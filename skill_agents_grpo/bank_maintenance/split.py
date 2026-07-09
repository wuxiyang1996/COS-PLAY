"""
Bank Maintenance — SPLIT: detect multi-modal skills and split into children.

Pipeline:
  1. Cheap trigger filters (pass-rate, failure concentration, embedding variance).
  2. Effect-signature clustering (fastest) → sparse-vector clustering → hybrid.
  3. Child skill creation with contract learn/verify/refine inside each cluster.
  4. Acceptance gate: each child must meet min size and pass-rate thresholds.

Efficiency: only triggered skills enter the split queue; clustering starts with
the cheapest method and escalates only when needed.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig
from skill_agents_grpo.stage3_mvp.contract_learn import learn_effects_contract
from skill_agents_grpo.stage3_mvp.contract_refine import refine_effects_contract
from skill_agents_grpo.stage3_mvp.contract_verify import verify_effects_contract
from skill_agents_grpo.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents_grpo.bank_maintenance.config import BankMaintenanceConfig
from skill_agents_grpo.bank_maintenance.schemas import (
    BankDiffEntry,
    DiffOp,
    RedecodeRequest,
    SkillProfile,
)

logger = logging.getLogger(__name__)


# ── Split result ─────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Outcome of splitting one parent skill."""

    parent_id: str
    accepted: bool = False
    children: List[ChildSkill] = field(default_factory=list)
    reason: str = ""


@dataclass
class ChildSkill:
    """One child produced by a split."""

    skill_id: str
    parent_id: str
    contract: SkillEffectsContract = field(default_factory=lambda: SkillEffectsContract(skill_id=""))
    report: Optional[VerificationReport] = None
    instance_seg_ids: List[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════
# 1) Trigger detection
# ═════════════════════════════════════════════════════════════════════


def check_split_triggers(
    profile: SkillProfile,
    instances: List[SegmentRecord],
    config: BankMaintenanceConfig,
    embeddings: Optional[Dict[str, List[float]]] = None,
) -> Tuple[bool, str]:
    """Decide whether *profile* should enter the split queue.

    Checks are ordered cheapest-first; returns as soon as one fires.
    """
    # (1) Low overall pass rate
    if profile.overall_pass_rate < config.split_pass_rate_thresh:
        return True, "low_pass_rate"

    # (2) Failure concentration: top-2 signatures explain most failures
    if profile.failure_signature_counts:
        total_failures = sum(profile.failure_signature_counts.values())
        if total_failures > 0:
            top2 = sorted(
                profile.failure_signature_counts.values(), reverse=True,
            )[:2]
            concentration = sum(top2) / total_failures
            if concentration >= config.split_failure_concentration:
                return True, "failure_concentration"

    # (3) Embedding variance check (only if embeddings supplied)
    if embeddings and profile.embedding_var_diag:
        total_var = sum(profile.embedding_var_diag)
        if total_var > config.split_embedding_var_thresh:
            return True, "high_embedding_variance"

    # (4) Quick 2-means SSE drop on effect signatures
    if len(instances) >= 2 * config.min_child_size:
        fired, ratio = _quick_2means_sse(instances)
        if fired and ratio > config.split_sse_ratio:
            return True, "sse_drop"

    # (5) High duration variance — skill mixes short and long behaviors
    if len(instances) >= 2 * config.min_child_size:
        lengths = [inst.t_end - inst.t_start for inst in instances]
        if lengths:
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                cv = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5 / mean_len
                max_ratio = max(lengths) / max(min(lengths), 1)
                if cv > 1.5 and max_ratio > 10:
                    return True, "high_duration_variance"

    return False, ""


def _quick_2means_sse(instances: List[SegmentRecord]) -> Tuple[bool, float]:
    """Run a very cheap 2-partition on effect signatures and check SSE drop."""
    sigs = [inst.effect_signature() for inst in instances]
    sig_counter = Counter(sigs)
    if len(sig_counter) < 2:
        return False, 0.0

    top2 = sig_counter.most_common(2)
    group_a = {inst.seg_id for inst in instances if inst.effect_signature() == top2[0][0]}
    group_b = {inst.seg_id for inst in instances if inst.effect_signature() == top2[1][0]}

    if len(group_a) < 2 or len(group_b) < 2:
        return False, 0.0

    ratio = (len(group_a) + len(group_b)) / len(instances)
    return True, ratio


# ═════════════════════════════════════════════════════════════════════
# 2) Clustering strategies (fast → slow)
# ═════════════════════════════════════════════════════════════════════


def cluster_by_effect_signature(
    instances: List[SegmentRecord],
    min_cluster_size: int,
) -> Optional[List[List[SegmentRecord]]]:
    """Group instances by identical (EffAdd, EffDel, EffEvent) hash.

    Returns two largest groups if both are big enough, else None.
    """
    groups: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for inst in instances:
        groups[inst.effect_signature()].append(inst)

    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    viable = [g for g in sorted_groups if len(g) >= min_cluster_size]

    if len(viable) >= 2:
        return viable[:2]
    return None


def cluster_by_sparse_effects(
    instances: List[SegmentRecord],
    min_cluster_size: int,
) -> Optional[List[List[SegmentRecord]]]:
    """Agglomerative-ish clustering using Jaccard on effect literal sets.

    Greedy 2-partition: pick the instance pair with lowest Jaccard as seeds,
    then assign every other instance to the nearest seed.
    """
    if len(instances) < 2 * min_cluster_size:
        return None

    def _effect_set(r: SegmentRecord) -> Set[str]:
        return r.eff_add | r.eff_del | r.eff_event

    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    effects = [_effect_set(r) for r in instances]

    import random
    n = len(instances)
    sample_size = min(200, n * (n - 1) // 2)
    best_pair = (0, 1)
    best_jac = 1.0

    indices = list(range(n))
    for _ in range(sample_size):
        i, j = random.sample(indices, 2)
        jac = _jaccard(effects[i], effects[j])
        if jac < best_jac:
            best_jac = jac
            best_pair = (i, j)

    if best_jac > 0.8:
        return None

    seed_a, seed_b = effects[best_pair[0]], effects[best_pair[1]]
    cluster_a: List[SegmentRecord] = []
    cluster_b: List[SegmentRecord] = []

    for inst, eff in zip(instances, effects):
        ja = _jaccard(eff, seed_a)
        jb = _jaccard(eff, seed_b)
        if ja >= jb:
            cluster_a.append(inst)
        else:
            cluster_b.append(inst)

    if len(cluster_a) >= min_cluster_size and len(cluster_b) >= min_cluster_size:
        return [cluster_a, cluster_b]
    return None


# ═════════════════════════════════════════════════════════════════════
# 3) Execute split
# ═════════════════════════════════════════════════════════════════════


def execute_split(
    parent_id: str,
    instances: List[SegmentRecord],
    config: BankMaintenanceConfig,
    parent_version: int = 0,
) -> SplitResult:
    """Try to split *parent_id* into child skills.

    Clustering priority: effect-signature → sparse-effects.
    Each child is learned/verified/refined using contract verification functions.
    Split is accepted only if every child meets the pass-rate threshold.
    """
    result = SplitResult(parent_id=parent_id)

    clusters = cluster_by_effect_signature(instances, config.min_child_size)
    if clusters is None:
        clusters = cluster_by_sparse_effects(instances, config.min_child_size)
    if clusters is None:
        result.reason = "no_viable_clusters"
        return result

    s3_config = Stage3MVPConfig(
        eff_freq=config.eff_freq,
        min_instances_per_skill=config.min_instances_per_skill,
        max_effects_per_skill=config.max_effects_per_skill,
        instance_pass_literal_frac=config.instance_pass_literal_frac,
    )

    children: List[ChildSkill] = []
    all_pass = True

    for idx, cluster in enumerate(clusters):
        child_id = f"{parent_id}__child_{idx}"
        contract = learn_effects_contract(
            child_id, cluster, s3_config, prev_version=parent_version,
        )
        report = verify_effects_contract(contract, cluster, s3_config)
        refined = refine_effects_contract(contract, report, s3_config)
        report_refined = verify_effects_contract(refined, cluster, s3_config)

        child = ChildSkill(
            skill_id=child_id,
            parent_id=parent_id,
            contract=refined,
            report=report_refined,
            instance_seg_ids=[r.seg_id for r in cluster],
        )
        children.append(child)

        if report_refined.overall_pass_rate < config.child_pass_rate_thresh:
            all_pass = False

    if all_pass and len(children) >= 2:
        result.accepted = True
        result.children = children
        result.reason = "accepted"
        logger.info(
            "Split %s → %s (pass rates: %s)",
            parent_id,
            [c.skill_id for c in children],
            [c.report.overall_pass_rate for c in children if c.report],
        )
    else:
        result.reason = "child_pass_rate_below_threshold"
        logger.info("Split rejected for %s: %s", parent_id, result.reason)

    return result


# ═════════════════════════════════════════════════════════════════════
# 4) Build re-decode requests for split
# ═════════════════════════════════════════════════════════════════════


def redecode_requests_for_split(
    parent_id: str,
    instances: List[SegmentRecord],
    config: BankMaintenanceConfig,
    children: List[ChildSkill],
) -> List[RedecodeRequest]:
    """Generate local re-decode windows for trajectories affected by a split."""
    traj_windows: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for inst in instances:
        traj_windows[inst.traj_id].append((inst.t_start, inst.t_end))

    requests: List[RedecodeRequest] = []
    child_ids = [c.skill_id for c in children]
    pad = config.redecode_window_pad

    for traj_id, windows in traj_windows.items():
        merged = _merge_windows(windows, pad)
        for ws, we in merged:
            requests.append(RedecodeRequest(
                traj_id=traj_id,
                window_start=max(0, ws - pad),
                window_end=we + pad,
                reason=f"split:{parent_id}",
                affected_skills=child_ids,
            ))

    return requests


def _merge_windows(
    windows: List[Tuple[int, int]], gap: int,
) -> List[Tuple[int, int]]:
    """Merge overlapping or close windows."""
    if not windows:
        return []
    sorted_w = sorted(windows)
    merged = [sorted_w[0]]
    for s, e in sorted_w[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + gap:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged
