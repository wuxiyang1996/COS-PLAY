"""
Bank Maintenance — MERGE: detect near-duplicate skills and merge them.

Pipeline:
  1. Candidate retrieval via LSH/inverted index (never all-pairs).
  2. Strict multi-criterion verification (Jaccard, cosine, transitions, cross-score).
  3. Merge execution: union instances, re-learn/verify/refine, alias mapping.
  4. Generate local re-decode requests when transition priors change.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
from skill_agents_grpo.bank_maintenance.indices import (
    EffectInvertedIndex,
    EmbeddingANN,
    MinHashLSH,
)
from skill_agents_grpo.bank_maintenance.schemas import (
    BankDiffEntry,
    DiffOp,
    RedecodeRequest,
    SkillProfile,
)

logger = logging.getLogger(__name__)


# ── Merge result ─────────────────────────────────────────────────────

@dataclass
class MergeResult:
    """Outcome of one merge operation."""

    canonical_id: str
    merged_ids: List[str] = field(default_factory=list)
    contract: Optional[SkillEffectsContract] = None
    report: Optional[VerificationReport] = None
    alias_map: Dict[str, str] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""


# ═════════════════════════════════════════════════════════════════════
# 1) Candidate retrieval (never all-pairs)
# ═════════════════════════════════════════════════════════════════════


def retrieve_merge_candidates(
    profiles: Dict[str, SkillProfile],
    inv_index: EffectInvertedIndex,
    lsh: MinHashLSH,
    ann: Optional[EmbeddingANN] = None,
    config: Optional[BankMaintenanceConfig] = None,
) -> Set[FrozenSet[str]]:
    """Collect candidate merge pairs from indices.

    Union of candidates from:
      - LSH bucket collisions
      - inverted-index high-overlap pairs
      - (optional) ANN nearest centroid pairs
      - transition-overlap pairs (for skills with empty effects)
    """
    pairs: Set[FrozenSet[str]] = set()

    pairs |= lsh.candidate_pairs()

    for sid, prof in profiles.items():
        hits = inv_index.candidates_for(
            prof.all_effects, min_shared=3, exclude={sid},
        )
        for peer_id, _ in hits:
            pairs.add(frozenset((sid, peer_id)))

    if ann is not None:
        for sid, prof in profiles.items():
            if prof.embedding_centroid is None:
                continue
            neighbours = ann.query(
                prof.embedding_centroid, k=3, exclude={sid},
            )
            for peer_id, sim in neighbours:
                if sim > (config.merge_emb_cosine_thresh if config else 0.90):
                    pairs.add(frozenset((sid, peer_id)))

    # For skills with empty effects, use transition overlap to find merge
    # candidates that the effect-based indices would miss.
    empty_effect_ids = [
        sid for sid, prof in profiles.items()
        if not prof.all_effects
    ]
    if len(empty_effect_ids) >= 2:
        k = config.merge_transition_overlap_k if config else 5
        min_overlap = config.merge_transition_overlap_min if config else 0.50
        for i, sid_a in enumerate(empty_effect_ids):
            pa = profiles[sid_a]
            for sid_b in empty_effect_ids[i + 1:]:
                pb = profiles[sid_b]
                if pa.transition_topk_next and pb.transition_topk_next:
                    t_next = _transition_overlap(pa.transition_topk_next, pb.transition_topk_next, k)
                    t_prev = _transition_overlap(pa.transition_topk_prev, pb.transition_topk_prev, k)
                    if (t_next + t_prev) / 2 >= min_overlap:
                        pairs.add(frozenset((sid_a, sid_b)))

    return pairs


# ═════════════════════════════════════════════════════════════════════
# 2) Strict verification
# ═════════════════════════════════════════════════════════════════════


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a and not b:
        return 1.0  # both empty → identical signatures
    return len(a & b) / len(a | b)


def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if a is None or b is None:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _transition_overlap(
    topk_a: List[Tuple[str, float]],
    topk_b: List[Tuple[str, float]],
    k: int,
) -> float:
    """Fraction of top-k transition neighbours shared."""
    set_a = {s for s, _ in topk_a[:k]}
    set_b = {s for s, _ in topk_b[:k]}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def verify_merge_pair(
    p1: SkillProfile,
    p2: SkillProfile,
    config: BankMaintenanceConfig,
) -> Tuple[bool, Dict[str, float]]:
    """Run strict checks on a candidate merge pair.

    Multiple criteria must pass simultaneously:
      - Effect Jaccard (eff_add, eff_del, eff_event combined)
      - Embedding cosine (if available)
      - Transition top-K overlap (if available)

    When both skills have empty effects, uses relaxed criteria:
    transition overlap alone is sufficient if high enough.
    """
    scores: Dict[str, float] = {}
    both_empty = not p1.all_effects and not p2.all_effects

    eff_jac = _jaccard(p1.all_effects, p2.all_effects)
    scores["eff_jaccard"] = eff_jac
    if not both_empty and eff_jac < config.merge_eff_jaccard_thresh:
        return False, scores

    if p1.embedding_centroid and p2.embedding_centroid:
        emb_cos = _cosine(p1.embedding_centroid, p2.embedding_centroid)
        scores["emb_cosine"] = emb_cos
        if emb_cos < config.merge_emb_cosine_thresh:
            return False, scores

    if p1.transition_topk_next and p2.transition_topk_next:
        k = config.merge_transition_overlap_k
        t_next = _transition_overlap(
            p1.transition_topk_next, p2.transition_topk_next, k,
        )
        t_prev = _transition_overlap(
            p1.transition_topk_prev, p2.transition_topk_prev, k,
        )
        scores["transition_next_overlap"] = t_next
        scores["transition_prev_overlap"] = t_prev
        avg_overlap = (t_next + t_prev) / 2
        if both_empty:
            if avg_overlap < config.merge_transition_overlap_min:
                return False, scores
        else:
            if avg_overlap < config.merge_transition_overlap_min:
                return False, scores
    elif both_empty:
        # Both effects empty AND no transition data → look at duration similarity
        if p1.duration_mean > 0 and p2.duration_mean > 0:
            dur_ratio = min(p1.duration_mean, p2.duration_mean) / max(p1.duration_mean, p2.duration_mean)
            scores["duration_ratio"] = dur_ratio
            if dur_ratio < 0.3:
                return False, scores
        # Same action verb in the label? (e.g. "overworld:CLEAR" and "dialog:CLEAR")
        verb1 = p1.skill_id.split(":")[-1] if ":" in p1.skill_id else p1.skill_id
        verb2 = p2.skill_id.split(":")[-1] if ":" in p2.skill_id else p2.skill_id
        scores["same_verb"] = 1.0 if verb1 == verb2 else 0.0
        if verb1 != verb2:
            return False, scores

    return True, scores


# ═════════════════════════════════════════════════════════════════════
# 3) Execute merge
# ═════════════════════════════════════════════════════════════════════


def execute_merge(
    k1: str,
    k2: str,
    instances_k1: List[SegmentRecord],
    instances_k2: List[SegmentRecord],
    config: BankMaintenanceConfig,
    prev_version: int = 0,
) -> MergeResult:
    """Merge two skills into one canonical skill.

    Chooses the skill with more instances as canonical. Unions instance sets,
    re-learns/re-verifies/re-refines the merged contract.
    """
    if len(instances_k1) >= len(instances_k2):
        canonical, retired = k1, k2
    else:
        canonical, retired = k2, k1

    all_instances = instances_k1 + instances_k2
    for inst in all_instances:
        inst.skill_label = canonical

    s3_config = Stage3MVPConfig(
        eff_freq=config.eff_freq,
        min_instances_per_skill=config.min_instances_per_skill,
        max_effects_per_skill=config.max_effects_per_skill,
        instance_pass_literal_frac=config.instance_pass_literal_frac,
    )

    contract = learn_effects_contract(
        canonical, all_instances, s3_config, prev_version=prev_version,
    )
    report = verify_effects_contract(contract, all_instances, s3_config)
    refined = refine_effects_contract(contract, report, s3_config)
    report_refined = verify_effects_contract(refined, all_instances, s3_config)

    result = MergeResult(
        canonical_id=canonical,
        merged_ids=[k1, k2],
        contract=refined,
        report=report_refined,
        alias_map={retired: canonical},
        accepted=True,
        reason="merged",
    )
    logger.info(
        "Merged %s + %s → %s (pass_rate=%.3f, n=%d)",
        k1, k2, canonical,
        report_refined.overall_pass_rate,
        len(all_instances),
    )
    return result


# ═════════════════════════════════════════════════════════════════════
# 4) Re-decode requests for merge
# ═════════════════════════════════════════════════════════════════════


def redecode_requests_for_merge(
    k1: str,
    k2: str,
    instances_k1: List[SegmentRecord],
    instances_k2: List[SegmentRecord],
    config: BankMaintenanceConfig,
    canonical_id: str,
) -> List[RedecodeRequest]:
    """Generate re-decode windows for trajectories affected by a merge."""
    traj_windows: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for inst in instances_k1 + instances_k2:
        traj_windows[inst.traj_id].append((inst.t_start, inst.t_end))

    pad = config.redecode_window_pad
    requests: List[RedecodeRequest] = []

    for traj_id, windows in traj_windows.items():
        merged = _merge_windows(windows, pad)
        for ws, we in merged:
            requests.append(RedecodeRequest(
                traj_id=traj_id,
                window_start=max(0, ws - pad),
                window_end=we + pad,
                reason=f"merge:{k1}+{k2}->{canonical_id}",
                affected_skills=[canonical_id],
            ))

    return requests


def _merge_windows(
    windows: List[Tuple[int, int]], gap: int,
) -> List[Tuple[int, int]]:
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
