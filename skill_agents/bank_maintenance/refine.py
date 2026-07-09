"""
Bank Maintenance — REFINE: adjust contract strength and update duration/boundary models.

Operations:
  A) Weaken: drop fragile literals with low success rates.
  B) Strengthen: add discriminative literals that separate from confusers.
  C) Duration model update.
  D) (Optional) start/end boundary classifier stub.

Confusers are the top-2/3 confusion partners from Stage 2 diagnostics, never
all skills — keeping the cost proportional to the number of triggered skills.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents.bank_maintenance.config import BankMaintenanceConfig
from skill_agents.bank_maintenance.duration_model import DurationModelStore
from skill_agents.bank_maintenance.schemas import (
    BankDiffEntry,
    DiffOp,
    SkillProfile,
)

logger = logging.getLogger(__name__)


# ── Refine result ────────────────────────────────────────────────────

@dataclass
class RefineResult:
    """Outcome of refining one skill."""

    skill_id: str
    dropped_literals: List[str] = field(default_factory=list)
    added_literals: List[str] = field(default_factory=list)
    new_contract: Optional[SkillEffectsContract] = None
    reason: str = ""


# ═════════════════════════════════════════════════════════════════════
# Trigger checks
# ═════════════════════════════════════════════════════════════════════


def check_refine_triggers(
    profile: SkillProfile,
    confusion_partners: Optional[List[str]] = None,
    config: Optional[BankMaintenanceConfig] = None,
) -> Tuple[bool, str]:
    """Decide whether *profile* should enter the refine queue.

    Fires for:
      - Too strong: low pass rate or many low-success literals.
      - Too weak: confusion partners supplied and pass rate is adequate but
        discriminability is poor (heuristic: many confusers listed).
    """
    cfg = config or BankMaintenanceConfig()

    if profile.top_violating_literals:
        return True, "too_strong"

    if profile.overall_pass_rate < cfg.refine_drop_success_rate:
        return True, "too_strong_low_pass_rate"

    if confusion_partners and len(confusion_partners) >= 1:
        return True, "too_weak_confusers"

    if not profile.all_effects and profile.n_instances >= 5:
        return True, "empty_contract"

    return False, ""


# ═════════════════════════════════════════════════════════════════════
# A) Weaken — drop fragile literals
# ═════════════════════════════════════════════════════════════════════


def weaken_contract(
    contract: SkillEffectsContract,
    report: VerificationReport,
    config: BankMaintenanceConfig,
    min_literals: int = 1,
) -> Tuple[SkillEffectsContract, List[str]]:
    """Drop effect literals whose per-instance success rate is below threshold.

    Returns the new contract and the list of dropped literal names.

    If dropping would leave fewer than *min_literals* total literals, the
    top-scoring literals are retained to prevent empty contracts.
    """
    thresh = config.refine_drop_success_rate
    dropped: List[str] = []

    new_add: Set[str] = set()
    for p in contract.eff_add:
        if report.eff_add_success_rate.get(p, 0.0) >= thresh:
            new_add.add(p)
        else:
            dropped.append(f"add:{p}")

    new_del: Set[str] = set()
    for p in contract.eff_del:
        if report.eff_del_success_rate.get(p, 0.0) >= thresh:
            new_del.add(p)
        else:
            dropped.append(f"del:{p}")

    new_event: Set[str] = set()
    for e in contract.eff_event:
        if report.eff_event_rate.get(e, 0.0) >= thresh:
            new_event.add(e)
        else:
            dropped.append(f"evt:{e}")

    total_kept = len(new_add) + len(new_del) + len(new_event)
    total_original = len(contract.eff_add) + len(contract.eff_del) + len(contract.eff_event)

    if total_kept < min_literals and total_original > 0:
        all_rates: List[Tuple[str, str, float]] = []
        for p in contract.eff_add:
            all_rates.append(("add", p, report.eff_add_success_rate.get(p, 0.0)))
        for p in contract.eff_del:
            all_rates.append(("del", p, report.eff_del_success_rate.get(p, 0.0)))
        for e in contract.eff_event:
            all_rates.append(("evt", e, report.eff_event_rate.get(e, 0.0)))
        all_rates.sort(key=lambda x: -x[2])

        rescued: List[str] = []
        for category, literal, _rate in all_rates:
            if total_kept >= min_literals:
                break
            tag = f"{category}:{literal}"
            if tag not in dropped:
                continue
            if category == "add":
                new_add.add(literal)
            elif category == "del":
                new_del.add(literal)
            else:
                new_event.add(literal)
            dropped.remove(tag)
            rescued.append(tag)
            total_kept += 1

        if rescued:
            logger.info(
                "Rescued %d literal(s) for %s to avoid empty contract: %s",
                len(rescued), contract.skill_id, rescued,
            )

    new_support = {
        k: v for k, v in contract.support.items()
        if k in new_add or k in new_del or k in new_event
    }

    new_contract = SkillEffectsContract(
        skill_id=contract.skill_id,
        version=contract.version + 1,
        name=contract.name,
        description=contract.description,
        eff_add=new_add,
        eff_del=new_del,
        eff_event=new_event,
        support=new_support,
        n_instances=contract.n_instances,
    )
    return new_contract, dropped


# ═════════════════════════════════════════════════════════════════════
# B) Strengthen — add discriminative literals vs confusers
# ═════════════════════════════════════════════════════════════════════


def _literal_frequencies(
    instances: List[SegmentRecord],
) -> Dict[str, float]:
    """Compute normalised frequency of every effect literal across instances."""
    n = len(instances)
    if n == 0:
        return {}
    counts: Counter = Counter()
    for inst in instances:
        for p in inst.eff_add:
            counts[f"add:{p}"] += 1
        for p in inst.eff_del:
            counts[f"del:{p}"] += 1
        for e in inst.eff_event:
            counts[f"evt:{e}"] += 1
    return {lit: cnt / n for lit, cnt in counts.items()}


def strengthen_contract(
    contract: SkillEffectsContract,
    self_instances: List[SegmentRecord],
    confuser_instances: Dict[str, List[SegmentRecord]],
    config: BankMaintenanceConfig,
) -> Tuple[SkillEffectsContract, List[str]]:
    """Add discriminative literals that are common in *self* but rare in confusers.

    score(p) = freq_self(p) - max_{c in confusers} freq_c(p)
    Add top-N literals with score > 0 and freq_self >= threshold.
    """
    self_freq = _literal_frequencies(self_instances)

    confuser_freqs: List[Dict[str, float]] = [
        _literal_frequencies(insts) for insts in confuser_instances.values()
    ]

    existing = (
        {f"add:{p}" for p in contract.eff_add}
        | {f"del:{p}" for p in contract.eff_del}
        | {f"evt:{e}" for e in contract.eff_event}
    )

    scored: List[Tuple[str, float]] = []
    for lit, freq_k in self_freq.items():
        if lit in existing:
            continue
        if freq_k < config.refine_add_freq_self:
            continue
        max_confuser = max(
            (cf.get(lit, 0.0) for cf in confuser_freqs), default=0.0,
        )
        if max_confuser > config.refine_add_max_confuser_freq:
            continue
        score = freq_k - max_confuser
        if score > 0:
            scored.append((lit, score))

    scored.sort(key=lambda x: -x[1])
    to_add = scored[: config.refine_top_n_add]

    new_add = set(contract.eff_add)
    new_del = set(contract.eff_del)
    new_event = set(contract.eff_event)
    new_support = dict(contract.support)
    added_names: List[str] = []

    for lit, _ in to_add:
        category, predicate = lit.split(":", 1)
        n_self = len(self_instances)
        support_count = int(self_freq[lit] * n_self)

        if category == "add":
            new_add.add(predicate)
        elif category == "del":
            new_del.add(predicate)
        elif category == "evt":
            new_event.add(predicate)
        new_support[predicate] = support_count
        added_names.append(lit)

    new_contract = SkillEffectsContract(
        skill_id=contract.skill_id,
        version=contract.version + 1,
        name=contract.name,
        description=contract.description,
        eff_add=new_add,
        eff_del=new_del,
        eff_event=new_event,
        support=new_support,
        n_instances=contract.n_instances,
    )
    return new_contract, added_names


# ═════════════════════════════════════════════════════════════════════
# C) Full refine pass (weaken + strengthen)
# ═════════════════════════════════════════════════════════════════════


def refine_skill(
    contract: SkillEffectsContract,
    report: VerificationReport,
    self_instances: List[SegmentRecord],
    confuser_instances: Dict[str, List[SegmentRecord]],
    config: BankMaintenanceConfig,
) -> RefineResult:
    """Full refine: weaken fragile literals then strengthen vs confusers."""
    result = RefineResult(skill_id=contract.skill_id)

    weakened, dropped = weaken_contract(contract, report, config)
    result.dropped_literals = dropped

    if confuser_instances:
        strengthened, added = strengthen_contract(
            weakened, self_instances, confuser_instances, config,
        )
        result.added_literals = added
        result.new_contract = strengthened
    else:
        result.new_contract = weakened

    result.reason = (
        f"dropped={len(dropped)}, added={len(result.added_literals)}"
    )
    logger.info("Refined %s: %s", contract.skill_id, result.reason)
    return result


# ═════════════════════════════════════════════════════════════════════
# D) Duration update helper
# ═════════════════════════════════════════════════════════════════════


def update_duration_model(
    duration_store: DurationModelStore,
    skill_id: str,
    instances: List[SegmentRecord],
) -> None:
    """Recompute duration model from scratch for one skill."""
    lengths = [inst.t_end - inst.t_start for inst in instances]
    duration_store.replace(skill_id, lengths)


# ═════════════════════════════════════════════════════════════════════
# E) Confusion partners extraction (from Stage 2 diagnostics)
# ═════════════════════════════════════════════════════════════════════


def extract_confusion_partners(
    skill_id: str,
    stage2_results: List[dict],
    top_n: int = 3,
) -> List[str]:
    """Extract top confusion partners for *skill_id* from Stage 2 results.

    Parameters
    ----------
    stage2_results : list[dict]
        Each entry has ``assigned_skill`` and ``candidates`` (list of
        ``{skill, total_score}``).  These come from
        ``SegmentationResult.to_dict()["segments"]``.
    top_n : int
        Number of confusers to return.

    Returns the skill_ids that most frequently appear as the #2 candidate
    when *skill_id* is assigned.
    """
    confuser_counts: Counter = Counter()
    for seg in stage2_results:
        if seg.get("assigned_skill") != skill_id:
            continue
        cands = seg.get("candidates", [])
        for c in cands[1:top_n + 1]:
            cid = c.get("skill") or c.get("skill_id", "")
            if cid and cid != skill_id:
                confuser_counts[cid] += 1

    return [sid for sid, _ in confuser_counts.most_common(top_n)]
