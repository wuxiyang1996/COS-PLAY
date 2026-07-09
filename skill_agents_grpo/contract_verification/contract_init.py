"""
Steps 2-3 — Compute per-instance effects and build initial contracts.

Step 2: For each ``SegmentRecord``, derive ``effects_add``, ``effects_del``,
        and candidate invariants from its predicate summaries.

Step 3: For each skill (excluding ``__NEW__``), aggregate its segment instances
        into a candidate ``SkillContract`` using frequency thresholds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from skill_agents_grpo.contract_verification.config import (
    ContractAggregationConfig,
    PredicateConfig,
)
from skill_agents_grpo.contract_verification.schemas import SegmentRecord, SkillContract

NEW_SKILL = "__NEW__"


# ── Step 2: per-instance effects ─────────────────────────────────────

def _booleanize(preds: Dict[str, float], p_thresh: float) -> Set[str]:
    """Convert probabilistic predicates to a boolean set by thresholding."""
    return {k for k, v in preds.items() if v >= p_thresh}


def compute_instance_effects(
    record: SegmentRecord,
    p_thresh: float = 0.7,
    inv_freq: float = 0.9,
) -> SegmentRecord:
    """Populate ``effects_add``, ``effects_del`` on a segment record in-place.

    Parameters
    ----------
    record : SegmentRecord
        Must have ``P_start``, ``P_end``, ``P_all`` already populated.
    p_thresh : float
        Threshold to convert probabilistic predicates to boolean.
    inv_freq : float
        Minimum fraction of timesteps a predicate must hold to be an invariant candidate.

    Returns
    -------
    SegmentRecord
        Same object, mutated with effects and invariant annotations.
    """
    start_set = _booleanize(record.P_start, p_thresh)
    end_set = _booleanize(record.P_end, p_thresh)

    record.effects_add = end_set - start_set
    record.effects_del = start_set - end_set

    return record


def compute_instance_invariants(
    record: SegmentRecord,
    p_thresh: float = 0.7,
    inv_freq: float = 0.9,
) -> Set[str]:
    """Compute candidate invariants for one segment instance.

    An invariant is a predicate that holds (>= p_thresh) in >= inv_freq fraction
    of timesteps within the segment.
    """
    if not record.P_all:
        return set()
    T = len(record.P_all)
    all_preds: Set[str] = set()
    for p_dict in record.P_all:
        all_preds.update(p_dict.keys())

    invariants: Set[str] = set()
    for pred in all_preds:
        hold_count = sum(1 for p_dict in record.P_all if p_dict.get(pred, 0.0) >= p_thresh)
        if hold_count / T >= inv_freq:
            invariants.add(pred)
    return invariants


def compute_all_effects(
    records: List[SegmentRecord],
    config: PredicateConfig,
    agg_config: ContractAggregationConfig,
) -> List[SegmentRecord]:
    """Batch: compute effects for all segment records."""
    for rec in records:
        compute_instance_effects(rec, config.p_thresh, agg_config.inv_freq)
    return records


# ── Step 3: aggregate into initial contracts ─────────────────────────

def build_initial_contracts(
    records: List[SegmentRecord],
    pred_config: PredicateConfig,
    agg_config: ContractAggregationConfig,
    existing_contracts: Optional[Dict[str, SkillContract]] = None,
) -> Dict[str, SkillContract]:
    """Build (or update) initial contracts from segment records.

    For each skill k (excluding ``__NEW__``), collect all instances and compute:
      - Pre_k: predicates present at start in >= ``pre_freq`` fraction of instances.
      - EffAdd_k: predicates in ``effects_add`` in >= ``eff_freq`` fraction.
      - EffDel_k: predicates in ``effects_del`` in >= ``eff_freq`` fraction.
      - Inv_k: predicates holding within segments in >= ``inv_freq`` fraction.

    If ``existing_contracts`` is provided, contracts are updated (version bumped)
    rather than created from scratch.

    Returns
    -------
    dict[str, SkillContract]
        Mapping skill_id -> SkillContract (version 0 if new, bumped if existing).
    """
    by_skill: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for rec in records:
        if rec.skill_label != NEW_SKILL:
            by_skill[rec.skill_label].append(rec)

    contracts: Dict[str, SkillContract] = {}

    for skill_id, instances in by_skill.items():
        n = len(instances)
        if n == 0:
            continue

        pre_counter: Counter = Counter()
        eff_add_counter: Counter = Counter()
        eff_del_counter: Counter = Counter()
        inv_counter: Counter = Counter()

        for rec in instances:
            start_bools = _booleanize(rec.P_start, pred_config.p_thresh)
            for p in start_bools:
                pre_counter[p] += 1
            for p in rec.effects_add:
                eff_add_counter[p] += 1
            for p in rec.effects_del:
                eff_del_counter[p] += 1

            inv_cands = compute_instance_invariants(
                rec, pred_config.p_thresh, agg_config.inv_freq,
            )
            for p in inv_cands:
                inv_counter[p] += 1

        pre = {p for p, c in pre_counter.items() if c / n >= agg_config.pre_freq}
        eff_add = {p for p, c in eff_add_counter.items() if c / n >= agg_config.eff_freq}
        eff_del = {p for p, c in eff_del_counter.items() if c / n >= agg_config.eff_freq}
        inv = {p for p, c in inv_counter.items() if c / n >= agg_config.inv_freq}

        if existing_contracts and skill_id in existing_contracts:
            contract = existing_contracts[skill_id]
            contract.pre = pre
            contract.eff_add = eff_add
            contract.eff_del = eff_del
            contract.inv = inv
            contract.bump_version()
        else:
            contract = SkillContract(
                skill_id=skill_id,
                version=0,
                pre=pre,
                eff_add=eff_add,
                eff_del=eff_del,
                inv=inv,
            )

        contracts[skill_id] = contract

    return contracts
