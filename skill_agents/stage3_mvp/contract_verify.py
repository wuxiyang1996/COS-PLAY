"""
Step 4 — Verify an effects contract against segment instances.

For each instance, check whether the contract's eff_add / eff_del / eff_event
literals are satisfied.  Produces per-literal success rates, an overall
pass rate, worst segments, and failure signatures.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from skill_agents.stage3_mvp.config import Stage3MVPConfig
from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)


def _instance_failures(
    contract: SkillEffectsContract,
    rec: SegmentRecord,
) -> List[str]:
    """Return list of failure tokens for one instance (used for signatures)."""
    fails: List[str] = []
    for p in contract.eff_add:
        if p not in rec.eff_add and p not in rec.B_end:
            fails.append(f"miss_add:{p}")
    for p in contract.eff_del:
        if p not in rec.eff_del and p in rec.B_end:
            fails.append(f"miss_del:{p}")
    for e in contract.eff_event:
        if e not in rec.eff_event:
            fails.append(f"miss_evt:{e}")
    return fails


def verify_effects_contract(
    contract: SkillEffectsContract,
    instances: List[SegmentRecord],
    config: Stage3MVPConfig,
) -> VerificationReport:
    """Verify *contract* against all *instances* and produce a report.

    Parameters
    ----------
    contract : SkillEffectsContract
    instances : list[SegmentRecord]
        Segment records for the same skill (effects already computed).
    config : Stage3MVPConfig

    Returns
    -------
    VerificationReport
    """
    n = len(instances)
    if n == 0:
        return VerificationReport(skill_id=contract.skill_id)

    # Per-literal success counters
    add_ok: Counter = Counter()
    del_ok: Counter = Counter()
    evt_ok: Counter = Counter()

    # Per-instance pass / failure tracking
    instance_failure_counts: Dict[str, int] = {}
    failure_sig_counter: Counter = Counter()
    total_literals = contract.total_literals

    for rec in instances:
        fails = _instance_failures(contract, rec)
        instance_failure_counts[rec.seg_id] = len(fails)

        if fails:
            sig = "|".join(sorted(fails))
            failure_sig_counter[sig] += 1

        for p in contract.eff_add:
            if p in rec.eff_add or p in rec.B_end:
                add_ok[p] += 1
        for p in contract.eff_del:
            if p in rec.eff_del or p not in rec.B_end:
                del_ok[p] += 1
        for e in contract.eff_event:
            if e in rec.eff_event:
                evt_ok[e] += 1

    # Per-literal success rates
    eff_add_sr = {p: add_ok[p] / n for p in contract.eff_add}
    eff_del_sr = {p: del_ok[p] / n for p in contract.eff_del}
    eff_event_r = {e: evt_ok[e] / n for e in contract.eff_event}

    # Overall pass rate: an instance "passes" if >= instance_pass_literal_frac
    # of the contract literals are satisfied.
    pass_thresh = config.instance_pass_literal_frac
    passing = 0
    for rec in instances:
        n_fails = instance_failure_counts[rec.seg_id]
        if total_literals == 0:
            passing += 1
        elif (total_literals - n_fails) / total_literals >= pass_thresh:
            passing += 1
    overall_pass_rate = passing / n

    # Worst segments: sorted by most failures, take top N
    sorted_worst = sorted(
        instance_failure_counts.items(), key=lambda x: -x[1],
    )
    worst_segments = [
        seg_id for seg_id, cnt in sorted_worst[:config.max_worst_segments]
        if cnt > 0
    ]

    return VerificationReport(
        skill_id=contract.skill_id,
        n_instances=n,
        eff_add_success_rate=eff_add_sr,
        eff_del_success_rate=eff_del_sr,
        eff_event_rate=eff_event_r,
        overall_pass_rate=overall_pass_rate,
        worst_segments=worst_segments,
        failure_signatures=dict(failure_sig_counter),
    )
