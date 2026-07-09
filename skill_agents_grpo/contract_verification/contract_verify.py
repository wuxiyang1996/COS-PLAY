"""
Step 4 — Verify contracts and produce counterexamples.

For each skill k, evaluate its ``SkillContract`` against all segment instances:
  - Pre check: does P_start satisfy each precondition?
  - EffAdd check: does P_end contain each expected add-effect?
  - EffDel check: is each expected delete-effect absent in P_end?
  - Inv check: does each invariant hold across enough timesteps?

Outputs a ``VerificationReport`` with per-literal rates, overall pass rate,
failure signature strings (for clustering), and ranked counterexamples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from skill_agents_grpo.contract_verification.config import (
    ContractVerificationConfig,
    PredicateConfig,
    ContractAggregationConfig,
    VerificationConfig,
)
from skill_agents_grpo.contract_verification.schemas import (
    SegmentRecord,
    SkillContract,
    VerificationReport,
)


def _check_pre(
    contract: SkillContract,
    record: SegmentRecord,
    p_thresh: float,
) -> Tuple[Dict[str, bool], List[str]]:
    """Check preconditions.  Returns per-literal pass/fail and violation list."""
    results: Dict[str, bool] = {}
    violations: List[str] = []
    for p in contract.pre:
        holds = record.P_start.get(p, 0.0) >= p_thresh
        results[p] = holds
        if not holds:
            violations.append(f"pre_miss:{p}")
    return results, violations


def _check_eff_add(
    contract: SkillContract,
    record: SegmentRecord,
    p_thresh: float,
) -> Tuple[Dict[str, bool], List[str]]:
    """Check add-effects.  Returns per-literal pass/fail and violation list."""
    results: Dict[str, bool] = {}
    violations: List[str] = []
    for p in contract.eff_add:
        holds = record.P_end.get(p, 0.0) >= p_thresh
        results[p] = holds
        if not holds:
            violations.append(f"eff_add_miss:{p}")
    return results, violations


def _check_eff_del(
    contract: SkillContract,
    record: SegmentRecord,
    p_thresh: float,
) -> Tuple[Dict[str, bool], List[str]]:
    """Check delete-effects.  Returns per-literal pass/fail and violation list."""
    results: Dict[str, bool] = {}
    violations: List[str] = []
    for p in contract.eff_del:
        gone = record.P_end.get(p, 0.0) < p_thresh
        results[p] = gone
        if not gone:
            violations.append(f"eff_del_miss:{p}")
    return results, violations


def _check_inv(
    contract: SkillContract,
    record: SegmentRecord,
    p_thresh: float,
    inv_freq: float,
) -> Tuple[Dict[str, bool], List[str]]:
    """Check invariants.  Returns per-literal pass/fail and violation list."""
    results: Dict[str, bool] = {}
    violations: List[str] = []
    if not record.P_all:
        return results, violations

    T = len(record.P_all)
    for p in contract.inv:
        hold_count = sum(1 for d in record.P_all if d.get(p, 0.0) >= p_thresh)
        frac = hold_count / T if T > 0 else 0.0
        holds = frac >= inv_freq
        results[p] = holds
        if not holds:
            violations.append(f"inv_miss:{p}")
    return results, violations


def verify_contract(
    contract: SkillContract,
    instances: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> VerificationReport:
    """Verify a skill contract against all its segment instances.

    Parameters
    ----------
    contract : SkillContract
        The contract to verify.
    instances : list[SegmentRecord]
        All segment records labelled with this skill.
    config : ContractVerificationConfig
        Thresholds for predicate booleanization and invariant checking.

    Returns
    -------
    VerificationReport
        Aggregated per-literal rates, overall pass rate, counterexamples,
        and failure signature histogram.
    """
    p_thresh = config.predicates.p_thresh
    inv_freq = config.aggregation.inv_freq
    max_cx = config.verification.max_counterexamples

    n = len(instances)
    if n == 0:
        return VerificationReport(skill_id=contract.skill_id)

    pre_pass: Dict[str, int] = defaultdict(int)
    eff_add_pass: Dict[str, int] = defaultdict(int)
    eff_del_pass: Dict[str, int] = defaultdict(int)
    inv_pass: Dict[str, int] = defaultdict(int)

    failure_sigs: Counter = Counter()
    instance_violations: List[Tuple[str, int, List[str]]] = []

    for rec in instances:
        all_violations: List[str] = []

        pre_res, pre_viol = _check_pre(contract, rec, p_thresh)
        for p, ok in pre_res.items():
            if ok:
                pre_pass[p] += 1
        all_violations.extend(pre_viol)

        eff_add_res, eff_add_viol = _check_eff_add(contract, rec, p_thresh)
        for p, ok in eff_add_res.items():
            if ok:
                eff_add_pass[p] += 1
        all_violations.extend(eff_add_viol)

        eff_del_res, eff_del_viol = _check_eff_del(contract, rec, p_thresh)
        for p, ok in eff_del_res.items():
            if ok:
                eff_del_pass[p] += 1
        all_violations.extend(eff_del_viol)

        inv_res, inv_viol = _check_inv(contract, rec, p_thresh, inv_freq)
        for p, ok in inv_res.items():
            if ok:
                inv_pass[p] += 1
        all_violations.extend(inv_viol)

        if all_violations:
            sig = "|".join(sorted(all_violations))
            failure_sigs[sig] += 1
            instance_violations.append((rec.seg_id, len(all_violations), all_violations))

    pre_viol_rate = {p: 1.0 - (pre_pass.get(p, 0) / n) for p in contract.pre}
    eff_add_succ = {p: eff_add_pass.get(p, 0) / n for p in contract.eff_add}
    eff_del_succ = {p: eff_del_pass.get(p, 0) / n for p in contract.eff_del}
    inv_hold = {p: inv_pass.get(p, 0) / n for p in contract.inv}

    total_checks = 0
    total_passed = 0
    for p in contract.pre:
        total_checks += n
        total_passed += pre_pass.get(p, 0)
    for p in contract.eff_add:
        total_checks += n
        total_passed += eff_add_pass.get(p, 0)
    for p in contract.eff_del:
        total_checks += n
        total_passed += eff_del_pass.get(p, 0)
    for p in contract.inv:
        total_checks += n
        total_passed += inv_pass.get(p, 0)

    overall_pass = total_passed / total_checks if total_checks > 0 else 1.0

    instance_violations.sort(key=lambda x: -x[1])
    counterexample_ids = [seg_id for seg_id, _, _ in instance_violations[:max_cx]]

    return VerificationReport(
        skill_id=contract.skill_id,
        n_instances=n,
        pre_violation_rate=pre_viol_rate,
        eff_add_success_rate=eff_add_succ,
        eff_del_success_rate=eff_del_succ,
        inv_hold_rate=inv_hold,
        overall_pass_rate=overall_pass,
        counterexample_ids=counterexample_ids,
        failure_signatures=dict(failure_sigs),
    )


def verify_all_contracts(
    contracts: Dict[str, SkillContract],
    records: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> Dict[str, VerificationReport]:
    """Verify all contracts against their respective segment instances.

    Parameters
    ----------
    contracts : dict[str, SkillContract]
    records : list[SegmentRecord]
    config : ContractVerificationConfig

    Returns
    -------
    dict[str, VerificationReport]
    """
    by_skill: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for rec in records:
        if rec.skill_label in contracts:
            by_skill[rec.skill_label].append(rec)

    reports: Dict[str, VerificationReport] = {}
    for skill_id, contract in contracts.items():
        instances = by_skill.get(skill_id, [])
        reports[skill_id] = verify_contract(contract, instances, config)

    return reports
