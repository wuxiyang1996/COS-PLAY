"""
Steps 5, 7 — Decide update actions and materialize NEW skills.

Deterministic policy (reproducible):

  KEEP   — overall pass rate >= min_pass_rate_keep, no structured failure clusters.
  REFINE — pass rate >= min_pass_rate_refine; drop high-violation literals,
           optionally demote to soft_pre.
  SPLIT  — failures cluster into >= 2 distinct effect modes with sufficient
           Jaccard gap; create child skills with per-cluster contracts.
  MATERIALIZE_NEW — repeated ``__NEW__`` segments cluster into a coherent group
           with a verifiable contract and distinctive effects.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from skill_agents_grpo.contract_verification.config import ContractVerificationConfig
from skill_agents_grpo.contract_verification.schemas import (
    SegmentRecord,
    SkillContract,
    UpdateAction,
    VerificationReport,
)
from skill_agents_grpo.contract_verification.clustering import (
    cluster_records,
    cluster_effect_jaccard_gap,
)
from skill_agents_grpo.contract_verification.contract_init import (
    build_initial_contracts,
    compute_all_effects,
)
from skill_agents_grpo.contract_verification.contract_verify import verify_contract

NEW_SKILL = "__NEW__"


# ── Step 5: decide action per skill ──────────────────────────────────

def decide_action(
    contract: SkillContract,
    report: VerificationReport,
    instances: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> UpdateAction:
    """Decide KEEP / REFINE / SPLIT for an existing skill.

    Uses a deterministic cascade:
      1. If pass rate >= min_pass_rate_keep → KEEP (unless structured failures).
      2. If failures cluster into >= 2 distinct modes → SPLIT.
      3. Else → REFINE (drop unstable literals).
    """
    vconf = config.verification
    cconf = config.clustering

    if report.overall_pass_rate >= vconf.min_pass_rate_keep:
        if not _has_structured_failures(report, instances, config):
            return UpdateAction(skill_id=contract.skill_id, action="KEEP")

    if len(instances) >= cconf.split_min_clusters * 2:
        if _should_split(report, instances, config):
            return _build_split_action(contract, instances, config)

    if report.overall_pass_rate >= vconf.min_pass_rate_refine:
        return _build_refine_action(contract, report, config)

    return _build_refine_action(contract, report, config)


def _has_structured_failures(
    report: VerificationReport,
    instances: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> bool:
    """Check whether failures cluster into distinct modes (not just noise)."""
    if len(report.failure_signatures) < 2:
        return False
    top_sigs = sorted(report.failure_signatures.values(), reverse=True)
    if len(top_sigs) >= 2 and top_sigs[1] >= 3:
        return True
    return False


def _should_split(
    report: VerificationReport,
    instances: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> bool:
    """Decide whether to attempt a SPLIT based on clustering feasibility."""
    labels, n_clusters, quality = cluster_records(instances, config.clustering)
    if n_clusters < config.clustering.split_min_clusters:
        return False
    gap = cluster_effect_jaccard_gap(instances, labels)
    return gap >= config.clustering.split_effect_jaccard_gap


# ── REFINE ───────────────────────────────────────────────────────────

def _build_refine_action(
    contract: SkillContract,
    report: VerificationReport,
    config: ContractVerificationConfig,
) -> UpdateAction:
    """Drop high-violation literals and optionally demote to soft_pre."""
    vconf = config.verification
    drop_thresh = vconf.violation_drop_thresh

    dropped: Dict[str, List[str]] = {"pre": [], "eff_add": [], "eff_del": [], "inv": []}
    demoted: List[str] = []

    for p, viol_rate in report.pre_violation_rate.items():
        if viol_rate > drop_thresh:
            if viol_rate < drop_thresh * 2:
                demoted.append(p)
            else:
                dropped["pre"].append(p)

    for p, succ_rate in report.eff_add_success_rate.items():
        if succ_rate < (1.0 - drop_thresh):
            dropped["eff_add"].append(p)

    for p, succ_rate in report.eff_del_success_rate.items():
        if succ_rate < (1.0 - drop_thresh):
            dropped["eff_del"].append(p)

    for p, hold_rate in report.inv_hold_rate.items():
        if hold_rate < config.aggregation.inv_freq:
            dropped["inv"].append(p)

    return UpdateAction(
        skill_id=contract.skill_id,
        action="REFINE",
        dropped_literals=dropped,
        demoted_to_soft=demoted,
        details={
            "pre_pass_rate": report.overall_pass_rate,
            "n_counterexamples": len(report.counterexample_ids),
        },
    )


def apply_refine(
    contract: SkillContract,
    action: UpdateAction,
) -> SkillContract:
    """Apply a REFINE action to a contract, mutating it in place."""
    for p in action.dropped_literals.get("pre", []):
        contract.pre.discard(p)
    for p in action.dropped_literals.get("eff_add", []):
        contract.eff_add.discard(p)
    for p in action.dropped_literals.get("eff_del", []):
        contract.eff_del.discard(p)
    for p in action.dropped_literals.get("inv", []):
        contract.inv.discard(p)
    for p in action.demoted_to_soft:
        contract.pre.discard(p)
        contract.soft_pre.add(p)
    contract.bump_version()
    return contract


# ── SPLIT ────────────────────────────────────────────────────────────

def _build_split_action(
    contract: SkillContract,
    instances: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> UpdateAction:
    """Cluster instances and create child skill ids."""
    labels, n_clusters, quality = cluster_records(instances, config.clustering)

    child_ids: List[str] = []
    for c in range(n_clusters):
        child_ids.append(f"{contract.skill_id}_v{contract.version + 1}c{c}")

    return UpdateAction(
        skill_id=contract.skill_id,
        action="SPLIT",
        new_skill_ids=child_ids,
        details={
            "n_clusters": n_clusters,
            "cluster_quality": quality,
            "labels": labels.tolist(),
        },
    )


def apply_split(
    contract: SkillContract,
    action: UpdateAction,
    instances: List[SegmentRecord],
    config: ContractVerificationConfig,
) -> Tuple[List[SkillContract], List[VerificationReport]]:
    """Execute a SPLIT: build per-cluster contracts and verify them.

    Returns
    -------
    children : list[SkillContract]
    reports : list[VerificationReport]
    """
    labels_list = action.details.get("labels", [])
    if not labels_list:
        labels, n_clusters, _ = cluster_records(instances, config.clustering)
        labels_list = labels.tolist()
    else:
        labels = np.array(labels_list)
        n_clusters = len(set(labels_list))

    clusters: Dict[int, List[SegmentRecord]] = defaultdict(list)
    for rec, label in zip(instances, labels_list):
        rec_copy = SegmentRecord(
            seg_id=rec.seg_id,
            traj_id=rec.traj_id,
            t_start=rec.t_start,
            t_end=rec.t_end,
            skill_label=action.new_skill_ids[int(label)] if int(label) < len(action.new_skill_ids) else rec.skill_label,
            P_start=rec.P_start,
            P_end=rec.P_end,
            P_all=rec.P_all,
            effects_add=rec.effects_add,
            effects_del=rec.effects_del,
            embedding=rec.embedding,
        )
        clusters[int(label)].append(rec_copy)

    contract.deprecated = True
    contract.children = action.new_skill_ids
    contract.bump_version()

    children: List[SkillContract] = []
    reports: List[VerificationReport] = []

    for c_idx, child_id in enumerate(action.new_skill_ids):
        cluster_recs = clusters.get(c_idx, [])
        if not cluster_recs:
            continue

        child_contracts = build_initial_contracts(
            cluster_recs, config.predicates, config.aggregation,
        )
        if child_id in child_contracts:
            child = child_contracts[child_id]
        else:
            child = SkillContract(skill_id=child_id, version=0)
            if cluster_recs:
                temp = build_initial_contracts(
                    cluster_recs, config.predicates, config.aggregation,
                )
                if temp:
                    first_contract = next(iter(temp.values()))
                    child.pre = first_contract.pre
                    child.eff_add = first_contract.eff_add
                    child.eff_del = first_contract.eff_del
                    child.inv = first_contract.inv

        children.append(child)
        report = verify_contract(child, cluster_recs, config)
        reports.append(report)

    return children, reports


# ── Step 7: MATERIALIZE_NEW ─────────────────────────────────────────

def materialize_new_skills(
    new_records: List[SegmentRecord],
    existing_contracts: Dict[str, SkillContract],
    config: ContractVerificationConfig,
) -> Tuple[List[SkillContract], List[UpdateAction], List[VerificationReport], bool]:
    """Turn repeated ``__NEW__`` segments into new bank skills.

    Procedure:
      1. Cluster NEW_POOL by effect signatures.
      2. For each sufficiently large cluster, learn a contract.
      3. If pass rate and effect distinctiveness are sufficient, create a new skill.

    Parameters
    ----------
    new_records : list[SegmentRecord]
        All segments labelled ``__NEW__``.
    existing_contracts : dict[str, SkillContract]
        Current bank for distinctiveness check.
    config : ContractVerificationConfig

    Returns
    -------
    new_contracts : list[SkillContract]
    actions : list[UpdateAction]
    reports : list[VerificationReport]
    resegment_needed : bool
        True if any new skills were created (Stage 2 should re-run).
    """
    new_cfg = config.new_skill

    if len(new_records) < new_cfg.min_cluster_size:
        return [], [], [], False

    compute_all_effects(new_records, config.predicates, config.aggregation)
    labels, n_clusters, quality = cluster_records(new_records, config.clustering)

    clusters: Dict[int, List[SegmentRecord]] = defaultdict(list)
    for rec, label in zip(new_records, labels):
        clusters[int(label)].append(rec)

    new_contracts: List[SkillContract] = []
    actions: List[UpdateAction] = []
    reports: List[VerificationReport] = []
    created_any = False
    ts = int(time.time())

    for c_idx, cluster_recs in clusters.items():
        if len(cluster_recs) < new_cfg.min_cluster_size:
            continue

        new_id = f"S_new_{ts}_{c_idx}"
        for rec in cluster_recs:
            rec.skill_label = new_id

        candidate_contracts = build_initial_contracts(
            cluster_recs, config.predicates, config.aggregation,
        )
        if new_id not in candidate_contracts:
            continue
        candidate = candidate_contracts[new_id]

        report = verify_contract(candidate, cluster_recs, config)
        if report.overall_pass_rate < new_cfg.min_pass_rate_create:
            continue

        if not _is_distinctive(candidate, existing_contracts, new_cfg.min_effect_distinctiveness):
            continue

        new_contracts.append(candidate)
        reports.append(report)
        actions.append(UpdateAction(
            skill_id=NEW_SKILL,
            action="MATERIALIZE_NEW",
            new_skill_ids=[new_id],
            details={
                "cluster_size": len(cluster_recs),
                "pass_rate": report.overall_pass_rate,
                "seg_ids": [r.seg_id for r in cluster_recs],
            },
        ))
        created_any = True

    return new_contracts, actions, reports, created_any


def _is_distinctive(
    candidate: SkillContract,
    existing: Dict[str, SkillContract],
    min_dist: float,
) -> bool:
    """Check that a candidate skill's effects are sufficiently different from all existing skills."""
    cand_effects = candidate.eff_add | candidate.eff_del
    if not cand_effects:
        return False

    for eid, econtr in existing.items():
        if econtr.deprecated:
            continue
        exist_effects = econtr.eff_add | econtr.eff_del
        if not exist_effects:
            continue
        union = len(cand_effects | exist_effects)
        inter = len(cand_effects & exist_effects)
        jacc_dist = 1.0 - (inter / union) if union > 0 else 1.0
        if jacc_dist < min_dist:
            return False
    return True
