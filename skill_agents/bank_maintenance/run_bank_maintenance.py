"""
Bank Maintenance — Orchestrator: Split / Merge / Refine + fast re-decode.

Run order:
  1. Build / update SkillProfiles (only for changed skills).
  2. SplitQueue → execute splits → local re-decode.
  3. MergeCandidates → verify → execute merges → local re-decode if needed.
  4. RefineQueue → refine contracts + duration/start-end updates.
  5. Update indices incrementally.
  6. Emit bank diff report.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents.bank_maintenance.config import BankMaintenanceConfig
from skill_agents.bank_maintenance.duration_model import DurationModelStore
from skill_agents.bank_maintenance.indices import (
    EffectInvertedIndex,
    EmbeddingANN,
    MinHashLSH,
)
from skill_agents.bank_maintenance.local_redecode import (
    build_redecode_requests,
    collect_affected_trajectories,
    redecode_windows,
    relabel_via_alias,
)
from skill_agents.bank_maintenance.merge import (
    MergeResult,
    execute_merge,
    retrieve_merge_candidates,
    verify_merge_pair,
)
from skill_agents.bank_maintenance.refine import (
    RefineResult,
    check_refine_triggers,
    extract_confusion_partners,
    refine_skill,
    update_duration_model,
)
from skill_agents.bank_maintenance.schemas import (
    BankDiffEntry,
    BankDiffReport,
    DiffOp,
    RedecodeRequest,
    SkillProfile,
)
from skill_agents.bank_maintenance.split import (
    SplitResult,
    check_split_triggers,
    execute_split,
    redecode_requests_for_split,
)

logger = logging.getLogger(__name__)


# ── Pipeline result ──────────────────────────────────────────────────


class BankMaintenanceResult:
    """Collects all outputs from a bank maintenance run."""

    def __init__(self) -> None:
        self.diff_report = BankDiffReport()
        self.profiles: Dict[str, SkillProfile] = {}
        self.split_results: List[SplitResult] = []
        self.merge_results: List[MergeResult] = []
        self.refine_results: List[RefineResult] = []
        self.materialized_ids: List[str] = []
        self.promoted_ids: List[str] = []
        self.redecode_requests: List[RedecodeRequest] = []
        self.alias_map: Dict[str, str] = {}

    def to_dict(self) -> dict:
        return {
            "diff_report": self.diff_report.to_dict(),
            "n_profiles": len(self.profiles),
            "splits": [
                {
                    "parent": sr.parent_id,
                    "accepted": sr.accepted,
                    "children": [c.skill_id for c in sr.children],
                }
                for sr in self.split_results
            ],
            "merges": [
                {
                    "canonical": mr.canonical_id,
                    "merged_ids": mr.merged_ids,
                    "accepted": mr.accepted,
                }
                for mr in self.merge_results
            ],
            "refines": [
                {
                    "skill_id": rr.skill_id,
                    "dropped": rr.dropped_literals,
                    "added": rr.added_literals,
                }
                for rr in self.refine_results
            ],
            "materialized": self.materialized_ids,
            "promoted": self.promoted_ids,
            "n_redecode_requests": len(self.redecode_requests),
            "alias_map": self.alias_map,
        }


# ═════════════════════════════════════════════════════════════════════
# Profile builder
# ═════════════════════════════════════════════════════════════════════


def build_profile(
    skill_id: str,
    contract: SkillEffectsContract,
    report: Optional[VerificationReport],
    instances: List[SegmentRecord],
    transition_bigrams: Optional[Dict[str, Counter]] = None,
    embeddings: Optional[Dict[str, List[float]]] = None,
    topk: int = 5,
) -> SkillProfile:
    """Build a SkillProfile from contract, report, and instance data."""
    eff_add = frozenset(contract.eff_add)
    eff_del = frozenset(contract.eff_del)
    eff_event = frozenset(contract.eff_event)

    all_eff = eff_add | eff_del | eff_event
    sig_hash = hash(tuple(sorted(all_eff)))

    sparse_vec: Dict[str, float] = {}
    n = contract.n_instances or len(instances) or 1
    for lit, cnt in contract.support.items():
        sparse_vec[lit] = cnt / n

    lengths = [inst.t_end - inst.t_start for inst in instances]
    dur_mean = sum(lengths) / len(lengths) if lengths else 0.0
    dur_var = (
        sum((l - dur_mean) ** 2 for l in lengths) / len(lengths)
        if len(lengths) > 1
        else 0.0
    )

    emb_centroid: Optional[List[float]] = None
    emb_var: Optional[List[float]] = None
    if embeddings:
        seg_embs = [
            embeddings[inst.seg_id]
            for inst in instances
            if inst.seg_id in embeddings
        ]
        if seg_embs:
            dim = len(seg_embs[0])
            emb_centroid = [
                sum(e[d] for e in seg_embs) / len(seg_embs) for d in range(dim)
            ]
            emb_var = [
                sum((e[d] - emb_centroid[d]) ** 2 for e in seg_embs) / len(seg_embs)
                for d in range(dim)
            ]

    topk_prev: List[Tuple[str, float]] = []
    topk_next: List[Tuple[str, float]] = []
    if transition_bigrams:
        prev_counts = transition_bigrams.get(f"_to_{skill_id}", Counter())
        next_counts = transition_bigrams.get(f"{skill_id}_to_", Counter())
        topk_prev = prev_counts.most_common(topk)
        topk_next = next_counts.most_common(topk)

    pass_rate = report.overall_pass_rate if report else 0.0
    fail_sigs = report.failure_signatures if report else {}
    top_viol = _top_violating(report) if report else []

    return SkillProfile(
        skill_id=skill_id,
        eff_add=eff_add,
        eff_del=eff_del,
        eff_event=eff_event,
        effect_signature_hash=sig_hash,
        effect_sparse_vec=sparse_vec,
        embedding_centroid=emb_centroid,
        embedding_var_diag=emb_var,
        transition_topk_prev=topk_prev,
        transition_topk_next=topk_next,
        duration_mean=dur_mean,
        duration_var=dur_var,
        overall_pass_rate=pass_rate,
        top_violating_literals=top_viol,
        failure_signature_counts=fail_sigs,
        n_instances=len(instances),
    )


def _top_violating(report: VerificationReport, n: int = 5) -> List[str]:
    all_rates: List[Tuple[str, float]] = []
    for p, r in report.eff_add_success_rate.items():
        all_rates.append((f"add:{p}", r))
    for p, r in report.eff_del_success_rate.items():
        all_rates.append((f"del:{p}", r))
    for e, r in report.eff_event_rate.items():
        all_rates.append((f"evt:{e}", r))
    all_rates = [(lit, r) for lit, r in all_rates if r < 1.0]
    all_rates.sort(key=lambda x: x[1])
    return [lit for lit, _ in all_rates[:n]]


# ═════════════════════════════════════════════════════════════════════
# Transition bigram builder
# ═════════════════════════════════════════════════════════════════════


def build_transition_bigrams(
    all_segments: List[SegmentRecord],
) -> Dict[str, Counter]:
    """Build forward and backward bigram counts from ordered segment lists."""
    by_traj: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for seg in all_segments:
        by_traj[seg.traj_id].append(seg)

    bigrams: Dict[str, Counter] = defaultdict(Counter)

    for traj_id, segs in by_traj.items():
        ordered = sorted(segs, key=lambda s: s.t_start)
        for i in range(len(ordered) - 1):
            curr = ordered[i].skill_label
            nxt = ordered[i + 1].skill_label
            bigrams[f"{curr}_to_"][nxt] += 1
            bigrams[f"_to_{nxt}"][curr] += 1

    return dict(bigrams)


def _collect_curator_candidates(
    result: BankMaintenanceResult,
    bank: Optional[SkillBankMVP] = None,
) -> List[Dict[str, Any]]:
    """Summarize all maintenance actions as curator candidates.

    Covers all 5 action types: split, merge, refine, materialize, promote.
    Uses ``"type"`` as the key for the action kind, matching
    ``_format_action`` in ``llm_curator.py``.
    """
    def _get_skill_score(sid: str) -> float:
        if bank is None:
            return 0.5
        skill = bank.get_skill(sid)
        return skill.compute_skill_score() if skill else 0.5

    candidates: List[Dict[str, Any]] = []
    for sr in result.split_results:
        if sr.accepted:
            candidates.append({
                "type": "split",
                "skill_id": sr.parent_id,
                "skill_score": _get_skill_score(sr.parent_id),
                "n_instances": len(sr.children),
                "details": {
                    "children": [c.skill_id for c in sr.children],
                },
            })
    for mr in result.merge_results:
        if mr.accepted:
            candidates.append({
                "type": "merge",
                "skill_id": mr.canonical_id,
                "skill_score": _get_skill_score(mr.canonical_id),
                "pass_rate": mr.report.overall_pass_rate if mr.report else 0,
                "n_instances": len(mr.merged_ids),
                "details": {
                    "merged_ids": mr.merged_ids,
                },
            })
    for rr in result.refine_results:
        if rr.new_contract is not None:
            candidates.append({
                "type": "refine",
                "skill_id": rr.skill_id,
                "skill_score": _get_skill_score(rr.skill_id),
                "details": {
                    "dropped": rr.dropped_literals[:5] if rr.dropped_literals else [],
                    "added": rr.added_literals[:5] if rr.added_literals else [],
                },
            })
    for mid in result.materialized_ids:
        skill = bank.get_skill(mid) if bank else None
        candidates.append({
            "type": "materialize",
            "skill_id": mid,
            "skill_score": _get_skill_score(mid),
            "trigger": "recurring pattern",
            "n_instances": len(getattr(skill, "sub_episodes", [])) if skill else 0,
            "details": {
                "name": getattr(skill, "name", mid) if skill else mid,
            },
        })
    for pid in result.promoted_ids:
        skill = bank.get_skill(pid) if bank else None
        report = bank.get_report(pid) if bank else None
        candidates.append({
            "type": "promote",
            "skill_id": pid,
            "skill_score": _get_skill_score(pid),
            "trigger": "proto-skill qualified",
            "pass_rate": report.overall_pass_rate if report else 0,
            "n_instances": getattr(skill, "n_instances", 0) if skill else 0,
            "details": {
                "name": getattr(skill, "name", pid) if skill else pid,
            },
        })
    return candidates


def _build_curator_outcomes(
    candidates: List[Dict[str, Any]],
    result: "BankMaintenanceResult",
    bank: Optional[SkillBankMVP] = None,
) -> List[Dict[str, Any]]:
    """Build per-candidate ground-truth outcomes for curator reward.

    Each entry: ``{"succeeded": bool, "quality_delta": float}``.
    ``succeeded`` is True if the action improved or maintained bank quality
    (accepted split with good child pass rates, accepted merge, etc.).
    """
    split_map = {sr.parent_id: sr for sr in result.split_results}
    merge_map = {mr.canonical_id: mr for mr in result.merge_results}
    refine_map = {rr.skill_id: rr for rr in result.refine_results}

    outcomes: List[Dict[str, Any]] = []
    for cand in candidates:
        atype = cand.get("type", "")
        sid = cand.get("skill_id", "")
        succeeded = False
        qd = 0.0

        if atype == "split":
            sr = split_map.get(sid)
            if sr and sr.accepted and sr.children:
                child_prs = [
                    c.report.overall_pass_rate
                    for c in sr.children if c.report
                ]
                if child_prs:
                    succeeded = True
                    qd = sum(child_prs) / len(child_prs) - cand.get("skill_score", 0.5)

        elif atype == "merge":
            mr = merge_map.get(sid)
            if mr and mr.accepted and mr.report:
                succeeded = mr.report.overall_pass_rate > 0.5
                qd = mr.report.overall_pass_rate - cand.get("pass_rate", 0.5)

        elif atype == "refine":
            rr = refine_map.get(sid)
            if rr and rr.new_contract is not None:
                new_rpt = bank.get_report(sid) if bank else None
                if new_rpt:
                    succeeded = new_rpt.overall_pass_rate > cand.get("skill_score", 0.5)
                    qd = new_rpt.overall_pass_rate - cand.get("skill_score", 0.5)
                else:
                    succeeded = True
                    qd = 0.05

        elif atype == "materialize":
            succeeded = True
            qd = 0.05

        elif atype == "promote":
            pr = cand.get("pass_rate", 0)
            ni = cand.get("n_instances", 0)
            succeeded = pr > 0.5 and ni >= 3
            qd = pr - 0.5

        outcomes.append({"succeeded": succeeded, "quality_delta": qd})
    return outcomes


# ═════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════


def run_bank_maintenance(
    bank: SkillBankMVP,
    all_segments: List[SegmentRecord],
    config: Optional[BankMaintenanceConfig] = None,
    embeddings: Optional[Dict[str, List[float]]] = None,
    stage2_diagnostics: Optional[List[dict]] = None,
    decode_fn: Optional[Callable] = None,
    traj_lengths: Optional[Dict[str, int]] = None,
    report_path: Optional[str] = None,
) -> BankMaintenanceResult:
    """Run the full bank maintenance pipeline.

    Parameters
    ----------
    bank : SkillBankMVP
        Skill bank with contracts and reports from contract verification.
    all_segments : list[SegmentRecord]
        Every segment record (with effects computed).
    config : BankMaintenanceConfig, optional
    embeddings : dict[str, list[float]], optional
        ``seg_id -> embedding`` for multimodality checks.
    stage2_diagnostics : list[dict], optional
        ``SegmentationResult.to_dict()["segments"]`` for confusion extraction.
    decode_fn : callable, optional
        ``(traj_id, window_start, window_end, skill_ids) -> list[dict]``
        for local re-decode.
    traj_lengths : dict[str, int], optional
        ``traj_id -> length`` for clamping re-decode windows.
    report_path : str, optional
        Path to write JSON bank diff report.

    Returns
    -------
    BankMaintenanceResult
    """
    if config is None:
        config = BankMaintenanceConfig()

    result = BankMaintenanceResult()
    diff = result.diff_report

    # ── Group segments by skill ──────────────────────────────────
    by_skill: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for seg in all_segments:
        if seg.skill_label.upper() != "NEW":
            by_skill[seg.skill_label].append(seg)

    # ── Transition bigrams ───────────────────────────────────────
    bigrams = build_transition_bigrams(all_segments)

    # ── 1. Build SkillProfiles ───────────────────────────────────
    logger.info("BankMaint 1: Building skill profiles (%d skills)", len(bank))
    profiles: Dict[str, SkillProfile] = {}

    for skill_id in bank.skill_ids:
        contract = bank.get_contract(skill_id)
        report = bank.get_report(skill_id)
        if contract is None:
            continue
        instances = by_skill.get(skill_id, [])
        profiles[skill_id] = build_profile(
            skill_id, contract, report, instances,
            transition_bigrams=bigrams,
            embeddings=embeddings,
            topk=config.profile_topk_transitions,
        )

    result.profiles = profiles

    # ── Build indices ────────────────────────────────────────────
    inv_index = EffectInvertedIndex()
    inv_index.build_from_profiles(profiles)

    lsh = MinHashLSH(
        num_perm=config.minhash_num_perm,
        threshold=config.lsh_threshold,
    )
    lsh.build_from_profiles(profiles)

    ann: Optional[EmbeddingANN] = None
    if embeddings:
        ann = EmbeddingANN()
        ann.build_from_profiles(profiles)

    # ── Duration model store ─────────────────────────────────────
    duration_store = DurationModelStore(
        n_bins=config.duration_n_bins,
        min_len=config.duration_min_len,
        max_len=config.duration_max_len,
        smoothing=config.duration_smoothing,
    )
    for skill_id, instances in by_skill.items():
        if skill_id in profiles:
            update_duration_model(duration_store, skill_id, instances)

    # ── 2. SPLIT ─────────────────────────────────────────────────
    logger.info("BankMaint 2: Checking split triggers")
    split_queue: List[str] = []
    for skill_id, prof in profiles.items():
        instances = by_skill.get(skill_id, [])
        triggered, reason = check_split_triggers(
            prof, instances, config, embeddings,
        )
        if triggered:
            split_queue.append(skill_id)
            logger.debug("Split triggered for %s: %s", skill_id, reason)

    already_merged: Set[str] = set()

    for skill_id in split_queue:
        instances = by_skill.get(skill_id, [])
        if len(instances) < 2 * config.min_child_size:
            continue

        contract = bank.get_contract(skill_id)
        if contract is None:
            continue

        sr = execute_split(
            skill_id, instances, config,
            parent_version=contract.version,
        )
        result.split_results.append(sr)

        if sr.accepted:
            for child in sr.children:
                bank.add_or_update(child.contract, child.report)
                profiles[child.skill_id] = build_profile(
                    child.skill_id, child.contract, child.report,
                    [s for s in instances if s.seg_id in set(child.instance_seg_ids)],
                    transition_bigrams=bigrams,
                    embeddings=embeddings,
                )
                inv_index.update_skill(child.skill_id, profiles[child.skill_id].all_effects)
                lsh.update_skill(child.skill_id, profiles[child.skill_id].all_effects)

            bank.remove(skill_id)
            inv_index.remove(skill_id)
            lsh.remove(skill_id)
            profiles.pop(skill_id, None)
            already_merged.add(skill_id)

            diff.add(BankDiffEntry(
                op=DiffOp.SPLIT,
                skill_id=skill_id,
                details={
                    "children": [c.skill_id for c in sr.children],
                    "child_pass_rates": [
                        c.report.overall_pass_rate
                        for c in sr.children if c.report
                    ],
                },
            ))

            reqs = redecode_requests_for_split(
                skill_id, instances, config, sr.children,
            )
            result.redecode_requests.extend(reqs)

    # ── 3. MERGE ─────────────────────────────────────────────────
    logger.info("BankMaint 3: Retrieving merge candidates")
    candidate_pairs = retrieve_merge_candidates(
        profiles, inv_index, lsh, ann, config,
    )
    logger.info("  %d candidate pairs", len(candidate_pairs))

    merged_away: Set[str] = set()

    for pair in candidate_pairs:
        k1, k2 = sorted(pair)
        if k1 in merged_away or k2 in merged_away:
            continue
        if k1 in already_merged or k2 in already_merged:
            continue

        p1 = profiles.get(k1)
        p2 = profiles.get(k2)
        if p1 is None or p2 is None:
            continue

        passed, scores = verify_merge_pair(p1, p2, config)
        if not passed:
            continue

        instances_k1 = by_skill.get(k1, [])
        instances_k2 = by_skill.get(k2, [])

        mr = execute_merge(
            k1, k2, instances_k1, instances_k2, config,
            prev_version=max(
                (bank.get_contract(k1) or SkillEffectsContract(skill_id=k1)).version,
                (bank.get_contract(k2) or SkillEffectsContract(skill_id=k2)).version,
            ),
        )
        result.merge_results.append(mr)

        if mr.accepted and mr.contract is not None:
            canonical = mr.canonical_id
            retired = [sid for sid in mr.merged_ids if sid != canonical][0]

            bank.add_or_update(mr.contract, mr.report)
            bank.remove(retired)

            result.alias_map.update(mr.alias_map)
            merged_away.add(retired)

            merged_instances = instances_k1 + instances_k2
            by_skill[canonical] = merged_instances
            profiles[canonical] = build_profile(
                canonical, mr.contract, mr.report, merged_instances,
                transition_bigrams=bigrams,
                embeddings=embeddings,
            )
            inv_index.update_skill(canonical, profiles[canonical].all_effects)
            lsh.update_skill(canonical, profiles[canonical].all_effects)

            profiles.pop(retired, None)
            inv_index.remove(retired)
            lsh.remove(retired)

            diff.add(BankDiffEntry(
                op=DiffOp.MERGE,
                skill_id=canonical,
                details={
                    "retired": retired,
                    "scores": scores,
                    "pass_rate": mr.report.overall_pass_rate if mr.report else 0,
                },
            ))

            update_duration_model(duration_store, canonical, merged_instances)

    # ── 4. REFINE ────────────────────────────────────────────────
    logger.info("BankMaint 4: Checking refine triggers")

    for skill_id, prof in list(profiles.items()):
        if skill_id in merged_away:
            continue

        confuser_ids: List[str] = []
        if stage2_diagnostics:
            confuser_ids = extract_confusion_partners(
                skill_id, stage2_diagnostics,
                top_n=config.refine_top_confusers,
            )

        triggered, reason = check_refine_triggers(prof, confuser_ids, config)
        if not triggered:
            instances = by_skill.get(skill_id, [])
            if instances:
                update_duration_model(duration_store, skill_id, instances)
                diff.add(BankDiffEntry(
                    op=DiffOp.DURATION_UPDATE,
                    skill_id=skill_id,
                    details={"n_instances": len(instances)},
                ))
            continue

        contract = bank.get_contract(skill_id)
        report = bank.get_report(skill_id)
        instances = by_skill.get(skill_id, [])

        if contract is None or report is None or not instances:
            continue

        confuser_instances: Dict[str, List[SegmentRecord]] = {}
        for cid in confuser_ids:
            if cid in by_skill:
                confuser_instances[cid] = by_skill[cid]

        rr = refine_skill(
            contract, report, instances, confuser_instances, config,
        )
        result.refine_results.append(rr)

        if rr.new_contract is not None:
            bank.add_or_update(rr.new_contract)

            from skill_agents.stage3_mvp.contract_verify import (
                verify_effects_contract,
            )
            from skill_agents.stage3_mvp.config import Stage3MVPConfig

            s3_cfg = Stage3MVPConfig(
                eff_freq=config.eff_freq,
                instance_pass_literal_frac=config.instance_pass_literal_frac,
            )
            new_report = verify_effects_contract(rr.new_contract, instances, s3_cfg)
            bank.add_or_update(rr.new_contract, new_report)

            profiles[skill_id] = build_profile(
                skill_id, rr.new_contract, new_report, instances,
                transition_bigrams=bigrams,
                embeddings=embeddings,
            )
            inv_index.update_skill(skill_id, profiles[skill_id].all_effects)
            lsh.update_skill(skill_id, profiles[skill_id].all_effects)

            diff.add(BankDiffEntry(
                op=DiffOp.REFINE,
                skill_id=skill_id,
                details={
                    "dropped": rr.dropped_literals,
                    "added": rr.added_literals,
                    "new_pass_rate": new_report.overall_pass_rate,
                },
            ))

        update_duration_model(duration_store, skill_id, instances)
        diff.add(BankDiffEntry(
            op=DiffOp.DURATION_UPDATE,
            skill_id=skill_id,
            details={"n_instances": len(instances)},
        ))

    # ── 4b. CURATOR — LLM filtering of maintenance actions ──────
    # Summarize all proposed actions and call the curator so GRPO
    # training data is generated for the curator adapter.
    curator_candidates = _collect_curator_candidates(result, bank=bank)
    if curator_candidates:
        try:
            from skill_agents.bank_maintenance.llm_curator import (
                filter_candidates,
                set_curator_reward_context,
            )

            action_outcomes = _build_curator_outcomes(
                curator_candidates, result, bank,
            )
            set_curator_reward_context(action_outcomes=action_outcomes)
            filter_candidates(curator_candidates, bank)
        except Exception as exc:
            logger.debug("Curator filtering skipped: %s", exc)

    # ── 5. Execute local re-decode (if decode_fn provided) ───────
    if decode_fn and result.redecode_requests:
        logger.info(
            "BankMaint 5: Executing %d re-decode requests",
            len(result.redecode_requests),
        )
        redecode_windows(result.redecode_requests, decode_fn)

    # ── 6. Finalise ──────────────────────────────────────────────
    diff.finalize()
    result.profiles = profiles

    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Bank diff report written to %s", report_path)

    logger.info(
        "Bank maintenance complete: %d splits, %d merges, %d refines",
        diff.n_splits, diff.n_merges, diff.n_refines,
    )

    return result
