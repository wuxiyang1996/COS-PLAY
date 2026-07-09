"""
Skill Evaluation — Main Orchestrator (LLM-agentic).

Runs LLM-as-a-judge on every skill in the bank across six quality
dimensions, optionally followed by a holistic synthesis pass.  All
quality reasoning is performed by the LLM — no hardcoded heuristic
thresholds.

Integration points:
  - Consumes the same ``SegmentRecord`` / ``SkillEffectsContract`` /
    ``VerificationReport`` / ``SkillProfile`` types used by Stage 3
    and Bank Maintenance.
  - Produces ``EvaluationSummary`` which can feed back into:
      * Bank Maintenance (guide split/merge/refine priorities).
      * Stage 2 LLM teacher (highlight low-quality skills for re-labelling).
      * Downstream agents (filter or weight skills during planning).

Typical call site::

    from skill_agents.skill_evaluation import run_skill_evaluation
    summary = run_skill_evaluation(bank, all_segments)
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents.bank_maintenance.schemas import SkillProfile
from skill_agents.skill_evaluation.config import SkillEvaluationConfig
from skill_agents.skill_evaluation.schemas import (
    EvaluationSummary,
    QualityDimension,
    SkillQualityReport,
)
from skill_agents.skill_evaluation.evaluators import (
    evaluate_coherence,
    evaluate_composability,
    evaluate_discriminability,
    evaluate_generalization,
    evaluate_granularity,
    evaluate_holistic,
    evaluate_utility,
)

logger = logging.getLogger(__name__)


# ── Lightweight profile builder ──────────────────────────────────────

def _build_lightweight_profile(
    skill_id: str,
    contract: SkillEffectsContract,
    report: Optional[VerificationReport],
    instances: List[SegmentRecord],
) -> SkillProfile:
    """Build a SkillProfile without importing bank_maintenance internals."""
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

    return SkillProfile(
        skill_id=skill_id,
        eff_add=eff_add,
        eff_del=eff_del,
        eff_event=eff_event,
        effect_signature_hash=sig_hash,
        effect_sparse_vec=sparse_vec,
        duration_mean=dur_mean,
        duration_var=dur_var,
        overall_pass_rate=report.overall_pass_rate if report else 0.0,
        n_instances=len(instances),
    )


# ── Transition bigram builder ────────────────────────────────────────

def _build_transition_bigrams(
    all_segments: List[SegmentRecord],
) -> Dict[str, Counter]:
    by_traj: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for seg in all_segments:
        by_traj[seg.traj_id].append(seg)

    bigrams: Dict[str, Counter] = defaultdict(Counter)
    for _, segs in by_traj.items():
        ordered = sorted(segs, key=lambda s: s.t_start)
        for i in range(len(ordered) - 1):
            curr = ordered[i].skill_label
            nxt = ordered[i + 1].skill_label
            bigrams[f"{curr}_to_"][nxt] += 1
            bigrams[f"_to_{nxt}"][curr] += 1

    return dict(bigrams)


# ── Per-trajectory pass-rate estimation ──────────────────────────────

def _per_trajectory_pass_rates(
    skill_id: str,
    contract: SkillEffectsContract,
    instances: List[SegmentRecord],
) -> Dict[str, float]:
    """Estimate contract pass rate per trajectory (simplified)."""
    by_traj: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for inst in instances:
        by_traj[inst.traj_id].append(inst)

    rates: Dict[str, float] = {}
    for traj_id, traj_instances in by_traj.items():
        if not traj_instances:
            continue
        passes = 0
        for inst in traj_instances:
            ok = True
            for lit in contract.eff_add:
                if lit not in inst.eff_add:
                    ok = False
                    break
            if ok:
                for lit in contract.eff_del:
                    if lit not in inst.eff_del:
                        ok = False
                        break
            if ok:
                passes += 1
        rates[traj_id] = passes / len(traj_instances)
    return rates


# ═════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════


def run_skill_evaluation(
    bank: SkillBankMVP,
    all_segments: List[SegmentRecord],
    config: Optional[SkillEvaluationConfig] = None,
    profiles: Optional[Dict[str, SkillProfile]] = None,
    episode_outcomes: Optional[Dict[str, bool]] = None,
    report_path: Optional[str] = None,
) -> EvaluationSummary:
    """Run the full LLM-agentic skill evaluation pipeline.

    For each skill in the bank, calls the LLM judge on each enabled
    quality dimension, then optionally runs a holistic synthesis pass.

    Parameters
    ----------
    bank : SkillBankMVP
        Skill bank containing contracts and verification reports.
    all_segments : list[SegmentRecord]
        All segment records (with effects computed) from Stage 3.
    config : SkillEvaluationConfig, optional
    profiles : dict[str, SkillProfile], optional
        Pre-built profiles (from bank maintenance); built on the fly
        if not provided.
    episode_outcomes : dict[str, bool], optional
        ``traj_id -> success`` for utility scoring.
    report_path : str, optional
        Path to write JSON evaluation report.

    Returns
    -------
    EvaluationSummary
        Bank-wide evaluation summary with per-skill reports.
    """
    cfg = config or SkillEvaluationConfig()
    llm_cfg = cfg.llm
    summary = EvaluationSummary()

    # Group segments by skill
    by_skill: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for seg in all_segments:
        if seg.skill_label.upper() != "NEW":
            by_skill[seg.skill_label].append(seg)

    # Build transition bigrams
    bigrams = _build_transition_bigrams(all_segments)

    # Build or reuse profiles
    if profiles is None:
        profiles = {}
        for skill_id in bank.skill_ids:
            contract = bank.get_contract(skill_id)
            if contract is None:
                continue
            report = bank.get_report(skill_id)
            instances = by_skill.get(skill_id, [])
            profiles[skill_id] = _build_lightweight_profile(
                skill_id, contract, report, instances,
            )

    all_skill_ids = list(profiles.keys())
    enabled = set(cfg.enabled_dimensions)

    logger.info(
        "Skill Evaluation (LLM-agentic): evaluating %d skills across %d dimensions",
        len(all_skill_ids), len(enabled),
    )

    # Evaluate each skill
    for skill_id in all_skill_ids:
        contract = bank.get_contract(skill_id)
        if contract is None:
            continue
        report = bank.get_report(skill_id)
        instances = by_skill.get(skill_id, [])
        profile = profiles.get(skill_id)

        if len(instances) < cfg.min_instances_for_eval:
            logger.debug(
                "Skipping %s: only %d instances (need %d)",
                skill_id, len(instances), cfg.min_instances_for_eval,
            )
            continue

        skill_report = SkillQualityReport(
            skill_id=skill_id,
            version=contract.version,
        )

        logger.info("  Evaluating skill: %s (%d instances)", skill_id, len(instances))

        # ── Dimension evaluations (each is an LLM call) ──────────

        if "coherence" in enabled:
            logger.debug("    [%s] coherence", skill_id)
            coh = evaluate_coherence(
                skill_id, contract, instances, report, llm_cfg,
            )
            skill_report.dimensions[QualityDimension.COHERENCE.value] = coh

        if "discriminability" in enabled:
            logger.debug("    [%s] discriminability", skill_id)
            disc = evaluate_discriminability(
                skill_id, contract, profiles, llm_cfg,
            )
            skill_report.dimensions[QualityDimension.DISCRIMINABILITY.value] = disc

        if "composability" in enabled:
            logger.debug("    [%s] composability", skill_id)
            comp = evaluate_composability(
                skill_id, contract, bigrams, all_skill_ids, llm_cfg,
            )
            skill_report.dimensions[QualityDimension.COMPOSABILITY.value] = comp

        if "generalization" in enabled:
            logger.debug("    [%s] generalization", skill_id)
            per_traj_pr = _per_trajectory_pass_rates(skill_id, contract, instances)
            gen = evaluate_generalization(
                skill_id, contract, instances, report, per_traj_pr, llm_cfg,
            )
            skill_report.dimensions[QualityDimension.GENERALIZATION.value] = gen

        if "utility" in enabled:
            logger.debug("    [%s] utility", skill_id)
            util = evaluate_utility(
                skill_id, contract, instances, episode_outcomes, llm_cfg,
            )
            skill_report.dimensions[QualityDimension.UTILITY.value] = util

        if "granularity" in enabled:
            logger.debug("    [%s] granularity", skill_id)
            gran = evaluate_granularity(
                skill_id, contract, instances, profile, llm_cfg,
            )
            skill_report.dimensions[QualityDimension.GRANULARITY.value] = gran

        # ── Weighted overall from dimension scores ───────────────
        skill_report.compute_overall(cfg.dimension_weights)

        # ── Holistic synthesis pass (optional, one more LLM call) ─
        if cfg.run_holistic_pass:
            logger.debug("    [%s] holistic synthesis", skill_id)
            holistic = evaluate_holistic(
                skill_id, contract, skill_report.dimensions, llm_cfg,
            )
            _apply_holistic(skill_report, holistic)

        summary.skill_reports[skill_id] = skill_report

        logger.info(
            "  %s: %.3f (%s) %s",
            skill_id,
            skill_report.overall_score,
            skill_report.overall_grade.value,
            " ".join(f"[{a}]" for a in _action_tags(skill_report)),
        )

    # Bank-level summary
    summary.compute_summary()

    # Persist
    path = report_path or cfg.report_path
    if path:
        _save_report(summary, path)

    logger.info(
        "Skill Evaluation complete: mean=%.3f, %d excellent, %d good, "
        "%d fair, %d poor, %d failing",
        summary.mean_overall,
        summary.n_excellent, summary.n_good,
        summary.n_fair, summary.n_poor, summary.n_failing,
    )

    return summary


# ── Apply holistic LLM judgement ─────────────────────────────────────

def _apply_holistic(
    report: SkillQualityReport,
    holistic: dict,
) -> None:
    """Apply the holistic LLM synthesis to a SkillQualityReport."""
    raw_score = holistic.get("score", 5)
    if isinstance(raw_score, (int, float)):
        report.overall_score = max(0.0, min(float(raw_score) / 10.0, 1.0))

    from skill_agents.skill_evaluation.schemas import QualityGrade
    report.overall_grade = QualityGrade.from_score(report.overall_score)

    recommendation = holistic.get("recommendation", "KEEP").upper()
    if recommendation == "DISCARD":
        report.recommend_discard = True
    elif recommendation == "SPLIT":
        report.recommend_split = True
    elif recommendation == "MERGE":
        merge_with = holistic.get("merge_with")
        if merge_with:
            report.recommend_merge_with.append(merge_with)
    elif recommendation == "REFINE":
        report.recommend_refine = True

    evidence = holistic.get("evidence", [])
    if isinstance(evidence, list):
        report.warnings.extend(evidence)

    reasoning = holistic.get("reasoning", "")
    if reasoning:
        report.warnings.append(f"Holistic: {reasoning}")


def _action_tags(report: SkillQualityReport) -> List[str]:
    tags = []
    if report.recommend_discard:
        tags.append("DISCARD")
    if report.recommend_split:
        tags.append("SPLIT")
    if report.recommend_merge_with:
        tags.append("MERGE")
    if report.recommend_refine:
        tags.append("REFINE")
    return tags


def _save_report(summary: EvaluationSummary, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)
    logger.info("Evaluation report written to %s", filepath)
