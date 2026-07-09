"""
Step 9 — Stage 3 pipeline orchestrator and agent-facing summary.

Orchestrates the full contract verification loop:
  1. Build predicate summaries from Stage 2 segmentation results.
  2. Compute per-instance effects.
  3. Build initial contracts from aggregated instances.
  4. Verify contracts against instances.
  5. Decide update actions (KEEP / REFINE / SPLIT).
  6. Materialize NEW skills from ``__NEW__`` segments.
  7. Persist updates to the skill bank.
  8. Produce an agent-facing summary for downstream consumers
     (LLM teacher, preference learning, bank maintenance).

Integration points:
  - Consumes ``SegmentationResult`` from Stage 2.
  - Provides ``compat_fn`` back to Stage 2's ``SegmentScorer``.
  - Produces diagnostics for the LLM teacher / preference loop.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from skill_agents_grpo.contract_verification.config import ContractVerificationConfig
from skill_agents_grpo.contract_verification.schemas import (
    SegmentRecord,
    SkillContract,
    UpdateAction,
    VerificationReport,
)
from skill_agents_grpo.contract_verification.predicates import (
    build_records_from_result,
    build_segment_predicates,
    default_extract_predicates,
)
from skill_agents_grpo.contract_verification.contract_init import (
    compute_all_effects,
    build_initial_contracts,
)
from skill_agents_grpo.contract_verification.contract_verify import (
    verify_all_contracts,
    verify_contract,
)
from skill_agents_grpo.contract_verification.updates import (
    decide_action,
    apply_refine,
    apply_split,
    materialize_new_skills,
    NEW_SKILL,
)
from skill_agents_grpo.contract_verification.skill_bank import SkillBank

logger = logging.getLogger(__name__)


# ── Agent-facing summary ────────────────────────────────────────────

@dataclass
class Stage3Summary:
    """Compact summary of Stage 3 results for downstream agents.

    This is what the LLM teacher / preference learner reads to:
      - Propose names for new skills.
      - Propose contract patches.
      - Generate preference labels on confusing segments.
    """

    actions: List[Dict[str, Any]] = field(default_factory=list)
    new_skills_created: List[str] = field(default_factory=list)
    resegment_needed: bool = False
    bank_summary: Dict[str, dict] = field(default_factory=dict)
    skill_diagnostics: Dict[str, dict] = field(default_factory=dict)
    action_language_output: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "actions": self.actions,
            "new_skills_created": self.new_skills_created,
            "resegment_needed": self.resegment_needed,
            "bank_summary": self.bank_summary,
            "skill_diagnostics": self.skill_diagnostics,
        }
        if self.action_language_output is not None:
            d["action_language_output"] = self.action_language_output
        return d

    def format_for_llm(self, include_action_language: bool = True) -> str:
        """Format as a text block suitable for LLM context injection.

        Parameters
        ----------
        include_action_language : bool
            If True and ``action_language_output`` is set, append the
            action language representation of the skill bank.
        """
        lines = ["=== Stage 3: Contract Verification Summary ==="]
        for act in self.actions:
            skill = act.get("skill_id", "?")
            action = act.get("action", "?")
            lines.append(f"  [{action}] {skill}")
            if act.get("dropped_literals"):
                for category, lits in act["dropped_literals"].items():
                    if lits:
                        lines.append(f"    dropped {category}: {', '.join(lits)}")
            if act.get("new_skill_ids"):
                lines.append(f"    new skills: {', '.join(act['new_skill_ids'])}")

        if self.new_skills_created:
            lines.append(f"\n  New skills materialized: {', '.join(self.new_skills_created)}")
        if self.resegment_needed:
            lines.append("  ** Re-segmentation recommended (new skills added) **")

        lines.append("\n  Active bank:")
        for skill_id, info in self.bank_summary.items():
            pr = info.get("pass_rate", "n/a")
            n = info.get("n_instances", 0)
            lines.append(f"    {skill_id} v{info.get('version',0)} -- pass={pr:.2f}, n={n}")

        if include_action_language and self.action_language_output:
            lines.append("\n  === Action Language Representation ===")
            for al_line in self.action_language_output.splitlines():
                lines.append(f"  {al_line}")

        return "\n".join(lines)


# ── Main pipeline ───────────────────────────────────────────────────

def run_stage3(
    result,
    traj_id: str,
    observations: Sequence,
    bank: Optional[SkillBank] = None,
    config: Optional[ContractVerificationConfig] = None,
    extract_fn: Callable = default_extract_predicates,
    cached_predicates: Optional[List[Optional[Dict[str, float]]]] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> tuple[SkillBank, Stage3Summary]:
    """Run the full Stage 3 pipeline.

    Parameters
    ----------
    result : SegmentationResult
        Output from Stage 2 (has ``.segments`` list of ``SegmentDiagnostic``).
    traj_id : str
        Trajectory identifier.
    observations : Sequence
        Full trajectory observations (indexed by timestep).
    bank : SkillBank, optional
        Existing bank to update; created fresh if None.
    config : ContractVerificationConfig, optional
    extract_fn : callable
        ``extract_predicates(obs) -> {predicate: prob}``.
    cached_predicates : list, optional
        Pre-extracted per-timestep predicate dicts.
    embeddings : dict, optional
        Mapping seg_id -> embedding vector.

    Returns
    -------
    bank : SkillBank
        Updated skill bank.
    summary : Stage3Summary
        Agent-facing summary of all actions taken.
    """
    cfg = config or ContractVerificationConfig()
    if bank is None:
        bank = SkillBank(path=cfg.bank_path)

    # Step 1: build segment predicate summaries
    records = build_records_from_result(
        result, traj_id, observations, cfg.predicates,
        extract_fn=extract_fn,
        cached_predicates=cached_predicates,
        embeddings=embeddings,
    )
    logger.info("Stage 3: built %d segment records", len(records))

    # Step 2: compute per-instance effects
    compute_all_effects(records, cfg.predicates, cfg.aggregation)

    # Separate NEW vs existing-skill records
    new_records = [r for r in records if r.skill_label == NEW_SKILL]
    existing_records = [r for r in records if r.skill_label != NEW_SKILL]

    # Step 3: build initial contracts from existing-skill records
    contracts = build_initial_contracts(
        existing_records, cfg.predicates, cfg.aggregation,
        existing_contracts=bank.active_contracts,
    )
    logger.info("Stage 3: built/updated %d contracts", len(contracts))

    # Step 4: verify contracts
    reports = verify_all_contracts(contracts, existing_records, cfg)

    # Step 5: decide update actions
    by_skill: Dict[str, List[SegmentRecord]] = defaultdict(list)
    for rec in existing_records:
        by_skill[rec.skill_label].append(rec)

    summary = Stage3Summary()

    for skill_id, contract in contracts.items():
        report = reports.get(skill_id)
        if report is None:
            continue
        instances = by_skill.get(skill_id, [])
        action = decide_action(contract, report, instances, cfg)

        logger.info("Stage 3: %s -> %s (pass_rate=%.3f, n=%d)",
                     skill_id, action.action, report.overall_pass_rate, report.n_instances)

        if action.action == "KEEP":
            bank.add_or_update(contract, report)
        elif action.action == "REFINE":
            apply_refine(contract, action)
            refined_report = verify_contract(contract, instances, cfg)
            bank.add_or_update(contract, refined_report)
        elif action.action == "SPLIT":
            children, child_reports = apply_split(contract, action, instances, cfg)
            bank.apply_action(action, contract=contract, children=children, reports=child_reports)

        summary.actions.append(action.to_dict())
        summary.skill_diagnostics[skill_id] = {
            "action": action.action,
            "pass_rate": report.overall_pass_rate,
            "n_instances": report.n_instances,
            "top_violations": _top_violations(report),
            "counterexamples": report.counterexample_ids[:5],
        }

    # Step 7: materialize NEW skills
    if new_records:
        new_contracts, new_actions, new_reports, resegment = materialize_new_skills(
            new_records, bank.active_contracts, cfg,
        )
        for nc, na, nr in zip(new_contracts, new_actions, new_reports):
            bank.add_or_update(nc, nr)
            summary.actions.append(na.to_dict())
            summary.new_skills_created.append(nc.skill_id)
        summary.resegment_needed = resegment

    # Step 8: persist
    if cfg.bank_path:
        bank.save(cfg.bank_path)

    summary.bank_summary = bank.summary()

    # Generate action language output if configured
    if cfg.action_language_format:
        summary.action_language_output = bank.to_action_language(
            fmt=cfg.action_language_format,
        )

    return bank, summary


# ── Batch: process multiple trajectories ─────────────────────────────

def run_stage3_batch(
    results_and_obs: List[tuple],
    bank: Optional[SkillBank] = None,
    config: Optional[ContractVerificationConfig] = None,
    extract_fn: Callable = default_extract_predicates,
) -> tuple[SkillBank, List[Stage3Summary]]:
    """Run Stage 3 on multiple trajectories, accumulating into one bank.

    Parameters
    ----------
    results_and_obs : list of (result, traj_id, observations)
        Each element is a tuple of (SegmentationResult, traj_id, observations).
    bank : SkillBank, optional
    config : ContractVerificationConfig, optional
    extract_fn : callable

    Returns
    -------
    bank : SkillBank
    summaries : list[Stage3Summary]
    """
    cfg = config or ContractVerificationConfig()
    if bank is None:
        bank = SkillBank(path=cfg.bank_path)

    summaries: List[Stage3Summary] = []
    for result, traj_id, observations in results_and_obs:
        bank, summary = run_stage3(
            result, traj_id, observations,
            bank=bank, config=cfg, extract_fn=extract_fn,
        )
        summaries.append(summary)

    return bank, summaries


# ── Helpers ──────────────────────────────────────────────────────────

def _top_violations(report: VerificationReport, n: int = 5) -> List[str]:
    """Extract the top-N most violated literals from a verification report."""
    violations: List[tuple] = []
    for p, rate in report.pre_violation_rate.items():
        if rate > 0:
            violations.append((rate, f"pre:{p}"))
    for p, rate in report.eff_add_success_rate.items():
        if rate < 1.0:
            violations.append((1.0 - rate, f"eff_add:{p}"))
    for p, rate in report.eff_del_success_rate.items():
        if rate < 1.0:
            violations.append((1.0 - rate, f"eff_del:{p}"))
    for p, rate in report.inv_hold_rate.items():
        if rate < 1.0:
            violations.append((1.0 - rate, f"inv:{p}"))
    violations.sort(key=lambda x: -x[0])
    return [v[1] for v in violations[:n]]
