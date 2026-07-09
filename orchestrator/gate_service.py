"""`GateService` — composes the seven canonical gate stages.

Spec: PLAN-UNIFIED-SKILL-GATE §7.

  Stage 0  — Static          (this module, syntax / contract sanity)
  Stage 1  — Replay          (delegates to harness.ReplayValidator)
  Stage 2  — Shadow          (read-only check — orchestrator runs SHADOW
                              skills in production and we read their stats
                              from the reward log)
  Stage 3a — Transfer        (few-shot adaptation against TRANSFER_TARGET_DOMAINS
                              via harness.FewShotAdapter; PLAN-SKILL-BANK §0.4)
  Stage 4  — Non-regression  (compare new bank against last release)
  Stage 5  — Promotion       (handed to PromotionOrchestrator on PASS)
  Stage 6  — Rollback/depr.  (handed to PromotionOrchestrator on regression)

This module owns Stages 0–4. Stages 5 and 6 are *promotion actions*, not
verdicts, and live in `promotion_orchestrator`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from common.enums import (
    DOMAINS,
    SOURCE_DOMAINS,
    TRANSFER_TARGET_DOMAINS,
    GateStage,
    GateVerdict,
)
from common.ids import new_proposal_id, schema_hash
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
    RewriteProposal,
)
from data_structure.extensions.gate_verdict import GateVerdictPayload, StageVerdict
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillRecord
from harness import SkillHarness
from harness.few_shot_adapter import AdaptResult, FewShotAdapter, FewShotDemo
from harness.reward_logger import RewardLogger
from orchestrator.config import GateThresholds


class _EvalSuiteLike(Protocol):
    """Structural shape of :class:`harness.gate_runner.EvalSuite`.

    Declared here as a Protocol so :class:`GateService` can accept the
    runtime payload without importing the concrete dataclass at module
    load time. Any object with ``suite_id`` / ``pre_score`` /
    ``post_score`` (and optionally ``metrics``) satisfies it.
    """

    suite_id: str
    pre_score: float
    post_score: float


@dataclass
class NonRegressionResult:
    delta: float
    pre: float
    post: float
    passed: bool

    def as_stage_verdict(self, *, max_delta: float) -> StageVerdict:
        verdict = GateVerdict.PASS if self.passed else GateVerdict.FAIL
        return StageVerdict(
            stage=GateStage.NON_REGRESSION,
            verdict=verdict,
            metrics={"delta": self.delta, "pre": self.pre, "post": self.post, "max_delta": max_delta},
        )


class GateService:
    """Runs gate stages against a candidate proposal+skill."""

    def __init__(
        self,
        *,
        harness: SkillHarness,
        thresholds: Optional[GateThresholds] = None,
        few_shot_adapter: Optional[FewShotAdapter] = None,
    ) -> None:
        self._harness = harness
        self._thresholds = thresholds or GateThresholds()
        fs_cfg = self._thresholds.few_shot
        self._few_shot = few_shot_adapter or FewShotAdapter(
            harness=harness,
            k_shot_default=fs_cfg.k_shot_default,
            k_shot_max=fs_cfg.k_shot_max,
            target_domain_pass_rate_min=fs_cfg.target_domain_pass_rate_min,
            adaptation_cost_max_tokens=fs_cfg.adaptation_cost_max_tokens,
        )

    # -- public API --------------------------------------------------------

    def evaluate(
        self,
        *,
        proposal: BankMutationProposal,
        skill: SkillRecord,
        replay_seeds: Iterable[SkillEpisode] = (),
        shadow_log: Optional[RewardLogger] = None,
        baseline_score: Optional[float] = None,
        post_score: Optional[float] = None,
        eval_suite: Optional["_EvalSuiteLike"] = None,
        few_shot_demos: Optional[Mapping[str, Sequence[FewShotDemo]]] = None,
    ) -> SkillEvaluationRecord:
        # Stage-4 input normalisation: caller may pass either a frozen
        # ``EvalSuite`` (preferred — produced by orchestrator/eval_suite.py)
        # or the legacy (baseline, post) scalar pair, never both.
        if eval_suite is not None and (
            baseline_score is not None or post_score is not None
        ):
            raise ValueError(
                "GateService.evaluate: pass either eval_suite=… OR "
                "(baseline_score, post_score)=…, not both."
            )
        eff_baseline = baseline_score
        eff_post = post_score
        if eval_suite is not None:
            eff_baseline = float(eval_suite.pre_score)
            eff_post = float(eval_suite.post_score)

        evaluation_id = f"eval-{new_proposal_id().split('-', 1)[1]}"
        verdicts: List[StageVerdict] = []

        verdicts.append(self._run_static(skill, proposal))
        verdicts.append(self._run_replay(skill, list(replay_seeds)))
        verdicts.append(self._run_shadow(skill, shadow_log))
        transfer_verdict, verified_targets = self._run_transfer(
            skill, few_shot_demos=few_shot_demos
        )
        verdicts.append(transfer_verdict)
        verdicts.append(self._run_non_regression(eff_baseline, eff_post))

        rationale, final_verdict, eligible = self._aggregate(
            verdicts, skill, verified_targets=verified_targets
        )

        verdict_payload = GateVerdictPayload(
            proposal_id=proposal.proposal_id,
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            stages=verdicts,
            final_verdict=final_verdict,
            rationale=rationale,
            eligible_domains=eligible,
        )
        ev = SkillEvaluationRecord(
            evaluation_id=evaluation_id,
            proposal_id=proposal.proposal_id,
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            episode_ids=[ep.episode_id for ep in replay_seeds],
            verdict=verdict_payload,
            metrics={
                f"{v.stage.value}.{k}": float(val)
                for v in verdicts
                for k, val in v.metrics.items()
            },
            started_at=time.time(),
            finished_at=time.time(),
        )
        return ev

    # -- stage 0 -----------------------------------------------------------

    def _run_static(
        self, skill: SkillRecord, proposal: BankMutationProposal
    ) -> StageVerdict:
        failures: List[str] = []
        # T1.3d: dropped legacy "feasible_domains ≥ 2 (general-protocol
        # invariant)" Stage-0 echo. The lane-(a) replacement
        # (``min_retrievals_per_skill``) is enforced by the lifecycle
        # manager at the ACTIVE transition only, since Stage 0 fires
        # for every proposal (incl. DRAFT→CANDIDATE, where retrievals
        # are necessarily 0). See
        # ``SkillLifecycleManager._validate_invariants`` and
        # ``OrchestratorConfig.gate_thresholds.min_retrievals_per_skill``.
        for d in skill.feasible_domains:
            if d not in DOMAINS:
                failures.append(f"unknown_domain={d!r}")
        # Evidence-driven invariant
        if not skill.contract.expected_evidence_roles and skill.skill_type.value != "action":
            failures.append("contract.expected_evidence_roles empty (G0)")
        # Protocol non-empty (except retirements)
        if not isinstance(proposal, RetireProposal) and not skill.protocol:
            failures.append("skill.protocol is empty")
        # Source-type sanity. RewriteProposal is exempt: by construction
        # it does not change the underlying skill's source_type (T1.3b).
        if not isinstance(proposal, RewriteProposal) and proposal.source_type != skill.source_type:
            failures.append(
                f"source_type mismatch: proposal={proposal.source_type.value}, "
                f"skill={skill.source_type.value}"
            )
        # Lineage check (composition / repair / rewrite must reference parents)
        if isinstance(proposal, (ComposeProposal,)) and not proposal.component_skill_ids:
            failures.append("ComposeProposal.component_skill_ids is empty")
        if isinstance(proposal, (GeneralizeProposal, PatchProposal, RewriteProposal)) and not getattr(proposal, "base_skill_id", ""):
            failures.append("base_skill_id is empty")
        verdict = GateVerdict.PASS if not failures else GateVerdict.FAIL
        return StageVerdict(stage=GateStage.STATIC, verdict=verdict, failures=failures)

    # -- stage 1 -----------------------------------------------------------

    def _run_replay(
        self, skill: SkillRecord, seeds: List[SkillEpisode]
    ) -> StageVerdict:
        if not seeds:
            return StageVerdict(
                stage=GateStage.REPLAY,
                verdict=GateVerdict.LIMITED_PASS,
                metrics={"n_seeds": 0.0},
                notes="no replay seeds provided",
            )
        result = self._harness.replay_validate(skill, seeds=seeds)
        return result.as_stage_verdict(threshold=self._thresholds.replay_pass_rate)

    # -- stage 2 -----------------------------------------------------------

    def _run_shadow(
        self, skill: SkillRecord, log: Optional[RewardLogger]
    ) -> StageVerdict:
        if log is None:
            return StageVerdict(
                stage=GateStage.SHADOW,
                verdict=GateVerdict.LIMITED_PASS,
                notes="no shadow log",
            )
        entries = list(log.filter(skill_id=skill.skill_id))
        if not entries:
            return StageVerdict(
                stage=GateStage.SHADOW,
                verdict=GateVerdict.LIMITED_PASS,
                notes="no shadow data yet",
            )
        rate = sum(1 for e in entries if e.success) / len(entries)
        verdict = GateVerdict.PASS if rate >= self._thresholds.shadow_pass_rate else GateVerdict.FAIL
        return StageVerdict(
            stage=GateStage.SHADOW,
            verdict=verdict,
            metrics={"shadow_pass_rate": rate, "n_shadow_episodes": float(len(entries))},
        )

    # -- stage 3a (few-shot transfer) -------------------------------------

    def _run_transfer(
        self,
        skill: SkillRecord,
        *,
        few_shot_demos: Optional[Mapping[str, Sequence[FewShotDemo]]] = None,
    ) -> tuple[StageVerdict, List[str]]:
        """Stage 3a — few-shot adaptation against TRANSFER_TARGET_DOMAINS.

        Returns the StageVerdict plus the list of target domains that
        actually verified (so the caller can decide whether to promote
        and what to write into `SkillRecord.verified_domains`).

        Asymmetric path (PLAN-SKILL-BANK §0.4): if the skill carries
        any source/target metadata, we run the few-shot adapter against
        each declared transfer target. Otherwise we fall back to the
        legacy "≥ N feasible domains" check so historical proposals
        keep evaluating.
        """

        thresholds = self._thresholds
        targets = self._infer_targets(skill)

        if not skill.source_domains and not targets:
            # Legacy fallback — preserves PLAN-UNIFIED-SKILL-GATE pre-asymmetry.
            n = len(set(skill.feasible_domains))
            verdict = (
                GateVerdict.PASS
                if n >= thresholds.transfer_min_domains
                else GateVerdict.LIMITED_PASS
            )
            return (
                StageVerdict(
                    stage=GateStage.TRANSFER,
                    verdict=verdict,
                    metrics={
                        "n_domains": float(n),
                        "min_domains": float(thresholds.transfer_min_domains),
                    },
                    notes="legacy_path:no source/target metadata",
                ),
                [],
            )

        if not targets:
            return (
                StageVerdict(
                    stage=GateStage.TRANSFER,
                    verdict=GateVerdict.FAIL,
                    failures=["no transfer_target_domains declared"],
                    notes="few_shot_skipped:no_targets",
                ),
                [],
            )

        if not any(d in SOURCE_DOMAINS for d in skill.source_domains):
            return (
                StageVerdict(
                    stage=GateStage.TRANSFER,
                    verdict=GateVerdict.FAIL,
                    failures=[
                        f"source_domains={sorted(set(skill.source_domains))} "
                        f"has no game-foundry lineage (SOURCE_DOMAINS={SOURCE_DOMAINS})"
                    ],
                ),
                [],
            )

        adapt_results: List[AdaptResult] = []
        for tgt in targets:
            shots = (few_shot_demos or {}).get(tgt, ())
            adapt_results.append(
                self._few_shot.adapt(skill=skill, target_domain=tgt, demos=shots)
            )

        verified = [
            r.target_domain
            for r in adapt_results
            if r.n_total > 0 and r.pass_rate >= thresholds.few_shot.target_domain_pass_rate_min
        ]
        diagnostics = sorted(
            {r.diagnostic_label for r in adapt_results if r.diagnostic_label}
        )

        metrics: Dict[str, float] = {
            "n_targets": float(len(targets)),
            "n_verified_targets": float(len(verified)),
            "min_verified_targets": float(thresholds.transfer_min_target_domains_verified),
        }
        for r in adapt_results:
            metrics[f"pass_rate.{r.target_domain}"] = float(r.pass_rate)
            metrics[f"k_used.{r.target_domain}"] = float(r.k_used)

        if len(verified) >= thresholds.transfer_min_target_domains_verified:
            verdict = GateVerdict.PASS
        elif len(verified) >= 1:
            verdict = GateVerdict.LIMITED_PASS
        else:
            verdict = GateVerdict.FAIL

        return (
            StageVerdict(
                stage=GateStage.TRANSFER,
                verdict=verdict,
                metrics=metrics,
                failures=(
                    [f"transfer_diagnostic:{d}" for d in diagnostics]
                    if verdict == GateVerdict.FAIL
                    else []
                ),
                notes=(
                    f"verified_targets={verified}; "
                    f"diagnostics={diagnostics}"
                ),
            ),
            verified,
        )

    def _infer_targets(self, skill: SkillRecord) -> List[str]:
        declared = list(skill.transfer_target_domains)
        if declared:
            return [d for d in declared if d in TRANSFER_TARGET_DOMAINS]
        # Fallback: any feasible_domain that is a transfer target and
        # not yet verified.
        return [
            d
            for d in skill.feasible_domains
            if d in TRANSFER_TARGET_DOMAINS and d not in skill.verified_domains
        ]

    # -- stage 4 -----------------------------------------------------------

    def _run_non_regression(
        self, baseline: Optional[float], post: Optional[float]
    ) -> StageVerdict:
        if baseline is None or post is None:
            return StageVerdict(
                stage=GateStage.NON_REGRESSION,
                verdict=GateVerdict.LIMITED_PASS,
                notes="no baseline / post score provided",
            )
        delta = post - baseline
        passed = delta >= -self._thresholds.non_regression_max_delta
        return NonRegressionResult(
            delta=delta, pre=baseline, post=post, passed=passed
        ).as_stage_verdict(max_delta=self._thresholds.non_regression_max_delta)

    # -- aggregate ---------------------------------------------------------

    def _aggregate(
        self,
        verdicts: List[StageVerdict],
        skill: SkillRecord,
        *,
        verified_targets: Optional[List[str]] = None,
    ) -> tuple[str, GateVerdict, List[str]]:
        any_fail = any(v.verdict == GateVerdict.FAIL for v in verdicts)
        any_limited = any(v.verdict == GateVerdict.LIMITED_PASS for v in verdicts)
        # Eligible-domain set: source_domains (game lineage) plus the
        # subset of target domains that just verified through Stage 3a.
        eligible: List[str] = []
        if skill.source_domains or verified_targets:
            eligible.extend(skill.source_domains)
            eligible.extend(verified_targets or [])
            eligible = sorted({d for d in eligible if d in DOMAINS})
        else:
            eligible = list(skill.feasible_domains)
        if any_fail:
            failing = [v.stage.value for v in verdicts if v.verdict == GateVerdict.FAIL]
            return f"failed_stages={failing}", GateVerdict.FAIL, []
        if any_limited:
            return "promotion_to_provisional_only", GateVerdict.LIMITED_PASS, eligible
        return "all_stages_pass", GateVerdict.PASS, eligible


__all__ = ["GateService", "NonRegressionResult"]
