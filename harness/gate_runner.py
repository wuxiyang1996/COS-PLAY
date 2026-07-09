"""`GateRunner` — the offline gate surface (PLAN-UNIFIED-SKILL-GATE §6).

The spec calls it the "Harness `GateRunner`"; the live composition has
historically lived in [`orchestrator.gate_service.GateService`](
../orchestrator/gate_service.py). Day-7 of the harness/README §22
roadmap closes the naming gap and lays the structural foundation for
Day-8's reproducibility-anchor work without touching the
already-tested `GateService` body — the alias keeps every existing
caller working.

What's new in this module on top of `GateService`:

* **`GateRunner`** — a thin subclass of `GateService` that accepts a
  reproducibility-anchor block (`bank_snapshot_id`, `eval_suite_id`,
  `adapter_versions`, `ontology_version`, `seed`, `judge_model`,
  `status_before`) at construction and threads it into the resulting
  `SkillEvaluationRecord` so two evaluations against different snapshots
  are now distinguishable on disk (closes harness/README §11).

* **Additive `evaluate(...)` shape**:
  * **`rollout_batch`** — replacement for the bare `RewardLogger` shadow
    input. Accepts a `Sequence[SkillEpisode]` directly so callers don't
    have to round-trip through a logger when their data already lives
    as episodes (e.g. an offline I/O dump driver). The `RewardLogger`
    path remains supported.
  * **`eval_suite`** — replacement for the scalar `(baseline_score,
    post_score)` Stage-4 input. Accepts an `EvalSuite` value object that
    pins a `suite_id` plus before/after metrics so the persisted record
    can name *which* evaluation suite produced the verdict.

* **`GateRunnerConfig`** — small immutable container for the
  reproducibility anchors. Pin once at GateRunner construction; every
  emitted `SkillEvaluationRecord` inherits them.

This is an additive surface — passing none of the new arguments
reproduces `GateService.evaluate(...)` exactly, modulo the extra
fields populated on the returned `SkillEvaluationRecord`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from common.enums import GateStage, GateVerdict, SkillStatus
from data_structure.extensions.bank_mutation_proposal import BankMutationProposal
from data_structure.extensions.gate_verdict import StageVerdict
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillRecord
from harness.few_shot_adapter import FewShotDemo
from harness.reward_logger import RewardLogger
from orchestrator.eval_suite import EvalSuite  # canonical home (T2.2); re-exported here
from orchestrator.gate_service import GateService

__all__ = ["EvalSuite", "GateRunner", "GateRunnerConfig"]


@dataclass(frozen=True)
class GateRunnerConfig:
    """Reproducibility anchors pinned at GateRunner construction.

    Closes harness/README §11. Each evaluation emitted by this runner
    inherits these — the `SkillEvaluationRecord.extra` payload will
    surface them so two evaluations against different bank snapshots,
    eval suites, or adapter versions are distinguishable on disk.
    """

    bank_snapshot_id: Optional[str] = None
    eval_suite_id: Optional[str] = None
    adapter_versions: Mapping[str, str] = field(default_factory=dict)
    ontology_version: Optional[str] = None
    judge_model: Optional[str] = None
    seed: Optional[int] = None


class GateRunner(GateService):
    """Subclass of `GateService` that pins reproducibility anchors and
    accepts the spec-shaped `rollout_batch` / `eval_suite` Stage I/O.

    Old callers using `GateService(...)` work unchanged. New callers
    should prefer `GateRunner(...)` to get the full spec contract.
    """

    def __init__(
        self,
        *,
        config: Optional[GateRunnerConfig] = None,
        **gate_service_kwargs: Any,
    ) -> None:
        super().__init__(**gate_service_kwargs)
        self._gr_config = config or GateRunnerConfig()

    @property
    def runner_config(self) -> GateRunnerConfig:
        return self._gr_config

    def evaluate(
        self,
        *,
        proposal: BankMutationProposal,
        skill: SkillRecord,
        replay_seeds: Iterable[SkillEpisode] = (),
        shadow_log: Optional[RewardLogger] = None,
        rollout_batch: Optional[Sequence[SkillEpisode]] = None,
        baseline_score: Optional[float] = None,
        post_score: Optional[float] = None,
        eval_suite: Optional[EvalSuite] = None,
        few_shot_demos: Optional[Mapping[str, Sequence[FewShotDemo]]] = None,
        status_before: Optional[SkillStatus] = None,
    ) -> SkillEvaluationRecord:
        """Run all five stages and emit a `SkillEvaluationRecord` with
        the runner's reproducibility anchors attached.

        The new `rollout_batch` and `eval_suite` parameters are
        additive: when provided, they replace `shadow_log` /
        `(baseline_score, post_score)` respectively for the underlying
        stage. Passing both old and new shapes for the same stage is
        an error — the spec contract picks one input source per stage.
        """

        # --- normalise Stage-2 input: rollout_batch wins over shadow_log
        if rollout_batch is not None and shadow_log is not None:
            raise ValueError(
                "GateRunner.evaluate: pass either rollout_batch=… OR "
                "shadow_log=…, not both (Stage 2 has one input source)."
            )
        synthesised_log: Optional[RewardLogger] = shadow_log
        if rollout_batch is not None:
            synthesised_log = _rollout_batch_to_log(rollout_batch, skill_id=skill.skill_id)

        # --- normalise Stage-4 input: eval_suite wins over scalars
        if eval_suite is not None and (baseline_score is not None or post_score is not None):
            raise ValueError(
                "GateRunner.evaluate: pass either eval_suite=… OR "
                "(baseline_score, post_score)=…, not both."
            )
        eff_baseline = baseline_score
        eff_post = post_score
        if eval_suite is not None:
            eff_baseline = eval_suite.pre_score
            eff_post = eval_suite.post_score

        # --- delegate to the existing GateService body
        record = super().evaluate(
            proposal=proposal,
            skill=skill,
            replay_seeds=replay_seeds,
            shadow_log=synthesised_log,
            baseline_score=eff_baseline,
            post_score=eff_post,
            few_shot_demos=few_shot_demos,
        )

        # --- pin reproducibility anchors directly on the record
        # (Day-8 expanded SkillEvaluationRecord with these fields).
        cfg = self._gr_config
        if cfg.bank_snapshot_id is not None:
            record.bank_snapshot_id = cfg.bank_snapshot_id
        if cfg.eval_suite_id is not None or eval_suite is not None:
            record.eval_suite_id = (
                eval_suite.suite_id if eval_suite is not None
                else cfg.eval_suite_id
            )
        if cfg.adapter_versions:
            record.adapter_versions = dict(cfg.adapter_versions)
        if cfg.ontology_version is not None:
            record.ontology_version = cfg.ontology_version
        if cfg.seed is not None and record.seed is None:
            record.seed = cfg.seed
        if cfg.judge_model is not None and record.judge_model is None:
            record.judge_model = cfg.judge_model
        record.version = skill.version
        if status_before is not None:
            record.status_before = status_before
        # `rejected_domains` = transfer_target_domains \ verified
        if skill.transfer_target_domains and record.verdict is not None:
            verified = set(record.verdict.eligible_domains)
            record.rejected_domains = sorted(
                set(skill.transfer_target_domains) - verified
            )
        # Per-suite metrics flow into the existing metrics dict so any
        # downstream consumer that reads `record.metrics` sees them
        # without traversing a new payload.
        if eval_suite is not None:
            for k, v in eval_suite.metrics.items():
                record.metrics[f"eval_suite.{k}"] = float(v)
            record.metrics["eval_suite.delta"] = float(eval_suite.delta())

        return record


def _rollout_batch_to_log(
    rollout_batch: Sequence[SkillEpisode],
    *,
    skill_id: str,
) -> RewardLogger:
    """Convert a list of `SkillEpisode`s to an in-memory
    `RewardLogger` so the downstream Stage-2 reader (which expects a
    logger) sees the same shape regardless of source.

    Filters episodes to the ones whose `skill_id` matches the
    proposal's skill — Stage 2 reads `log.filter(skill_id=...)`, so
    this short-circuits the filter for callers that already have the
    relevant subset.
    """
    log = RewardLogger()
    for ep in rollout_batch:
        if ep.skill_id != skill_id:
            # Defensive: callers may pass an unfiltered batch. We
            # cooperate with the Stage-2 filter by only emitting the
            # matching subset.
            continue
        log.log_episode(ep)
    return log


__all__ = ["EvalSuite", "GateRunner", "GateRunnerConfig"]
