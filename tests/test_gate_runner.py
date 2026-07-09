"""Day-7: tests for `harness.gate_runner.GateRunner` — the spec-named
offline gate surface. Closes harness/README §11 + §12.

Pins:

  * `GateRunner` is a `GateService` subclass — old callers work.
  * `GateRunnerConfig` reproducibility anchors flow into the
    `SkillEvaluationRecord` (`bank_snapshot_id`, `eval_suite_id`,
    `adapter_versions`, `ontology_version`, `seed`, `version`).
  * `EvalSuite.delta()` computes ``post - pre`` and replaces the
    scalar `(baseline_score, post_score)` parameters end-to-end.
  * `rollout_batch=…` Stage-2 input replaces the `RewardLogger`
    surface; episodes for the wrong skill are dropped automatically.
  * Mixing the old and new shape for the same stage is an error.
"""
from __future__ import annotations

from typing import Any, Iterable, List

import pytest

from common.enums import (
    GateStage,
    GateVerdict,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    HypothesisProposal,
)
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_record import (
    SkillContract,
    SkillRecord,
)
from harness.adapter_registry import AdapterRegistry
from harness.gate_runner import EvalSuite, GateRunner, GateRunnerConfig
from harness.replay_validator import ReplayValidator
from harness.reward_logger import RewardLogger
from harness.skill_harness import HarnessConfig, SkillHarness


def _fresh_skill(
    *,
    skill_id: str = "s",
    feasible_domains: List[str] = None,
    source_domains: List[str] = None,
) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id,
        name=skill_id,
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        status=SkillStatus.PROVISIONAL,
        feasible_domains=feasible_domains or ["gymv", "browser"],
        source_domains=source_domains or ["gymv"],
        protocol=[{"action": "STEP", "payload": {}}],
        contract=SkillContract(),
    )


def _hypothesis_for(skill: SkillRecord) -> BankMutationProposal:
    return HypothesisProposal(
        proposal_id="prop-test",
        rationale="test",
        name=skill.name,
        novel_protocol=list(skill.protocol),
        contract=skill.contract,
    )


def _harness() -> SkillHarness:
    registry = AdapterRegistry()
    return SkillHarness(registry, config=HarnessConfig())


def test_gate_runner_pins_reproducibility_anchors() -> None:
    skill = _fresh_skill()
    proposal = _hypothesis_for(skill)
    runner = GateRunner(
        config=GateRunnerConfig(
            bank_snapshot_id="snap-2026-05-01",
            eval_suite_id="gymv-smoke-v1",
            adapter_versions={"gymv": "v3"},
            ontology_version="ont-1.2",
            seed=7,
            judge_model="gpt-test",
        ),
        harness=_harness(),
    )
    rec = runner.evaluate(
        proposal=proposal,
        skill=skill,
        status_before=SkillStatus.PROVISIONAL,
    )
    assert rec.bank_snapshot_id == "snap-2026-05-01"
    assert rec.eval_suite_id == "gymv-smoke-v1"
    assert rec.adapter_versions == {"gymv": "v3"}
    assert rec.ontology_version == "ont-1.2"
    assert rec.seed == 7
    assert rec.judge_model == "gpt-test"
    assert rec.version == skill.version
    assert rec.status_before == SkillStatus.PROVISIONAL


def test_gate_runner_eval_suite_replaces_scalar_scores() -> None:
    skill = _fresh_skill()
    proposal = _hypothesis_for(skill)
    runner = GateRunner(harness=_harness())
    rec = runner.evaluate(
        proposal=proposal,
        skill=skill,
        eval_suite=EvalSuite(
            suite_id="gymv-suite-v0",
            pre_score=0.40,
            post_score=0.65,
            metrics={"twenty_forty_eight": 0.7, "tetris": 0.6},
        ),
    )
    # Stage 4 ran with the EvalSuite scores.
    sv = rec.verdict.stage_for(GateStage.NON_REGRESSION)
    assert sv is not None
    assert sv.metrics["pre"] == pytest.approx(0.40)
    assert sv.metrics["post"] == pytest.approx(0.65)
    # Per-suite metrics flowed into record.metrics.
    assert rec.metrics["eval_suite.delta"] == pytest.approx(0.25)
    assert rec.metrics["eval_suite.twenty_forty_eight"] == pytest.approx(0.7)
    assert rec.eval_suite_id == "gymv-suite-v0"


def test_gate_runner_rollout_batch_filters_to_skill() -> None:
    skill = _fresh_skill()
    proposal = _hypothesis_for(skill)

    # Two episodes — only one matches the skill_id.
    ep_match = SkillEpisode(
        episode_id="e1",
        skill_id=skill.skill_id,
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
        outcome=SkillEpisodeOutcome(success=True, contract_satisfied=True),
    )
    ep_other = SkillEpisode(
        episode_id="e2",
        skill_id="other",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
        outcome=SkillEpisodeOutcome(success=False, contract_satisfied=False),
    )

    runner = GateRunner(harness=_harness())
    rec = runner.evaluate(
        proposal=proposal,
        skill=skill,
        rollout_batch=[ep_match, ep_other],
    )
    sv = rec.verdict.stage_for(GateStage.SHADOW)
    # Stage 2 saw only the match (1/1 success rate).
    assert sv is not None
    # `n_shadow_episodes` is 1.0 even though the batch had 2 episodes.
    assert sv.metrics.get("n_shadow_episodes") == 1.0
    assert sv.metrics.get("shadow_pass_rate") == pytest.approx(1.0)


def test_gate_runner_rejects_mixed_stage2_inputs() -> None:
    skill = _fresh_skill()
    proposal = _hypothesis_for(skill)
    runner = GateRunner(harness=_harness())
    log = RewardLogger()
    with pytest.raises(ValueError, match="rollout_batch.*shadow_log"):
        runner.evaluate(
            proposal=proposal,
            skill=skill,
            rollout_batch=[],
            shadow_log=log,
        )


def test_gate_runner_rejects_mixed_stage4_inputs() -> None:
    skill = _fresh_skill()
    proposal = _hypothesis_for(skill)
    runner = GateRunner(harness=_harness())
    with pytest.raises(ValueError, match="eval_suite.*baseline_score"):
        runner.evaluate(
            proposal=proposal,
            skill=skill,
            eval_suite=EvalSuite(suite_id="x", pre_score=0.5, post_score=0.6),
            baseline_score=0.4,
            post_score=0.5,
        )
