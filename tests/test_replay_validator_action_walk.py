"""Day-7b: tests for `ReplayValidator(mode="action_level")`.

The action-level walk compares ``seed.steps[i]`` against the proposal's
adapter output step-by-step and pins:

  * action_type sequence equality (extra proposed steps tolerated;
    truncation is a regression),
  * evidence-role non-worsening (proposal's roles ⊇ seed's roles),
  * payload equality (per-step boolean diagnostic).

PLAN-UNIFIED-SKILL-GATE §7.1.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.replay_validator import (
    REPLAY_MODE_ACTION,
    ReplayValidator,
    StepDiff,
)
from harness.skill_adapter import (
    AdapterRunContext,
    AdapterRunResult,
    SkillAdapter,
)


class _StubAdapter(SkillAdapter):
    """Minimal adapter that returns a scripted step list per call."""

    name = "gymv"
    supported_types = (SkillType.ACTION,)

    def __init__(
        self,
        scripted_steps: List[Dict[str, Any]],
        *,
        new_evidence: Optional[List[EvidenceRef]] = None,
        success: bool = True,
    ) -> None:
        self._steps = scripted_steps
        self._new_evidence = new_evidence or []
        self._success = success

    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool:
        return True

    def run(
        self, skill: SkillRecord, ctx: AdapterRunContext
    ) -> AdapterRunResult:
        return AdapterRunResult(
            steps=list(self._steps),
            success=self._success,
            contract_satisfied=self._success,
            new_evidence=list(self._new_evidence),
            score=1.0 if self._success else 0.0,
        )


def _ev(role: str, source: str = "schema") -> EvidenceRef:
    return EvidenceRef(
        source=source, locator=f"loc-{role}", role=role,
    )


def _seed(*, steps: List[SkillEpisodeStep]) -> SkillEpisode:
    return SkillEpisode(
        episode_id=f"seed-{len(steps)}",
        skill_id="skill-X",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
        steps=steps,
        outcome=SkillEpisodeOutcome(success=True, contract_satisfied=True),
    )


def _skill() -> SkillRecord:
    return SkillRecord(
        skill_id="skill-X",
        name="X",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        status=SkillStatus.PROVISIONAL,
        feasible_domains=["gymv"],
        protocol=[{"action": "STEP", "payload": {}}],
        contract=SkillContract(),
    )


def _registry(adapter: _StubAdapter) -> AdapterRegistry:
    reg = AdapterRegistry()
    reg.register(adapter)
    return reg


def test_action_level_pass_on_identical_replay() -> None:
    seed_step = SkillEpisodeStep(
        step_index=0, action_type="SLIDE",
        action_payload={"direction": "left"},
        pre_state=None, post_state=None,
        evidence=[_ev("GATHER")],
    )
    proposed_step = {
        "action_type": "SLIDE",
        "payload": {"direction": "left"},
        "evidence": [_ev("GATHER")],
    }
    adapter = _StubAdapter([proposed_step], new_evidence=[_ev("GATHER")])
    rv = ReplayValidator(_registry(adapter))
    out = rv.validate(skill=_skill(), seeds=[_seed(steps=[seed_step])],
                      mode=REPLAY_MODE_ACTION)

    assert out.n_seeds == 1
    assert out.n_pass == 1
    assert out.mode == REPLAY_MODE_ACTION
    assert out.n_steps_compared == 1
    assert out.n_steps_action_match == 1
    assert out.n_steps_evidence_non_worse == 1
    diffs = out.per_seed_outcomes[0]["step_diffs"]
    assert len(diffs) == 1
    assert diffs[0]["action_match"] is True
    assert diffs[0]["evidence_non_worse"] is True


def test_action_level_fail_on_step_count_regression() -> None:
    seed_steps = [
        SkillEpisodeStep(step_index=0, action_type="A", action_payload={},
                         pre_state=None, post_state=None, evidence=[_ev("GATHER")]),
        SkillEpisodeStep(step_index=1, action_type="B", action_payload={},
                         pre_state=None, post_state=None, evidence=[_ev("GATHER")]),
    ]
    proposed = [{"action_type": "A", "payload": {}, "evidence": [_ev("GATHER")]}]
    adapter = _StubAdapter(proposed, new_evidence=[_ev("GATHER")])
    rv = ReplayValidator(_registry(adapter))
    out = rv.validate(skill=_skill(), seeds=[_seed(steps=seed_steps)],
                      mode=REPLAY_MODE_ACTION)

    assert out.n_pass == 0
    assert any(
        "step_count_regressed" in f
        for f in out.per_seed_outcomes[0]["step_failures"]
    )


def test_action_level_fail_on_evidence_regression() -> None:
    """Seed's step gathered evidence; proposal omits the gather role
    entirely — that's monotonic-worsening and must fail."""
    seed_step = SkillEpisodeStep(
        step_index=0, action_type="A", action_payload={},
        pre_state=None, post_state=None,
        evidence=[_ev("GATHER"), _ev("VERIFY")],
    )
    proposed = [{
        "action_type": "A", "payload": {},
        "evidence": [_ev("VERIFY")],
    }]
    adapter = _StubAdapter(proposed, new_evidence=[_ev("VERIFY")])
    rv = ReplayValidator(_registry(adapter))
    out = rv.validate(skill=_skill(), seeds=[_seed(steps=[seed_step])],
                      mode=REPLAY_MODE_ACTION)

    assert out.n_pass == 0
    assert out.n_steps_action_match == 1     # action_type still matches
    assert out.n_steps_evidence_non_worse == 0   # roles regressed
    assert any(
        "evidence_regressed" in f
        for f in out.per_seed_outcomes[0]["step_failures"]
    )


def test_action_level_extra_proposed_steps_are_tolerated() -> None:
    """The proposal may emit *additional* steps after the seed's last
    step — only the seed-prefix is compared, extras are fine."""
    seed_step = SkillEpisodeStep(
        step_index=0, action_type="A", action_payload={},
        pre_state=None, post_state=None, evidence=[_ev("GATHER")],
    )
    proposed = [
        {"action_type": "A", "payload": {}, "evidence": [_ev("GATHER")]},
        {"action_type": "INSPECT", "payload": {}, "evidence": [_ev("VERIFY")]},
    ]
    adapter = _StubAdapter(proposed, new_evidence=[_ev("GATHER")])
    rv = ReplayValidator(_registry(adapter))
    out = rv.validate(skill=_skill(), seeds=[_seed(steps=[seed_step])],
                      mode=REPLAY_MODE_ACTION)

    assert out.n_pass == 1
    assert out.n_steps_compared == 1


def test_adapter_level_mode_is_default_and_unchanged() -> None:
    """Default mode (`adapter_level`) preserves the Day-3 behaviour:
    one outcome per seed, no per-step diff."""
    seed_step = SkillEpisodeStep(
        step_index=0, action_type="A", action_payload={},
        pre_state=None, post_state=None, evidence=[_ev("GATHER")],
    )
    proposed = [{"action_type": "B", "payload": {}, "evidence": [_ev("GATHER")]}]
    adapter = _StubAdapter(proposed, new_evidence=[_ev("GATHER")])
    rv = ReplayValidator(_registry(adapter))
    # No `mode=` → adapter_level → action mismatch is invisible.
    out = rv.validate(skill=_skill(), seeds=[_seed(steps=[seed_step])])
    assert out.mode == "adapter_level"
    assert out.n_steps_compared == 0
    assert out.n_pass == 1   # adapter said success, that's enough


def test_action_level_invalid_mode_raises() -> None:
    rv = ReplayValidator(AdapterRegistry())
    with pytest.raises(ValueError, match="unknown"):
        rv.validate(skill=_skill(), seeds=[], mode="bogus")
