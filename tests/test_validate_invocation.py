"""Day-8a: tests for `SkillHarness.validate_invocation` and the
`EligibilityFilter.filter_with_rejections` companion.

Closes harness/README §9: the second-pass invocation veto and the
rejected-skill channel were missing — actor had no way to reason
about *why* a candidate was excluded.
"""
from __future__ import annotations

from typing import Any

import pytest

from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness import AdapterRegistry
from harness.adapters import GymvAdapter
from harness.eligibility import EligibleSkill, RejectedSkill
from harness.skill_harness import (
    HarnessConfig,
    SkillHarness,
    ValidateInvocationResult,
)


def _state(domain: str = "gymv", task: str = "make_gaming_env/tetris",
           evidence=None) -> StateSchema:
    return StateSchema(
        task=task,
        domain=domain,
        evidence=list(evidence or []),
    )


def _skill(
    *,
    skill_type: SkillType = SkillType.ACTION,
    protocol: list = None,
    expected_evidence_roles: list = None,
    preconditions: list = None,
    feasible_tasks: list = None,
) -> SkillRecord:
    return SkillRecord(
        skill_id="s",
        name="s",
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        status=SkillStatus.ACTIVE,
        feasible_domains=["gymv"],
        source_domains=["gymv"],
        feasible_tasks=list(feasible_tasks or []),
        protocol=protocol if protocol is not None else [
            {"action": "STEP", "payload": {}},
        ],
        contract=SkillContract(
            preconditions=list(preconditions or []),
            expected_evidence_roles=list(expected_evidence_roles or []),
        ),
    )


def _harness() -> SkillHarness:
    reg = AdapterRegistry()
    reg.register(GymvAdapter())
    return SkillHarness(reg, config=HarnessConfig())


def test_validate_invocation_pass_on_simple_action_skill() -> None:
    h = _harness()
    res = h.validate_invocation(_skill(), _state(), bindings={})
    assert isinstance(res, ValidateInvocationResult)
    assert res.ok is True
    assert res.adapter_ok is True
    assert res.binding_ok is True
    assert res.precondition_ok is True
    assert res.evidence_ok is True
    assert res.veto_reasons == []


def test_validate_invocation_vetoes_on_missing_binding() -> None:
    h = _harness()
    skill = _skill(protocol=[{"action": "MERGE", "payload": {"row": "${target_row}"}}])
    # No binding for ${target_row}.
    res = h.validate_invocation(skill, _state(), bindings={})
    assert res.ok is False
    assert res.binding_ok is False
    assert "target_row" in res.missing_bindings
    assert any("missing_bindings" in r for r in res.veto_reasons)


def test_validate_invocation_pass_when_binding_supplied() -> None:
    h = _harness()
    skill = _skill(protocol=[{"action": "MERGE", "payload": {"row": "${target_row}"}}])
    res = h.validate_invocation(skill, _state(), bindings={"target_row": 2})
    assert res.ok is True
    assert res.binding_ok is True
    assert res.missing_bindings == []


def test_validate_invocation_vetoes_on_unbound_precondition_slot() -> None:
    h = _harness()
    skill = _skill(
        protocol=[{"action": "STEP", "payload": {}}],
        preconditions=["have ${needed_thing}"],
    )
    res = h.validate_invocation(skill, _state(), bindings={})
    assert res.ok is False
    assert res.precondition_ok is False
    assert any("needed_thing" in f for f in res.failed_preconditions)


def test_validate_invocation_propagates_shadow_only() -> None:
    h = _harness()
    skill = _skill()
    elig = EligibleSkill(skill=skill, adapter_name="gymv", shadow_only=True)
    res = h.validate_invocation(skill, _state(), bindings={}, eligible=elig)
    assert res.shadow_only is True


def test_validate_invocation_vetoes_when_no_adapter() -> None:
    skill = _skill()
    # Empty registry → no adapter for the state's domain.
    h = SkillHarness(AdapterRegistry(), config=HarnessConfig())
    res = h.validate_invocation(skill, _state(), bindings={})
    assert res.ok is False
    assert res.adapter_ok is False
    assert res.adapter_name is None


def test_validate_invocation_action_skill_skips_evidence_check() -> None:
    """ACTION skills are exempt from `evidence_in` checks (G0 doesn't
    apply to ACTION, per harness/README §10)."""
    h = _harness()
    skill = _skill(
        skill_type=SkillType.ACTION,
        expected_evidence_roles=["GATHER", "VERIFY"],
    )
    res = h.validate_invocation(skill, _state(evidence=[]), bindings={})
    assert res.evidence_ok is True


def test_filter_with_rejections_emits_rejection_channel() -> None:
    """Day-8 channel: skills rejected by the eligibility filter are
    surfaced with a reason rather than silently dropped."""
    h = _harness()
    skill_wrong_domain = _skill()
    object.__setattr__(skill_wrong_domain, "feasible_domains", ["browser"])
    skill_ok = _skill()
    eligible, rejected = h._eligibility.filter_with_rejections(
        [skill_wrong_domain, skill_ok], _state(),
    )
    assert len(eligible) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], RejectedSkill)
    assert rejected[0].veto == "domain_mismatch"
    assert "browser" in rejected[0].veto_reason


def test_eligible_skill_to_json_carries_per_check_booleans() -> None:
    skill = _skill()
    elig = EligibleSkill(
        skill=skill, adapter_name="gymv",
        binding_ok=False, evidence_ok=True,
    )
    j = elig.to_json()
    assert j["binding_ok"] is False
    assert j["evidence_ok"] is True
    assert j["adapter_ok"] is True
    assert j["precondition_ok"] is True
