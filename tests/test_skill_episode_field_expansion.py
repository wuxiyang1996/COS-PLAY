"""Day-8b/c: pin the new `SkillEpisode` / `SkillEpisodeStep` /
`SkillEpisodeOutcome` / `SkillEvaluationRecord` fields.

Closes harness/README §10 + §11. The new fields are *additive* — the
prior shape continues to round-trip through `to_json()` unchanged.
"""
from __future__ import annotations

from common.enums import (
    EVIDENCE_ROLES,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.state_schema import EvidenceRef
from data_structure.extensions.gate_verdict import GateVerdictPayload
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord


def _ev(role: str = "GATHER") -> EvidenceRef:
    return EvidenceRef(source="schema", locator="x", role=role)


def test_skill_episode_step_evidence_in_out_split() -> None:
    step = SkillEpisodeStep(
        step_index=0,
        action_type="VERIFY",
        action_payload={},
        pre_state=None,
        post_state=None,
        evidence_in=[_ev("GATHER")],
        evidence_out=[_ev("VERIFY")],
        protocol_index=2,
        evidence_warrant={"ref": "ev-1"},
        verify_verdict={"verdict": "matches"},
    )
    j = step.to_json()
    assert len(j["evidence_in"]) == 1
    assert j["evidence_in"][0]["role"] == "GATHER"
    assert len(j["evidence_out"]) == 1
    assert j["evidence_out"][0]["role"] == "VERIFY"
    assert j["protocol_index"] == 2
    assert j["evidence_warrant"] == {"ref": "ev-1"}
    assert j["verify_verdict"] == {"verdict": "matches"}


def test_skill_episode_step_legacy_evidence_mirrors_into_out() -> None:
    """Adapter authors writing the legacy `evidence` field still
    populate `evidence_out` for new readers."""
    step = SkillEpisodeStep(
        step_index=0,
        action_type="STEP",
        action_payload={},
        pre_state=None,
        post_state=None,
        evidence=[_ev("GATHER")],
    )
    assert len(step.evidence_out) == 1
    assert step.evidence_out[0].role == "GATHER"
    # Round-trips both ways.
    j = step.to_json()
    assert len(j["evidence_out"]) == 1
    assert len(j["evidence"]) == 1


def test_skill_episode_outcome_contract_progress_and_reward_components() -> None:
    out = SkillEpisodeOutcome(
        success=True,
        contract_satisfied=True,
        evidence_role=["GATHER"],
        contract_progress={
            "effects_add[0]": True,
            "effects_add[1]": False,
            "expected_evidence_role[GATHER]": True,
        },
        reward_components={
            "r_env": 0.7,
            "r_follow": 0.2,
            "r_cost": -0.05,
            "r_total": 0.85,
        },
    )
    j = out.to_json()
    assert j["contract_progress"]["effects_add[1]"] is False
    assert j["reward_components"]["r_total"] == 0.85


def test_skill_episode_shadow_and_diagnostic_labels() -> None:
    ep = SkillEpisode(
        episode_id="e1",
        skill_id="s",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
        shadow=True,
        transfer_label="g0_violation",
    )
    # transfer_label is auto-mirrored into diagnostic_labels.
    assert "g0_violation" in ep.diagnostic_labels
    j = ep.to_json()
    assert j["shadow"] is True
    assert "g0_violation" in j["diagnostic_labels"]
    # Re-init via `__post_init__` doesn't double-add when re-constructed.
    ep2 = SkillEpisode(
        episode_id="e2",
        skill_id="s",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
        diagnostic_labels=["g0_violation", "transfer:domain_mismatch"],
        transfer_label="g0_violation",
    )
    assert ep2.diagnostic_labels.count("g0_violation") == 1


def test_skill_episode_protocol_trace_mirror() -> None:
    ep = SkillEpisode(
        episode_id="e1",
        skill_id="s",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
    )
    ep.add_step(SkillEpisodeStep(
        step_index=0, action_type="A", action_payload={},
        pre_state=None, post_state=None, protocol_index=0,
    ))
    ep.add_step(SkillEpisodeStep(
        step_index=1, action_type="OBS", action_payload={},
        pre_state=None, post_state=None, protocol_index=None,
    ))
    ep.add_step(SkillEpisodeStep(
        step_index=2, action_type="B", action_payload={},
        pre_state=None, post_state=None, protocol_index=1,
    ))
    assert ep.protocol_trace == [0, None, 1]
    assert ep.to_json()["protocol_trace"] == [0, None, 1]


def test_skill_evaluation_record_anchors_round_trip() -> None:
    r = SkillEvaluationRecord(
        evaluation_id="eval-1",
        proposal_id="p-1",
        skill_id="s-1",
        skill_content_hash="ABCD",
        bank_snapshot_id="snap-2026-04-30",
        eval_suite_id="gymv-suite-v0",
        adapter_versions={"gymv": "v3", "browser": "v1"},
        ontology_version="ont-1.2",
        version="v3",
        status_before=SkillStatus.PROVISIONAL,
        status_after=SkillStatus.ACTIVE,
        rejected_domains=["video"],
        rollback_target=None,
        diagnostic_labels=["transfer:label_unbound"],
    )
    j = r.to_json()
    assert j["bank_snapshot_id"] == "snap-2026-04-30"
    assert j["eval_suite_id"] == "gymv-suite-v0"
    assert j["adapter_versions"] == {"gymv": "v3", "browser": "v1"}
    assert j["ontology_version"] == "ont-1.2"
    assert j["version"] == "v3"
    assert j["status_before"] == SkillStatus.PROVISIONAL.value
    assert j["status_after"] == SkillStatus.ACTIVE.value
    assert j["rejected_domains"] == ["video"]
    assert j["diagnostic_labels"] == ["transfer:label_unbound"]


def test_skill_evaluation_record_anchors_default_to_none() -> None:
    """Back-compat: a record constructed without anchors emits them
    as None / [] rather than missing keys."""
    r = SkillEvaluationRecord(
        evaluation_id="eval-x",
        proposal_id="p-x",
        skill_id="s-x",
        skill_content_hash="ZZZ",
    )
    j = r.to_json()
    assert j["bank_snapshot_id"] is None
    assert j["eval_suite_id"] is None
    assert j["adapter_versions"] == {}
    assert j["status_before"] is None
    assert j["rejected_domains"] == []
