"""Day-9a: tests for the `GateRunner` → `PromotionOrchestrator` anchor
flow.

The PromotionOrchestrator is the audit-trail authority — when it
signs off on a transition, the persisted SkillEvaluationRecord must
carry every reproducibility anchor that drove the gate decision so a
post-hoc rollback can identify the exact ``(snapshot, eval_suite,
adapter_versions)`` bundle that was active.

Pins:

  * ``status_before`` is captured *before* the lifecycle transition
    fires, so it reflects the source state, not the post-transition
    state;
  * ``status_after`` matches the target_status from the
    PromotionPlan;
  * ``bank_snapshot_id`` is pinned to the just-minted snapshot
    (override path also covered);
  * GateRunner-supplied ``eval_suite_id`` / ``adapter_versions`` /
    ``ontology_version`` survive the round trip;
  * The audit row carries the union of anchors across all transitions.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import (
    GateVerdict,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from data_structure.extensions.gate_verdict import GateVerdictPayload
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from orchestrator import (
    ArtifactStore,
    PromotionOrchestrator,
    PromotionPlan,
    SnapshotManager,
)
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(root, "archive")),
    )


def _new_skill(name: str = "transfer_demo") -> SkillRecord:
    return SkillRecord.new(
        name=name,
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=["gymv", "browser"],
        protocol=[
            {"action": "VERIFY", "payload": {"target": "x"}},
            {"action": "COMMIT", "payload": {"target": "x"}},
        ],
        contract=SkillContract(
            preconditions=["have_target"],
            expected_evidence_roles=["VERIFY", "COMMIT"],
            success_criteria=["committed"],
        ),
    )


def _evaluation_with_anchors(
    skill: SkillRecord,
    *,
    eval_suite_id: str = "suite-day9a",
    adapter_versions=None,
    ontology_version: str = "ont-v3",
) -> SkillEvaluationRecord:
    return SkillEvaluationRecord(
        evaluation_id="eval-day9a",
        proposal_id="p-day9a",
        skill_id=skill.skill_id,
        skill_content_hash=skill.content_hash(),
        verdict=GateVerdictPayload(
            proposal_id="p-day9a",
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            final_verdict=GateVerdict.PASS,
            rationale="all stages PASS",
        ),
        eval_suite_id=eval_suite_id,
        adapter_versions=dict(adapter_versions or {"gymv": "v3", "browser": "v2"}),
        ontology_version=ontology_version,
    )


def _setup_promotion(tmp_path) -> "tuple[SkillRepository, SkillLifecycleManager, ArtifactStore, PromotionOrchestrator, SkillRecord]":
    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    skill = _new_skill()
    lifecycle.ingest_draft(skill)
    lifecycle.transition(skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ready")
    artifacts = ArtifactStore(str(tmp_path / "art"))
    snap_mgr = SnapshotManager(artifacts)
    promo = PromotionOrchestrator(
        lifecycle=lifecycle, snapshot_manager=snap_mgr, artifact_store=artifacts
    )
    return repo, lifecycle, artifacts, promo, skill


def _read_audit(artifacts: ArtifactStore) -> "list[dict]":
    """`ArtifactStore.append_audit` writes a JSONL log; readers parse
    it directly. Returns rows in append order."""
    path = os.path.join(artifacts.root, "audit.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_promote_pins_status_before_to_candidate(tmp_path) -> None:
    repo, _lc, artifacts, promo, skill = _setup_promotion(tmp_path)
    ev = _evaluation_with_anchors(skill)
    plan = PromotionPlan()
    plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=ev, rationale="day9a-test")

    result = promo.promote(plan, adapter_signature=["gymv"], config_payload={})
    assert result.release is not None

    persisted = artifacts.get_evaluation(ev.evaluation_id)
    assert persisted is not None
    assert persisted["status_before"] == SkillStatus.CANDIDATE.value
    assert persisted["status_after"] == SkillStatus.ACTIVE.value


def test_promote_pins_bank_snapshot_id(tmp_path) -> None:
    repo, _lc, artifacts, promo, skill = _setup_promotion(tmp_path)
    ev = _evaluation_with_anchors(skill)
    plan = PromotionPlan()
    plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=ev, rationale="x")

    promo.promote(plan, adapter_signature=["gymv"], config_payload={})
    persisted = artifacts.get_evaluation(ev.evaluation_id)
    assert persisted is not None
    snap_id = persisted.get("bank_snapshot_id")
    assert isinstance(snap_id, str) and len(snap_id) > 0
    # And the same snapshot id appears in the release's bank_snapshot_path.
    rels = [r for r in artifacts.list_releases()
            if ev.evaluation_id in (r.get("gate_evaluation_ids") or [])]
    assert rels, "no release found for the evaluation"
    assert snap_id in rels[0]["bank_snapshot_path"]


def test_promote_caller_supplied_bank_snapshot_id_wins(tmp_path) -> None:
    repo, _lc, artifacts, promo, skill = _setup_promotion(tmp_path)
    ev = _evaluation_with_anchors(skill)
    plan = PromotionPlan()
    plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=ev, rationale="x")

    promo.promote(
        plan,
        adapter_signature=["gymv"],
        config_payload={},
        bank_snapshot_id="custom-snap-007",
    )
    persisted = artifacts.get_evaluation(ev.evaluation_id)
    assert persisted is not None
    assert persisted["bank_snapshot_id"] == "custom-snap-007"


def test_promote_preserves_gate_runner_anchors(tmp_path) -> None:
    repo, _lc, artifacts, promo, skill = _setup_promotion(tmp_path)
    ev = _evaluation_with_anchors(
        skill,
        eval_suite_id="suite-cross-domain",
        adapter_versions={"gymv": "v9", "osworld": "v1"},
        ontology_version="ont-day9a",
    )
    plan = PromotionPlan()
    plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=ev, rationale="x")

    promo.promote(plan, adapter_signature=["gymv"], config_payload={})
    persisted = artifacts.get_evaluation(ev.evaluation_id)
    assert persisted is not None
    assert persisted["eval_suite_id"] == "suite-cross-domain"
    assert persisted["adapter_versions"] == {"gymv": "v9", "osworld": "v1"}
    assert persisted["ontology_version"] == "ont-day9a"


def test_promote_audit_carries_anchor_union(tmp_path) -> None:
    repo, _lc, artifacts, promo, skill = _setup_promotion(tmp_path)
    ev = _evaluation_with_anchors(
        skill,
        eval_suite_id="suite-A",
        adapter_versions={"gymv": "v9"},
        ontology_version="ont-A",
    )
    plan = PromotionPlan()
    plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=ev, rationale="x")

    promo.promote(plan, adapter_signature=["gymv"], config_payload={})
    audit = _read_audit(artifacts)
    rows = [r for r in audit if r.get("kind") == "release"]
    assert rows, "no release row in audit"
    last = rows[-1]
    anchors = last.get("reproducibility_anchors") or {}
    assert anchors.get("eval_suite_ids") == ["suite-A"]
    assert anchors.get("ontology_versions") == ["ont-A"]
    assert anchors.get("adapter_versions") == {"gymv": "v9"}
