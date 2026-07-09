"""Day-7c: tests for `SkillLifecycleManager.record_task_verification`.

Closes the Day-5b → Day-6 hand-off: when a Stage-3a transfer cycle
admits a target task, the verdict needs to land in the bank so the
next eligibility filter sees ``task_match == "verified"``.

Pins:

  * `record_task_verification` is the *only* sanctioned writer of
    `verified_tasks` and the matching `adapter_history` entries;
  * The function appends one ``adapter_history`` entry per task with
    ``kind == "task_verification"``;
  * Calling the function on a frozen-store skill round-trips: a fresh
    repo loaded from disk sees the new `verified_tasks`;
  * Empty rationale / empty `verified_tasks` are rejected;
  * Re-registering an already-verified task is a no-op (idempotent).
"""
from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.lifecycle import LifecycleError
from skill_bank.stores import StoreName


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(root, "archive")),
    )


def _draft_skill() -> SkillRecord:
    return SkillRecord.new(
        name="commit_merge",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        source_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight", "tetris"],
        protocol=[{"action": "STEP", "payload": {}}],
        contract=SkillContract(),
    )


def test_record_task_verification_appends_to_verified_tasks(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    rec = lifecycle.record_task_verification(
        skill.skill_id,
        verified_tasks=["tetris"],
        evaluation_id="eval-007",
        per_task_metrics={"tetris": {"pass_rate": 0.83, "k_used": 3}},
        rationale="Day-5b transfer cycle PASS",
    )

    assert rec.verified_tasks == ["tetris"]
    # adapter_history gained one entry tagged kind="task_verification".
    th = [e for e in rec.adapter_history if e.get("kind") == "task_verification"]
    assert len(th) == 1
    assert th[0]["target_task"] == "tetris"
    assert th[0]["evaluation_id"] == "eval-007"
    assert th[0]["metrics"] == {"pass_rate": 0.83, "k_used": 3}
    assert th[0]["rationale"] == "Day-5b transfer cycle PASS"


def test_record_task_verification_round_trips_to_disk(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)
    lifecycle.record_task_verification(
        skill.skill_id,
        verified_tasks=["tetris", "candy_crush"],
        rationale="ok",
    )

    repo2 = _new_repo(str(tmp_path))
    loaded = repo2.get(skill.skill_id)
    assert loaded is not None
    assert sorted(loaded.verified_tasks) == ["candy_crush", "tetris"]


def test_record_task_verification_idempotent(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    lifecycle.record_task_verification(
        skill.skill_id, verified_tasks=["tetris"], rationale="first"
    )
    rec = lifecycle.record_task_verification(
        skill.skill_id, verified_tasks=["tetris"], rationale="dup"
    )
    # `verified_tasks` doesn't double-up.
    assert rec.verified_tasks == ["tetris"]
    # …but the second call still leaves an audit trail.
    th = [e for e in rec.adapter_history if e.get("kind") == "task_verification"]
    assert len(th) == 2


def test_record_task_verification_rejects_empty_rationale(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    with pytest.raises(LifecycleError, match="rationale"):
        lifecycle.record_task_verification(
            skill.skill_id, verified_tasks=["tetris"], rationale=""
        )


def test_record_task_verification_rejects_empty_tasks(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    with pytest.raises(LifecycleError, match="non-empty verified_tasks"):
        lifecycle.record_task_verification(
            skill.skill_id, verified_tasks=[], rationale="x"
        )


def test_record_task_verification_unknown_skill_id(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    with pytest.raises(LifecycleError, match="Unknown skill"):
        lifecycle.record_task_verification(
            "no-such-skill",
            verified_tasks=["tetris"],
            rationale="x",
        )
