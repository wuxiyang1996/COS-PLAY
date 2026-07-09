"""Day-9b: integration test for the `--persist` wiring on
`labeling_supplement._phase4_transfer_cycle`.

The phase-4 driver mostly exercises the `FewShotAdapter` cycle, which
is itself heavily integration-tested by `test_few_shot_transfer.py`.
This test pins the *new bit* — that calling
`_seed_lifecycle_for_persistence` on a writable bank root, followed
by `lifecycle.record_task_verification(...)`, lands the change on
disk such that re-loading the repo from a fresh handle sees the
verified_tasks entry.

This is the round-trip the driver's `--persist` flag relies on; if
this test holds, the driver's persistence is sound.
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import SkillSourceType, SkillStatus, SkillType
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from labeling_supplement._phase4_transfer_cycle import (
    _seed_lifecycle_for_persistence,
)
from skill_bank import SkillRepository, SkillStore
from skill_bank.stores import StoreName


def _draft_action_skill(name: str = "merge_pair") -> SkillRecord:
    return SkillRecord.new(
        name=name,
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        source_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
        protocol=[{"action": "STEP", "payload": {}}],
        contract=SkillContract(),
    )


def _open_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(root, "archive")),
    )


def test_seed_lifecycle_creates_provisional_records(tmp_path) -> None:
    bank_root = tmp_path / "bank"
    skill_a = _draft_action_skill("a")
    skill_b = _draft_action_skill("b")
    object.__setattr__(skill_a, "status", SkillStatus.DRAFT)
    object.__setattr__(skill_b, "status", SkillStatus.DRAFT)

    lifecycle, seeded = _seed_lifecycle_for_persistence(
        bank_root, [skill_a, skill_b],
    )
    assert set(seeded.keys()) == {skill_a.skill_id, skill_b.skill_id}
    repo = lifecycle.repository
    a = repo.get(skill_a.skill_id)
    b = repo.get(skill_b.skill_id)
    assert a is not None and a.status == SkillStatus.PROVISIONAL
    assert b is not None and b.status == SkillStatus.PROVISIONAL


def test_seed_lifecycle_idempotent(tmp_path) -> None:
    bank_root = tmp_path / "bank"
    skill = _draft_action_skill("a")
    object.__setattr__(skill, "status", SkillStatus.DRAFT)

    lifecycle1, seeded1 = _seed_lifecycle_for_persistence(bank_root, [skill])
    lifecycle2, seeded2 = _seed_lifecycle_for_persistence(bank_root, [skill])
    assert seeded1.keys() == seeded2.keys()
    # Second call is a no-op transition-wise; record still PROVISIONAL.
    rec = lifecycle2.repository.get(skill.skill_id)
    assert rec is not None and rec.status == SkillStatus.PROVISIONAL


def test_persist_round_trips_verified_tasks(tmp_path) -> None:
    """Mirror what the driver's --persist branch does: seed →
    record_task_verification → reopen fresh → see the change."""
    bank_root = tmp_path / "bank"
    skill = _draft_action_skill("a")
    object.__setattr__(skill, "status", SkillStatus.DRAFT)

    lifecycle, _ = _seed_lifecycle_for_persistence(bank_root, [skill])
    lifecycle.record_task_verification(
        skill.skill_id,
        verified_tasks=["tetris"],
        evaluation_id="phase4-2048-to-tetris",
        per_task_metrics={"tetris": {"pass_rate": 0.83, "k_used": 4}},
        rationale="phase4: pass_rate=0.83 k_used=4",
    )

    # Fresh repo handle picks up the on-disk change.
    repo2 = _open_repo(str(bank_root))
    loaded = repo2.get(skill.skill_id)
    assert loaded is not None
    assert "tetris" in loaded.verified_tasks
    th = [e for e in loaded.adapter_history if e.get("kind") == "task_verification"]
    assert th and th[0]["target_task"] == "tetris"
