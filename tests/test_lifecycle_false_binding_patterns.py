"""Day-9c: tests for `SkillLifecycleManager.record_false_binding_pattern`.

Closes the hand-off between the eligibility filter's `RejectedSkill`
channel (Day-8a) and the Crafter's `false_binding_patterns` evidence
loop (PLAN-SKILL-BANK §4.3b).

Pins:

  * `record_false_binding_pattern` is the *only* sanctioned writer of
    `SkillRecord.false_binding_patterns`;
  * Re-recording the same ``(veto, domain, task)`` triple is dedup'd
    (the existing entry's ``count`` is incremented; no duplicate row);
  * Different domains / tasks become separate entries;
  * The list is capped at ``max_patterns`` via FIFO eviction;
  * The change round-trips to disk.
"""
from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import SkillSourceType, SkillType
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


def _draft_skill(name: str = "merge_pair") -> SkillRecord:
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


def test_record_false_binding_pattern_appends(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    rec = lifecycle.record_false_binding_pattern(
        skill.skill_id,
        veto="binding_failed",
        veto_reason="slot ${target_tile} unbound",
        domain="gymv",
        task="tetris",
    )
    assert len(rec.false_binding_patterns) == 1
    p = rec.false_binding_patterns[0]
    assert p["veto"] == "binding_failed"
    assert p["count"] == 1
    assert p["domain"] == "gymv"
    assert p["task"] == "tetris"
    assert "first_observed_at" in p
    assert "last_observed_at" in p


def test_record_false_binding_pattern_dedupes_same_triple(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    lifecycle.record_false_binding_pattern(
        skill.skill_id,
        veto="binding_failed",
        veto_reason="r1",
        domain="gymv",
        task="tetris",
    )
    rec = lifecycle.record_false_binding_pattern(
        skill.skill_id,
        veto="binding_failed",
        veto_reason="r2",
        domain="gymv",
        task="tetris",
    )
    # Single entry, count=2.
    assert len(rec.false_binding_patterns) == 1
    p = rec.false_binding_patterns[0]
    assert p["count"] == 2


def test_record_false_binding_pattern_different_triples_split(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    lifecycle.record_false_binding_pattern(
        skill.skill_id, veto="binding_failed",
        veto_reason="r", domain="gymv", task="tetris",
    )
    lifecycle.record_false_binding_pattern(
        skill.skill_id, veto="binding_failed",
        veto_reason="r", domain="gymv", task="candy_crush",
    )
    rec = lifecycle.record_false_binding_pattern(
        skill.skill_id, veto="precondition_failed",
        veto_reason="r", domain="gymv", task="tetris",
    )
    assert len(rec.false_binding_patterns) == 3


def test_record_false_binding_pattern_round_trips_to_disk(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)
    lifecycle.record_false_binding_pattern(
        skill.skill_id, veto="binding_failed",
        veto_reason="reason", domain="osworld", task="bash_open_app",
    )

    repo2 = _new_repo(str(tmp_path))
    loaded = repo2.get(skill.skill_id)
    assert loaded is not None
    assert len(loaded.false_binding_patterns) == 1
    assert loaded.false_binding_patterns[0]["veto"] == "binding_failed"
    assert loaded.false_binding_patterns[0]["domain"] == "osworld"


def test_record_false_binding_pattern_max_cap_fifo(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    for i in range(5):
        lifecycle.record_false_binding_pattern(
            skill.skill_id, veto=f"v{i}", veto_reason="r",
            domain="gymv", task=f"t{i}",
            max_patterns=3,
        )
    rec = repo.get(skill.skill_id)
    assert rec is not None
    assert len(rec.false_binding_patterns) == 3
    # FIFO: the kept slice should be the most recent three (v2, v3, v4).
    assert [p["veto"] for p in rec.false_binding_patterns] == ["v2", "v3", "v4"]


def test_record_false_binding_pattern_unknown_skill_id(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    with pytest.raises(LifecycleError, match="Unknown skill"):
        lifecycle.record_false_binding_pattern(
            "no-such-skill",
            veto="binding_failed",
            veto_reason="r",
        )
