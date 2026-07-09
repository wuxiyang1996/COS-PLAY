"""Day-9c: tests for `harness.RejectedSkillSink`.

The sink aggregates `RejectedSkill` records emitted by the eligibility
filter (Day-8a) and lands them on `SkillRecord.false_binding_patterns`
through `SkillLifecycleManager.record_false_binding_pattern` (Day-9c).
This is the in-process bridge between the harness and the Crafter
that closes PLAN-SKILL-BANK §4.3b.

Pins:

  * `observe()` dedupes on ``(skill_id, veto, domain, task)`` and
    accumulates a count;
  * `flush_to(lifecycle)` writes every aggregated pattern through the
    *only* sanctioned writer;
  * Patterns for unknown skill_ids are skipped (not raised) so a
    transient repository in the dump driver doesn't blow up the flush;
  * `min_count` filter drops noise patterns;
  * Sink resets after a flush by default.
"""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import SkillSourceType, SkillType
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.eligibility import RejectedSkill
from harness.rejected_skill_sink import RejectedSkillSink
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
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


def _rejected(skill: SkillRecord, veto: str = "binding_failed") -> RejectedSkill:
    return RejectedSkill(skill=skill, veto=veto, veto_reason=f"{veto}-detail")


def test_sink_observe_dedupes_on_quad(tmp_path) -> None:
    sink = RejectedSkillSink()
    skill = _draft_skill()
    sink.observe([_rejected(skill)], domain="gymv", task="tetris")
    sink.observe([_rejected(skill)], domain="gymv", task="tetris")
    sink.observe([_rejected(skill)], domain="gymv", task="tetris")
    assert len(sink) == 1
    p = sink.patterns()[0]
    assert p.count == 3


def test_sink_observe_splits_by_domain_task(tmp_path) -> None:
    sink = RejectedSkillSink()
    skill = _draft_skill()
    sink.observe([_rejected(skill)], domain="gymv", task="tetris")
    sink.observe([_rejected(skill)], domain="gymv", task="candy_crush")
    sink.observe([_rejected(skill, veto="adapter_missing")],
                 domain="gymv", task="tetris")
    assert len(sink) == 3


def test_sink_flush_writes_patterns(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    sink = RejectedSkillSink()
    sink.observe([_rejected(skill, veto="binding_failed")] * 3,
                 domain="gymv", task="tetris")
    sink.observe([_rejected(skill, veto="adapter_missing")],
                 domain="gymv", task="tetris")

    report = sink.flush_to(lifecycle)
    assert report.n_skills_touched == 1
    assert report.n_patterns_written == 2
    assert report.n_errors == 0

    rec = repo.get(skill.skill_id)
    assert rec is not None
    vetoes = sorted(p["veto"] for p in rec.false_binding_patterns)
    assert vetoes == ["adapter_missing", "binding_failed"]
    # Sink reset by default.
    assert len(sink) == 0


def test_sink_flush_skips_unknown_skill_id(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    sink = RejectedSkillSink()
    skill = _draft_skill()  # NOT ingested

    sink.observe([_rejected(skill)], domain="gymv", task="tetris")
    report = sink.flush_to(lifecycle)
    assert report.n_patterns_written == 0
    assert report.n_skills_touched == 0
    assert skill.skill_id in report.skipped_unknown_skill_ids


def test_sink_flush_min_count_filter(tmp_path) -> None:
    repo = _new_repo(str(tmp_path))
    lifecycle = SkillLifecycleManager(repo)
    skill = _draft_skill()
    lifecycle.ingest_draft(skill)

    sink = RejectedSkillSink()
    sink.observe([_rejected(skill)], domain="gymv", task="tetris")
    sink.observe([_rejected(skill, veto="adapter_missing")] * 5,
                 domain="gymv", task="tetris")

    report = sink.flush_to(lifecycle, min_count=3)
    assert report.n_patterns_written == 1  # only adapter_missing (count=5)
    rec = repo.get(skill.skill_id)
    assert rec is not None
    assert [p["veto"] for p in rec.false_binding_patterns] == ["adapter_missing"]


def test_sink_class_distribution() -> None:
    sink = RejectedSkillSink()
    skill = _draft_skill()
    sink.observe([_rejected(skill, veto="binding_failed")] * 3,
                 domain="gymv", task="tetris")
    sink.observe([_rejected(skill, veto="adapter_missing")],
                 domain="gymv", task="tetris")
    dist = sink.class_distribution()
    assert dist == {"binding_failed": 1, "adapter_missing": 1}
