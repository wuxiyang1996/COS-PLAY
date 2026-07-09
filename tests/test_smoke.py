"""End-to-end smoke test: a tiny rollout exercising every Phase A/B/C module.

This is *not* a correctness test — it's a wiring test. It builds a
minimal fake env + actor, runs one outer episode with one ACTIVE skill,
ingests a synthetic failure into the crafter, and verifies that all
artifact subdirectories are populated.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.enums import SkillSourceType, SkillStatus, SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness import AdapterRegistry, HarnessConfig, SkillHarness
from harness.adapters import BrowserAdapter, GymvAdapter
from orchestrator import (
    ArtifactStore,
    BudgetController,
    EpisodeRunner,
    OrchestratorConfig,
)
from orchestrator.runner import ActorChoice
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName


class _Env:
    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> StateSchema:
        return StateSchema(task="demo", domain="gymv", facts={"score": 0})

    def step(self, ep):
        self.tick += 1
        next_state = StateSchema(task="demo", domain="gymv", facts={"score": self.tick})
        next_state.outer_step = self.tick
        return next_state, self.tick >= 2


class _Actor:
    def __init__(self, skill: SkillRecord) -> None:
        self._skill = skill

    def choose_action(self, state, eligible):
        if not eligible:
            return None
        return ActorChoice(skill=eligible[0].skill, rationale="first_eligible")


def test_smoke_end_to_end(tmp_path) -> None:
    bank_root = str(tmp_path / "bank")
    art_root = str(tmp_path / "art")

    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(bank_root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(bank_root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(bank_root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(bank_root, "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(art_root)

    skill = SkillRecord.new(
        name="press_then_check",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=["gymv", "browser"],
        protocol=[
            {"action": "PRESS", "payload": {"key": "${target}"}},
            {"action": "VERIFY", "payload": {"target": "score>0"}},
        ],
        contract=SkillContract(
            preconditions=["have_target"],
            expected_evidence_roles=["VERIFY"],
            success_criteria=["committed"],
        ),
    )
    lifecycle.ingest_draft(skill)
    lifecycle.transition(skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok")
    lifecycle.transition(skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="seed")

    registry = AdapterRegistry()
    registry.register(GymvAdapter())
    registry.register(BrowserAdapter())
    harness = SkillHarness(registry, config=HarnessConfig())

    env = _Env()
    actor = _Actor(skill)
    runner = EpisodeRunner(
        env=env, actor=actor, harness=harness, bank=repo, artifact_store=artifacts
    )
    budget = BudgetController()
    result = runner.run(budget=budget, max_outer_steps=4)

    assert not result.aborted, result.abort_reason
    assert result.outer_steps >= 1
    # Skill episodes were persisted.
    assert any(p.endswith(".json") for p in os.listdir(os.path.join(art_root, "skill_episodes")))
    # Outer episode meta persisted.
    assert any(p.endswith(".json") for p in os.listdir(os.path.join(art_root, "episodes")))


def test_smoke_crafter_cycle(tmp_path) -> None:
    from crafter import SkillCrafterService

    bank_root = str(tmp_path / "bank")
    art_root = str(tmp_path / "art")
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(bank_root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(bank_root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(bank_root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(bank_root, "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(art_root)
    crafter = SkillCrafterService(
        lifecycle=lifecycle, artifact_store=artifacts, hot_pattern_threshold=2
    )

    failures = [
        FailureTrace(
            skill_id="s-broken",
            skill_episode_id=f"ep-{i}",
            domain="gymv",
            failed_step_index=1,
            failure_class="INVARIANT_VIOLATION",
            abort_reason="invariant: empty evidence",
        )
        for i in range(3)
    ]
    result = crafter.cycle(new_failures=failures)
    assert result.n_failures_ingested == 3
    assert result.n_patterns_examined >= 1
    # Hypothesizer should have proposed at least one new draft skill.
    assert len(repo.draft.all()) >= 1
    assert len(result.proposals) >= 1
