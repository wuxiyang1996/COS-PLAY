"""Invariant tests for the new MVP (P0 + Phases A, B, C).

These tests enforce architectural rules that the system relies on for
safety. Run with `pytest tests/test_invariants.py`.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure the project root is importable when invoked via `pytest tests/`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import (
    DOMAINS,
    GateVerdict,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.bank_mutation_proposal import HypothesisProposal
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.gate_verdict import GateVerdictPayload, StageVerdict
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness import (
    AdapterRegistry,
    HarnessConfig,
    SkillHarness,
)
from harness.adapters import BrowserAdapter, GymvAdapter
from orchestrator import (
    ArtifactStore,
    GateService,
    OrchestratorConfig,
    PromotionOrchestrator,
    PromotionPlan,
    SnapshotManager,
)
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore, StoreLockedError
from skill_bank.stores import StoreName


# --------------------------------------------------------------------- helpers


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(root, "archive")),
    )


def _new_skill(
    *,
    domains=("gymv", "browser"),
    skill_type=SkillType.MIXED,
    expected_roles=("VERIFY", "COMMIT"),
    name="demo",
) -> SkillRecord:
    return SkillRecord.new(
        name=name,
        skill_type=skill_type,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=list(domains),
        protocol=[{"action": "VERIFY", "payload": {"target": "x"}}, {"action": "COMMIT", "payload": {"target": "x"}}],
        contract=SkillContract(
            preconditions=["have_target"],
            expected_evidence_roles=list(expected_roles),
            success_criteria=["committed"],
        ),
    )


# --------------------------------------------------------------------- G0


class TestG0EvidenceInvariant:
    """A successful non-action SkillEpisode without evidence must not finalize."""

    def test_finalize_rejects_evidence_free_success(self) -> None:
        ep = SkillEpisode.begin(
            skill_id="s1",
            skill_version="v1",
            skill_type=SkillType.REASONING,
            domain="visual_reasoning",
            parent_run_id=None,
        )
        ep.add_step(
            SkillEpisodeStep(
                step_index=0,
                action_type="REASON",
                action_payload={},
                pre_state=None,
                post_state=None,
                evidence=[],
            )
        )
        with pytest.raises(ValueError, match="evidence-driven invariant"):
            ep.finalize(
                outcome=SkillEpisodeOutcome(
                    success=True,
                    contract_satisfied=True,
                    evidence_role=[],
                )
            )

    def test_finalize_accepts_when_evidence_carried(self) -> None:
        ep = SkillEpisode.begin(
            skill_id="s1",
            skill_version="v1",
            skill_type=SkillType.REASONING,
            domain="visual_reasoning",
            parent_run_id=None,
        )
        ep.add_step(
            SkillEpisodeStep(
                step_index=0,
                action_type="VERIFY",
                action_payload={},
                pre_state=None,
                post_state=None,
                evidence=[EvidenceRef(source="t", locator="0", role="VERIFY")],
            )
        )
        ep.finalize(
            outcome=SkillEpisodeOutcome(
                success=True,
                contract_satisfied=True,
                evidence_role=["VERIFY"],
            )
        )
        assert ep.outcome is not None and ep.outcome.success


# --------------------------------------------------------------------- no-mem


class TestNoMemoryInvariant:
    def test_step_rejects_query_mem_action(self) -> None:
        with pytest.raises(ValueError, match="no-memory"):
            SkillEpisodeStep(
                step_index=0,
                action_type="QUERY_MEM_KV",
                action_payload={},
                pre_state=None,
                post_state=None,
            )


# --------------------------------------------------------------------- bank lock


class TestBankWriteIsolation:
    """Only `SkillLifecycleManager` may write to a SkillStore."""

    def test_direct_put_raises(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path))
        skill = _new_skill()
        with pytest.raises(StoreLockedError):
            repo.draft.put(skill, token=object())

    def test_lifecycle_can_write_and_transition(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo)
        skill = _new_skill()
        lifecycle.ingest_draft(skill)
        assert repo.draft.get(skill.skill_id) is not None
        lifecycle.transition(skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ready for gate")
        assert repo.candidate.get(skill.skill_id) is not None
        assert repo.draft.get(skill.skill_id) is None

    def test_active_promotion_allows_single_domain_when_no_retrievals_threshold(
        self, tmp_path
    ) -> None:
        """T1.3d: lane-(a) drops the legacy 2-domain ACTIVE invariant.

        Default ``min_retrievals_per_skill=0`` means no enforcement, so
        a single-domain skill (e.g. gymv-only) MAY now transition
        DRAFT→CANDIDATE→SHADOW→ACTIVE provided the other invariants
        (evidence-roles, source/target asymmetry) hold.
        """
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo)
        single_domain = _new_skill(domains=("gymv",))
        lifecycle.ingest_draft(single_domain)
        lifecycle.transition(
            single_domain.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok"
        )
        lifecycle.transition(
            single_domain.skill_id, to_status=SkillStatus.SHADOW, rationale="shadow"
        )
        lifecycle.transition(
            single_domain.skill_id, to_status=SkillStatus.ACTIVE, rationale="lane-a"
        )
        assert repo.get(single_domain.skill_id).status == SkillStatus.ACTIVE

    def test_active_promotion_requires_min_retrievals_when_threshold_set(
        self, tmp_path
    ) -> None:
        """T1.3d: the lane-(a) replacement gate.

        When the lifecycle is constructed with
        ``min_retrievals_per_skill=N``, ACTIVE is rejected until the
        record carries ``metrics['retrievals'] >= N``. Replaces the
        legacy ``feasible_domains < 2`` check.
        """
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo, min_retrievals_per_skill=3)
        skill = _new_skill(domains=("gymv",))
        lifecycle.ingest_draft(skill)
        lifecycle.transition(
            skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok"
        )
        lifecycle.transition(
            skill.skill_id, to_status=SkillStatus.SHADOW, rationale="shadow"
        )
        with pytest.raises(Exception, match="min_retrievals_per_skill"):
            lifecycle.transition(
                skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="too-soon"
            )
        # Bump retrievals past the threshold; promotion now succeeds.
        skill.metrics["retrievals"] = 3.0
        lifecycle.transition(
            skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="ready"
        )
        assert repo.get(skill.skill_id).status == SkillStatus.ACTIVE

    def test_active_promotion_requires_evidence_roles(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo)
        no_roles = _new_skill(expected_roles=())
        lifecycle.ingest_draft(no_roles)
        lifecycle.transition(no_roles.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok")
        with pytest.raises(Exception, match="G0"):
            lifecycle.transition(no_roles.skill_id, to_status=SkillStatus.ACTIVE, rationale="try")


# --------------------------------------------------------------------- gate-bound promotion


class TestAtomicPromotion:
    def test_promote_rejects_failed_verdict(self, tmp_path) -> None:
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
        eval_record = SkillEvaluationRecord(
            evaluation_id="eval-1",
            proposal_id="p-1",
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            verdict=GateVerdictPayload(
                proposal_id="p-1",
                skill_id=skill.skill_id,
                skill_content_hash=skill.content_hash(),
                final_verdict=GateVerdict.FAIL,
                rationale="forced fail",
            ),
        )
        plan = PromotionPlan()
        plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=eval_record, rationale="x")
        with pytest.raises(Exception, match="FAIL"):
            promo.promote(plan, adapter_signature=[], config_payload={}, notes="")
        # Skill must still be CANDIDATE after the failed promotion attempt.
        assert repo.get(skill.skill_id).status == SkillStatus.CANDIDATE  # type: ignore[union-attr]

    def test_promote_rejects_content_hash_drift(self, tmp_path) -> None:
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
        eval_record = SkillEvaluationRecord(
            evaluation_id="eval-1",
            proposal_id="p-1",
            skill_id=skill.skill_id,
            skill_content_hash="deadbeef",  # wrong
            verdict=GateVerdictPayload(
                proposal_id="p-1",
                skill_id=skill.skill_id,
                skill_content_hash="deadbeef",
                final_verdict=GateVerdict.PASS,
                rationale="pass",
            ),
        )
        plan = PromotionPlan()
        plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=eval_record, rationale="x")
        with pytest.raises(Exception, match="content hash drift"):
            promo.promote(plan, adapter_signature=[], config_payload={}, notes="")

    def test_promote_succeeds_on_pass(self, tmp_path) -> None:
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
        ev = SkillEvaluationRecord(
            evaluation_id="eval-2",
            proposal_id="p-2",
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            verdict=GateVerdictPayload(
                proposal_id="p-2",
                skill_id=skill.skill_id,
                skill_content_hash=skill.content_hash(),
                final_verdict=GateVerdict.PASS,
                rationale="pass",
            ),
        )
        plan = PromotionPlan()
        plan.add(skill=skill, target_status=SkillStatus.ACTIVE, evaluation=ev, rationale="ok")
        result = promo.promote(plan, adapter_signature=["gymv", "browser"], config_payload={"k": 1})
        assert result.release is not None
        assert skill.skill_id in result.promoted_skill_ids
        assert repo.get(skill.skill_id).status == SkillStatus.ACTIVE  # type: ignore[union-attr]


# --------------------------------------------------------------------- crafter scope


class TestCrafterCandidateOnly:
    """The crafter materialises proposals as DRAFT; nothing ever lands in
    `active_store` without going through the gate + PromotionOrchestrator.
    """

    def test_crafter_writes_only_drafts(self, tmp_path) -> None:
        from crafter import SkillCrafterService

        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)

        # Seed two existing skills so we have something to compose.
        a = _new_skill(name="a")
        b = _new_skill(name="b")
        lifecycle.ingest_draft(a)
        lifecycle.ingest_draft(b)

        proposal = crafter.propose_composition(
            [a, b], name="a_then_b", rationale="demo"
        )
        # The proposal lives in artifacts; the new draft skill in draft_store.
        assert any(p["proposal_id"] == proposal.proposal_id for p in artifacts._list_json("proposals"))
        # No record landed in candidate / active / archive.
        assert len(repo.candidate.all()) == 0
        assert len(repo.active.all()) == 0


# --------------------------------------------------------------------- harness


class TestHarnessEligibility:
    def test_eligibility_excludes_candidate_status(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        s = _new_skill(skill_type=SkillType.ACTION)
        lifecycle.ingest_draft(s)
        lifecycle.transition(s.skill_id, to_status=SkillStatus.CANDIDATE, rationale="x")
        registry = AdapterRegistry()
        registry.register(GymvAdapter())
        registry.register(BrowserAdapter())
        h = SkillHarness(registry, config=HarnessConfig())
        state = StateSchema(task="t", domain="gymv")
        eligible = h.select_eligible_skills([s], state)
        assert eligible == []  # CANDIDATE is not runnable

    def test_eligibility_includes_active(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        s = _new_skill(skill_type=SkillType.ACTION)
        lifecycle.ingest_draft(s)
        lifecycle.transition(s.skill_id, to_status=SkillStatus.CANDIDATE, rationale="x")
        lifecycle.transition(s.skill_id, to_status=SkillStatus.ACTIVE, rationale="y")
        registry = AdapterRegistry()
        registry.register(GymvAdapter())
        registry.register(BrowserAdapter())
        h = SkillHarness(registry, config=HarnessConfig())
        state = StateSchema(task="t", domain="gymv")
        eligible = h.select_eligible_skills([s], state)
        assert len(eligible) == 1
        assert eligible[0].skill.skill_id == s.skill_id


# --------------------------------------------------------------------- gate


class TestGateService:
    def test_static_stage_rejects_single_domain(self, tmp_path) -> None:
        registry = AdapterRegistry()
        registry.register(GymvAdapter())
        h = SkillHarness(registry)
        gate = GateService(harness=h)
        skill = _new_skill(domains=("gymv",))
        proposal = HypothesisProposal(
            name="x",
            target_domains=["gymv"],
            novel_protocol=skill.protocol,
            contract=skill.contract,
        )
        ev = gate.evaluate(proposal=proposal, skill=skill, replay_seeds=[])
        assert ev.verdict is not None and ev.verdict.final_verdict == GateVerdict.FAIL


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
