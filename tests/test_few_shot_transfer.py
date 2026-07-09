"""Few-shot transfer tests (PLAN-SKILL-BANK §0.4 + PLAN-UNIFIED-SKILL-GATE Stage 3a).

These tests exercise the source/target asymmetry end-to-end:

  * the new SkillRecord fields round-trip through the bank store;
  * the lifecycle invariant rejects ACTIVE promotion when no target
    domain has been verified;
  * the FewShotAdapter actually runs against the registered target
    adapter and reports a pass-rate;
  * the GateService Stage 3a fails when no target domain verifies and
    passes when ≥1 does.
"""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import (
    SOURCE_DOMAINS,
    TRANSFER_TARGET_DOMAINS,
    GateStage,
    GateVerdict,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from data_structure.extensions.gate_verdict import GateVerdictPayload, StageVerdict
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord
from common.state_schema import StateSchema
from data_structure.extensions.bank_mutation_proposal import HypothesisProposal
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness import (
    AdapterRegistry,
    FewShotAdapter,
    FewShotAdapterError,
    FewShotDemo,
    HarnessConfig,
    SkillHarness,
)
from harness.adapters import (
    BrowserAdapter,
    GymvAdapter,
    OsworldAdapter,
    VideoAdapter,
    VisualReasoningAdapter,
)
from orchestrator import GateService, OrchestratorConfig
from orchestrator.artifact_store import ArtifactStore
from orchestrator.promotion_orchestrator import PromotionOrchestrator, PromotionPlan
from orchestrator.snapshot_manager import SnapshotManager
from skill_bank import (
    SkillLifecycleManager,
    SkillRepository,
    SkillStore,
)
from skill_bank.stores import StoreName


# --------------------------------------------------------------------- helpers


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(root, "archive")),
    )


def _registry_with_all_adapters() -> AdapterRegistry:
    r = AdapterRegistry()
    r.register(GymvAdapter())
    r.register(BrowserAdapter())
    r.register(OsworldAdapter())
    r.register(VideoAdapter())
    r.register(VisualReasoningAdapter())
    return r


def _game_skill(
    *,
    name: str = "collect_evidence_chain",
    feasible_domains=("gymv", "video"),
    source_domains=("gymv",),
    transfer_target_domains=("video",),
    verified_domains=(),
    source_type: SkillSourceType = SkillSourceType.CRAFTED,
) -> SkillRecord:
    return SkillRecord.new(
        name=name,
        skill_type=SkillType.MIXED,
        source_type=source_type,
        feasible_domains=list(feasible_domains),
        source_domains=list(source_domains),
        transfer_target_domains=list(transfer_target_domains),
        verified_domains=list(verified_domains),
        protocol=[
            {"action": "VERIFY", "payload": {"target": "${target}"}},
            {"action": "COMMIT", "payload": {"target": "${target}"}},
        ],
        contract=SkillContract(
            preconditions=["have_target"],
            expected_evidence_roles=["VERIFY", "COMMIT"],
            success_criteria=["committed"],
        ),
    )


# ---------------------------------------------------------------- canonical sets


class TestCanonicalDomainSets:
    def test_source_and_target_partition(self) -> None:
        assert set(SOURCE_DOMAINS).isdisjoint(set(TRANSFER_TARGET_DOMAINS))
        assert "gymv" in SOURCE_DOMAINS
        assert {"browser", "osworld", "video", "visual_reasoning"} == set(
            TRANSFER_TARGET_DOMAINS
        )


# --------------------------------------------------------------------- record


class TestSkillRecordSourceTargetFields:
    def test_round_trip_through_bank(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo)
        s = _game_skill()
        lifecycle.ingest_draft(s)
        # Re-read from a fresh repo to force a JSON load.
        repo2 = _new_repo(str(tmp_path))
        loaded = repo2.get(s.skill_id)
        assert loaded is not None
        assert loaded.source_domains == ["gymv"]
        assert loaded.transfer_target_domains == ["video"]
        assert loaded.verified_domains == []

    def test_invalid_source_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="source_domains"):
            SkillRecord.new(
                name="bad",
                skill_type=SkillType.MIXED,
                source_type=SkillSourceType.MINED,
                feasible_domains=["gymv", "video"],
                source_domains=["browser"],  # browser is a target, not source
            )

    def test_invalid_transfer_target_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="transfer_target_domains"):
            SkillRecord.new(
                name="bad",
                skill_type=SkillType.MIXED,
                source_type=SkillSourceType.MINED,
                feasible_domains=["gymv", "video"],
                source_domains=["gymv"],
                transfer_target_domains=["gymv"],  # gymv is a source, not target
            )


# --------------------------------------------------------------------- lifecycle


class TestAsymmetricLifecycleInvariant:
    def test_active_requires_verified_target(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo)
        s = _game_skill(verified_domains=())  # nothing verified yet
        lifecycle.ingest_draft(s)
        lifecycle.transition(s.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok")
        with pytest.raises(Exception, match="verified target"):
            lifecycle.transition(
                s.skill_id, to_status=SkillStatus.ACTIVE, rationale="try"
            )

    def test_active_succeeds_when_target_verified(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path))
        lifecycle = SkillLifecycleManager(repo)
        s = _game_skill(verified_domains=("video",))
        lifecycle.ingest_draft(s)
        lifecycle.transition(s.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok")
        lifecycle.transition(s.skill_id, to_status=SkillStatus.ACTIVE, rationale="ok")
        record = repo.get(s.skill_id)
        assert record is not None and record.status == SkillStatus.ACTIVE


# --------------------------------------------------------------------- adapter


class TestFewShotAdapter:
    def test_unknown_target_domain_raises(self) -> None:
        registry = _registry_with_all_adapters()
        h = SkillHarness(registry, config=HarnessConfig())
        adapter = FewShotAdapter(harness=h)
        with pytest.raises(FewShotAdapterError, match="TRANSFER_TARGET_DOMAINS"):
            adapter.adapt(skill=_game_skill(), target_domain="gymv")

    def test_missing_adapter_returns_diagnostic(self) -> None:
        registry = AdapterRegistry()
        registry.register(GymvAdapter())  # no transfer-target adapters
        h = SkillHarness(registry, config=HarnessConfig(fail_on_missing_adapter=False))
        adapter = FewShotAdapter(harness=h)
        result = adapter.adapt(skill=_game_skill(), target_domain="video")
        assert result.diagnostic_label == "target_domain_demo_unavailable"
        assert result.n_total == 0
        assert result.success is False

    def test_adapt_runs_against_target_adapter(self) -> None:
        registry = _registry_with_all_adapters()
        h = SkillHarness(registry, config=HarnessConfig())
        adapter = FewShotAdapter(harness=h, k_shot_default=3)
        demos = [
            FewShotDemo(state=StateSchema(task=f"shot-{i}", domain="video"))
            for i in range(3)
        ]
        result = adapter.adapt(skill=_game_skill(), target_domain="video", demos=demos)
        assert result.k_used == 3
        assert result.n_total == 3
        # Stub adapter always succeeds → pass_rate == 1.0
        assert result.pass_rate == 1.0
        assert result.success is True


# --------------------------------------------------------------------- gate stage


class TestGateTransferStage:
    def test_stage_fails_when_no_target_verifies(self) -> None:
        registry = AdapterRegistry()
        registry.register(GymvAdapter())  # no target adapter at all
        h = SkillHarness(registry, config=HarnessConfig(fail_on_missing_adapter=False))
        gate = GateService(harness=h)
        skill = _game_skill()
        proposal = HypothesisProposal(
            name="x",
            target_domains=["video"],
            novel_protocol=skill.protocol,
            contract=skill.contract,
        )
        ev = gate.evaluate(proposal=proposal, skill=skill, replay_seeds=[])
        assert ev.verdict is not None
        # static stage will pass (T1.3d dropped the ≥2 feasible_domains
        # check; only domain-validity + evidence roles are gated at G0);
        # transfer stage will FAIL because no target verified.
        stage_verdicts = {s.stage.value: s.verdict for s in ev.verdict.stages}
        assert stage_verdicts["transfer"] == GateVerdict.FAIL
        assert ev.verdict.final_verdict == GateVerdict.FAIL

    def test_stage_passes_when_target_verifies(self) -> None:
        registry = _registry_with_all_adapters()
        h = SkillHarness(registry, config=HarnessConfig())
        gate = GateService(harness=h)
        skill = _game_skill()
        proposal = HypothesisProposal(
            name="x",
            target_domains=["video"],
            novel_protocol=skill.protocol,
            contract=skill.contract,
        )
        demos = {
            "video": [
                FewShotDemo(state=StateSchema(task=f"shot-{i}", domain="video"))
                for i in range(3)
            ]
        }
        ev = gate.evaluate(
            proposal=proposal,
            skill=skill,
            replay_seeds=[],
            few_shot_demos=demos,
        )
        assert ev.verdict is not None
        stage_verdicts = {s.stage.value: s.verdict for s in ev.verdict.stages}
        assert stage_verdicts["transfer"] == GateVerdict.PASS
        # Eligible domains include the source (gymv) plus the verified target (video).
        assert set(ev.verdict.eligible_domains) == {"gymv", "video"}


# --------------------------------------------------------------------- promotion


class TestPromotionWritesVerifiedDomains:
    """The PromotionOrchestrator must mirror Stage 3a verifications into
    `SkillRecord.verified_domains` *before* the lifecycle ACTIVE check."""

    def test_promote_records_verified_targets(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)

        # Skill starts with no verified targets — promotion to ACTIVE
        # would otherwise fail invariant 7.
        skill = _game_skill(verified_domains=())
        lifecycle.ingest_draft(skill)
        lifecycle.transition(
            skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ready"
        )

        # Hand-craft a passing gate evaluation whose Stage 3a (TRANSFER)
        # reports `video` as verified with concrete pass-rate metrics.
        # We bypass GateService here to isolate the promotion-path test
        # from upstream stage outcomes.
        transfer_stage = StageVerdict(
            stage=GateStage.TRANSFER,
            verdict=GateVerdict.PASS,
            metrics={
                "n_targets": 1.0,
                "n_verified_targets": 1.0,
                "min_verified_targets": 1.0,
                "pass_rate.video": 1.0,
                "k_used.video": 3.0,
            },
            notes="verified_targets=['video']",
        )
        verdict = GateVerdictPayload(
            proposal_id="prop-test",
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            stages=[transfer_stage],
            final_verdict=GateVerdict.PASS,
            rationale="all_stages_pass",
            eligible_domains=["gymv", "video"],
        )
        ev = SkillEvaluationRecord(
            evaluation_id="eval-promotion-test",
            proposal_id="prop-test",
            skill_id=skill.skill_id,
            skill_content_hash=skill.content_hash(),
            verdict=verdict,
        )

        artifacts = ArtifactStore(str(tmp_path / "art"))
        snap = SnapshotManager(artifacts)
        promo = PromotionOrchestrator(
            lifecycle=lifecycle, snapshot_manager=snap, artifact_store=artifacts
        )
        plan = PromotionPlan()
        plan.add(
            skill=skill,
            target_status=SkillStatus.ACTIVE,
            evaluation=ev,
            rationale="ok",
        )
        result = promo.promote(plan, adapter_signature=["gymv"], config_payload={})

        promoted = repo.get(skill.skill_id)
        assert promoted is not None
        assert promoted.status == SkillStatus.ACTIVE
        assert "video" in promoted.verified_domains
        # adapter_history records the verification with the eval id and
        # the per-target metrics that the gate emitted.
        history = [
            entry for entry in promoted.adapter_history
            if entry.get("target_domain") == "video"
        ]
        assert len(history) == 1
        assert history[0]["evaluation_id"] == "eval-promotion-test"
        assert history[0]["metrics"] == {"pass_rate": 1.0, "k_used": 3.0}
        assert result.release is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
