"""Phase D + Phase F crafter coverage.

* `TestPhaseDRepair` — `Repairer` rule path per `RecoveryStrategy`,
  `SkillCrafterService.propose_repair` end-to-end (artifact + DRAFT
  record), and the `cycle()` dispatch order (repair > hypothesize for
  in-bank skills, hypothesize for unknown skills).

* `TestPhaseFFrozenTeacher` — Qwen3-VL constants live in
  `common.models`, are deferred-by-default, and surface through both
  `SkillCrafterService.with_qwen3_vl_teacher` and the
  `VLM_AGENT_PHASE_F_TEACHER` env-var path. The project-wide
  `BACKBONE_TEACHER_MODEL` invariant (Qwen/Qwen3.5-35B-A3B by default)
  is preserved.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.enums import RecoveryStrategy, SkillSourceType, SkillStatus, SkillType
from common.models import (
    BACKBONE_TEACHER_MODEL,
    DEFERRED_MODELS,
    QWEN3_VL_TEACHERS,
    is_frozen_qwen_teacher,
    qwen3_vl_teacher,
)
from crafter import (
    FailureMemory,
    Repairer,
    SkillCrafterService,
)
from crafter.failure_memory import FailurePattern
from data_structure.extensions.bank_mutation_proposal import (
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from orchestrator import ArtifactStore
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName


# --------------------------------------------------------------------- helpers


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, os.path.join(root, "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, os.path.join(root, "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, os.path.join(root, "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, os.path.join(root, "archive")),
    )


def _seed_active_skill(lifecycle: SkillLifecycleManager) -> SkillRecord:
    skill = SkillRecord.new(
        name="press_then_check",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=["gymv", "browser"],
        protocol=[
            {"action": "PRESS", "payload": {"key": "${target}"}},
            {"action": "EXECUTE", "payload": {"target": "score>0"}},
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
    return skill


def _diagnosis(strategy: RecoveryStrategy, *, root_cause: str = "x") -> FailureDiagnosis:
    return FailureDiagnosis(
        failure_id="fail-stub",
        locus="protocol_step",
        root_cause=root_cause,
        recommended_strategy=strategy,
        confidence=0.5,
    )


def _pattern(skill_id: str, *, idx: int = 1, n: int = 3) -> FailurePattern:
    return FailurePattern(
        pattern_id="pat-test",
        skill_id=skill_id,
        failure_class="INVARIANT_VIOLATION",
        failed_step_index=idx,
        domains=["gymv"],
        failure_ids=[f"fail-{i}" for i in range(n)],
        sample_abort_reasons=["missing target"],
    )


# --------------------------------------------------------------------- Phase D


class TestPhaseDRepair:
    @pytest.mark.parametrize(
        ("strategy", "check"),
        [
            (
                RecoveryStrategy.HOP_INSERTION,
                lambda p: any(h["action"] == "VERIFY" for h in p.patched_protocol)
                and "VERIFY" in p.patched_contract.expected_evidence_roles,
            ),
            (
                RecoveryStrategy.PRECONDITION_STRENGTHENING,
                lambda p: "have_target" in p.patched_contract.preconditions,
            ),
            (
                RecoveryStrategy.FALLBACK_INJECTION,
                lambda p: any("_fallback" in h for h in p.patched_protocol),
            ),
            (
                RecoveryStrategy.REGROUNDING_TRIGGER,
                lambda p: any(
                    h["action"] == "GROUND" and h["payload"].get("force_refresh")
                    for h in p.patched_protocol
                )
                and "GATHER" in p.patched_contract.expected_evidence_roles,
            ),
            (
                RecoveryStrategy.PROTOCOL_PATCH,
                lambda p: [h["action"] for h in p.patched_protocol[:2]]
                in (["PRESS", "GROUND"], ["GROUND", "EXECUTE"])
                or any(h["action"] == "GROUND" for h in p.patched_protocol),
            ),
            (
                RecoveryStrategy.SKILL_DECOMPOSITION,
                lambda p: any(h.get("_decompose_hint") for h in p.patched_protocol),
            ),
        ],
    )
    def test_rule_repair_emits_patch_for_each_strategy(
        self, tmp_path, strategy, check
    ) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        base = _seed_active_skill(lifecycle)

        repairer = Repairer()
        patch = repairer.repair(
            base=base,
            pattern=_pattern(base.skill_id),
            diagnosis=_diagnosis(strategy),
            teacher_model=BACKBONE_TEACHER_MODEL,
        )
        assert isinstance(patch, PatchProposal)
        assert patch.base_skill_id == base.skill_id
        assert patch.recovery_strategy == strategy.value
        assert patch.target_domains == base.feasible_domains
        assert patch.parent_skill_ids == [base.skill_id]
        assert check(patch), f"strategy {strategy} produced unexpected patch body"

    def test_rule_repair_returns_none_for_retirement(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        base = _seed_active_skill(lifecycle)
        patch = Repairer().repair(
            base=base,
            pattern=_pattern(base.skill_id),
            diagnosis=_diagnosis(RecoveryStrategy.SKILL_RETIREMENT),
        )
        assert patch is None

    def test_rule_repair_changes_content_hash(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        base = _seed_active_skill(lifecycle)
        original_hash = base.content_hash()
        patch = Repairer().repair(
            base=base,
            pattern=_pattern(base.skill_id),
            diagnosis=_diagnosis(RecoveryStrategy.HOP_INSERTION),
        )
        # Build a draft from the patch and verify the gate-binding hash differs.
        draft = SkillRecord.new(
            name="x",
            skill_type=base.skill_type,
            source_type=patch.source_type,
            feasible_domains=base.feasible_domains,
            protocol=patch.patched_protocol,
            contract=patch.patched_contract,
            parent_skill_ids=[base.skill_id],
        )
        assert draft.content_hash() != original_hash

    def test_propose_repair_persists_patch_and_draft(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        # Lane-(b) opt-in — this test asserts the public propose_repair
        # entry point returns a PatchProposal. Live trainer default is
        # False (T1.3a / skill-lane-decision.md).
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            enable_protocol_patching=True,
        )

        base = _seed_active_skill(lifecycle)
        # FailureMemory must contain the trace the diagnoser will look up.
        memory: FailureMemory = crafter._failures  # noqa: SLF001
        for i in range(3):
            memory.add(
                FailureTrace(
                    failure_id=f"fail-{i}",
                    skill_id=base.skill_id,
                    skill_episode_id=f"ep-{i}",
                    domain="gymv",
                    failed_step_index=1,
                    failure_class="INVARIANT_VIOLATION",
                    abort_reason="invariant: empty evidence",
                )
            )
        pattern = memory.hot_patterns(min_count=2)[0]

        proposal = crafter.propose_repair(base=base, pattern=pattern)
        assert isinstance(proposal, PatchProposal)
        assert proposal.base_skill_id == base.skill_id

        # Patch proposal landed in the artifact store.
        proposal_jsons = list(artifacts._list_json("proposals"))  # noqa: SLF001
        assert any(p["proposal_id"] == proposal.proposal_id for p in proposal_jsons)
        # Draft skill landed in draft_store, with parent lineage and the
        # repaired source_type.
        drafts = repo.draft.all()
        assert len(drafts) == 1
        d = drafts[0]
        assert d.parent_skill_ids == [base.skill_id]
        assert d.source_type == SkillSourceType.REPAIRED
        assert d.proposal_id == proposal.proposal_id
        # Active store untouched (crafter scope invariant 6).
        assert len(repo.active.all()) == 1
        assert repo.active.all()[0].skill_id == base.skill_id

    def test_propose_repair_routes_to_retirement_via_cycle(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            hot_pattern_threshold=2,
        )
        base = _seed_active_skill(lifecycle)

        # Force the diagnoser to emit SKILL_RETIREMENT every time so the
        # cycle's dispatch should hit RetireProposal, not patch.
        crafter._diagnoser.set_llm_diagnoser(  # noqa: SLF001
            lambda trace: _diagnosis(
                RecoveryStrategy.SKILL_RETIREMENT, root_cause="persistent failure"
            )
        )

        failures = [
            FailureTrace(
                skill_id=base.skill_id,
                skill_episode_id=f"ep-{i}",
                domain="gymv",
                failed_step_index=1,
                failure_class="MISSING_ADAPTER",
            )
            for i in range(2)
        ]
        result = crafter.cycle(new_failures=failures)
        assert any(isinstance(p, RetireProposal) for p in result.proposals)
        # No PatchProposal for this hot pattern.
        assert not any(isinstance(p, PatchProposal) for p in result.proposals)

    def test_cycle_prefers_repair_when_skill_is_in_bank(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        # Lane-(b) opt-in — asserts the cycle dispatch emits a PatchProposal.
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            hot_pattern_threshold=2,
            enable_protocol_patching=True,
        )
        base = _seed_active_skill(lifecycle)
        n_before = len(repo.draft.all())

        failures = [
            FailureTrace(
                skill_id=base.skill_id,
                skill_episode_id=f"ep-{i}",
                domain="gymv",
                failed_step_index=1,
                failure_class="INVARIANT_VIOLATION",
                abort_reason="invariant: empty evidence",
            )
            for i in range(3)
        ]
        result = crafter.cycle(new_failures=failures)
        # At least one PatchProposal landed for the in-bank skill.
        patches = [p for p in result.proposals if isinstance(p, PatchProposal)]
        assert patches, "cycle should emit a PatchProposal for an in-bank skill"
        assert patches[0].base_skill_id == base.skill_id
        # Draft store grew by exactly one (the patched draft).
        assert len(repo.draft.all()) == n_before + len(patches)

    def test_cycle_falls_back_to_hypothesis_for_unknown_skill(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            hot_pattern_threshold=2,
        )
        failures = [
            FailureTrace(
                skill_id="s-not-in-bank",
                skill_episode_id=f"ep-{i}",
                domain="gymv",
                failed_step_index=1,
                failure_class="INVARIANT_VIOLATION",
            )
            for i in range(3)
        ]
        result = crafter.cycle(new_failures=failures)
        assert not any(isinstance(p, PatchProposal) for p in result.proposals)
        assert result.proposals, "hypothesizer should still propose for unknown skills"


# --------------------------------------------------------------------- Phase F


class TestPhaseFFrozenTeacher:
    def test_qwen3_vl_constants_registered(self) -> None:
        assert qwen3_vl_teacher("32b") == "Qwen/Qwen3-VL-32B"
        assert qwen3_vl_teacher("235B-A22B") == "Qwen/Qwen3-VL-235B-A22B"
        # Aliases — case-insensitive.
        assert qwen3_vl_teacher("32B") == "Qwen/Qwen3-VL-32B"
        with pytest.raises(ValueError, match="Unknown Qwen3-VL teacher size"):
            qwen3_vl_teacher("8b")

    def test_qwen3_vl_models_are_deferred(self) -> None:
        for m in QWEN3_VL_TEACHERS.values():
            assert m in DEFERRED_MODELS
            assert is_frozen_qwen_teacher(m)
        # GPT-style judges are not classified as a frozen Qwen teacher.
        assert not is_frozen_qwen_teacher("gpt-5.5")
        assert not is_frozen_qwen_teacher("gpt-4o")
        # Default crafter teacher is the project-wide control-plane backbone.
        assert BACKBONE_TEACHER_MODEL == "Qwen/Qwen3.5-35B-A3B"

    def test_with_qwen3_vl_teacher_constructor(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService.with_qwen3_vl_teacher(
            lifecycle=lifecycle, artifact_store=artifacts, size="32b"
        )
        assert crafter.teacher_model == "Qwen/Qwen3-VL-32B"
        assert crafter.is_phase_f_active

    def test_set_teacher_model_swap(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)
        assert crafter.teacher_model == BACKBONE_TEACHER_MODEL
        assert not crafter.is_phase_f_active

        crafter.set_teacher_model(qwen3_vl_teacher("235b-a22b"))
        assert crafter.teacher_model == "Qwen/Qwen3-VL-235B-A22B"
        assert crafter.is_phase_f_active
        with pytest.raises(ValueError, match="non-empty"):
            crafter.set_teacher_model("")

    def test_from_env_phase_f_switch(self, tmp_path, monkeypatch) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))

        monkeypatch.setenv("VLM_AGENT_PHASE_F_TEACHER", "32b")
        crafter = SkillCrafterService.from_env(
            lifecycle=lifecycle, artifact_store=artifacts
        )
        assert crafter.teacher_model == "Qwen/Qwen3-VL-32B"

        monkeypatch.delenv("VLM_AGENT_PHASE_F_TEACHER", raising=False)
        crafter2 = SkillCrafterService.from_env(
            lifecycle=lifecycle, artifact_store=artifacts
        )
        # Without the env-var, the default backbone teacher is used.
        assert crafter2.teacher_model == BACKBONE_TEACHER_MODEL

    def test_from_env_invalid_phase_f_value_raises(
        self, tmp_path, monkeypatch
    ) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))

        monkeypatch.setenv("VLM_AGENT_PHASE_F_TEACHER", "totally-bogus")
        with pytest.raises(ValueError, match="Unknown Qwen3-VL teacher size"):
            SkillCrafterService.from_env(
                lifecycle=lifecycle, artifact_store=artifacts
            )

    def test_phase_f_teacher_appears_on_emitted_proposals(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        # Lane-(b) opt-in: this test checks the teacher slot stamps on
        # the emitted PatchProposal, which requires the Repairer path
        # live. Live trainer default (T1.3a) is False.
        crafter = SkillCrafterService.with_qwen3_vl_teacher(
            lifecycle=lifecycle, artifact_store=artifacts, size="32b",
            enable_protocol_patching=True,
        )
        base = _seed_active_skill(lifecycle)
        # Inject the failure pattern directly so the test doesn't depend
        # on the diagnoser default rule path.
        memory: FailureMemory = crafter._failures  # noqa: SLF001
        memory.add(
            FailureTrace(
                failure_id="fail-1",
                skill_id=base.skill_id,
                domain="gymv",
                failed_step_index=1,
                failure_class="INVARIANT_VIOLATION",
                abort_reason="invariant: empty evidence",
            )
        )
        pattern = memory.hot_patterns(min_count=1)[0]
        patch = crafter.propose_repair(base=base, pattern=pattern)
        assert isinstance(patch, PatchProposal)
        assert patch.teacher_model == "Qwen/Qwen3-VL-32B"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
