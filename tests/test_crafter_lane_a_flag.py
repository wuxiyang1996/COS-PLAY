"""Lane-(a) feature flag (T1.3a) — `enable_protocol_patching`.

Spec: ``implementation_notes/legacy/skill-lane-decision.md`` §3.4 (Crafter modes
that go dark) + ``implementation_notes/pre-training-readiness-audit.md``
§0.4 (T1.3a row).

Verifies the live trainer Crafter default — ``enable_protocol_patching=False`` —
parks the Repairer / PatchProposal mint path and routes failure signals
to the Hypothesizer fall-through. Lane-(b) opt-in (``=True``) restores
the legacy Repairer path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from data_structure.extensions.bank_mutation_proposal import (
    HypothesisProposal,
    PatchProposal,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from crafter.service import SkillCrafterService
from orchestrator.artifact_store import ArtifactStore
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName


def _new_repo(root: str) -> SkillRepository:
    return SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(Path(root) / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(Path(root) / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(Path(root) / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(Path(root) / "archive")),
    )


def _seed_active_skill(lifecycle: SkillLifecycleManager) -> SkillRecord:
    """Seed an ACTIVE multi-domain skill into the lifecycle.

    Historical note: the legacy ``feasible_domains < 2`` invariant was
    dropped in T1.3d (lane-(a)). This helper still seeds a multi-domain
    skill because the test fixture pre-dates the change; the behaviour
    is unaffected because the new ``min_retrievals_per_skill`` gate
    defaults to 0 (no enforcement) when the lifecycle is constructed
    without a threshold.
    """
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
            effects_add=["committed"],
            expected_evidence_roles=["GATHER", "COMMIT"],
            success_criteria=["committed"],
        ),
    )
    lifecycle.ingest_draft(skill)
    lifecycle.transition(
        skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="seed",
    )
    lifecycle.transition(
        skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="seed",
    )
    return skill


# ---------------------------------------------------------------------- tests


def test_default_disables_protocol_patching(tmp_path):
    """Default ``enable_protocol_patching`` must be ``False`` (lane-(a))."""
    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(str(tmp_path / "art"))
    crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)
    assert crafter.enable_protocol_patching is False, (
        "Live trainer default must be False under the lane-(a) decision "
        "(skill = retrieval payload). See "
        "implementation_notes/legacy/skill-lane-decision.md."
    )


def test_lane_a_default_routes_known_skill_failure_to_hypothesizer(tmp_path):
    """With ``enable_protocol_patching=False`` (default) and a failing
    pattern whose ``skill_id`` resolves in the bank, the dispatcher's
    ``_STATUS_NO_OP`` fall-through must route to the Hypothesizer.

    Lane (a) carries failure signal through the Hypothesizer instead of
    minting protocol-edits on a non-executable retrieval payload.
    """
    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(str(tmp_path / "art"))
    crafter = SkillCrafterService(
        lifecycle=lifecycle, artifact_store=artifacts,
        # Explicit default — lane-(a).
        enable_protocol_patching=False,
        # This test asserts the dispatcher routes to the Hypothesizer
        # on a single failure when the patch path is parked. Disable
        # the post-v11 hypothesizer-fallthrough gates (recurrence ≥ 3
        # AND no related skill) so the test exercises the dispatch
        # mechanic in isolation. Production runs use the gate
        # defaults — see SkillCrafterService.__init__.
        hypothesize_min_recurrences=1,
        hypothesize_related_skill_jaccard=0.0,
    )
    base = _seed_active_skill(lifecycle)

    failure = FailureTrace(
        skill_id=base.skill_id,
        skill_episode_id="ep-0",
        domain="gymv",
        failed_step_index=1,
        failure_class="INVARIANT_VIOLATION",
        abort_reason="invariant: empty evidence",
    )
    result = crafter.reflect_on_episode(
        EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[failure],
        )
    )

    # Patches must be parked entirely.
    assert not any(isinstance(p, PatchProposal) for p in result.proposals), (
        "lane-(a) default must NOT mint PatchProposal — got "
        f"{[type(p).__name__ for p in result.proposals]}"
    )
    # Hypothesizer should still fire as the fall-through.
    hypotheses = [p for p in result.proposals if isinstance(p, HypothesisProposal)]
    assert hypotheses, (
        "lane-(a) default must route failure signal to the Hypothesizer; "
        f"got {[type(p).__name__ for p in result.proposals]}"
    )
    # Coalesce / cooldown must be inert in lane-(a).
    assert result.n_patches_coalesced == 0
    assert result.n_patches_skipped_cooldown == 0


def test_lane_b_optin_restores_patch_proposal_path(tmp_path):
    """With ``enable_protocol_patching=True``, the legacy Repairer path
    is live and PatchProposal records appear again — the lane-(b)
    regression surface."""
    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(str(tmp_path / "art"))
    crafter = SkillCrafterService(
        lifecycle=lifecycle, artifact_store=artifacts,
        enable_protocol_patching=True,
    )
    assert crafter.enable_protocol_patching is True
    base = _seed_active_skill(lifecycle)

    failure = FailureTrace(
        skill_id=base.skill_id,
        skill_episode_id="ep-0",
        domain="gymv",
        failed_step_index=1,
        failure_class="INVARIANT_VIOLATION",
        abort_reason="invariant: empty evidence",
    )
    result = crafter.reflect_on_episode(
        EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[failure],
        )
    )
    # At least one PatchProposal landed.
    assert any(isinstance(p, PatchProposal) for p in result.proposals), (
        "lane-(b) opt-in must mint a PatchProposal — got "
        f"{[type(p).__name__ for p in result.proposals]}"
    )


def test_lane_a_unknown_skill_still_falls_through_to_hypothesizer(tmp_path):
    """With patching disabled and a pattern whose ``skill_id`` is NOT in
    the bank, the dispatcher already takes the existing fall-through to
    the Hypothesizer (the ``base is None`` branch). The lane-(a) gate
    must not regress this — the proposal volume should be unchanged
    from pre-flag behavior on this code path."""
    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(str(tmp_path / "art"))
    crafter = SkillCrafterService(
        lifecycle=lifecycle, artifact_store=artifacts,
        hot_pattern_threshold=2,
        enable_protocol_patching=False,
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
    # Hypothesizer should still produce at least one proposal.
    assert result.proposals, (
        "Hypothesizer must continue to fire on unknown-skill failures "
        "regardless of the protocol-patching flag."
    )
    assert not any(isinstance(p, PatchProposal) for p in result.proposals)


def test_lane_a_propose_repair_public_api_returns_none(tmp_path):
    """The public ``propose_repair`` entrypoint must respect the flag —
    callers that want a PatchProposal under lane-(a) get ``None`` back,
    matching the existing diagnoser short-circuit contract."""
    from crafter.failure_memory import FailureMemory

    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(str(tmp_path / "art"))
    crafter = SkillCrafterService(
        lifecycle=lifecycle, artifact_store=artifacts,
        enable_protocol_patching=False,
    )
    base = _seed_active_skill(lifecycle)
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
    assert proposal is None, (
        "lane-(a) default must short-circuit propose_repair to None; "
        "the dispatcher carries the signal through the Hypothesizer."
    )
    # No draft skill landed in the draft store either.
    assert len(repo.draft.all()) == 0
