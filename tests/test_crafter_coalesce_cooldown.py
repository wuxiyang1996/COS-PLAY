"""Coverage for the early-training noise filters introduced alongside
the two-tier trigger model:

  * **Coalesce** — when an open DRAFT ``PatchProposal`` already covers
    ``(base_skill_id, recovery_strategy)`` the new evidence is appended
    in place; ``CrafterCycleResult.n_patches_coalesced`` is bumped and
    ``proposals`` does NOT include a duplicate record.
  * **Cooldown** — when no coalescable open patch exists *and* the
    same base was patched within the last
    ``SkillCrafterService.cooldown_passes`` Crafter passes, the mint
    is skipped; ``CrafterCycleResult.n_patches_skipped_cooldown`` is
    bumped and the failure still landed in ``FailureMemory``.

Spec: implementation_notes/legacy/crafter-harness-orchestrator-roles.md
§"Two-tier trigger" (early-training noise filters), and
``crafter/service.py::CrafterCycleResult`` docstring.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from crafter import SkillCrafterService
from data_structure.extensions.bank_mutation_proposal import (
    PatchProposal,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureTrace
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
            effects_add=["committed"],
            expected_evidence_roles=["VERIFY"],
            success_criteria=["committed"],
        ),
    )
    lifecycle.ingest_draft(skill)
    lifecycle.transition(skill.skill_id, to_status=SkillStatus.CANDIDATE, rationale="ok")
    lifecycle.transition(skill.skill_id, to_status=SkillStatus.ACTIVE, rationale="seed")
    return skill


def _make_failure(base_skill_id: str, ep_id: str, *, idx: int = 1) -> FailureTrace:
    return FailureTrace(
        skill_id=base_skill_id,
        skill_episode_id=ep_id,
        domain="gymv",
        failed_step_index=idx,
        failure_class="INVARIANT_VIOLATION",
        abort_reason="invariant: empty evidence",
    )


def _build_service(tmp_path, *, cooldown_passes: int = 5) -> tuple:
    repo = _new_repo(str(tmp_path / "bank"))
    lifecycle = SkillLifecycleManager(repo)
    artifacts = ArtifactStore(str(tmp_path / "art"))
    crafter = SkillCrafterService(
        lifecycle=lifecycle,
        artifact_store=artifacts,
        cooldown_passes=cooldown_passes,
        # Coalesce/cooldown lives on the Repairer path; opt into lane-(b)
        # patching for these unit tests (live trainer default is False
        # under the lane-(a) decision — see T1.3a).
        enable_protocol_patching=True,
    )
    return repo, lifecycle, artifacts, crafter


# --------------------------------------------------------------------- tests


class TestCoalesce:
    def test_repeat_same_base_and_strategy_coalesces(self, tmp_path) -> None:
        """Five episodes worth of identical INVARIANT_VIOLATION on the
        same base ⇒ one minted PatchProposal (episode 1) and four
        coalesces (episodes 2-5). The single proposal accumulates all
        five failure_ids in its `seed_failure_ids`."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=0,        # disable cooldown to isolate coalesce
        )
        base = _seed_active_skill(lifecycle)

        first = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        assert len([p for p in first.proposals if isinstance(p, PatchProposal)]) == 1
        assert first.n_patches_coalesced == 0
        first_patch = next(p for p in first.proposals if isinstance(p, PatchProposal))
        first_proposal_id = first_patch.proposal_id
        assert len(first_patch.seed_failure_ids) == 1

        for i in range(1, 5):
            r = crafter.reflect_on_episode(EpisodeReflection(
                episode_id=f"ep-{i}",
                domain="gymv",
                failure_traces=[_make_failure(base.skill_id, f"ep-{i}")],
            ))
            patches = [p for p in r.proposals if isinstance(p, PatchProposal)]
            assert patches == [], (
                f"ep-{i}: expected coalesce, got {len(patches)} fresh PatchProposal(s)"
            )
            assert r.n_patches_coalesced == 1, (
                f"ep-{i}: expected n_patches_coalesced==1, got {r.n_patches_coalesced}"
            )

        # Single proposal artifact persists; seed_failure_ids has grown.
        live = artifacts.get_proposal(first_proposal_id)
        assert live is not None, "coalesce must overwrite the same artifact JSON"
        assert len(live["seed_failure_ids"]) == 5, (
            f"expected 5 accumulated failure_ids, got {len(live['seed_failure_ids'])}"
        )

        # Exactly ONE DRAFT skill record materialized for this base.
        drafts_for_base = [
            r for r in repo.draft.all() if r.parent_skill_ids == [base.skill_id]
        ]
        assert len(drafts_for_base) == 1, (
            f"expected one DRAFT for base, got {len(drafts_for_base)}"
        )

    def test_coalesce_evicts_when_draft_leaves_DRAFT(self, tmp_path) -> None:
        """If the gate has already promoted the DRAFT (or archived it),
        the cached coalesce entry must be lazily evicted and the next
        failure mints a fresh patch instead of overwriting a stale
        proposal."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=0,
        )
        base = _seed_active_skill(lifecycle)

        first = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        first_patch = next(p for p in first.proposals if isinstance(p, PatchProposal))
        # Locate the draft that the patch materialized.
        draft = next(r for r in repo.draft.all()
                     if r.proposal_id == first_patch.proposal_id)

        # Simulate the gate / orchestrator promoting the patch out of DRAFT.
        lifecycle.transition(
            draft.skill_id,
            to_status=SkillStatus.CANDIDATE,
            rationale="gate-test-promotion",
        )

        second = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-1",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-1")],
        ))
        fresh_patches = [p for p in second.proposals if isinstance(p, PatchProposal)]
        assert len(fresh_patches) == 1, (
            "post-promotion failure must mint a NEW patch; stale cache must be evicted"
        )
        assert fresh_patches[0].proposal_id != first_patch.proposal_id
        assert second.n_patches_coalesced == 0

    def test_different_strategies_do_not_coalesce(self, tmp_path) -> None:
        """Coalesce key includes the recovery_strategy. A second
        failure that diagnoses to a *different* strategy on the same
        base must mint a fresh proposal, not append to the existing
        one."""
        from common.enums import RecoveryStrategy
        from data_structure.extensions.failure_trace import FailureDiagnosis

        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=0,
        )
        base = _seed_active_skill(lifecycle)

        # Stub the diagnoser so we can deterministically flip strategy
        # between calls without going through the rule pipeline.
        strategy_seq = iter([
            RecoveryStrategy.HOP_INSERTION,
            RecoveryStrategy.PRECONDITION_STRENGTHENING,
        ])

        def _stub_diagnose(trace):
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="stubbed",
                recommended_strategy=next(strategy_seq),
            )

        crafter._diagnoser.diagnose = _stub_diagnose                  # type: ignore[assignment]

        ep0 = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        ep1 = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-1",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-1", idx=0)],
        ))
        patches_0 = [p for p in ep0.proposals if isinstance(p, PatchProposal)]
        patches_1 = [p for p in ep1.proposals if isinstance(p, PatchProposal)]
        assert len(patches_0) == 1
        assert len(patches_1) == 1, (
            "different strategy ⇒ different coalesce key ⇒ fresh patch"
        )
        assert patches_0[0].recovery_strategy != patches_1[0].recovery_strategy
        assert ep1.n_patches_coalesced == 0

    def test_public_propose_repair_returns_none_on_coalesce(self, tmp_path) -> None:
        """Callers using the public API see ``None`` on coalesce; they
        can recover the running proposal through ``_open_patches`` if
        they need it."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=0,
        )
        base = _seed_active_skill(lifecycle)

        # Seed the failure memory and dispatch once via reflect.
        first = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        assert any(isinstance(p, PatchProposal) for p in first.proposals)

        # Now drive a second failure manually through the public API.
        # The pattern_id stays the same since failure_class+skill_id
        # are identical.
        crafter.ingest_failures([_make_failure(base.skill_id, "ep-1", idx=2)])
        # Recover the now-hot pattern (single-trace patterns are visible at min_count=1).
        hot = crafter._failures.hot_patterns(min_count=1)
        assert hot, "failure memory must surface the new pattern"
        out = crafter.propose_repair(base=base, pattern=hot[0])
        assert out is None, "second call with the same (base, strategy) must coalesce"


class TestCooldown:
    def test_cooldown_blocks_repeat_mint_within_window(self, tmp_path) -> None:
        """With ``cooldown_passes=3`` and the open DRAFT eagerly drained
        between calls (so coalesce is bypassed), a fresh patch on the
        same base in passes 2 and 3 must be skipped via cooldown,
        and the same call in pass 4+ becomes eligible again."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=3,
        )
        base = _seed_active_skill(lifecycle)

        # Pass 1 — fresh mint.
        r1 = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        patches_1 = [p for p in r1.proposals if isinstance(p, PatchProposal)]
        assert len(patches_1) == 1
        assert r1.n_patches_skipped_cooldown == 0

        # Drain the open DRAFT so coalesce can't fire — we want the
        # cooldown path, not the coalesce path.
        draft = next(r for r in repo.draft.all()
                     if r.proposal_id == patches_1[0].proposal_id)
        lifecycle.transition(
            draft.skill_id,
            to_status=SkillStatus.CANDIDATE,
            rationale="drain-for-cooldown-test",
        )

        # Passes 2 and 3 — within cooldown, mint blocked.
        for i in range(1, 3):
            r = crafter.reflect_on_episode(EpisodeReflection(
                episode_id=f"ep-{i}",
                domain="gymv",
                failure_traces=[_make_failure(base.skill_id, f"ep-{i}")],
            ))
            patches = [p for p in r.proposals if isinstance(p, PatchProposal)]
            assert patches == [], (
                f"pass {i+1}: expected cooldown-skip, got {len(patches)} fresh PatchProposal(s)"
            )
            assert r.n_patches_skipped_cooldown == 1, (
                f"pass {i+1}: expected n_patches_skipped_cooldown==1, "
                f"got {r.n_patches_skipped_cooldown}"
            )
            assert r.n_patches_coalesced == 0

        # Pass 4 — cooldown elapsed, mint eligible again.
        r4 = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-3",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-3")],
        ))
        patches_4 = [p for p in r4.proposals if isinstance(p, PatchProposal)]
        assert len(patches_4) == 1, (
            "pass 4: cooldown elapsed; new fresh patch should mint"
        )
        assert r4.n_patches_skipped_cooldown == 0

    def test_cooldown_disabled_with_zero_passes(self, tmp_path) -> None:
        """``cooldown_passes=0`` reverts to the pre-fix behaviour: every
        pass either mints or coalesces; nothing is ever cooldown-skipped."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=0,
        )
        base = _seed_active_skill(lifecycle)

        for i in range(5):
            r = crafter.reflect_on_episode(EpisodeReflection(
                episode_id=f"ep-{i}",
                domain="gymv",
                failure_traces=[_make_failure(base.skill_id, f"ep-{i}")],
            ))
            assert r.n_patches_skipped_cooldown == 0

    def test_cooldown_does_not_block_coalesce(self, tmp_path) -> None:
        """Coalesce always wins over cooldown: when an open DRAFT
        exists, the failure must be coalesced even within the cooldown
        window."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=10,        # huge cooldown
        )
        base = _seed_active_skill(lifecycle)

        crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        r2 = crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-1",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-1")],
        ))
        # Coalesce path must win; cooldown counter stays zero.
        assert r2.n_patches_coalesced == 1
        assert r2.n_patches_skipped_cooldown == 0
        assert not any(isinstance(p, PatchProposal) for p in r2.proposals)


class TestPassCounter:
    def test_no_signal_does_not_advance_counter(self, tmp_path) -> None:
        """Healthy episodes (no failures, no fresh candidates) must not
        bleed off cooldown — otherwise a few quiet episodes after a
        patch would silently re-open the mint window."""
        repo, lifecycle, artifacts, crafter = _build_service(
            tmp_path, cooldown_passes=2,
        )
        base = _seed_active_skill(lifecycle)

        crafter.reflect_on_episode(EpisodeReflection(
            episode_id="ep-0",
            domain="gymv",
            failure_traces=[_make_failure(base.skill_id, "ep-0")],
        ))
        before = crafter._pass_counter

        # 5 silent reflections shouldn't advance the counter.
        for i in range(1, 6):
            crafter.reflect_on_episode(EpisodeReflection(
                episode_id=f"ep-{i}",
                domain="gymv",
            ))
        assert crafter._pass_counter == before, (
            f"silent reflections must not bump pass_counter "
            f"(was {before}, became {crafter._pass_counter})"
        )
