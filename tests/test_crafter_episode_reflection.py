"""Per-episode reactive Crafter pass — coverage for the two-tier trigger.

Spec: implementation_notes/legacy/crafter-harness-orchestrator-roles.md
§"Two-tier trigger" + crafter/service.py::reflect_on_episode docstring.

These tests cover four properties:

  1. ``reflect_on_episode`` fires on a single failure (threshold=1),
     where ``cycle()`` with default ``hot_pattern_threshold=3`` would
     not.
  2. A reflection that contains neither failures nor freshly-minted
     candidate ids is a no-op (``trigger=reflect_on_episode``,
     zero proposals).
  3. Subsumption-retire fires when a candidate's ``parent_skill_ids``
     points at an active and the candidate's contract is a strict
     superset (effects + roles + success criteria).
  4. Stale failures from earlier episodes do not re-fire the per-episode
     pass (the per-episode pass restricts to ``failure_id``s present
     in *this* reflection).

The tests share the same ``_new_repo`` / ``_seed_active_skill`` helpers
as ``test_crafter_repair.py`` so changes to the bank fixture stay
co-located.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.enums import (
    RecoveryStrategy,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from crafter import BankView, SkillCrafterService
from data_structure.extensions.bank_mutation_proposal import (
    PatchProposal,
    RetireProposal,
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


def _seed_candidate_subsuming(
    lifecycle: SkillLifecycleManager,
    *,
    parent: SkillRecord,
) -> SkillRecord:
    """Create a CANDIDATE skill whose contract strictly covers ``parent``'s.

    Linked via ``parent_skill_ids`` so the BankView subsumption heuristic
    fires (mimicking what the per-episode bank-mgmt agent's `refine`
    step would emit).
    """
    cand = SkillRecord.new(
        name=f"{parent.name}__refined",
        skill_type=parent.skill_type,
        source_type=SkillSourceType.CRAFTED,
        feasible_domains=list(parent.feasible_domains),
        protocol=list(parent.protocol),
        contract=SkillContract(
            preconditions=list(parent.contract.preconditions) + ["fresh_state"],
            effects_add=list(parent.contract.effects_add) + ["score_delta"],
            effects_del=list(parent.contract.effects_del),
            expected_evidence_roles=(
                list(parent.contract.expected_evidence_roles) + ["GATHER"]
            ),
            success_criteria=list(parent.contract.success_criteria),
        ),
        parent_skill_ids=[parent.skill_id],
    )
    lifecycle.ingest_draft(cand)
    lifecycle.transition(
        cand.skill_id, to_status=SkillStatus.CANDIDATE, rationale="bank-agent refine"
    )
    return cand


# --------------------------------------------------------------------- tests


class TestReflectOnEpisode:
    def test_fires_on_single_failure_below_batch_threshold(self, tmp_path) -> None:
        """One failure ⇒ reflect_on_episode emits a proposal,
        cycle() with default threshold=3 does NOT."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            hot_pattern_threshold=3,           # batch threshold left at default
            # Lane-(b) opt-in: this test asserts a PatchProposal is
            # emitted, which only happens with the Repairer path live.
            # Live trainer default is False (T1.3a).
            enable_protocol_patching=True,
        )
        base = _seed_active_skill(lifecycle)

        single_failure = FailureTrace(
            skill_id=base.skill_id,
            skill_episode_id="ep-0",
            domain="gymv",
            failed_step_index=1,
            failure_class="INVARIANT_VIOLATION",
            abort_reason="invariant: empty evidence",
        )

        # Per-episode pass fires (threshold=1).
        result = crafter.reflect_on_episode(
            EpisodeReflection(
                episode_id="ep-0",
                domain="gymv",
                failure_traces=[single_failure],
            )
        )
        assert result.trigger == "reflect_on_episode"
        assert result.episode_id == "ep-0"
        assert result.n_failures_ingested == 1
        assert result.n_patterns_examined == 1
        assert any(isinstance(p, PatchProposal) for p in result.proposals)
        assert result.bank_view_summary["n_active"] == 1

        # Per-batch pass with the SAME single failure does not fire (default threshold=3).
        # We use a fresh service so failure memory state doesn't leak.
        repo2 = _new_repo(str(tmp_path / "bank2"))
        lifecycle2 = SkillLifecycleManager(repo2)
        artifacts2 = ArtifactStore(str(tmp_path / "art2"))
        crafter2 = SkillCrafterService(
            lifecycle=lifecycle2, artifact_store=artifacts2, hot_pattern_threshold=3
        )
        _seed_active_skill(lifecycle2)
        batch_result = crafter2.cycle(new_failures=[single_failure])
        assert batch_result.trigger == "cycle"
        assert batch_result.n_patterns_examined == 0
        assert batch_result.proposals == []

    def test_no_signal_is_a_no_op(self, tmp_path) -> None:
        """A reflection with no failures and no fresh candidates
        produces zero proposals but still reports the bank view size."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)
        _seed_active_skill(lifecycle)

        result = crafter.reflect_on_episode(
            EpisodeReflection(episode_id="ep-healthy", domain="gymv")
        )
        assert result.trigger == "reflect_on_episode"
        assert result.proposals == []
        assert result.n_failures_ingested == 0
        assert result.n_patterns_examined == 0
        assert result.n_subsumption_retires == 0
        assert result.bank_view_summary["n_active"] == 1

    def test_subsumption_retire_path(self, tmp_path) -> None:
        """A freshly-minted candidate that strictly covers an active
        skill yields a RetireProposal (subsumed_by=...)."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)

        active = _seed_active_skill(lifecycle)
        candidate = _seed_candidate_subsuming(lifecycle, parent=active)

        result = crafter.reflect_on_episode(
            EpisodeReflection(
                episode_id="ep-refine",
                domain="gymv",
                new_candidate_skill_ids=[candidate.skill_id],
            )
        )
        assert result.n_subsumption_retires == 1
        retires = [p for p in result.proposals if isinstance(p, RetireProposal)]
        assert len(retires) == 1
        r = retires[0]
        assert r.target_skill_id == active.skill_id
        assert candidate.skill_id in r.reason
        assert "subsumed_by" in r.reason
        # And the active store is untouched (crafter scope invariant 6).
        assert len(repo.active.all()) == 1

    def test_subsumption_skipped_when_contract_not_strictly_stronger(
        self, tmp_path
    ) -> None:
        """A candidate that omits one of the active's effects must NOT
        produce a RetireProposal — false-positive avoidance."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)

        active = _seed_active_skill(lifecycle)
        # Build a candidate that drops `committed` from effects_add
        # (active had it; this candidate doesn't) ⇒ NOT subsumes.
        weak_cand = SkillRecord.new(
            name="press_then_check__weakened",
            skill_type=active.skill_type,
            source_type=SkillSourceType.CRAFTED,
            feasible_domains=list(active.feasible_domains),
            protocol=list(active.protocol),
            contract=SkillContract(
                preconditions=list(active.contract.preconditions),
                effects_add=[],                        # weaker than active
                expected_evidence_roles=(
                    list(active.contract.expected_evidence_roles)
                ),
                success_criteria=list(active.contract.success_criteria),
            ),
            parent_skill_ids=[active.skill_id],
        )
        lifecycle.ingest_draft(weak_cand)
        lifecycle.transition(
            weak_cand.skill_id, to_status=SkillStatus.CANDIDATE, rationale="weak refine"
        )

        result = crafter.reflect_on_episode(
            EpisodeReflection(
                episode_id="ep-weak-refine",
                domain="gymv",
                new_candidate_skill_ids=[weak_cand.skill_id],
            )
        )
        assert result.n_subsumption_retires == 0
        assert not any(isinstance(p, RetireProposal) for p in result.proposals)

    def test_stale_failures_do_not_re_fire(self, tmp_path) -> None:
        """Reflecting twice in a row on disjoint episodes must not
        re-fire the first episode's pattern from FailureMemory."""
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        # Lane-(b) opt-in — assertion below requires the Repairer path.
        crafter = SkillCrafterService(
            lifecycle=lifecycle, artifact_store=artifacts,
            enable_protocol_patching=True,
        )
        base = _seed_active_skill(lifecycle)

        first_failure = FailureTrace(
            skill_id=base.skill_id,
            skill_episode_id="ep-0",
            domain="gymv",
            failed_step_index=1,
            failure_class="INVARIANT_VIOLATION",
        )
        first = crafter.reflect_on_episode(
            EpisodeReflection(
                episode_id="ep-0",
                domain="gymv",
                failure_traces=[first_failure],
            )
        )
        assert first.proposals, "first episode must produce a proposal"

        # Second reflection: NEW episode with a totally different
        # (unknown) failure. The first episode's pattern is still in
        # FailureMemory but must not re-emit because its failure_id is
        # not in this reflection.
        second_failure = FailureTrace(
            skill_id="s-unrelated",
            skill_episode_id="ep-1",
            domain="gymv",
            failed_step_index=0,
            failure_class="MISSING_ADAPTER",
        )
        second = crafter.reflect_on_episode(
            EpisodeReflection(
                episode_id="ep-1",
                domain="gymv",
                failure_traces=[second_failure],
            )
        )
        # The second result's patterns must reference ep-1's failure, not ep-0's.
        for p in second.proposals:
            assert base.skill_id != getattr(p, "base_skill_id", None) or (
                second_failure.failure_id in p.seed_failure_ids
            ), "second episode pass must not re-fire the first episode's pattern"


class TestBankView:
    def test_view_is_read_only_snapshot(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)
        active = _seed_active_skill(lifecycle)

        view: BankView = crafter._take_bank_view()  # noqa: SLF001
        assert active.skill_id in view.actives
        assert view.status_of(active.skill_id) == "active"
        assert view.status_of("not-in-bank") is None
        assert view.size_summary() == {"n_active": 1, "n_candidate": 0, "n_draft": 0}

        # frozen=True dataclass — assignment must raise.
        with pytest.raises(Exception):
            view.actives = {}                # type: ignore[misc]

    def test_subsumed_pairs_returns_empty_when_no_parent_link(self, tmp_path) -> None:
        repo = _new_repo(str(tmp_path / "bank"))
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(tmp_path / "art"))
        crafter = SkillCrafterService(lifecycle=lifecycle, artifact_store=artifacts)

        active = _seed_active_skill(lifecycle)
        # Candidate WITHOUT parent_skill_ids — even if its contract
        # strictly subsumes the active, the heuristic must not fire
        # (we require explicit lineage to avoid false positives).
        orphan = SkillRecord.new(
            name="orphan_strong",
            skill_type=active.skill_type,
            source_type=SkillSourceType.CRAFTED,
            feasible_domains=list(active.feasible_domains),
            protocol=list(active.protocol),
            contract=SkillContract(
                preconditions=list(active.contract.preconditions),
                effects_add=list(active.contract.effects_add) + ["extra"],
                expected_evidence_roles=(
                    list(active.contract.expected_evidence_roles)
                ),
                success_criteria=list(active.contract.success_criteria),
            ),
            parent_skill_ids=[],
        )
        lifecycle.ingest_draft(orphan)
        lifecycle.transition(
            orphan.skill_id,
            to_status=SkillStatus.CANDIDATE,
            rationale="orphan candidate",
        )

        view: BankView = crafter._take_bank_view()  # noqa: SLF001
        pairs = view.subsumed_pairs(candidate_ids=[orphan.skill_id])
        assert pairs == []
