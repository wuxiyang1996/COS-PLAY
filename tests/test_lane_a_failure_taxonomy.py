"""Tests for T1.3c — lane-(a) FailureClass taxonomy.

Covers:
    1. ``RecoveryStrategy`` exposes the three new values
       (BANK_GAP / RETRIEVAL_MISLEAD / STALE_DESCRIPTION).
    2. ``LANE_A_RECOVERY_STRATEGIES`` set contains exactly those three.
    3. ``FailureDiagnoser._rule_diagnose`` emits the correct strategy
       for each lane-(a) failure_class, and aliases the synthesis
       signals (OUTCOME_FAILURE / NO_SKILL_BOUND / LOW_APPLICABILITY
       → BANK_GAP; MISSING_EFFECTS → STALE_DESCRIPTION).
    4. ``_run_failure_dispatch`` skips the Repairer for lane-(a)
       strategies and falls through to the Hypothesizer.
"""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import LANE_A_RECOVERY_STRATEGIES, RecoveryStrategy
from data_structure.extensions.failure_trace import FailureTrace


def _trace(failure_class: str, *, extra=None) -> FailureTrace:
    return FailureTrace(
        failure_id=f"f-{failure_class}",
        skill_id="sk-test",
        skill_episode_id="ep-1",
        domain="gymv",
        failure_class=failure_class,
        extra=dict(extra or {}),
    )


# --------------------------------------------------------------------- enum surface


def test_recovery_strategy_has_lane_a_values() -> None:
    assert RecoveryStrategy.BANK_GAP.value == "bank_gap"
    assert RecoveryStrategy.RETRIEVAL_MISLEAD.value == "retrieval_mislead"
    assert RecoveryStrategy.STALE_DESCRIPTION.value == "stale_description"


def test_lane_a_recovery_strategies_set() -> None:
    assert LANE_A_RECOVERY_STRATEGIES == frozenset({
        RecoveryStrategy.BANK_GAP,
        RecoveryStrategy.RETRIEVAL_MISLEAD,
        RecoveryStrategy.STALE_DESCRIPTION,
    })
    # Protocol-edit strategies are NOT lane-(a).
    assert RecoveryStrategy.PROTOCOL_PATCH not in LANE_A_RECOVERY_STRATEGIES
    assert RecoveryStrategy.HOP_INSERTION not in LANE_A_RECOVERY_STRATEGIES
    assert RecoveryStrategy.SKILL_RETIREMENT not in LANE_A_RECOVERY_STRATEGIES


# --------------------------------------------------------------------- diagnoser


def test_diagnoser_emits_bank_gap_for_canonical_class() -> None:
    from crafter.failure_diagnoser import FailureDiagnoser

    d = FailureDiagnoser().diagnose(_trace("BANK_GAP"))
    assert d.recommended_strategy == RecoveryStrategy.BANK_GAP


def test_diagnoser_emits_retrieval_mislead() -> None:
    from crafter.failure_diagnoser import FailureDiagnoser

    d = FailureDiagnoser().diagnose(_trace("RETRIEVAL_MISLEAD"))
    assert d.recommended_strategy == RecoveryStrategy.RETRIEVAL_MISLEAD


def test_diagnoser_emits_stale_description() -> None:
    from crafter.failure_diagnoser import FailureDiagnoser

    d = FailureDiagnoser().diagnose(_trace("STALE_DESCRIPTION"))
    assert d.recommended_strategy == RecoveryStrategy.STALE_DESCRIPTION


@pytest.mark.parametrize(
    "signal",
    ["OUTCOME_FAILURE", "NO_SKILL_BOUND", "LOW_APPLICABILITY"],
)
def test_diagnoser_aliases_synthesis_signals_to_bank_gap(signal: str) -> None:
    """When ``failure_class`` itself is unmappable, the diagnoser
    consults ``trace.extra['synthesis_signal']`` and aliases the lane-(a)
    signals to ``BANK_GAP`` (per ``configs/failure_routing.yaml``).
    """
    from crafter.failure_diagnoser import FailureDiagnoser

    d = FailureDiagnoser().diagnose(
        _trace("OTHER", extra={"synthesis_signal": signal})
    )
    assert d.recommended_strategy == RecoveryStrategy.BANK_GAP


def test_diagnoser_aliases_missing_effects_to_stale_description() -> None:
    from crafter.failure_diagnoser import FailureDiagnoser

    d = FailureDiagnoser().diagnose(
        _trace("OTHER", extra={"synthesis_signal": "MISSING_EFFECTS"})
    )
    assert d.recommended_strategy == RecoveryStrategy.STALE_DESCRIPTION


def test_diagnoser_falls_through_unknown_signal_to_protocol_patch() -> None:
    """Backward-compat: unrecognized synthesis signals do NOT crash;
    they fall through to the legacy ``PROTOCOL_PATCH`` default.
    """
    from crafter.failure_diagnoser import FailureDiagnoser

    d = FailureDiagnoser().diagnose(
        _trace("WHATEVER", extra={"synthesis_signal": "XYZ"})
    )
    assert d.recommended_strategy == RecoveryStrategy.PROTOCOL_PATCH


# --------------------------------------------------------------------- dispatch


def test_dispatcher_lane_a_strategies_route_to_hypothesizer() -> None:
    """T1.3c: when the diagnoser returns a lane-(a) strategy and the
    pattern's skill_id resolves to an existing bank record, the
    dispatcher must NOT call the Repairer; it falls through to the
    Hypothesizer.
    """
    from unittest.mock import MagicMock

    from crafter.service import SkillCrafterService
    from data_structure.extensions.bank_mutation_proposal import HypothesisProposal
    from data_structure.extensions.failure_trace import FailureDiagnosis
    from data_structure.extensions.skill_record import SkillContract, SkillRecord
    from common.enums import SkillSourceType, SkillStatus, SkillType

    base_skill = SkillRecord(
        skill_id="sk-base",
        name="x",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.SEEDED,
        status=SkillStatus.CANDIDATE,
        feasible_domains=["gymv", "browser"],
        protocol=[{"action": "PRESS", "payload": {"key": "x"}}],
        contract=SkillContract(expected_evidence_roles=["GATHER"], success_criteria=["x"]),
    )

    # Build a SkillCrafterService skeleton with the few attributes
    # ``_run_failure_dispatch`` actually consults (no live LLM, no
    # repository, no real artifact store).
    svc = SkillCrafterService.__new__(SkillCrafterService)
    svc._artifacts = MagicMock()
    svc._pass_counter = 1
    svc._patch_last_pass = {}
    svc._cooldown_passes = 0
    svc._open_patches = {}
    svc._teacher = None
    svc._enable_protocol_patching = True
    svc._failures = MagicMock()
    # Post-v11 hypothesizer-fallthrough gates: this test exercises
    # the dispatch routing in isolation (no recurrence assertion, no
    # bank lookup), so disable both gates by setting min_recurrences=1
    # and jaccard=0.0. Production callers go through ``__init__``,
    # which keeps the gate defaults (3 / 0.30).
    svc._hypothesize_min_recurrences = 1
    svc._hypothesize_related_jaccard = 0.0

    # Stub _resolve_base to return our skill, bypass the repository.
    svc._resolve_base = MagicMock(return_value=base_skill)

    # Diagnoser returns BANK_GAP — the lane-(a) signal.
    svc._diagnose_pattern = MagicMock(
        return_value=FailureDiagnosis(
            failure_id="f-1",
            locus="retrieval",
            root_cause="bank gap",
            recommended_strategy=RecoveryStrategy.BANK_GAP,
            confidence=0.7,
        )
    )

    # Hypothesizer returns a fake proposal.
    fake_hypothesis = HypothesisProposal(
        name="new-skill",
        novel_protocol=[{"action": "EXECUTE", "payload": {}}],
    )
    svc._hypothesizer = MagicMock()
    svc._hypothesizer.propose = MagicMock(return_value=fake_hypothesis)

    # Repairer must NOT be called for a lane-(a) strategy.
    svc._repairer = MagicMock()
    svc._repairer.repair = MagicMock(side_effect=AssertionError(
        "Repairer must not be called for lane-(a) strategies"
    ))

    # _persist is a no-op for the test.
    svc._persist = MagicMock(return_value=None)

    pattern = MagicMock()
    pattern.skill_id = "sk-base"
    pattern.pattern_id = "p-1"
    pattern.failure_ids = ["f-1"]
    pattern.count = 1  # post-v11 hypothesizer-fallthrough gate reads .count

    proposals, n_coalesced, n_cooldown = svc._run_failure_dispatch([pattern])

    # The hypothesizer fired; the repairer did not.
    assert len(proposals) == 1
    assert proposals[0] is fake_hypothesis
    svc._hypothesizer.propose.assert_called_once()
    svc._repairer.repair.assert_not_called()
    assert n_coalesced == 0
    assert n_cooldown == 0


def test_dispatcher_protocol_edit_strategies_still_call_repairer() -> None:
    """Sanity: lane-(b) strategies (e.g. PRECONDITION_STRENGTHENING)
    keep the original Repairer route. Only the three new lane-(a)
    values are short-circuited.
    """
    from unittest.mock import MagicMock

    from crafter.service import SkillCrafterService
    from data_structure.extensions.failure_trace import FailureDiagnosis
    from data_structure.extensions.skill_record import SkillContract, SkillRecord
    from common.enums import SkillSourceType, SkillStatus, SkillType

    base_skill = SkillRecord(
        skill_id="sk-base",
        name="x",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.SEEDED,
        status=SkillStatus.CANDIDATE,
        feasible_domains=["gymv", "browser"],
        protocol=[{"action": "PRESS", "payload": {"key": "x"}}],
        contract=SkillContract(expected_evidence_roles=["GATHER"], success_criteria=["x"]),
    )

    svc = SkillCrafterService.__new__(SkillCrafterService)
    svc._artifacts = MagicMock()
    svc._pass_counter = 1
    svc._patch_last_pass = {}
    svc._cooldown_passes = 0
    svc._open_patches = {}
    svc._teacher = None
    svc._enable_protocol_patching = True
    svc._failures = MagicMock()
    svc._resolve_base = MagicMock(return_value=base_skill)
    svc._diagnose_pattern = MagicMock(
        return_value=FailureDiagnosis(
            failure_id="f-1",
            locus="precondition",
            root_cause="x",
            recommended_strategy=RecoveryStrategy.PRECONDITION_STRENGTHENING,
            confidence=0.7,
        )
    )
    svc._hypothesizer = MagicMock()
    # NB: the dispatcher takes the early "if base is not None" path and
    # short-circuits with _STATUS_MINTED on a PatchProposal; the
    # hypothesizer should NOT fire when the Repairer succeeds.

    # Stub _propose_repair_internal to mint a fake PatchProposal so we
    # avoid threading through the Repairer's full plumbing.
    from data_structure.extensions.bank_mutation_proposal import PatchProposal
    fake_patch = PatchProposal(
        base_skill_id="sk-base",
        recovery_strategy=RecoveryStrategy.PRECONDITION_STRENGTHENING.value,
    )
    svc._propose_repair_internal = MagicMock(
        return_value=(fake_patch, SkillCrafterService._STATUS_MINTED)
    )
    svc._lookup_open_patch = MagicMock(return_value=None)
    svc._is_under_cooldown = MagicMock(return_value=False)

    pattern = MagicMock()
    pattern.skill_id = "sk-base"
    pattern.pattern_id = "p-1"
    pattern.failure_ids = ["f-1"]
    pattern.count = 1  # post-v11 hypothesizer-fallthrough gate reads .count

    proposals, _, _ = svc._run_failure_dispatch([pattern])

    assert len(proposals) == 1
    assert proposals[0] is fake_patch
    svc._propose_repair_internal.assert_called_once()
    svc._hypothesizer.propose.assert_not_called()
