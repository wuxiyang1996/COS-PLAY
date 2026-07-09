"""Unit tests for ``RewriteProposal`` + ``MergeProposal`` alias (T1.3b).

Covers:
    1. ``RewriteProposal`` instantiation and JSON round-trip.
    2. ``MergeProposal`` is an alias for ``ComposeProposal`` (same class,
       same JSON serialization).
    3. ``BankMutationProposal`` Union accepts both new symbols.
    4. ``GateService._run_static`` accepts ``RewriteProposal`` without a
       ``source_type`` mismatch failure (the rewrite preserves provenance).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import GateVerdict, SkillSourceType, SkillStatus, SkillType
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    MergeProposal,
    RewriteProposal,
    proposal_to_json,
)
from data_structure.extensions.skill_record import SkillContract, SkillRecord


def _seeded_skill() -> SkillRecord:
    return SkillRecord(
        skill_id="sk-test-1",
        name="press_then_check",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.SEEDED,
        status=SkillStatus.CANDIDATE,
        feasible_domains=["gymv"],
        protocol=[{"action": "PRESS", "payload": {"key": "${target}"}}],
        contract=SkillContract(
            preconditions=["have_target"],
            expected_evidence_roles=["GATHER", "COMMIT"],
            success_criteria=["committed"],
        ),
    )


# --------------------------------------------------------------------- merge alias


def test_merge_proposal_is_compose_alias() -> None:
    """``MergeProposal`` aliases ``ComposeProposal``: same class object,
    same wire format, same isinstance() relations.
    """
    assert MergeProposal is ComposeProposal
    p = MergeProposal(
        name="m",
        component_skill_ids=["a", "b"],
        composed_protocol=[{"action": "EXECUTE", "payload": {}}],
    )
    assert isinstance(p, ComposeProposal)
    assert isinstance(p, MergeProposal)
    out = proposal_to_json(p)
    # JSON wire format MUST keep ``"ComposeProposal"`` so persisted
    # artifacts and offline driver readers continue to work.
    assert out["type"] == "ComposeProposal"


# --------------------------------------------------------------------- rewrite


def test_rewrite_proposal_default_source_type_is_crafted() -> None:
    p = RewriteProposal(base_skill_id="sk-1")
    assert p.source_type == SkillSourceType.CRAFTED


def test_rewrite_proposal_json_roundtrip_preserves_unset_fields_as_none() -> None:
    """Unset rewrite fields serialize as ``None`` so readers can tell
    "leave existing value untouched" from "explicit empty value".
    """
    p = RewriteProposal(
        base_skill_id="sk-1",
        rewritten_description="A clearer description.",
        rewritten_tags=["browser", "click"],
    )
    out = proposal_to_json(p)
    assert out["type"] == "RewriteProposal"
    assert out["base_skill_id"] == "sk-1"
    assert out["rewritten_description"] == "A clearer description."
    assert out["rewritten_tags"] == ["browser", "click"]
    # Unset rewrite fields: None == "leave alone".
    assert out["rewritten_name"] is None
    assert out["rewritten_retrieval_text"] is None
    assert out["rewritten_slot_guidance"] is None
    assert out["rewritten_notes"] is None


def test_rewrite_proposal_explicit_empty_distinguishable_from_unset() -> None:
    """Explicit empty containers serialize differently from unset
    fields: empty dict / empty list survive the round-trip and are
    distinguishable from ``None``.
    """
    p = RewriteProposal(
        base_skill_id="sk-1",
        rewritten_tags=[],                  # explicit clear
        rewritten_slot_guidance={},         # explicit clear
    )
    out = proposal_to_json(p)
    assert out["rewritten_tags"] == []
    assert out["rewritten_slot_guidance"] == {}
    # The ones we did NOT touch remain None.
    assert out["rewritten_description"] is None


def test_rewrite_proposal_in_bank_mutation_union() -> None:
    """``BankMutationProposal`` Union must accept ``RewriteProposal``
    (typing assertion only — typing is structural at runtime, but we
    can at least confirm the symbol is exported).
    """
    p: BankMutationProposal = RewriteProposal(base_skill_id="sk-1")
    assert isinstance(p, RewriteProposal)


# --------------------------------------------------------------------- gate static


def test_gate_static_accepts_rewrite_proposal() -> None:
    """T1.3b: ``GateService._run_static`` must NOT report a
    ``source_type mismatch`` failure when the proposal is a
    ``RewriteProposal`` against a skill with a different source_type
    (e.g. SEEDED). Rewrites preserve the underlying provenance.
    """
    from orchestrator.gate_service import GateService
    from orchestrator.config import GateThresholds

    skill = _seeded_skill()
    # SEEDED skill, CRAFTED rewrite — would otherwise fail source_type
    # mismatch under the pre-T1.3b check.
    proposal = RewriteProposal(
        base_skill_id=skill.skill_id,
        rewritten_description="Clearer description.",
    )
    gate = GateService.__new__(GateService)
    gate._thresholds = GateThresholds()
    verdict = gate._run_static(skill, proposal)
    # No source_type mismatch in failures.
    assert not any("source_type mismatch" in f for f in verdict.failures), (
        f"unexpected failures: {verdict.failures}"
    )
    assert verdict.verdict == GateVerdict.PASS, verdict.failures


def test_gate_static_rewrite_requires_base_skill_id() -> None:
    """Lineage check fires for RewriteProposal without ``base_skill_id``."""
    from orchestrator.gate_service import GateService
    from orchestrator.config import GateThresholds

    skill = _seeded_skill()
    proposal = RewriteProposal(base_skill_id="")
    gate = GateService.__new__(GateService)
    gate._thresholds = GateThresholds()
    verdict = gate._run_static(skill, proposal)
    assert verdict.verdict == GateVerdict.FAIL
    assert any("base_skill_id is empty" in f for f in verdict.failures), (
        f"missing base_skill_id check: {verdict.failures}"
    )
