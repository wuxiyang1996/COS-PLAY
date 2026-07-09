"""Day-7d: tests for the deterministic stub executor's typed-hop
awareness.

Before Day-7d the deterministic stub emitted a single ``GATHER``
evidence ref per hop regardless of the hop's `action_type`. That
trivially satisfied the G0 evidence-driven invariant, but it masked
real role regressions: the action-level `ReplayValidator` walk
couldn't tell the difference between a `GATHER → VERIFY → COMMIT`
canonical seed and a stub re-run that emitted three `GATHER`s.

After Day-7d the stub maps action_type → evidence role
(`GATHER`/`VERIFY`/`REASON`/`COMMIT`) so the action-level walk has
something to compare against on transfer-target seeds. The stub also
emits a directional `evidence_in / evidence_out` split that matches
the Day-8b SkillEpisodeStep expansion.

Pins:

  * Each canonical action verb maps to its expected role;
  * Unknown verbs degrade to ``GATHER`` (G0 still satisfied);
  * The stub emits both legacy ``evidence`` and the directional
    ``evidence_in / evidence_out`` keys;
  * The hop loop forwards the directional split into the per-step
    record;
  * Run round-trips through SkillHarness.run_skill, populating
    `SkillEpisodeStep.evidence_in / evidence_out / protocol_index`.
"""
from __future__ import annotations

import os
import sys
from typing import List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.enums import SkillSourceType, SkillStatus, SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness import AdapterRegistry, HarnessConfig, SkillHarness
from harness.adapters._stub_base import (
    _ACTION_VERB_TO_ROLE,
    _role_for_action,
    make_deterministic_executor,
)
from harness.adapters.osworld_adapter import OsworldAdapter
from harness.skill_adapter import AdapterRunContext


def _state(domain: str = "osworld") -> StateSchema:
    return StateSchema(task=f"task/{domain}", domain=domain)


def _ctx(domain: str = "osworld") -> AdapterRunContext:
    return AdapterRunContext(
        state=_state(domain),
        bindings={},
        budget={"hops": 8, "ms": 30_000.0},
        seed=0,
    )


def test_role_for_action_canonical_mapping() -> None:
    assert _role_for_action("GATHER") == "GATHER"
    assert _role_for_action("OBSERVE") == "GATHER"
    assert _role_for_action("VERIFY") == "VERIFY"
    assert _role_for_action("CHECK") == "VERIFY"
    assert _role_for_action("REASON") == "REASON"
    assert _role_for_action("INFER") == "REASON"
    assert _role_for_action("COMMIT") == "COMMIT"
    assert _role_for_action("ANSWER") == "COMMIT"


def test_role_for_action_unknown_falls_back_to_gather() -> None:
    assert _role_for_action("FROBNICATE") == "GATHER"
    assert _role_for_action("") == "GATHER"


def test_role_for_action_prefix_match() -> None:
    # "VERIFY_TILE" should resolve to VERIFY via the prefix path.
    assert _role_for_action("VERIFY_TILE") == "VERIFY"
    assert _role_for_action("COMMIT_BUTTON") == "COMMIT"


def test_executor_emits_directional_split() -> None:
    exec_fn = make_deterministic_executor("osworld")
    ctx = _ctx("osworld")
    out = exec_fn("VERIFY", {"target": "tile"}, ctx)
    assert out["ok"] is True
    # Legacy uni-directional union still present.
    legacy = out["evidence"]
    assert isinstance(legacy, list) and len(legacy) == 1
    ev = legacy[0]
    assert isinstance(ev, EvidenceRef)
    assert ev.role == "VERIFY"
    # Directional split.
    assert "evidence_in" in out and isinstance(out["evidence_in"], list)
    assert "evidence_out" in out and isinstance(out["evidence_out"], list)
    assert out["evidence_out"][0].role == "VERIFY"


def test_executor_evidence_in_carries_prior_state_evidence() -> None:
    exec_fn = make_deterministic_executor("osworld")
    ctx = _ctx("osworld")
    seed_ev = EvidenceRef(source="seed", locator="step=0", role="GATHER", confidence=0.9)
    ctx.state.evidence = [seed_ev]
    out = exec_fn("REASON", {}, ctx)
    # evidence_in mirrors the state's prior evidence.
    assert len(out["evidence_in"]) == 1
    assert out["evidence_in"][0].source == "seed"


def test_run_skill_propagates_directional_split() -> None:
    """End-to-end: run a 3-hop skill on the osworld stub adapter and
    confirm the SkillEpisodeStep records pick up the directional
    split + protocol_index."""
    registry = AdapterRegistry()
    registry.register(OsworldAdapter())
    harness = SkillHarness(registry=registry, config=HarnessConfig(seed=0))

    skill = SkillRecord.new(
        name="osworld_walk",
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.SEEDED,
        feasible_domains=["osworld"],
        protocol=[
            {"action": "GATHER", "payload": {"x": 1}},
            {"action": "VERIFY", "payload": {"x": 1}},
            {"action": "COMMIT", "payload": {"x": 1}},
        ],
        contract=SkillContract(
            preconditions=[],
            expected_evidence_roles=["GATHER", "VERIFY", "COMMIT"],
            success_criteria=["committed"],
        ),
    )
    object.__setattr__(skill, "status", SkillStatus.PROVISIONAL)

    ep = harness.run_skill(
        skill, _state("osworld"), bindings={}, parent_run_id="run-day7d",
    )
    assert ep.outcome is not None
    # 3 protocol hops → 3 SkillEpisodeSteps.
    assert len(ep.steps) == 3
    expected_roles = ["GATHER", "VERIFY", "COMMIT"]
    for i, step in enumerate(ep.steps):
        assert step.protocol_index == i
        # Directional out role matches the hop type.
        assert step.evidence_out, f"step {i} missing evidence_out"
        assert step.evidence_out[0].role == expected_roles[i]
    # protocol_trace mirrors the protocol indices.
    assert ep.protocol_trace == [0, 1, 2]


def test_canonical_role_table_covers_all_four_roles() -> None:
    """Sanity: every spec'd evidence role appears in the verb table."""
    seen = set(_ACTION_VERB_TO_ROLE.values())
    assert seen >= {"GATHER", "VERIFY", "REASON", "COMMIT"}
