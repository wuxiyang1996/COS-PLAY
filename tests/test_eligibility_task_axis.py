"""Regression test for the task-axis F2′ veto in `EligibilityFilter`.

Targets `harness/README.md` §22's empirical fix: a 2048-mined skill must
*not* be admitted for a tetris state (and vice versa) when the skill has
non-empty `feasible_tasks`. Skills with `feasible_tasks=[]` remain
admissible everywhere — that's the back-compat clause for cold-start
banks decorated before the task axis landed.

Companion of `labeling_supplement/_phase0_cross_eligibility_probe.py` —
the probe is the empirical sweep against the real cold-start corpus, this
test is the unit-level guard against accidental regression.
"""
from __future__ import annotations

import pytest

from common.enums import SkillSourceType, SkillStatus, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.adapters import GymvAdapter
from harness.eligibility import EligibilityFilter, task_id_from_state


# ───────────── helpers ─────────────


def _state(task: str) -> StateSchema:
    return StateSchema(task=task, domain="gymv")


def _skill(
    *,
    skill_id: str,
    feasible_tasks=None,
    verified_tasks=None,
    feasible_domains=("gymv",),
    skill_type=SkillType.ACTION,
    status=SkillStatus.PROVISIONAL,
) -> SkillRecord:
    sk = SkillRecord.new(
        name=skill_id,
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=list(feasible_domains),
        feasible_tasks=list(feasible_tasks) if feasible_tasks is not None else None,
        verified_tasks=list(verified_tasks) if verified_tasks is not None else None,
        protocol=[{"action": "EXEC", "payload": {}, "notes": "noop"}],
    )
    object.__setattr__(sk, "skill_id", skill_id)
    object.__setattr__(sk, "status", status)
    return sk


@pytest.fixture
def filt() -> EligibilityFilter:
    registry = AdapterRegistry()
    registry.register(GymvAdapter())
    return EligibilityFilter(registry)


# ───────────── task-id extraction ─────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("twenty_forty_eight", "twenty_forty_eight"),
        ("make_gaming_env/twenty_forty_eight", "twenty_forty_eight"),
        ("foo/bar/baz/qux", "qux"),
        ("", None),
        ("   ", None),
        ("/", None),  # trailing-empty segment
    ],
)
def test_task_id_extraction(raw: str, expected) -> None:
    state = StateSchema(task=raw, domain="gymv")
    assert task_id_from_state(state) == expected


# ───────────── F2′ same-task admission ─────────────


def test_same_task_admitted(filt: EligibilityFilter) -> None:
    sk = _skill(skill_id="A_2048", feasible_tasks=["twenty_forty_eight"])
    out = filt.filter([sk], _state("make_gaming_env/twenty_forty_eight"))
    assert len(out) == 1
    assert out[0].task_match == "same_task"


def test_verified_task_recorded(filt: EligibilityFilter) -> None:
    sk = _skill(
        skill_id="A_2048_v",
        feasible_tasks=["twenty_forty_eight"],
        verified_tasks=["twenty_forty_eight"],
    )
    out = filt.filter([sk], _state("twenty_forty_eight"))
    assert len(out) == 1
    assert out[0].task_match == "verified"


# ───────────── F2′ cross-task veto (the §22 regression case) ─────────────


def test_cross_task_vetoed(filt: EligibilityFilter) -> None:
    sk_2048 = _skill(skill_id="A_2048", feasible_tasks=["twenty_forty_eight"])
    sk_tetris = _skill(skill_id="B_tetris", feasible_tasks=["tetris"])

    # 2048 state: only the 2048 skill should be admitted.
    out_2048 = filt.filter([sk_2048, sk_tetris], _state("twenty_forty_eight"))
    assert {es.skill.skill_id for es in out_2048} == {"A_2048"}

    # Tetris state: only the tetris skill should be admitted.
    out_tetris = filt.filter([sk_2048, sk_tetris], _state("tetris"))
    assert {es.skill.skill_id for es in out_tetris} == {"B_tetris"}


# ───────────── back-compat: empty feasible_tasks is task-agnostic ─────────────


def test_empty_feasible_tasks_is_agnostic(filt: EligibilityFilter) -> None:
    """A skill with `feasible_tasks=[]` (i.e. pre-v2 decorator output, or a
    deliberately task-agnostic skill) must remain admissible regardless of
    the state's task. This is the load-bearing back-compat clause."""

    sk = _skill(skill_id="legacy", feasible_tasks=[])  # explicitly empty
    out_a = filt.filter([sk], _state("twenty_forty_eight"))
    out_b = filt.filter([sk], _state("tetris"))
    out_c = filt.filter([sk], _state(""))  # state task missing too

    assert len(out_a) == 1 and out_a[0].task_match == "agnostic"
    assert len(out_b) == 1 and out_b[0].task_match == "agnostic"
    assert len(out_c) == 1 and out_c[0].task_match == "agnostic"


def test_state_with_no_task_does_not_blind_veto(filt: EligibilityFilter) -> None:
    """When the *state* has no task tag, a skill with `feasible_tasks` must
    NOT be vetoed (degraded admit). Single-step adapters and synthesised
    states that don't carry a task identifier still need to dispatch."""

    sk = _skill(skill_id="A_2048", feasible_tasks=["twenty_forty_eight"])
    out = filt.filter([sk], _state(""))  # no task on the state side
    assert len(out) == 1
    assert out[0].task_match == "agnostic"


# ───────────── multi-task feasibility ─────────────


def test_multi_task_feasibility(filt: EligibilityFilter) -> None:
    """A skill that has been verified across two tasks (e.g. after a
    successful Stage-3a transfer cycle) should be admissible on either."""

    sk = _skill(
        skill_id="bridge",
        feasible_tasks=["twenty_forty_eight", "tetris"],
        verified_tasks=["twenty_forty_eight"],
    )
    out_a = filt.filter([sk], _state("twenty_forty_eight"))
    assert len(out_a) == 1 and out_a[0].task_match == "verified"
    out_b = filt.filter([sk], _state("tetris"))
    assert len(out_b) == 1 and out_b[0].task_match == "same_task"  # not yet verified
    out_c = filt.filter([sk], _state("candy_crush"))  # unrelated task
    assert out_c == []
