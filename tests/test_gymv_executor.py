"""End-to-end test for the Day-3 gymv executor wiring.

PLAN-HARNESS §22 / harness/README §22. These tests exercise the path from
a (lifted) `SkillRecord` through `GymvAdapter.run` with a real
`HopExecutor` bound to a tiny fake env. They do *not* require GamingAgent
to be installed — the fake `_FakeTwoFortyEightEnv` mimics the surface
contract `env_wrappers.gym_like.make_gaming_env` exposes (`step` returns
a 5-tuple with a dict observation that carries a canonical `<state>`
block, `action_names` lists the env's vocabulary).

What we cover:

* `make_gymv_executor` resolves SLIDE.up → "up" via `ACTION_ALIAS_MAP`.
* `GymvAdapter.run` records pre/post StateSchema for every hop.
* `_evaluate_effects` rolls up `effects_add` predicates against the
  recorded snapshots and surfaces the result as
  `AdapterRunResult.extra["per_hop_effects"]`.
* The roll-up correctly identifies `cumulative_reward_increased` and
  `entity_value_increased` for `highest_tile`.
* Observational ops (`INSPECT`) are NOT translated into env steps.
* Unresolvable ops (`SLIDE.diagonal`) abort with `no_env_action_for_op`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pytest

from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness import (
    AdapterRegistry,
    HarnessConfig,
    SkillHarness,
)
from harness.adapters import GymvAdapter
from harness.gymv_executor import (
    ACTION_ALIAS_MAP,
    GymvExecutorState,
    OBSERVATIONAL_OPS,
    _resolve_action,
    initial_state_from_env,
    make_gymv_executor,
)


# ───────────── A tiny fake 2048-shape env ─────────────

_2048_INITIAL = """<state>
domain=gymv
task=make_gaming_env/twenty_forty_eight
goal=Play 2048
step=0

<entities>
e1[type=region, label=board, ontology=container_entity]
e2[type=object, label=tile_2, ontology=selectable_entity]
e3[type=object, label=tile_2, ontology=selectable_entity]
e5[type=text, label=highest_tile, ontology=goal_indicator]
e6[type=text, label=score, ontology=goal_indicator]

<attributes>
e1.state=visible
e2.state=visible
e2.value=2
e3.state=visible
e3.value=2
e5.value=2
e6.value=0

<state_flags>
phase=play
progress=0.05
</state>"""

_2048_AFTER_UP = """<state>
domain=gymv
task=make_gaming_env/twenty_forty_eight
goal=Play 2048
step=1

<entities>
e1[type=region, label=board, ontology=container_entity]
e2[type=object, label=tile_4, ontology=selectable_entity]
e5[type=text, label=highest_tile, ontology=goal_indicator]
e6[type=text, label=score, ontology=goal_indicator]

<attributes>
e1.state=visible
e2.state=visible
e2.value=4
e5.value=4
e6.value=4

<state_flags>
phase=play
progress=0.1
</state>"""


@dataclass
class _FakeStep:
    obs: Dict[str, Any]
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class _FakeTwoFortyEightEnv:
    """Mimics `env_wrappers.gym_like._GymLikeWrapper.step`'s contract."""

    action_names: List[str] = ["up", "down", "left", "right"]

    def __init__(self) -> None:
        self._steps_taken = 0
        self._last_obs = {"text": _2048_INITIAL, "schema_canonical": _2048_INITIAL}

    def reset(self, *, seed: Any = None, options: Any = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._steps_taken = 0
        self._last_obs = {"text": _2048_INITIAL, "schema_canonical": _2048_INITIAL}
        return self._last_obs, {"action_names": list(self.action_names)}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._steps_taken += 1
        if action == "up":
            obs = {"text": _2048_AFTER_UP, "schema_canonical": _2048_AFTER_UP}
            self._last_obs = obs
            return obs, 4.0, False, False, {"action_names": list(self.action_names)}
        # other moves: no-op, no reward
        return (
            self._last_obs,
            0.0,
            False,
            False,
            {"action_names": list(self.action_names)},
        )


# ───────────── Helpers ─────────────


def _slide_skill_with_effects() -> SkillRecord:
    """A SLIDE skill whose first env-mutating hop carries a predicate
    bundle that the parser+success_fn can decide on the post-state."""

    return SkillRecord.new(
        name="COMMIT__SLIDE_UP",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
        protocol=[
            {
                "op": "SLIDE",
                "payload": {"direction": "up"},
                "slot_types": {"direction": "enum"},
                "preconditions": [],
                "effects_add": [
                    {
                        "type": "cumulative_reward_increased",
                        "args": {},
                        "from_phrase": "score increases",
                    },
                    {
                        "type": "entity_value_increased",
                        "args": {"entity_label": "highest_tile"},
                        "from_phrase": "highest tile grows",
                    },
                ],
                "effects_del": [],
                "evidence_role": "COMMIT",
                "notes": "slide upward",
                "lift_mode": "first",
            }
        ],
    )


def _registry_with_executor(
    env: Any, *, on_unresolved: str = "skip",
) -> Tuple[AdapterRegistry, GymvExecutorState, GymvAdapter]:
    adapter = GymvAdapter()
    executor, holder = make_gymv_executor(
        env, domain="gymv", task="twenty_forty_eight",
        on_unresolved=on_unresolved,
    )
    adapter.set_executor(executor)
    registry = AdapterRegistry()
    registry.register(adapter)
    return registry, holder, adapter


# ───────────── Action-resolution unit tests ─────────────


def test_resolve_action_up_maps_to_up() -> None:
    """SLIDE.up under `ACTION_ALIAS_MAP` should resolve to "up" against
    a 4-direction action vocabulary (the 2048 / Tetris shape)."""

    out = _resolve_action(
        "SLIDE",
        {"direction": "up"},
        action_names=["up", "down", "left", "right"],
        alias_map=ACTION_ALIAS_MAP,
    )
    assert out == "up"


def test_resolve_action_falls_back_to_op_token() -> None:
    """EXECUTE has no payload-keyed alias but should still pick a
    sensible token when one is in the action vocabulary."""

    out = _resolve_action(
        "EXECUTE",
        {},
        action_names=["merge", "noop"],
        alias_map=ACTION_ALIAS_MAP,
    )
    assert out == "merge"


def test_resolve_action_returns_none_when_unmappable() -> None:
    out = _resolve_action(
        "SLIDE",
        {"direction": "diagonal"},  # not in alias map
        action_names=["up", "down", "left", "right"],
        alias_map=ACTION_ALIAS_MAP,
    )
    assert out is None


def test_resolve_action_payload_value_rescue() -> None:
    """The payload-value rescue clause: when a slot we didn't statically
    alias carries a token that *is* directly a known env action, we
    should still use it. Lifted COMMIT/MERGE in 2048 has
    `SELECT.target=${target}` — when the actor binds `${target}="up"`,
    the executor must dispatch "up" to the env."""

    out = _resolve_action(
        "SELECT",
        {"target": "up"},  # SELECT has no payload alias for `target`
        action_names=["up", "down", "left", "right"],
        alias_map=ACTION_ALIAS_MAP,
    )
    assert out == "up"


def test_resolve_action_skips_placeholder_payload_values() -> None:
    """A `${slot}` placeholder must NOT be treated as a valid action
    token — that would make the executor cheerfully pretend to have
    resolved a hop when the actor failed to bind the slot."""

    out = _resolve_action(
        "SELECT",
        {"target": "${target}"},
        action_names=["up", "down", "left", "right"],
        alias_map=ACTION_ALIAS_MAP,
    )
    assert out is None


def test_observational_ops_are_complete() -> None:
    """The OBSERVATIONAL_OPS set must cover every reason/control verb so
    the executor never tries to translate them into env actions. Ties
    the runtime contract back to the lift's verb taxonomy."""

    expected = {
        "INSPECT", "READ", "TRACK",
        "COMPARE", "EVALUATE", "SIMULATE", "PREFER", "PENALIZE", "VERIFY",
        "KEEP", "STOP", "CONTINUE",
    }
    assert OBSERVATIONAL_OPS == expected


# ───────────── End-to-end harness run on a fake env ─────────────


def test_run_skill_steps_env_and_records_post_state() -> None:
    env = _FakeTwoFortyEightEnv()
    env.reset()
    registry, holder, _ = _registry_with_executor(env)
    harness = SkillHarness(registry, config=HarnessConfig(seed=0))

    skill = _slide_skill_with_effects()
    state = initial_state_from_env(env, domain="gymv", task="twenty_forty_eight")
    # Sanity: parser should have surfaced score=0 from the reset state.
    assert state.facts.get("score") == 0
    assert state.facts.get("highest_tile") == 2

    episode = harness.run_skill(skill, state, parent_run_id=None)

    assert episode.outcome is not None
    assert episode.outcome.success is True

    # Every step has both pre and post snapshots.
    assert len(episode.steps) == 1
    step = episode.steps[0]
    assert step.pre_state is not None
    assert step.post_state is not None

    # Post-state carries the parsed score / highest_tile from the env.
    post_facts = (step.post_state or {}).get("facts") or {}
    assert post_facts.get("score") == 4
    assert post_facts.get("highest_tile") == 4
    assert post_facts.get("cumulative_reward") == pytest.approx(4.0)
    assert post_facts.get("phase") == "play"

    # Closure state is updated.
    assert holder.cumulative_reward == pytest.approx(4.0)
    assert holder.outer_step == 1


def test_per_hop_effects_rolls_up_to_outcome_extra() -> None:
    env = _FakeTwoFortyEightEnv()
    env.reset()
    registry, _, _ = _registry_with_executor(env)
    harness = SkillHarness(registry, config=HarnessConfig(seed=0))

    skill = _slide_skill_with_effects()
    state = initial_state_from_env(env, domain="gymv", task="twenty_forty_eight")
    episode = harness.run_skill(skill, state, parent_run_id=None)

    extra = (episode.outcome.extra if episode.outcome else {}) or {}
    roll = extra.get("per_hop_effects") or {}
    assert roll, "expected per_hop_effects to be surfaced on outcome.extra"
    assert roll.get("n_hops_evaluated") == 1
    assert roll.get("n_hops_passed") == 1
    assert roll.get("pass_rate") == 1.0

    per_hop = roll.get("per_hop") or []
    assert len(per_hop) == 1
    pred_types = {p["type"] for p in per_hop[0]["predicates"]}
    assert pred_types == {"cumulative_reward_increased", "entity_value_increased"}
    for p in per_hop[0]["predicates"]:
        assert p["passed"] is True, p


def test_observational_op_does_not_step_env() -> None:
    """An INSPECT-only skill must NOT step the env, but the harness
    should still produce a successful episode (no env-side delta is
    expected)."""

    env = _FakeTwoFortyEightEnv()
    env.reset()
    registry, holder, _ = _registry_with_executor(env)
    harness = SkillHarness(registry, config=HarnessConfig(seed=0))

    skill = SkillRecord.new(
        name="GATHER__INSPECT",
        skill_type=SkillType.GROUNDING,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
        protocol=[
            {
                "op": "INSPECT",
                "payload": {"target": "board"},
                "slot_types": {"target": "container_entity"},
                "preconditions": [],
                "effects_add": [],
                "effects_del": [],
                "evidence_role": "GATHER",
                "notes": "inspect the board",
                "lift_mode": "first",
            }
        ],
    )
    state = initial_state_from_env(env, domain="gymv", task="twenty_forty_eight")
    episode = harness.run_skill(skill, state, parent_run_id=None)

    assert episode.outcome is not None
    assert episode.outcome.success is True
    assert holder.outer_step == 0     # zero env steps
    assert holder.cumulative_reward == pytest.approx(0.0)


def test_unresolvable_op_aborts_cleanly_in_strict_mode() -> None:
    """An unresolvable env-mutating op aborts with `no_env_action_for_op`
    when the executor is configured in strict (`abort`) mode. This is
    the path the gate hardening pass takes to surface missing slot
    bindings as adapter failures."""

    env = _FakeTwoFortyEightEnv()
    env.reset()
    registry, _, _ = _registry_with_executor(env, on_unresolved="abort")
    harness = SkillHarness(registry, config=HarnessConfig(seed=0))

    skill = SkillRecord.new(
        name="COMMIT__SLIDE_DIAGONAL",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
        protocol=[
            {
                "op": "SLIDE",
                "payload": {"direction": "diagonal"},
                "slot_types": {"direction": "enum"},
                "preconditions": [],
                "effects_add": [],
                "effects_del": [],
                "evidence_role": "COMMIT",
                "notes": "slide diagonally",
                "lift_mode": "first",
            }
        ],
    )
    state = initial_state_from_env(env, domain="gymv", task="twenty_forty_eight")
    episode = harness.run_skill(skill, state, parent_run_id=None)

    assert episode.outcome is not None
    assert episode.outcome.success is False
    assert "no_env_action_for_op" in (episode.outcome.abort_reason or "")


def test_unresolvable_op_soft_skips_in_default_mode() -> None:
    """The default executor mode is `skip` so a redundant env-mutating
    hop (e.g. lifted EXECUTE() following SLIDE.direction=up) is recorded
    as observational evidence and the run completes. This is the
    load-bearing behaviour for actually executing real cold-start
    protocols on the real env."""

    env = _FakeTwoFortyEightEnv()
    env.reset()
    registry, holder, _ = _registry_with_executor(env)  # default: skip
    harness = SkillHarness(registry, config=HarnessConfig(seed=0))

    skill = SkillRecord.new(
        name="COMMIT__SLIDE_PLUS_REDUNDANT_EXECUTE",
        skill_type=SkillType.ACTION,
        source_type=SkillSourceType.MINED,
        feasible_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
        protocol=[
            {
                "op": "SLIDE",
                "payload": {"direction": "up"},
                "slot_types": {"direction": "enum"},
                "preconditions": [],
                "effects_add": [
                    {"type": "cumulative_reward_increased", "args": {}},
                ],
                "effects_del": [],
                "evidence_role": "COMMIT",
                "notes": "slide upward",
                "lift_mode": "first",
            },
            {
                "op": "EXECUTE",
                "payload": {},  # no slot context
                "slot_types": {},
                "preconditions": [],
                "effects_add": [],
                "effects_del": [],
                "evidence_role": "COMMIT",
                "notes": "execute the slide",
                "lift_mode": "first",
            },
        ],
    )
    state = initial_state_from_env(env, domain="gymv", task="twenty_forty_eight")
    episode = harness.run_skill(skill, state, parent_run_id=None)

    assert episode.outcome is not None
    assert episode.outcome.success is True
    # The env stepped exactly once (SLIDE.up), not twice — the redundant
    # EXECUTE was soft-skipped.
    assert holder.outer_step == 1
    assert holder.cumulative_reward == pytest.approx(4.0)
