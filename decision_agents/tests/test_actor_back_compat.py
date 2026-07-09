"""Backward-compatibility tests for the unified single-MDP migration.

The unified-harness pivot intentionally preserves the legacy
``run_actor_episode(env, agent, ...)`` entry point: callers that
pre-date the harness contract should keep working byte-identical to
before — the actor auto-binds a :class:`GymHarness` over the supplied
env, and the resulting :class:`ActorDecision` reports an ``"action_kind"
== "primitive"`` (the GymHarness default), which is the same cost
bucket the old code used.

We also pin the deprecation contract: passing the legacy
``hop_policy`` / ``max_hops_per_step`` kwargs continues to compile
without error (so the existing
:class:`decision_agents.SFT.GPT4oCollectorActor` and
:class:`decision_agents.grpo.QwenVLActor` constructors keep
working) — but the call emits a :class:`DeprecationWarning` so callers
notice they're using a removed scaffold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pytest

from decision_agents.actor_agent import ActorAgent, run_actor_episode
from decision_agents.core import GymHarness
from decision_agents.skill_interface import NullSkillProvider


_GYMV_SCHEMA = """\
<state>
domain=gymv
task=Game2048-v0
goal=Reach 2048
step=0

<entities>
e1[type=object, label=tile_2, bid=null, pos=0,0,1,1, ontology=selectable_entity]

<attributes>
e1.state=visible

<affordances>
e1.affords=[select]

<state_flags>
progress=0.1
phase=mid
scene_type=game_play
error=null
dialog_open=false
input_pending=false

<targets>
target=e1
blocker=null
constraint=null
candidate_set=[e1]
history_anchor=null

<actions>
a1=[Up]
a2=[Down]
</state>
"""


@dataclass
class _StubEnv:
    """Mimics the in-tree gymv stub env used by ``test_actor_agent.py``.

    Returns a 5-tuple from ``step`` so we exercise the GymHarness's
    Gymnasium-style fold of ``term/trunc → done``.
    """

    schemas: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=lambda: [0.0, 1.0])
    _t: int = 0

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self._t = 0
        return "obs0", {
            "schema": self.schemas[0] if self.schemas else None,
            "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
            "game": "gymv-stub",
        }

    def step(
        self, action: str
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        idx = min(self._t, len(self.schemas) - 1) if self.schemas else 0
        done = self._t >= max(1, len(self.schemas) - 1)
        r = self.rewards[self._t - 1] if self._t - 1 < len(self.rewards) else 0.0
        return (
            f"obs{self._t}",
            r,
            done,
            False,
            {
                "schema": self.schemas[idx] if self.schemas else None,
                "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
                "game": "gymv-stub",
            },
        )


# ══════════════════════════════════════════════════════════════════════
# Legacy entry-point: run_actor_episode(env, agent) auto-binds GymHarness
# ══════════════════════════════════════════════════════════════════════


class TestRunActorEpisodeBackCompat:
    def test_no_harness_auto_binds_gym(self) -> None:
        env = _StubEnv(schemas=[_GYMV_SCHEMA, _GYMV_SCHEMA, _GYMV_SCHEMA])
        agent = ActorAgent(skill_provider=NullSkillProvider())
        # Pre-condition: actor has no harness bound yet.
        assert agent.harness is None

        episode = run_actor_episode(env, agent=agent, max_steps=5)

        # Post-condition: a GymHarness was auto-bound (the runner
        # contract that lets callers keep passing ``env=`` unchanged).
        assert isinstance(agent.harness, GymHarness)
        assert episode.experiences
        # Every experience reports the ``primitive`` action kind, i.e.
        # the cost bucket gym-style envs have always used.
        for exp in episode.experiences:
            assert exp.extras.get("action_kind") == "primitive"

    def test_explicit_harness_is_respected(self) -> None:
        env = _StubEnv(schemas=[_GYMV_SCHEMA, _GYMV_SCHEMA])
        h = GymHarness(env)
        agent = ActorAgent(skill_provider=NullSkillProvider())
        episode = run_actor_episode(env, agent=agent, harness=h, max_steps=5)
        # The explicit harness is the one bound on the actor.
        assert agent.harness is h
        assert episode.experiences

    def test_metadata_preserved(self) -> None:
        env = _StubEnv(schemas=[_GYMV_SCHEMA, _GYMV_SCHEMA, _GYMV_SCHEMA])
        episode = run_actor_episode(
            env,
            agent=ActorAgent(skill_provider=NullSkillProvider()),
            task="Reach 2048",
            max_steps=5,
        )
        assert "steps" in episode.metadata
        assert "cumulative_reward" in episode.metadata


# ══════════════════════════════════════════════════════════════════════
# Deprecation contract: ``hop_policy`` / ``max_hops_per_step`` still compile
# ══════════════════════════════════════════════════════════════════════


class TestDeprecatedKwargs:
    def test_hop_policy_kwarg_emits_warning_but_works(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = ActorAgent(
                skill_provider=NullSkillProvider(),
                hop_policy=object(),  # any sentinel value
            )
        assert any(issubclass(rec.category, DeprecationWarning) for rec in w)
        assert agent.harness is None  # nothing got bound

    def test_max_hops_kwarg_emits_warning_but_works(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = ActorAgent(
                skill_provider=NullSkillProvider(),
                max_hops_per_step=4,
            )
        assert any(issubclass(rec.category, DeprecationWarning) for rec in w)
        assert agent.harness is None

    def test_run_actor_episode_hop_policy_kwarg_warns(self) -> None:
        env = _StubEnv(schemas=[_GYMV_SCHEMA, _GYMV_SCHEMA])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_actor_episode(
                env,
                agent=ActorAgent(skill_provider=NullSkillProvider()),
                hop_policy=object(),
                max_steps=2,
            )
        assert any(issubclass(rec.category, DeprecationWarning) for rec in w)


# ══════════════════════════════════════════════════════════════════════
# Deprecation shim on ``decision_agents`` package
# ══════════════════════════════════════════════════════════════════════


class TestPackageDeprecationShim:
    @pytest.mark.parametrize(
        "name",
        [
            "HopAction",
            "HopPolicy",
            "HopStep",
            "HopTrace",
            "HeuristicHopPolicy",
            "parse_hop_action",
        ],
    )
    def test_inner_mdp_attributes_warn_then_raise(self, name: str) -> None:
        import decision_agents

        # The shim emits a DeprecationWarning *and* raises AttributeError
        # so callers get both signals.  Using ``pytest.warns`` keeps the
        # warning visible regardless of the active filter.
        with pytest.warns(DeprecationWarning):
            with pytest.raises(AttributeError):
                getattr(decision_agents, name)
