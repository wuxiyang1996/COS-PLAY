"""Smoke tests for the schema-native Actor Agent stack.

Covers the new modules (post unified-harness migration):

* ``schema_parser`` — ``<state>`` → :class:`StateSchema`
* ``skill_interface`` — ``NullSkillProvider`` + custom stub provider
* ``skill_tracker`` — reselect triggers + slot coverage (PLAN §10)
* ``actor_agent`` — end-to-end step + run_actor_episode over a stub env

End-to-end coverage of the per-task :class:`Harness` lives in
``test_harness.py`` (per-harness ``valid_actions`` / ``step``),
``test_actor_with_vr_harness.py`` (VR rollout), and
``test_actor_back_compat.py`` (legacy ``run_actor_episode(env, agent)``
auto-binds a :class:`GymHarness`).

The tests run offline (no LLM / API calls).  When ``API_func.ask_model``
is unavailable the actor falls back to "first valid action" which keeps
the tests deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Import directly from submodules so we can exercise the actor stack
# without paying the ``decision_agents.__init__`` cost (which pulls in
# ``API_func`` and its optional ``anthropic`` dependency).
from decision_agents.schema_parser import (
    StateSchema,
    parse_state_schema,
    resolve_entity_action,
)
from decision_agents.skill_interface import (
    NullSkillProvider,
    SkillGuidance,
)
from decision_agents.skill_tracker import SkillTracker
from decision_agents.actor_agent import (
    ActorAgent,
    ActorDecision,
    InnerScratchpad,
    _extract_action_from_reply,
    run_actor_episode,
)


# ══════════════════════════════════════════════════════════════════════
# Fixture: a realistic 2048-style schema
# ══════════════════════════════════════════════════════════════════════


GYMV_SCHEMA = """\
<state>
domain=gymv
task=Game2048-v0
goal=Reach 2048
step=12

<entities>
e1[type=object, label=tile_2, bid=null, pos=0,0,1,1, ontology=selectable_entity]
e2[type=object, label=tile_4, bid=null, pos=0,1,1,1, ontology=selectable_entity]
e3[type=object, label=tile_2, bid=null, pos=1,1,1,1, ontology=selectable_entity]
e4[type=object, label=tile_4, bid=null, pos=1,2,1,1, ontology=selectable_entity]
e5[type=region, label=empty, bid=null, pos=2,3,1,1, ontology=navigable_region]

<attributes>
e1.state=visible
e1.value=2
e2.state=visible
e2.value=4
e3.state=visible
e3.value=2

<affordances>
e1.affords=[select, track]
e2.affords=[select, track]
e3.affords=[select, track]

<relations>
adjacent(e1,e2)
adjacent(e3,e4)

<state_flags>
progress=0.25
phase=mid
scene_type=game_play
error=null
dialog_open=false
input_pending=false

<targets>
target=e1
blocker=null
constraint=merge tiles with same value
candidate_set=[e1,e3]
history_anchor=null

<uncertainty>
e5.label=high

<actions>
a1=[Up]
a2=[Down]
a3=[Left]
a4=[Right]
</state>
"""


IMAGE_QA_SCHEMA = """\
<state>
domain=image_qa
task=tir_bench-q1
goal=What color is the cube?
step=0

<entities>
e1[type=object, label=red cube, bid=null, pos=120,80,40,40, ontology=tracked_entity]
e2[type=object, label=blue sphere, bid=null, pos=200,80,40,40, ontology=tracked_entity]

<attributes>
e1.state=visible
e1.value=red
e2.state=visible
e2.value=blue

<affordances>
e1.affords=[inspect]
e2.affords=[inspect]

<relations>
adjacent(e1,e2)

<state_flags>
progress=null
phase=null
scene_type=image_qa
error=null
dialog_open=false
input_pending=false

<targets>
target=e1
blocker=null
constraint=null
candidate_set=[e1]
history_anchor=null

<evidence>
hop1.abstract_op=GROUND
hop1.tool=detect_objects
hop1.result_ref={e1,e2}
hop1.confidence=high
hop2.abstract_op=CHECK
hop2.tool=extract_colors
hop2.result_ref={e1}
hop2.confidence=high

<answer>
answer=red
grounding=[e1]
evidence_chain=[hop1,hop2]
confidence=high
</state>
"""


# ══════════════════════════════════════════════════════════════════════
# Schema parser
# ══════════════════════════════════════════════════════════════════════


class TestSchemaParser:
    def test_parse_gymv_schema(self) -> None:
        schema = parse_state_schema(GYMV_SCHEMA)
        assert schema is not None
        assert isinstance(schema, StateSchema)
        assert schema.domain == "gymv"
        assert schema.goal == "Reach 2048"
        assert schema.step == 12
        assert len(schema.entities) == 5
        assert schema.entity_order[:2] == ["e1", "e2"]

        e1 = schema.get_entity("e1")
        assert e1 is not None
        assert e1.type == "object"
        assert e1.label == "tile_2"
        assert e1.ontology == "selectable_entity"
        assert e1.pos == (0, 0, 1, 1)
        assert e1.state == "visible"
        assert e1.value == "2"
        assert e1.affords == ["select", "track"]

        assert schema.targets.target == "e1"
        assert schema.targets.blocker is None
        assert schema.targets.candidate_set == ["e1", "e3"]
        assert schema.targets.constraint == "merge tiles with same value"

        assert schema.state_flags.progress == pytest.approx(0.25)
        assert schema.state_flags.phase == "mid"
        assert schema.state_flags.scene_type == "game_play"
        assert schema.state_flags.dialog_open is False

        assert schema.actions == ["[Up]", "[Down]", "[Left]", "[Right]"]
        assert len(schema.relations) == 2
        assert schema.relations[0].name == "adjacent"

        # Uncertainty should be attached to the referenced entity
        e5 = schema.get_entity("e5")
        assert e5 is not None
        assert e5.uncertainty.get("label") == "high"
        assert "e5" in schema.high_uncertainty_eids("label")

    def test_parse_image_qa_schema(self) -> None:
        schema = parse_state_schema(IMAGE_QA_SCHEMA)
        assert schema is not None
        assert schema.domain == "image_qa"
        assert schema.answer is not None
        assert schema.answer.answer == "red"
        assert schema.answer.grounding == ["e1"]
        assert schema.answer.confidence == "high"
        assert len(schema.evidence) == 2
        hop1 = schema.evidence[0]
        assert hop1.abstract_op == "GROUND"
        assert hop1.tool == "detect_objects"
        assert set(hop1.result_ref) == {"e1", "e2"}

    def test_parse_rejects_empty(self) -> None:
        assert parse_state_schema("") is None
        assert parse_state_schema(None) is None
        assert parse_state_schema("no tags here") is None

    def test_slot_coverage(self) -> None:
        schema = parse_state_schema(GYMV_SCHEMA)
        assert schema is not None
        cov = schema.slot_coverage(
            ["target", "blocker", "candidate_set", "constraint", "e1", "e99", "e1.value"]
        )
        assert cov == {
            "target": True,
            "blocker": False,
            "candidate_set": True,
            "constraint": True,
            "e1": True,
            "e99": False,
            "e1.value": True,
        }
        missing = schema.missing_slots(["target", "blocker", "candidate_set"])
        assert missing == ["blocker"]

    def test_compact_summary_fits_budget(self) -> None:
        schema = parse_state_schema(GYMV_SCHEMA)
        assert schema is not None
        summary = schema.compact_summary(max_chars=220)
        assert len(summary) <= 220
        assert "domain=gymv" in summary
        assert "target=e1" in summary


class TestResolveEntityAction:
    def test_resolve_click_entity(self) -> None:
        schema = parse_state_schema(GYMV_SCHEMA)
        assert schema is not None
        r = resolve_entity_action("select(e2)", schema)
        assert r.eid == "e2"
        assert r.verb == "select"
        # no bid available on a gym tile → resolved uses label
        assert r.resolved == "select(tile_4)"

    def test_resolve_raw_action_passthrough(self) -> None:
        schema = parse_state_schema(GYMV_SCHEMA)
        r = resolve_entity_action("[Up]", schema)
        assert r.eid is None
        assert r.resolved == "[Up]"

    def test_resolve_unknown_entity_stays_raw(self) -> None:
        schema = parse_state_schema(GYMV_SCHEMA)
        r = resolve_entity_action("click(e99)", schema)
        assert r.eid == "e99"
        # unknown eid → resolved falls back to the raw string
        assert r.resolved == "click(e99)"


# ══════════════════════════════════════════════════════════════════════
# Skill interface / tracker
# ══════════════════════════════════════════════════════════════════════


class _StubSkillProvider:
    """Minimal in-memory provider that surfaces a single hard-coded skill."""

    def __init__(self, guidance: SkillGuidance) -> None:
        self.guidance = guidance
        self.select_calls: int = 0
        self.outcome_calls: List[Dict[str, Any]] = []

    def select(
        self,
        query: str,
        *,
        state_summary: str = "",
        structured_state: Any = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 1,
    ) -> List[SkillGuidance]:
        self.select_calls += 1
        return [self.guidance]

    def record_outcome(
        self,
        skill_id: str,
        *,
        outcome: str,
        reward: float = 0.0,
        steps_taken: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.outcome_calls.append({
            "skill_id": skill_id,
            "outcome": outcome,
            "reward": reward,
            "steps_taken": steps_taken,
        })

    def available_skills(self) -> List[str]:
        return [self.guidance.skill_id]


class TestNullSkillProvider:
    def test_select_returns_empty(self) -> None:
        p = NullSkillProvider()
        assert p.select("anything") == []
        assert p.available_skills() == []
        p.record_outcome("x", outcome="success")


class TestSkillTracker:
    def _guidance(self, **overrides: Any) -> SkillGuidance:
        defaults = dict(
            skill_id="merge_to_corner",
            skill_name="Merge to Corner",
            protocol_steps=["[Up]", "[Right]", "[Up]", "[Right]"],
            success_criteria=["tile_16 visible"],
            abort_criteria=["game_over"],
            expected_duration=6,
            required_slots=["target"],
            eff_add=["tile_16"],
        )
        defaults.update(overrides)
        return SkillGuidance(**defaults)

    def test_reselect_when_no_active_skill(self) -> None:
        tracker = SkillTracker()
        schema = parse_state_schema(GYMV_SCHEMA)
        reselect, reason = tracker.should_reselect(schema)
        assert reselect
        assert reason == "no_active_skill"

    def test_activate_with_target_populates_ready(self) -> None:
        tracker = SkillTracker()
        schema = parse_state_schema(GYMV_SCHEMA)
        g = self._guidance()
        check = tracker.activate(g, schema)
        assert check.activated
        assert check.ready
        assert check.missing_slots == []
        assert tracker.active_skill_id == "merge_to_corner"
        assert tracker.current_step() == "[Up]"
        assert tracker.progress_marker() == "0/4"

    def test_activate_with_missing_slot_flags_ground(self) -> None:
        tracker = SkillTracker()
        schema = parse_state_schema(GYMV_SCHEMA)
        # Need blocker which isn't set in the fixture schema.
        g = self._guidance(required_slots=["target", "blocker"])
        check = tracker.activate(g, schema)
        assert check.activated
        assert not check.ready
        assert "blocker" in check.missing_slots
        assert tracker.needs_ground
        assert tracker.pending_ground_slots == ["blocker"]

    def test_stall_triggers_reselect(self) -> None:
        tracker = SkillTracker(stall_window=3)
        schema = parse_state_schema(GYMV_SCHEMA)
        tracker.activate(self._guidance(), schema)
        for _ in range(3):
            tracker.record_step(reward=0.0, schema_after=schema)
        reselect, reason = tracker.should_reselect(schema)
        assert reselect
        assert reason == "stall"

    def test_duration_cap_triggers_reselect(self) -> None:
        tracker = SkillTracker(stall_window=100)
        schema = parse_state_schema(GYMV_SCHEMA)
        tracker.activate(
            self._guidance(expected_duration=2, required_slots=[]),
            schema,
        )
        # Positive reward so stall isn't the trigger.
        tracker.record_step(reward=1.0, schema_after=schema)
        tracker.record_step(reward=1.0, schema_after=schema)
        reselect, reason = tracker.should_reselect(schema)
        assert reselect
        assert reason == "duration_exceeded"

    def test_finalize_active_skill_records_stats(self) -> None:
        tracker = SkillTracker()
        schema = parse_state_schema(GYMV_SCHEMA)
        tracker.activate(self._guidance(), schema)
        tracker.record_step(reward=0.5, schema_after=schema)
        rec = tracker.finalize_active_skill("success")
        assert rec is not None
        assert rec["skill_id"] == "merge_to_corner"
        assert rec["outcome"] == "success"
        assert rec["reward"] == pytest.approx(0.5)
        assert rec["steps_taken"] == 1


# ══════════════════════════════════════════════════════════════════════
# ActorAgent end-to-end (stub env, no LLM calls)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class _StubEnv:
    """Deterministic 1-D env that rewards ``[Up]`` and terminates."""

    schemas: List[str] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    _t: int = 0

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self._t = 0
        info = {
            "schema": self.schemas[0] if self.schemas else None,
            "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
            "game": "gymv-stub",
            "task": "Reach 2048",
        }
        return "stub obs 0", info

    def step(
        self, action: str
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        idx = min(self._t, len(self.schemas) - 1) if self.schemas else 0
        done = self._t >= max(1, len(self.schemas) - 1)
        reward = (
            self.step_rewards[self._t - 1]
            if self._t - 1 < len(self.step_rewards)
            else 0.0
        )
        info = {
            "schema": self.schemas[idx] if self.schemas else None,
            "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
            "game": "gymv-stub",
        }
        return f"stub obs {self._t}", reward, done, False, info


class TestActorAgent:
    def test_single_step_with_null_provider(self) -> None:
        agent = ActorAgent(skill_provider=NullSkillProvider())
        agent.reset()
        decision = agent.step(
            observation="obs",
            schema_text=GYMV_SCHEMA,
            task="Reach 2048",
            valid_actions=["[Up]", "[Down]", "[Left]", "[Right]"],
        )
        assert isinstance(decision, ActorDecision)
        assert decision.action in {"[Up]", "[Down]", "[Left]", "[Right]"}
        assert decision.summary
        # The unified single-MDP loop no longer emits a hop trace; it picks
        # an action directly from ``harness.valid_actions(state)``.
        assert decision.parse_path  # non-empty parse-strategy tag

    def test_skill_provider_is_consulted_on_first_step(self) -> None:
        g = SkillGuidance(
            skill_id="merge_to_corner",
            skill_name="Merge to Corner",
            protocol_steps=["[Up]", "[Right]"],
            required_slots=["target"],
            expected_duration=4,
        )
        provider = _StubSkillProvider(g)
        agent = ActorAgent(skill_provider=provider)
        agent.reset()
        decision = agent.step(
            observation="obs",
            schema_text=GYMV_SCHEMA,
            task="Reach 2048",
            valid_actions=["[Up]", "[Down]", "[Left]", "[Right]"],
        )
        assert decision.active_skill_id == "merge_to_corner"
        assert decision.reselected
        # Protocol step 0 is `[Up]` which IS in the valid action set —
        # the actor should follow it without calling the LLM.
        assert decision.action == "[Up]"
        assert provider.select_calls == 1

    def test_episode_runner_produces_experiences(self) -> None:
        env = _StubEnv(
            schemas=[GYMV_SCHEMA, GYMV_SCHEMA, GYMV_SCHEMA],
            step_rewards=[0.0, 1.0],
        )
        episode = run_actor_episode(
            env,
            agent=ActorAgent(skill_provider=NullSkillProvider()),
            task="Reach 2048",
            max_steps=10,
        )
        assert episode.experiences
        assert episode.metadata["steps"] >= 1
        assert "cumulative_reward" in episode.metadata
        for exp in episode.experiences:
            assert hasattr(exp, "reward_details")
            assert "r_total" in exp.reward_details
            assert hasattr(exp, "extras")
            # Unified loop replaces the ``hop_trace`` extra with the
            # per-action ``action_kind`` tag from the bound harness.
            assert "action_kind" in exp.extras

    def test_reselect_sets_queried_skill_and_adds_cost(self) -> None:
        """PLAN §4 — a reselect event fires ``query_skill_cost``."""
        g = SkillGuidance(
            skill_id="merge_to_corner",
            protocol_steps=["[Up]"],
            required_slots=["target"],
        )
        provider = _StubSkillProvider(g)
        agent = ActorAgent(skill_provider=provider)
        agent.reset()
        decision = agent.step(
            observation="obs",
            schema_text=GYMV_SCHEMA,
            task="Reach 2048",
            valid_actions=["[Up]", "[Down]", "[Left]", "[Right]"],
        )
        assert decision.queried_skill is True
        rr = agent.observe_result(decision, reward=0.0, next_schema_text=GYMV_SCHEMA)
        # query_skill_cost is -0.01 by default, so r_cost <= that.
        assert rr.r_cost <= agent.reward_computer.cfg.query_skill_cost + 1e-9

    def test_decision_to_dict_is_self_describing(self) -> None:
        agent = ActorAgent(skill_provider=NullSkillProvider())
        agent.reset()
        decision = agent.step(
            observation="obs",
            schema_text=GYMV_SCHEMA,
            task="Reach 2048",
            valid_actions=["[Up]", "[Down]", "[Left]", "[Right]"],
        )
        d = decision.to_dict()
        # All new fields are rendered so the GRPO logs can see them.
        for key in (
            "reasoning",
            "queried_skill",
            "queried_mem",
            "parse_path",
            "action_kind",
            "intention",
        ):
            assert key in d, f"ActorDecision.to_dict is missing {key}"
        # ``hop_trace`` is gone post-unified-MDP migration; document the
        # invariant so we don't regress.
        assert "hop_trace" not in d

    def test_anti_repetition_switches_action(self) -> None:
        agent = ActorAgent(
            skill_provider=NullSkillProvider(),
            anti_repetition_window=2,
        )
        agent.reset()
        valid = ["[Up]", "[Down]", "[Left]", "[Right]"]
        # Seed history with two identical zero-reward actions.
        for _ in range(2):
            decision = agent.step(
                observation="obs",
                schema_text=GYMV_SCHEMA,
                task="Reach 2048",
                valid_actions=valid,
            )
            agent.observe_result(
                decision, reward=0.0, next_schema_text=GYMV_SCHEMA
            )

        # The first valid action has been repeated — anti-rep should swap.
        decision = agent.step(
            observation="obs",
            schema_text=GYMV_SCHEMA,
            task="Reach 2048",
            valid_actions=valid,
        )
        assert decision.anti_repetition_triggered
        assert decision.action != agent.state.last_actions[-2]


# ══════════════════════════════════════════════════════════════════════
# Multi-strategy action parser (PLAN §1 step 6)
# ══════════════════════════════════════════════════════════════════════


class TestExtractActionFromReply:
    VALID = ["[Up]", "[Down]", "[Left]", "[Right]"]

    def test_exact_match(self) -> None:
        action, path = _extract_action_from_reply(
            "THOUGHT: go up.\nACTION: [Up]", self.VALID
        )
        assert action == "[Up]"
        assert path == "exact"

    def test_numbered_selection(self) -> None:
        action, path = _extract_action_from_reply(
            "ACTION: 3", self.VALID
        )
        assert action == "[Left]"
        assert path == "numbered"

    def test_numbered_with_dot(self) -> None:
        action, path = _extract_action_from_reply("ACTION: 4.", self.VALID)
        assert action == "[Right]"
        assert path == "numbered"

    def test_edit_distance_typo(self) -> None:
        action, path = _extract_action_from_reply("ACTION: [Righ]", self.VALID)
        assert action == "[Right]"
        assert path == "edit_distance"

    def test_token_overlap(self) -> None:
        # "press up button" shares 'up' with '[Up]'.
        action, path = _extract_action_from_reply(
            "ACTION: press up button", self.VALID
        )
        assert action == "[Up]"
        assert path in {"token_overlap", "loose"}  # 'Up' also matches loosely

    def test_entity_reference_passthrough(self) -> None:
        # Entity-ref actions like click(e5) should survive even when not in
        # the valid-action list — the env runner resolves them later.
        action, path = _extract_action_from_reply(
            "ACTION: click(e5)", ["[Up]", "[Down]"]
        )
        assert action == "click(e5)"
        assert path == "entity_ref"

    def test_loose_fallback(self) -> None:
        action, path = _extract_action_from_reply(
            "I want to press [Down] now.", self.VALID
        )
        assert action == "[Down]"
        assert path == "loose"

    def test_trailing_digit_after_loose_fails(self) -> None:
        # No ACTION: line, no valid action in text, but a trailing digit
        # should still decode as a numbered selection.
        action, path = _extract_action_from_reply("pick option 2", self.VALID)
        assert action == "[Down]"
        assert path == "numbered"

    def test_no_match_returns_empty(self) -> None:
        action, path = _extract_action_from_reply("garbage reply", self.VALID)
        assert action is None
        assert path == ""
