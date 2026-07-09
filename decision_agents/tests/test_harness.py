"""Per-harness unit tests for the unified single-MDP migration.

Covers the five reference harnesses that ship with
:mod:`decision_agents.core`:

* :class:`GymHarness`     — gym-style env wrapper (reset / step / valid_actions).
* :class:`BrowserHarness` — schema-derived ``click`` / ``type`` / ``scroll``;
                            ``step`` raises until a Playwright env is plugged in.
* :class:`OSWorldHarness` — fixed core vocabulary + entity-keyed ``xdotool`` clicks;
                            ``step`` raises until an OSWorld env is plugged in.
* :class:`VRHarness`      — read-only image, scratchpad-mutating ops,
                            ``ANSWER`` terminates with ``+1`` on gold match.
* :class:`VideoHarness`   — extends :class:`VRHarness` with frame cursor
                            + temporal ops (``NEXT_FRAME / JUMP / WINDOW``).

These tests are deliberately offline — no LLM, no env, no Playwright.
The harnesses are exercised directly so each one's ``valid_actions`` /
``step`` contract is locked in independently of the actor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

from decision_agents.core import (
    BrowserHarness,
    GymHarness,
    HarnessState,
    OSWorldHarness,
    VRHarness,
    VideoHarness,
    VisualInput,
)
from decision_agents.core.harness import (
    ACTION_KIND_PRIMITIVE,
    ACTION_KIND_VR_ANSWER,
    ACTION_KIND_VR_LOOK,
    ACTION_KIND_VR_NOTE,
    ACTION_KIND_VR_RETRIEVE,
    ACTION_KIND_VIDEO_JUMP,
    ACTION_KIND_VIDEO_NEXT_FRAME,
    parse_op_call,
)
from decision_agents.schema_parser import parse_state_schema


# ══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════


_BROWSER_SCHEMA = """\
<state>
domain=browser
task=search-bookmark
goal=find a hotel
step=0

<entities>
e1[type=widget, label=search_box, bid=b1, pos=null, ontology=interactive_element]
e2[type=widget, label=submit, bid=b2, pos=null, ontology=interactive_element]

<attributes>
e1.state=visible
e2.state=visible

<affordances>
e1.affords=[type, click]
e2.affords=[click]

<state_flags>
progress=null
phase=null
scene_type=web_page
error=null
dialog_open=false
input_pending=false

<targets>
target=e1
blocker=null
constraint=null
candidate_set=[e1,e2]
history_anchor=null

<actions>
a1=click
a2=type
</state>
"""


_OS_SCHEMA = """\
<state>
domain=os
task=open-terminal
goal=launch terminal
step=0

<entities>
e1[type=widget, label=terminal_icon, bid=null, pos=100,200,32,32, ontology=interactive_element]
e2[type=widget, label=close_button, bid=null, pos=400,50,16,16, ontology=interactive_element]

<attributes>
e1.state=visible
e2.state=visible

<affordances>
e1.affords=[click]
e2.affords=[click]

<state_flags>
progress=null
phase=null
scene_type=desktop
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
a1=click
</state>
"""


@pytest.fixture
def fake_image() -> VisualInput:
    """Tiny stand-in for a real screenshot — no bytes ever decoded."""
    return VisualInput(image_url="https://example.com/test.png", caption="fake")


# ══════════════════════════════════════════════════════════════════════
# GymHarness
# ══════════════════════════════════════════════════════════════════════


@dataclass
class _Stub5TupleEnv:
    """Returns the 5-tuple ``(obs, reward, term, trunc, info)``."""

    valid: List[str] = field(default_factory=lambda: ["a", "b", "c"])
    rewards: List[float] = field(default_factory=lambda: [0.0, 1.0])
    _t: int = 0

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self._t = 0
        return "obs0", {"valid_actions": list(self.valid)}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        r = self.rewards[self._t - 1] if self._t - 1 < len(self.rewards) else 0.0
        term = self._t >= len(self.rewards)
        return f"obs{self._t}", r, term, False, {"valid_actions": list(self.valid)}


@dataclass
class _Stub4TupleEnv:
    """Returns the 4-tuple ``(obs, reward, done, info)``."""

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        return "obs0", {"available_actions": ["x", "y"]}

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        return "obs1", 0.5, True, {"available_actions": ["x", "y"]}


class TestGymHarness:
    def test_reset_passes_through_5tuple(self) -> None:
        h = GymHarness(_Stub5TupleEnv())
        obs, info = h.reset()
        assert obs == "obs0"
        assert info["valid_actions"] == ["a", "b", "c"]

    def test_step_folds_term_trunc_into_done(self) -> None:
        h = GymHarness(_Stub5TupleEnv(rewards=[0.0]))
        h.reset()
        obs, reward, done, info = h.step("a")
        assert obs == "obs1"
        assert reward == 0.0
        assert done is True

    def test_step_accepts_4tuple(self) -> None:
        h = GymHarness(_Stub4TupleEnv())
        h.reset()
        obs, reward, done, info = h.step("x")
        assert obs == "obs1"
        assert reward == pytest.approx(0.5)
        assert done is True

    def test_valid_actions_prefer_info_then_schema(self) -> None:
        h = GymHarness(_Stub5TupleEnv())
        # info wins.
        out = h.valid_actions(
            HarnessState(observation="", info={"valid_actions": ["i1", "i2"]})
        )
        assert out == ["i1", "i2"]
        # Empty info → falls back to schema.actions.
        schema = parse_state_schema(_BROWSER_SCHEMA)
        out = h.valid_actions(HarnessState(observation="", schema=schema))
        assert out and all(isinstance(a, str) for a in out)
        # No info, no schema → empty list (the actor will fall back to
        # "no-op", but that's the actor's problem, not ours).
        assert h.valid_actions(HarnessState(observation="")) == []

    def test_action_kind_is_primitive(self) -> None:
        h = GymHarness(_Stub5TupleEnv())
        assert h.action_kind("anything") == ACTION_KIND_PRIMITIVE


# ══════════════════════════════════════════════════════════════════════
# BrowserHarness
# ══════════════════════════════════════════════════════════════════════


class TestBrowserHarness:
    def test_step_without_env_raises(self) -> None:
        h = BrowserHarness()
        with pytest.raises(NotImplementedError):
            h.step('click(b1)')

    def test_valid_actions_emits_per_entity_templates(self) -> None:
        h = BrowserHarness()
        schema = parse_state_schema(_BROWSER_SCHEMA)
        actions = h.valid_actions(HarnessState(observation="", schema=schema))
        # ``e1`` has both ``click`` and ``type`` affordances + bid=b1.
        assert any(a.startswith("click(b1)") for a in actions)
        assert any('type(b1' in a for a in actions)
        # Always-on chrome / answer sentinel.
        assert any('key("Enter")' == a for a in actions)
        assert any('ANSWER(' in a for a in actions)

    def test_valid_actions_fallback_when_no_schema(self) -> None:
        h = BrowserHarness()
        actions = h.valid_actions(HarnessState(observation=""))
        # At minimum the chrome / answer templates land.
        assert 'ANSWER("<text>")' in actions
        assert 'key("Enter")' in actions

    def test_action_kind_is_primitive(self) -> None:
        h = BrowserHarness()
        assert h.action_kind('click(b1)') == ACTION_KIND_PRIMITIVE


# ══════════════════════════════════════════════════════════════════════
# OSWorldHarness
# ══════════════════════════════════════════════════════════════════════


class TestOSWorldHarness:
    def test_step_without_env_raises(self) -> None:
        h = OSWorldHarness()
        with pytest.raises(NotImplementedError):
            h.step('xdotool("click 100 200")')

    def test_valid_actions_includes_core_vocab(self) -> None:
        h = OSWorldHarness()
        actions = h.valid_actions(HarnessState(observation=""))
        joined = " | ".join(actions)
        for needle in ("xdotool", "type", "bash", "ANSWER"):
            assert needle in joined, f"core vocab missing: {needle}"

    def test_valid_actions_pre_fills_clicks_from_schema(self) -> None:
        h = OSWorldHarness()
        schema = parse_state_schema(_OS_SCHEMA)
        actions = h.valid_actions(HarnessState(observation="", schema=schema))
        # e1.pos=100,200,32,32 → click target should land somewhere
        # inside that bounding box (centre is 116,216).
        assert any('xdotool("click ' in a for a in actions)


# ══════════════════════════════════════════════════════════════════════
# VRHarness
# ══════════════════════════════════════════════════════════════════════


class TestVRHarness:
    def test_reset_emits_question_obs(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="how many cubes?")
        obs, info = h.reset()
        assert obs == "how many cubes?"
        assert info["task"] == "how many cubes?"
        assert info["image"]["image_url"].endswith("test.png")

    def test_look_writes_grounded_slot(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="q")
        h.reset()
        obs, reward, done, info = h.step("LOOK(scene)")
        assert reward == 0.0
        assert done is False
        assert "scene" in h.scratchpad.grounded_slots

    def test_note_appends_to_scratchpad(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="q")
        h.reset()
        h.step('NOTE("two cubes")')
        assert any("two cubes" in n for n in h.scratchpad.notes)

    def test_retrieve_writes_memory_hits(self, fake_image: VisualInput) -> None:
        class _Mem:
            def __init__(self) -> None:
                self.queries: List[str] = []

            def query(self, q: str, k: int = 3) -> List[Dict[str, Any]]:
                self.queries.append(q)
                return [{"summary": "prev hit"}]

            def add_experience(self, **kw: Any) -> None:  # pragma: no cover
                pass

        mem = _Mem()
        h = VRHarness(image=fake_image, question="q")
        h.bind_actor(memory=mem)
        h.reset()
        obs, r, done, info = h.step('RETRIEVE("colors")')
        assert mem.queries == ['"colors"']  # quotes survive — the harness
        # leaves the arg as-typed so memory backends can dequote however
        # they like.
        assert info["memory_hits"] == 1
        assert h.scratchpad.memory_hits

    def test_answer_terminates_and_scores(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="how many?", gold_answer="3")
        h.reset()
        obs, reward, done, info = h.step('ANSWER("3")')
        assert done is True
        assert reward == pytest.approx(1.0)
        assert info["correct"] is True
        # Wrong answer → 0 reward, but still terminates.
        h2 = VRHarness(image=fake_image, question="q", gold_answer="3")
        h2.reset()
        _, r2, d2, _ = h2.step('ANSWER("7")')
        assert d2 is True
        assert r2 == 0.0

    def test_max_steps_truncates(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="q", max_steps=2)
        h.reset()
        h.step("LOOK(scene)")
        obs, reward, done, info = h.step("LOOK(scene)")
        assert done is True
        assert info.get("truncated") is True

    def test_action_kind_routing(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="q")
        assert h.action_kind("LOOK(scene)") == ACTION_KIND_VR_LOOK
        assert h.action_kind('RETRIEVE("x")') == ACTION_KIND_VR_RETRIEVE
        assert h.action_kind('NOTE("hi")') == ACTION_KIND_VR_NOTE
        assert h.action_kind('ANSWER("3")') == ACTION_KIND_VR_ANSWER

    def test_valid_actions_surfaces_answer_first(self, fake_image: VisualInput) -> None:
        h = VRHarness(image=fake_image, question="q")
        actions = h.valid_actions(HarnessState(observation="q"))
        assert actions[0].startswith("ANSWER(")


# ══════════════════════════════════════════════════════════════════════
# VideoHarness
# ══════════════════════════════════════════════════════════════════════


class TestVideoHarness:
    def _frames(self, n: int = 4) -> List[VisualInput]:
        return [
            VisualInput(image_url=f"https://example.com/f{i}.png", caption=f"f{i}")
            for i in range(n)
        ]

    def test_reset_pins_cursor_at_zero(self) -> None:
        h = VideoHarness(frames=self._frames(3), question="who entered?")
        obs, info = h.reset()
        assert info["cursor"] == 0
        assert info["n_frames"] == 3
        assert h.current_frame is not None
        assert h.current_frame.image_url.endswith("f0.png")

    def test_next_frame_advances_cursor(self) -> None:
        h = VideoHarness(frames=self._frames(3), question="q")
        h.reset()
        _, _, _, info = h.step("NEXT_FRAME()")
        assert info["cursor"] == 1
        assert h.current_frame.image_url.endswith("f1.png")

    def test_jump_clamps_to_clip_bounds(self) -> None:
        h = VideoHarness(frames=self._frames(3), question="q")
        h.reset()
        _, _, _, info = h.step("JUMP(99)")
        assert info["cursor"] == 2  # clamped to len-1
        _, _, _, info = h.step("JUMP(-5)")
        assert info["cursor"] == 0

    def test_window_records_range_and_moves_cursor(self) -> None:
        h = VideoHarness(frames=self._frames(5), question="q")
        h.reset()
        _, _, _, info = h.step("WINDOW(2,4)")
        assert info["cursor"] == 2
        assert any("WINDOW=" in n for n in h.scratchpad.notes)

    def test_action_kind_routes_temporal_ops(self) -> None:
        h = VideoHarness(frames=self._frames(2), question="q")
        assert h.action_kind("NEXT_FRAME()") == ACTION_KIND_VIDEO_NEXT_FRAME
        assert h.action_kind("JUMP(1)") == ACTION_KIND_VIDEO_JUMP
        # Inherits VR routing for shared ops.
        assert h.action_kind('ANSWER("yes")') == ACTION_KIND_VR_ANSWER

    def test_inherited_vr_ops_still_work(self) -> None:
        h = VideoHarness(frames=self._frames(2), question="q", gold_answer="yes")
        h.reset()
        h.step("LOOK(scene)")
        assert "scene" in h.scratchpad.grounded_slots
        _, reward, done, _ = h.step('ANSWER("yes")')
        assert done is True
        assert reward == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# parse_op_call helper
# ══════════════════════════════════════════════════════════════════════


class TestParseOpCall:
    def test_basic_split(self) -> None:
        assert parse_op_call("LOOK(scene)") == ("LOOK", "scene")

    def test_handles_missing_close_paren(self) -> None:
        assert parse_op_call("LOOK(scene") == ("LOOK", "scene")

    def test_uppercases_op_only(self) -> None:
        op, arg = parse_op_call("look(Scene)")
        assert op == "LOOK"
        assert arg == "Scene"

    def test_multi_arg_returned_as_string(self) -> None:
        op, arg = parse_op_call("COMPARE(e1,e2,color)")
        assert op == "COMPARE"
        assert arg == "e1,e2,color"

    def test_empty_action(self) -> None:
        assert parse_op_call("") == ("", "")

    def test_no_paren_returns_empty_op(self) -> None:
        op, arg = parse_op_call("just a string")
        assert op == ""
        assert arg == "just a string"
