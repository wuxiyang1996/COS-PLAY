"""Per-task ``Harness`` contract for the unified single-MDP actor.

The actor's MDP signature is task-independent — see
``decision_agents/README.md`` "How the actor agent works".  What
varies between game / web / OS / visual-reasoning / video-understanding
is **only**:

* what an *observation* looks like,
* what *actions* are valid in a given state, and
* what ``step(action)`` does to the world (or to a read-only-world
  scratchpad).

A :class:`Harness` packages those three concerns so :class:`ActorAgent`
can stay task-agnostic.  Five reference implementations ship under
``decision_agents/core/``:

* :class:`~decision_agents.core.harness_gym.GymHarness` —
  thin wrapper over a gym-style env.  Mutable world.
* :class:`~decision_agents.core.harness_browser.BrowserHarness` —
  Playwright / BrowserGym action vocabulary.  Mutable world (``step``
  is a stub until the env layer is wired).
* :class:`~decision_agents.core.harness_osworld.OSWorldHarness` —
  desktop primitives + bash.  Mutable world (``step`` is a stub).
* :class:`~decision_agents.core.harness_vr.VRHarness` — read-only
  image + reasoning-op vocabulary that mutates a scratchpad until
  the agent emits ``ANSWER``.
* :class:`~decision_agents.core.harness_video.VideoHarness` — read-only
  clip + frame cursor; same scratchpad ops as :class:`VRHarness` plus
  ``NEXT_FRAME / JUMP / WINDOW / FOCUS / TRACK``.

Compatibility seam
------------------
:class:`ActorAgent` keeps its old ``step(observation, ..., info=...)``
signature (and ``run_actor_episode`` keeps its ``env`` arg) so existing
callers do not break: when no ``harness=`` is supplied, the actor
auto-binds a :class:`GymHarness` over the supplied env / info.  See
``decision_agents/actor_agent.py`` for the dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from decision_agents.schema_parser import StateSchema


# ──────────────────────────────────────────────────────────────────────
# HarnessState — what the actor passes to harness.valid_actions
# ──────────────────────────────────────────────────────────────────────


@dataclass
class HarnessState:
    """Snapshot of everything the harness may need to make a decision.

    The actor builds one of these per outer step (after schema parsing
    and intention inference) and hands it to
    :meth:`Harness.valid_actions`.  Harnesses are expected to be
    **stateless w.r.t. ``HarnessState``** — they read it but don't
    mutate it; their own state lives in instance attributes (e.g. the
    underlying gym env, the VR scratchpad, the video frame cursor).

    Attributes
    ----------
    observation
        Whatever the runner / env handed the actor.  Typically a
        string (game text, URL, screenshot caption) but harnesses may
        accept arbitrary objects.
    schema
        Parsed :class:`~decision_agents.schema_parser.StateSchema`
        when available.  ``None`` for callers that bypassed the VLM
        grounding head.
    info
        The ``info`` dict from the env's most recent
        ``reset`` / ``step``.  Game / web / OS harnesses lean on it
        (``valid_actions`` etc.); VR / Video harnesses ignore it.
    intention
        Outer-step ``[TAG] subgoal`` from
        :func:`decision_agents.agent_helper.infer_intention`.  Used
        by harnesses that want to bias their action enumeration on
        the current strategic intent (e.g. surface ``ANSWER`` early
        for ``[OPTIMIZE]`` subgoals on the VR harness).
    t
        Outer-step counter.  VR / Video harnesses use it as the
        deliberation budget; mutable-world harnesses use it for
        logging.
    """

    observation: Any = ""
    schema: Optional[StateSchema] = None
    info: Dict[str, Any] = field(default_factory=dict)
    intention: str = ""
    t: int = 0


# ──────────────────────────────────────────────────────────────────────
# Action-kind vocabulary (used for r_cost lookup)
# ──────────────────────────────────────────────────────────────────────


# Stable string tags that :class:`~decision_agents.reward_func.RewardConfig`
# maps to per-action costs.  Harnesses return one of these from
# :meth:`Harness.action_kind`.  Adding a new harness?  Pick an existing
# kind when the cost semantics match (``"primitive"`` for "an env
# action that advances the world") rather than inventing a new one,
# unless you also add a matching ``RewardConfig`` field.
ACTION_KIND_PRIMITIVE: str = "primitive"           # default for game / web / OS env actions
ACTION_KIND_VR_LOOK: str = "vr_look"
ACTION_KIND_VR_RETRIEVE: str = "vr_retrieve"
ACTION_KIND_VR_NOTE: str = "vr_note"
ACTION_KIND_VR_ANSWER: str = "vr_answer"
ACTION_KIND_VIDEO_NEXT_FRAME: str = "video_next_frame"
ACTION_KIND_VIDEO_JUMP: str = "video_jump"
ACTION_KIND_VIDEO_FOCUS: str = "video_focus"
ACTION_KIND_VIDEO_TRACK: str = "video_track"


# ──────────────────────────────────────────────────────────────────────
# Harness protocol
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class Harness(Protocol):
    """Per-task action-source + step transition.

    Required methods
    ----------------
    reset() -> (observation, info)
        Initialise / re-initialise the harness for a new episode.

    step(action) -> (next_obs, reward, done, info)
        Apply *action*, returning a 4-tuple shaped like a Gymnasium
        step (terminated and truncated folded into ``done`` for
        simplicity — episodes can be told apart later from ``info``).

    valid_actions(state) -> List[str]
        Enumerate the actions the actor may pick from in *state*.
        The result is the candidate list passed to ``_call_llm`` and
        matched against by the multi-strategy action parser; the
        order matters for the "numbered" parse path.

    Optional methods (defaulted by duck-typing inside ``ActorAgent``):

    action_kind(action) -> str
        Return the cost bucket for *action* (one of the
        ``ACTION_KIND_*`` constants above).  Defaults to
        :data:`ACTION_KIND_PRIMITIVE` when the harness omits it.

    summarize_state(state) -> str
        Return a compact ``key=value`` summary string when the harness
        wants to override the schema-driven default.  Defaults to
        ``state.schema.compact_summary()`` when omitted.
    """

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        ...

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        ...

    def valid_actions(self, state: HarnessState) -> List[str]:
        ...


# ──────────────────────────────────────────────────────────────────────
# Helpers (used by reference impls)
# ──────────────────────────────────────────────────────────────────────


def parse_op_call(action: str) -> Tuple[str, str]:
    """Split an ``OP(arg)`` action string into ``(op, arg)``.

    Returns ``(op_uppercased, arg_stripped)`` — or ``("", action)``
    when *action* doesn't have the ``OP(arg)`` shape.  Used by the
    VR / Video harnesses to dispatch ``LOOK(scene)`` ↦ scratchpad
    write etc.

    Tolerates whitespace, trailing punctuation, and missing closing
    parens (the LLM occasionally drops the ``)``).  The match is on
    the first ``(`` so multi-arg actions like
    ``COMPARE(e1,e2,color)`` come back as ``("COMPARE", "e1,e2,color")``.
    """
    if not action:
        return "", ""
    s = action.strip().rstrip(".;,")
    paren = s.find("(")
    if paren <= 0:
        return "", s
    op = s[:paren].strip().upper()
    rest = s[paren + 1:]
    if rest.endswith(")"):
        rest = rest[:-1]
    return op, rest.strip()


__all__ = [
    "Harness",
    "HarnessState",
    "parse_op_call",
    "ACTION_KIND_PRIMITIVE",
    "ACTION_KIND_VR_LOOK",
    "ACTION_KIND_VR_RETRIEVE",
    "ACTION_KIND_VR_NOTE",
    "ACTION_KIND_VR_ANSWER",
    "ACTION_KIND_VIDEO_NEXT_FRAME",
    "ACTION_KIND_VIDEO_JUMP",
    "ACTION_KIND_VIDEO_FOCUS",
    "ACTION_KIND_VIDEO_TRACK",
]
