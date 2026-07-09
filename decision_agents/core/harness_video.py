"""Video-understanding harness — frame cursor + scratchpad ops.

Extends :class:`VRHarness` with a frame-cursor ``t`` and the temporal
ops ``NEXT_FRAME / JUMP / WINDOW / FOCUS / TRACK`` on top of the same
scratchpad-mutating ops (``LOOK / RETRIEVE / NOTE / ANSWER``).  The
clip is read-only; only the cursor and the scratchpad change.

The harness stores the clip as a sequence of :class:`VisualInput`
frames so the actor's existing multimodal builders work unchanged —
``valid_actions`` reads ``len(frames)`` to bound ``JUMP`` arguments,
and ``current_frame`` exposes the frame at the cursor for the prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decision_agents.core.harness import (
    ACTION_KIND_VIDEO_FOCUS,
    ACTION_KIND_VIDEO_JUMP,
    ACTION_KIND_VIDEO_NEXT_FRAME,
    ACTION_KIND_VIDEO_TRACK,
    HarnessState,
    parse_op_call,
)
from decision_agents.core.harness_vr import VR_OPS, VRHarness
from decision_agents.core.multimodal import VisualInput

_LOGGER = logging.getLogger(__name__)

MAX_VALID_ACTIONS_IN_PROMPT: int = 16


# Temporal ops added on top of :data:`VR_OPS`.
VIDEO_OPS: Tuple[str, ...] = (
    "NEXT_FRAME",
    "JUMP",
    "WINDOW",
    "FOCUS",
    "TRACK",
)


class VideoHarness(VRHarness):
    """Read-only video clip + frame cursor + reasoning scratchpad.

    Parameters
    ----------
    frames
        Ordered list of :class:`VisualInput` frames.  May be empty for
        unit tests; the harness still runs but the cursor stays at 0.
    question / gold_answer / max_steps / candidate_args
        Same semantics as :class:`VRHarness`.
    """

    def __init__(
        self,
        *,
        frames: Optional[Sequence[VisualInput]] = None,
        question: str = "",
        gold_answer: Optional[str] = None,
        max_steps: int = 16,
        candidate_args: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Pass the first frame (if any) up to VRHarness so its ``image``
        # attribute always points at the *current* frame.  We then
        # override it on cursor moves.
        first = frames[0] if frames else None
        super().__init__(
            image=first,
            question=question,
            gold_answer=gold_answer,
            max_steps=max_steps,
            candidate_args=candidate_args,
        )
        self.frames: List[VisualInput] = list(frames or [])
        self._cursor: int = 0

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        out = super().reset()
        self._cursor = 0
        if self.frames:
            self.image = self.frames[0]
        # Annotate info with cursor / clip length so runners can log.
        obs, info = out
        info["cursor"] = self._cursor
        info["n_frames"] = len(self.frames)
        return obs, info

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        op, arg = parse_op_call(action)

        # Temporal ops — handled here, then exit (no scratchpad mutation
        # beyond an optional NOTE).
        if op in VIDEO_OPS:
            self._t += 1
            reward = 0.0
            info: Dict[str, Any] = {"op": op, "arg": arg, "harness": "VideoHarness"}

            if op == "NEXT_FRAME":
                self._move_cursor(self._cursor + 1)
            elif op == "JUMP":
                target = _safe_int(arg, default=self._cursor)
                self._move_cursor(target)
            elif op == "WINDOW":
                # ``WINDOW(t1,t2)`` → set cursor to t1 and remember the range.
                t1, t2 = _parse_window(arg, default=(self._cursor, self._cursor))
                self._move_cursor(t1)
                self._scratchpad_note(f"WINDOW=[{t1},{t2}]")
            elif op == "FOCUS":
                # ``FOCUS(bbox,t)`` — record on scratchpad; harness
                # doesn't manipulate pixels itself.
                self._scratchpad_note(f"FOCUS({arg})")
            elif op == "TRACK":
                self._scratchpad_note(f"TRACK({arg})")

            done = False
            if self._t >= self.max_steps:
                done = True
                info["truncated"] = True
            self._done = done
            self._last_obs = self.question
            info["cursor"] = self._cursor
            return self._last_obs, reward, done, info

        # Non-temporal op → fall through to VRHarness behaviour.
        obs, reward, done, info = super().step(action)
        info.setdefault("harness", "VideoHarness")
        info["cursor"] = self._cursor
        return obs, reward, done, info

    # ── action enumeration ───────────────────────────────────────────

    def valid_actions(self, state: HarnessState) -> List[str]:
        """VR ops + temporal ops, ordered for prompt economy."""
        actions: List[str] = []
        seen: set[str] = set()

        ans = 'ANSWER("<text>")'
        actions.append(ans)
        seen.add(ans)

        # Temporal ops first — videos usually need cursor motion before
        # any reasoning op makes sense.
        n = max(0, len(self.frames) - 1)
        for rendered in (
            f"NEXT_FRAME()",
            f"JUMP({min(n, self._cursor + 5)})",
            f"WINDOW({self._cursor},{min(n, self._cursor + 8)})",
        ):
            if rendered not in seen:
                seen.add(rendered)
                actions.append(rendered)

        # FOCUS / TRACK: schema-keyed templates when entities exist.
        if state.schema is not None and state.schema.entities:
            for eid in state.schema.entity_order[:2]:
                rendered = f"FOCUS({eid},{self._cursor})"
                if rendered not in seen:
                    seen.add(rendered)
                    actions.append(rendered)
                rendered = f"TRACK({eid})"
                if rendered not in seen:
                    seen.add(rendered)
                    actions.append(rendered)

        # Plus the inherited reasoning ops (LOOK/NOTE/RETRIEVE).
        for fallback in (
            'LOOK(scene)',
            'RETRIEVE("<keywords>")',
            'NOTE("<text>")',
        ):
            if fallback not in seen:
                seen.add(fallback)
                actions.append(fallback)

        return actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    # ── optional cost lookup ─────────────────────────────────────────

    def action_kind(self, action: str) -> str:
        """Temporal ops first, then defer to VR for everything else."""
        op, _ = parse_op_call(action)
        if op == "NEXT_FRAME":
            return ACTION_KIND_VIDEO_NEXT_FRAME
        if op in ("JUMP", "WINDOW"):
            return ACTION_KIND_VIDEO_JUMP
        if op == "FOCUS":
            return ACTION_KIND_VIDEO_FOCUS
        if op == "TRACK":
            return ACTION_KIND_VIDEO_TRACK
        return super().action_kind(action)

    # ── exposed accessors ────────────────────────────────────────────

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def current_frame(self) -> Optional[VisualInput]:
        if not self.frames:
            return None
        idx = max(0, min(self._cursor, len(self.frames) - 1))
        return self.frames[idx]

    # ── private helpers ──────────────────────────────────────────────

    def _move_cursor(self, target: int) -> None:
        if not self.frames:
            self._cursor = 0
            return
        self._cursor = max(0, min(int(target), len(self.frames) - 1))
        self.image = self.frames[self._cursor]


# ──────────────────────────────────────────────────────────────────────
# Arg parsers (kept private — small enough to inline)
# ──────────────────────────────────────────────────────────────────────


def _safe_int(s: str, *, default: int) -> int:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return default


def _parse_window(s: str, *, default: Tuple[int, int]) -> Tuple[int, int]:
    """Parse ``"t1,t2"`` into ``(int, int)`` — tolerant of whitespace."""
    parts = [p.strip() for p in str(s or "").split(",") if p.strip()]
    if len(parts) >= 2:
        return (_safe_int(parts[0], default=default[0]),
                _safe_int(parts[1], default=default[1]))
    if len(parts) == 1:
        t1 = _safe_int(parts[0], default=default[0])
        return (t1, t1)
    return default


__all__ = ["VideoHarness", "VIDEO_OPS"]
