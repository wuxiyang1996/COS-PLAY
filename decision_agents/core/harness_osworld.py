"""OSWorld harness — desktop primitives + bash + ANSWER.

Phase 1 status: ``valid_actions`` is **real** (a fixed core vocabulary
plus per-entity click templates derived from the parsed
``<state>`` schema).  ``step`` is a stub that raises
:class:`NotImplementedError` until the OSWorld env wrapper is plumbed
in.

The action vocabulary mirrors the one in PLAN-ACTION-AGENT §6 (xdotool
clicks, keyboard input, bash, file reads, and a final ``ANSWER`` for
information-retrieval tasks).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from decision_agents.core.harness import (
    ACTION_KIND_PRIMITIVE,
    Harness,
    HarnessState,
)

_LOGGER = logging.getLogger(__name__)

MAX_VALID_ACTIONS_IN_PROMPT: int = 16


# Core OSWorld vocabulary, always available.  Templates use ``<…>``
# placeholders the LLM is expected to fill in; the multi-strategy
# action parser tolerates that since the harness's ``step`` will
# accept any fully-rendered concrete action that matches the
# ``OP(arg)`` shape.
_CORE_ACTIONS: List[str] = [
    'xdotool("click <x> <y>")',
    'xdotool("key Return")',
    'xdotool("key Tab")',
    'xdotool("key ctrl+s")',
    'type("<text>")',
    'bash("<cmd>")',
    'read("<path>")',
    'screenshot()',
    'ANSWER("<text>")',
]


class OSWorldHarness(Harness):
    """OSWorld-shaped harness.

    Parameters
    ----------
    env
        Optional underlying OSWorld env.  When ``None``, ``step``
        raises :class:`NotImplementedError`; the rest of the harness
        still works (schema-driven ``valid_actions``).
    """

    def __init__(self, env: Optional[Any] = None) -> None:
        self.env = env

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        if self.env is None:
            return "", {"warning": "OSWorldHarness has no env wired"}
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, dict(info or {})

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if self.env is None:
            raise NotImplementedError(
                "OSWorldHarness.step requires an OSWorld env; wire one via "
                "OSWorldHarness(env=...) once the env layer is plugged in "
                "(see PLAN-ACTION-AGENT §6 Phase 3)."
            )
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, term, trunc, info = out
            done = bool(term or trunc)
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            done = bool(done)
        else:
            raise TypeError(
                f"OSWorldHarness expected a 4- or 5-tuple from env.step, got {type(out)}"
            )
        return obs, float(reward or 0.0), done, dict(info or {})

    # ── action enumeration ───────────────────────────────────────────

    def valid_actions(self, state: HarnessState) -> List[str]:
        """Return the fixed core vocab plus per-entity click templates."""
        actions: List[str] = []
        seen: set[str] = set()

        if state.schema is not None and state.schema.entities:
            for eid in state.schema.entity_order:
                ent = state.schema.entities.get(eid)
                if ent is None or not ent.pos:
                    continue
                # Pre-fill the click x,y from the entity's pos centre.
                x = ent.pos[0] + (ent.pos[2] // 2 if len(ent.pos) >= 4 else 0)
                y = ent.pos[1] + (ent.pos[3] // 2 if len(ent.pos) >= 4 else 0)
                rendered = f'xdotool("click {x} {y}")'
                if rendered not in seen:
                    seen.add(rendered)
                    actions.append(rendered)
                if len(actions) >= 6:
                    break  # leave room for the core vocab

        for core in _CORE_ACTIONS:
            if core not in seen:
                seen.add(core)
                actions.append(core)

        return actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    # ── optional cost lookup ─────────────────────────────────────────

    def action_kind(self, action: str) -> str:
        return ACTION_KIND_PRIMITIVE


__all__ = ["OSWorldHarness"]
