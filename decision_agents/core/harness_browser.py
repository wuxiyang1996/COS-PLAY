"""Browser harness — BrowserGym / Playwright action vocabulary.

Phase 1 status: ``valid_actions`` is **real** (it derives concrete
``click(bid) / type(bid, str) / scroll / goto / key / ANSWER`` actions
from the parsed ``<state>`` schema's interactive entities).  ``step``
is a stub that raises :class:`NotImplementedError` until the env
wrapper is plumbed in — see ``plans/02-action-agent/PLAN-ACTION-AGENT.md``.

This split lets the actor exercise the unified-MDP loop end-to-end
against the browser even before a real backend exists: a
``(obs, info)`` pair carrying a schema is enough to enumerate actions
and call the LLM, and tests can mock ``step`` per-case.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from decision_agents.core.harness import (
    ACTION_KIND_PRIMITIVE,
    Harness,
    HarnessState,
)
from decision_agents.schema_parser import StateSchema

_LOGGER = logging.getLogger(__name__)

MAX_VALID_ACTIONS_IN_PROMPT: int = 16
"""Cap matches :data:`decision_agents.actor_agent.MAX_VALID_ACTIONS_IN_PROMPT`."""

# Affordance → action template.  When an entity advertises one of these
# affordances (via the schema's ``e1.affords=[click,type,scroll]``
# block), the harness emits the matching action pre-filled with the
# entity's ``bid``.  Anything not listed here gets dropped silently.
_AFFORDANCE_TEMPLATES: Dict[str, str] = {
    "click": "click({bid})",
    "select": "click({bid})",
    "tap": "click({bid})",
    "type": 'type({bid}, "<text>")',
    "input": 'type({bid}, "<text>")',
    "scroll": "scroll({bid}, 0, 400)",
    "navigate": "goto({bid})",
    "open": "goto({bid})",
}


class BrowserHarness(Harness):
    """BrowserGym-shaped harness.

    Parameters
    ----------
    env
        Optional underlying BrowserGym / Playwright env.  When ``None``,
        ``step`` raises :class:`NotImplementedError`; the rest of the
        harness still works (schema-driven ``valid_actions``).
    answer_action
        Token used to terminate browser tasks where the goal is to
        return a textual answer (e.g. WebArena Q&A pages).  Always
        appended to the action list with placeholder text.
    """

    def __init__(self, env: Optional[Any] = None, *, answer_action: str = 'ANSWER("<text>")') -> None:
        self.env = env
        self.answer_action = answer_action

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        if self.env is None:
            return "", {"warning": "BrowserHarness has no env wired"}
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, dict(info or {})

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if self.env is None:
            raise NotImplementedError(
                "BrowserHarness.step requires a Playwright/BrowserGym env; "
                "wire one via BrowserHarness(env=...) once the env layer "
                "is plugged in (see PLAN-ACTION-AGENT §6 Phase 2)."
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
                f"BrowserHarness expected a 4- or 5-tuple from env.step, got {type(out)}"
            )
        return obs, float(reward or 0.0), done, dict(info or {})

    # ── action enumeration ───────────────────────────────────────────

    def valid_actions(self, state: HarnessState) -> List[str]:
        """Derive a concrete action list from the BrowserGym schema.

        Walks the schema's entities, emits one action per advertised
        affordance, then appends ``key("Enter")`` and the ``ANSWER``
        sentinel.  Falls back to a fixed core vocabulary when no
        schema is available so the actor still has something to pick.
        """
        actions: List[str] = []
        seen: set[str] = set()

        if state.schema is not None and state.schema.entities:
            for eid in state.schema.entity_order:
                ent = state.schema.entities.get(eid)
                if ent is None or not ent.bid:
                    continue
                for aff in ent.affords:
                    template = _AFFORDANCE_TEMPLATES.get(aff.lower())
                    if template is None:
                        continue
                    rendered = template.format(bid=ent.bid)
                    if rendered not in seen:
                        seen.add(rendered)
                        actions.append(rendered)

        # Always-available chrome / fallback.
        for fallback in (
            'key("Enter")',
            'key("Tab")',
            "scroll(0, 400)",
            "scroll(0, -400)",
            self.answer_action,
        ):
            if fallback not in seen:
                seen.add(fallback)
                actions.append(fallback)

        return actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    # ── optional cost lookup ─────────────────────────────────────────

    def action_kind(self, action: str) -> str:
        return ACTION_KIND_PRIMITIVE


__all__ = ["BrowserHarness"]
