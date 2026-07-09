"""Gym-style harness — wraps any env with a ``reset`` + ``step`` pair.

This is the harness :class:`~decision_agents.actor_agent.ActorAgent`
auto-binds when no explicit ``harness=`` is provided, so the legacy
``run_actor_episode(env, agent, ...)`` path keeps working byte-identical
to the pre-harness era.

What it owns
------------
* ``reset()`` / ``step()`` delegate to the underlying env.  Both
  4-tuple ``(obs, reward, done, info)`` and 5-tuple
  ``(obs, reward, terminated, truncated, info)`` envs are accepted —
  the second case folds ``terminated or truncated`` into a single
  ``done`` bool.
* ``valid_actions`` mirrors the priority that lived in the legacy
  ``_resolve_valid_actions`` (info-dict → schema → empty), so the
  back-compat shim produces an identical action vocabulary.
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
"""Mirrors :data:`decision_agents.actor_agent.MAX_VALID_ACTIONS_IN_PROMPT`
so the back-compat path produces the same truncated list the legacy
``_resolve_valid_actions`` did.  Kept module-local to avoid a circular
import."""


class GymHarness(Harness):
    """Thin wrapper over a Gymnasium-shaped env.

    Parameters
    ----------
    env
        Object with ``reset() -> (obs, info)`` and ``step(action) ->
        (obs, reward, term, trunc, info)`` (or the 4-tuple variant).
        The harness never inspects ``env`` beyond those two methods,
        so any duck-typed stand-in (the test ``_StubEnv``, a
        Gymnasium-wrapped game, etc.) works.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:  # legacy single-return reset
            obs, info = out, {}
        return obs, dict(info or {})

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, term, trunc, info = out
            done = bool(term or trunc)
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            done = bool(done)
        else:
            raise TypeError(
                f"GymHarness expected a 4- or 5-tuple from env.step, got {type(out)}"
            )
        return obs, float(reward or 0.0), done, dict(info or {})

    # ── action enumeration ───────────────────────────────────────────

    def valid_actions(self, state: HarnessState) -> List[str]:
        """Resolve the per-step action vocabulary.

        Priority (mirrors the legacy ``_resolve_valid_actions``):

        1. ``info["valid_actions"]`` / ``info["available_actions"]``
        2. ``state.schema.actions`` from the parsed schema
        3. Empty list (the actor will fall back to ``"no-op"``).
        """
        info = state.info or {}
        candidate = info.get("valid_actions") or info.get("available_actions")
        if candidate:
            return [str(a) for a in candidate][:MAX_VALID_ACTIONS_IN_PROMPT]
        if state.schema is not None and state.schema.actions:
            return list(state.schema.actions[:MAX_VALID_ACTIONS_IN_PROMPT])
        return []

    # ── optional cost lookup (default = primitive) ───────────────────

    def action_kind(self, action: str) -> str:
        """All gym actions share the ``primitive`` cost bucket."""
        return ACTION_KIND_PRIMITIVE


__all__ = ["GymHarness"]
