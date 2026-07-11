"""WebShop NL wrapper for the COS-PLAY co-evolution loop.

Wraps the WebShop text environment (``web_agent_site.envs.WebAgentTextEnv``)
into the standard COS-PLAY Gymnasium-like NL interface::

    obs_nl, info = env.reset()
    obs_nl, reward, terminated, truncated, info = env.step("click[buy now]")

``info["action_names"]`` is populated on every step so the decision agent
knows which actions are available (search, click on specific elements).

``info["structured_state"]`` provides predicate-checkable state keys
compatible with ``decision_agents.protocol_utils.check_predicate``.

``info["effect_tags"]`` provides per-step effect tags from the canonical
``TASK_EFFECT_SUBSET["webshop"]`` vocabulary for the skill-selection LoRA.

WebShop repo must be on ``sys.path`` (e.g. ``/workspace/WebShop``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_WEBSHOP_ROOT = os.environ.get("WEBSHOP_ROOT", "/workspace/WebShop")


def _ensure_webshop_on_path() -> None:
    if _WEBSHOP_ROOT not in sys.path:
        sys.path.insert(0, _WEBSHOP_ROOT)


class WebShopNLWrapper:
    """COS-PLAY NL wrapper around ``WebAgentTextEnv``.

    Parameters
    ----------
    num_products : int
        Number of products to load (use a small number for fast init).
    observation_mode : str
        ``"text"`` (NL) or ``"html"`` (raw HTML).
    max_steps : int
        Maximum steps per episode.
    human_goals : bool
        Whether to use human-written goals (more diverse but slower init).
    """

    def __init__(
        self,
        num_products: Optional[int] = None,
        observation_mode: str = "text",
        max_steps: int = 50,
        human_goals: bool = False,
    ):
        _ensure_webshop_on_path()
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        kwargs: Dict[str, Any] = {
            "observation_mode": observation_mode,
            # Always pass explicitly: SkillRL small-env default is synthetic
            # goals (human_goals=False). Human goals only cover ~13/1000
            # products in the shipped items_shuffle_1000 subset.
            "human_goals": bool(human_goals),
        }
        if num_products is not None:
            kwargs["num_products"] = num_products

        self._env = WebAgentTextEnv(**kwargs)
        self._max_steps = max_steps
        self._step_count = 0
        self._done = False
        self._goal = ""
        self._goal_idx: Optional[int] = None
        self._action_history: List[str] = []
        self._cumulative_reward = 0.0
        self._prev_obs = ""

    @property
    def num_goals(self) -> int:
        return len(self._env.server.goals)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        goal_idx: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Reset the env.

        If ``goal_idx`` is set, force that goal from ``server.goals`` (stable
        across paired runs when the goal list is built with the same RNG seed).
        Uses a unique session id so repeated indices do not reuse stale state.
        """
        if goal_idx is not None:
            goals = self._env.server.goals
            if goal_idx < 0 or goal_idx >= len(goals):
                raise IndexError(
                    f"goal_idx={goal_idx} out of range [0, {len(goals)})"
                )
            session = f"g{goal_idx}-{self._step_count}-{os.getpid()}"
            # Pre-register so SimServer.receive keeps this goal for the session.
            self._env.server.user_sessions[session] = {
                "goal": goals[goal_idx],
                "done": False,
            }
            self._env.reset(session=session)
            self._goal_idx = goal_idx
        else:
            self._env.reset()
            self._goal_idx = None

        self._step_count = 0
        self._done = False
        self._action_history = []
        self._cumulative_reward = 0.0

        obs_nl = self._env.observation
        self._goal = self._env.get_instruction_text()
        self._prev_obs = obs_nl

        available = self._get_action_names()
        page_type = self._detect_page_type(obs_nl)

        from decision_agents.protocol_utils import compute_webshop_structured_state
        structured = compute_webshop_structured_state(
            goal=self._goal,
            page_type=page_type,
            step=0,
            action_history=[],
            cumulative_reward=0.0,
        )

        info: Dict[str, Any] = {
            "action_names": available,
            "env_name": "webshop",
            "game_name": "webshop",
            "goal": self._goal,
            "goal_idx": self._goal_idx,
            "structured_state": structured,
            "effect_tags": {"state_observed": "true"},
        }
        obs_with_goal = f"Goal: {self._goal}\n\n{obs_nl}"
        return obs_with_goal, info

    def step(
        self, action: str,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            return "", 0.0, True, False, {"action_names": [], "effect_tags": {}}

        self._step_count += 1
        self._action_history.append(action)

        try:
            state, reward, done, env_info = self._env.step(action)
        except Exception as exc:
            logger.warning("WebShop step error: %s", exc)
            state = self._env.observation if hasattr(self._env, "observation") else ""
            reward = 0.0
            done = False

        obs_nl = self._env.observation if hasattr(self._env, "observation") else state
        self._cumulative_reward += reward

        truncated = self._step_count >= self._max_steps and not done
        terminated = bool(done)
        self._done = terminated or truncated

        available = self._get_action_names() if not self._done else []
        page_type = self._detect_page_type(obs_nl) if not self._done else "done"

        from decision_agents.protocol_utils import (
            compute_webshop_effects,
            compute_webshop_structured_state,
        )
        effects = compute_webshop_effects(
            action=action,
            prev_obs=self._prev_obs,
            curr_obs=obs_nl,
            reward=reward,
        )
        structured = compute_webshop_structured_state(
            goal=self._goal,
            page_type=page_type,
            step=self._step_count,
            action_history=self._action_history,
            cumulative_reward=self._cumulative_reward,
        )
        self._prev_obs = obs_nl

        info: Dict[str, Any] = {
            "action_names": available,
            "env_name": "webshop",
            "game_name": "webshop",
            "goal": self._goal,
            "structured_state": structured,
            "effect_tags": effects,
            "raw_env_reward": float(reward),
        }
        obs_with_goal = f"Goal: {self._goal}\n\n{obs_nl}"
        return obs_with_goal, float(reward), terminated, truncated, info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def _get_action_names(self) -> List[str]:
        """Extract available actions from the WebShop environment."""
        try:
            avail = self._env.get_available_actions()
        except Exception:
            return ["search[query]"]

        actions: List[str] = []
        if avail.get("has_search_bar"):
            actions.append("search[query]")
        clickables = avail.get("clickables", [])
        for c in clickables:
            actions.append(f"click[{c}]")
        if not actions:
            actions.append("search[query]")
        return actions

    @staticmethod
    def _detect_page_type(obs: str) -> str:
        obs_lower = obs.lower() if obs else ""
        if "search" in obs_lower and "results" not in obs_lower and "back to search" not in obs_lower:
            return "search"
        elif "back to search" in obs_lower or "next >" in obs_lower:
            return "results"
        elif "price" in obs_lower or "buy now" in obs_lower:
            return "product"
        return "unknown"


def make_webshop_env(
    max_steps: int = 50,
    num_products: Optional[int] = None,
) -> WebShopNLWrapper:
    """Factory function matching the COS-PLAY convention."""
    return WebShopNLWrapper(
        max_steps=max_steps,
        num_products=num_products,
        observation_mode="text",
    )
