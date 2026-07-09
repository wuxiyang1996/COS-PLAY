"""
OSWorld NL wrapper: desktop observations → natural language for LLM agents.

Wraps OSWorldGymWrapper so that:
- Observations are natural language strings (not raw screenshot arrays).
- The accessibility tree is parsed into a compact structured summary.
- step() accepts string actions (pyautogui commands or DONE/FAIL/WAIT).

This follows the same pattern as GamingAgentNLWrapper and BrowserGym
adapters in this project, providing a consistent interface for LLM-based
decision agents.

Usage:

    from env_wrappers.osworld_wrapper import OSWorldGymWrapper
    from env_wrappers.osworld_nl_wrapper import OSWorldNLWrapper

    base_env = OSWorldGymWrapper(provider_name="docker", max_steps=15)
    env = OSWorldNLWrapper(base_env)
    obs, info = env.reset()           # obs: str (NL description)
    obs, reward, term, trunc, info = env.step("pyautogui.click(500, 300)")
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accessibility tree → compact text
# ---------------------------------------------------------------------------

def _parse_a11y_tree(raw: str, max_elements: int = 30) -> str:
    """Parse an accessibility tree string into a compact summary.

    Extracts interactive elements (buttons, links, inputs, text fields)
    and their states, producing a concise representation suitable for
    LLM context windows.
    """
    if not raw or not raw.strip():
        return ""

    lines = raw.strip().split("\n")
    elements: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Common a11y tree patterns: [role] "name" {state}
        role_match = re.match(
            r'\[(\w+)\]\s*["\']?([^"\']*)["\']?\s*(?:\{([^}]*)\})?', stripped
        )
        if role_match:
            role, name, state = role_match.groups()
            name = name.strip()[:50]
            parts = [f"{role}: {name}" if name else role]
            if state:
                parts.append(f"({state.strip()})")
            elements.append(" ".join(parts))
        elif len(stripped) > 5 and not stripped.startswith("---"):
            elements.append(stripped[:80])

        if len(elements) >= max_elements:
            break

    return "\n".join(elements)


def _estimate_screen_regions(a11y_text: str) -> Dict[str, str]:
    """Heuristic extraction of screen region descriptions from a11y tree."""
    regions: Dict[str, str] = {}

    lower = a11y_text.lower()
    if any(w in lower for w in ("menu bar", "menubar", "menu_bar")):
        regions["menu"] = "Menu bar present"
    if any(w in lower for w in ("toolbar", "tool_bar", "tool bar")):
        regions["toolbar"] = "Toolbar visible"
    if any(w in lower for w in ("dialog", "modal", "popup")):
        regions["dialog"] = "Dialog/modal open"
    if any(w in lower for w in ("status bar", "statusbar")):
        regions["status"] = "Status bar present"

    return regions


# ---------------------------------------------------------------------------
# Structured state summary for retrieval / skill matching
# ---------------------------------------------------------------------------

def build_osworld_state_summary(
    obs: Dict[str, Any],
    step: int = 0,
    last_action: str | None = None,
    last_reward: float | None = None,
) -> Dict[str, Any]:
    """Build a compact structured state dict from an OSWorld observation.

    Designed to parallel ``build_structured_state_summary`` in
    ``gamingagent_nl_wrapper`` for consistency across environments.
    """
    instruction = obs.get("instruction", "")
    a11y = obs.get("accessibility_tree", "")
    terminal = obs.get("terminal", "")
    has_screenshot = obs.get("screenshot") is not None

    summary: Dict[str, Any] = {
        "env": "osworld",
        "domain": "desktop",
        "step": step,
        "instruction": instruction[:100] if instruction else "",
        "has_screenshot": has_screenshot,
    }

    if a11y:
        regions = _estimate_screen_regions(a11y)
        if regions:
            summary["screen_regions"] = regions
        n_elements = len(re.findall(r'\[(\w+)\]', a11y))
        summary["ui_element_count"] = n_elements
        summary["has_dialog"] = "dialog" in regions

    if terminal:
        terminal_lines = terminal.strip().split("\n")
        summary["terminal_lines"] = len(terminal_lines)
        if terminal_lines:
            summary["terminal_last"] = terminal_lines[-1][:60]

    if last_action is not None:
        summary["last_action"] = str(last_action)[:60]
    if last_reward is not None:
        summary["reward"] = last_reward

    return summary


# ---------------------------------------------------------------------------
# Observation → natural language
# ---------------------------------------------------------------------------

def obs_to_natural_language(
    obs: Dict[str, Any],
    *,
    include_a11y: bool = True,
    include_terminal: bool = True,
    max_a11y_elements: int = 30,
) -> str:
    """Convert an OSWorld observation dict to a natural language string.

    Parameters
    ----------
    obs : dict
        Observation from OSWorldGymWrapper with keys: screenshot,
        accessibility_tree, terminal, instruction.
    include_a11y : bool
        Include parsed accessibility tree in the output.
    include_terminal : bool
        Include terminal output in the output.
    max_a11y_elements : int
        Maximum number of a11y elements to include.

    Returns
    -------
    str : Natural language description of the current desktop state.
    """
    parts: list[str] = []

    instruction = obs.get("instruction", "")
    if instruction:
        parts.append(f"Task: {instruction}")

    has_screenshot = obs.get("screenshot") is not None
    if has_screenshot:
        screenshot = obs["screenshot"]
        if isinstance(screenshot, np.ndarray) and screenshot.ndim >= 2:
            h, w = screenshot.shape[:2]
            parts.append(f"Screen: {w}x{h} desktop screenshot visible.")
        else:
            parts.append("Screen: Desktop screenshot visible.")

    a11y = obs.get("accessibility_tree", "")
    if include_a11y and a11y:
        parsed = _parse_a11y_tree(a11y, max_elements=max_a11y_elements)
        if parsed:
            parts.append(f"UI Elements:\n{parsed}")

    terminal = obs.get("terminal", "")
    if include_terminal and terminal:
        trimmed = terminal.strip()
        if len(trimmed) > 500:
            trimmed = trimmed[-500:]
            trimmed = "...\n" + trimmed
        if trimmed:
            parts.append(f"Terminal Output:\n{trimmed}")

    if not parts:
        return "Desktop environment state. (No details available.)"

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The NL wrapper class
# ---------------------------------------------------------------------------

_ACTION_HELP = (
    "You can issue pyautogui commands (e.g. pyautogui.click(x, y), "
    "pyautogui.typewrite('text'), pyautogui.hotkey('ctrl', 'c')) "
    "or special actions: DONE (task complete), FAIL (task impossible), "
    "WAIT (pause and observe)."
)


class OSWorldNLWrapper:
    """Natural-language observation wrapper for OSWorld.

    Converts screenshot + accessibility tree observations into text
    descriptions suitable for LLM-based agents. Follows the same
    interface pattern as GamingAgentNLWrapper.

    Parameters
    ----------
    env : OSWorldGymWrapper
        The base Gymnasium-wrapped OSWorld environment.
    include_a11y : bool
        Include parsed accessibility tree in NL observations.
    include_terminal : bool
        Include terminal output in NL observations.
    include_action_hint : bool
        Append available action instructions to observations.
    max_a11y_elements : int
        Max UI elements to include from accessibility tree.
    """

    def __init__(
        self,
        env: Any,
        include_a11y: bool = True,
        include_terminal: bool = True,
        include_action_hint: bool = True,
        max_a11y_elements: int = 30,
    ):
        self._env = env
        self._include_a11y = include_a11y
        self._include_terminal = include_terminal
        self._include_action_hint = include_action_hint
        self._max_a11y_elements = max_a11y_elements
        self._step_count = 0
        self._last_action: str | None = None
        self._last_reward: float | None = None

    @property
    def env(self):
        """The underlying OSWorldGymWrapper."""
        return self._env

    @property
    def unwrapped(self):
        """Unwrap to the native DesktopEnv."""
        if hasattr(self._env, "unwrapped"):
            return self._env.unwrapped
        return self._env

    def _obs_to_nl(self, obs: Dict[str, Any]) -> str:
        nl = obs_to_natural_language(
            obs,
            include_a11y=self._include_a11y,
            include_terminal=self._include_terminal,
            max_a11y_elements=self._max_a11y_elements,
        )
        if self._include_action_hint:
            nl += f"\n\n{_ACTION_HELP}"
        return nl

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Reset and return NL observation string.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Forwarded to OSWorldGymWrapper.reset().

        Returns
        -------
        observation : str
            Natural language description of the desktop state.
        info : dict
            Task metadata plus ``state_natural_language``,
            ``structured_state``, and ``raw_obs`` keys.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._last_action = None
        self._last_reward = None

        nl = self._obs_to_nl(obs)

        info["state_natural_language"] = nl
        info["raw_obs"] = obs
        info["structured_state"] = build_osworld_state_summary(obs, step=0)
        info["env_name"] = "osworld"
        info["action_type"] = "pyautogui"

        return nl, info

    def step(
        self,
        action: str | Dict[str, Any],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Execute action and return NL observation.

        Parameters
        ----------
        action : str or dict
            Pyautogui command string, or DONE/FAIL/WAIT.

        Returns
        -------
        observation : str
            Natural language description of the resulting state.
        reward : float
            Reward signal (evaluation score on DONE).
        terminated : bool
            True if agent sent DONE or FAIL.
        truncated : bool
            True if max_steps reached.
        info : dict
            Step metadata.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        self._last_action = str(action) if not isinstance(action, str) else action
        self._last_reward = reward

        nl = self._obs_to_nl(obs)

        info["state_natural_language"] = nl
        info["raw_obs"] = obs
        info["step"] = self._step_count
        info["structured_state"] = build_osworld_state_summary(
            obs,
            step=self._step_count,
            last_action=self._last_action,
            last_reward=self._last_reward,
        )
        info["env_name"] = "osworld"
        info["action_type"] = "pyautogui"

        return nl, reward, terminated, truncated, info

    def evaluate(self) -> float:
        """Manually trigger task evaluation."""
        return self._env.evaluate()

    def render(self, mode: str = "rgb_array"):
        return self._env.render(mode=mode)

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_names(self) -> List[str]:
        return getattr(self._env, "action_names", ["DONE", "FAIL", "WAIT"])

    @property
    def tasks(self):
        return getattr(self._env, "tasks", [])

    @property
    def current_task(self):
        return getattr(self._env, "current_task", None)

    def __repr__(self) -> str:
        return f"OSWorldNLWrapper({self._env!r})"
