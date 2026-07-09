"""
VideoGameBench env wrapper: state <-> natural language.

Wraps VideoGameBenchGBEnv (from videogamebench/gym_like) so that:
- Observation is a natural language description (step, valid actions).
- step() accepts string actions (e.g. "A", "press up", "no-op").

The underlying VideoGameBenchGBEnv returns RGB screen and discrete actions 0..8.
This wrapper presents an NL interface for language-model policies.

Action mapping: 0=no-op, 1=A, 2=B, 3=SELECT, 4=START, 5=RIGHT, 6=LEFT, 7=UP, 8=DOWN.

Usage:

    from env_wrappers.videogamebench_nl_wrapper import VideoGameBenchNLWrapper
    from videogamebench.gym_like import make_videogamebench_env

    base_env = make_videogamebench_env(game="kirby", max_steps=200)
    env = VideoGameBenchNLWrapper(base_env)
    obs, info = env.reset()   # obs: str (NL state)
    obs, reward, term, trunc, info = env.step("A")
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Button order: 0=no-op, 1=A, 2=B, 3=SELECT, 4=START, 5=RIGHT, 6=LEFT, 7=UP, 8=DOWN
VIDEOGAMEBENCH_BUTTON_NAMES = ["no-op", "A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN"]

_ACTION_INDEX_TO_NAME = {i: n for i, n in enumerate(VIDEOGAMEBENCH_BUTTON_NAMES)}
_NL_TO_INDEX: Dict[str, int] = {
    "no-op": 0, "noop": 0, "none": 0, "nothing": 0, "stay": 0, "wait": 0, "hold": 0,
    "a": 1, "press a": 1, "button a": 1,
    "b": 2, "press b": 2, "button b": 2,
    "select": 3, "press select": 3,
    "start": 4, "press start": 4,
    "right": 5, "press right": 5, "move right": 5, "→": 5,
    "left": 6, "press left": 6, "move left": 6, "←": 6,
    "up": 7, "press up": 7, "move up": 7, "↑": 7,
    "down": 8, "press down": 8, "move down": 8, "↓": 8,
}


# ---------------------------------------------------------------------------
# Compact structured state summary (for agent context / retrieval)
# ---------------------------------------------------------------------------

_STALL_WINDOW = 8


def build_structured_state_summary(
    step: int,
    last_action: Optional[int] = None,
    last_reward: float = 0.0,
    recent_actions: Optional[List[int]] = None,
    recent_rewards: Optional[List[float]] = None,
    game_name: str = "videogame",
) -> dict:
    """Build a compact interaction-driven structured dict for VideoGameBench.

    Since raw semantic state is typically unavailable (only screen pixels),
    this summary uses interaction signals: action history, reward trend,
    stagnation detection, and repeated-action patterns.

    Designed to be fed into ``compact_structured_state()`` from
    ``decision_agents.agent_helper``.

    Returns:
        Dict with short key=value-friendly fields.  Example::

            {"game": "kirby", "step": 37, "last_action": "RIGHT",
             "progress": "stall:8", "critical": "repeat:RIGHTx4",
             "reward": "0.0", "objective": "break_stall"}
    """
    summary: dict = {"game": game_name, "step": step}

    if last_action is not None:
        summary["last_action"] = _ACTION_INDEX_TO_NAME.get(last_action, str(last_action))

    # Reward trend
    if recent_rewards:
        last_n = recent_rewards[-_STALL_WINDOW:]
        summary["reward"] = str(last_reward)
        nonzero = sum(1 for r in last_n if r != 0.0)
        if nonzero == 0 and len(last_n) >= 3:
            summary["progress"] = f"stall:{len(last_n)}"
        elif nonzero > 0:
            trend = sum(last_n)
            summary["progress"] = f"trend:{trend:+.1f}({nonzero}nz/{len(last_n)})"
    elif last_reward != 0.0:
        summary["reward"] = str(last_reward)

    # Repeated action detection
    if recent_actions and len(recent_actions) >= 3:
        last_n = recent_actions[-_STALL_WINDOW:]
        if len(set(last_n)) == 1:
            name = _ACTION_INDEX_TO_NAME.get(last_n[0], str(last_n[0]))
            summary["critical"] = f"repeat:{name}x{len(last_n)}"

    # Objective heuristic
    if "stall" in summary.get("progress", ""):
        summary["objective"] = "break_stall"
    elif "repeat" in summary.get("critical", ""):
        summary["objective"] = "vary_action"
    else:
        summary["objective"] = "continue"

    return summary


def state_to_natural_language(
    step: int,
    valid_actions: Optional[List[str]] = None,
) -> str:
    """
    Build natural language state description for VideoGameBench.

    Args:
        step: Current step number.
        valid_actions: Optional list of action names; defaults to all buttons.

    Returns:
        NL string for LLM consumption.
    """
    actions = valid_actions or VIDEOGAMEBENCH_BUTTON_NAMES[1:]  # exclude no-op for brevity
    actions_str = ", ".join(actions)
    return (
        f"You see the game screen. Step {step}.\n\n"
        f"Choose one action: no-op, {actions_str}.\n"
        f"(Examples: no-op, A, B, SELECT, START, RIGHT, LEFT, UP, DOWN)"
    )


def natural_language_to_action_index(text: Union[str, int]) -> int:
    """
    Convert NL action to VideoGameBench action index 0..8.

    Args:
        text: Action as string (e.g. "press A", "up", "no-op") or int 0..8.

    Returns:
        Action index in [0, 8]. Falls back to 0 (no-op) if unparseable.
    """
    if isinstance(text, (int, np.integer)):
        idx = int(text)
        if 0 <= idx <= 8:
            return idx
        return 0

    s = (text or "").strip().lower()
    if not s:
        return 0

    # Direct lookup
    if s in _NL_TO_INDEX:
        return _NL_TO_INDEX[s]

    # "press X", "button X"
    for prefix in ("press ", "button ", "hit "):
        if s.startswith(prefix):
            rest = s[len(prefix):].strip()
            if rest in _NL_TO_INDEX:
                return _NL_TO_INDEX[rest]
            if rest in ("a", "b", "select", "start", "right", "left", "up", "down"):
                return _NL_TO_INDEX.get(rest, 0)

    # First word match
    words = re.split(r"[\s,]+", s)
    first = words[0] if words else ""
    if first in _NL_TO_INDEX:
        return _NL_TO_INDEX[first]

    # Partial match
    for key, idx in _NL_TO_INDEX.items():
        if key and first.startswith(key) or key.startswith(first):
            return idx

    return 0


class VideoGameBenchNLWrapper:
    """
    Wraps VideoGameBenchGBEnv so observations are NL strings
    and step() accepts string or int actions.

    Compatible with agents.dummy_agent.language_agent_action when game=GAME_VIDEOGAMEBENCH.
    """

    def __init__(self, env: Any, game_name: str = "videogame"):
        """
        Args:
            env: VideoGameBenchGBEnv (or any env with step(int) -> obs, reward, term, trunc, info).
            game_name: Short game identifier for structured summaries (e.g. "kirby").
        """
        self._env = env
        self._game_name = game_name
        self._step_count = 0
        self._recent_actions: List[int] = []
        self._recent_rewards: List[float] = []
        self._last_action: Optional[int] = None

    @property
    def env(self):
        return self._env

    def _build_nl_state(self, info: Dict[str, Any]) -> str:
        step = info.get("step", self._step_count)
        return state_to_natural_language(step=step)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._recent_actions = []
        self._recent_rewards = []
        self._last_action = None
        info["step"] = 0
        nl = self._build_nl_state(info)
        info["state_natural_language"] = nl
        info["raw_obs"] = obs
        info["structured_state"] = build_structured_state_summary(
            step=0, game_name=self._game_name,
        )
        info["env_name"] = "videogamebench"
        info["game_name"] = self._game_name
        return nl, info

    def step(
        self,
        action: Union[str, int, np.integer],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, str):
            action_idx = natural_language_to_action_index(action)
        else:
            action_idx = int(action)
        obs, reward, terminated, truncated, info = self._env.step(action_idx)
        self._step_count = info.get("step", self._step_count + 1)
        self._last_action = action_idx
        self._recent_actions.append(action_idx)
        self._recent_rewards.append(float(reward))
        # Keep bounded history
        if len(self._recent_actions) > 16:
            self._recent_actions = self._recent_actions[-16:]
            self._recent_rewards = self._recent_rewards[-16:]
        nl = self._build_nl_state(info)
        info["state_natural_language"] = nl
        info["raw_obs"] = obs
        info["action_index"] = action_idx
        info["structured_state"] = build_structured_state_summary(
            step=self._step_count,
            last_action=self._last_action,
            last_reward=float(reward),
            recent_actions=self._recent_actions,
            recent_rewards=self._recent_rewards,
            game_name=self._game_name,
        )
        info["env_name"] = "videogamebench"
        info["game_name"] = self._game_name
        return nl, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)
