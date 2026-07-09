"""
GamingAgent (LMGame-Bench) env wrapper: state <-> natural language.

Wraps GamingAgentEnv (from GamingAgent/gym_like) so that:
- Observation is the textual state as natural language (obs["text"]).
- step() accepts string actions (e.g. "up", "left", "hard_drop").

The underlying GamingAgentEnv already provides text observations and accepts
string actions; this wrapper presents a consistent NL interface for the
dummy_agent and other language-model policies.

Usage:

    from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
    from GamingAgent.gym_like import make_gaming_env

    base_env = make_gaming_env("sokoban", max_steps=200)
    env = GamingAgentNLWrapper(base_env)
    obs, info = env.reset()   # obs: str (text state)
    obs, reward, term, trunc, info = env.step("push up")
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Compact structured state summary (for agent context / retrieval)
# ---------------------------------------------------------------------------

_GAME_HINTS = [
    (re.compile(r"sokoban|box|wall|goal|push", re.I), "sokoban"),
    (re.compile(r"maze|corridor|exit", re.I), "maze"),
    (re.compile(r"tetris|block|rotate|drop", re.I), "tetris"),
    (re.compile(r"2048|twenty.?forty.?eight|highest.?tile", re.I), "2048"),
    (re.compile(r"grid|tile|cell", re.I), "grid_game"),
]


def _detect_game_hint(text: str) -> str:
    """Return a short game name hint from observation text."""
    for pat, name in _GAME_HINTS:
        if pat.search(text):
            return name
    return "text_game"


def _extract_clauses(text: str, limit: int = 6) -> List[str]:
    """Split *text* into short informative clauses (up to *limit*)."""
    clauses = re.split(r"[.\n|]+", text)
    clauses = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 2]
    clauses = [c for c in clauses if not re.match(r"^[-=]{3,}", c)]
    return clauses[:limit]


# ---------------------------------------------------------------------------
# Numpy-string parsing helpers (for games that emit np.uint8/np.int64 etc.)
# ---------------------------------------------------------------------------

_NUMPY_RE = re.compile(r"np\.(?:u?int|float)\d*\(([0-9.eE+-]+)\)")


def _clean_numpy_str(text: str) -> str:
    """Replace ``np.uint8(2)`` style wrappers with plain Python literals."""
    return _NUMPY_RE.sub(r"\1", text)


def _try_parse_dict(text: str) -> Optional[dict]:
    """Try to parse a dict-like string after cleaning numpy wrappers."""
    cleaned = _clean_numpy_str(text)
    m = re.search(r"\{.+\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(0))
    except (ValueError, SyntaxError):
        return None


# ---------------------------------------------------------------------------
# 2048-specific helpers
# ---------------------------------------------------------------------------

def _compact_board(board: list) -> str:
    """Compact 2D board: ``'0,0,0,0/0,0,0,0/2,0,0,0/0,0,0,2'``."""
    return "/".join(",".join(str(int(c)) for c in row) for row in board)


def _count_merges(board: list) -> int:
    """Count adjacent same-value non-zero pairs (merge opportunities)."""
    count = 0
    for r in range(len(board)):
        for c in range(len(board[r])):
            v = int(board[r][c])
            if v == 0:
                continue
            if c + 1 < len(board[r]) and int(board[r][c + 1]) == v:
                count += 1
            if r + 1 < len(board) and int(board[r + 1][c]) == v:
                count += 1
    return count


def _2048_phase(highest: int, empty: int) -> str:
    """Matches phase_detector._extract_2048_phases logic."""
    occupancy = 1.0 - (empty / 16.0)
    if occupancy < 0.35 and highest <= 32:
        return "opening"
    if occupancy > 0.7 or highest >= 256:
        return "endgame"
    return "midgame"


def _build_2048_summary(
    parsed: dict,
    step: int,
    action_names: Optional[List[str]],
    last_reward: Optional[float],
) -> dict:
    """Build structured state for a 2048 board observation."""
    board = parsed["board"]
    highest = int(parsed.get("highest_tile", 0))
    if highest == 0:
        highest = max(int(c) for row in board for c in row)
    empty = sum(1 for row in board for c in row if int(c) == 0)
    merges = _count_merges(board)

    flat = [int(c) for row in board for c in row]
    max_count = flat.count(highest) if highest > 0 else 0

    summary: dict = {
        "game": "2048",
        "board": _compact_board(board),
        "max_tile": highest,
        "empty": empty,
        "merges": merges,
        "max_count": max_count,
        "phase": _2048_phase(highest, empty),
        "step": step,
    }
    if last_reward is not None:
        summary["reward"] = str(last_reward)
    if action_names:
        summary["affordance"] = ",".join(action_names[:8])
    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_structured_state_summary(
    obs: Union[Dict[str, Any], str],
    step: int = 0,
    action_names: Optional[List[str]] = None,
    last_reward: Optional[float] = None,
) -> dict:
    """Build a compact structured dict for a GamingAgent observation.

    Designed to be fed into ``compact_structured_state()`` from
    ``decision_agents.agent_helper``.

    Returns:
        Dict with short key=value-friendly fields.  Example::

            {"game": "2048", "board": "0,0,0,0/0,0,2,0/...",
             "max_tile": 16, "empty": 12, "merges": 2, "step": 5}
    """
    if isinstance(obs, str):
        raw_text = obs
    elif isinstance(obs, dict):
        raw_text = str(obs.get("text") or obs.get("textual_representation", ""))
        if not raw_text:
            raw_text = str(obs)
    else:
        raw_text = str(obs)

    # --- Try structured parsing for games with dict-like observations ---
    parsed = None
    if isinstance(obs, dict) and "board" in obs:
        parsed = {k: (v if not hasattr(v, 'item') else v.item())
                  for k, v in obs.items()}
    elif raw_text:
        parsed = _try_parse_dict(raw_text)

    if parsed and "board" in parsed:
        return _build_2048_summary(parsed, step, action_names, last_reward)

    # --- Generic text-based extraction (non-2048 games) ---
    game_hint = _detect_game_hint(raw_text)
    clauses = _extract_clauses(raw_text)

    summary: dict = {"game": game_hint, "step": step}

    objective_clause = ""
    self_clause = ""
    critical_clause = ""
    progress_clause = ""

    for c in clauses:
        cl = c.lower()
        if not objective_clause and any(
            w in cl for w in ("goal", "objective", "task", "target", "deliver", "push")
        ):
            objective_clause = c[:60]
        elif not self_clause and any(
            w in cl for w in ("you", "player", "agent", "position", "at (")
        ):
            self_clause = c[:60]
        elif not critical_clause and any(
            w in cl for w in ("enemy", "hazard", "obstacle", "pot", "box", "wall", "door", "key")
        ):
            critical_clause = c[:60]
        elif not progress_clause and any(
            w in cl for w in ("score", "progress", "step", "remaining", "left", "complete")
        ):
            progress_clause = c[:60]

    if not self_clause and clauses:
        self_clause = clauses[0][:60]

    if self_clause:
        summary["self"] = self_clause
    if objective_clause:
        summary["objective"] = objective_clause
    if critical_clause:
        summary["critical"] = critical_clause
    if progress_clause:
        summary["progress"] = progress_clause
    if last_reward is not None:
        summary["reward"] = str(last_reward)
    if action_names:
        summary["affordance"] = ",".join(action_names[:8])

    return summary


def state_to_natural_language(obs: Union[Dict[str, Any], str]) -> str:
    """
    Convert GamingAgent observation to natural language string.

    Args:
        obs: Observation from GamingAgentEnv (dict with "text", "screen", "raw")
             or already a string.

    Returns:
        Natural language state description for LLM consumption.
    """
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        text = obs.get("text") or obs.get("textual_representation", "")
        if text:
            return str(text)
        if "screen" in obs and obs["screen"] is not None:
            return "You see the game screen. (Text representation not available.)"
        return "Game state. (No text representation.)"
    return str(obs)


class GamingAgentNLWrapper:
    """
    Wraps GamingAgentEnv so observations are natural language strings
    and step() accepts string actions.

    Compatible with agents.dummy_agent.language_agent_action when game=GAME_GAMINGAGENT.
    """

    def __init__(
        self,
        env: Any,
        include_action_hint: bool = True,
        game_name: Optional[str] = None,
    ):
        """
        Args:
            env: GamingAgentEnv (or any env with obs["text"] and string-action step).
            include_action_hint: If True, append valid action names to the NL state.
            game_name: Specific game identifier (e.g. "sokoban", "2048").
                       Auto-detected from observation text if not provided.
        """
        self._env = env
        self._include_action_hint = include_action_hint
        self._game_name: Optional[str] = game_name
        self._action_names: List[str] = []
        self._step_count = 0

    @property
    def env(self):
        return self._env

    def _obs_to_nl(self, obs: Union[Dict[str, Any], str], info: Dict[str, Any]) -> str:
        nl = state_to_natural_language(obs)
        if self._include_action_hint and self._action_names:
            nl += f"\n\nValid actions: {', '.join(self._action_names)}. Choose one."
        return nl

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._last_reward: Optional[float] = None
        self._action_names = info.get("action_names", getattr(self._env, "action_names", []))
        nl = self._obs_to_nl(obs, info)
        info["state_natural_language"] = nl
        info["action_names"] = self._action_names
        info["structured_state"] = build_structured_state_summary(
            obs, step=0, action_names=self._action_names,
        )
        detected = self._game_name or info["structured_state"].get("game", "text_game")
        info["env_name"] = "gamingagent"
        info["game_name"] = detected
        return nl, info

    def step(
        self,
        action: Union[str, int, np.integer],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        self._last_reward = float(reward)
        self._action_names = info.get("action_names", self._action_names)
        nl = self._obs_to_nl(obs, info)
        info["state_natural_language"] = nl
        info["action_names"] = self._action_names
        info["step"] = self._step_count
        info["structured_state"] = build_structured_state_summary(
            obs,
            step=self._step_count,
            action_names=self._action_names,
            last_reward=self._last_reward,
        )
        detected = self._game_name or info["structured_state"].get("game", "text_game")
        info["env_name"] = "gamingagent"
        info["game_name"] = detected
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

    @property
    def action_names(self) -> List[str]:
        return self._action_names or getattr(self._env, "action_names", [])
