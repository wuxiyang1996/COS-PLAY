"""
Gymnasium-compatible interface for Orak game environments (all 12 games).

Provides make_orak_gaming_env() so evaluation harnesses can create and
interact with Orak environments using the standard Gymnasium API.

The Orak repo must be cloned at ../Orak/ (krafton-ai/Orak release branch).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent
_ORAK_REPO = _CODEBASE_ROOT.parent / "Orak"
_ORAK_SRC = _ORAK_REPO / "src"
_ORAK_MCP_GAMES = _ORAK_SRC / "mcp_game_servers"
_ORAK_MCP_AGENTS = _ORAK_SRC / "mcp_agent_client"

for _p in [str(_ORAK_SRC), str(_ORAK_MCP_GAMES)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate_orak.orak_nl_wrapper import ORAK_GAMES


def _cfg(game: str) -> str:
    p = _ORAK_MCP_AGENTS / "configs" / game / "config.yaml"
    return str(p)


ORAK_GAME_CONFIG_MAPPING: Dict[str, Dict[str, Any]] = {}
for _gname, _gmeta in ORAK_GAMES.items():
    ORAK_GAME_CONFIG_MAPPING[f"orak_{_gname}"] = {
        "config_yaml": _gmeta["config_yaml"],
        "action_names": _gmeta["action_names"],
        "display_name": f"{_gname.replace('_', ' ').title()} (Orak)",
        "genre": _gmeta.get("genre", ""),
    }


def list_orak_games() -> List[str]:
    return sorted(ORAK_GAME_CONFIG_MAPPING.keys())


class _OrakGymWrapper:
    """Wraps an Orak BaseEnv to expose standard Gymnasium semantics."""

    def __init__(
        self,
        orak_env: Any,
        action_names: List[str],
        game_name: str,
        max_steps: int,
    ):
        self._env = orak_env
        self._action_names = action_names
        self._game_name = game_name
        self._max_steps = max_steps
        self._step_count = 0
        self._obs = None

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._step_count = 0

        kw = {}
        if seed is not None:
            kw["seed"] = seed
        if options is not None:
            kw["options"] = options

        if hasattr(self._env, "initial_obs"):
            obs_obj = self._env.initial_obs()
        else:
            obs_obj = self._env.reset(**kw) if kw else self._env.reset()
            if isinstance(obs_obj, tuple):
                obs_obj = obs_obj[0]
        self._obs = obs_obj

        text = self._env.obs2text(obs_obj) or ""
        game_info = self._env.get_game_info() if hasattr(self._env, "get_game_info") else {}

        obs_dict = {"text": text}
        info: Dict[str, Any] = {
            "action_names": self._action_names,
            "game_info": game_info,
        }
        return obs_dict, info

    def step(
        self,
        action: Union[str, int],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        action_str = str(action).strip()

        action_obj = self._env.text2action(action_str)
        result = self._env.step(action_obj)
        obs_obj, reward_raw, terminated, truncated, step_info = (
            result[0], result[1], result[2], result[3], result[4],
        )
        self._obs = obs_obj

        score, done = self._env.evaluate(obs_obj)
        reward = float(score) if score else float(reward_raw or 0)

        if self._step_count >= self._max_steps and not (terminated or truncated):
            truncated = True

        text = self._env.obs2text(obs_obj) or ""
        game_info = self._env.get_game_info() if hasattr(self._env, "get_game_info") else {}

        obs_dict = {"text": text}
        info: Dict[str, Any] = {
            "action_names": self._action_names,
            "game_info": game_info,
            "score": score,
            "done": done,
        }
        return obs_dict, reward, bool(terminated or done), bool(truncated), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def render(self):
        return None


def make_orak_gaming_env(
    game: str,
    max_steps: int = 1000,
    config_override: Optional[str] = None,
) -> _OrakGymWrapper:
    """Create a Gymnasium-compatible Orak game environment.

    Args:
        game: One of list_orak_games() (prefixed with "orak_", e.g. "orak_star_craft").
              Also accepts un-prefixed names (e.g. "star_craft").
        max_steps: Maximum steps per episode before truncation.
        config_override: Optional path to a custom config YAML.
    """
    if game not in ORAK_GAME_CONFIG_MAPPING:
        prefixed = f"orak_{game}"
        if prefixed in ORAK_GAME_CONFIG_MAPPING:
            game = prefixed
        else:
            raise ValueError(f"Unknown Orak game '{game}'. Available: {list_orak_games()}")

    meta = ORAK_GAME_CONFIG_MAPPING[game]
    config_path = config_override or meta["config_yaml"]

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Ensure Orak repo is cloned at {_ORAK_REPO}"
        )

    import omegaconf
    from mcp_game_servers.utils.module_creator import EnvCreator

    cfg = omegaconf.OmegaConf.load(config_path)

    _NO_SCREENSHOT_GAMES = {"orak_pokemon_red", "orak_super_mario"}

    log_dir = str(_CODEBASE_ROOT / "orak_logs" / game)
    os.makedirs(log_dir, exist_ok=True)
    with omegaconf.open_dict(cfg):
        if "env" in cfg:
            cfg.env.log_path = log_dir
            if hasattr(cfg.env, "show_graphic"):
                cfg.env.show_graphic = False
            if game in _NO_SCREENSHOT_GAMES:
                cfg.env.save_screenshots = False
        cfg.log_path = log_dir

    env = EnvCreator(cfg).create()
    action_names = meta["action_names"]

    return _OrakGymWrapper(env, action_names, game, max_steps)
