"""
Gymnasium-compatible interface for GamingAgent (LMGame-Bench) environments.

Provides make_gaming_env() and list_games() so the evaluate_gamingagent
test harness can create and interact with GamingAgent environments using
the standard Gymnasium API:

    from evaluate_gamingagent.gym_like import make_gaming_env, list_games

    env = make_gaming_env("twenty_forty_eight", max_steps=50)
    obs, info = env.reset()          # obs: dict with "text" key
    obs, reward, term, trunc, info = env.step("up")   # string action

The external GamingAgent repo must be installed or on PYTHONPATH; this
module only imports from it at runtime.
"""

import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent
_GAMINGAGENT_ROOT = _CODEBASE_ROOT.parent / "GamingAgent"

for _p in [str(_CODEBASE_ROOT), str(_GAMINGAGENT_ROOT)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

_ENVS_DIR = str(_GAMINGAGENT_ROOT / "gamingagent" / "envs")

GAME_CONFIG_MAPPING = {
    "twenty_forty_eight": "custom_01_2048",
    "sokoban": "custom_02_sokoban",
    "candy_crush": "custom_03_candy_crush",
    "tetris": "custom_04_tetris",
    "pokemon_red": "custom_06_pokemon_red",
    "tictactoe": "zoo_01_tictactoe",
    "texasholdem": "zoo_02_texasholdem",
}

# Orak benchmark (krafton-ai/Orak, 12 games) are handled via evaluate_orak/
ORAK_GAME_NAMES = [
    "orak_twenty_fourty_eight",
    "orak_baba_is_you",
    "orak_super_mario",
    "orak_street_fighter",
    "orak_star_craft",
    "orak_star_craft_multi",
    "orak_slay_the_spire",
    "orak_pokemon_red",
    "orak_darkest_dungeon",
    "orak_pwaat",
    "orak_her_story",
    "orak_minecraft",
    "orak_stardew_valley",
]


def list_games() -> List[str]:
    """Return the names of games that can be created via make_gaming_env."""
    return sorted(GAME_CONFIG_MAPPING.keys())


def _load_env_config(config_dir: str) -> dict:
    path = os.path.join(_ENVS_DIR, config_dir, "game_env_config.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _action_names_from_config(config: dict) -> List[str]:
    mapping = config.get("action_mapping", {})
    return list(mapping.keys()) if mapping else []


class _GymLikeWrapper:
    """Wraps a native GamingAgent env to expose standard Gymnasium semantics.

    * reset(seed=, options=) -> (obs_dict, info)
    * step(action_str)       -> (obs_dict, reward, terminated, truncated, info)

    obs_dict always contains a "text" key suitable for GamingAgentNLWrapper.
    """

    def __init__(self, native_env: Any, action_names: List[str],
                 game_name: str, max_steps: int,
                 dynamic_actions: bool = False):
        self._env = native_env
        self._action_names = action_names
        self._game_name = game_name
        self._max_steps = max_steps
        self._step_count = 0
        self._episode_id = 0
        self._dynamic_actions = dynamic_actions

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    @staticmethod
    def _obs_to_dict(obs_obj: Any) -> Dict[str, Any]:
        """Convert a native GamingAgent Observation into a plain dict."""
        text = ""
        if hasattr(obs_obj, "textual_representation") and obs_obj.textual_representation:
            text = obs_obj.textual_representation
        elif hasattr(obs_obj, "processed_visual_description") and obs_obj.processed_visual_description:
            text = obs_obj.processed_visual_description
        elif isinstance(obs_obj, dict):
            text = str(obs_obj.get("text", obs_obj))
        else:
            text = str(obs_obj)
        return {"text": text}

    def _resolve_dynamic_actions(self, info: Dict[str, Any]) -> List[str]:
        """For games with dynamic action spaces (e.g. candy_crush), derive
        human-readable action names from the native info dict."""
        if not self._dynamic_actions:
            return self._action_names

        effective_idx = info.get("effective_actions", [])
        idx_to_move = getattr(self._env, "env_action_idx_to_move", {})

        if effective_idx and idx_to_move:
            names = [idx_to_move[i] for i in effective_idx if i in idx_to_move]
            if names:
                return names[:20]

        if idx_to_move:
            return list(idx_to_move.values())[:20]

        return self._action_names

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._step_count = 0
        self._episode_id += 1

        kw: Dict[str, Any] = {"episode_id": self._episode_id}
        if seed is not None:
            kw["seed"] = seed

        obs_obj, info = self._env.reset(**kw)
        obs_dict = self._obs_to_dict(obs_obj)

        resolved = self._resolve_dynamic_actions(info)
        if resolved:
            self._action_names = resolved
        info["action_names"] = self._action_names

        return obs_dict, info

    def step(
        self,
        action: Union[str, int],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        action_str = str(action) if not isinstance(action, str) else action

        result = self._env.step(agent_action_str=action_str)
        obs_obj = result[0]
        reward = result[1]
        terminated = result[2]
        truncated = result[3]
        info = result[4]
        perf_score = result[5] if len(result) > 5 else 0.0

        obs_dict = self._obs_to_dict(obs_obj)

        resolved = self._resolve_dynamic_actions(info)
        if resolved:
            self._action_names = resolved
        info["action_names"] = self._action_names
        info["perf_score"] = perf_score

        if self._step_count >= self._max_steps and not (terminated or truncated):
            truncated = True

        return obs_dict, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
        tmp = getattr(self._env, "_rom_tmp_dir", None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def render(self):
        if hasattr(self._env, "render"):
            return self._env.render()
        return None


def make_gaming_env(
    game: str,
    max_steps: int = 200,
    observation_mode: str = "text",
) -> _GymLikeWrapper:
    """Create a Gymnasium-compatible GamingAgent environment.

    Args:
        game: One of list_games() (e.g. "twenty_forty_eight", "sokoban").
        max_steps: Maximum steps per episode before truncation.
        observation_mode: "text", "vision", or "both".

    Returns:
        A wrapper with standard reset()/step() Gymnasium interface.
    """
    config_dir = GAME_CONFIG_MAPPING.get(game)
    if config_dir is None:
        raise ValueError(f"Unknown game '{game}'. Available: {list_games()}")

    config = _load_env_config(config_dir)
    action_names = _action_names_from_config(config)
    config_path = os.path.join(_ENVS_DIR, config_dir, "game_env_config.json")
    cache_dir = tempfile.mkdtemp(prefix=f"gamingagent_{game}_")
    dynamic_actions = False

    common_adapter_kw = {
        "observation_mode_for_adapter": observation_mode,
        "agent_cache_dir_for_adapter": cache_dir,
        "game_specific_config_path_for_adapter": config_path,
    }

    if game == "twenty_forty_eight":
        from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import (
            TwentyFortyEightEnv,
        )
        init_kw = config.get("env_init_kwargs", {})
        env = TwentyFortyEightEnv(
            render_mode=None,
            size=init_kw.get("size", 4),
            max_pow=init_kw.get("max_pow", 16),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 10
            ),
            **common_adapter_kw,
        )

    elif game == "sokoban":
        from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv
        init_kw = config.get("env_init_kwargs", {})
        env = SokobanEnv(
            render_mode=None,
            dim_room=tuple(init_kw.get("dim_room", [10, 10])),
            max_steps_episode=init_kw.get("max_steps_episode", 200),
            num_boxes=init_kw.get("num_boxes", 3),
            num_gen_steps=init_kw.get("num_gen_steps"),
            level_to_load=config.get("level_to_load"),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 5
            ),
            **common_adapter_kw,
        )

    elif game == "candy_crush":
        from gamingagent.envs.custom_03_candy_crush.candyCrushEnv import (
            CandyCrushEnv,
        )
        init_kw = config.get("env_init_kwargs", {})
        env = CandyCrushEnv(
            num_rows_override=init_kw.get("num_rows", 8),
            num_cols_override=init_kw.get("num_cols", 8),
            num_colours_override=init_kw.get("num_colours", 4),
            num_moves_override=init_kw.get("num_moves", 50),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 50
            ),
            **common_adapter_kw,
        )
        dynamic_actions = True

    elif game == "tetris":
        from gamingagent.envs.custom_04_tetris.tetrisEnv import TetrisEnv
        init_kw = config.get("env_init_kwargs", {})
        env = TetrisEnv(
            render_mode=None,
            board_width=init_kw.get("board_width", 10),
            board_height=init_kw.get("board_height", 20),
            gravity=init_kw.get("gravity", True),
            render_upscale=init_kw.get("render_upscale", 25),
            queue_size=init_kw.get("queue_size", 4),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 30
            ),
            **common_adapter_kw,
        )

    elif game == "pokemon_red":
        from gamingagent.envs.custom_06_pokemon_red.pokemonRedEnv import (
            PokemonRedEnv,
        )
        init_kw = config.get("env_init_kwargs", {})
        rom_path = init_kw.get("rom_path", "")
        if rom_path and not os.path.isabs(rom_path):
            rom_path = str(_GAMINGAGENT_ROOT / rom_path)
        if not os.path.isfile(rom_path):
            raise FileNotFoundError(
                f"Pokemon Red ROM not found at '{rom_path}'. "
                f"Place the .gb ROM at {_GAMINGAGENT_ROOT / 'gamingagent' / 'configs' / 'custom_06_pokemon_red' / 'rom' / 'pokemon.gb'}"
            )
        _tmp_dir = tempfile.mkdtemp(prefix="pokemon_red_")
        _rom_copy = os.path.join(_tmp_dir, os.path.basename(rom_path))
        shutil.copy2(rom_path, _rom_copy)
        atexit.register(shutil.rmtree, _tmp_dir, True)

        env = PokemonRedEnv(
            render_mode=None,
            rom_path=_rom_copy,
            sound=init_kw.get("sound", False),
            game_name_for_adapter=game,
            observation_mode_for_adapter=observation_mode,
            agent_cache_dir_for_adapter=cache_dir,
            game_specific_config_path_for_adapter=config_path,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 100
            ),
        )
        env._rom_tmp_dir = _tmp_dir

    elif game == "tictactoe":
        from gamingagent.envs.zoo_01_tictactoe.TicTacToeEnv import (
            SingleTicTacToeEnv,
        )
        init_kw = config.get("env_init_kwargs", {})
        env = SingleTicTacToeEnv(
            render_mode=None,
            opponent_policy=init_kw.get("opponent_policy", "random"),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 5
            ),
            **common_adapter_kw,
        )

    elif game == "texasholdem":
        from gamingagent.envs.zoo_02_texasholdem.TexasHoldemEnv import (
            SingleTexasHoldemEnv,
        )
        init_kw = config.get("env_init_kwargs", {})
        env = SingleTexasHoldemEnv(
            render_mode=None,
            opponent_policy=init_kw.get("opponent_policy", "random"),
            num_players=init_kw.get("num_players", 2),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 50
            ),
            **common_adapter_kw,
        )

    else:
        raise ValueError(
            f"Game '{game}' is in the mapping but not yet implemented."
        )

    return _GymLikeWrapper(env, action_names, game, max_steps,
                           dynamic_actions=dynamic_actions)
