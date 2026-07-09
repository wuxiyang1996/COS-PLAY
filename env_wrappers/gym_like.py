"""
Gymnasium-compatible interface for GamingAgent (LMGame-Bench) environments.

Provides make_gaming_env() and list_games() so the env_wrappers
test harness can create and interact with GamingAgent environments using
the standard Gymnasium API:

    from env_wrappers.gym_like import make_gaming_env, list_games

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
    "candy_crush": "custom_03_candy_crush",
    "tetris": "custom_04_tetris",
    "tictactoe": "zoo_01_tictactoe",
    "texasholdem": "zoo_02_texasholdem",
}

# Orak benchmark (krafton-ai/Orak) games are handled via orak_nl_wrapper.py
ORAK_GAME_NAMES = [
    "orak_twenty_fourty_eight",
    "orak_baba_is_you",
    "orak_super_mario",
    "orak_street_fighter",
    "orak_slay_the_spire",
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

    obs_dict always contains a ``"text"`` key suitable for GamingAgentNLWrapper.
    When the native env was constructed with ``observation_mode`` ``"vision"``
    or ``"both"``, the GamingAgent adapter writes a PNG of the current board
    each step; this wrapper additionally exposes:

    * ``obs["img_path"]`` — absolute path to the PNG written for this step.
    * ``obs["image"]``    — ``np.ndarray`` of shape ``(H, W, 3)`` ``uint8``,
                            loaded from disk when ``load_image_array=True``
                            and from ``self._env.render()`` if ``render_mode``
                            is ``"rgb_array"``.

    These keys are absent in pure ``"text"`` mode so the legacy contract is
    unchanged.
    """

    def __init__(self, native_env: Any, action_names: List[str],
                 game_name: str, max_steps: int,
                 dynamic_actions: bool = False,
                 observation_mode: str = "text",
                 render_mode: Optional[str] = None,
                 load_image_array: bool = True):
        self._env = native_env
        self._action_names = action_names
        self._game_name = game_name
        self._max_steps = max_steps
        self._step_count = 0
        self._episode_id = 0
        self._dynamic_actions = dynamic_actions
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        self._load_image_array = load_image_array

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    def _obs_to_dict(self, obs_obj: Any) -> Dict[str, Any]:
        """Convert a native GamingAgent ``Observation`` into a plain dict.

        Mirrors the canonical GamingAgent perception contract from
        ``gamingagent.envs.gym_env_adapter.GymEnvAdapter`` /
        ``gamingagent.modules.core_module.Observation``:

        * ``text``                       — best-effort string suitable for the
                                           NL wrapper (textual_representation
                                           > processed_visual_description >
                                           ``str(obs)``).
        * ``textual_representation``     — raw GamingAgent textual board.
        * ``processed_visual_description`` — populated by ``PerceptionModule``
                                           when the agent runs vision
                                           pre-processing (often ``None``).
        * ``background``                 — static background blurb if the
                                           env attached one.
        * ``img_path``                   — absolute path to the canonical PNG
                                           the adapter writes for this step
                                           (``vision`` / ``both`` modes).
        * ``image``                      — uint8 RGB ``np.ndarray`` (loaded
                                           via ``env.render()`` when
                                           ``render_mode='rgb_array'``, then
                                           via the PNG on disk).

        Legacy ``text`` and ``img_path`` keys are preserved so existing
        callers keep working.
        """
        result: Dict[str, Any] = {}
        perception: Dict[str, Any] = {}

        if hasattr(obs_obj, "get_perception_summary"):
            try:
                perception = obs_obj.get_perception_summary() or {}
            except Exception:
                perception = {}

        textual = perception.get("textual_representation") or getattr(
            obs_obj, "textual_representation", None,
        )
        proc_desc = perception.get("processed_visual_description") or getattr(
            obs_obj, "processed_visual_description", None,
        )
        img_path = perception.get("img_path") or getattr(obs_obj, "img_path", None)
        background = getattr(obs_obj, "background", None)

        if isinstance(obs_obj, dict):
            textual = textual or obs_obj.get("textual_representation") or obs_obj.get("text")
            img_path = img_path or obs_obj.get("img_path") or obs_obj.get("image_path")
            proc_desc = proc_desc or obs_obj.get("processed_visual_description")
            background = background or obs_obj.get("background")

        if textual:
            result["text"] = str(textual)
            result["textual_representation"] = str(textual)
        elif proc_desc:
            result["text"] = str(proc_desc)
        else:
            result["text"] = str(obs_obj)

        if proc_desc:
            result["processed_visual_description"] = str(proc_desc)
        if background:
            result["background"] = str(background)

        if not img_path and self._observation_mode in ("vision", "both"):
            img_path = self._render_and_save_frame()

        if img_path:
            result["img_path"] = img_path

        image = self._maybe_load_image(img_path)
        if image is not None:
            result["image"] = image

        return result

    def _render_and_save_frame(self) -> Optional[str]:
        """Render the env to an ``rgb_array`` and persist it through the
        GamingAgent adapter so the saved path matches the rest of the run.

        This is a fallback for ``observation_mode='text'`` envs that still
        expose ``render(render_mode='rgb_array')`` — the wrapper user often
        wants pixels regardless of which native mode the env was built in.
        Returns the saved path, or ``None`` when neither rendering nor
        saving works.
        """
        if not hasattr(self._env, "render"):
            return None
        try:
            frame = self._env.render()
        except Exception:
            return None
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            try:
                arr = (
                    (arr * 255.0).clip(0, 255).astype(np.uint8)
                    if float(arr.max()) <= 1.0
                    else arr.astype(np.uint8)
                )
            except Exception:
                return None

        adapter = getattr(self._env, "adapter", None)
        if adapter is not None and hasattr(adapter, "save_frame_and_get_path"):
            try:
                return adapter.save_frame_and_get_path(arr)
            except Exception:
                return None
        return None

    def _maybe_load_image(self, img_path: Optional[str]) -> Optional[np.ndarray]:
        """Best-effort RGB ``np.ndarray`` for the current step.

        Priority:
          1. ``self._env.render()`` when constructed with ``render_mode='rgb_array'``
             (no disk I/O, fastest path).
          2. PNG at ``img_path`` (written by the GamingAgent adapter).
        Returns ``None`` if neither source is available or both fail.
        """
        if self._render_mode == "rgb_array":
            try:
                rendered = self._env.render() if hasattr(self._env, "render") else None
                if rendered is not None:
                    arr = np.asarray(rendered)
                    if arr.dtype != np.uint8:
                        arr = (
                            (arr * 255).clip(0, 255).astype(np.uint8)
                            if arr.max() <= 1.0
                            else arr.astype(np.uint8)
                        )
                    return arr
            except Exception:
                pass

        if img_path and self._load_image_array and os.path.exists(img_path):
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(img_path) as im:
                    return np.asarray(im.convert("RGB"))
            except Exception:
                return None

        return None

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
    render_mode: Optional[str] = None,
    load_image_array: bool = True,
) -> _GymLikeWrapper:
    """Create a Gymnasium-compatible GamingAgent environment.

    Args:
        game: One of list_games() (e.g. "twenty_forty_eight", "candy_crush").
        max_steps: Maximum steps per episode before truncation.
        observation_mode: ``"text"``, ``"vision"``, or ``"both"``. ``"vision"``
            and ``"both"`` make the GamingAgent adapter write a PNG snapshot
            each step to its cache directory; the wrapper exposes the path as
            ``obs["img_path"]``.
        render_mode: Forwarded to the underlying env. When set to
            ``"rgb_array"``, the wrapper additionally calls ``env.render()``
            after each ``reset``/``step`` and stores the resulting frame in
            ``obs["image"]`` (no disk round-trip).
        load_image_array: If True (default) and ``observation_mode`` includes
            vision, lazily load the PNG referenced by ``obs["img_path"]`` into
            ``obs["image"]`` as a uint8 RGB ``np.ndarray``. Disable to skip
            disk reads when only the path is needed.

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
            render_mode=render_mode,
            size=init_kw.get("size", 4),
            max_pow=init_kw.get("max_pow", 16),
            game_name_for_adapter=game,
            max_stuck_steps_for_adapter=config.get(
                "max_unchanged_steps_for_termination", 10
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
            render_mode=render_mode,
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

    elif game == "tictactoe":
        from gamingagent.envs.zoo_01_tictactoe.TicTacToeEnv import (
            SingleTicTacToeEnv,
        )
        init_kw = config.get("env_init_kwargs", {})
        env = SingleTicTacToeEnv(
            render_mode=render_mode,
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
            render_mode=render_mode,
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

    return _GymLikeWrapper(
        env, action_names, game, max_steps,
        dynamic_actions=dynamic_actions,
        observation_mode=observation_mode,
        render_mode=render_mode,
        load_image_array=load_image_array,
    )
