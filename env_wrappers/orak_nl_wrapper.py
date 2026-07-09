"""
Orak game environment NL wrapper.

Wraps Orak game environments (BaseEnv subclasses from krafton-ai/Orak) so that:
- Observations are natural language strings (via obs2text).
- step() accepts string actions (parsed via text2action).

Orak benchmark covers 9 games across 6 genres:
  Action:      street_fighter, super_mario
  Adventure:   pwaat (Ace Attorney), her_story
  RPG:         darkest_dungeon
  Simulation:  minecraft, stardew_valley
  Strategy:    slay_the_spire
  Puzzle:      baba_is_you, twenty_fourty_eight

Usage:
    from env_wrappers.orak_nl_wrapper import OrakNLWrapper, make_orak_env

    env = make_orak_env("super_mario")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(...)
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent

# Primary: cloned Orak repo
_ORAK_REPO = _CODEBASE_ROOT.parent / "Orak"
_ORAK_SRC = _ORAK_REPO / "src"
_ORAK_MCP_GAMES = _ORAK_SRC / "mcp_game_servers"
_ORAK_MCP_AGENTS = _ORAK_SRC / "mcp_agent_client"

for _p in [str(_ORAK_SRC), str(_ORAK_MCP_GAMES)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


# ── Orak release-branch contract constants ────────────────────────────────
# Source: docs/eval_metrics.md and src/mcp_agent_client/configs/{game}/config.yaml
# in https://github.com/krafton-ai/Orak/tree/release.

_INPUT_MODALITIES = {"text", "image", "text_image"}

# Per-game maximum traversable distance used to normalise the raw score
# returned by ``evaluate()``.  ``Score = traversed / max_distance * 100``
# (see ``Orak/docs/eval_metrics.md``).
_SCORE_NORMALISERS: Dict[str, float] = {
    "super_mario": 3161.0,  # SuperMarioBros stage 1-1 distance to flag
}


def _patch_supermario_for_headless() -> None:
    """Make Orak's hardcoded ``render_mode='human'`` Mario env headless-safe.

    Orak's release branch hardcodes ``render_mode='human'`` in
    ``mcp_game_servers/super_mario/game/super_mario_env.py`` (line 213),
    which requires an X display.  When ``DISPLAY`` is unset (e.g. in CI or
    headless servers) we monkey-patch ``gym_super_mario_bros.make`` to swap
    the render mode to ``'rgb_array'`` so callers don't need ``xvfb-run``.
    """
    if os.environ.get("DISPLAY"):
        return
    try:
        import gym_super_mario_bros as _gsmb  # type: ignore
    except Exception:
        return
    if getattr(_gsmb.make, "_orak_headless_patched", False):
        return
    _orig_make = _gsmb.make

    def _patched_make(*args: Any, **kwargs: Any):
        if kwargs.get("render_mode") == "human":
            kwargs["render_mode"] = "rgb_array"
        return _orig_make(*args, **kwargs)

    _patched_make._orak_headless_patched = True  # type: ignore[attr-defined]
    _gsmb.make = _patched_make


@contextlib.contextmanager
def _orak_cwd():
    """Temporarily chdir to the Orak repo root for games that use relative paths."""
    prev = os.getcwd()
    try:
        os.chdir(str(_ORAK_REPO))
        yield
    finally:
        os.chdir(prev)


def _cfg_path(game: str) -> str:
    """Resolve config.yaml path from the Orak repo."""
    return str(_ORAK_MCP_AGENTS / "configs" / game / "config.yaml")



ORAK_GAMES: Dict[str, Dict[str, Any]] = {
    # ── Puzzle ──────────────────────────────────────────────────────────
    "twenty_fourty_eight": {
        "config_yaml": _cfg_path("twenty_fourty_eight"),
        "action_names": ["up", "down", "left", "right"],
        "task": "Merge tiles to reach 2048. Score = min(score/20000*100, 100).",
        "genre": "puzzle",
    },
    "baba_is_you": {
        "config_yaml": _cfg_path("baba_is_you"),
        "action_names": ["idle", "left", "right", "up", "down"],
        "task": "Solve the Baba Is You puzzle by manipulating rules. 100=win, 40=WIN exists, 20=WALL broken, 0=fail.",
        "genre": "puzzle",
    },
    # ── Action ──────────────────────────────────────────────────────────
    "super_mario": {
        "config_yaml": _cfg_path("super_mario"),
        "action_names": [f"Jump Level: {i}" for i in range(7)],
        "task": "Advance Mario as far right as possible. Score = x_pos / 3161 * 100.",
        "genre": "action",
    },
    "street_fighter": {
        "config_yaml": _cfg_path("street_fighter"),
        "action_names": [
            "Move Closer", "Move Away", "Fireball", "Megapunch", "Hurricane",
            "Low Kick", "Medium Kick", "High Kick", "Jump Closer", "Jump Away",
            "Crouch", "Block", "Low Punch", "Medium Punch", "High Punch",
        ],
        "task": "Defeat the opponent in Street Fighter III. Score = stages cleared.",
        "genre": "action",
    },
    # ── Strategy ────────────────────────────────────────────────────────
    "slay_the_spire": {
        "config_yaml": _cfg_path("slay_the_spire"),
        "action_names": ["PLAY", "END", "CHOOSE", "SKIP"],
        "task": "Climb the Spire, defeat enemies with card combos. Score = floor reached (max 50).",
        "genre": "strategy",
    },
    # ── RPG ─────────────────────────────────────────────────────────────
    "darkest_dungeon": {
        "config_yaml": _cfg_path("darkest_dungeon"),
        "action_names": ["attack", "heal", "swap", "idle", "skip"],
        "task": "Survive dungeon raids. Score = 0.4*combat + 0.3*survival + 0.3*(1-stress).",
        "genre": "rpg",
    },
    # ── Adventure ───────────────────────────────────────────────────────
    "pwaat": {
        "config_yaml": _cfg_path("pwaat"),
        "action_names": [
            "Ok", "Back", "Down", "Up", "Left", "Right",
            "Present evidence", "Press",
        ],
        "task": "Solve cases in Ace Attorney. Score = milestone rewards.",
        "genre": "adventure",
    },
    "her_story": {
        "config_yaml": _cfg_path("her_story"),
        "action_names": ["Search", "Play Video"],
        "task": "Uncover the story by searching keywords and watching videos. Score = videos viewed / 272.",
        "genre": "adventure",
    },
    # ── Simulation ──────────────────────────────────────────────────────
    "minecraft": {
        "config_yaml": _cfg_path("minecraft"),
        "action_names": [],
        "task": "Craft target items in Minecraft. Actions are JavaScript async functions.",
        "genre": "simulation",
    },
    "stardew_valley": {
        "config_yaml": _cfg_path("stardew_valley"),
        "action_names": [
            "till_soil", "plant_seeds", "water_seeds", "harvest_crops",
            "sell_item", "buy_item", "get_out_of_house", "go_house_and_sleep",
        ],
        "task": "Complete farming tasks in Stardew Valley (cleanup, cultivation, shopping, earn money).",
        "genre": "simulation",
    },
}


def _obs_to_text(obs_obj: Any, env: Any) -> str:
    """Convert an Orak Obs dataclass to text via env.obs2text."""
    text = env.obs2text(obs_obj)
    if text is None:
        if hasattr(obs_obj, "to_text"):
            text = obs_obj.to_text()
        else:
            text = str(obs_obj)
    return text or ""


def _extract_visual(obs_obj: Any) -> Dict[str, Any]:
    """Best-effort extraction of pixel-level state from an Orak Obs dataclass.

    Currently targets ``SuperMarioObs`` (which carries ``image: PIL.Image`` and
    ``state["image"]`` as either ``LazyFrames`` or an ``np.ndarray`` normalised
    to ``[0, 1]``). Returns a dict with up to two keys:

    * ``image_pil`` — ``PIL.Image.Image`` in RGB.
    * ``frame``    — ``np.ndarray`` of shape ``(H, W, 3)`` ``uint8``.

    Returns an empty dict when no visual information is available.
    """
    out: Dict[str, Any] = {}

    img = getattr(obs_obj, "image", None)
    if img is not None:
        try:
            from PIL import Image as _PILImage
            if isinstance(img, _PILImage.Image):
                out["image_pil"] = img
                if "frame" not in out:
                    out["frame"] = np.asarray(img.convert("RGB"))
        except Exception:
            pass

    state = getattr(obs_obj, "state", None)
    if isinstance(state, dict) and "image" in state and "frame" not in out:
        raw = state["image"]
        try:
            try:
                from gym.wrappers.frame_stack import LazyFrames as _LF
                if _LF is not None and isinstance(raw, _LF):
                    raw = np.array(raw)
            except Exception:
                pass
            if hasattr(raw, "numpy"):
                raw = raw.numpy()
            arr = np.asarray(raw)
            arr = np.squeeze(arr)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                out["frame"] = arr
        except Exception:
            pass

    return out


def _save_orak_frame(visual: Dict[str, Any], save_dir: Path, episode_id: int, step_num: int) -> Optional[str]:
    """Persist the current Orak frame to disk and return the path.

    Mirrors the layout used by GamingAgent's
    ``GymEnvAdapter._create_agent_observation_path``::

        <save_dir>/observations/env_obs_e<episode_id:03d>_s<step_num:04d>.png

    so the on-disk contract matches between the two wrapper families.
    Returns ``None`` if no frame is available.
    """
    if not visual:
        return None
    arr = visual.get("frame")
    pil = visual.get("image_pil")
    if arr is None and pil is None:
        return None
    try:
        from PIL import Image as _PILImage
        save_dir = Path(save_dir) / "observations"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"env_obs_e{int(episode_id):03d}_s{int(step_num):04d}.png"
        img = pil if pil is not None else _PILImage.fromarray(arr.astype(np.uint8))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(str(path))
        return str(path)
    except Exception:
        return None


class OrakNLWrapper:
    """
    Wraps an Orak BaseEnv so observations are NL strings and step()
    accepts string actions. Presents the same interface as other
    Game-AI-Agent NL wrappers.
    """

    def __init__(
        self,
        env: Any,
        game_name: str,
        include_action_hint: bool = True,
        max_steps: int = 1000,
        input_modality: str = "text",
        save_frames: bool = True,
        frame_save_dir: Optional[str] = None,
        episode_id: int = 1,
    ):
        self._env = env
        self._game_name = game_name
        self._include_action_hint = include_action_hint
        self._max_steps = max_steps
        if input_modality not in _INPUT_MODALITIES:
            raise ValueError(
                f"input_modality must be one of {_INPUT_MODALITIES}, got {input_modality!r}"
            )
        self._input_modality = input_modality
        self._save_frames = save_frames
        self._frame_save_dir: Path = Path(
            frame_save_dir
            if frame_save_dir
            else _CODEBASE_ROOT / "orak_logs" / game_name
        )
        self._episode_id = int(episode_id)
        self._step_count = 0
        self._last_reward: Optional[float] = None
        self._prev_score_val: float = 0.0

        game_info = ORAK_GAMES.get(game_name, {})
        self._action_names: List[str] = game_info.get("action_names", [])
        self._task: str = game_info.get("task", "")
        self._score_normaliser: Optional[float] = _SCORE_NORMALISERS.get(game_name)


    @property
    def env(self):
        return self._env

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    def _format_obs(self, obs_text: str) -> str:
        nl = obs_text
        if self._include_action_hint and self._action_names:
            if self._game_name == "slay_the_spire":
                nl += "\n\nChoose: PLAY <card_idx> [target_idx], END, CHOOSE <idx>, or SKIP."
            elif self._game_name == "minecraft":
                nl += "\n\nWrite a JavaScript async function with bot parameter."
            elif self._game_name == "stardew_valley":
                nl += f"\n\nReturn a Python list of skill calls. Available: {', '.join(self._action_names)}"
            else:
                nl += f"\n\nValid actions: {', '.join(self._action_names[:20])}. Choose one."
        return nl

    def _build_raw_obs(self, obs_obj: Any, obs_text: str, step_num: int) -> Dict[str, Any]:
        """Assemble the canonical ``raw_obs`` payload for a single step.

        Honors the active ``input_modality`` so callers using ``"text"`` only
        skip pixel I/O entirely.  Always includes ``text`` (raw, no
        action-hint suffix) so ``input_modality='image'`` runs can still log
        ground-truth state for downstream evaluation if desired.
        """
        raw_obs: Dict[str, Any] = {"text": obs_text}
        if self._input_modality in ("image", "text_image"):
            visual = _extract_visual(obs_obj)
            raw_obs.update(visual)
            if self._save_frames and visual:
                img_path = _save_orak_frame(
                    visual, self._frame_save_dir, self._episode_id, step_num,
                )
                if img_path:
                    raw_obs["img_path"] = img_path
        return raw_obs

    def _promote_visual_fields(self, raw_obs: Dict[str, Any], info: Dict[str, Any]) -> None:
        """Lift visual fields from ``raw_obs`` to the top level of ``info``.

        Mirrors :meth:`GamingAgentNLWrapper._promote_visual_fields` so callers
        can read pixels off ``info`` directly without diving into
        ``info['raw_obs']``.
        """
        for key in ("img_path", "frame", "image_pil"):
            if raw_obs.get(key) is not None and key not in info:
                info[key] = raw_obs[key]
        if "frame" in raw_obs and "image" not in info:
            info["image"] = raw_obs["frame"]

    def _normalise_score(self, raw_score: Any) -> Tuple[float, Optional[float]]:
        """Return ``(score_val, score_normalised)`` per Orak's eval spec.

        ``evaluate()`` return formats vary by game:
          - Mario:  ``(int_distance, done)``
          - Others: ``(float | str | None, done)``

        For games listed in ``_SCORE_NORMALISERS`` we additionally compute
        the ``Score = traversed / max_distance * 100`` value documented in
        ``Orak/docs/eval_metrics.md``.
        """
        score_val = 0.0
        if isinstance(raw_score, str):
            try:
                score_val = float(raw_score.split("(")[0].strip())
            except (ValueError, AttributeError):
                score_val = 0.0
        elif raw_score is not None:
            try:
                score_val = float(raw_score)
            except (ValueError, TypeError):
                score_val = 0.0
        normalised = (
            min(100.0, max(0.0, score_val / self._score_normaliser * 100.0))
            if self._score_normaliser
            else None
        )
        return score_val, normalised

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._step_count = 0
        self._last_reward = None
        self._prev_score_val = 0.0

        with _orak_cwd():
            # Orak BaseEnv uses initial_obs() rather than Gymnasium's reset()
            if hasattr(self._env, "initial_obs"):
                obs_obj = self._env.initial_obs()
            else:
                kw = {}
                if seed is not None:
                    kw["seed"] = seed
                if options is not None:
                    kw["options"] = options
                obs_obj = self._env.reset(**kw) if kw else self._env.reset()
                if isinstance(obs_obj, tuple):
                    obs_obj = obs_obj[0]

            obs_text = _obs_to_text(obs_obj, self._env)

        nl = self._format_obs(obs_text)

        game_info = {}
        if hasattr(self._env, "get_game_info"):
            game_info = self._env.get_game_info() or {}

        raw_obs = self._build_raw_obs(obs_obj, obs_text, step_num=0)

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "task": self._task,
            "input_modality": self._input_modality,
            "episode_id": self._episode_id,
            "raw_obs": raw_obs,
            **game_info,
        }
        self._promote_visual_fields(raw_obs, info)
        return nl, info

    def step(
        self,
        action: Union[str, int],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        action_str = str(action).strip()

        with _orak_cwd():
            action_obj = self._env.text2action(action_str)

            result = self._env.step(action_obj)
            obs_obj, reward_raw, terminated, truncated, step_info = (
                result[0], result[1], result[2], result[3], result[4]
            )

            score, done = self._env.evaluate(obs_obj)
            obs_text = _obs_to_text(obs_obj, self._env)

        score_val, score_normalised = self._normalise_score(score)
        reward = score_val - self._prev_score_val
        self._prev_score_val = score_val

        self._step_count += 1
        self._last_reward = reward

        if self._step_count >= self._max_steps and not (terminated or truncated):
            truncated = True

        nl = self._format_obs(obs_text)

        game_info = {}
        if hasattr(self._env, "get_game_info"):
            game_info = self._env.get_game_info() or {}

        raw_obs = self._build_raw_obs(obs_obj, obs_text, step_num=self._step_count)

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "step": self._step_count,
            "score": score,
            "score_value": score_val,
            "score_normalised": score_normalised,
            "task": self._task,
            "input_modality": self._input_modality,
            "episode_id": self._episode_id,
            "raw_obs": raw_obs,
            **game_info,
        }
        self._promote_visual_fields(raw_obs, info)
        return nl, reward, bool(terminated or done), bool(truncated), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)


def make_orak_env(
    game_name: str,
    max_steps: int = 1000,
    config_override: Optional[str] = None,
    input_modality: str = "text",
    save_frames: bool = True,
    frame_save_dir: Optional[str] = None,
    episode_id: int = 1,
) -> OrakNLWrapper:
    """Create a wrapped Orak game environment.

    Args:
        game_name: One of the keys in ORAK_GAMES (e.g. "super_mario", "baba_is_you").
        max_steps: Max steps before truncation.
        config_override: Optional path to a custom config YAML.
        input_modality: One of ``"text"``, ``"image"``, ``"text_image"`` —
            mirrors ``cfg.env.input_modality`` from the Orak release branch.
            ``"image"`` and ``"text_image"`` cause the wrapper to extract
            ``frame`` / ``image_pil`` and (when ``save_frames`` is True)
            persist a PNG to ``frame_save_dir``; ``"text"`` skips pixel I/O.
        save_frames: If True (default), and ``input_modality`` includes
            vision, save each step's frame to disk under
            ``<frame_save_dir>/observations/env_obs_eXXX_sXXXX.png``,
            matching GamingAgent's on-disk layout.
        frame_save_dir: Directory to save frames into. Defaults to
            ``<codebase>/orak_logs/<game_name>``.
        episode_id: Used to name saved frames when ``save_frames`` is True.

    Returns:
        OrakNLWrapper with standard reset()/step() Gymnasium interface.
    """
    if game_name not in ORAK_GAMES:
        raise ValueError(f"Unknown Orak game '{game_name}'. Available: {sorted(ORAK_GAMES.keys())}")
    if input_modality not in _INPUT_MODALITIES:
        raise ValueError(
            f"input_modality must be one of {_INPUT_MODALITIES}, got {input_modality!r}"
        )

    config_path = config_override or ORAK_GAMES[game_name]["config_yaml"]

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Ensure the Orak repo is cloned at {_ORAK_REPO}"
        )

    if game_name == "super_mario":
        _patch_supermario_for_headless()

    import omegaconf
    from mcp_game_servers.utils.module_creator import EnvCreator

    cfg = omegaconf.OmegaConf.load(config_path)

    log_dir = str(_CODEBASE_ROOT / "orak_logs" / game_name)
    os.makedirs(log_dir, exist_ok=True)

    _NO_SCREENSHOT_GAMES = {"super_mario"}

    with omegaconf.open_dict(cfg):
        if "env" in cfg:
            cfg.env.log_path = log_dir
            if hasattr(cfg.env, "input_modality"):
                cfg.env.input_modality = input_modality
            if hasattr(cfg.env, "show_graphic"):
                cfg.env.show_graphic = False
            if game_name in _NO_SCREENSHOT_GAMES:
                # Some Orak game configs (e.g. super_mario) don't define
                # ``save_screenshots`` in their structured schema; setting
                # an unknown key under strict mode raises. Skip silently
                # when absent.
                try:
                    if "save_screenshots" in cfg.env:
                        cfg.env.save_screenshots = False
                except Exception:
                    pass
            # Resolve rom_path relative to Orak repo root so it works
            # regardless of the caller's cwd.
            if hasattr(cfg.env, "rom_path") and not os.path.isabs(cfg.env.rom_path):
                abs_rom = os.path.normpath(os.path.join(str(_ORAK_REPO), cfg.env.rom_path))
                cfg.env.rom_path = abs_rom
        cfg.log_path = log_dir

    with _orak_cwd():
        env = EnvCreator(cfg).create()

    wrapper = OrakNLWrapper(
        env,
        game_name=game_name,
        max_steps=max_steps,
        input_modality=input_modality,
        save_frames=save_frames,
        frame_save_dir=frame_save_dir or log_dir,
        episode_id=episode_id,
    )

    # Use the real action space from the env when available.
    if hasattr(env, "action_dict") and env.action_dict:
        wrapper._action_names = sorted(
            env.action_dict.keys(), key=lambda a: env.action_dict[a]
        )

    return wrapper
