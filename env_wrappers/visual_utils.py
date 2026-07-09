"""
Shared helpers for accessing visual state from env_wrappers.

Different wrappers expose pixels under different keys:

- GamingAgent (2048 / Candy Crush / Tetris) via ``make_gaming_env(..., observation_mode="vision")``:
    obs dict may carry ``image`` (np.ndarray) and/or ``img_path`` (PNG on disk).
- ``GamingAgentNLWrapper``: same dict is preserved on ``info["raw_obs"]``.
- ``OrakNLWrapper`` for Super Mario:
    ``info["raw_obs"]`` carries ``image_pil`` and/or ``frame``.
- ``OSWorldGymWrapper``: ``obs["screenshot"]`` is the raw RGB array.
- ``OSWorldNLWrapper``: ``info["raw_obs"]["screenshot"]``.

``get_obs_image`` and ``get_obs_pil_image`` normalise these layouts so callers
can write::

    frame = get_obs_image(obs_or_info)   # np.ndarray (H, W, 3) uint8 | None
    pil   = get_obs_pil_image(obs_or_info)  # PIL.Image.Image | None

When callers explicitly pass ``allow_replay_fallback=True``, these helpers can
also reconstruct a frame from text observations using ``replay`` renderers.
That path is intentionally opt-in because direct wrapper pixels are faster and
better represent the live environment.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np


def _coerce_uint8_rgb(arr: Any) -> Optional[np.ndarray]:
    """Best-effort coercion of an array-like to ``(H, W, 3)`` ``uint8``."""
    if arr is None:
        return None
    try:
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        a = np.asarray(arr)
    except Exception:
        return None
    if a.size == 0:
        return None
    a = np.squeeze(a)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim != 3:
        return None
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.shape[-1] != 3:
        return None
    if a.dtype != np.uint8:
        try:
            mx = float(a.max())
        except Exception:
            mx = 255.0
        if mx <= 1.0:
            a = (a * 255.0).clip(0, 255).astype(np.uint8)
        else:
            a = a.astype(np.uint8)
    return a


def _from_path(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not isinstance(path, str) or not os.path.exists(path):
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:
        return None


def _from_pil(img: Any) -> Optional[np.ndarray]:
    try:
        from PIL import Image as _PILImage
        if isinstance(img, _PILImage.Image):
            return np.asarray(img.convert("RGB"))
    except Exception:
        return None
    return None


_IMAGE_KEYS = (
    "image",
    "frame",
    "screenshot",
    "rgb",
    "rgb_array",
)
_PIL_KEYS = ("image_pil", "pil_image")
_PATH_KEYS = ("img_path", "image_path", "screenshot_path")


def _as_mapping(value: Any) -> Optional[dict]:
    return value if isinstance(value, dict) else None


def _first_present(mapping: dict, keys: Tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_game_name(name: Any) -> Optional[str]:
    if not name:
        return None
    normalised = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "2048": "twenty_forty_eight",
        "twenty_fourty_eight": "twenty_forty_eight",
        "orak_twenty_fourty_eight": "twenty_forty_eight",
        "orak_super_mario": "super_mario",
    }
    return aliases.get(normalised, normalised)


def _replay_fallback_fields(obs_or_info: Any) -> Tuple[Optional[str], str, int, float, float, str]:
    """Extract replay renderer arguments from a wrapper ``info``/``obs`` dict."""
    mapping = _as_mapping(obs_or_info)
    if mapping is None:
        return None, "", 0, 0.0, 0.0, ""

    raw = mapping.get("raw_obs") if isinstance(mapping.get("raw_obs"), dict) else {}
    raw = raw or {}

    game = _normalise_game_name(
        _first_present(mapping, ("game_name", "game", "env_name"))
        or _first_present(raw, ("game_name", "game", "env_name"))
    )
    state = _first_present(
        mapping,
        (
            "state_natural_language",
            "state",
            "text",
            "textual_representation",
            "processed_visual_description",
        ),
    )
    if state is None:
        state = _first_present(
            raw,
            (
                "state_natural_language",
                "state",
                "text",
                "textual_representation",
                "processed_visual_description",
            ),
        )

    step = _safe_int(_first_present(mapping, ("step", "step_count", "timestep")), 0)
    reward = _safe_float(_first_present(mapping, ("reward", "last_reward", "raw_env_reward")), 0.0)
    total_reward = _safe_float(
        _first_present(
            mapping,
            ("total_reward", "score_value", "score_normalised", "score", "perf_score"),
        ),
        reward,
    )
    action = _first_present(mapping, ("action", "last_action", "agent_action")) or ""

    return game, str(state or ""), step, reward, total_reward, str(action)


def _render_replay_fallback(obs_or_info: Any) -> Optional[np.ndarray]:
    game, state, step, reward, total_reward, action = _replay_fallback_fields(obs_or_info)
    if not game or not state:
        return None

    try:
        from replay.generate_replay_gifs import RENDERERS
    except Exception:
        return None

    renderer = RENDERERS.get(game)
    if renderer is None:
        return None

    try:
        image = renderer(state, step, reward, total_reward, action)
    except Exception:
        return None
    return _from_pil(image)


def get_obs_image(obs_or_info: Any, *, allow_replay_fallback: bool = False) -> Optional[np.ndarray]:
    """Return the current frame as a ``(H, W, 3)`` ``uint8`` array, or ``None``.

    Accepts an obs dict, an info dict, or anything with a ``raw_obs``
    sub-dictionary (GamingAgent NL wrapper, Orak NL wrapper, OSWorld NL
    wrapper). PIL images and paths to PNGs are resolved transparently.

    Set ``allow_replay_fallback=True`` to reconstruct a frame from text state
    with ``replay.generate_replay_gifs`` when no live pixels are exposed.
    The replay renderer is imported lazily and never used on the default path.
    """
    if obs_or_info is None:
        return None

    if isinstance(obs_or_info, np.ndarray):
        return _coerce_uint8_rgb(obs_or_info)

    if not isinstance(obs_or_info, dict):
        try:
            from PIL import Image as _PILImage
            if isinstance(obs_or_info, _PILImage.Image):
                return _from_pil(obs_or_info)
        except Exception:
            pass
        return None

    raw = obs_or_info.get("raw_obs")
    if isinstance(raw, dict):
        nested = get_obs_image(raw, allow_replay_fallback=False)
        if nested is not None:
            return nested

    for key in _IMAGE_KEYS:
        if key in obs_or_info:
            arr = _coerce_uint8_rgb(obs_or_info[key])
            if arr is not None:
                return arr

    for key in _PIL_KEYS:
        if key in obs_or_info:
            arr = _from_pil(obs_or_info[key])
            if arr is not None:
                return arr

    for key in _PATH_KEYS:
        if key in obs_or_info:
            arr = _from_path(obs_or_info[key])
            if arr is not None:
                return arr

    if allow_replay_fallback:
        return _render_replay_fallback(obs_or_info)

    return None


def get_obs_pil_image(obs_or_info: Any, *, allow_replay_fallback: bool = False):
    """Return the current frame as a ``PIL.Image.Image`` in RGB, or ``None``."""
    try:
        from PIL import Image
    except ImportError:
        return None

    if isinstance(obs_or_info, dict):
        for key in _PIL_KEYS:
            if key in obs_or_info and isinstance(obs_or_info[key], Image.Image):
                return obs_or_info[key]
        raw = obs_or_info.get("raw_obs")
        if isinstance(raw, dict):
            for key in _PIL_KEYS:
                if key in raw and isinstance(raw[key], Image.Image):
                    return raw[key]

    arr = get_obs_image(obs_or_info, allow_replay_fallback=allow_replay_fallback)
    if arr is None:
        return None
    return Image.fromarray(arr, mode="RGB")


__all__ = ["get_obs_image", "get_obs_pil_image"]
