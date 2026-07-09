"""Observation-modifying wrappers for gym-v.

These wrappers operate on the multimodal :class:`gym_v.Observation` and are
useful for retro / image-heavy environments such as the Genesis games in
:mod:`gym_v.envs.multi_turn.temporal`. They cover three common needs:

* :class:`GrayscaleObservation` — convert the RGB frame to grayscale.
* :class:`ResizeObservation` — downscale the frame to a fixed (W, H).
* :class:`FrameStack` — stack the last *k* frames as a list of PIL Images.
* :class:`TextStateAugmenter` — append step / episode bookkeeping that lives
  in ``info`` to ``Observation.text`` so language-model agents can see it
  without parsing ``info`` themselves.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from PIL import Image

from gym_v.core import Env, Observation, ObservationWrapper, Wrapper
from gym_v.utils import RecordConstructorArgs

__all__ = [
    "GrayscaleObservation",
    "ResizeObservation",
    "FrameStack",
    "TextStateAugmenter",
]


# --------------------------------------------------------------------------- #
# Image-side transforms
# --------------------------------------------------------------------------- #


def _map_image(
    img: Image.Image | list[Image.Image] | None,
    fn: Callable[[Image.Image], Image.Image],
) -> Image.Image | list[Image.Image] | None:
    if img is None:
        return None
    if isinstance(img, list):
        return [fn(x) for x in img]
    return fn(img)


class GrayscaleObservation(ObservationWrapper, RecordConstructorArgs):
    """Convert ``Observation.image`` to single-channel grayscale.

    Args:
        env: The environment to wrap.
        keep_dim: If ``True``, the grayscale image is converted back to
            3-channel RGB so downstream code expecting an RGB image keeps
            working. If ``False`` (default), the image stays in mode ``"L"``.
    """

    def __init__(self, env: Env, keep_dim: bool = False):
        RecordConstructorArgs.__init__(self, keep_dim=keep_dim)
        ObservationWrapper.__init__(self, env)
        self._keep_dim = bool(keep_dim)

    def _convert(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        return gray.convert("RGB") if self._keep_dim else gray

    def observation(self, observation: Observation) -> Observation:
        return Observation(
            image=_map_image(observation.image, self._convert),
            text=observation.text,
            metadata=observation.metadata,
        )


class ResizeObservation(ObservationWrapper, RecordConstructorArgs):
    """Resize ``Observation.image`` to a fixed ``(width, height)``.

    Args:
        env: The environment to wrap.
        size: Either a single int (square image) or a ``(width, height)`` tuple.
        resample: PIL resampling filter (default :data:`PIL.Image.BILINEAR`).
    """

    def __init__(
        self,
        env: Env,
        size: int | tuple[int, int],
        resample: int = Image.BILINEAR,
    ):
        if isinstance(size, int):
            wh: tuple[int, int] = (size, size)
        else:
            if len(size) != 2:
                raise ValueError(f"size must be int or (w, h) tuple, got {size!r}")
            wh = (int(size[0]), int(size[1]))
        if wh[0] <= 0 or wh[1] <= 0:
            raise ValueError(f"size components must be positive, got {wh}")

        RecordConstructorArgs.__init__(self, size=wh, resample=int(resample))
        ObservationWrapper.__init__(self, env)
        self._wh = wh
        self._resample = int(resample)

    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize(self._wh, resample=self._resample)

    def observation(self, observation: Observation) -> Observation:
        return Observation(
            image=_map_image(observation.image, self._resize),
            text=observation.text,
            metadata=observation.metadata,
        )


class FrameStack(Wrapper, RecordConstructorArgs):
    """Replace ``Observation.image`` with the last ``num_stack`` frames.

    The stacked image is exposed as ``list[PIL.Image.Image]`` in chronological
    order (oldest first, newest last). Per-agent buffers are maintained so
    multi-agent envs work too. After :meth:`reset`, the buffer is filled by
    repeating the initial frame ``num_stack`` times.

    Args:
        env: The environment to wrap.
        num_stack: Number of frames to keep in each stack (>= 1).
    """

    def __init__(self, env: Env, num_stack: int = 4):
        if num_stack < 1:
            raise ValueError(f"num_stack must be >= 1, got {num_stack}")
        RecordConstructorArgs.__init__(self, num_stack=num_stack)
        Wrapper.__init__(self, env)
        self._k = int(num_stack)
        self._buffers: dict[str, deque[Image.Image]] = {}

    def _push(self, agent_id: str, img: Image.Image) -> list[Image.Image]:
        buf = self._buffers.get(agent_id)
        if buf is None:
            buf = deque(maxlen=self._k)
            self._buffers[agent_id] = buf
        buf.append(img)
        return list(buf)

    def _stack_obs(self, agent_id: str, observation: Observation) -> Observation:
        img = observation.image
        if isinstance(img, list):
            # Already a list (e.g. another FrameStack below us). Use the most recent.
            single = img[-1] if img else None
        else:
            single = img
        if single is None:
            return observation
        # On reset, fill buffer with `single` so we always emit `k` frames.
        if agent_id not in self._buffers or len(self._buffers[agent_id]) == 0:
            for _ in range(self._k):
                self._push(agent_id, single)
            stacked = list(self._buffers[agent_id])
        else:
            stacked = self._push(agent_id, single)
        return Observation(
            image=stacked, text=observation.text, metadata=observation.metadata
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        self._buffers.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        return (
            {aid: self._stack_obs(aid, o) for aid, o in obs.items()},
            info,
        )

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            {aid: self._stack_obs(aid, o) for aid, o in obs.items()},
            reward,
            terminated,
            truncated,
            info,
        )


# --------------------------------------------------------------------------- #
# Text-side augmentation
# --------------------------------------------------------------------------- #


class TextStateAugmenter(Wrapper, RecordConstructorArgs):
    """Append structured state info to ``Observation.text``.

    Some env wrappers (notably :class:`gym_v.wrappers.HistoryRecorder`) put rich
    bookkeeping into ``info`` that a language-model agent can't see unless it's
    embedded in ``Observation.text``. This wrapper formats a configurable set
    of fields from each agent's ``info`` and appends them — pipe-separated —
    after the underlying env's text. By default it surfaces the bookkeeping
    that :class:`~gym_v.envs.multi_turn.temporal.retro_env.RetroGymVEnv` adds.

    Args:
        env: The environment to wrap.
        include_fields: Ordered iterable of ``info`` keys to surface.
        prefix: String prefix prepended to the augmented block (default ``""``).
        separator: Separator between the original text and the augmented block
            (default ``" | "``).
    """

    DEFAULT_FIELDS = (
        "frame_index",
        "episode_reward",
        "last_action",
        "action_history",
    )

    def __init__(
        self,
        env: Env,
        include_fields: tuple[str, ...] | list[str] | None = None,
        prefix: str = "",
        separator: str = " | ",
    ):
        fields = (
            tuple(include_fields)
            if include_fields is not None
            else tuple(self.DEFAULT_FIELDS)
        )
        RecordConstructorArgs.__init__(
            self,
            include_fields=list(fields),
            prefix=prefix,
            separator=separator,
        )
        Wrapper.__init__(self, env)
        self._fields = fields
        self._prefix = prefix
        self._separator = separator

    def _format(self, info_for_agent: dict[str, Any]) -> str:
        chunks: list[str] = []
        for key in self._fields:
            if key not in info_for_agent:
                continue
            value = info_for_agent[key]
            if isinstance(value, (list, tuple)):
                rendered = "[" + ",".join(str(v) for v in value) + "]"
            elif isinstance(value, float):
                rendered = f"{value:.3f}"
            else:
                rendered = str(value)
            chunks.append(f"{key}={rendered}")
        return self._separator.join(chunks)

    def _augment_obs(
        self, obs: Observation, info_for_agent: dict[str, Any]
    ) -> Observation:
        block = self._format(info_for_agent)
        if not block:
            return obs
        new_text = (
            (obs.text + self._separator if obs.text else "")
            + (self._prefix if self._prefix else "")
            + block
        )
        return Observation(
            image=obs.image, text=new_text, metadata=obs.metadata
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return (
            {
                aid: self._augment_obs(o, info.get(aid, {}) if isinstance(info, dict) else {})
                for aid, o in obs.items()
            },
            info,
        )

    def step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            {
                aid: self._augment_obs(o, info.get(aid, {}) if isinstance(info, dict) else {})
                for aid, o in obs.items()
            },
            reward,
            terminated,
            truncated,
            info,
        )
