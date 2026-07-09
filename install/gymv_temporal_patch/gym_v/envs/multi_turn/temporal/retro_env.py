"""Stable-Retro environment wrapper for gym-v.

This adapter wraps stable-retro's ``RetroEnv`` to provide a ``gym_v.Env`` interface
returning multimodal :class:`~gym_v.Observation` (image + text). Designed to be used
together with the wrappers in :mod:`gym_v.wrappers` (frame skip, frame stack,
grayscale/resize image transforms, text augmenters, history recorder, ...).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image
import stable_retro

from gym_v import Env, Observation, get_logger

logger = get_logger()


class RetroGymVEnv(Env):
    """Wrapper for stable-retro environments.

    This adapter wraps stable-retro's RetroEnv to provide a gym_v.Env interface.
    It converts string-based actions (e.g., "UP", "A", "B+UP") into button masks
    and converts numpy image observations to PIL Images.

    Args:
        game: Name of the game (e.g., "Airstriker-Genesis-v0"). If the suffix
            ``-v0`` is missing it is appended automatically when the bare name
            doesn't resolve in stable-retro's ROM index.
        state: Game state to load (default: stable_retro.State.DEFAULT).
        scenario: Scenario file to use (default: None).
        players: Number of in-game players (passed to stable-retro).
        num_players: Number of agents exposed by gym-v (default: 1).
        action_history_len: How many of the most recent button presses to surface
            in the textual observation (default: 8).
        **kwargs: Additional arguments passed to stable_retro.make.

    Example:
        >>> env = RetroGymVEnv(game="Airstriker-Genesis-v0")
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step({"agent_0": "A"})
    """

    # Meta: source=Retro, category=temporal, turn=multi

    def __init__(
        self,
        game: str,
        state: Any | None = None,
        scenario: str | None = None,
        players: int = 1,
        num_players: int = 1,
        action_history_len: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._game = self._resolve_game_name(game)
        self._players = players
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._action_history_len = max(0, int(action_history_len))

        retro_kwargs: dict[str, Any] = {
            "game": self._game,
            "players": players,
            "render_mode": "rgb_array",
        }
        if state is not None:
            retro_kwargs["state"] = state
        if scenario is not None:
            retro_kwargs["scenario"] = scenario

        # FILTERED actions enables natural single/combo button mapping.
        retro_kwargs["use_restricted_actions"] = stable_retro.Actions.FILTERED

        self._retro_env = stable_retro.make(**retro_kwargs)

        # --- Button configuration ---
        self.buttons = self._retro_env.buttons
        self._num_buttons = self._retro_env.num_buttons
        self.available_actions = [b for b in self.buttons if b and b != "NULL"]

        self._button_to_idx: dict[str, int] = {
            button.upper(): idx
            for idx, button in enumerate(self.buttons)
            if button and button != "NULL"
        }

        # --- Stable-retro ROM watch variables (from data.json) ---
        self._watch_keys: list[str] = self._discover_watch_keys()

        # --- Episode bookkeeping ---
        self._frame_index: int = 0
        self._episode_reward: float = 0.0
        self._last_action: str | None = None
        self._action_history: deque[str] = deque(maxlen=self._action_history_len)
        self._last_info: dict[str, Any] = {}

        logger.info(f"Initialized RetroGymVEnv for game: {self._game}")
        logger.info(f"Available buttons: {self.available_actions}")
        if self._watch_keys:
            logger.info(f"Watch variables: {self._watch_keys}")

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_game_name(game: str) -> str:
        """Resolve a stable-retro game name, auto-appending ``-v0`` if needed."""
        try:
            stable_retro.data.get_romfile_path(game)
            return game
        except FileNotFoundError:
            if not game.endswith("-v0"):
                alt = f"{game}-v0"
                try:
                    stable_retro.data.get_romfile_path(alt)
                    logger.warning(
                        f"Game '{game}' not found; using '{alt}' instead. "
                        "Please update your registration."
                    )
                    return alt
                except FileNotFoundError:
                    pass
            raise

    def _discover_watch_keys(self) -> list[str]:
        """Read ``data.json`` for the active ROM and return its watch variables.

        Falls back to an empty list if the file is missing or malformed; the
        text observation logic will then surface whatever stable-retro chooses
        to put in ``info``.
        """
        try:
            rom_path = Path(stable_retro.data.get_romfile_path(self._game))
        except FileNotFoundError:
            return []
        data_json = rom_path.parent / "data.json"
        if not data_json.exists():
            return []
        try:
            payload = json.loads(data_json.read_text())
        except (OSError, ValueError):
            return []
        return list(payload.get("info", {}).keys())

    # ------------------------------------------------------------------ #
    # gym_v.Env API
    # ------------------------------------------------------------------ #

    @property
    def description(self) -> str:
        watch_block = (
            f"\n            Game-state variables exposed in info: {self._watch_keys}"
            if self._watch_keys
            else ""
        )
        return dedent(f"""
            This is a retro game environment: {self._game}.

            Available buttons: {self.available_actions}{watch_block}

            ## Output Format
            You must output ONLY the action string, nothing else. No explanation, no reasoning, just the action.

            ## Valid Actions
            - Single button: A, B, C, UP, DOWN, LEFT, RIGHT, START
            - Combined buttons: Use "+" to press multiple buttons simultaneously

            ## Examples
            - Move right: RIGHT
            - Move up-right: UP+RIGHT
            - Jump: A
            - Jump right: A+RIGHT
            - Attack: B
            - Attack while moving: B+LEFT
            - Special move: A+B+DOWN
            - No action: NOOP

            ## Your Response
            Output only one action per step. Example valid responses:
            RIGHT
            A+UP
            B
            DOWN+LEFT
            NOOP
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        obs_array, retro_info = self._retro_env.reset(seed=seed, options=options)

        self._frame_index = 0
        self._episode_reward = 0.0
        self._last_action = None
        self._action_history.clear()
        self._last_info = dict(retro_info) if retro_info else {}

        image = Image.fromarray(obs_array)
        text = self._get_observation_text(self._last_info, reward=0.0)
        obs = Observation(
            image=image,
            text=text,
            metadata=self._build_metadata(self._last_info, reward=0.0),
        )
        info: dict[str, Any] = {
            "retro_info": self._last_info,
            "frame_index": self._frame_index,
            "episode_reward": self._episode_reward,
            "available_actions": list(self.available_actions),
        }

        logger.info(f"Reset {self._game} environment.")

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: info for agent_id in self._agent_ids},
        )

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        button_mask = self._action_to_mask(action_str)

        obs_array, reward, terminated, truncated, retro_info = self._retro_env.step(
            button_mask
        )

        # --- Bookkeeping ---
        self._frame_index += 1
        self._episode_reward += float(reward)
        self._last_action = action_str
        if self._action_history_len > 0:
            self._action_history.append(action_str)
        self._last_info = dict(retro_info) if retro_info else {}

        image = Image.fromarray(obs_array)
        text = self._get_observation_text(self._last_info, reward=float(reward))
        obs = Observation(
            image=image,
            text=text,
            metadata=self._build_metadata(self._last_info, reward=float(reward)),
        )

        info: dict[str, Any] = {
            "retro_info": self._last_info,
            "action_parsed": self._get_action_meaning(action_str),
            "frame_index": self._frame_index,
            "episode_reward": self._episode_reward,
            "last_action": action_str,
            "action_history": list(self._action_history),
            "available_actions": list(self.available_actions),
        }

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: float(reward) for agent_id in self._agent_ids},
            {
                **{agent_id: bool(terminated) for agent_id in self._agent_ids},
                "__all__": bool(terminated),
            },
            {
                **{agent_id: bool(truncated) for agent_id in self._agent_ids},
                "__all__": bool(truncated),
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the current frame as a PIL Image."""
        rendered = self._retro_env.render()
        if rendered is not None and isinstance(rendered, np.ndarray):
            return Image.fromarray(rendered)
        return None

    def close(self):
        if hasattr(self, "_retro_env") and self._retro_env is not None:
            self._retro_env.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _action_to_mask(self, action_str: str) -> np.ndarray:
        """Convert a string action ("A", "A+UP", "NOOP", ...) to a button mask."""
        mask = np.zeros(self._num_buttons, dtype=np.uint8)

        action_upper = action_str.upper().strip() if action_str else ""
        if action_upper in ("NOOP", "NONE", ""):
            return mask

        for button in (b.strip() for b in action_upper.split("+")):
            if button in self._button_to_idx:
                mask[self._button_to_idx[button]] = 1
            else:
                logger.warning(
                    f"Unknown button '{button}'. Available: {list(self._button_to_idx.keys())}"
                )
        return mask

    def _get_action_meaning(self, action_str: str) -> list[str]:
        """Get a list of button names being pressed."""
        action_upper = action_str.upper().strip() if action_str else ""
        if action_upper in ("NOOP", "NONE", ""):
            return []
        return [
            b.strip()
            for b in action_upper.split("+")
            if b.strip() in self._button_to_idx
        ]

    def _build_metadata(
        self, info: dict[str, Any], reward: float
    ) -> dict[str, Any]:
        """Structured per-step metadata stored on the Observation."""
        return {
            "game": self._game,
            "frame_index": self._frame_index,
            "step_reward": reward,
            "episode_reward": self._episode_reward,
            "last_action": self._last_action,
            "action_history": list(self._action_history),
            "ram_watch": {
                k: info[k] for k in self._watch_keys if k in info
            },
            "available_actions": list(self.available_actions),
        }

    def _get_observation_text(self, info: dict[str, Any], reward: float) -> str:
        """Generate text description from game state info.

        The textual representation is intentionally compact, single-line, and
        contains only stable fields so language-model agents can rely on its
        layout. Game-specific RAM watch variables (from ``data.json``) come
        first, followed by step / episode bookkeeping and recent actions.
        """
        parts: list[str] = [f"Game: {self._game}"]

        # Prefer canonical ordering for the well-known keys.
        canonical = ("score", "lives", "health", "level", "gameover")
        seen: set[str] = set()
        for key in canonical:
            if key in info:
                parts.append(f"{key.capitalize()}: {info[key]}")
                seen.add(key)

        # Then the rest of the data.json watch keys.
        for key in self._watch_keys:
            if key in seen or key not in info:
                continue
            value = info[key]
            if isinstance(value, (int, float, np.integer, np.floating)):
                parts.append(f"{key}: {value}")
                seen.add(key)

        # Step / episode bookkeeping.
        parts.append(f"Frame: {self._frame_index}")
        parts.append(f"StepReward: {reward:.3f}")
        parts.append(f"EpReward: {self._episode_reward:.3f}")

        if self._last_action is not None:
            parts.append(f"LastAction: {self._last_action}")
        if self._action_history_len > 0 and len(self._action_history) > 0:
            recent = ",".join(self._action_history)
            parts.append(f"Recent: [{recent}]")

        return " | ".join(parts)
