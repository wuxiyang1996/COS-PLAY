"""Gym-V Temporal/* (stable-retro Genesis) → NL-wrapper for the co-evolution loop.

Mirrors :class:`env_wrappers.gamingagent_nl_wrapper.GamingAgentNLWrapper`
in surface area so :mod:`trainer.coevolution.episode_runner` can dispatch
to it via the same ``(obs_nl, info)`` / ``(obs_nl, reward, term, trunc, info)``
contract:

* ``env.reset()`` returns ``(obs_nl: str, info: dict)`` with
  ``info["action_names"]``, ``info["structured_state"]``,
  ``info["raw_obs"]``, and ``info["image"]`` populated.
* ``env.step(action_str)`` returns the gym-style 5-tuple, with
  ``action_str`` looked up against the per-step ``available_actions`` list
  emitted by ``RetroGymVEnv``. Gym-V's multi-agent dict API
  (``{agent_id: action}``) is hidden inside the wrapper.

Construction
------------

The wrapper takes ownership of an already-`gym_v.make("Temporal/<Env>-v0")`-ed
env.  The :func:`make_gymv_temporal_env` factory is the convenience
constructor that handles ``StochasticFrameSkip`` (``frame_skip=8`` by
default — see ``baselines/README.md`` § "Gym-V benchmark scope" for why
the skip matters) and the gym_v / `stable_retro` import dance so callers
in the trainer don't have to reach into ``gymv_wrapper`` themselves.

Visual grounding
----------------

The wrapper prefers the project's
:func:`gymv_wrapper.temporal_visual_grounding.build_temporal_visual_schema`
output — a JSON-serialisable dict with frame geometry, parsed
``obs.text``, RAM watch, and per-game grounding focus — for
``info["structured_state"]``.  When the schema builder errors out for
any reason (missing watch keys, unfamiliar env id), the wrapper falls
back to a minimal hand-rolled summary so the rest of the runner pipeline
keeps moving.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public env-id ↔ short-slug mapping
# ---------------------------------------------------------------------------
#
# Slugs are kebab-free, lowercase, snake_case so they survive shell argv,
# wandb run names, and on-disk run-dir names without quoting.  The mapping
# covers the 8-game default benchmark scope from
# ``baselines/README.md`` § "Gym-V benchmark scope" (commit 4f97dd6).
# The 5 dropped games are intentionally NOT here — re-add them only when
# the §11.2-Option-A back-stop is invoked.

GYMV_TEMPORAL_GAMES: Dict[str, str] = {
    # Phase-1 source (4 games — see training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1,
    # refreshed 2026-05-03 PM from new Cold-start-out-gymv/latest 4-backbone teacher data)
    "gymv_thunder_force_iii": "Temporal/ThunderForceIII-v0",
    "gymv_altered_beast": "Temporal/AlteredBeast-v0",
    "gymv_columns": "Temporal/Columns-v0",
    "gymv_dynamite_headdy": "Temporal/DynamiteHeaddy-v0",
    # Phase-2 holdouts (4 games — see §7.1)
    "gymv_streets_of_rage_2": "Temporal/StreetsOfRage2-v0",
    "gymv_space_harrier_ii": "Temporal/SpaceHarrierII-v0",
    "gymv_airstriker": "Temporal/Airstriker-v0",
    "gymv_strider": "Temporal/Strider-v0",
}


def is_gymv_temporal_game(slug: str) -> bool:
    """True if *slug* is one of the 8 wired Gym-V Temporal/* slugs."""
    return slug in GYMV_TEMPORAL_GAMES


def gymv_env_id_for(slug: str) -> str:
    """Resolve a slug (e.g. ``gymv_columns``) to the gym_v env id
    (``Temporal/Columns-v0``).  Raises ``KeyError`` if unknown."""
    return GYMV_TEMPORAL_GAMES[slug]


# ---------------------------------------------------------------------------
# Factory + wrapper
# ---------------------------------------------------------------------------


def make_gymv_temporal_env(
    slug_or_env_id: str,
    *,
    max_steps: int = 80,
    frame_skip: int = 8,
    seed: Optional[int] = None,
) -> "GymVTemporalNLWrapper":
    """Build a :class:`GymVTemporalNLWrapper` ready for the runner.

    Parameters
    ----------
    slug_or_env_id
        Either a short slug from :data:`GYMV_TEMPORAL_GAMES` (preferred —
        used by ``trainer.coevolution.episode_runner``'s ``--games`` CLI
        path) or a raw env id like ``"Temporal/Airstriker-v0"`` (used by
        ad-hoc smoke tests).
    max_steps
        Episode horizon enforced by the wrapper itself; ``RetroGymVEnv``
        does not cap episodes natively.  Default ``80`` matches the
        Gym-V cold-start sweep (``baselines/README.md`` § "Gym-V benchmark
        scope").
    frame_skip
        Hold each agent action for this many emulator frames.  Default
        ``8`` is the value the 4-backbone success-rate sweep used to land
        the 8-game benchmark scope; 1.33 s of game time per step at
        ``frame_skip=1`` is below the 5–10 s title-screen-to-first-reward
        window for most ROMs, so leaving this at ``1`` puts most games at
        zero reward.
    seed
        Forwarded to ``env.reset(seed=...)``.  Stable-retro is
        deterministic at the emulator level when seeded.
    """
    env_id = (
        GYMV_TEMPORAL_GAMES[slug_or_env_id]
        if slug_or_env_id in GYMV_TEMPORAL_GAMES
        else slug_or_env_id
    )

    try:
        import gym_v  # noqa: F401  (registers Temporal/* envs on import)
        import gym_v.envs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "make_gymv_temporal_env() requires gym_v + stable-retro + the "
            "Mega Drive ROMs imported via install/install_gymv.sh + "
            "install/gymv_temporal_patch/apply_patch.sh.  Got: %s" % exc
        ) from exc

    import gym_v as _gym_v  # second alias keeps mypy happy below

    env = _gym_v.make(env_id)

    if frame_skip and frame_skip > 1:
        try:
            from gym_v.wrappers import StochasticFrameSkip
            env = StochasticFrameSkip(env, n=frame_skip, stickprob=0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "frame_skip=%d requested but StochasticFrameSkip unavailable "
                "(%s); falling back to skip=1.",
                frame_skip, exc,
            )

    return GymVTemporalNLWrapper(
        env,
        env_id=env_id,
        max_steps=max_steps,
        reset_seed=seed,
    )


class GymVTemporalNLWrapper:
    """Adapt a Gym-V multi-agent ``Temporal/*`` env to the co-evolution
    runner's single-agent NL contract.

    See module docstring for the contract details.  This wrapper does
    not perform any LLM calls — vision / action selection happens in the
    runner; we only translate observations and route action strings back
    to the underlying multi-agent ``env.step({agent_id: action})``.
    """

    def __init__(
        self,
        env: Any,
        *,
        env_id: str,
        max_steps: int = 80,
        reset_seed: Optional[int] = None,
    ) -> None:
        self._env = env
        self._env_id = env_id
        self._max_steps = max_steps
        self._reset_seed = reset_seed
        self._agent_id: Optional[str] = None
        self._action_names: List[str] = []
        self._step_count: int = 0
        self._last_reward: float = 0.0

        self._game_slug = next(
            (s for s, eid in GYMV_TEMPORAL_GAMES.items() if eid == env_id),
            env_id.replace("/", "_").replace("-v0", "").lower(),
        )

        try:
            from gymv_wrapper.temporal_visual_grounding import (
                build_temporal_visual_schema,
                TEMPORAL_GAME_SPECS,
            )
            self._build_schema = build_temporal_visual_schema
            self._spec = TEMPORAL_GAME_SPECS.get(env_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gymv_wrapper.temporal_visual_grounding unavailable (%s); "
                "structured_state will fall back to a minimal summary.",
                exc,
            )
            self._build_schema = None
            self._spec = None

    @property
    def env(self):
        return self._env

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    @property
    def action_names(self) -> List[str]:
        return list(self._action_names)

    def _normalise_obs(self, odict: Any, info_dict: Any) -> Tuple[Any, Dict[str, Any]]:
        """Pull the single-agent ``Observation`` out of gym_v's multi-agent
        return dict.  Stores ``self._agent_id`` on first call."""
        if self._agent_id is None:
            try:
                self._agent_id = next(iter(odict))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"GymVTemporalNLWrapper: env.reset() did not return a "
                    f"multi-agent dict ({type(odict).__name__}); cannot "
                    f"infer agent_id"
                ) from exc

        obs = odict[self._agent_id]
        if isinstance(info_dict, dict) and self._agent_id in info_dict:
            info = dict(info_dict[self._agent_id])
        elif isinstance(info_dict, dict):
            info = dict(info_dict)
        else:
            info = {}
        return obs, info

    def _structured_state(self, obs: Any) -> Dict[str, Any]:
        """Return a JSON-serialisable summary the runner can drop straight
        into ``info["structured_state"]``.  Prefers the project's
        :func:`build_temporal_visual_schema`; falls back to a minimal
        dict when that's unavailable."""
        if self._build_schema is not None:
            try:
                return dict(self._build_schema(self._env_id, obs))
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "build_temporal_visual_schema(%s) failed: %s",
                    self._env_id, exc,
                )

        meta = dict(getattr(obs, "metadata", None) or {})
        return {
            "schema_kind": "gymv.temporal_visual_grounding.fallback",
            "gym_env_id": self._env_id,
            "display_name": getattr(self._spec, "display_name", self._env_id),
            "genre": getattr(self._spec, "genre", "unknown"),
            "simulation": {
                "frame_index": meta.get("frame_index"),
                "episode_reward": meta.get("episode_reward"),
                "step_reward": meta.get("step_reward"),
                "last_action": meta.get("last_action"),
            },
            "control": {
                "available_actions": list(meta.get("available_actions") or []),
            },
            "ram_watch": dict(meta.get("ram_watch") or {}),
        }

    def _obs_to_nl(self, obs: Any) -> str:
        """Build the natural-language observation string the runner reads
        as ``obs_nl``.  Uses ``obs.text`` (always populated by the patched
        ``RetroGymVEnv`` — see ``install/gymv_temporal_patch/README.md``)
        plus an action-affordance hint."""
        text = getattr(obs, "text", None) or ""
        if not text:
            text = (
                f"[{self._env_id} step {self._step_count} — text channel empty; "
                f"image-only observation]"
            )
        if self._action_names:
            text += (
                f"\n\nValid actions: {', '.join(self._action_names[:25])}. "
                f"Choose one."
            )
        return text

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if seed is None:
            seed = self._reset_seed

        try:
            if seed is not None:
                odict, info_dict = self._env.reset(seed=seed)
            else:
                odict, info_dict = self._env.reset()
        except TypeError:
            odict, info_dict = self._env.reset()

        obs, info = self._normalise_obs(odict, info_dict)
        self._step_count = 0
        self._last_reward = 0.0

        meta = dict(getattr(obs, "metadata", None) or {})
        self._action_names = [str(a) for a in (meta.get("available_actions") or [])][:25]
        if not self._action_names:
            self._action_names = ["NOOP"]

        nl = self._obs_to_nl(obs)
        info.update({
            "state_natural_language": nl,
            "action_names": self._action_names,
            "structured_state": self._structured_state(obs),
            "env_name": "gymv_temporal",
            "game_name": self._game_slug,
            "raw_obs": obs,
            "image": getattr(obs, "image", None),
            "gym_env_id": self._env_id,
            "step": 0,
        })
        return nl, info

    def step(
        self,
        action: Any,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._agent_id is None:
            raise RuntimeError(
                "GymVTemporalNLWrapper.step() called before reset()"
            )

        action_str = str(action)
        if self._action_names and action_str not in self._action_names:
            try:
                idx = int(action_str)
                if 0 <= idx < len(self._action_names):
                    action_str = self._action_names[idx]
            except (TypeError, ValueError):
                pass

        try:
            odict, reward_dict, term_dict, trunc_dict, info_dict = self._env.step(
                {self._agent_id: action_str}
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "GymVTemporalNLWrapper(%s).step(%r) failed: %s",
                self._env_id, action_str, exc,
            )
            raise

        obs, info = self._normalise_obs(odict, info_dict)

        def _scalar(d: Any, default: Any) -> Any:
            if isinstance(d, dict) and self._agent_id in d:
                return d[self._agent_id]
            if isinstance(d, dict) and d:
                return next(iter(d.values()))
            return default if d is None else d

        reward = float(_scalar(reward_dict, 0.0))
        terminated = bool(_scalar(term_dict, False))
        truncated = bool(_scalar(trunc_dict, False))

        self._step_count += 1
        self._last_reward = reward

        if self._step_count >= self._max_steps:
            truncated = True

        meta = dict(getattr(obs, "metadata", None) or {})
        avail = [str(a) for a in (meta.get("available_actions") or [])][:25]
        if avail:
            self._action_names = avail

        nl = self._obs_to_nl(obs)
        info.update({
            "state_natural_language": nl,
            "action_names": self._action_names,
            "structured_state": self._structured_state(obs),
            "env_name": "gymv_temporal",
            "game_name": self._game_slug,
            "raw_obs": obs,
            "image": getattr(obs, "image", None),
            "gym_env_id": self._env_id,
            "step": self._step_count,
            "last_action": action_str,
            "last_reward": reward,
        })
        return nl, reward, terminated, truncated, info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
