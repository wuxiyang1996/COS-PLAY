"""ALFWorld text environment wrapper for COS-PLAY.

The wrapper keeps ALFWorld isolated in its own conda env but exposes the same
reset/step shape used by the other natural-language wrappers:

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("open fridge 1")

Only text-mode ALFWorld is used by default. Visual/THOR runs require installing
the env with ALFWORLD_EXTRAS=vis or ALFWORLD_EXTRAS=full and a working display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ALFWORLD_SPLITS = (
    "train",
    "eval_in_distribution",
    "eval_out_of_distribution",
)


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _commands_from_info(info: Dict[str, Any]) -> List[str]:
    commands = info.get("admissible_commands")
    commands = _first(commands)
    if commands is None:
        return []
    return [str(cmd) for cmd in commands]


def alfworld_obs_to_natural_language(
    obs: Any,
    info: Optional[Dict[str, Any]] = None,
    *,
    include_admissible: bool = True,
    max_actions: int = 40,
) -> str:
    """Convert one ALFWorld observation batch into compact text."""
    parts = [str(_first(obs)).strip()]
    if include_admissible and info:
        commands = _commands_from_info(info)[:max_actions]
        if commands:
            parts.append("Admissible actions: " + "; ".join(commands))
    return "\n\n".join(part for part in parts if part)


@dataclass
class ALFWorldNLWrapper:
    """Natural-language wrapper around ALFWorld's batched text env."""

    env: Any
    include_admissible: bool = True
    max_actions: int = 40
    max_steps: int = 50
    _step_count: int = 0
    _last_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def action_names(self) -> List[str]:
        return _commands_from_info(self._last_info)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        del args, kwargs
        self._step_count = 0
        obs, info = self.env.reset()
        self._last_info = dict(info or {})
        text = alfworld_obs_to_natural_language(
            obs,
            self._last_info,
            include_admissible=self.include_admissible,
            max_actions=self.max_actions,
        )
        return text, self._build_info(info)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        obs, scores, dones, info = self.env.step([str(action)])
        self._last_info = dict(info or {})
        reward = float(_first(scores) or 0.0)
        terminated = bool(_first(dones))
        truncated = self._step_count >= self.max_steps and not terminated
        text = alfworld_obs_to_natural_language(
            obs,
            self._last_info,
            include_admissible=self.include_admissible,
            max_actions=self.max_actions,
        )
        out_info = self._build_info(info)
        out_info["last_action"] = str(action)
        return text, reward, terminated, truncated, out_info

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _build_info(self, info: Any) -> Dict[str, Any]:
        result = dict(info or {})
        result["env"] = "alfworld"
        result["step"] = self._step_count
        result["admissible_actions"] = _commands_from_info(result)
        return result


def make_alfworld_env(
    *,
    split: str = "eval_out_of_distribution",
    env_type: str = "AlfredTWEnv",
    batch_size: int = 1,
    max_steps: int = 50,
    include_admissible: bool = True,
    config_path: Optional[str] = None,
) -> ALFWorldNLWrapper:
    """Create a text-mode ALFWorld NL wrapper.

    Args:
        split: One of ``train``, ``eval_in_distribution``, or
            ``eval_out_of_distribution``.
        env_type: ALFWorld environment type. Defaults to ``AlfredTWEnv``.
        batch_size: ALFWorld batch size. COS-PLAY wrappers expect 1.
        max_steps: Wrapper-side truncation horizon.
        include_admissible: Include valid text commands in observations.
        config_path: Optional path passed to ALFWorld's ``load_config``.
    """
    if split not in ALFWORLD_SPLITS:
        raise ValueError(f"split must be one of {ALFWORLD_SPLITS}, got {split!r}")
    if batch_size != 1:
        raise ValueError("COS-PLAY ALFWorldNLWrapper currently expects batch_size=1")

    try:
        from alfworld.agents.environment import get_environment  # type: ignore
        import alfworld.agents.modules.generic as generic  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise ImportError(
            "ALFWorld is not installed. Run install/install_alfworld.sh and "
            "activate the 'alfworld' conda env."
        ) from exc

    config = generic.load_config(config_path) if config_path else generic.load_config()
    config.setdefault("env", {})["type"] = env_type
    env = get_environment(env_type)(config, train_eval=split)
    env = env.init_env(batch_size=batch_size)
    return ALFWorldNLWrapper(
        env=env,
        include_admissible=include_admissible,
        max_steps=max_steps,
    )


__all__ = [
    "ALFWORLD_SPLITS",
    "ALFWorldNLWrapper",
    "alfworld_obs_to_natural_language",
    "make_alfworld_env",
]
