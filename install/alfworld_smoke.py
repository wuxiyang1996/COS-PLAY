"""Smoke test for the COS-PLAY ALFWorld environment.

This intentionally starts the text-mode environment only. It verifies the
package import, config load, environment construction, reset, and one
admissible-command step when data are present.
"""

from __future__ import annotations

import random
import sys
from typing import Any


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def main() -> int:
    try:
        import alfworld  # type: ignore
        from alfworld.agents.environment import get_environment  # type: ignore
        import alfworld.agents.modules.generic as generic  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[FAIL] ALFWorld import failed: {exc}")
        return 1

    print(f"[OK] alfworld import: {getattr(alfworld, '__version__', 'unknown')}")

    try:
        config = generic.load_config()
        config.setdefault("env", {})["type"] = "AlfredTWEnv"
        env_type = config["env"]["type"]
        env = get_environment(env_type)(config, train_eval="eval_out_of_distribution")
        env = env.init_env(batch_size=1)
        obs, info = env.reset()
    except Exception as exc:
        print(f"[FAIL] ALFWorld text env reset failed: {exc}")
        print("       If this is a missing-data error, run: alfworld-download")
        return 1

    obs0 = str(_first(obs)).replace("\n", " ")[:160]
    print(f"[OK] reset: {obs0}")

    admissible = info.get("admissible_commands") if isinstance(info, dict) else None
    commands = _first(admissible) if admissible else None
    if commands:
        action = random.choice(list(commands))
    else:
        action = "look"

    try:
        next_obs, scores, dones, infos = env.step([action])
    except Exception as exc:
        print(f"[FAIL] step({action!r}) failed: {exc}")
        return 1

    print(
        "[OK] step: "
        f"action={action!r} score={_first(scores)} done={_first(dones)} "
        f"obs={str(_first(next_obs)).replace(chr(10), ' ')[:120]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
