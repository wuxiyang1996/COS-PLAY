"""Smoke test for the `gymv` conda env (ModalMinds/gym-v)."""
from __future__ import annotations

import sys

failures: list[tuple[str, str]] = []


def check(label: str, fn, required: bool = True) -> None:
    try:
        out = fn()
        print(f"  [OK]   {label}{(': ' + str(out)) if out else ''}")
    except Exception as exc:
        if required:
            failures.append((label, str(exc)))
            print(f"  [FAIL] {label}: {exc}")
        else:
            print(f"  [WARN] {label}: {exc}")


print(f"Python {sys.version.split()[0]}\n")

print("Core:")
check("gymnasium",          lambda: __import__('gymnasium').__version__)
check("pydantic",           lambda: __import__('pydantic').__version__)
check("scipy",              lambda: __import__('scipy').__version__)
check("networkx",           lambda: __import__('networkx').__version__)
check("matplotlib",         lambda: __import__('matplotlib').__version__)
print()

print("gym-v package:")
check("gym_v",              lambda: (__import__('gym_v', fromlist=['*']), "imported")[-1])


def _gymv_list() -> str:
    import gym_v
    # gym_v.envs.registration keeps a registry; just import registration module
    from gym_v.envs import registration as _reg  # noqa: F401
    return "gym_v.envs.registration imported"
check("gym-v registration",  _gymv_list)


def _gymv_make_ticTacToe() -> str:
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gym_v
    env = gym_v.make("Games/TicTacToe-v0")
    obs, info = env.reset(seed=0)
    env.close()
    keys = list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
    return f"make(Games/TicTacToe-v0) ok, obs keys={keys}"
check("gym-v make TicTacToe", _gymv_make_ticTacToe)
print()

print("Optional extras:")
check("textarena",          lambda: (__import__('textarena', fromlist=['*']), "imported")[-1], required=False)
check("pettingzoo",         lambda: __import__('pettingzoo').__version__, required=False)
check("minigrid",           lambda: __import__('minigrid').__version__, required=False)
check("miniworld",          lambda: (__import__('miniworld', fromlist=['*']), "imported")[-1], required=False)
print()

print("=" * 50)
if failures:
    print(f"{len(failures)} REQUIRED check(s) FAILED:")
    for label, err in failures:
        print(f"  - {label}: {err}")
    sys.exit(1)
print("All required checks passed.")
print("=" * 50)
