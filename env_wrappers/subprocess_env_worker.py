#!/usr/bin/env python
"""Subprocess env worker — runs an Orak game env and serves it over stdin/stdout.

This script is meant to be executed by a *different* Python interpreter
(e.g. the ``orak-mario`` conda env) than the main training process.
Communication uses newline-delimited JSON over stdin/stdout.

Protocol
--------
Request  (parent -> worker, one JSON line on stdin):
    {"cmd": "reset"}
    {"cmd": "step", "action": "<action string>"}
    {"cmd": "close"}
    {"cmd": "get_action_names"}

Response (worker -> parent, one JSON line on stdout):
    {"ok": true, ...payload...}
    {"ok": false, "error": "<message>"}

The worker exits cleanly when stdin is closed or a "close" command arrives.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# We communicate JSON over fd 1 (stdout).  Game envs may print debug
# output to stdout which would corrupt the protocol.  We keep a
# reference to the *real* stdout for our JSON messages and redirect
# sys.stdout to stderr so any game prints go there instead.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr  # game prints go to stderr


def _write(obj: dict) -> None:
    _REAL_STDOUT.write(json.dumps(obj, default=str) + "\n")
    _REAL_STDOUT.flush()


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect even stderr-routed stdout to devnull during noisy env calls."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    # Add project paths so evaluate_orak / Orak imports work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    codebase_root = os.path.dirname(script_dir)
    orak_src = os.path.join(codebase_root, "..", "Orak", "src")
    for p in [codebase_root, orak_src]:
        rp = os.path.realpath(p)
        if os.path.isdir(rp) and rp not in sys.path:
            sys.path.insert(0, rp)

    from evaluate_orak.orak_nl_wrapper import make_orak_env

    with _suppress_stdout():
        env = make_orak_env(args.game, max_steps=args.max_steps)

    _write({"ok": True, "status": "ready", "action_names": env.action_names})

    # Read commands from the real stdin (fd 0).
    real_stdin = open(0, "r")

    for raw_line in real_stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _write({"ok": False, "error": f"bad json: {exc}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "reset":
                with _suppress_stdout():
                    obs, info = env.reset()
                info.pop("state_natural_language", None)
                _write({"ok": True, "obs": obs, "info": info})

            elif cmd == "step":
                action = req.get("action", "")
                with _suppress_stdout():
                    obs, reward, terminated, truncated, info = env.step(action)
                info.pop("state_natural_language", None)
                _write({
                    "ok": True,
                    "obs": obs,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                })

            elif cmd == "get_action_names":
                _write({"ok": True, "action_names": env.action_names})

            elif cmd == "close":
                with _suppress_stdout():
                    env.close()
                _write({"ok": True, "status": "closed"})
                break

            else:
                _write({"ok": False, "error": f"unknown cmd: {cmd}"})

        except Exception:
            _write({"ok": False, "error": traceback.format_exc()})

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
