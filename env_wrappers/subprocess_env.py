"""Subprocess-based environment wrapper.

Spawns the game environment in a child process (potentially using a different
conda env / Python interpreter) and communicates over JSON-over-pipes.  This
solves dependency conflicts where the training env (e.g. ``game-ai-agent``,
NumPy 2.x) is incompatible with a game's env (e.g. ``orak-mario``, NumPy 1.x).

Usage::

    env = SubprocessEnv(
        python="/workspace/miniconda3/envs/orak-mario/bin/python",
        game="super_mario",
        max_steps=500,
    )
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step("Jump Level: 3")
    env.close()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = str(
    Path(__file__).resolve().parent / "subprocess_env_worker.py"
)

_DEFAULT_ORAK_PYTHON = "/workspace/miniconda3/envs/orak-mario/bin/python"


class SubprocessEnv:
    """Gymnasium-style env backed by a child process."""

    def __init__(
        self,
        game: str,
        max_steps: int = 500,
        python: Optional[str] = None,
        startup_timeout: float = 120.0,
    ):
        self._game = game
        self._max_steps = max_steps
        self._python = python or os.environ.get("ORAK_PYTHON", _DEFAULT_ORAK_PYTHON)
        self._action_names: List[str] = []
        self._proc: Optional[subprocess.Popen] = None

        self._start(startup_timeout)

    def _start(self, timeout: float) -> None:
        codebase_root = str(Path(__file__).resolve().parent.parent)
        orak_src = str(Path(codebase_root).parent / "Orak" / "src")

        env = {
            **os.environ,
            "SDL_VIDEODRIVER": "dummy",
            "PYTHONPATH": os.pathsep.join(
                filter(None, [codebase_root, orak_src, os.environ.get("PYTHONPATH", "")])
            ),
        }
        if not os.environ.get("DISPLAY"):
            env["PYGLET_HEADLESS"] = "1"
        else:
            env.pop("PYGLET_HEADLESS", None)

        cmd = [
            self._python,
            _WORKER_SCRIPT,
            "--game", self._game,
            "--max-steps", str(self._max_steps),
        ]
        logger.info("SubprocessEnv: spawning %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        resp = self._read_response(timeout=timeout)
        if not resp.get("ok"):
            stderr_out = ""
            if self._proc.stderr:
                import select
                if select.select([self._proc.stderr], [], [], 2.0)[0]:
                    stderr_out = self._proc.stderr.read(4096)
            raise RuntimeError(
                f"SubprocessEnv failed to start: {resp.get('error', 'unknown')}\n"
                f"stderr: {stderr_out}"
            )
        self._action_names = resp.get("action_names", [])
        logger.info(
            "SubprocessEnv(%s) ready, pid=%d, %d actions",
            self._game, self._proc.pid, len(self._action_names),
        )

    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, timeout: float = 300.0) -> dict:
        assert self._proc and self._proc.stdout
        import select
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            self._kill()
            raise TimeoutError(
                f"SubprocessEnv({self._game}): no response within {timeout}s"
            )
        line = self._proc.stdout.readline()
        if not line:
            stderr_tail = ""
            if self._proc.stderr:
                try:
                    stderr_tail = self._proc.stderr.read(4096)
                except Exception:
                    pass
            raise RuntimeError(
                f"SubprocessEnv({self._game}): worker died unexpectedly.\n"
                f"stderr: {stderr_tail}"
            )
        return json.loads(line)

    def _call(self, req: dict, timeout: float = 300.0) -> dict:
        self._send(req)
        resp = self._read_response(timeout=timeout)
        if not resp.get("ok"):
            raise RuntimeError(
                f"SubprocessEnv({self._game}) error: {resp.get('error', 'unknown')}"
            )
        return resp

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        resp = self._call({"cmd": "reset"}, timeout=120.0)
        return resp["obs"], resp.get("info", {})

    def step(
        self,
        action: Union[str, int],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        resp = self._call({"cmd": "step", "action": str(action)})
        return (
            resp["obs"],
            resp["reward"],
            resp["terminated"],
            resp["truncated"],
            resp.get("info", {}),
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send({"cmd": "close"})
            self._proc.wait(timeout=10)
        except Exception:
            self._kill()
        self._proc = None

    def _kill(self) -> None:
        if self._proc:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return None
