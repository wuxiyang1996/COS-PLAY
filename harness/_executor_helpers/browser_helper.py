"""BrowserGym helper subprocess.

Runs in the ``browsergym`` conda env (which carries playwright +
browsergym-{core,miniwob,webarena,...}). The harness's main pipeline
talks to this helper via newline-delimited JSON-RPC over stdin/stdout
(see :mod:`harness._executor_helpers._proto`).

Operations
----------

* ``ping`` -- liveness probe (auto-injected).
* ``start`` -- ``{"task_id": str, "headless": bool=True, "slow_mo": int=0}``
  -- ``gym.make("browsergym/<task_id>", headless=..., slow_mo=...)``,
  ``env.reset()``, return the first observation summary +
  screenshot path on disk.
* ``step`` -- ``{"action": str}`` -- ``env.step(action)``, return
  observation summary + screenshot path. ``action`` is a
  BrowserGym high-level action string (e.g. ``'click("47")'``,
  ``'fill("17", "hello")'``, ``'goto("https://...")'``,
  ``'noop()'``). The high-level action set is documented at
  :mod:`browsergym.core.action.highlevel`.
* ``screenshot`` -- ``{}`` -- re-grab the current screenshot without
  stepping. Useful when the per-sample executor only does visual
  grounding hops without env mutation.
* ``close`` -- tear down the env.

Response shape (on ``ok=True``)::

    {
      "ok": true,
      "url": "file:///.../email-inbox-star-reply.html",
      "goal": "Find the email by Cecile and click the star icon...",
      "screenshot_path": "/tmp/_bg_helper_<pid>_step_<N>.png",
      "axtree_excerpt": "<truncated str repr>",
      "focused_bid": "10",
      "last_action": null,
      "last_action_error": null,
      "reward": 0.0,
      "terminated": false,
      "truncated": false,
      "step_index": 0
    }

Screenshots are written to ``/tmp/_bg_helper_<pid>_step_<N>.png`` and
the path returned to the parent. The parent reads the file (with
``PIL.Image.open``) inside the main env's interpreter so we don't
have to base64-shuttle multi-MB images through the JSON pipe.

axtree
~~~~~~

The full BrowserGym AXTree is a multi-megabyte dict that the parent
rarely needs verbatim. We return only an ``axtree_excerpt`` (truncated
``str(axtree)[:axtree_chars]``) by default; callers that need the
full tree can request it via ``axtree_chars=-1``. The full
``axtree_object`` JSON dump is also written to
``/tmp/_bg_helper_<pid>_step_<N>.axtree.json`` alongside the
screenshot for callers that want to read it directly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the harness package to sys.path so we can import _proto. The
# helper is launched as a standalone python script, so the parent's
# cwd / PYTHONPATH may or may not include the harness package; play
# it safe by climbing two levels up from this file.
_THIS_DIR = Path(__file__).resolve().parent
_HARNESS_PARENT = _THIS_DIR.parent.parent  # .../Multi-hop-Reasoning-VLM-Agent
if str(_HARNESS_PARENT) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PARENT))

from harness._executor_helpers._proto import run_helper_loop  # noqa: E402

logger = logging.getLogger("harness.executor_helpers.browser_helper")
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[browser_helper] %(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Per-process state
# ---------------------------------------------------------------------------


class _State:
    def __init__(self) -> None:
        self.env: Any = None
        self.task_id: str = ""
        self.step_index: int = 0
        self.last_obs: Dict[str, Any] = {}

    def reset_state(self) -> None:
        self.step_index = 0
        self.last_obs = {}


_state = _State()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_screenshot(screenshot: Any, step_index: int) -> Optional[str]:
    """Write the screenshot ndarray to /tmp and return its path."""
    if screenshot is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        if not isinstance(screenshot, np.ndarray):
            return None
        path = Path(f"/tmp/_bg_helper_{os.getpid()}_step_{step_index:04d}.png")
        Image.fromarray(screenshot).save(path)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_screenshot failed: %r", exc)
        return None


def _save_axtree(axtree: Any, step_index: int) -> Optional[str]:
    if axtree is None:
        return None
    try:
        path = Path(f"/tmp/_bg_helper_{os.getpid()}_step_{step_index:04d}.axtree.json")
        path.write_text(json.dumps(axtree, default=str))
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("save_axtree failed: %r", exc)
        return None


def _summarize(
    obs: Dict[str, Any],
    *,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    info: Optional[Dict[str, Any]] = None,
    axtree_chars: int = 4000,
) -> Dict[str, Any]:
    """Pack an observation dict + step result into the JSON-safe response."""
    screenshot = obs.get("screenshot")
    axtree = obs.get("axtree_object")
    sshot_path = _save_screenshot(screenshot, _state.step_index)
    axtree_path = _save_axtree(axtree, _state.step_index)

    if axtree_chars == -1:
        axtree_excerpt = str(axtree) if axtree is not None else ""
    else:
        excerpt = str(axtree) if axtree is not None else ""
        axtree_excerpt = excerpt[: max(0, int(axtree_chars))]

    return {
        "ok": True,
        "url": obs.get("url"),
        "goal": (
            obs.get("goal")
            or (obs.get("goal_object") and str(obs["goal_object"])[:500])
            or ""
        ),
        "screenshot_path": sshot_path,
        "axtree_path": axtree_path,
        "axtree_excerpt": axtree_excerpt,
        "axtree_truncated": (
            axtree_chars != -1
            and axtree is not None
            and len(str(axtree)) > axtree_chars
        ),
        "focused_bid": obs.get("focused_element_bid"),
        "last_action": obs.get("last_action"),
        "last_action_error": obs.get("last_action_error") or "",
        "open_pages_titles": obs.get("open_pages_titles") or [],
        "open_pages_urls": obs.get("open_pages_urls") or [],
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info or {},
        "step_index": _state.step_index,
    }


# ---------------------------------------------------------------------------
# RPC handlers
# ---------------------------------------------------------------------------


def _h_start(req: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(req.get("task_id") or "").strip()
    if not task_id:
        return {"ok": False, "error": "missing task_id"}
    if "/" in task_id and not task_id.startswith("browsergym/"):
        # Tolerate "miniwob.click-button" as well as
        # "browsergym/miniwob.click-button" -- normalize to the
        # gym registry convention.
        full_id = task_id
    else:
        full_id = task_id if task_id.startswith("browsergym/") else f"browsergym/{task_id}"

    headless = bool(req.get("headless", True))
    slow_mo = int(req.get("slow_mo", 0))

    # MINIWOB_URL must be set if the task is a MiniWoB one. Default
    # to the in-tree fixtures so the parent doesn't have to set the
    # env var. WebArena / VisualWebArena tasks ignore this.
    miniwob_default = (
        "file:///workspace/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"
    )
    os.environ.setdefault("MINIWOB_URL", miniwob_default)

    if _state.env is not None:
        try:
            _state.env.close()
        except Exception:  # noqa: BLE001
            pass
        _state.env = None

    import gymnasium as gym

    # Side-effect import: register the task families. Wrapped in
    # try/except because not every browsergym install has every
    # subpackage.
    for sub in ("miniwob", "webarena", "visualwebarena", "assistantbench",
                "workarena"):
        try:
            __import__(f"browsergym.{sub}")
        except Exception:  # noqa: BLE001
            pass

    try:
        env = gym.make(full_id, headless=headless, slow_mo=slow_mo)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"gym.make({full_id!r}) failed: {type(exc).__name__}: {exc}",
        }
    _state.env = env
    _state.task_id = full_id
    _state.reset_state()

    try:
        obs, info = env.reset()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"env.reset() failed: {type(exc).__name__}: {exc}",
        }
    _state.last_obs = obs
    return _summarize(
        obs,
        info=info,
        axtree_chars=int(req.get("axtree_chars", 4000)),
    )


def _h_step(req: Dict[str, Any]) -> Dict[str, Any]:
    if _state.env is None:
        return {"ok": False, "error": "no env -- call start first"}
    action = str(req.get("action") or "").strip()
    if not action:
        return {"ok": False, "error": "missing action"}
    _state.step_index += 1
    try:
        obs, reward, terminated, truncated, info = _state.env.step(action)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"env.step({action!r}) failed: {type(exc).__name__}: {exc}",
        }
    _state.last_obs = obs
    return _summarize(
        obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        axtree_chars=int(req.get("axtree_chars", 4000)),
    )


def _h_screenshot(req: Dict[str, Any]) -> Dict[str, Any]:
    """Return a snapshot of the current obs without stepping the env."""
    if _state.env is None:
        return {"ok": False, "error": "no env -- call start first"}
    obs = _state.last_obs
    if not obs:
        return {"ok": False, "error": "no last_obs cached"}
    return _summarize(obs, axtree_chars=int(req.get("axtree_chars", 4000)))


def _h_close(_req: Dict[str, Any]) -> Dict[str, Any]:
    if _state.env is not None:
        try:
            _state.env.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("env.close() raised: %r", exc)
    _state.env = None
    _state.task_id = ""
    _state.reset_state()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    handlers = {
        "start": _h_start,
        "step": _h_step,
        "screenshot": _h_screenshot,
        "close": _h_close,
    }
    run_helper_loop(handlers, name="browser_helper")


if __name__ == "__main__":
    main()
