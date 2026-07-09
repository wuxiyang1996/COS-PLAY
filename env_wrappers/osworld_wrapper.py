"""
Gymnasium-compatible wrapper for OSWorld (xlang-ai/OSWorld).

Wraps OSWorld's DesktopEnv to provide a standard Gymnasium API:

    from env_wrappers.osworld_wrapper import OSWorldGymWrapper

    env = OSWorldGymWrapper(provider_name="docker")
    obs, info = env.reset(options={"task_id": "94d95f96-..."})
    obs, reward, term, trunc, info = env.step("pyautogui.click(960, 540)")

OSWorld's native DesktopEnv already subclasses gym.Env but deviates
from the Gymnasium spec in several ways:

  - reset() requires a ``task_config`` dict (not seed/options)
  - step()  returns old-style 4-tuple (obs, reward, done, info)
  - No formal observation_space / action_space as gymnasium.spaces
  - Reward is always 0 during stepping; evaluation is a separate call

This wrapper fixes all of the above and adds:

  - Gymnasium 5-tuple step returns (obs, reward, terminated, truncated, info)
  - Task catalog: load tasks from JSON, cycle or select by ID
  - Proper observation dict (screenshot, a11y_tree, instruction)
  - Max-step truncation
  - Auto-evaluation on DONE/FAIL for reward signals

Requirements:
  - OSWorld must be installed: ``pip install desktop-env`` or clone + pip install
  - A VM provider must be configured (vmware, docker, or aws)
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# OSWorld actions that signal episode end
_TERMINAL_ACTIONS = {"DONE", "FAIL", "WAIT"}

# Default task for smoke-testing (from OSWorld quickstart.py)
_DEFAULT_TASK = {
    "id": "94d95f96-9699-4208-98ba-3c3119edf9c2",
    "instruction": "I want to install Spotify on my current system. Could you please help me?",
    "config": [
        {
            "type": "execute",
            "parameters": {
                "command": [
                    "python",
                    "-c",
                    "import pyautogui; import time; pyautogui.click(960, 540); time.sleep(0.5);",
                ]
            },
        }
    ],
    "evaluator": {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "which spotify",
        },
        "expected": {
            "type": "rule",
            "rules": {"include": ["spotify"], "exclude": ["not found"]},
        },
    },
}


def load_task_catalog(
    path: str | Path,
    *,
    domain: str | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Load tasks from an OSWorld JSON catalog file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file (e.g. ``evaluation_examples/test_all.json``).
    domain : str, optional
        Filter to tasks from a specific domain (e.g. ``"libreoffice_calc"``).
    limit : int, optional
        Max number of tasks to return.

    Returns
    -------
    List of task config dicts ready for ``DesktopEnv.reset(task_config=...)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task catalog not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples_root = path.parent / "examples"

    def _resolve_entry(entry, default_domain: str | None) -> Dict[str, Any] | None:
        """Resolve one catalog entry into a full task-config dict.

        Catalog files such as ``test_small.json`` enumerate task *IDs*
        (strings) grouped by domain. The full configuration for each task
        lives at ``evaluation_examples/examples/<domain>/<id>.json``. This
        helper accepts either raw dicts (already-inlined tasks) or strings
        (IDs to look up on disk) and returns a usable task dict.
        """
        if isinstance(entry, dict):
            return entry
        if not isinstance(entry, str):
            return None
        candidates = []
        if default_domain:
            candidates.append(examples_root / default_domain / f"{entry}.json")
        candidates.extend(examples_root.glob(f"*/{entry}.json"))
        for cand in candidates:
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                except Exception:
                    continue
        return None

    if isinstance(data, dict):
        tasks: List[Dict[str, Any]] = []
        for domain_name, domain_tasks in data.items():
            if domain and domain_name != domain:
                continue
            entries = []
            if isinstance(domain_tasks, list):
                entries = list(domain_tasks)
            elif isinstance(domain_tasks, dict):
                entries = list(domain_tasks.values())
            for entry in entries:
                resolved = _resolve_entry(entry, domain_name)
                if resolved is not None:
                    tasks.append(resolved)
        data = tasks
    elif isinstance(data, list):
        resolved_list: List[Dict[str, Any]] = []
        for entry in data:
            r = _resolve_entry(entry, None)
            if r is not None:
                resolved_list.append(r)
        data = resolved_list
        if domain:
            data = [t for t in data if t.get("domain", "") == domain]

    if limit:
        data = data[:limit]

    return data


class OSWorldGymWrapper:
    """Gymnasium-compatible wrapper around OSWorld's DesktopEnv.

    Provides standard ``reset()`` / ``step()`` interface with 5-tuple
    returns, task catalog management, and automatic evaluation.

    Parameters
    ----------
    provider_name : str
        VM provider: ``"vmware"``, ``"docker"``, ``"aws"``, etc.
    path_to_vm : str, optional
        Path to VM image (provider-specific).
    os_type : str
        Guest OS type: ``"Ubuntu"`` or ``"Windows"``.
    action_space_type : str
        OSWorld action space: ``"pyautogui"`` or ``"computer_13"``.
    headless : bool
        Run VM without GUI.
    max_steps : int
        Max steps per episode before truncation.
    require_a11y_tree : bool
        Include accessibility tree in observations.
    require_terminal : bool
        Include terminal output in observations.
    auto_evaluate : bool
        Call ``DesktopEnv.evaluate()`` when agent sends DONE.
    task_catalog : str or list, optional
        Path to JSON task catalog, or a pre-loaded list of task dicts.
        If None, only the default smoke-test task is available.
    task_domain : str, optional
        Filter task catalog to this domain.
    task_shuffle : bool
        Shuffle task order on each full cycle through the catalog.
    screen_size : tuple of int
        VM screen resolution (width, height).
    pause_after_action : float
        Seconds to wait after each action (passed to DesktopEnv.step).
    client_password : str
        VM sudo/login password.
    enable_proxy : bool
        Enable proxy support for tasks that need it.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        provider_name: str = "vmware",
        path_to_vm: str | None = None,
        os_type: str = "Ubuntu",
        action_space_type: str = "pyautogui",
        headless: bool = False,
        max_steps: int = 15,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        auto_evaluate: bool = True,
        task_catalog: str | Path | List[Dict[str, Any]] | None = None,
        task_domain: str | None = None,
        task_shuffle: bool = False,
        screen_size: Tuple[int, int] = (1920, 1080),
        pause_after_action: float = 2.0,
        client_password: str = "",
        enable_proxy: bool = False,
    ):
        self._provider_name = provider_name
        self._path_to_vm = path_to_vm
        self._os_type = os_type
        self._action_space_type = action_space_type
        self._headless = headless
        self._max_steps = max_steps
        self._auto_evaluate = auto_evaluate
        self._pause_after_action = pause_after_action
        self._task_shuffle = task_shuffle

        # Lazy-init the native env (deferred so import doesn't require OSWorld)
        self._env = None
        self._env_kwargs = dict(
            provider_name=provider_name,
            path_to_vm=path_to_vm,
            os_type=os_type,
            action_space=action_space_type,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            screen_size=screen_size,
            client_password=client_password,
            enable_proxy=enable_proxy,
        )

        # Task catalog
        if isinstance(task_catalog, (str, Path)):
            self._tasks = load_task_catalog(task_catalog, domain=task_domain)
        elif isinstance(task_catalog, list):
            self._tasks = list(task_catalog)
        else:
            self._tasks = [_DEFAULT_TASK]

        self._task_index = 0
        self._task_cycle_count = 0

        # Episode state
        self._step_count = 0
        self._episode_count = 0
        self._current_task: Dict[str, Any] | None = None
        self._last_obs: Dict[str, Any] | None = None
        self._terminated = False
        self._truncated = False

    def _ensure_env(self):
        """Lazy-initialize the native DesktopEnv."""
        if self._env is not None:
            return
        try:
            from desktop_env.desktop_env import DesktopEnv
        except ImportError:
            raise ImportError(
                "OSWorld not installed. Install with: pip install desktop-env\n"
                "Or clone https://github.com/xlang-ai/OSWorld and install."
            )

        kwargs = {k: v for k, v in self._env_kwargs.items() if v is not None}
        self._env = DesktopEnv(**kwargs)

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        """The loaded task catalog."""
        return self._tasks

    @property
    def current_task(self) -> Dict[str, Any] | None:
        """The task config for the current episode."""
        return self._current_task

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    @property
    def action_names(self) -> List[str]:
        """Valid special actions (non-pyautogui)."""
        return ["DONE", "FAIL", "WAIT"]

    def _select_task(self, task_id: str | None = None) -> Dict[str, Any]:
        """Pick the next task from the catalog or by ID."""
        if task_id:
            for t in self._tasks:
                if t.get("id") == task_id:
                    return t
            raise ValueError(
                f"Task '{task_id}' not found in catalog "
                f"({len(self._tasks)} tasks loaded)"
            )

        if self._task_index >= len(self._tasks):
            self._task_index = 0
            self._task_cycle_count += 1
            if self._task_shuffle:
                random.shuffle(self._tasks)

        task = self._tasks[self._task_index]
        self._task_index += 1
        return task

    @staticmethod
    def _normalize_obs(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OSWorld observation into a clean dict.

        OSWorld's :meth:`PythonController.get_screenshot` returns the raw
        PNG bytes of the VM's framebuffer (body of the ``GET /screenshot``
        response), not a numpy array. We decode the PNG with PIL into an
        ``HxWx3`` uint8 ``np.ndarray`` so downstream callers can treat
        ``obs["screenshot"]`` like any other gym frame.
        """
        screenshot = raw_obs.get("screenshot")
        if isinstance(screenshot, (bytes, bytearray)):
            try:
                from io import BytesIO

                from PIL import Image as _PILImage

                img = _PILImage.open(BytesIO(screenshot)).convert("RGB")
                screenshot = np.asarray(img, dtype=np.uint8)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to decode screenshot bytes: %s", exc)
                screenshot = None
        elif screenshot is not None and not isinstance(screenshot, np.ndarray):
            try:
                screenshot = np.asarray(screenshot)
            except Exception:
                screenshot = None

        return {
            "screenshot": screenshot,
            "accessibility_tree": raw_obs.get("accessibility_tree") or "",
            "terminal": raw_obs.get("terminal") or "",
            "instruction": raw_obs.get("instruction") or "",
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment for a new episode.

        Parameters
        ----------
        seed : int, optional
            Random seed (used for task shuffling).
        options : dict, optional
            - ``task_id``: select a specific task by ID
            - ``task_config``: pass a full task config dict directly
            - ``task_index``: select task by catalog index

        Returns
        -------
        observation : dict
            Keys: ``screenshot`` (np.ndarray HxWx3), ``accessibility_tree``
            (str), ``terminal`` (str), ``instruction`` (str).
        info : dict
            Task metadata and wrapper state.
        """
        self._ensure_env()
        options = options or {}

        if seed is not None:
            random.seed(seed)

        # Select task
        if "task_config" in options:
            task = options["task_config"]
        elif "task_index" in options:
            idx = options["task_index"]
            task = self._tasks[idx % len(self._tasks)]
        else:
            task = self._select_task(options.get("task_id"))

        self._current_task = task
        self._step_count = 0
        self._episode_count += 1
        self._terminated = False
        self._truncated = False

        raw_obs = self._env.reset(task_config=task)
        obs = self._normalize_obs(raw_obs)
        self._last_obs = obs

        info = {
            "task_id": task.get("id", ""),
            "instruction": task.get("instruction", ""),
            "domain": task.get("domain", ""),
            "episode": self._episode_count,
            "task_index": max(0, self._task_index - 1),
            "num_tasks": len(self._tasks),
            "max_steps": self._max_steps,
            "provider": self._provider_name,
        }

        return obs, info

    def step(
        self,
        action: str | Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute an action in the desktop environment.

        Parameters
        ----------
        action : str or dict
            A pyautogui command string (e.g. ``"pyautogui.click(960, 540)"``),
            or a special action (``"DONE"``, ``"FAIL"``, ``"WAIT"``),
            or a dict with ``action_type`` key for computer_13 format.

        Returns
        -------
        observation : dict
            Same structure as reset().
        reward : float
            0.0 during episode; on DONE with auto_evaluate, returns evaluation
            score (0.0 or 1.0).
        terminated : bool
            True if agent sent DONE or FAIL.
        truncated : bool
            True if max_steps reached.
        info : dict
            Step metadata.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._terminated or self._truncated:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        self._step_count += 1

        raw_obs, raw_reward, raw_done, raw_info = self._env.step(
            action, pause=self._pause_after_action
        )
        obs = self._normalize_obs(raw_obs)
        self._last_obs = obs

        terminated = bool(raw_done)
        truncated = False
        reward = float(raw_reward)

        if not terminated and self._step_count >= self._max_steps:
            truncated = True

        # Auto-evaluate on terminal actions
        eval_score = None
        if terminated and self._auto_evaluate:
            try:
                eval_score = float(self._env.evaluate())
                reward = eval_score
            except Exception as e:
                logger.warning("Auto-evaluation failed: %s", e)
                eval_score = None

        self._terminated = terminated
        self._truncated = truncated

        info = {
            **raw_info,
            "step": self._step_count,
            "task_id": self._current_task.get("id", "") if self._current_task else "",
            "instruction": self._current_task.get("instruction", "") if self._current_task else "",
            "action": action,
            "eval_score": eval_score,
        }

        return obs, reward, terminated, truncated, info

    def evaluate(self) -> float:
        """Manually trigger OSWorld task evaluation.

        Returns
        -------
        float
            Evaluation score (typically 0.0 or 1.0).
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized.")
        return float(self._env.evaluate())

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:
        """Return the current screenshot as an RGB numpy array."""
        if self._env is None:
            return None
        if mode == "rgb_array":
            return self._env.render(mode="rgb_array")
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        """Shut down the VM and release resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning("Error closing OSWorld env: %s", e)
            self._env = None

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return (
            f"OSWorldGymWrapper("
            f"provider={self._provider_name!r}, "
            f"tasks={len(self._tasks)}, "
            f"max_steps={self._max_steps})"
        )

    # ------------------------------------------------------------------
    # Convenience: iterate through all tasks
    # ------------------------------------------------------------------

    def task_ids(self) -> List[str]:
        """Return all task IDs in the catalog."""
        return [t.get("id", f"task_{i}") for i, t in enumerate(self._tasks)]

    def task_instructions(self) -> Dict[str, str]:
        """Return a mapping of task_id -> instruction."""
        return {
            t.get("id", f"task_{i}"): t.get("instruction", "")
            for i, t in enumerate(self._tasks)
        }
