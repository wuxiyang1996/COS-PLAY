"""
Shared fixtures for OSWorld wrapper tests.

Provides mock DesktopEnv, pre-configured wrapper instances, and
temporary task catalog files.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure env_wrappers is importable
_CODEBASE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEBASE_ROOT))


# ======================================================================
# Mock DesktopEnv
# ======================================================================

class MockDesktopEnv:
    """Mimics OSWorld's DesktopEnv interface without requiring a VM.

    Produces synthetic screenshots, accessibility trees, and terminal
    output that exercise all observation paths in the wrappers.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._step = 0
        self._instruction = ""
        self._task_id = ""
        self._done = False
        self._screen_w = kwargs.get("screen_size", (1920, 1080))[0]
        self._screen_h = kwargs.get("screen_size", (1920, 1080))[1]
        self._screen = self._make_gradient_screen()

        self.evaluator = {}
        self.metric = None
        self.result_getter = None
        self.expected_getter = None
        self.action_history: list = []

    def _make_gradient_screen(self) -> np.ndarray:
        screen = np.zeros((self._screen_h, self._screen_w, 3), dtype=np.uint8)
        screen[:, :, 0] = np.linspace(0, 200, self._screen_w, dtype=np.uint8)
        screen[:, :, 2] = np.linspace(200, 0, self._screen_w, dtype=np.uint8)
        return screen

    def reset(self, task_config=None, seed=None, options=None):
        self._step = 0
        self._done = False
        self.action_history.clear()
        if task_config:
            self._instruction = task_config.get("instruction", "")
            self._task_id = task_config.get("id", "")
        return self._get_obs()

    def step(self, action, pause=2):
        self._step += 1
        self.action_history.append(action)
        done = False
        info = {}

        if action in ("DONE", "FAIL") or (
            isinstance(action, dict)
            and action.get("action_type") in ("DONE", "FAIL")
        ):
            done = True
            if action == "DONE" or (isinstance(action, dict) and action.get("action_type") == "DONE"):
                info["done"] = True
            else:
                info["fail"] = True

        self._done = done
        return self._get_obs(), 0.0, done, info

    def evaluate(self):
        if self.action_history and (
            self.action_history[-1] == "FAIL"
            or (isinstance(self.action_history[-1], dict) and self.action_history[-1].get("action_type") == "FAIL")
        ):
            return 0.0
        return 1.0

    def render(self, mode="rgb_array"):
        return self._screen.copy()

    def close(self):
        pass

    def _get_obs(self):
        return {
            "screenshot": self._screen.copy(),
            "accessibility_tree": self._make_a11y(),
            "terminal": self._make_terminal(),
            "instruction": self._instruction,
        }

    def _make_a11y(self) -> str:
        base = (
            '[window] "Ubuntu Desktop" {active}\n'
            '  [panel] "Top Bar" {}\n'
            '    [button] "Activities" {}\n'
            '    [label] "Apr 12, 2026" {}\n'
            '  [panel] "Desktop" {}\n'
        )
        if self._step == 0:
            base += '    [icon] "Files" {}\n    [icon] "Firefox" {}\n    [icon] "Terminal" {}\n'
        else:
            base += (
                f'    [icon] "Files" {{}}\n'
                f'    [icon] "Firefox" {{}}\n'
                f'    [dialog] "Open File" {{modal}}\n'
                f'      [button] "OK" {{focused}}\n'
                f'      [button] "Cancel" {{}}\n'
            )
        base += '  [statusbar] "Status Bar" {}\n'
        return base

    def _make_terminal(self) -> str:
        if self._step == 0:
            return ""
        return f"$ step {self._step}\nAction executed successfully.\nuser@ubuntu:~$"


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_desktop_env():
    """Return a fresh MockDesktopEnv instance."""
    return MockDesktopEnv()


@pytest.fixture
def sample_tasks() -> List[Dict[str, Any]]:
    """Return a list of sample task configs."""
    return [
        {
            "id": "task-001",
            "instruction": "Open Firefox and navigate to google.com",
            "domain": "firefox",
            "config": [],
            "evaluator": {
                "func": "check_include_exclude",
                "result": {"type": "vm_command_line", "command": "pgrep firefox"},
                "expected": {"type": "rule", "rules": {"include": ["firefox"]}},
            },
        },
        {
            "id": "task-002",
            "instruction": "Create a new spreadsheet in LibreOffice Calc",
            "domain": "libreoffice_calc",
            "config": [],
            "evaluator": {
                "func": "check_include_exclude",
                "result": {"type": "vm_command_line", "command": "echo ok"},
                "expected": {"type": "rule", "rules": {"include": ["ok"]}},
            },
        },
        {
            "id": "task-003",
            "instruction": "Install vlc media player using apt",
            "domain": "terminal",
            "config": [],
            "evaluator": {
                "func": "check_include_exclude",
                "result": {"type": "vm_command_line", "command": "which vlc"},
                "expected": {"type": "rule", "rules": {"include": ["vlc"]}},
            },
        },
    ]


@pytest.fixture
def task_catalog_file(sample_tasks, tmp_path) -> Path:
    """Write sample tasks to a temporary JSON catalog (dict-of-lists format)."""
    catalog = {}
    for t in sample_tasks:
        domain = t.get("domain", "unknown")
        catalog.setdefault(domain, []).append(t)

    path = tmp_path / "test_catalog.json"
    path.write_text(json.dumps(catalog, indent=2))
    return path


@pytest.fixture
def task_catalog_list_file(sample_tasks, tmp_path) -> Path:
    """Write sample tasks to a temporary JSON catalog (flat list format)."""
    path = tmp_path / "test_catalog_list.json"
    path.write_text(json.dumps(sample_tasks, indent=2))
    return path


@pytest.fixture
def gym_wrapper(mock_desktop_env, sample_tasks):
    """Return an OSWorldGymWrapper with a mock env injected."""
    from env_wrappers.osworld_wrapper import OSWorldGymWrapper

    env = OSWorldGymWrapper.__new__(OSWorldGymWrapper)
    env._provider_name = "mock"
    env._path_to_vm = None
    env._os_type = "Ubuntu"
    env._action_space_type = "pyautogui"
    env._headless = True
    env._max_steps = 10
    env._auto_evaluate = True
    env._pause_after_action = 0.0
    env._task_shuffle = False
    env._env_kwargs = {}
    env._tasks = sample_tasks
    env._task_index = 0
    env._task_cycle_count = 0
    env._step_count = 0
    env._episode_count = 0
    env._current_task = None
    env._last_obs = None
    env._terminated = False
    env._truncated = False
    env._env = mock_desktop_env
    return env


@pytest.fixture
def nl_wrapper(gym_wrapper):
    """Return an OSWorldNLWrapper around the gym wrapper."""
    from env_wrappers.osworld_nl_wrapper import OSWorldNLWrapper
    return OSWorldNLWrapper(gym_wrapper)
