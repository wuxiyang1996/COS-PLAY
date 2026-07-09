"""
Test / demo script for the OSWorld Gymnasium wrapper.

Runs in two modes:

  1. MOCK mode (default, no VM required) — validates wrapper logic,
     NL conversion, state summaries, and task catalog loading with a
     mock DesktopEnv that returns synthetic observations.

  2. LIVE mode (--live flag, requires VM) — actually spins up OSWorld
     and runs a short episode.

Usage:
    # Mock tests (no VM, no OSWorld install needed)
    python -m env_wrappers.test_osworld_wrapper

    # Live test (requires OSWorld + VM provider)
    python -m env_wrappers.test_osworld_wrapper --live --provider docker

    # With a task catalog
    python -m env_wrappers.test_osworld_wrapper --live --catalog path/to/test_all.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure the parent directory (COS-PLAY) is on sys.path so
# ``from env_wrappers.xxx`` works when running this file directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent
if str(_CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEBASE_ROOT))

import numpy as np


# ======================================================================
# Mock DesktopEnv for offline testing
# ======================================================================

class _MockDesktopEnv:
    """Mimics OSWorld's DesktopEnv interface for testing without a VM."""

    def __init__(self, **kwargs):
        self._step = 0
        self._instruction = ""
        self._screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Draw a simple gradient so screenshots aren't blank
        self._screen[:, :, 0] = np.linspace(0, 200, 1920, dtype=np.uint8)
        self._screen[:, :, 2] = np.linspace(200, 0, 1920, dtype=np.uint8)

    def reset(self, task_config=None):
        self._step = 0
        self._instruction = (task_config or {}).get("instruction", "Test task")
        return {
            "screenshot": self._screen.copy(),
            "accessibility_tree": self._mock_a11y(),
            "terminal": "",
            "instruction": self._instruction,
        }

    def step(self, action, pause=2):
        self._step += 1
        done = action in ("DONE", "FAIL")
        info = {}
        if action == "DONE":
            info["done"] = True
        elif action == "FAIL":
            info["fail"] = True
        return (
            {
                "screenshot": self._screen.copy(),
                "accessibility_tree": self._mock_a11y(),
                "terminal": f"$ step {self._step}\nAction executed: {action}",
                "instruction": self._instruction,
            },
            0.0,
            done,
            info,
        )

    def evaluate(self):
        return 1.0

    def render(self, mode="rgb_array"):
        return self._screen.copy()

    def close(self):
        pass

    @staticmethod
    def _mock_a11y():
        return (
            '[window] "Ubuntu Desktop" {active}\n'
            '  [panel] "Top Bar" {}\n'
            '    [button] "Activities" {}\n'
            '    [label] "Apr 12, 2026" {}\n'
            '  [panel] "Desktop" {}\n'
            '    [icon] "Files" {}\n'
            '    [icon] "Firefox" {}\n'
            '    [icon] "Terminal" {}\n'
            '  [statusbar] "Status Bar" {}\n'
        )


# ======================================================================
# Tests
# ======================================================================

def test_mock_wrapper():
    """Test OSWorldGymWrapper with mock DesktopEnv."""
    from env_wrappers.osworld_wrapper import OSWorldGymWrapper

    env = OSWorldGymWrapper.__new__(OSWorldGymWrapper)
    env._provider_name = "mock"
    env._path_to_vm = None
    env._os_type = "Ubuntu"
    env._action_space_type = "pyautogui"
    env._headless = True
    env._max_steps = 5
    env._auto_evaluate = True
    env._pause_after_action = 0.0
    env._task_shuffle = False
    env._env_kwargs = {}
    env._tasks = [
        {
            "id": "test-001",
            "instruction": "Open Firefox and navigate to google.com",
            "config": [],
            "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}},
        },
        {
            "id": "test-002",
            "instruction": "Create a new spreadsheet in LibreOffice Calc",
            "config": [],
            "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}},
        },
    ]
    env._task_index = 0
    env._task_cycle_count = 0
    env._step_count = 0
    env._episode_count = 0
    env._current_task = None
    env._last_obs = None
    env._terminated = False
    env._truncated = False

    # Inject mock env
    env._env = _MockDesktopEnv()

    # --- Test reset ---
    obs, info = env.reset()
    assert isinstance(obs, dict), "obs should be dict"
    assert "screenshot" in obs, "obs should have screenshot"
    assert isinstance(obs["screenshot"], np.ndarray), "screenshot should be np.ndarray"
    assert obs["screenshot"].shape == (1080, 1920, 3), f"wrong shape: {obs['screenshot'].shape}"
    assert obs["instruction"] == "Open Firefox and navigate to google.com"
    assert info["task_id"] == "test-001"
    print("  [PASS] reset() returns correct observation and info")

    # --- Test step ---
    obs, reward, term, trunc, info = env.step("pyautogui.click(500, 300)")
    assert not term and not trunc, "should not be done after 1 step"
    assert info["step"] == 1
    print("  [PASS] step() returns correct 5-tuple")

    # --- Test truncation ---
    for i in range(4):
        obs, reward, term, trunc, info = env.step("pyautogui.click(100, 100)")
    assert trunc, "should be truncated at max_steps=5"
    print("  [PASS] truncation at max_steps works")

    # --- Test DONE action with auto-evaluate ---
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step("DONE")
    assert term, "DONE should terminate"
    assert reward == 1.0, f"auto-evaluate should return 1.0, got {reward}"
    print("  [PASS] DONE triggers evaluation, reward=1.0")

    # --- Test task cycling ---
    # After the above resets consumed task[0] and task[1], the index wraps.
    obs, info = env.reset()
    assert info["task_id"] in ("test-001", "test-002"), "should cycle through catalog"
    first_cycled = info["task_id"]
    obs2, info2 = env.reset()
    second_cycled = info2["task_id"]
    assert first_cycled != second_cycled, "consecutive resets should cycle tasks"
    print("  [PASS] task cycling works")

    # --- Test specific task selection ---
    obs, info = env.reset(options={"task_id": "test-001"})
    assert info["task_id"] == "test-001", "should select specific task"
    print("  [PASS] task_id selection works")

    env.close()
    print("  [PASS] close() works")


def test_nl_wrapper():
    """Test OSWorldNLWrapper with mock observations."""
    from env_wrappers.osworld_wrapper import OSWorldGymWrapper
    from env_wrappers.osworld_nl_wrapper import (
        OSWorldNLWrapper,
        obs_to_natural_language,
        build_osworld_state_summary,
    )

    # Test standalone NL conversion
    mock_obs = {
        "screenshot": np.zeros((720, 1280, 3), dtype=np.uint8),
        "accessibility_tree": (
            '[window] "LibreOffice Calc" {active}\n'
            '  [menubar] "Menu Bar" {}\n'
            '    [menu] "File" {}\n'
            '    [menu] "Edit" {}\n'
            '  [toolbar] "Standard" {}\n'
            '    [button] "Save" {}\n'
            '    [button] "Undo" {}\n'
            '  [grid] "Sheet1" {focused}\n'
            '    [cell] "A1" {selected}\n'
        ),
        "terminal": "$ ls\nDocuments  Downloads  Desktop",
        "instruction": "Create a budget spreadsheet with monthly expenses",
    }

    nl = obs_to_natural_language(mock_obs)
    assert "Task: Create a budget" in nl
    assert "1280x720" in nl
    assert "LibreOffice Calc" in nl or "window" in nl
    assert "ls" in nl or "Documents" in nl
    print("  [PASS] obs_to_natural_language produces correct text")

    # Test structured state summary
    summary = build_osworld_state_summary(mock_obs, step=3, last_action="click(500, 300)")
    assert summary["env"] == "osworld"
    assert summary["step"] == 3
    assert summary["has_screenshot"] is True
    assert "last_action" in summary
    print("  [PASS] build_osworld_state_summary produces correct dict")

    # Test full NL wrapper with mock env
    base_env = OSWorldGymWrapper.__new__(OSWorldGymWrapper)
    base_env._provider_name = "mock"
    base_env._path_to_vm = None
    base_env._os_type = "Ubuntu"
    base_env._action_space_type = "pyautogui"
    base_env._headless = True
    base_env._max_steps = 10
    base_env._auto_evaluate = True
    base_env._pause_after_action = 0.0
    base_env._task_shuffle = False
    base_env._env_kwargs = {}
    base_env._tasks = [{
        "id": "nl-test-001",
        "instruction": "Install Spotify",
        "config": [],
        "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}},
    }]
    base_env._task_index = 0
    base_env._task_cycle_count = 0
    base_env._step_count = 0
    base_env._episode_count = 0
    base_env._current_task = None
    base_env._last_obs = None
    base_env._terminated = False
    base_env._truncated = False
    base_env._env = _MockDesktopEnv()

    env = OSWorldNLWrapper(base_env)

    obs_nl, info = env.reset()
    assert isinstance(obs_nl, str), "NL obs should be string"
    assert "Task: Install Spotify" in obs_nl
    assert "structured_state" in info
    assert info["env_name"] == "osworld"
    print("  [PASS] NL wrapper reset() returns correct text + info")

    obs_nl, reward, term, trunc, info = env.step("pyautogui.click(960, 540)")
    assert isinstance(obs_nl, str)
    assert info["step"] == 1
    assert "structured_state" in info
    print("  [PASS] NL wrapper step() returns correct 5-tuple")

    env.close()
    print("  [PASS] NL wrapper close() works")


def test_task_catalog():
    """Test task catalog loading."""
    from env_wrappers.osworld_wrapper import load_task_catalog

    # Create a temp catalog file
    catalog = {
        "libreoffice_calc": [
            {"id": "calc-001", "instruction": "Sum column A", "config": [], "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}}},
            {"id": "calc-002", "instruction": "Format as currency", "config": [], "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}}},
        ],
        "firefox": [
            {"id": "ff-001", "instruction": "Open Google", "config": [], "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}}},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(catalog, f)
        f.flush()
        path = f.name

    tasks = load_task_catalog(path)
    assert len(tasks) == 3, f"expected 3 tasks, got {len(tasks)}"
    print("  [PASS] load_task_catalog loads all tasks")

    tasks = load_task_catalog(path, domain="libreoffice_calc")
    assert len(tasks) == 2, f"expected 2 calc tasks, got {len(tasks)}"
    print("  [PASS] load_task_catalog filters by domain")

    tasks = load_task_catalog(path, limit=1)
    assert len(tasks) == 1, f"expected 1 task with limit, got {len(tasks)}"
    print("  [PASS] load_task_catalog respects limit")

    Path(path).unlink()


def test_live(provider: str, catalog: str | None, max_steps: int):
    """Live test with an actual VM (requires OSWorld installed)."""
    from env_wrappers.osworld_wrapper import OSWorldGymWrapper
    from env_wrappers.osworld_nl_wrapper import OSWorldNLWrapper

    kwargs = dict(
        provider_name=provider,
        headless=True,
        max_steps=max_steps,
        auto_evaluate=True,
    )
    if catalog:
        kwargs["task_catalog"] = catalog

    base_env = OSWorldGymWrapper(**kwargs)
    env = OSWorldNLWrapper(base_env)

    print(f"\n  Environment: {base_env}")
    print(f"  Tasks loaded: {base_env.num_tasks}")

    obs, info = env.reset()
    print(f"\n  Task: {info.get('instruction', 'N/A')}")
    print(f"  Observation (first 300 chars):\n    {obs[:300]}...")

    # Take a single action to verify the loop works
    obs, reward, term, trunc, info = env.step("WAIT")
    print(f"\n  After WAIT: step={info['step']}, term={term}, trunc={trunc}")

    obs, reward, term, trunc, info = env.step("DONE")
    print(f"  After DONE: reward={reward}, term={term}")

    env.close()
    print("  [PASS] Live test completed successfully")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Test OSWorld Gymnasium wrapper")
    parser.add_argument("--live", action="store_true", help="Run live test with VM")
    parser.add_argument("--provider", default="docker", help="VM provider for live test")
    parser.add_argument("--catalog", default=None, help="Path to task catalog JSON")
    parser.add_argument("--max-steps", type=int, default=5, help="Max steps per episode")
    args = parser.parse_args()

    print("=" * 60)
    print("OSWorld Gymnasium Wrapper Tests")
    print("=" * 60)

    if args.live:
        print("\n[LIVE] Running with actual VM...")
        test_live(args.provider, args.catalog, args.max_steps)
    else:
        print("\n[MOCK] test_task_catalog")
        test_task_catalog()

        print("\n[MOCK] test_mock_wrapper")
        test_mock_wrapper()

        print("\n[MOCK] test_nl_wrapper")
        test_nl_wrapper()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
