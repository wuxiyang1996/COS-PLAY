"""
Unit tests for OSWorldGymWrapper — the core Gymnasium adapter.

Tests cover:
  - Observation structure and types
  - 5-tuple step returns
  - Terminated vs truncated semantics
  - Special actions (DONE, FAIL, WAIT)
  - Auto-evaluation on terminal actions
  - Task catalog loading (dict, list, domain filter, limit)
  - Task cycling and selection
  - Edge cases (step after done, empty catalog, etc.)
  - Real DesktopEnv import path (without starting a VM)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ======================================================================
# Observation structure
# ======================================================================

class TestObservation:
    """Verify observation dict structure from reset() and step()."""

    def test_reset_obs_keys(self, gym_wrapper):
        obs, info = gym_wrapper.reset()
        assert set(obs.keys()) == {"screenshot", "accessibility_tree", "terminal", "instruction"}

    def test_reset_screenshot_is_ndarray(self, gym_wrapper):
        obs, _ = gym_wrapper.reset()
        assert isinstance(obs["screenshot"], np.ndarray)

    def test_reset_screenshot_shape(self, gym_wrapper):
        obs, _ = gym_wrapper.reset()
        assert obs["screenshot"].shape == (1080, 1920, 3)
        assert obs["screenshot"].dtype == np.uint8

    def test_reset_instruction_matches_task(self, gym_wrapper):
        obs, info = gym_wrapper.reset()
        assert obs["instruction"] == info["instruction"]

    def test_reset_a11y_tree_is_string(self, gym_wrapper):
        obs, _ = gym_wrapper.reset()
        assert isinstance(obs["accessibility_tree"], str)
        assert len(obs["accessibility_tree"]) > 0

    def test_step_obs_has_same_keys(self, gym_wrapper):
        gym_wrapper.reset()
        obs, _, _, _, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert set(obs.keys()) == {"screenshot", "accessibility_tree", "terminal", "instruction"}

    def test_step_terminal_updates(self, gym_wrapper):
        obs0, _ = gym_wrapper.reset()
        assert obs0["terminal"] == ""
        obs1, _, _, _, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert len(obs1["terminal"]) > 0


# ======================================================================
# Info dict
# ======================================================================

class TestInfo:
    """Verify info dict contents from reset() and step()."""

    def test_reset_info_fields(self, gym_wrapper):
        _, info = gym_wrapper.reset()
        for key in ("task_id", "instruction", "episode", "max_steps", "provider"):
            assert key in info, f"missing info key: {key}"

    def test_reset_info_task_id(self, gym_wrapper):
        _, info = gym_wrapper.reset()
        assert info["task_id"] == "task-001"

    def test_step_info_has_step_count(self, gym_wrapper):
        gym_wrapper.reset()
        _, _, _, _, info = gym_wrapper.step("pyautogui.click(100, 100)")
        assert info["step"] == 1

    def test_step_info_records_action(self, gym_wrapper):
        gym_wrapper.reset()
        _, _, _, _, info = gym_wrapper.step("pyautogui.hotkey('ctrl', 's')")
        assert info["action"] == "pyautogui.hotkey('ctrl', 's')"


# ======================================================================
# Step return semantics (5-tuple)
# ======================================================================

class TestStepReturns:
    """Verify Gymnasium 5-tuple (obs, reward, terminated, truncated, info)."""

    def test_step_returns_5_values(self, gym_wrapper):
        gym_wrapper.reset()
        result = gym_wrapper.step("pyautogui.click(100, 100)")
        assert len(result) == 5

    def test_normal_step_not_done(self, gym_wrapper):
        gym_wrapper.reset()
        _, reward, term, trunc, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert not term
        assert not trunc
        assert reward == 0.0

    def test_reward_is_float(self, gym_wrapper):
        gym_wrapper.reset()
        _, reward, _, _, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert isinstance(reward, float)

    def test_terminated_is_bool(self, gym_wrapper):
        gym_wrapper.reset()
        _, _, term, _, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert isinstance(term, bool)

    def test_truncated_is_bool(self, gym_wrapper):
        gym_wrapper.reset()
        _, _, _, trunc, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert isinstance(trunc, bool)


# ======================================================================
# Terminal actions (DONE, FAIL, WAIT)
# ======================================================================

class TestTerminalActions:
    """Test special action handling."""

    def test_done_terminates(self, gym_wrapper):
        gym_wrapper.reset()
        _, reward, term, trunc, info = gym_wrapper.step("DONE")
        assert term is True
        assert trunc is False

    def test_done_evaluates_reward(self, gym_wrapper):
        gym_wrapper.reset()
        _, reward, _, _, info = gym_wrapper.step("DONE")
        assert reward == 1.0
        assert info["eval_score"] == 1.0

    def test_fail_terminates(self, gym_wrapper):
        gym_wrapper.reset()
        _, reward, term, _, info = gym_wrapper.step("FAIL")
        assert term is True

    def test_fail_evaluates_zero(self, gym_wrapper):
        gym_wrapper.reset()
        _, reward, _, _, info = gym_wrapper.step("FAIL")
        assert reward == 0.0
        assert info["eval_score"] == 0.0

    def test_wait_does_not_terminate(self, gym_wrapper):
        gym_wrapper.reset()
        _, _, term, trunc, _ = gym_wrapper.step("WAIT")
        assert not term
        assert not trunc


# ======================================================================
# Truncation
# ======================================================================

class TestTruncation:
    """Test max-step truncation logic."""

    def test_truncation_at_max_steps(self, gym_wrapper):
        gym_wrapper._max_steps = 3
        gym_wrapper.reset()
        for _ in range(2):
            _, _, term, trunc, _ = gym_wrapper.step("pyautogui.click(100, 100)")
            assert not term and not trunc
        _, _, term, trunc, _ = gym_wrapper.step("pyautogui.click(100, 100)")
        assert trunc is True
        assert term is False

    def test_done_before_max_steps_is_terminated_not_truncated(self, gym_wrapper):
        gym_wrapper._max_steps = 100
        gym_wrapper.reset()
        _, _, term, trunc, _ = gym_wrapper.step("DONE")
        assert term is True
        assert trunc is False


# ======================================================================
# Step-after-done guard
# ======================================================================

class TestStepAfterDone:
    """Verify that stepping after episode end raises an error."""

    def test_step_after_terminated_raises(self, gym_wrapper):
        gym_wrapper.reset()
        gym_wrapper.step("DONE")
        with pytest.raises(RuntimeError, match="Episode has ended"):
            gym_wrapper.step("pyautogui.click(100, 100)")

    def test_step_after_truncated_raises(self, gym_wrapper):
        gym_wrapper._max_steps = 1
        gym_wrapper.reset()
        gym_wrapper.step("pyautogui.click(100, 100)")
        with pytest.raises(RuntimeError, match="Episode has ended"):
            gym_wrapper.step("pyautogui.click(100, 100)")

    def test_reset_after_done_works(self, gym_wrapper):
        gym_wrapper.reset()
        gym_wrapper.step("DONE")
        obs, info = gym_wrapper.reset()
        assert obs["screenshot"] is not None


# ======================================================================
# Auto-evaluation toggle
# ======================================================================

class TestAutoEvaluation:
    """Test auto_evaluate flag."""

    def test_auto_evaluate_on(self, gym_wrapper):
        gym_wrapper._auto_evaluate = True
        gym_wrapper.reset()
        _, reward, _, _, info = gym_wrapper.step("DONE")
        assert info["eval_score"] is not None

    def test_auto_evaluate_off(self, gym_wrapper):
        gym_wrapper._auto_evaluate = False
        gym_wrapper.reset()
        _, reward, _, _, info = gym_wrapper.step("DONE")
        assert info["eval_score"] is None
        assert reward == 0.0

    def test_manual_evaluate(self, gym_wrapper):
        gym_wrapper._auto_evaluate = False
        gym_wrapper.reset()
        gym_wrapper.step("DONE")
        score = gym_wrapper.evaluate()
        assert score == 1.0


# ======================================================================
# Task catalog loading
# ======================================================================

class TestTaskCatalog:
    """Test load_task_catalog with different formats."""

    def test_load_dict_format(self, task_catalog_file):
        from env_wrappers.osworld_wrapper import load_task_catalog
        tasks = load_task_catalog(task_catalog_file)
        assert len(tasks) == 3

    def test_load_list_format(self, task_catalog_list_file):
        from env_wrappers.osworld_wrapper import load_task_catalog
        tasks = load_task_catalog(task_catalog_list_file)
        assert len(tasks) == 3

    def test_filter_by_domain(self, task_catalog_file):
        from env_wrappers.osworld_wrapper import load_task_catalog
        tasks = load_task_catalog(task_catalog_file, domain="firefox")
        assert len(tasks) == 1
        assert tasks[0]["id"] == "task-001"

    def test_limit(self, task_catalog_file):
        from env_wrappers.osworld_wrapper import load_task_catalog
        tasks = load_task_catalog(task_catalog_file, limit=2)
        assert len(tasks) == 2

    def test_missing_file_raises(self, tmp_path):
        from env_wrappers.osworld_wrapper import load_task_catalog
        with pytest.raises(FileNotFoundError):
            load_task_catalog(tmp_path / "nonexistent.json")

    def test_empty_catalog(self, tmp_path):
        from env_wrappers.osworld_wrapper import load_task_catalog
        path = tmp_path / "empty.json"
        path.write_text("[]")
        tasks = load_task_catalog(path)
        assert tasks == []


# ======================================================================
# Task cycling and selection
# ======================================================================

class TestTaskCycling:
    """Test task rotation and selection via options."""

    def test_sequential_cycling(self, gym_wrapper):
        """Tasks should cycle through in order."""
        ids = []
        for _ in range(len(gym_wrapper.tasks)):
            _, info = gym_wrapper.reset()
            ids.append(info["task_id"])
            gym_wrapper.step("DONE")
        assert ids == ["task-001", "task-002", "task-003"]

    def test_wrap_around(self, gym_wrapper):
        """After exhausting catalog, should wrap to the beginning."""
        for _ in range(3):
            gym_wrapper.reset()
            gym_wrapper.step("DONE")
        _, info = gym_wrapper.reset()
        assert info["task_id"] == "task-001"

    def test_select_by_task_id(self, gym_wrapper):
        _, info = gym_wrapper.reset(options={"task_id": "task-003"})
        assert info["task_id"] == "task-003"

    def test_select_by_index(self, gym_wrapper):
        _, info = gym_wrapper.reset(options={"task_index": 1})
        assert info["task_id"] == "task-002"

    def test_select_by_task_config(self, gym_wrapper):
        custom = {"id": "custom-999", "instruction": "Do something custom", "config": [], "evaluator": {"func": "check_include_exclude", "result": {"type": "vm_command_line", "command": "echo ok"}, "expected": {"type": "rule", "rules": {"include": ["ok"]}}}}
        _, info = gym_wrapper.reset(options={"task_config": custom})
        assert info["task_id"] == "custom-999"

    def test_invalid_task_id_raises(self, gym_wrapper):
        with pytest.raises(ValueError, match="not found"):
            gym_wrapper.reset(options={"task_id": "nonexistent"})

    def test_shuffle_on_wrap(self, sample_tasks, mock_desktop_env):
        """With task_shuffle=True, order should change after wrap-around."""
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
        env._task_shuffle = True
        env._env_kwargs = {}
        env._tasks = sample_tasks * 3  # 9 tasks so shuffling is observable
        env._task_index = 0
        env._task_cycle_count = 0
        env._step_count = 0
        env._episode_count = 0
        env._current_task = None
        env._last_obs = None
        env._terminated = False
        env._truncated = False
        env._env = mock_desktop_env

        # Complete first cycle
        first_order = []
        for _ in range(9):
            _, info = env.reset()
            first_order.append(info["task_id"])
            env.step("DONE")

        # Complete second cycle
        second_order = []
        for _ in range(9):
            _, info = env.reset()
            second_order.append(info["task_id"])
            env.step("DONE")

        # Orders should differ (with very high probability for 9 items)
        assert first_order != second_order, "shuffle should change order"
        assert sorted(first_order) == sorted(second_order), "same tasks, different order"


# ======================================================================
# Utility methods
# ======================================================================

class TestUtilityMethods:
    """Test helper/utility methods on the wrapper."""

    def test_task_ids(self, gym_wrapper):
        ids = gym_wrapper.task_ids()
        assert ids == ["task-001", "task-002", "task-003"]

    def test_task_instructions(self, gym_wrapper):
        mapping = gym_wrapper.task_instructions()
        assert "task-001" in mapping
        assert "Firefox" in mapping["task-001"]

    def test_num_tasks(self, gym_wrapper):
        assert gym_wrapper.num_tasks == 3

    def test_action_names(self, gym_wrapper):
        assert set(gym_wrapper.action_names) == {"DONE", "FAIL", "WAIT"}

    def test_repr(self, gym_wrapper):
        r = repr(gym_wrapper)
        assert "OSWorldGymWrapper" in r
        assert "mock" in r

    def test_render(self, gym_wrapper):
        gym_wrapper.reset()
        img = gym_wrapper.render()
        assert isinstance(img, np.ndarray)
        assert img.shape == (1080, 1920, 3)

    def test_close_is_idempotent(self, gym_wrapper):
        gym_wrapper.reset()
        gym_wrapper.close()
        gym_wrapper.close()  # should not raise


# ======================================================================
# DesktopEnv import path (no VM needed)
# ======================================================================

class TestDesktopEnvImport:
    """Verify we can import DesktopEnv and inspect its interface."""

    def test_desktop_env_importable(self):
        from desktop_env.desktop_env import DesktopEnv
        assert hasattr(DesktopEnv, "reset")
        assert hasattr(DesktopEnv, "step")
        assert hasattr(DesktopEnv, "close")
        assert hasattr(DesktopEnv, "evaluate")

    def test_desktop_env_is_gym_env(self):
        from desktop_env.desktop_env import DesktopEnv
        import gymnasium as gym
        assert issubclass(DesktopEnv, gym.Env)

    def test_desktop_env_step_signature(self):
        """Verify step() accepts (action, pause) — the interface we wrap."""
        import inspect
        from desktop_env.desktop_env import DesktopEnv
        sig = inspect.signature(DesktopEnv.step)
        params = list(sig.parameters.keys())
        assert "action" in params
        assert "pause" in params

    def test_desktop_env_reset_signature(self):
        """Verify reset() accepts task_config — the interface we wrap."""
        import inspect
        from desktop_env.desktop_env import DesktopEnv
        sig = inspect.signature(DesktopEnv.reset)
        params = list(sig.parameters.keys())
        assert "task_config" in params


# ======================================================================
# Multi-episode stress test
# ======================================================================

class TestMultiEpisode:
    """Run multiple episodes to verify state doesn't leak."""

    def test_ten_episodes(self, gym_wrapper):
        for ep in range(10):
            obs, info = gym_wrapper.reset()
            assert info["episode"] == ep + 1
            for step_i in range(3):
                obs, reward, term, trunc, info = gym_wrapper.step(
                    "pyautogui.click(100, 100)"
                )
                assert info["step"] == step_i + 1
                if term or trunc:
                    break
            gym_wrapper.step("DONE")

    def test_alternating_done_fail(self, gym_wrapper):
        for i in range(6):
            gym_wrapper.reset()
            action = "DONE" if i % 2 == 0 else "FAIL"
            _, reward, term, _, _ = gym_wrapper.step(action)
            assert term is True
            expected = 1.0 if action == "DONE" else 0.0
            assert reward == expected
