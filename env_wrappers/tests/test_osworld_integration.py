"""
Integration tests — verify OSWorldGymWrapper can lazy-init DesktopEnv.

These tests do NOT start a VM. They verify:
  - Lazy import works (desktop_env is only loaded on first reset)
  - Constructor accepts all documented parameters
  - Task catalog can be passed at construction time
  - Docker provider can be selected (import path, not actual container)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ======================================================================
# Lazy import
# ======================================================================

class TestLazyImport:
    """Verify DesktopEnv is not imported at wrapper construction time."""

    def test_constructor_does_not_import_desktop_env(self):
        """Creating the wrapper should NOT trigger DesktopEnv import."""
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper

        env = OSWorldGymWrapper(
            provider_name="docker",
            headless=True,
            max_steps=5,
        )
        assert env._env is None
        env.close()

    @patch("env_wrappers.osworld_wrapper.OSWorldGymWrapper._ensure_env")
    def test_reset_triggers_ensure_env(self, mock_ensure):
        """reset() should call _ensure_env to lazy-init."""
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper

        env = OSWorldGymWrapper(provider_name="docker", headless=True)
        mock_ensure.side_effect = RuntimeError("mock: would start VM")
        with pytest.raises(RuntimeError, match="mock: would start VM"):
            env.reset()
        mock_ensure.assert_called_once()


# ======================================================================
# Constructor parameter validation
# ======================================================================

class TestConstructorParams:
    """Verify all documented constructor parameters are accepted."""

    def test_default_params(self):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper()
        assert env._provider_name == "vmware"
        assert env._max_steps == 15
        assert env._auto_evaluate is True
        env.close()

    def test_docker_provider(self):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper(provider_name="docker", os_type="Ubuntu")
        assert env._provider_name == "docker"
        env.close()

    def test_aws_provider(self):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper(provider_name="aws", os_type="Ubuntu")
        assert env._provider_name == "aws"
        env.close()

    def test_custom_screen_size(self):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper(screen_size=(1280, 720))
        assert env._env_kwargs["screen_size"] == (1280, 720)
        env.close()

    def test_computer_13_action_space(self):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper(action_space_type="computer_13")
        assert env._action_space_type == "computer_13"
        env.close()

    def test_catalog_from_file(self, task_catalog_file):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper(task_catalog=task_catalog_file)
        assert env.num_tasks == 3
        env.close()

    def test_catalog_from_list(self, sample_tasks):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper(task_catalog=sample_tasks)
        assert env.num_tasks == 3
        env.close()

    def test_no_catalog_uses_default(self):
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        env = OSWorldGymWrapper()
        assert env.num_tasks == 1
        assert env.tasks[0]["id"] == "94d95f96-9699-4208-98ba-3c3119edf9c2"
        env.close()


# ======================================================================
# Full wrapper → mock DesktopEnv integration
# ======================================================================

class TestMockDesktopEnvIntegration:
    """Test the full wrapper with a patched DesktopEnv constructor."""

    def _make_mock_env(self):
        """Create a mock that mimics DesktopEnv's interface."""
        mock = MagicMock()
        screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock.reset.return_value = {
            "screenshot": screen,
            "accessibility_tree": '[window] "Desktop" {}',
            "terminal": "",
            "instruction": "Test task",
        }
        mock.step.return_value = (
            {
                "screenshot": screen,
                "accessibility_tree": '[window] "Desktop" {}',
                "terminal": "$ echo hello\nhello",
                "instruction": "Test task",
            },
            0.0,
            False,
            {},
        )
        mock.evaluate.return_value = 1.0
        mock.render.return_value = screen
        return mock

    @patch("desktop_env.desktop_env.DesktopEnv")
    def test_full_episode_with_patched_init(self, MockDE):
        """Patch DesktopEnv constructor to avoid VM start."""
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper

        mock_env = self._make_mock_env()
        MockDE.return_value = mock_env

        env = OSWorldGymWrapper(provider_name="docker", headless=True, max_steps=5)
        obs, info = env.reset()

        assert "screenshot" in obs
        assert info["task_id"] == "94d95f96-9699-4208-98ba-3c3119edf9c2"
        MockDE.assert_called_once()
        mock_env.reset.assert_called_once()

        obs, reward, term, trunc, info = env.step("pyautogui.click(960, 540)")
        assert not term
        mock_env.step.assert_called_once()

        # Terminal action
        mock_env.step.return_value = (
            mock_env.step.return_value[0],
            0.0,
            True,
            {"done": True},
        )
        obs, reward, term, trunc, info = env.step("DONE")
        assert term is True
        assert reward == 1.0

        env.close()
        mock_env.close.assert_called_once()

    @patch("desktop_env.desktop_env.DesktopEnv")
    def test_nl_wrapper_with_patched_env(self, MockDE):
        """Full NL wrapper flow with patched DesktopEnv."""
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper
        from env_wrappers.osworld_nl_wrapper import OSWorldNLWrapper

        mock_env = self._make_mock_env()
        MockDE.return_value = mock_env

        base = OSWorldGymWrapper(provider_name="docker", headless=True)
        env = OSWorldNLWrapper(base)

        obs, info = env.reset()
        assert isinstance(obs, str)
        assert "Task:" in obs
        assert info["env_name"] == "osworld"

        obs, reward, term, trunc, info = env.step("pyautogui.click(500, 300)")
        assert isinstance(obs, str)
        assert info["step"] == 1

        env.close()
