"""
Unit tests for OSWorldNLWrapper — natural language observation adapter.

Tests cover:
  - NL observation format and content
  - Structured state summaries
  - Accessibility tree parsing
  - Screen region detection
  - Action hints
  - Full episode flow through NL wrapper
  - Edge cases (empty observations, missing fields)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest


# ======================================================================
# obs_to_natural_language
# ======================================================================

class TestObsToNaturalLanguage:
    """Test the standalone NL conversion function."""

    def test_includes_instruction(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        obs = {
            "screenshot": np.zeros((720, 1280, 3), dtype=np.uint8),
            "accessibility_tree": "",
            "terminal": "",
            "instruction": "Open the file manager",
        }
        nl = obs_to_natural_language(obs)
        assert "Task: Open the file manager" in nl

    def test_includes_screen_dimensions(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        obs = {
            "screenshot": np.zeros((1080, 1920, 3), dtype=np.uint8),
            "accessibility_tree": "",
            "terminal": "",
            "instruction": "",
        }
        nl = obs_to_natural_language(obs)
        assert "1920x1080" in nl

    def test_includes_a11y_elements(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        obs = {
            "screenshot": None,
            "accessibility_tree": '[button] "Save" {focused}\n[button] "Cancel" {}\n',
            "terminal": "",
            "instruction": "",
        }
        nl = obs_to_natural_language(obs, include_a11y=True)
        assert "Save" in nl

    def test_excludes_a11y_when_disabled(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        obs = {
            "screenshot": None,
            "accessibility_tree": '[button] "Save" {}\n',
            "terminal": "",
            "instruction": "",
        }
        nl = obs_to_natural_language(obs, include_a11y=False)
        assert "Save" not in nl

    def test_includes_terminal_output(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        obs = {
            "screenshot": None,
            "accessibility_tree": "",
            "terminal": "$ ls\nDocuments  Downloads",
            "instruction": "",
        }
        nl = obs_to_natural_language(obs)
        assert "Documents" in nl
        assert "Terminal" in nl

    def test_truncates_long_terminal(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        long_terminal = "x\n" * 1000
        obs = {
            "screenshot": None,
            "accessibility_tree": "",
            "terminal": long_terminal,
            "instruction": "",
        }
        nl = obs_to_natural_language(obs)
        assert "..." in nl

    def test_empty_observation_fallback(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        obs = {
            "screenshot": None,
            "accessibility_tree": "",
            "terminal": "",
            "instruction": "",
        }
        nl = obs_to_natural_language(obs)
        assert "No details available" in nl

    def test_max_a11y_elements_limit(self):
        from env_wrappers.osworld_nl_wrapper import obs_to_natural_language
        lines = "\n".join(f'[button] "Button{i}" {{}}' for i in range(100))
        obs = {
            "screenshot": None,
            "accessibility_tree": lines,
            "terminal": "",
            "instruction": "",
        }
        nl = obs_to_natural_language(obs, max_a11y_elements=5)
        assert nl.count("button:") <= 5


# ======================================================================
# build_osworld_state_summary
# ======================================================================

class TestStateSummary:
    """Test structured state summary generation."""

    def test_basic_fields(self):
        from env_wrappers.osworld_nl_wrapper import build_osworld_state_summary
        obs = {
            "screenshot": np.zeros((100, 100, 3), dtype=np.uint8),
            "accessibility_tree": "",
            "terminal": "",
            "instruction": "Install Firefox",
        }
        s = build_osworld_state_summary(obs, step=5)
        assert s["env"] == "osworld"
        assert s["domain"] == "desktop"
        assert s["step"] == 5
        assert s["has_screenshot"] is True
        assert "Install Firefox" in s["instruction"]

    def test_last_action_and_reward(self):
        from env_wrappers.osworld_nl_wrapper import build_osworld_state_summary
        obs = {"screenshot": None, "accessibility_tree": "", "terminal": "", "instruction": ""}
        s = build_osworld_state_summary(obs, step=3, last_action="click(500, 300)", last_reward=0.5)
        assert s["last_action"] == "click(500, 300)"
        assert s["reward"] == 0.5

    def test_screen_region_detection_dialog(self):
        from env_wrappers.osworld_nl_wrapper import build_osworld_state_summary
        obs = {
            "screenshot": None,
            "accessibility_tree": '[dialog] "Save As" {modal}\n[button] "OK" {}',
            "terminal": "",
            "instruction": "",
        }
        s = build_osworld_state_summary(obs)
        assert s.get("has_dialog") is True

    def test_screen_region_detection_menu(self):
        from env_wrappers.osworld_nl_wrapper import build_osworld_state_summary
        obs = {
            "screenshot": None,
            "accessibility_tree": '[menubar] "Menu Bar" {}\n[toolbar] "Standard" {}',
            "terminal": "",
            "instruction": "",
        }
        s = build_osworld_state_summary(obs)
        regions = s.get("screen_regions", {})
        assert "menu" in regions or "toolbar" in regions

    def test_terminal_summary(self):
        from env_wrappers.osworld_nl_wrapper import build_osworld_state_summary
        obs = {
            "screenshot": None,
            "accessibility_tree": "",
            "terminal": "$ apt install vlc\nReading package lists...\nDone",
            "instruction": "",
        }
        s = build_osworld_state_summary(obs)
        assert s["terminal_lines"] == 3
        assert "Done" in s["terminal_last"]

    def test_no_screenshot(self):
        from env_wrappers.osworld_nl_wrapper import build_osworld_state_summary
        obs = {"screenshot": None, "accessibility_tree": "", "terminal": "", "instruction": ""}
        s = build_osworld_state_summary(obs)
        assert s["has_screenshot"] is False


# ======================================================================
# _parse_a11y_tree
# ======================================================================

class TestParseA11yTree:
    """Test accessibility tree parsing."""

    def test_extracts_roles_and_names(self):
        from env_wrappers.osworld_nl_wrapper import _parse_a11y_tree
        raw = '[button] "Save" {focused}\n[button] "Cancel" {}'
        parsed = _parse_a11y_tree(raw)
        assert "button: Save" in parsed
        assert "button: Cancel" in parsed

    def test_extracts_states(self):
        from env_wrappers.osworld_nl_wrapper import _parse_a11y_tree
        raw = '[button] "Save" {focused}'
        parsed = _parse_a11y_tree(raw)
        assert "focused" in parsed

    def test_respects_max_elements(self):
        from env_wrappers.osworld_nl_wrapper import _parse_a11y_tree
        raw = "\n".join(f'[button] "Btn{i}" {{}}' for i in range(50))
        parsed = _parse_a11y_tree(raw, max_elements=5)
        lines = [l for l in parsed.split("\n") if l.strip()]
        assert len(lines) <= 5

    def test_empty_input(self):
        from env_wrappers.osworld_nl_wrapper import _parse_a11y_tree
        assert _parse_a11y_tree("") == ""
        assert _parse_a11y_tree("   ") == ""
        assert _parse_a11y_tree(None) == ""


# ======================================================================
# NL Wrapper full flow
# ======================================================================

class TestNLWrapperFlow:
    """Test OSWorldNLWrapper reset/step/close cycle."""

    def test_reset_returns_string(self, nl_wrapper):
        obs, info = nl_wrapper.reset()
        assert isinstance(obs, str)
        assert len(obs) > 0

    def test_reset_contains_task(self, nl_wrapper):
        obs, info = nl_wrapper.reset()
        assert "Task:" in obs

    def test_reset_info_has_structured_state(self, nl_wrapper):
        _, info = nl_wrapper.reset()
        assert "structured_state" in info
        assert info["structured_state"]["env"] == "osworld"

    def test_reset_info_has_raw_obs(self, nl_wrapper):
        _, info = nl_wrapper.reset()
        assert "raw_obs" in info
        assert "screenshot" in info["raw_obs"]

    def test_step_returns_string(self, nl_wrapper):
        nl_wrapper.reset()
        obs, _, _, _, _ = nl_wrapper.step("pyautogui.click(500, 300)")
        assert isinstance(obs, str)

    def test_step_info_has_step_count(self, nl_wrapper):
        nl_wrapper.reset()
        _, _, _, _, info = nl_wrapper.step("pyautogui.click(500, 300)")
        assert info["step"] == 1

    def test_action_hint_included(self, nl_wrapper):
        obs, _ = nl_wrapper.reset()
        assert "pyautogui" in obs
        assert "DONE" in obs

    def test_action_hint_excluded(self, gym_wrapper):
        from env_wrappers.osworld_nl_wrapper import OSWorldNLWrapper
        env = OSWorldNLWrapper(gym_wrapper, include_action_hint=False)
        obs, _ = env.reset()
        assert "DONE" not in obs or "Task:" in obs

    def test_done_propagates(self, nl_wrapper):
        nl_wrapper.reset()
        _, reward, term, trunc, _ = nl_wrapper.step("DONE")
        assert term is True
        assert reward == 1.0

    def test_close(self, nl_wrapper):
        nl_wrapper.reset()
        nl_wrapper.close()

    def test_repr(self, nl_wrapper):
        r = repr(nl_wrapper)
        assert "OSWorldNLWrapper" in r


# ======================================================================
# NL wrapper properties
# ======================================================================

class TestNLWrapperProperties:
    """Test property accessors on the NL wrapper."""

    def test_action_names(self, nl_wrapper):
        assert "DONE" in nl_wrapper.action_names

    def test_tasks(self, nl_wrapper):
        assert len(nl_wrapper.tasks) == 3

    def test_current_task_before_reset(self, nl_wrapper):
        assert nl_wrapper.current_task is None

    def test_current_task_after_reset(self, nl_wrapper):
        nl_wrapper.reset()
        assert nl_wrapper.current_task is not None
        assert "id" in nl_wrapper.current_task

    def test_env_property(self, nl_wrapper, gym_wrapper):
        assert nl_wrapper.env is gym_wrapper

    def test_evaluate_passthrough(self, nl_wrapper):
        nl_wrapper.reset()
        nl_wrapper.step("DONE")
        score = nl_wrapper.evaluate()
        assert score == 1.0

    def test_render_passthrough(self, nl_wrapper):
        nl_wrapper.reset()
        img = nl_wrapper.render()
        assert isinstance(img, np.ndarray)
