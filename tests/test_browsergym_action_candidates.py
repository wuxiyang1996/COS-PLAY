"""Regression tests for ``browsergym_wrapper.tools._h_list_valid_actions``.

These tests lock in the May-3 2026 fix that surfaces ``fill(<bid>, "...")``
candidate actions for visible textbox / searchbox / combobox elements
**regardless** of whether BrowserGym populated the
``set_of_marks=True`` flag on those elements. The pre-fix behaviour
silently dropped every input candidate on classifieds / shopping
listings pages where ``set_of_marks`` was sparsely populated, which in
turn caused the actor to thrash through navigation actions
(``scroll/go_back/go_forward``) on multi-constraint search tasks like
``visualwebarena.92``.

See:
* ``browsergym_wrapper/tools.py:_h_list_valid_actions`` — patched code.
* ``legacy/visualwebarena/vwa-improvement-plan.md`` §3 Tier-1 change B
    — full root-cause analysis and the diagnostic episode this test
    guards against regressing.
"""
from __future__ import annotations

import sys

import pytest


# Allow the test to run from the repo root without an editable install.
sys.path.insert(0, "/workspace/Multi-hop-Reasoning-VLM-Agent")

from browsergym_wrapper.tools import (  # type: ignore  # noqa: E402
    _BrowserState,
    _h_list_valid_actions,
)


def _node(bid: str, role: str, name: str = "") -> dict:
    """Mirror the AXTree node shape produced by BrowserGym."""
    return {
        "browsergym_id": bid,
        "role": {"value": role},
        "name": {"value": name},
    }


def _make_obs(nodes: list[dict], extras: dict[str, dict]) -> dict:
    return {
        "axtree_object": {"nodes": nodes},
        "extra_element_properties": extras,
        "focused_element_bid": "",
        "url": "http://localhost:9980/",
        "last_action": "",
        "last_action_error": "",
    }


def test_fill_surfaced_for_textbox_without_set_of_marks_flag():
    """The smoking-gun test for the May-3 fix.

    Build a synthetic search page where the only ``<input>`` element
    has visibility=1 but neither ``set_of_marks`` nor ``clickable`` set
    (this is the exact state we observed on classifieds.92).  The
    pre-fix candidate list would emit zero ``fill(...)``.  The patched
    list MUST emit ``fill("a3", "...")``.
    """
    nodes = [
        _node("a1", "main", "page-root"),
        _node("a3", "searchbox", "site search"),
        _node("a7", "button", "Submit"),
    ]
    extras = {
        # Search box — visible, but BrowserGym did not flag it as
        # set-of-marks or clickable. Pre-fix this dropped it.
        "a3": {"visibility": 1.0},
        # Submit button — clickable, has a SoM tag.
        "a7": {"visibility": 1.0, "set_of_marks": True, "clickable": True},
    }
    state = _BrowserState(_make_obs(nodes, extras))

    out = _h_list_valid_actions(state)
    actions = [a["action"] for a in out["actions"]]

    assert any(a.startswith('fill("a3"') for a in actions), (
        f"Expected ``fill(\"a3\", ...)`` in candidates; got {actions!r}"
    )
    assert out.get("fill_candidates", 0) >= 1, (
        f"fill_candidates counter should be >=1; got {out.get('fill_candidates')}"
    )


def test_click_still_requires_som_or_clickable_flag():
    """Click candidates still need a flag to avoid dumping the entire DOM.

    The patched filter only relaxes the gate for **input** roles. A
    plain ``main`` / ``generic`` / ``listitem`` node without either
    flag is still a no-op for the candidate list.
    """
    nodes = [
        _node("a2", "generic", "ambient div"),
        _node("a5", "link", "go to detail"),
    ]
    extras = {
        # Ambient div — visible but no SoM/clickable.  Should NOT be
        # surfaced as a click candidate.
        "a2": {"visibility": 1.0},
        # A real anchor — has clickable=True. Should be surfaced.
        "a5": {"visibility": 1.0, "clickable": True},
    }
    state = _BrowserState(_make_obs(nodes, extras))
    out = _h_list_valid_actions(state)
    actions = [a["action"] for a in out["actions"]]

    assert not any(a.startswith('click("a2"') for a in actions), (
        f"Bare generic should not produce click candidate; got {actions!r}"
    )
    assert any(a.startswith('click("a5"') for a in actions), (
        f"Clickable link should produce click candidate; got {actions!r}"
    )


def test_invisible_input_is_skipped():
    """Visibility < 0.5 still suppresses the candidate even for inputs.

    Hidden inputs (off-screen filter chips, collapsed search drawers,
    aria-hidden modals) must not pollute the candidate list — the
    visibility floor still guards against that.
    """
    nodes = [
        _node("a1", "textbox", "hidden filter"),
        _node("a2", "textbox", "visible search"),
    ]
    extras = {
        "a1": {"visibility": 0.0},
        "a2": {"visibility": 1.0},
    }
    state = _BrowserState(_make_obs(nodes, extras))
    actions = [a["action"] for a in _h_list_valid_actions(state)["actions"]]

    assert not any('fill("a1"' in a for a in actions), (
        f"Invisible textbox a1 should be filtered; got {actions!r}"
    )
    assert any('fill("a2"' in a for a in actions), (
        f"Visible textbox a2 should be surfaced; got {actions!r}"
    )


def test_navigation_fallback_still_present():
    """Even on a totally bare obs, scroll / go_back must be available.

    The actor relies on these as escape hatches when no real candidate
    is surfaced; removing them would leave the action selector with a
    legitimately empty list and a hard parse failure.
    """
    state = _BrowserState(_make_obs([], {}))
    actions = [a["action"] for a in _h_list_valid_actions(state)["actions"]]
    assert "scroll(down)" in actions
    assert "go_back()" in actions


def test_telemetry_counters_present_in_response():
    """The patched response must expose fill/click/toggle counters.

    The 4-model 200-task aggregator (``scripts/aggregate_vwa_baseline.py``)
    consumes these to report the per-task action-distribution audit
    matrix from §6 of the parent baseline plan.
    """
    nodes = [
        _node("a1", "searchbox"),
        _node("a2", "checkbox"),
        _node("a3", "link"),
    ]
    extras = {
        "a1": {"visibility": 1.0},
        "a2": {"visibility": 1.0, "clickable": True},
        "a3": {"visibility": 1.0, "set_of_marks": True},
    }
    state = _BrowserState(_make_obs(nodes, extras))
    out = _h_list_valid_actions(state)
    assert out["fill_candidates"] == 1
    assert out["toggle_candidates"] == 1
    assert out["click_candidates"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
