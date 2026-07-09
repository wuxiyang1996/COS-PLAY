"""Regression tests for the anti-thrash override (#6d) in the BrowserGym actor.

These tests lock in the May-3 2026 anti-thrash safeguard that catches the
post-search blocked-page recovery pattern surfaced in ``visualwebarena.92``,
where gpt-5.5 low looped through 28 navigation-only actions
(``scroll/go_back/go_forward``) without re-attempting the search box once
the bid had been re-numbered after the Magento interstitial.

The override is split into three pure helpers in
``cold_start/generate_cold_start_actor_browsergym.py``:

* ``_is_nav_only_action`` — classifies an action string into the
  navigation-only set ``{scroll, go_back, go_forward, noop}``.
* ``_extract_search_query`` — heuristic keyword extraction from the goal.
* ``_build_anti_thrash_action`` — synthesises ``fill(<bid>, <query>)``
  given the current candidate list and goal.

Together they implement the override that fires when
``consecutive_nav_actions >= _MAX_CONSECUTIVE_NAV`` and the next action
would itself be nav-only.

See:
* ``cold_start/generate_cold_start_actor_browsergym.py`` — patched code.
* ``legacy/visualwebarena/vwa-improvement-plan.md`` §3 Tier-1 change A
    — full root-cause analysis.
"""
from __future__ import annotations

import sys

import pytest


sys.path.insert(0, "/workspace/Multi-hop-Reasoning-VLM-Agent")


from cold_start.generate_cold_start_actor_browsergym import (  # type: ignore  # noqa: E402
    _MAX_CONSECUTIVE_NAV,
    _build_anti_thrash_action,
    _extract_search_query,
    _is_nav_only_action,
)


# ---------------------------------------------------------------------------
# _is_nav_only_action
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,expected",
    [
        ('scroll(0, 300)', True),
        ('scroll(0, -200)', True),
        ('go_back()', True),
        ('go_forward()', True),
        ('noop()', True),
        ('  scroll(0, 100)  ', True),  # leading whitespace tolerated
        ('click("12")', False),
        ('fill("56", "tv")', False),
        ('check("9")', False),
        ('press("12", "Enter")', False),
        ('', False),
        ('# scroll(0, 100)', False),  # commented-out doesn't count
    ],
)
def test_is_nav_only_action_classifies_correctly(action: str, expected: bool):
    assert _is_nav_only_action(action) is expected


# ---------------------------------------------------------------------------
# _extract_search_query
# ---------------------------------------------------------------------------

def test_extract_query_prefers_quoted_phrase():
    goal = 'Find the listing titled "Vintage Sony TV 1980" on classifieds.'
    assert _extract_search_query(goal) == "Vintage Sony TV 1980"


def test_extract_query_prefers_smart_quotes():
    goal = "Find the post titled \u201cBest pizza in NYC\u201d on reddit."
    assert _extract_search_query(goal) == "Best pizza in NYC"


def test_extract_query_falls_back_to_capitalized_run():
    goal = "Find the most expensive TV from Maryland on the page."
    # "TV" then break, "Maryland" alone — Maryland is the longest cap run.
    out = _extract_search_query(goal)
    assert out, "should not be empty"
    assert "Maryland" in out or "TV" in out


def test_extract_query_falls_back_to_longest_content_words():
    # No quoted phrase, no capitalized nouns.
    goal = "find the cheapest electronic device available"
    out = _extract_search_query(goal)
    # Longest content tokens: 'electronic' (10), 'available' (9), 'cheapest' (8), 'device' (6)
    assert "electronic" in out


def test_extract_query_handles_empty_goal():
    assert _extract_search_query("") == ""
    assert _extract_search_query("   ") == ""


def test_extract_query_skips_action_verbs_in_capitalized_run():
    # "Find" is in stop list — should not be treated as the start of the run.
    goal = "Find Maryland TV listings now."
    out = _extract_search_query(goal)
    assert "Find" not in out
    assert "Maryland" in out


# ---------------------------------------------------------------------------
# _build_anti_thrash_action
# ---------------------------------------------------------------------------

def test_build_returns_none_when_no_fill_candidate():
    candidates = ['click("12")', 'scroll(0, 300)', 'go_back()']
    assert _build_anti_thrash_action(candidates, "Find the TV listings") is None


def test_build_returns_none_when_goal_is_empty():
    candidates = ['fill("56", "...")', 'click("12")']
    assert _build_anti_thrash_action(candidates, "") is None


def test_build_synthesises_fill_with_query():
    candidates = ['scroll(0, 300)', 'fill("56", "...")', 'click("12")']
    out = _build_anti_thrash_action(candidates, 'Find "Vintage Sony TV"')
    assert out == 'fill("56", "Vintage Sony TV")'


def test_build_picks_first_fill_candidate_in_order():
    # When there are multiple fill candidates, take the first (matches the
    # order produced by ``_h_list_valid_actions``: searchboxes first).
    candidates = [
        'scroll(0, 300)',
        'fill("12", "...")',
        'fill("99", "...")',
    ]
    out = _build_anti_thrash_action(candidates, "Find Maryland TV")
    assert out is not None
    assert out.startswith('fill("12", ')


def test_build_escapes_quotes_in_query():
    # Queries containing literal double-quotes must be escaped for the
    # generated python action string to remain syntactically valid.
    candidates = ['fill("12", "...")']
    out = _build_anti_thrash_action(candidates, 'Find "Sony \\"hd\\" TV"')
    assert out is not None
    # Must not contain an unescaped quote inside the string literal.
    body = out[len('fill("12", "'):-len('")')]
    # An odd number of *escaped* quotes is fine; verify by re-evaluating:
    import ast
    tree = ast.parse(out, mode="eval")
    # Will raise SyntaxError if the action string is malformed.
    assert tree is not None


# ---------------------------------------------------------------------------
# Integration: threshold constant matches docstring expectations
# ---------------------------------------------------------------------------

def test_threshold_is_within_sane_bounds():
    # Too small fires on every page with <3 nav actions (false positives on
    # legitimate scroll-to-find loops). Too large lets the original 28-step
    # thrash continue most of the way before the override kicks in.
    assert 2 <= _MAX_CONSECUTIVE_NAV <= 6, (
        f"_MAX_CONSECUTIVE_NAV={_MAX_CONSECUTIVE_NAV} is outside sane "
        "anti-thrash bounds"
    )
