"""Regression tests for the anti-repetition override (#6e) in the BrowserGym actor.

The override addresses the May-3 2026 finding that ~73 % of steps on
``visualwebarena.96`` and ~70 % on ``.433`` are wasted on repeating the
same action signature (e.g. ``click("211")`` 7×, ``go_back()`` 10×).
The override:

  1. Tracks each executed action's *signature* in a sliding window of
     ``_REPEAT_WINDOW`` recent steps.
  2. Computes a *discouraged* set: signatures with count ≥
     ``_MAX_REPEATS_BEFORE_DISCOURAGE`` in the window.
  3. (#3b) Drops discouraged signatures from the candidate-action list
     handed to the action LLM, **except** for the protected types
     ``go_back / go_forward / noop`` which always remain available as
     recovery escape hatches.
  4. (#6c2) If the LLM picks a discouraged signature anyway (off-list
     pick), swaps to a non-discouraged candidate.

These tests cover the four pure helpers in
``cold_start/generate_cold_start_actor_browsergym.py``:

  * ``_action_signature``
  * ``_is_repeat_protected``
  * ``_build_discouraged_signatures``
  * ``_filter_repeat_candidates``

See also:
  * ``legacy/visualwebarena/vwa-improvement-plan.md`` §11
    (anti-repetition mechanism — full root-cause and impact)
"""
from __future__ import annotations

import sys

import pytest


sys.path.insert(0, "/workspace/Multi-hop-Reasoning-VLM-Agent")


from cold_start.generate_cold_start_actor_browsergym import (  # type: ignore  # noqa: E402
    _MAX_REPEATS_BEFORE_DISCOURAGE,
    _MIN_CANDIDATES_AFTER_FILTER,
    _REPEAT_WINDOW,
    _action_signature,
    _build_discouraged_signatures,
    _filter_repeat_candidates,
    _is_repeat_protected,
)


# ---------------------------------------------------------------------------
# _action_signature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,expected",
    [
        ('click("211")', "click(211)"),
        ('click( "211" )', "click(211)"),
        ('click(\'211\')', "click(211)"),
        ('click("211")  ', "click(211)"),
        ('fill("18", "music")', "fill(18,music)"),
        ('fill("18","music")', "fill(18,music)"),
        ('fill( "18" , "music" )', "fill(18,music)"),
        ('press("54", "Enter")', "press(54,Enter)"),
        ('scroll(0, 300)', "scroll(0,300)"),
        ('scroll(0, -300)', "scroll(0,-300)"),
        ('go_back()', "go_back"),
        ('go_forward()', "go_forward"),
        ('noop()', "noop"),
        ('', ""),
        ('click("211"', 'click("211"'),  # malformed — leave as-is
    ],
)
def test_action_signature_normalises_consistently(action, expected):
    assert _action_signature(action) == expected


def test_action_signature_collapses_quote_styles_to_same_key():
    """The whole point of the helper is dedup: ``click("211")`` and
    ``click('211')`` MUST hash to the same signature so the discouragement
    set can collapse them into one count."""
    sigs = {
        _action_signature('click("211")'),
        _action_signature("click('211')"),
        _action_signature('click(211)'),
        _action_signature('click( "211" )'),
    }
    assert len(sigs) == 1, f"expected 1 unique signature, got {sigs}"


# ---------------------------------------------------------------------------
# _is_repeat_protected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,protected",
    [
        ('go_back()', True),
        ('go_forward()', True),
        ('noop()', True),
        ('  go_back()', True),  # leading whitespace tolerated
        ('scroll(0, 300)', False),  # NOT protected — windowed dedup applies
        ('click("12")', False),
        ('fill("18", "music")', False),
        ('press("54", "Enter")', False),
        ('check("9")', False),
        ('', False),
    ],
)
def test_is_repeat_protected_classifies_correctly(action, protected):
    assert _is_repeat_protected(action) is protected


# ---------------------------------------------------------------------------
# _build_discouraged_signatures
# ---------------------------------------------------------------------------

def test_build_discouraged_returns_empty_for_empty_history():
    assert _build_discouraged_signatures([]) == {}


def test_build_discouraged_below_threshold_returns_empty():
    history = ["click(1)", "click(2)", "click(3)"]  # all distinct
    assert _build_discouraged_signatures(history) == {}


def test_build_discouraged_at_threshold_returns_signature():
    """``_MAX_REPEATS_BEFORE_DISCOURAGE`` defaults to 2 — so 2 occurrences
    in the window already trigger discouragement (the 3rd attempt is what
    gets blocked)."""
    history = ["click(211)"] * _MAX_REPEATS_BEFORE_DISCOURAGE
    out = _build_discouraged_signatures(history)
    assert "click(211)" in out
    assert out["click(211)"] == _MAX_REPEATS_BEFORE_DISCOURAGE


def test_build_discouraged_window_constraint_excludes_old_history():
    """Repetitions outside the sliding window must NOT count — the agent
    is allowed to revisit a signature after enough other actions have
    intervened."""
    history = (
        ["click(211)"] * 5  # 5 OLD repeats
        + ["click(99)"] * (_REPEAT_WINDOW - 1)  # crowd them out of the window
    )
    out = _build_discouraged_signatures(history)
    assert "click(211)" not in out, (
        "old repeats should fall outside the window"
    )


def test_build_discouraged_replicates_visualwebarena_96_pattern():
    """Pre-fix smoke: ``click("211")`` 7× and ``go_back()`` 10× across
    30 steps. The last _REPEAT_WINDOW steps should have both signatures
    over the threshold."""
    history = []
    for _ in range(7):
        history += ['click(211)', 'go_back']
    out = _build_discouraged_signatures(history)
    assert "click(211)" in out
    assert "go_back" in out


# ---------------------------------------------------------------------------
# _filter_repeat_candidates
# ---------------------------------------------------------------------------

def test_filter_returns_input_unchanged_when_nothing_discouraged():
    candidates = ['click("12")', 'fill("18", "abc")', 'go_back()']
    assert _filter_repeat_candidates(candidates, {}) is candidates


def test_filter_drops_discouraged_clicks_keeps_protected():
    candidates = [
        'click("211")',
        'click("99")',
        'go_back()',
        'fill("50", "music")',
        'scroll(0, 300)',
    ]
    discouraged = {"click(211)": 3}
    out = _filter_repeat_candidates(candidates, discouraged)
    # click("211") must be dropped
    assert 'click("211")' not in out
    # click("99") (different bid) must stay
    assert 'click("99")' in out
    # protected actions stay
    assert 'go_back()' in out


def test_filter_keeps_go_back_even_when_overrepresented():
    """``go_back`` is an escape hatch — even if its signature appears in
    the discouraged set, the filter must keep it in the candidate list."""
    candidates = ['click("99")', 'fill("50", "abc")', 'go_back()']
    discouraged = {"go_back": 5}  # over-represented
    out = _filter_repeat_candidates(candidates, discouraged)
    assert 'go_back()' in out


def test_filter_backs_off_if_too_few_survivors():
    """If discouraging would leave the agent with < ``_MIN_CANDIDATES_AFTER_FILTER``
    options, fall back to the unfiltered list — the agent always needs
    something to pick."""
    candidates = ['click("211")', 'click("212")']  # only 2 options, both clicks
    discouraged = {"click(211)": 3, "click(212)": 3}
    out = _filter_repeat_candidates(candidates, discouraged)
    # Surviving = 0 (both dropped, no protected) → must back off
    assert out == candidates
    assert len(out) >= _MIN_CANDIDATES_AFTER_FILTER or out == candidates


def test_filter_drops_repeated_fill_with_same_value():
    """Anti-repetition catches ``fill("54", "f/music")`` 4× pattern from
    visualwebarena.433. Different fill on same bid stays available."""
    candidates = [
        'fill("54", "f/music")',
        'fill("54", "music")',  # different value
        'fill("99", "f/music")',  # different bid
        'go_back()',
        'click("42")',
    ]
    discouraged = {"fill(54,f/music)": 3}
    out = _filter_repeat_candidates(candidates, discouraged)
    assert 'fill("54", "f/music")' not in out
    assert 'fill("54", "music")' in out
    assert 'fill("99", "f/music")' in out


def test_filter_drops_scroll_when_overrepeated():
    """Scroll IS NOT in the protected list — windowed scroll-spam should
    be filtered. The consecutive-NOOP override (#6) handles back-to-back
    identical scrolls; #6e adds the interleaved case."""
    candidates = [
        'scroll(0, 300)',
        'scroll(0, -300)',
        'click("99")',
        'go_back()',
    ]
    discouraged = {"scroll(0,300)": 3}
    out = _filter_repeat_candidates(candidates, discouraged)
    assert 'scroll(0, 300)' not in out
    assert 'scroll(0, -300)' in out  # different scroll direction = different sig


# ---------------------------------------------------------------------------
# Constants sanity check
# ---------------------------------------------------------------------------

def test_constants_are_within_sane_bounds():
    assert _REPEAT_WINDOW >= 4, (
        f"_REPEAT_WINDOW={_REPEAT_WINDOW} too small — would over-trigger "
        "on legitimate exploration"
    )
    assert _REPEAT_WINDOW <= 16, (
        f"_REPEAT_WINDOW={_REPEAT_WINDOW} too large — would let bad "
        "patterns persist for half the episode"
    )
    assert 2 <= _MAX_REPEATS_BEFORE_DISCOURAGE <= 4
    assert _MIN_CANDIDATES_AFTER_FILTER >= 2
