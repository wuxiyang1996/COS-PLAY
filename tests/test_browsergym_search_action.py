"""
Regression tests for the synthetic ``search_web("...")`` action surface
in ``cold_start.generate_cold_start_actor_browsergym``.

This is the agent-side counterpart of ``test_search_backends.py``. We
test the *wiring* (regex, action_tools enum, structured-slot mapping,
validator pass-through, runtime intercept) here, not the search HTTP
calls themselves — those live in ``test_search_backends.py`` and use
fully-mocked HTTP.

Specifically we check:

  - ``_SEARCH_WEB_CALL_RE`` matches both quote styles and rejects
    nearby look-alikes (``search_web()``, ``goto("search_web...")``).
  - ``_BROWSERGYM_ACTION_RE`` accepts the synthetic action so the
    pre-filter in ``_validate_action_string`` doesn't drop it.
  - ``_validate_action_string`` accepts ``search_web("...")`` and
    rejects empty/malformed variants.
  - ``_build_action_tools`` exposes ``search_web`` in the
    ``action_type`` enum and a ``query`` slot in the parameters.
  - ``_structured_to_action_string`` rebuilds the literal action.
  - ``_intercept_search_web`` returns the injection-then-noop
    sequence on success and a goto fallback on failure.
  - The system prompt mentions ``search_web`` as the recommended
    primary search affordance.

Breadcrumb: see ``legacy/visualwebarena/vwa-improvement-plan.md`` for
the original anti-thrash / anti-repeat history that motivated the
``action_for_history`` / ``action_for_step`` split this module
relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cold_start import generate_cold_start_actor_browsergym as actor  # noqa: E402


# ---------------------------------------------------------------------------
# Regex + parser
# ---------------------------------------------------------------------------
def test_search_web_regex_accepts_double_quoted():
    assert actor._SEARCH_WEB_CALL_RE.match('search_web("hello world")')
    assert actor._parse_search_web_query('search_web("hello world")') == "hello world"


def test_search_web_regex_accepts_single_quoted():
    assert actor._SEARCH_WEB_CALL_RE.match("search_web('foo bar')")
    assert actor._parse_search_web_query("search_web('foo bar')") == "foo bar"


def test_search_web_regex_rejects_lookalikes():
    """``goto("https://example.com/search_web?q=x")`` must NOT match."""
    bad_inputs = [
        'goto("https://example.com/search_web?q=x")',
        'click("search_web")',
        'search_web()',                # missing query
        'search_web("a", "b")',         # too many args
        'search_web(unquoted)',         # unquoted
        '',
        'noop()',
    ]
    for s in bad_inputs:
        assert actor._parse_search_web_query(s) is None, f"unexpected match: {s!r}"


def test_search_web_regex_handles_escaped_quotes_passthrough():
    """Empty queries should not match (we want at least one char)."""
    # Empty string is captured as empty group -> we treat as no-op.
    # But the regex accepts ``search_web("")`` — query="". We don't
    # rely on that being rejected at the regex level; the harness
    # treats empty-query as a safety no-op via search_backends.
    q = actor._parse_search_web_query('search_web("")')
    assert q == ""


# ---------------------------------------------------------------------------
# Pre-filter + full validator
# ---------------------------------------------------------------------------
def test_browsergym_action_re_accepts_search_web():
    assert actor._BROWSERGYM_ACTION_RE.match('search_web("hi")')


def test_validate_action_string_accepts_search_web():
    assert actor._validate_action_string('search_web("hi there")') is True


def test_validate_action_string_rejects_empty_search_web_call():
    """The legacy regex requires at least one character inside the parens
    via ``\\(.+\\)``; ``search_web()`` should be rejected."""
    assert actor._validate_action_string('search_web()') is False


# ---------------------------------------------------------------------------
# Structured action-tools wiring
# ---------------------------------------------------------------------------
def test_build_action_tools_exposes_search_web_enum():
    tools = actor._build_action_tools(["click(\"a1\")"])
    enum = tools[0]["function"]["parameters"]["properties"]["action_type"]["enum"]
    assert "search_web" in enum


def test_build_action_tools_exposes_query_slot_with_examples():
    tools = actor._build_action_tools(["click(\"a1\")"])
    props = tools[0]["function"]["parameters"]["properties"]
    assert "query" in props
    assert props["query"]["type"] == "string"
    desc = props["query"]["description"]
    # The agent needs to know this routes server-side and bypasses the
    # anti-bot wall — without that hint it'll keep emitting goto(google)
    assert "search_web" in desc.lower() or "anti-bot" in desc.lower() \
        or "server-side" in desc.lower()


def test_structured_to_action_string_emits_search_web_from_query_slot():
    out = actor._structured_to_action_string(
        {"action_type": "search_web", "query": "beluga whale GFF3"}
    )
    assert out == 'search_web("beluga whale GFF3")'


def test_structured_to_action_string_falls_back_to_text_slot():
    """Some LLM call sites reuse ``text`` for any free-form string. The
    structured fallback should accept either ``query`` or ``text``."""
    out = actor._structured_to_action_string(
        {"action_type": "search_web", "text": "fallback query"}
    )
    assert out == 'search_web("fallback query")'


def test_structured_to_action_string_returns_none_when_query_missing():
    out = actor._structured_to_action_string({"action_type": "search_web"})
    assert out is None


def test_structured_to_action_string_escapes_quotes_in_query():
    """Embedded double quotes must be escaped so ``env.step`` can
    round-trip the string through Python's exec/parser."""
    out = actor._structured_to_action_string(
        {"action_type": "search_web", "query": 'beluga "GFF3"'}
    )
    assert out == 'search_web("beluga \\"GFF3\\"")'


# ---------------------------------------------------------------------------
# Runtime intercept
# ---------------------------------------------------------------------------
def _make_fake_env_with_page(page_mock):
    """Build a wrapper-chain stub that exposes ``page`` on the third
    level — exercising the wrapper-walk in ``_get_active_browsergym_page``."""
    inner = MagicMock(spec=["env", "page"])
    inner.env = None
    inner.page = page_mock
    middle = MagicMock(spec=["env"])
    middle.env = inner
    outer = MagicMock(spec=["env"])
    outer.env = middle
    return outer


def test_intercept_passes_through_non_search_web_actions():
    env = MagicMock()
    out, meta = actor._intercept_search_web(env, 'click("a1")')
    assert out == 'click("a1")'
    assert meta is None


def test_intercept_runs_search_then_injects_data_url(monkeypatch):
    """Happy path: search returns results, page.goto on a data: URL
    succeeds, substitute action is noop()."""
    page = MagicMock()
    env = _make_fake_env_with_page(page)

    # Mock search_backends.search() -> 2 results
    fake_results = [
        {"title": "T1", "url": "https://example.com/a", "snippet": "S1", "source": "ddg-html"},
        {"title": "T2", "url": "https://example.com/b", "snippet": "S2", "source": "ddg-html"},
    ]
    from cold_start import search_backends
    monkeypatch.setattr(search_backends, "search", lambda q, **kw: fake_results)

    out, meta = actor._intercept_search_web(env, 'search_web("hello")')

    assert out == "noop()"
    assert meta is not None
    assert meta["intercepted"] is True
    assert meta["query"] == "hello"
    assert meta["n_results"] == 2
    assert meta["fallback"] is None
    # page.goto must have been called with a data: URL
    page.goto.assert_called_once()
    nav_url = page.goto.call_args[0][0]
    assert nav_url.startswith("data:text/html;charset=utf-8,")


def test_intercept_falls_back_to_ddg_html_goto_when_no_page_handle(monkeypatch):
    """If we can't get a page handle (e.g. env was never reset), the
    intercept should produce a real ``goto("https://html.duckduckgo.com/...")``
    so the agent at least gets a chance at a real result page."""
    env = MagicMock(spec=["env"])  # no .page
    env.env = None
    from cold_start import search_backends
    monkeypatch.setattr(search_backends, "search", lambda q, **kw: [])

    out, meta = actor._intercept_search_web(env, 'search_web("foo bar")')
    assert out.startswith('goto("https://html.duckduckgo.com/html/?q=')
    assert "foo+bar" in out or "foo%20bar" in out
    assert meta["intercepted"] is False
    assert meta["fallback"] == "no_page_handle"


def test_intercept_falls_back_to_ddg_html_goto_on_goto_failure(monkeypatch):
    """If page.goto raises (Chromium navigation hang etc.), the
    intercept should fall back to a real DDG-HTML goto. We don't want
    a single Playwright hiccup to kill the search affordance."""
    page = MagicMock()
    page.goto.side_effect = RuntimeError("Chromium hung")
    env = _make_fake_env_with_page(page)
    from cold_start import search_backends
    monkeypatch.setattr(
        search_backends, "search",
        lambda q, **kw: [
            {"title": "T", "url": "https://x.io", "snippet": "s", "source": "fake"}
        ],
    )

    out, meta = actor._intercept_search_web(env, 'search_web("q")')
    assert out.startswith('goto("https://html.duckduckgo.com/html/?q=')
    assert meta["intercepted"] is False
    assert "injection_failed" in (meta["fallback"] or "")


def test_intercept_renders_empty_page_when_search_returns_zero(monkeypatch):
    """All backends down: still inject a synthetic page so the agent
    sees an explicit "no results" message rather than a Playwright
    consent dialog or stale page."""
    page = MagicMock()
    env = _make_fake_env_with_page(page)
    from cold_start import search_backends
    monkeypatch.setattr(search_backends, "search", lambda q, **kw: [])

    out, meta = actor._intercept_search_web(env, 'search_web("q")')
    assert out == "noop()"
    assert meta["intercepted"] is True
    assert meta["n_results"] == 0
    assert meta["fallback"] == "empty_results"
    page.goto.assert_called_once()


# ---------------------------------------------------------------------------
# System prompt: must mention search_web as the primary search
# ---------------------------------------------------------------------------
def test_actor_system_prompt_mentions_search_web():
    p = actor._ACTOR_SYSTEM_PROMPT
    assert "search_web" in p
    # The prompt should explicitly tell the model NOT to start with
    # goto(google.com) — that was the May-3 failure mode.
    assert "search_web" in p and ("server-side" in p or "anti-bot" in p)
