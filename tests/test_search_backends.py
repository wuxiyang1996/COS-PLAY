"""
Regression tests for ``cold_start.search_backends``.

Exercises the multi-backend fallback chain, the per-backend HTML/JSON
parsers (using captured upstream payloads — no live HTTP), the result
dedupe, and the ``render_results_html`` / ``results_to_data_url``
output formats. Designed to catch the specific bugs we hit during
development of the AssistantBench search-API workaround:

  - DDG-HTML serving an "anomaly_modal" CAPTCHA page when our UA was
    flagged (must raise from the parser, not return zero quietly so
    the fallback chain can fall through)
  - DDG-HTML snippet regex mistakenly making the snippet group
    optional via ``(?:...)?`` — captured snippets were always None
  - DDG redirector URLs (``//duckduckgo.com/l/?uddg=...``) leaking
    into the result list when the unwrap step had a typo
  - Yahoo nav-only anchors (Images / Video / News tabs) being picked
    up as if they were organic results
  - data: URL output not being percent-encoded so Chromium would
    refuse the navigation when titles contained ``#`` or ``%``
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

import pytest

# Path setup — stand alone so the test runs even when CWD != repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cold_start import search_backends as sb  # noqa: E402


# ---------------------------------------------------------------------------
# Captured upstream HTML samples (kept as constants for hermetic testing).
# ---------------------------------------------------------------------------
_DDG_HTML_GOOD = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fmart.ensembl.org%2FDelphinapterus_leucas%2FInfo%2FIndex">Delphinapterus_leucas - Ensembl genome browser 115</a>
  <a class="result__url" href="https://mart.ensembl.org/Delphinapterus_leucas/Info/Index">https://mart.ensembl.org</a>
  <a class="result__snippet" href="//ddg/redir">Search Beluga whale (Delphinapterus leucas) e.g. SCN11A or ML702066.1</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.ncbi.nlm.nih.gov%2Fdatasets%2Fassembly%2FGCA_029941455.3%2F">Delphinapterus leucas genome assembly</a>
  <a class="result__snippet" href="//ddg/redir">An official website of the United States government</a>
</div>
</body></html>
"""

_DDG_HTML_ANOMALY = (
    """<html><body><form id="challenge-form" action="//duckduckgo.com/anomaly.js?cc=botnet">"""
    """<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>"""
    """</form></body></html>"""
)

_YAHOO_GOOD = """
<html><body>
<!-- nav-only result that should be filtered -->
<h3 class="title"><a href="https://images.search.yahoo.com/search/images;_ylt=AwrijnGn?p=cat">Images</a></h3>
<!-- real organic result via Yahoo redirector -->
<h3 class="title"><a href="https://r.search.yahoo.com/_ylt=AwrijnGn/RV=2/RE=1234/RU=https%3a%2f%2fdocs.python.org%2f3%2flibrary%2furllib.html/RK=2/RS=abc-">Python urllib documentation</a></h3>
<div class="compText"><p>The urllib package collects modules for working with URLs.</p></div>
<!-- second real result -->
<h3 class="title"><a href="https://r.search.yahoo.com/_ylt=AwrijnGn/RV=2/RE=1234/RU=https%3a%2f%2fwww.geeksforgeeks.org%2fpython%2ffoo/RK=2/RS=abc-">GeeksforGeeks: Python urllib</a></h3>
<div class="compText"><p>Tutorial-style overview of the Python urllib library.</p></div>
</body></html>
"""

_DDG_LITE_GOOD = """
<html><body>
<table>
<tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Example title</a></td></tr>
<tr><td class="result-snippet">Example snippet text.</td></tr>
</table>
</body></html>
"""

_WIKIPEDIA_OPENSEARCH = json.dumps(
    [
        "beluga",
        ["Beluga whale", "Beluga (sturgeon)"],
        ["The beluga whale is an Arctic cetacean.", "The beluga sturgeon is a species of fish."],
        ["https://en.wikipedia.org/wiki/Beluga_whale", "https://en.wikipedia.org/wiki/Beluga_(sturgeon)"],
    ]
)


# ---------------------------------------------------------------------------
# Patch helpers — feed canned bodies into the parser without HTTP.
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_http(monkeypatch):
    """Drop-in replacement for ``_http_get`` that serves canned bodies
    keyed by ``(host_substring, expected_param_substring)``. Returns a
    callable that registers ``(matcher, body_or_exception)`` pairs."""
    routes: list = []

    def fake_http_get(url, *, timeout=8.0, extra_headers=None):
        for matcher, body in routes:
            if all(needle in url for needle in matcher):
                if isinstance(body, BaseException):
                    raise body
                return body
        raise AssertionError(f"unmocked HTTP GET: {url!r}  (routes={routes})")

    monkeypatch.setattr(sb, "_http_get", fake_http_get)

    def add_route(matchers, body):
        routes.append((tuple(matchers), body))

    return add_route


# ---------------------------------------------------------------------------
# DDG-HTML parser
# ---------------------------------------------------------------------------
def test_ddg_html_extracts_titles_hrefs_snippets(mock_http):
    mock_http(["html.duckduckgo.com"], _DDG_HTML_GOOD)
    out = sb._search_ddg_html("beluga whale GFF3", k=5, timeout=8)
    assert len(out) == 2
    assert out[0]["title"].startswith("Delphinapterus_leucas")
    # URL was unwrapped from /l/?uddg=...
    assert out[0]["url"] == "https://mart.ensembl.org/Delphinapterus_leucas/Info/Index"
    # Snippet must NOT be empty (regression: combined regex made group(3) optional)
    assert "Beluga whale" in out[0]["snippet"]
    assert all(r["source"] == "ddg-html" for r in out)


def test_ddg_html_anomaly_modal_raises(mock_http):
    """When DDG serves the anti-bot challenge page, the parser MUST raise
    so the multi-backend chain falls through. Returning empty silently
    would leave the agent staring at zero results when in fact the
    next backend would have served real ones."""
    mock_http(["html.duckduckgo.com"], _DDG_HTML_ANOMALY)
    with pytest.raises(RuntimeError, match="anomaly"):
        sb._search_ddg_html("anything", k=5, timeout=8)


def test_ddg_html_respects_k(mock_http):
    mock_http(["html.duckduckgo.com"], _DDG_HTML_GOOD)
    out = sb._search_ddg_html("q", k=1, timeout=8)
    assert len(out) == 1


def test_ddg_redirect_unwrap():
    """``//duckduckgo.com/l/?uddg=<urlencoded>`` and the bare https
    variant both round-trip back to the original URL."""
    target = "https://example.com/some/page?x=1"
    encoded = urllib.parse.quote(target, safe="")
    assert sb._ddg_unwrap_url(f"//duckduckgo.com/l/?uddg={encoded}") == target
    assert sb._ddg_unwrap_url(f"https://duckduckgo.com/l/?uddg={encoded}") == target
    # Non-redirector URLs pass through unchanged
    assert sb._ddg_unwrap_url("https://example.com/foo") == "https://example.com/foo"


# ---------------------------------------------------------------------------
# DDG-Lite parser
# ---------------------------------------------------------------------------
def test_ddg_lite_extracts_results(mock_http):
    mock_http(["lite.duckduckgo.com"], _DDG_LITE_GOOD)
    out = sb._search_ddg_lite("anything", k=5, timeout=8)
    assert len(out) == 1
    assert out[0]["url"] == "https://example.com"
    assert out[0]["snippet"] == "Example snippet text."
    assert out[0]["source"] == "ddg-lite"


def test_ddg_lite_anomaly_raises(mock_http):
    mock_http(["lite.duckduckgo.com"], _DDG_HTML_ANOMALY)
    with pytest.raises(RuntimeError, match="anomaly"):
        sb._search_ddg_lite("q", k=5, timeout=8)


# ---------------------------------------------------------------------------
# Yahoo parser
# ---------------------------------------------------------------------------
def test_yahoo_filters_nav_tabs_and_unwraps_redirector(mock_http):
    mock_http(["search.yahoo.com"], _YAHOO_GOOD)
    out = sb._search_yahoo("python urllib", k=5, timeout=8)
    # Nav tab "Images" with images.search.yahoo.com host MUST be filtered
    assert len(out) == 2
    hosts = [urllib.parse.urlparse(r["url"]).hostname for r in out]
    assert "images.search.yahoo.com" not in hosts
    assert hosts[0] == "docs.python.org"
    assert hosts[1] == "www.geeksforgeeks.org"
    assert all(r["source"] == "yahoo" for r in out)
    # Snippet pairing — first compText goes to first result.
    assert "urllib package" in out[0]["snippet"]


def test_yahoo_unwrap_helper():
    href = (
        "https://r.search.yahoo.com/_ylt=AAA/RV=2/RE=1/"
        "RU=https%3a%2f%2fexample.com%2f/RK=2/RS=BBB-"
    )
    assert sb._yahoo_unwrap_url(href) == "https://example.com/"
    # Non-redirector URLs pass through.
    assert sb._yahoo_unwrap_url("https://example.com/x") == "https://example.com/x"


# ---------------------------------------------------------------------------
# Wikipedia parser
# ---------------------------------------------------------------------------
def test_wikipedia_parses_opensearch(mock_http):
    mock_http(["en.wikipedia.org"], _WIKIPEDIA_OPENSEARCH)
    out = sb._search_wikipedia("beluga", k=5, timeout=8)
    assert len(out) == 2
    assert out[0]["title"] == "Beluga whale"
    assert out[0]["url"].endswith("/wiki/Beluga_whale")
    assert out[0]["source"] == "wikipedia"


# ---------------------------------------------------------------------------
# Multi-backend fallback chain
# ---------------------------------------------------------------------------
def test_search_falls_through_to_next_backend_on_failure():
    """First backend raises; second succeeds. ``search`` should return
    the second backend's results without surfacing the first failure."""
    def boom(query, k, timeout):
        raise RuntimeError("backend 1 down")

    def good(query, k, timeout):
        return [sb._new_result("hit", "https://example.com", "snip", "fake")]

    chain = [("first", boom), ("second", good)]
    out = sb.search("foo", k=5, backend_chain=chain)
    assert len(out) == 1
    assert out[0]["source"] == "fake"


def test_search_returns_empty_on_total_failure():
    chain = [
        ("a", lambda q, k, t: (_ for _ in ()).throw(RuntimeError("a down"))),
        ("b", lambda q, k, t: (_ for _ in ()).throw(RuntimeError("b down"))),
    ]
    assert sb.search("foo", k=5, backend_chain=chain) == []


def test_search_skips_empty_backends_and_continues():
    chain = [
        ("first", lambda q, k, t: []),
        ("second", lambda q, k, t: [sb._new_result("ok", "https://x.io", "s", "fake")]),
    ]
    out = sb.search("foo", backend_chain=chain)
    assert len(out) == 1
    assert out[0]["url"] == "https://x.io"


def test_search_dedupes_results():
    """Duplicate URLs should be folded into a single result, with the
    first occurrence kept."""
    def two_dups(q, k, t):
        return [
            sb._new_result("a1", "https://example.com/x", "s1", "fake"),
            sb._new_result("a2", "https://example.com/x/", "s2", "fake"),  # trailing slash
            sb._new_result("a3", "https://other.com/y", "s3", "fake"),
        ]
    chain = [("dup", two_dups)]
    out = sb.search("q", k=5, backend_chain=chain)
    assert len(out) == 2
    assert out[0]["title"] == "a1"  # first occurrence kept
    assert out[1]["url"] == "https://other.com/y"


def test_search_strips_empty_query():
    assert sb.search("", k=5) == []
    assert sb.search("   ", k=5) == []


# ---------------------------------------------------------------------------
# Rendering: HTML page + data: URL
# ---------------------------------------------------------------------------
def test_render_results_html_well_formed():
    results = [
        sb._new_result("Title 1", "https://example.com/a", "Snippet A", "ddg-html"),
        sb._new_result("Title 2", "https://example.com/b", "Snippet B", "yahoo"),
    ]
    body = sb.render_results_html("my query", results)
    # Header
    assert "<title>Search results: my query</title>" in body
    # Per-result links
    assert 'href="https://example.com/a"' in body
    assert 'href="https://example.com/b"' in body
    # Snippets
    assert "Snippet A" in body and "Snippet B" in body
    # Source disclosure
    assert "ddg-html" in body and "yahoo" in body
    # Result count in meta line
    assert "2 results" in body


def test_render_results_html_empty_results_shows_friendly_msg():
    body = sb.render_results_html("nothing matched", [])
    assert "No results found" in body
    assert "nothing matched" in body


def test_render_results_html_escapes_html_in_titles_and_snippets():
    """Adversarial titles/snippets should NOT leak raw HTML into the
    rendered page (XSS avoidance + clean axtree extraction)."""
    results = [
        sb._new_result(
            "<script>alert(1)</script>",
            "https://example.com/x",
            "Snippet with <b>bold</b> & ampersand",
            "fake",
        )
    ]
    body = sb.render_results_html("q", results)
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body
    assert "&amp; ampersand" in body


def test_results_to_data_url_decodes_back_to_html():
    results = [sb._new_result("t", "https://x.io", "s", "fake")]
    data_url = sb.results_to_data_url("query", results)
    assert data_url.startswith("data:text/html;charset=utf-8,")
    payload = data_url[len("data:text/html;charset=utf-8,"):]
    decoded = urllib.parse.unquote(payload)
    assert "<title>Search results: query</title>" in decoded
    assert "https://x.io" in decoded


def test_results_to_data_url_handles_special_chars_in_query():
    """Queries with #, %, & must be URL-safe in the data URL."""
    results = [sb._new_result("t", "https://x.io", "s", "fake")]
    data_url = sb.results_to_data_url("a & b #c %d", results)
    payload = data_url[len("data:text/html;charset=utf-8,"):]
    # Naked '#' would break Chromium's data URL parse; must be percent-encoded.
    assert "#" not in payload
    decoded = urllib.parse.unquote(payload)
    assert "a &amp; b #c %d" in decoded


# ---------------------------------------------------------------------------
# Paid-API wrappers gracefully decline when keys absent
# ---------------------------------------------------------------------------
def test_tavily_serper_brave_raise_without_keys(monkeypatch):
    for env in ("TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_SEARCH_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    for fn in (sb._search_tavily, sb._search_serper, sb._search_brave):
        with pytest.raises(RuntimeError, match="not set"):
            fn("q", 5, 8)
