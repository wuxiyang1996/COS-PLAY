"""
Server-side web search for the BrowserGym actor (``search_web`` action).

WHY THIS MODULE EXISTS
----------------------
Playwright/Chromium has a distinctive TLS+JS fingerprint that Google,
DuckDuckGo (consumer & html), Mojeek, Startpage etc. detect and block:
the agent ends up on Google's ``/sorry/index`` CAPTCHA, DDG's
``static-pages/418 teapot``, or a generic "unusual traffic" page —
all of which yield an axtree with no clickable result links. That's
the root cause of the AssistantBench reward gap: the agent never
sees candidate result URLs to click, so it eventually emits
``report_infeasible`` (or guesses the wrong source — see
``validation.10`` where the agent went to NCBI when gold expected
Ensembl).

Plain Python ``urllib`` is NOT subject to the Chromium TLS
fingerprint check — DDG-HTML, DDG-Lite and Bing all return real
result HTML when fetched from a Python process (verified May 2026).
So we run the search **server-side** (via ``urllib`` from the
harness, not via the agent's browser), parse the HTML to a list of
``{title, url, snippet}`` dicts, and inject that as a synthetic
search-results page into the agent's live Playwright page.

BACKEND CHAIN
-------------
Tried in this order, each falls through on HTTPError / parse failure
/ empty result:

1. **Tavily** — paid API designed for AI agents. Requires
   ``TAVILY_API_KEY``. Free tier: 1000 req/month. Best snippets.
2. **Serper** — Google-quality. Requires ``SERPER_API_KEY``. Free
   trial: 2500 queries.
3. **Brave Search API** — privacy-focused. Requires
   ``BRAVE_SEARCH_API_KEY``. Free tier: 2000/month.
4. **DuckDuckGo HTML** scrape — ``html.duckduckgo.com/html/?q=...``.
   No key. Verified working May 2026 from non-Playwright clients
   *with a Firefox UA*; the Chrome UA gets greylisted to an
   "anomaly_modal" CAPTCHA and we detect+skip that explicitly.
5. **DuckDuckGo Lite** scrape — ``lite.duckduckgo.com/lite/?q=...``.
   Same domain, stripped HTML4 layout, used as fallback when ``html``
   subdomain rate-limits.
6. **Yahoo** scrape — ``search.yahoo.com/search?p=...``. No key.
   Yahoo's classic SERP is still server-rendered (unlike Bing's
   post-redesign JS layout), so it works from urllib. Most reliable
   non-paid fallback as of May 2026.
7. **Wikipedia opensearch** API — entity-only fallback. No key. Used
   if everything above fails (returns links but no snippets).

API
---
Public functions:

``search(query, k=5, timeout=8) -> list[dict]``
    Multi-backend chain. Returns up to ``k`` results as
    ``[{title, url, snippet, source}, ...]``. Empty list on total
    failure (no exceptions propagate to the caller).

``render_results_html(query, results) -> str``
    Build a clean HTML search-results page from the result list.
    Designed to round-trip cleanly through BrowserGym's axtree:
    each result has a ``<a href>`` and a snippet ``<p>`` so the
    agent sees them as numbered clickable entities.

``results_to_data_url(query, results) -> str``
    URL-encode the rendered HTML into a ``data:text/html;...`` URL
    suitable for ``page.goto()``. Keeps the page navigable (real
    Chromium navigation, links work as expected when the agent
    clicks them).
"""

from __future__ import annotations

import gzip
import html
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared HTTP plumbing
# ---------------------------------------------------------------------------
# Default to a desktop Firefox UA. Empirically (verified May 2026), DDG-HTML
# greylists the matching Chrome UA and serves an "anomaly_modal" CAPTCHA page
# instead of results — but the same query with this Firefox UA returns
# normal SERP markup. Yahoo, Startpage, Wikipedia and the paid APIs are
# UA-agnostic. Connection: close avoids urllib keep-alive corner cases.
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "close",
}


def _http_get(
    url: str,
    *,
    timeout: float = 8.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """Fetch ``url`` and return the decoded body as ``str``.

    Transparently handles ``gzip`` / ``deflate`` content-encoding that some
    backends return regardless of our ``Accept-Encoding`` request.
    """
    headers = dict(_DEFAULT_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        encoding = r.headers.get("Content-Encoding", "").lower()
    if encoding == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 8.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    headers = dict(_DEFAULT_HEADERS)
    headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, headers=headers, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
    return json.loads(raw.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Result type helpers
# ---------------------------------------------------------------------------
def _new_result(title: str, url: str, snippet: str, source: str) -> Dict[str, str]:
    """Build a normalized result dict with sensible defaults."""
    return {
        "title": (title or "").strip()[:280] or "(untitled)",
        "url": (url or "").strip(),
        "snippet": (snippet or "").strip()[:600],
        "source": source,
    }


def _dedupe_results(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop duplicate URLs (keep first occurrence) and obvious empties."""
    seen: set = set()
    out: List[Dict[str, str]] = []
    for r in results:
        u = r.get("url", "").strip()
        if not u:
            continue
        # Normalize trailing slashes for dedup
        key = u.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Backend 1: Tavily (paid API, AI-native)
# ---------------------------------------------------------------------------
def _search_tavily(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(k, 10)),
        "search_depth": "basic",
        "include_answer": False,
    }
    data = _http_post_json(url, payload, timeout=timeout)
    out = []
    for item in (data.get("results") or [])[:k]:
        out.append(
            _new_result(
                item.get("title", ""),
                item.get("url", ""),
                item.get("content") or item.get("snippet") or "",
                "tavily",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Backend 2: Serper (paid API, Google-quality)
# ---------------------------------------------------------------------------
def _search_serper(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set")
    url = "https://google.serper.dev/search"
    data = _http_post_json(
        url,
        {"q": query, "num": max(1, min(k, 10))},
        timeout=timeout,
        extra_headers={"X-API-KEY": api_key},
    )
    out = []
    for item in (data.get("organic") or [])[:k]:
        out.append(
            _new_result(
                item.get("title", ""),
                item.get("link", ""),
                item.get("snippet", ""),
                "serper",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Backend 3: Brave Search API (paid, free tier)
# ---------------------------------------------------------------------------
def _search_brave(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY not set")
    qs = urllib.parse.urlencode({"q": query, "count": max(1, min(k, 10))})
    url = f"https://api.search.brave.com/res/v1/web/search?{qs}"
    body = _http_get(
        url,
        timeout=timeout,
        extra_headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
    )
    data = json.loads(body)
    out = []
    for item in (data.get("web", {}).get("results") or [])[:k]:
        out.append(
            _new_result(
                item.get("title", ""),
                item.get("url", ""),
                item.get("description") or "",
                "brave",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Backend 4: DuckDuckGo HTML scrape (free, no key)
# ---------------------------------------------------------------------------
# ``html.duckduckgo.com/html/`` wraps every result in two anchors that appear
# in fixed order on the page:
#   <a class="result__a" href="<redirector>">Title</a>
#   ...some intermediate URL chrome...
#   <a class="result__snippet" href="<redirector>">snippet HTML</a>
# We extract titles+hrefs and snippets *independently* and pair by index —
# DDG emits them 1:1 in document order. A combined regex would let the
# snippet group go unmatched (see git blame: was a real bug May 2026).
_DDG_TITLE_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
# DDG wraps the *real* destination URL in a redirector
# ``//duckduckgo.com/l/?uddg=<urlencoded original URL>``. We normalise back
# to the original URL so the agent's ``click(...)`` lands on the real site.
_DDG_REDIRECT_RE = re.compile(
    r"^(?://duckduckgo\.com|https?://(?:www\.)?duckduckgo\.com)/l/\?"
)


def _ddg_unwrap_url(href: str) -> str:
    if not _DDG_REDIRECT_RE.match(href):
        return href
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parts.query)
    inner = qs.get("uddg", [None])[0]
    return inner or href


def _strip_html(s: str) -> str:
    """Remove HTML tags and unescape entities; collapse whitespace."""
    if not s:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", s)
    text = html.unescape(no_tags)
    return re.sub(r"\s+", " ", text).strip()


def _search_ddg_html(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    qs = urllib.parse.urlencode({"q": query})
    url = f"https://html.duckduckgo.com/html/?{qs}"
    body = _http_get(url, timeout=timeout)
    # DDG occasionally short-circuits to an "anomaly_modal" CAPTCHA page
    # if our IP/UA is flagged. Detect this explicitly so the multi-
    # backend chain falls through to the next source instead of
    # returning zero results from a "successful" fetch.
    if "anomaly-modal" in body or "anomaly_modal" in body or "anomaly.js" in body:
        raise RuntimeError("ddg-html anomaly challenge served")
    titles = list(_DDG_TITLE_RE.finditer(body))
    snippets_text = [_strip_html(m.group(1)) for m in _DDG_SNIPPET_RE.finditer(body)]
    out: List[Dict[str, str]] = []
    for i, m in enumerate(titles):
        href = _ddg_unwrap_url(m.group(1))
        title = _strip_html(m.group(2))
        if not title or not href:
            continue
        snippet = snippets_text[i] if i < len(snippets_text) else ""
        out.append(_new_result(title, href, snippet, "ddg-html"))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Backend 5: DuckDuckGo Lite scrape (free, no key)
# ---------------------------------------------------------------------------
# Same domain as DDG-HTML but a stripped HTML4 / table layout. Tested as a
# fallback when the "html" subdomain is rate-limited but ``lite`` is not.
_DDG_LITE_LINK_RE = re.compile(
    r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_DDG_LITE_SNIPPET_RE = re.compile(
    r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
    re.DOTALL | re.IGNORECASE,
)


def _search_ddg_lite(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    qs = urllib.parse.urlencode({"q": query})
    url = f"https://lite.duckduckgo.com/lite/?{qs}"
    body = _http_get(url, timeout=timeout)
    if "anomaly-modal" in body or "anomaly.js" in body:
        raise RuntimeError("ddg-lite anomaly challenge served")
    links = list(_DDG_LITE_LINK_RE.finditer(body))
    snippets = [_strip_html(m.group(1)) for m in _DDG_LITE_SNIPPET_RE.finditer(body)]
    out: List[Dict[str, str]] = []
    for i, m in enumerate(links):
        href = _ddg_unwrap_url(m.group(1))
        title = _strip_html(m.group(2))
        if not href or not title:
            continue
        snip = snippets[i] if i < len(snippets) else ""
        out.append(_new_result(title, href, snip, "ddg-lite"))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Backend 6: Yahoo scrape (free, no key)
# ---------------------------------------------------------------------------
# Yahoo's classic ``search.yahoo.com/search?p=...`` SERP is server-rendered,
# so unlike the post-redesign Bing it works from server-side urllib calls.
# Each organic result is wrapped in ``<h3 class="title"><a href=...>`` with
# a sibling ``<div class="compText"> ... <p>snippet</p> ... </div>``.
# We extract paired titles + descriptions; the first few <h3.title a> hits
# are nav tabs (Images, Video, News etc.) — we filter those by URL host.
_YAHOO_TITLE_RE = re.compile(
    r'<h3[^>]*class="[^"]*\btitle\b[^"]*"[^>]*>'
    r'\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_YAHOO_SNIPPET_RE = re.compile(
    r'<div[^>]*class="[^"]*\bcompText\b[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)
# Yahoo wraps result hrefs in a ``r.search.yahoo.com/.../RU=<urlencoded>/RK=...``
# redirector with multiple slash-separated tracking segments (``_ylt=`` /
# ``RV=`` / ``RE=``) before the ``RU=`` payload. ``.*?/RU=`` is lazy so
# we don't gobble past the URL; ``[^/]+`` then captures the encoded
# destination URL up to the next ``/RK=`` tracking marker.
_YAHOO_REDIRECT_RE = re.compile(r"https?://r\.search\.yahoo\.com/.*?/RU=([^/]+)/RK=")


def _yahoo_unwrap_url(href: str) -> str:
    m = _YAHOO_REDIRECT_RE.search(href)
    if m:
        try:
            return urllib.parse.unquote(m.group(1))
        except Exception:
            pass
    return href


def _search_yahoo(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    qs = urllib.parse.urlencode({"p": query})
    url = f"https://search.yahoo.com/search?{qs}"
    body = _http_get(url, timeout=timeout)
    snippets = [_strip_html(m.group(1))[:600] for m in _YAHOO_SNIPPET_RE.finditer(body)]
    # Filter nav-only anchors FIRST so the title-to-snippet pairing
    # below indexes against real organic results only — Yahoo emits a
    # ``compText`` only for real results, not for the Images/Video/etc.
    # vertical-search nav tabs that share the ``h3.title a`` shape.
    real_titles: List[Tuple[str, str]] = []
    for m in _YAHOO_TITLE_RE.finditer(body):
        href = _yahoo_unwrap_url(m.group(1).strip())
        title = _strip_html(m.group(2))
        if not href.startswith("http") or not title:
            continue
        host = urllib.parse.urlparse(href).hostname or ""
        if host.endswith("yahoo.com") or host.endswith("yimg.com"):
            continue
        real_titles.append((href, title))
    out: List[Dict[str, str]] = []
    for i, (href, title) in enumerate(real_titles):
        snip = snippets[i] if i < len(snippets) else ""
        out.append(_new_result(title, href, snip, "yahoo"))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Backend 7: Wikipedia opensearch (free, no key)
# ---------------------------------------------------------------------------
def _search_wikipedia(query: str, k: int, timeout: float) -> List[Dict[str, str]]:
    qs = urllib.parse.urlencode(
        {"action": "opensearch", "format": "json", "search": query, "limit": max(1, min(k, 10))}
    )
    url = f"https://en.wikipedia.org/w/api.php?{qs}"
    body = _http_get(url, timeout=timeout)
    data = json.loads(body)
    if not isinstance(data, list) or len(data) < 4:
        return []
    titles = data[1]
    descs = data[2] if isinstance(data[2], list) else [""] * len(titles)
    urls = data[3] if isinstance(data[3], list) else [""] * len(titles)
    out: List[Dict[str, str]] = []
    for i, t in enumerate(titles[:k]):
        u = urls[i] if i < len(urls) else ""
        d = descs[i] if i < len(descs) else ""
        if not u:
            continue
        out.append(_new_result(t, u, d, "wikipedia"))
    return out


# ---------------------------------------------------------------------------
# Public API: search() — multi-backend fallback chain
# ---------------------------------------------------------------------------
# Order: paid first (best quality), then free scrapes (fast & no rate quota),
# then Wikipedia (entity-only) as a last-ditch source of clickable URLs.
_BACKEND_CHAIN: List[Tuple[str, Any]] = [
    ("tavily", _search_tavily),
    ("serper", _search_serper),
    ("brave", _search_brave),
    ("ddg-html", _search_ddg_html),
    ("ddg-lite", _search_ddg_lite),
    ("yahoo", _search_yahoo),
    ("wikipedia", _search_wikipedia),
]


def search(
    query: str,
    *,
    k: int = 5,
    timeout: float = 8.0,
    backend_chain: Optional[List[Tuple[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Run a server-side web search and return up to ``k`` results.

    Returns an empty list on total backend failure rather than raising;
    callers should treat empty as "search blocked / no signal" and fall
    back to a different strategy (direct goto, etc.).

    Parameters
    ----------
    query : str
        The free-form query string. Must be non-empty after strip.
    k : int
        Max results to return. Most backends honour this verbatim;
        Wikipedia caps at 10.
    timeout : float
        Per-backend HTTP timeout (seconds).
    backend_chain : optional
        Override the default chain (mainly for tests).
    """
    q = (query or "").strip()
    if not q:
        return []
    chain = backend_chain or _BACKEND_CHAIN
    for name, fn in chain:
        try:
            results = fn(q, k, timeout)
        except Exception as e:
            logger.debug("search backend %s failed: %s", name, e)
            continue
        results = _dedupe_results(results)
        if results:
            logger.info(
                "search(%r) -> %d results from %s", q[:60], len(results), name
            )
            return results[:k]
    logger.warning("search(%r) -> all backends failed", q[:60])
    return []


# ---------------------------------------------------------------------------
# Rendering: turn results into a synthetic search-results page
# ---------------------------------------------------------------------------
_RESULTS_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Search results: {query_esc}</title>
<style>
body {{ font-family: sans-serif; max-width: 760px; margin: 16px auto; padding: 0 16px; color: #1a1a1a; }}
h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
.meta {{ font-size: 13px; color: #555; margin-bottom: 12px; }}
.result {{ margin-bottom: 16px; }}
.result h3 {{ font-size: 16px; margin: 4px 0; }}
.result h3 a {{ color: #1a0dab; text-decoration: none; }}
.result .url {{ font-size: 12px; color: #006621; word-break: break-all; }}
.result .snippet {{ font-size: 13px; color: #4d5156; line-height: 1.4; }}
.empty {{ font-size: 14px; color: #b00; }}
</style>
</head>
<body>
<h1>Web search results</h1>
<p class="meta">Query: <strong>{query_esc}</strong> &middot; {result_count} results &middot; <em>(server-side: {sources_str})</em></p>
{results_html}
</body>
</html>"""


def render_results_html(query: str, results: List[Dict[str, str]]) -> str:
    """Render results as a self-contained HTML page.

    The page is designed to round-trip cleanly through BrowserGym's
    axtree extractor: each result is a top-level ``<div class="result">``
    with an ``<a href>``, a separate URL line, and a snippet, so the
    set-of-marks pass labels them with discrete bids the agent can
    ``click("...")``.
    """
    query_esc = html.escape(query or "")
    if not results:
        results_html = (
            '<p class="empty">No results found. Try a different query or '
            'navigate directly to a known site (Wikipedia, news outlet, '
            'etc.) by URL.</p>'
        )
        sources_str = "no backends available"
    else:
        sources_str = ", ".join(sorted({r.get("source", "?") for r in results}))
        chunks = []
        for i, r in enumerate(results, 1):
            t = html.escape(r.get("title", "") or "(untitled)")
            u = html.escape(r.get("url", "") or "")
            s = html.escape(r.get("snippet", "") or "")
            chunks.append(
                f'<div class="result">'
                f'<h3>{i}. <a href="{u}">{t}</a></h3>'
                f'<div class="url">{u}</div>'
                f'<p class="snippet">{s}</p>'
                f'</div>'
            )
        results_html = "\n".join(chunks)
    return _RESULTS_HTML_TEMPLATE.format(
        query_esc=query_esc,
        result_count=len(results),
        sources_str=html.escape(sources_str),
        results_html=results_html,
    )


def results_to_data_url(query: str, results: List[Dict[str, str]]) -> str:
    """Encode a rendered results page as a ``data:text/html;...`` URL.

    Returns a string suitable for ``page.goto(...)`` — Chromium accepts
    data URLs natively. The encoding is percent-quoted (no base64) so
    log lines stay roughly readable when debugging.
    """
    body = render_results_html(query, results)
    # ``safe=""`` percent-encodes everything but URL gen-delims; this is
    # required for Chrome to accept the data URL when body contains '#'
    # in headings or URLs.
    encoded = urllib.parse.quote(body, safe="")
    return f"data:text/html;charset=utf-8,{encoded}"
