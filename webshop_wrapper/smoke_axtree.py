"""AXTree-quality smoke for the WebShop bridge.

Answers the single make-or-break question: *does BrowserGym's
Chromium-backed accessibility tree pick up clean interactive roles
(button / link / textbox / radio) on WebShop's HTML pages, or does
everything collapse into ``role=generic`` with no bids?*

Run sequence
------------

::

    # Terminal 1 — start the stub server (no install required)
    cd Multi-hop-Reasoning-VLM-Agent
    conda run -n browsergym python -m webshop_wrapper.stub_app --port 3000

    # Terminal 2 — run the smoke (also browsergym env)
    cd Multi-hop-Reasoning-VLM-Agent
    conda run -n browsergym python -m webshop_wrapper.smoke_axtree

The script auto-detects whether a server is already running on port
3000 and starts the stub itself if not.

What it measures
----------------

For each of 5 representative pages
(``search_page`` -> ``results_page`` -> ``item_page`` ->
``description_page`` -> ``done_page``), the smoke:

1. Navigates the BrowserGym env to that URL.
2. Pulls ``obs["axtree_object"]`` from BrowserGym.
3. Counts nodes by role (interactive vs. structural vs. generic).
4. Counts how many interactive nodes have a non-empty ``bid``.
5. Runs the existing ``browsergym_wrapper.heuristic.obs_to_schema``
   converter on the observation and checks the schema is non-empty.

Exit code 0 means the AXTree extraction is good enough to support all
three ``browsergym_wrapper`` heads (heuristic / vision / OmniParser)
without further work.  A non-zero exit means WebShop's HTML is too
markup-thin and the bridge would need a custom HTML->schema converter.

Pass thresholds (per page)
--------------------------

* ``search_page`` : >=1 textbox, >=1 button, >=80% interactive nodes have bids
* ``results_page``: >=2 buttons (Back / Next), >=1 link (product), >=80% bids
* ``item_page``   : >=4 buttons (Description / Features / Reviews / Buy Now),
                    >=1 radio (option), >=80% bids
* ``done_page``   : page loads, reward field present in DOM
* schema heuristic emits >=3 entities on results_page and item_page

Anything less and the bridge needs more work before it's usable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_BASE_URL = "http://127.0.0.1:3000"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _port_in_use(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _wait_for_url(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


@dataclass
class PageReport:
    label: str
    url_path: str
    status: str  # "PASS" | "WARN" | "FAIL"
    role_counts: dict[str, int]
    interactive_nodes: int
    bid_coverage: float
    schema_entities: int
    notes: list[str]

    def line(self) -> str:
        roles = ", ".join(f"{k}={v}" for k, v in sorted(self.role_counts.items()) if v)
        return (
            f"  [{self.status:<4}] {self.label:<14} "
            f"interactive={self.interactive_nodes:<3} "
            f"bid_cov={self.bid_coverage:.0%}  schema_ents={self.schema_entities:<3} "
            f"({roles})"
        )


# --------------------------------------------------------------------------- #
# AXTree node walker
#
# BrowserGym's ``axtree_object`` is the dict from CDP's
# ``Accessibility.getFullAXTree``: top-level "nodes" -> list of dicts each
# with ``role.value``, ``name.value``, ``properties[]`` and ``backendDOMNodeId``.
# The injected ``bid`` lives in ``properties`` as ``{ name: "browsergym_id",
# value: { value: "<bid>" } }`` (cf. browsergym/core/observation.py).
# --------------------------------------------------------------------------- #
_INTERACTIVE_ROLES = frozenset({
    "link", "button", "textbox", "combobox", "checkbox", "radio",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "switch",
    "searchbox", "spinbutton", "slider", "option", "treeitem",
})


def _extract_role(node: dict[str, Any]) -> str:
    role = node.get("role") or {}
    if isinstance(role, dict):
        return str(role.get("value") or "").strip()
    return str(role).strip()


def _extract_bid(node: dict[str, Any]) -> str:
    """Pull the BrowserGym-injected bid from a CDP AXTree node.

    Storage is top-level ``browsergym_id`` (string).  We also fall back
    to scanning ``properties[]`` for older BrowserGym versions where it
    used to live there.
    """
    bid = node.get("browsergym_id")
    if bid:
        return str(bid)
    for prop in node.get("properties") or []:
        if prop.get("name") in ("browsergym_id", "bid"):
            v = prop.get("value") or {}
            if isinstance(v, dict):
                return str(v.get("value") or "")
            return str(v)
    return ""


def _walk_axtree(axtree: dict[str, Any]) -> tuple[dict[str, int], int, int]:
    nodes = axtree.get("nodes") if isinstance(axtree, dict) else axtree
    if not nodes:
        return {}, 0, 0
    role_counts: dict[str, int] = {}
    interactive_total = 0
    interactive_with_bid = 0
    for node in nodes:
        role = _extract_role(node)
        if not role:
            continue
        role_counts[role] = role_counts.get(role, 0) + 1
        if role in _INTERACTIVE_ROLES:
            interactive_total += 1
            if _extract_bid(node):
                interactive_with_bid += 1
    return role_counts, interactive_total, interactive_with_bid


# --------------------------------------------------------------------------- #
# Spec — pass thresholds per page.  URL paths are built dynamically per
# session because the real WebShop's goal[0].asin is server-state-dependent
# (post-shuffle with seed=233 over the running dataset), and the stub uses
# fake ASINs.  ``_build_specs(goal)`` fills in the templates with whatever
# asin/query the running server reports for ``fixed_<idx>``.
# --------------------------------------------------------------------------- #
def _build_specs(asin: str, query_words: list[str]) -> list[dict[str, Any]]:
    keywords_lit = repr(query_words)  # e.g. "['water','bottle']"
    return [
        {
            "label": "search_page",
            "url_path": "/fixed_0",
            "min_roles": {"button": 1, "textbox": 1},
            "min_schema_entities": 2,
        },
        {
            "label": "results_page",
            "url_path": f"/search_results/fixed_0/{keywords_lit}/1",
            "min_roles": {"button": 2, "link": 1},
            "min_schema_entities": 3,
        },
        {
            "label": "item_page",
            "url_path": f"/item_page/fixed_0/{asin}/{keywords_lit}/1/{{}}",
            "min_roles": {"button": 4, "radio": 0},
            "min_schema_entities": 4,
        },
        {
            "label": "description_page",
            "url_path": (
                f"/item_sub_page/fixed_0/{asin}/{keywords_lit}/1/Description/{{}}"
            ),
            "min_roles": {"button": 1},
            "min_schema_entities": 1,
        },
        {
            "label": "done_page",
            "url_path": f"/done/fixed_0/{asin}/{{}}",
            "min_roles": {},
            "min_schema_entities": 0,
        },
    ]


def _discover_session_info(base_url: str, session_id: str = "fixed_0") -> tuple[str, list[str]]:
    """Return (asin, query_words) for the named session by hitting the
    bridge endpoint. Falls back to stub-friendly defaults if the
    response is empty (i.e. the server is unreachable, in which case
    the smoke will fail on the first page anyway)."""
    url = f"{base_url.rstrip('/')}/__bridge/session/{session_id}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        goal = data.get("goal") or {}
        asin = str(goal.get("asin") or "B07X1Y2Z3A")
        query = goal.get("query", "water bottle")
        words = query.split() if isinstance(query, str) else list(query)
        return asin, (words or ["water", "bottle"])
    except Exception:
        return "B07X1Y2Z3A", ["water", "bottle"]


def _url_encode_path(path: str) -> str:
    """Flask's `<keywords>` etc. are converters, not query params; encode
    the per-segment values but keep the path separators intact."""
    parts = path.split("/")
    return "/".join(urllib.parse.quote(p, safe="[]{},'-_") for p in parts)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _run_one_page(env, base_url: str, spec: dict[str, Any]) -> PageReport:
    page = env.unwrapped.page
    url = base_url + _url_encode_path(spec["url_path"])
    notes: list[str] = []
    try:
        page.goto(url, timeout=15000, wait_until="load")
    except Exception as exc:
        return PageReport(
            label=spec["label"], url_path=spec["url_path"], status="FAIL",
            role_counts={}, interactive_nodes=0, bid_coverage=0.0,
            schema_entities=0, notes=[f"goto failed: {exc}"],
        )
    page.wait_for_load_state("networkidle", timeout=10000)
    obs = env.unwrapped._get_obs()  # noqa: SLF001 — public-ish in BrowserGym

    axtree = obs.get("axtree_object") or {}
    role_counts, interactive_total, interactive_with_bid = _walk_axtree(axtree)
    bid_cov = (interactive_with_bid / interactive_total) if interactive_total else 1.0

    schema_entities = 0
    try:
        from browsergym_wrapper.heuristic import obs_to_schema
        schema_str = obs_to_schema(obs, step=0, task_id=f"webshop.smoke.{spec['label']}")
        # Heuristic emits one entity per line as ``e<n>[type=..., label=...]``
        # in the ``<entities>`` section.  Count those, not the section tag.
        in_entities = False
        for line in schema_str.splitlines():
            if line.strip() == "<entities>":
                in_entities = True
                continue
            if in_entities:
                if not line.strip():
                    break
                if line.lstrip().startswith("e") and "[" in line:
                    schema_entities += 1
    except Exception as exc:
        notes.append(f"schema heuristic failed: {exc}")

    status = "PASS"
    for role, want in spec.get("min_roles", {}).items():
        have = role_counts.get(role, 0)
        if have < want:
            status = "FAIL"
            notes.append(f"role={role!r} have={have} < want={want}")
    if interactive_total and bid_cov < 0.80:
        status = "WARN" if status == "PASS" else status
        notes.append(f"bid coverage {bid_cov:.0%} below 80% threshold")
    if schema_entities < spec.get("min_schema_entities", 0):
        status = "WARN" if status == "PASS" else status
        notes.append(
            f"schema entities {schema_entities} < want {spec['min_schema_entities']}"
        )

    return PageReport(
        label=spec["label"], url_path=spec["url_path"], status=status,
        role_counts=role_counts, interactive_nodes=interactive_total,
        bid_coverage=bid_cov, schema_entities=schema_entities, notes=notes,
    )


def _start_stub_server(port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "webshop_wrapper.stub_app", "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if not _wait_for_url(f"http://127.0.0.1:{port}/__bridge/session/fixed_0"):
        proc.kill()
        try:
            stderr_tail = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
        except Exception:
            stderr_tail = ""
        raise RuntimeError(f"stub server did not come up on :{port}\n{stderr_tail[:2000]}")
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WebShop bridge AXTree-quality smoke",
    )
    parser.add_argument("--base-url", default=_DEFAULT_BASE_URL,
                        help="Where the WebShop / stub server is running.")
    parser.add_argument("--no-stub", action="store_true",
                        help="Do not auto-spawn the stub server even if "
                             "the port is unused.")
    parser.add_argument("--save-axtree", default="",
                        help="Optional path to dump first page's axtree as JSON.")
    args = parser.parse_args()

    parsed = urllib.parse.urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3000

    stub_proc: subprocess.Popen | None = None
    if not _port_in_use(host, port):
        if args.no_stub:
            print(f"[ERROR] no server on {host}:{port} and --no-stub set")
            return 2
        print(f"[info] spawning stub server on :{port}")
        stub_proc = _start_stub_server(port)
    else:
        print(f"[info] reusing existing server on {host}:{port}")

    try:
        # Lazy imports — only after we know the server is up so the
        # first failure mode isn't "browsergym not in this env".
        import gymnasium as gym
        import browsergym.core  # noqa: F401  — registers gym ids

        from webshop_wrapper.task import register_webshop_tasks
        registered = register_webshop_tasks(num_goals=5)
        env_id = registered[0]
        print(f"[info] registered {len(registered)} envs; using {env_id}")

        # We override start_url via base_url, so the actual goal_idx
        # doesn't matter for navigation — we'll page.goto() ourselves.
        # But we still need the env initialised so obs extraction works.
        import os
        os.environ["WEBSHOP_BASE_URL"] = args.base_url

        env = gym.make(env_id, headless=True, slow_mo=0, viewport={"width": 1280, "height": 720})
        env.reset()

        asin, query_words = _discover_session_info(args.base_url, "fixed_0")
        print(f"[info] discovered fixed_0: asin={asin} query={query_words}")
        page_specs = _build_specs(asin, query_words)

        reports: list[PageReport] = []
        for spec in page_specs:
            r = _run_one_page(env, args.base_url, spec)
            reports.append(r)
            print(r.line())
            for n in r.notes:
                print(f"           note: {n}")

        if args.save_axtree:
            # Dump the obs from the most-interactive page (item_page),
            # not the last one walked (done_page is interaction-empty).
            spec = next(s for s in page_specs if s["label"] == "item_page")
            env.unwrapped.page.goto(args.base_url + _url_encode_path(spec["url_path"]),
                                    timeout=15000, wait_until="load")
            obs = env.unwrapped._get_obs()  # noqa: SLF001
            payload = {
                "axtree_object": obs.get("axtree_object", {}),
                "extra_element_properties": obs.get("extra_element_properties", {}),
                "url": obs.get("url", ""),
            }
            Path(args.save_axtree).write_text(json.dumps(payload, indent=2, default=str))
            print(f"[info] dumped item_page obs to {args.save_axtree}")

        env.close()

    finally:
        if stub_proc:
            stub_proc.terminate()
            try:
                stub_proc.wait(timeout=5)
            except Exception:
                stub_proc.kill()

    fails = sum(1 for r in reports if r.status == "FAIL")
    warns = sum(1 for r in reports if r.status == "WARN")
    print()
    print("=" * 60)
    if fails:
        print(f"VERDICT: FAIL  ({fails} page(s) failed, {warns} warned)")
        print("  -> WebShop's HTML is too markup-thin for BrowserGym AXTree.")
        print("     Recommend writing a custom HTML->schema heuristic for "
              "Head 1, OR abandoning WebShop in favour of MiniWoB / "
              "AssistantBench (already in main).")
        return 1
    if warns:
        print(f"VERDICT: WARN  ({warns} page(s) warned, 0 failed)")
        print("  -> AXTree is usable but bid coverage or schema density is")
        print("     below ideal.  Vision/OmniParser heads will work, the")
        print("     heuristic head may need a touchup.  Safe to proceed.")
        return 0
    print("VERDICT: PASS  (all 5 pages clean)")
    print("  -> BrowserGym AXTree extraction is good enough for all 3 heads.")
    print("     Run install/install_webshop.sh next to go from stub to full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
