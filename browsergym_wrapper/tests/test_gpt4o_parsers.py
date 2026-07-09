"""GPT-4o adapter contract + live schema test for BrowserGym.

Two tiers:

* **offline** — always run; verifies that
  :func:`browsergym_wrapper.adapter.generate_label` (and the legacy
  shim ``vlm_wrapper.browser_adapter.generate_label``) is importable
  and callable.

* **live**    — marked ``@pytest.mark.live``; skipped unless
  ``OPENAI_API_KEY`` (or ``VLM_TEST_API_KEY``) is set.  Sends a
  synthesized Wikipedia-style page to GPT-4o and asserts that a
  parseable ``<state>…</state>`` schema with an ``<entities>`` block
  comes back.  Schemas are saved under ``out/schemas/browser.schema.txt``.

Run offline only (default)::

    pytest browsergym_wrapper/tests/test_gpt4o_parsers.py -q

Run including the live GPT-4o call::

    pytest -m "live or not live" browsergym_wrapper/tests/test_gpt4o_parsers.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_vlm_parsers import _synthesize_wiki_page  # noqa: E402

SCHEMAS_DIR = REPO_ROOT / "out" / "schemas"
API_KEY = os.environ.get("VLM_TEST_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")

live = pytest.mark.live
needs_api = pytest.mark.skipif(
    not API_KEY, reason="OPENAI_API_KEY / VLM_TEST_API_KEY not set"
)


def _save_schema(case: str, schema: str) -> None:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEMAS_DIR / f"{case}.schema.txt").write_text(schema, encoding="utf-8")


def _assert_valid_schema(schema: str | None, case: str) -> None:
    """Structural sanity: presence of ``<state>`` block and either
    ``<entities>`` (interactive/image-QA domains) or
    ``<events>``/``<answer>`` (video domain)."""
    assert schema, f"{case}: parser returned no schema"
    assert "<state>" in schema and "</state>" in schema, (
        f"{case}: missing <state>…</state>:\n{schema[:400]}"
    )
    has_entities = "<entities>" in schema
    has_events = "<events>" in schema
    has_answer = "<answer>" in schema
    assert has_entities or has_events or has_answer, (
        f"{case}: schema lacks any of <entities>/<events>/<answer>:\n"
        f"{schema[:400]}"
    )


def test_offline_browser_adapter_import() -> None:
    from browsergym_wrapper.adapter import generate_label
    assert callable(generate_label)


def test_offline_legacy_shim_browser_adapter_import() -> None:
    """The vlm_wrapper compatibility shim must still expose the same callable."""
    from vlm_wrapper.browser_adapter import generate_label as shim_generate
    from browsergym_wrapper.adapter import generate_label as canon_generate
    assert shim_generate is canon_generate


@live
@needs_api
def test_live_browser_schema() -> None:
    from browsergym_wrapper.adapter import generate_label

    image = _synthesize_wiki_page()
    result = generate_label(
        image,
        goal="Identify the first section heading on the Wikipedia main page.",
        task_id="wiki.main_page.demo",
        step=0,
        url="https://en.wikipedia.org",
        max_entities=15,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = result.get("schema")
    _assert_valid_schema(schema, "browser")
    assert "<entities>" in schema
    _save_schema("browser", schema)
