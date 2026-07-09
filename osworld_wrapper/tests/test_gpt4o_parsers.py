"""GPT-4o adapter contract + live schema test for OSWorld (desktop).

Two tiers:

* **offline** — verifies that
  :func:`osworld_wrapper.adapter.generate_label` (and the legacy shim
  ``vlm_wrapper.osworld_adapter.generate_label``) is importable and
  callable.

* **live**    — marked ``@pytest.mark.live``; skipped unless
  ``OPENAI_API_KEY`` (or ``VLM_TEST_API_KEY``) is set.  Sends a
  synthesized desktop screenshot to GPT-4o and asserts that a
  parseable ``<state>…</state>`` schema comes back.  Schemas are saved
  under ``out/schemas/desktop.schema.txt``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_vlm_parsers import _synthesize_desktop  # noqa: E402

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


def test_offline_osworld_adapter_import() -> None:
    from osworld_wrapper.adapter import generate_label
    assert callable(generate_label)


def test_offline_legacy_shim_osworld_adapter_import() -> None:
    from vlm_wrapper.osworld_adapter import generate_label as shim_generate
    from osworld_wrapper.adapter import generate_label as canon_generate
    assert shim_generate is canon_generate


@live
@needs_api
def test_live_desktop_schema() -> None:
    from osworld_wrapper.adapter import generate_label

    image = _synthesize_desktop()
    result = generate_label(
        image,
        instruction="Open the Files application.  What pyautogui action would you take first?",
        task_id="osworld.demo",
        step=0,
        max_entities=15,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = result.get("schema")
    _assert_valid_schema(schema, "desktop")
    _save_schema("desktop", schema)
