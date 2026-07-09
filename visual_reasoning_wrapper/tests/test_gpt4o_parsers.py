"""Visual-reasoning benchmark contract + live schema tests.

Two tiers:

* **offline** — always run; verifies the synthetic-image helpers
  (``_synthesize_2048_board`` / ``_synthesize_wiki_page`` /
  ``_synthesize_desktop``) and the TIR-Bench / Video-Holmes loaders
  (``iter_*_samples`` + dataclass shape).

* **live**    — marked ``@pytest.mark.live``; skipped unless
  ``OPENAI_API_KEY`` (or ``VLM_TEST_API_KEY``) is set.  Runs
  ``parse_tir_bench_sample`` and ``parse_video_holmes_sample`` against
  one real sample each and asserts that the produced schema contains
  an ``<answer>`` block.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_vlm_parsers import (  # noqa: E402
    _synthesize_2048_board,
    _synthesize_desktop,
    _synthesize_wiki_page,
)

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


def test_offline_synthesizers_produce_rgb_images() -> None:
    for fn in (_synthesize_2048_board, _synthesize_wiki_page,
               _synthesize_desktop):
        img = fn()
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert min(img.size) > 64


def test_offline_tir_bench_loader_yields_samples() -> None:
    pytest.importorskip("datasets")
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        TIRBenchSample,
        iter_tir_bench_samples,
    )

    try:
        samples = list(iter_tir_bench_samples(split="test", limit=1))
    except Exception as exc:
        pytest.skip(f"TIR-Bench HF load skipped: {exc}")
    assert samples
    s = samples[0]
    assert isinstance(s, TIRBenchSample)
    assert s.prompt and s.sample_id


def test_offline_video_holmes_loader_yields_samples() -> None:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, VideoHolmesSample,
    )

    samples = list(iter_video_holmes_samples(split="test", limit=3))
    assert samples, "Video-Holmes loader returned no samples"
    for s in samples:
        assert isinstance(s, VideoHolmesSample)
        assert s.question and s.question_id
        assert s.options, "each Video-Holmes question should carry options"


@live
@needs_api
def test_live_tir_bench_schema() -> None:
    pytest.importorskip("datasets")
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples,
        parse_tir_bench_sample,
    )

    try:
        sample = next(iter_tir_bench_samples(split="test", limit=1))
    except Exception as exc:
        pytest.skip(f"TIR-Bench HF load skipped: {exc}")
    out = parse_tir_bench_sample(
        sample,
        model=MODEL,
        api_key=API_KEY,
        max_entities=10,
        max_rounds=4,
    )
    schema = out.get("schema")
    _assert_valid_schema(schema, "tir_bench")
    assert "<answer>" in schema, "TIR-Bench schema must contain <answer>"
    assert out.get("answer"), "TIR-Bench parser returned empty answer"
    _save_schema("tir_bench", schema)


@live
@needs_api
def test_live_video_holmes_schema() -> None:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, parse_video_holmes_sample,
    )

    chosen = None
    for s in iter_video_holmes_samples(split="test", limit=50):
        if s.video_path and s.video_path.exists():
            chosen = s
            break
    if chosen is None:
        pytest.skip(
            "No Video-Holmes video clips on disk — see "
            "install/INSTALL_BENCHMARKS.md §5 to download them.")

    out = parse_video_holmes_sample(
        chosen,
        num_frames=4,
        max_rounds=2,
        max_entities=12,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = out.get("schema")
    _assert_valid_schema(schema, "video_holmes")
    assert "<answer>" in schema, "Video-Holmes schema must contain <answer>"
    _save_schema("video_holmes", schema)
