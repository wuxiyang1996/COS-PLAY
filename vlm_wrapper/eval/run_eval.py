#!/usr/bin/env python
"""CLI entry point for the visual-grounding eval harness.

Wraps ``vlm_wrapper.eval.harness.run_eval`` with a benchmark dispatch
table so you can run the full Phase-1 evaluation suite from one
command.

Usage::

    python -m vlm_wrapper.eval.run_eval \\
        --benchmark tir_bench \\
        --limit 200 \\
        --model gpt-4o \\
        --output runs/eval/tir_bench.jsonl

Supported ``--benchmark`` values:

* ``visual_toolbench`` — image QA + tool use (HF); gold string (diagnostic)
* ``tir_bench``        — image QA (HF Agents-X/TIR-Bench)
* ``video_holmes``     — video QA MCQ, gold letter
* ``siv_bench``        — video QA MCQ, gold letter

Add a new benchmark by extending ``_BENCHMARKS`` below — provide an
iterator factory, a grounder, a sample-id function, and a
``gold_extractor``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("vlm_wrapper.eval")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_wrapper.eval.harness import run_eval  # noqa: E402


# ----------------------------------------------------------------------
# Benchmark dispatch
# ----------------------------------------------------------------------

def _gold_from_qa(field: str = "answer") -> Callable[[Any], dict[str, Any]]:
    """Extract gold answer + (optional) scene-graph entities for IoU."""
    def extractor(sample: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"answer": getattr(sample, field, None)}
        sg = getattr(sample, "scene_graph", None)
        if sg and isinstance(sg, dict):
            objs = sg.get("objects") or {}
            entities = []
            for _oid, obj in objs.items():
                if "x" in obj and "y" in obj and "w" in obj and "h" in obj:
                    entities.append({
                        "label": obj.get("name", ""),
                        "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                    })
            if entities:
                out["entities"] = entities
        return out
    return extractor


def _vtb_gold(sample: Any) -> dict[str, Any]:
    return {"answer": getattr(sample, "gold_answer", None)}


def _make_visual_toolbench(args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        iter_visual_toolbench_samples, parse_visual_toolbench_sample,
    )

    def grounder(sample):
        return parse_visual_toolbench_sample(
            sample,
            model=args.model, api_key=args.api_key, base_url=args.base_url,
            max_entities=args.max_entities, max_rounds=args.max_rounds,
        )

    return {
        "samples": iter_visual_toolbench_samples(limit=args.limit),
        "grounder": grounder,
        "domain": "image_qa",
        "gold_extractor": _vtb_gold,
        "sample_id_fn": lambda s: s.sample_id,
    }


def _make_tir_bench(args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples, parse_tir_bench_sample,
    )

    def grounder(sample):
        return parse_tir_bench_sample(
            sample,
            model=args.model, api_key=args.api_key, base_url=args.base_url,
            max_entities=args.max_entities, max_rounds=args.max_rounds,
        )

    return {
        "samples": iter_tir_bench_samples(split=args.split, limit=args.limit),
        "grounder": grounder,
        "domain": "image_qa",
        "gold_extractor": _gold_from_qa("answer"),
        "sample_id_fn": lambda s: s.sample_id,
    }


def _make_video_holmes(args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, parse_video_holmes_sample,
    )

    def grounder(sample):
        return parse_video_holmes_sample(
            sample,
            num_frames=args.num_frames,
            model=args.model, api_key=args.api_key, base_url=args.base_url,
            max_entities=args.max_entities, max_rounds=args.max_rounds,
        )

    return {
        "samples": iter_video_holmes_samples(
            split=args.split, limit=args.limit,
        ),
        "grounder": grounder,
        "domain": "video_qa",
        "gold_extractor": _gold_from_qa("answer"),
        "sample_id_fn": lambda s: f"{s.video_id}.Q{s.question_id}",
    }


def _make_siv_bench(args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.siv_bench import (
        iter_siv_bench_samples, parse_siv_bench_sample,
    )

    def grounder(sample):
        return parse_siv_bench_sample(
            sample,
            num_frames=args.num_frames,
            model=args.model, api_key=args.api_key, base_url=args.base_url,
            max_entities=args.max_entities, max_rounds=args.max_rounds,
        )

    return {
        "samples": iter_siv_bench_samples(
            limit=args.limit, subtitle=args.subtitle or "origin",
        ),
        "grounder": grounder,
        "domain": "video_qa",
        "gold_extractor": _gold_from_qa("answer"),
        "sample_id_fn": lambda s: f"{s.video_id}.Q{s.question_id}",
    }


_BENCHMARKS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "visual_toolbench": _make_visual_toolbench,
    "tir_bench": _make_tir_bench,
    "video_holmes": _make_video_holmes,
    "siv_bench": _make_siv_bench,
}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the visual-grounding eval harness",
    )
    p.add_argument("--benchmark", choices=list(_BENCHMARKS), required=True)
    p.add_argument("--split", default="test",
                   help="Dataset split (video: test/train; TIR-Bench: test only)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap iteration count (smoke tests)")
    p.add_argument("--model", default=os.environ.get("VLM_LABEL_MODEL", "gpt-4o"))
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--max_entities", type=int, default=20)
    p.add_argument("--max_rounds", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=8,
                   help="Frames per video (video benchmarks only)")
    p.add_argument("--subtitle", default="origin",
                   help="SIV-Bench subtitle condition")
    p.add_argument("--output", default=None,
                   help="Per-sample JSONL output path")
    p.add_argument("--report", default=None,
                   help="Aggregated metrics JSON output path "
                        "(defaults to <output>.report.json)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quiet:
        logger.setLevel(logging.WARNING)

    factory = _BENCHMARKS[args.benchmark]
    spec = factory(args)

    output_jsonl = args.output
    report_path = args.report
    if output_jsonl and not report_path:
        report_path = str(Path(output_jsonl).with_suffix(".report.json"))

    logger.info(
        "Running eval: benchmark=%s split=%s limit=%s model=%s",
        args.benchmark, args.split, args.limit, args.model,
    )
    report = run_eval(
        samples=spec["samples"],
        grounder=spec["grounder"],
        domain=spec["domain"],
        gold_extractor=spec.get("gold_extractor"),
        sample_id_fn=spec.get("sample_id_fn"),
        output_jsonl=output_jsonl,
        limit=args.limit,
    )

    summary = report.to_dict(include_rows=False)
    summary["benchmark"] = args.benchmark
    summary["split"] = args.split
    summary["model"] = args.model

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Wrote report to %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
