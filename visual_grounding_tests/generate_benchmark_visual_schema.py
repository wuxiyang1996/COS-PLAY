#!/usr/bin/env python
"""
Generate skill-ready visual grounding schemas for the four visual-reasoning
benchmarks.

Covered benchmarks come from ``visual_reasoning_wrapper``:

  - visual_toolbench: image tool-use reasoning
  - tir_bench: image thinking-with-images reasoning
  - video_holmes: short-video multi-hop evidence reasoning
  - siv_bench: social-interaction video reasoning

Each sample is passed through the benchmark parser, which calls the shared
``vlm_wrapper.ground.cascaded_ground`` pipeline and returns a structured
``<state>...</state>`` schema. This script writes a normalized JSONL record
with ``schema_for_skills`` so downstream skill learners can consume one field
without caring which benchmark produced it.

Usage from the repo root::

    export OPENAI_API_KEY=...
    python visual_grounding_tests/generate_benchmark_visual_schema.py --limit 2

    # Video-only smoke run.
    python visual_grounding_tests/generate_benchmark_visual_schema.py \\
        --benchmarks video_holmes siv_bench --limit 1 --num_frames 8

    # Metadata-only check, no model/API calls.
    python visual_grounding_tests/generate_benchmark_visual_schema.py \\
        --benchmarks visual_toolbench tir_bench --limit 3 --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))

DEFAULT_BENCHMARKS = (
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)
DEFAULT_MODEL = os.environ.get(
    "VLM_BENCH_SCHEMA_MODEL",
    os.environ.get("VLM_LABEL_MODEL", "gpt-5.5"),
)
DEFAULT_OUTPUT_TAG = "benchmark_visual_schemas"


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    modality: str
    sample_iter: Callable[[argparse.Namespace], Iterator[Any]]
    parse_sample: Callable[[Any, argparse.Namespace], dict[str, Any]]
    sample_id: Callable[[Any], str]
    question: Callable[[Any], str | None]
    gold_answer: Callable[[Any], str | None]


def _as_jsonable(value: Any) -> Any:
    """Best-effort conversion for dataclasses, Paths, PIL objects, etc."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _as_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {"image_size": list(value.size), "mode": value.mode}
    return str(value)


def _validation_to_dict(validation: Any) -> Any:
    if validation is None:
        return None
    if isinstance(validation, dict):
        return _as_jsonable(validation)
    if hasattr(validation, "as_dict"):
        try:
            return _as_jsonable(validation.as_dict())
        except Exception:
            return str(validation)
    return _as_jsonable(validation)


def _sample_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        try:
            out = sample.to_dict()
            if isinstance(out, dict):
                return _as_jsonable(out)
        except Exception:
            pass
    return _as_jsonable(sample) if isinstance(sample, dict) else {"repr": repr(sample)}


def _make_record(
    *,
    spec: BenchmarkSpec,
    sample: Any,
    result: dict[str, Any] | None,
    error: str | None,
    elapsed_s: float,
    dry_run: bool,
    model: str,
) -> dict[str, Any]:
    schema = result.get("schema") if result else None
    return {
        "benchmark": spec.key,
        "modality": spec.modality,
        "sample_id": spec.sample_id(sample),
        "question": spec.question(sample),
        "gold_answer": spec.gold_answer(sample),
        "answer": result.get("answer") if result else None,
        "answer_raw": result.get("answer_raw") if result else None,
        "correct": result.get("correct") if result else None,
        "schema_for_skills": schema,
        "schema_valid": bool(schema),
        "tool_trace": _as_jsonable(result.get("tool_trace") if result else []),
        "rounds": result.get("rounds") if result else 0,
        "validation": _validation_to_dict(result.get("validation") if result else None),
        "warnings": _as_jsonable(result.get("warnings") if result else []),
        "head_used": result.get("head_used") if result else None,
        "escalation_trace": _as_jsonable(result.get("escalation_trace") if result else None),
        "sample": _sample_dict(sample),
        "raw_result": _as_jsonable(result) if result else None,
        "error": error,
        "dry_run": dry_run,
        "model": model,
        "elapsed_seconds": round(elapsed_s, 3),
    }


def _iter_visual_toolbench(args: argparse.Namespace) -> Iterator[Any]:
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        iter_visual_toolbench_samples,
    )

    return iter_visual_toolbench_samples(
        limit=args.limit,
        single_turn_only=not args.include_multi_turn_visual_toolbench,
    )


def _parse_visual_toolbench(sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        parse_visual_toolbench_sample,
    )

    return parse_visual_toolbench_sample(
        sample,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
        chain=args.image_chain or None,
    )


def _iter_tir_bench(args: argparse.Namespace) -> Iterator[Any]:
    from visual_reasoning_wrapper.benchmarks.tir_bench import iter_tir_bench_samples

    return iter_tir_bench_samples(
        split=args.split,
        limit=args.limit,
        task_filter=args.tir_task_filter,
    )


def _parse_tir_bench(sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.tir_bench import parse_tir_bench_sample

    return parse_tir_bench_sample(
        sample,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
        chain=args.image_chain or None,
    )


def _iter_video_holmes(args: argparse.Namespace) -> Iterator[Any]:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples,
    )

    question_types = args.video_holmes_question_types or None
    return iter_video_holmes_samples(
        split=args.split,
        limit=args.limit,
        video_holmes_root=args.video_holmes_root,
        question_types=question_types,
    )


def _parse_video_holmes(sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        parse_video_holmes_sample,
    )

    return parse_video_holmes_sample(
        sample,
        num_frames=args.num_frames,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
        max_side=args.max_side,
    )


def _iter_siv_bench(args: argparse.Namespace) -> Iterator[Any]:
    from visual_reasoning_wrapper.benchmarks.siv_bench import iter_siv_bench_samples

    return iter_siv_bench_samples(
        limit=args.limit,
        siv_root=args.siv_root,
        subtitle=args.subtitle,
        dimensions=args.siv_dimensions or None,
        subtasks=args.siv_subtasks or None,
    )


def _parse_siv_bench(sample: Any, args: argparse.Namespace) -> dict[str, Any]:
    from visual_reasoning_wrapper.benchmarks.siv_bench import parse_siv_bench_sample

    return parse_siv_bench_sample(
        sample,
        num_frames=args.num_frames,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
        max_side=args.max_side,
    )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "visual_toolbench": BenchmarkSpec(
        key="visual_toolbench",
        modality="image",
        sample_iter=_iter_visual_toolbench,
        parse_sample=_parse_visual_toolbench,
        sample_id=lambda s: str(s.sample_id),
        question=lambda s: getattr(s, "question", None),
        gold_answer=lambda s: getattr(s, "gold_answer", None),
    ),
    "tir_bench": BenchmarkSpec(
        key="tir_bench",
        modality="image",
        sample_iter=_iter_tir_bench,
        parse_sample=_parse_tir_bench,
        sample_id=lambda s: str(s.sample_id),
        question=lambda s: getattr(s, "prompt", None),
        gold_answer=lambda s: getattr(s, "answer", None),
    ),
    "video_holmes": BenchmarkSpec(
        key="video_holmes",
        modality="video",
        sample_iter=_iter_video_holmes,
        parse_sample=_parse_video_holmes,
        sample_id=lambda s: f"{s.video_id}.Q{s.question_id}",
        question=lambda s: getattr(s, "question", None),
        gold_answer=lambda s: getattr(s, "answer", None),
    ),
    "siv_bench": BenchmarkSpec(
        key="siv_bench",
        modality="video",
        sample_iter=_iter_siv_bench,
        parse_sample=_parse_siv_bench,
        sample_id=lambda s: f"{s.video_id}.Q{s.question_id}",
        question=lambda s: getattr(s, "question", None),
        gold_answer=lambda s: getattr(s, "answer", None),
    ),
}


def _run_benchmark(
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    output_jsonl: Path,
) -> dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total = 0
    schema_ok = 0
    errors = 0

    with output_jsonl.open("w", encoding="utf-8") as fh:
        try:
            samples = iter(spec.sample_iter(args))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if not args.continue_on_error:
                raise
            fh.write(
                json.dumps(
                    {
                        "benchmark": spec.key,
                        "modality": spec.modality,
                        "sample_id": None,
                        "schema_for_skills": None,
                        "schema_valid": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "error_stage": "sample_iterator_setup",
                        "dry_run": args.dry_run,
                        "model": args.model,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n",
            )
            samples = iter(())

        while True:
            try:
                sample = next(samples)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if not args.continue_on_error:
                    raise
                fh.write(
                    json.dumps(
                        {
                            "benchmark": spec.key,
                            "modality": spec.modality,
                            "sample_id": None,
                            "schema_for_skills": None,
                            "schema_valid": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "error_stage": "sample_iteration",
                            "dry_run": args.dry_run,
                            "model": args.model,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n",
                )
                break

            total += 1
            t0 = time.time()
            result: dict[str, Any] | None = None
            error: str | None = None
            try:
                if not args.dry_run:
                    result = spec.parse_sample(sample, args)
                    if result.get("schema"):
                        schema_ok += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                error = f"{type(exc).__name__}: {exc}"
                if args.verbose:
                    traceback.print_exc()
                if not args.continue_on_error:
                    record = _make_record(
                        spec=spec,
                        sample=sample,
                        result=None,
                        error=error,
                        elapsed_s=time.time() - t0,
                        dry_run=args.dry_run,
                        model=args.model,
                    )
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    raise

            record = _make_record(
                spec=spec,
                sample=sample,
                result=result,
                error=error,
                elapsed_s=time.time() - t0,
                dry_run=args.dry_run,
                model=args.model,
            )
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()

            if args.verbose:
                status = "schema" if record["schema_for_skills"] else "no-schema"
                if error:
                    status = "error"
                print(f"  [{spec.key}] {total}: {record['sample_id']} {status}")

    return {
        "benchmark": spec.key,
        "modality": spec.modality,
        "records": total,
        "schema_for_skills_ok": schema_ok,
        "errors": errors,
        "output_jsonl": str(output_jsonl),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate skill-ready <state> schemas for visual reasoning benchmarks.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARKS),
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to run. Default: all four visual_reasoning_wrapper benchmarks.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Samples per benchmark.")
    parser.add_argument("--split", default="test", help="Split for TIR/Video-Holmes.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_entities", type=int, default=20)
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_side", type=int, default=640)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: visual_grounding_tests/output/benchmark_visual_schemas/<run_id>",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Iterate samples and write metadata, but do not call the VLM parser.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Write error records and continue instead of failing fast.",
    )
    parser.add_argument(
        "--include_multi_turn_visual_toolbench",
        action="store_true",
        help="Include VisualToolBench multi-turn rows. Default keeps single-turn only.",
    )
    parser.add_argument("--tir_task_filter", default=None)
    parser.add_argument(
        "--image_chain",
        nargs="+",
        default=None,
        choices=("vlm", "tool_loop", "omniparser", "heuristic"),
        help=(
            "Cascade chain for VisualToolBench / TIR-Bench. "
            "Default (None) lets each parser pick its own — currently "
            "['tool_loop'] for both, since both papers expect tool use. "
            "Pass 'vlm tool_loop' for the cheap escalating chain or "
            "'vlm' to bypass tools entirely."
        ),
    )
    parser.add_argument("--video_holmes_root", default=None)
    parser.add_argument("--video_holmes_question_types", nargs="*", default=None)
    parser.add_argument("--siv_root", default=None)
    parser.add_argument("--subtitle", default="origin")
    parser.add_argument("--siv_dimensions", nargs="*", default=None)
    parser.add_argument("--siv_subtasks", nargs="*", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "output" / DEFAULT_OUTPUT_TAG / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    master: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "model": args.model,
        "dry_run": args.dry_run,
        "limit_per_benchmark": args.limit,
        "benchmarks": list(args.benchmarks),
        "outputs": [],
    }

    for key in args.benchmarks:
        spec = BENCHMARKS[key]
        output_jsonl = out_dir / f"{key}.skills_schema.jsonl"
        print(f"\n-> {key} ({spec.modality}) -> {output_jsonl}")
        try:
            summary = _run_benchmark(spec, args, output_jsonl)
        except Exception as exc:  # noqa: BLE001
            summary = {
                "benchmark": key,
                "modality": spec.modality,
                "records": None,
                "schema_for_skills_ok": None,
                "errors": 1,
                "output_jsonl": str(output_jsonl),
                "fatal_error": f"{type(exc).__name__}: {exc}",
            }
            master["outputs"].append(summary)
            with (out_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
                json.dump(master, fh, ensure_ascii=False, indent=2, default=str)
            raise

        master["outputs"].append(summary)
        print(
            "   wrote {records} records, schema_ok={schema_for_skills_ok}, "
            "errors={errors}".format(**summary),
        )

    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(master, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nBatch summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
