"""Driver that turns a benchmark stream into per-sample + aggregated metrics.

The harness is benchmark-agnostic: callers provide a stream of
``EvalSample`` records (input + gold) and a callable that runs the
grounding pipeline on one input.  The harness handles iteration,
exception isolation, on-disk JSONL append, semantic validation, and
aggregation.

Typical use::

    from vlm_wrapper.eval import run_eval
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples,
        parse_tir_bench_sample,
    )

    def grounder(sample):
        return parse_tir_bench_sample(sample, model="gpt-4o", api_key=KEY)

    report = run_eval(
        samples=iter_tir_bench_samples(split="test", limit=200),
        grounder=grounder,
        domain="image_qa",
        gold_extractor=lambda s: {
            "answer": s.answer,
            "schema": None,
            "entities": None,
            "needed_tools": None,
        },
        sample_id_fn=lambda s: s.sample_id,
        output_jsonl="runs/eval/tir_bench.jsonl",
    )
    print(report.metrics.to_dict())
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..schema import (
    parse_schema_output,
    semantic_validate,
)
from .metrics import (
    EvalMetrics,
    PerSampleResult,
    compute_answer_accuracy,
    compute_entity_iou,
    compute_field_accuracy,
    compute_format_compliance,
    compute_target_accuracy,
    compute_tool_precision,
    summarise_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """The full output of ``run_eval``: per-sample rows + aggregated metrics."""

    domain: str
    n_samples: int
    elapsed_s: float
    metrics: EvalMetrics
    rows: list[PerSampleResult] = field(default_factory=list)
    output_jsonl: str | None = None

    def to_dict(self, include_rows: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "domain": self.domain,
            "n_samples": self.n_samples,
            "elapsed_s": round(self.elapsed_s, 2),
            "metrics": self.metrics.to_dict(),
            "output_jsonl": self.output_jsonl,
        }
        if include_rows:
            out["rows"] = [r.to_dict() for r in self.rows]
        return out


def _default_sample_id(sample: Any, idx: int) -> str:
    """Best-effort sample id when the caller didn't supply ``sample_id_fn``."""
    for attr in ("question_id", "sample_id", "image_filename", "video_id"):
        v = getattr(sample, attr, None)
        if v is not None:
            return str(v)
    return f"sample_{idx}"


def _normalise_grounder_output(out: Any) -> dict[str, Any]:
    """Coerce whatever the per-benchmark parser returns into a uniform dict.

    All ``parse_<benchmark>_sample`` helpers in
    ``visual_reasoning_wrapper.benchmarks.*`` already return dicts with the same key
    set we need (``schema``, ``answer``, ``tool_trace``, ``head_used``,
    ``escalation_trace``, ``warnings``).  This wrapper just guards
    against the harness being pointed at a custom grounder that returns
    a ``GroundingResult`` directly.
    """
    if isinstance(out, dict):
        return out
    return {
        "schema": getattr(out, "schema", None),
        "answer": getattr(out, "answer", None),
        "tool_trace": getattr(out, "tool_trace", []),
        "head_used": getattr(out, "head_used", None),
        "escalation_trace": getattr(out, "escalation_trace", []),
        "warnings": getattr(out, "warnings", []),
    }


def run_eval(
    samples: Iterable[Any],
    grounder: Callable[[Any], Any],
    *,
    domain: str,
    gold_extractor: Callable[[Any], dict[str, Any]] | None = None,
    sample_id_fn: Callable[[Any], str] | None = None,
    output_jsonl: str | Path | None = None,
    progress_every: int = 10,
    limit: int | None = None,
    image_size_fn: Callable[[Any], tuple[int, int] | None] | None = None,
) -> EvalReport:
    """Run the evaluation harness on a sample stream.

    Parameters
    ----------
    samples : iterable
        Yields one benchmark sample per item.  No schema is required —
        we just pass each sample to ``grounder``.
    grounder : callable(sample) -> dict | GroundingResult
        Runs the grounding pipeline on one sample.  The return value
        must include keys (or attributes) ``schema``, ``answer``,
        ``tool_trace``, ``head_used``, ``escalation_trace``.
    domain : str
        ``gymv`` / ``browser`` / ``desktop`` / ``image_qa`` / ``video_qa``.
        Drives the semantic validator's required-section list.
    gold_extractor : callable(sample) -> dict, optional
        Returns ground-truth data for the sample.  Recognised keys:

        - ``answer``      : str — expected answer (QA benchmarks)
        - ``schema``      : str — gold ``<state>…</state>`` text (rare)
        - ``entities``    : list[{label, bbox}] — for IoU
        - ``needed_tools``: list[str] — oracle of tools the gold uses
    sample_id_fn : callable(sample) -> str, optional
        Override the default sample-id derivation.
    output_jsonl : path, optional
        Append per-sample rows to this file as JSONL — survives
        interruption.
    progress_every : int
        Log a progress line every N samples.
    limit : int, optional
        Cap iteration count for smoke tests.
    image_size_fn : callable(sample) -> (w, h) or None, optional
        Used by the semantic validator to range-check ``pos=`` values.
    """
    rows: list[PerSampleResult] = []
    fh = None
    if output_jsonl is not None:
        out_path = Path(output_jsonl)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(out_path, "a", encoding="utf-8")

    t0 = time.time()
    try:
        for i, sample in enumerate(samples, 1):
            if limit is not None and i > limit:
                break

            sid = (
                sample_id_fn(sample) if sample_id_fn
                else _default_sample_id(sample, i)
            )
            try:
                raw = grounder(sample)
                out = _normalise_grounder_output(raw)
            except Exception as exc:
                logger.warning("[eval %s] sample %s failed: %s", domain, sid, exc)
                row = PerSampleResult(
                    sample_id=sid, domain=domain, error=str(exc),
                )
                rows.append(row)
                if fh is not None:
                    fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                    fh.flush()
                continue

            schema = out.get("schema")
            head_used = out.get("head_used")
            esc = out.get("escalation_trace") or []
            tool_trace = out.get("tool_trace") or []

            fmt_ok, fmt_warnings = compute_format_compliance(schema)
            sem = semantic_validate(
                schema, domain=domain,
                image_size=(image_size_fn(sample) if image_size_fn else None),
            )

            gold = gold_extractor(sample) if gold_extractor else {}
            gold_schema = gold.get("schema")
            gold_answer = gold.get("answer")
            gold_entities = gold.get("entities")
            needed_tools = gold.get("needed_tools")

            field_acc = (
                compute_field_accuracy(schema, gold_schema)
                if gold_schema else {}
            )
            ans_ok = (
                compute_answer_accuracy(out.get("answer"), gold_answer)
                if gold_answer is not None else None
            )
            tgt_ok, blk_ok = (
                compute_target_accuracy(schema, gold_schema)
                if gold_schema else (None, None)
            )
            iou, iou_breakdown = (
                compute_entity_iou(schema, gold_entities)
                if gold_entities else (None, {})
            )
            tp, tr = compute_tool_precision(tool_trace, needed_tools)

            row = PerSampleResult(
                sample_id=sid,
                domain=domain,
                head_used=head_used,
                n_escalations=max(0, len(esc) - 1),
                format_ok=fmt_ok,
                format_warnings=fmt_warnings,
                semantic_valid=sem.valid,
                semantic_errors=sem.errors,
                field_accuracy=field_acc,
                answer_correct=ans_ok,
                target_correct=tgt_ok,
                blocker_correct=blk_ok,
                entity_iou=iou,
                entity_iou_breakdown=iou_breakdown,
                tool_precision=tp,
                tool_recall=tr,
            )
            rows.append(row)

            if fh is not None:
                fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            if i % progress_every == 0:
                logger.info(
                    "[eval %s] %d samples — fmt=%.2f sem=%.2f ans=%s",
                    domain, i,
                    sum(1 for r in rows if r.format_ok) / len(rows),
                    sum(1 for r in rows if r.semantic_valid) / len(rows),
                    (
                        f"{sum(1 for r in rows if r.answer_correct)}/"
                        f"{sum(1 for r in rows if r.answer_correct is not None)}"
                    ) if any(r.answer_correct is not None for r in rows) else "n/a",
                )
    finally:
        if fh is not None:
            fh.close()

    metrics = summarise_metrics(rows, by_domain=False)
    return EvalReport(
        domain=domain,
        n_samples=len(rows),
        elapsed_s=time.time() - t0,
        metrics=metrics,
        rows=rows,
        output_jsonl=str(output_jsonl) if output_jsonl else None,
    )


__all__ = ["EvalReport", "run_eval"]
