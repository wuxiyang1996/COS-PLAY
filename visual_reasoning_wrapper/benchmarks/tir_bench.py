"""TIR-Bench loader + VLM parser (image QA).

`TIR-Bench: A Comprehensive Benchmark for Agentic Thinking-with-Images
Reasoning <https://arxiv.org/abs/2511.01833>`_ — 13 task families, 1 215
test questions, distributed as ``Agents-X/TIR-Bench`` on HuggingFace.

Requires the ``datasets`` package and network (or a fully-populated HF
cache) on first use.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from vlm_wrapper.ground import GroundingRequest, cascaded_ground
from ..question_router import classify_question
from ._hf_images import decode_hf_image

logger = logging.getLogger(__name__)

_HF_ID = "Agents-X/TIR-Bench"


@dataclass
class TIRBenchSample:
    """One TIR-Bench row (single or dual image; we feed ``image_1`` to the VLM)."""

    sample_id: str
    task: str
    prompt: str
    answer: str | None
    image_1: Any  # raw HF cell for decode_hf_image
    image_2: Any | None = None
    meta_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "prompt": self.prompt,
            "answer": self.answer,
            "meta_data": self.meta_data,
        }


def default_tir_bench_root(workspace_root: str | Path | None = None) -> Path:
    """Optional local mirror root (``data/TIR-Bench``); HF cache is the default source."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "TIR-Bench"


def _load_dataset():
    """Load TIR-Bench rows.

    Prefers the local mirror at ``data/datasets/TIR-Bench/TIR-Bench.json``
    (a plain list of dicts produced by ``hf snapshot_download``) so we
    don't require network access or ``datasets`` features that newer
    library versions have removed (e.g. ``trust_remote_code``).
    """
    repo_root = Path(__file__).resolve().parents[2]
    local_json = repo_root / "data" / "datasets" / "TIR-Bench" / "TIR-Bench.json"
    if local_json.exists():
        with local_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install HuggingFace `datasets` to use TIR-Bench "
            "(see pyproject optional / vlm_benchmarks env)."
        ) from exc
    try:
        return load_dataset(_HF_ID, split="test", trust_remote_code=True)
    except (ValueError, TypeError) as exc:
        if "trust_remote_code" not in str(exc):
            raise
        return load_dataset(_HF_ID, split="test")


def iter_tir_bench_samples(
    split: str = "test",
    *,
    limit: int | None = None,
    task_filter: str | None = None,
) -> Iterator[TIRBenchSample]:
    """Yield ``TIRBenchSample`` rows (HF ``test`` split only).

    Parameters
    ----------
    split
        Ignored except for logging — upstream only ships ``test``.
    task_filter
        If set, only rows whose ``task`` field equals this string.
    """
    if split != "test":
        logger.warning("TIR-Bench only defines a `test` split; ignoring split=%r", split)

    ds = _load_dataset()
    n = 0
    for i in range(len(ds)):
        row = ds[i]
        task = str(row.get("task") or "")
        if task_filter is not None and task != task_filter:
            continue
        sid = str(row.get("id", i))
        prompt = str(row.get("prompt") or "")
        ans = row.get("answer")
        ans_s = str(ans).strip() if ans is not None else None
        meta = row.get("meta_data")
        meta_d = meta if isinstance(meta, dict) else None
        yield TIRBenchSample(
            sample_id=sid,
            task=task,
            prompt=prompt,
            answer=ans_s,
            image_1=row.get("image_1"),
            image_2=row.get("image_2"),
            meta_data=meta_d,
        )
        n += 1
        if limit is not None and n >= limit:
            return


def load_tir_bench_image(sample: TIRBenchSample) -> Image.Image:
    """Decode ``image_1`` (and ignore ``image_2`` unless we add tiling later)."""
    return decode_hf_image(sample.image_1)


def parse_tir_bench_sample(
    sample: TIRBenchSample,
    *,
    image: Image.Image | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 20,
    max_rounds: int = 4,
    chain: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``cascaded_ground`` on one TIR-Bench question (``domain=image_qa``).

    TIR-Bench's whole framing is *agentic thinking-with-images* — the
    model is supposed to crop / zoom / re-detect across ≥ 2 hops.  We
    therefore default ``chain=["tool_loop"]`` so the single-shot ``vlm``
    head doesn't short-circuit the cascade with a hallucinated schema.
    Override with ``chain=["vlm", "tool_loop"]`` for the cheaper
    escalating chain.
    """
    if image is None:
        image = load_tir_bench_image(sample)

    task_id = f"tir_bench.{sample.task}.{sample.sample_id}"
    routing = classify_question(sample.prompt, modality="image")
    routing_block = routing.to_prompt_block()
    goal = (
        f"{sample.prompt}\n"
        "TIR-Bench is an agentic thinking-with-images benchmark — you "
        "MUST ground your answer with at least one tool call "
        "(detect_objects / grounded_detect / zoom_region / "
        "read_text_region / describe_region / spatial_query / "
        "count_objects / measure_distance) before emitting <answer>. "
        "Each <evidence> hop must cite the real `tool=` you called and "
        "reference the entity IDs it produced (e.g. `result_ref=e1,e2`).\n"
        "Reasoning tools available: count_value, compute_ratio, "
        "compare_values, verify_claim — call them to RECORD numeric "
        "computations (counts, ratios, comparisons) before claiming "
        "the answer.  Cite the resulting `derivation_id` (d1, d2, …) "
        "inside <derivations> and <answer>.\n"
        f"{routing_block}\n"
        "Answer concisely (MCQ: single letter A–J, otherwise digits or "
        "text exactly as the question requests). scene_type=image_qa."
    )
    req = GroundingRequest(
        images=image,
        goal=goal,
        domain="image_qa",
        output_mode="answer",
        task_id=task_id,
        step=0,
        context={
            "benchmark": "tir_bench",
            "task": sample.task,
            "meta_data": sample.meta_data,
            "question_classes": routing.classes,
            "required_reasoning_tools": routing.required_tools,
            "derivation_kinds": routing.derivation_kinds,
        },
        max_entities=max_entities,
        max_rounds=max_rounds,
        model=model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
        temperature=temperature,
    )
    result = cascaded_ground(
        req,
        image_size=image.size,
        chain=list(chain) if chain else ["tool_loop"],
    )
    predicted = (result.answer or "").strip()
    gt = (sample.answer or "").strip() if sample.answer else None
    correct: bool | None = None
    if gt and predicted:
        correct = predicted.lower() == gt.lower()

    return {
        "schema": result.schema,
        "answer": predicted or None,
        "ground_truth": gt,
        "correct": correct,
        "tool_trace": result.tool_trace,
        "rounds": result.rounds,
        "model": result.model,
        "warnings": result.warnings,
        "validation": result.validation.as_dict() if result.validation else None,
        "head_used": result.head_used,
        "escalation_trace": result.escalation_trace,
        "sample": sample.to_dict(),
    }


def parse_tir_bench_batch(
    samples: Iterable[TIRBenchSample],
    *,
    output_jsonl: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 20,
    max_rounds: int = 4,
    temperature: float | None = None,
    progress: bool = True,
    chain: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Batch driver — append each result to optional JSONL."""
    results: list[dict[str, Any]] = []
    fh = None
    if output_jsonl is not None:
        outp = Path(output_jsonl)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(outp, "a", encoding="utf-8")

    try:
        for i, sample in enumerate(samples, 1):
            try:
                out = parse_tir_bench_sample(
                    sample,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    max_entities=max_entities,
                    max_rounds=max_rounds,
                    temperature=temperature,
                    chain=chain,
                )
            except Exception as exc:
                logger.warning("TIR-Bench sample %s failed: %s", sample.sample_id, exc)
                out = {"error": str(exc), "sample": sample.to_dict()}
            results.append(out)
            if fh is not None:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if progress:
                tag = (
                    "OK" if out.get("correct") is True
                    else "NO" if out.get("correct") is False
                    else "??"
                )
                logger.info(
                    "[TIR-Bench %s] %d: id=%s pred=%r gt=%r",
                    tag, i, sample.sample_id, out.get("answer"), out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()
    return results


__all__ = [
    "TIRBenchSample",
    "default_tir_bench_root",
    "iter_tir_bench_samples",
    "load_tir_bench_image",
    "parse_tir_bench_sample",
    "parse_tir_bench_batch",
]
