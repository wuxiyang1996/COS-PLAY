"""VisualToolBench (VTB) loader + VLM parser (image QA + tool use).

`Beyond Seeing: Evaluating Multimodal LLMs On Tool-enabled Image
Perception, Transformation, and Reasoning <https://arxiv.org/abs/2510.12712>`_.
Dataset: ``ScaleAI/VisualToolBench`` on HuggingFace (1 204 tasks).

By default we stream **single-turn** tasks only so each sample maps to
one image and one user prompt (turn 0). Official leaderboard scoring
uses per-turn **rubrics**; this repo's harness applies plain string
match on ``turn_golden_answers[0]`` as a *diagnostic* signal only.
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

_HF_ID = "ScaleAI/VisualToolBench"


def _as_str_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x) for x in j]
            except json.JSONDecodeError:
                pass
        return [s]
    return [str(val)]


def _first_image_cell(row: dict[str, Any]) -> Any:
    imgs = row.get("images")
    if isinstance(imgs, list) and imgs:
        return imgs[0]
    ibt = row.get("images_by_turn")
    if isinstance(ibt, list) and ibt and isinstance(ibt[0], list) and ibt[0]:
        return ibt[0][0]
    raise ValueError("VisualToolBench row has no images")


@dataclass
class VisualToolBenchSample:
    """One VTB task (single-turn slice when ``single_turn_only`` is used)."""

    sample_id: str
    turncase: str
    prompt_category: str | None
    question: str
    gold_answer: str | None
    image_cell: Any
    eval_focus: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "turncase": self.turncase,
            "prompt_category": self.prompt_category,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "eval_focus": self.eval_focus,
        }


def default_visual_toolbench_root(workspace_root: str | Path | None = None) -> Path:
    """Optional local export path; HF cache is the default."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "VisualToolBench"


def _load_streaming():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install HuggingFace `datasets` to use VisualToolBench."
        ) from exc
    try:
        return load_dataset(_HF_ID, split="test", streaming=True, trust_remote_code=True)
    except ValueError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        return load_dataset(_HF_ID, split="test", streaming=True)


def iter_visual_toolbench_samples(
    *,
    limit: int | None = None,
    single_turn_only: bool = True,
) -> Iterator[VisualToolBenchSample]:
    """Yield VTB rows from the HF ``test`` split (streaming iterator)."""
    ds = _load_streaming()
    n = 0
    for row in ds:
        if single_turn_only:
            n_turns = row.get("num_turns")
            if isinstance(n_turns, int) and n_turns != 1:
                continue
            if not isinstance(n_turns, int):
                tc = str(row.get("turncase") or "").lower()
                if "multi" in tc:
                    continue
        prompts = _as_str_list(row.get("turn_prompts"))
        answers = _as_str_list(row.get("turn_golden_answers"))
        if not prompts:
            continue
        q = prompts[0]
        gold = answers[0] if answers else None
        sid = str(row.get("id", n))
        try:
            cell = _first_image_cell(row)
        except ValueError:
            continue
        yield VisualToolBenchSample(
            sample_id=sid,
            turncase=str(row.get("turncase") or ""),
            prompt_category=(
                str(row["prompt_category"]) if row.get("prompt_category") else None
            ),
            question=q,
            gold_answer=gold,
            image_cell=cell,
            eval_focus=str(row["eval_focus"]) if row.get("eval_focus") else None,
        )
        n += 1
        if limit is not None and n >= limit:
            return


def load_visual_toolbench_image(sample: VisualToolBenchSample) -> Image.Image:
    return decode_hf_image(sample.image_cell)


def parse_visual_toolbench_sample(
    sample: VisualToolBenchSample,
    *,
    image: Image.Image | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    chain: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``cascaded_ground`` on one VTB task (``domain=image_qa``).

    The default ``chain=["tool_loop"]`` skips the single-shot ``vlm``
    head: VisualToolBench is explicitly a *tool-enabled* image benchmark
    (paper §3 "Beyond Seeing"), so we force the multi-hop tool loop and
    refuse to accept a schema that hasn't actually grounded with
    ``crop`` / ``detect_objects`` / ``read_text_in_region`` / etc.  Pass
    ``chain=["vlm", "tool_loop"]`` for the cheaper escalating cascade
    (e.g. for sanity runs) and ``["vlm"]`` to bypass tools entirely.
    """
    if image is None:
        image = load_visual_toolbench_image(sample)

    task_id = f"visual_toolbench.{sample.sample_id}"
    routing = classify_question(sample.question, modality="image")
    routing_block = routing.to_prompt_block()
    goal = (
        f"{sample.question}\n"
        "This is a VisualToolBench item — you MUST exercise the visual "
        "tools (detect_objects, grounded_detect, zoom_region, "
        "read_text_region, describe_region, count_objects, "
        "spatial_query, measure_distance, …) before answering. "
        "Single-shot answers without tool evidence will be rejected. "
        "Each <evidence> hop must cite the real `tool=` you called and "
        "the entity IDs the tool produced (e.g. `result_ref=e1,e2`).\n"
        "Reasoning tools available: count_value, compute_ratio, "
        "compare_values, verify_claim — use them to RECORD computations "
        "(not just describe them), and cite the resulting "
        "`derivation_id` (d1, d2, …) inside <derivations> and <answer>.\n"
        f"{routing_block}\n"
        "Put the final response in <answer> as a concise string; align "
        "wording with the gold reference when possible (official "
        "scoring uses rubrics)."
    )
    req = GroundingRequest(
        images=image,
        goal=goal,
        domain="image_qa",
        output_mode="answer",
        task_id=task_id,
        step=0,
        context={
            "benchmark": "visual_toolbench",
            "prompt_category": sample.prompt_category,
            "eval_focus": sample.eval_focus,
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
    gt = (sample.gold_answer or "").strip() if sample.gold_answer else None

    correct: bool | None = None
    if gt and predicted:
        pl, gl = predicted.lower(), gt.lower()
        correct = pl == gl or gl in pl or pl in gl

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


def parse_visual_toolbench_batch(
    samples: Iterable[VisualToolBenchSample],
    *,
    output_jsonl: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    temperature: float | None = None,
    progress: bool = True,
    chain: list[str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    fh = None
    if output_jsonl is not None:
        outp = Path(output_jsonl)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fh = open(outp, "a", encoding="utf-8")

    try:
        for i, sample in enumerate(samples, 1):
            try:
                out = parse_visual_toolbench_sample(
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
                logger.warning("VTB sample %s failed: %s", sample.sample_id, exc)
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
                    "[VTB %s] %d: id=%s pred=%r",
                    tag, i, sample.sample_id, out.get("answer"),
                )
    finally:
        if fh is not None:
            fh.close()
    return results


__all__ = [
    "VisualToolBenchSample",
    "default_visual_toolbench_root",
    "iter_visual_toolbench_samples",
    "load_visual_toolbench_image",
    "parse_visual_toolbench_sample",
    "parse_visual_toolbench_batch",
]
