"""Data loader for the ``schema_gen`` SFT pipeline.

Reads the three input flavours produced by Phase-0 collection
(PLAN-VISUAL-GROUNDING-MILESTONES §5):

1. **Gym-V triples** — output of ``labeling/grounding/collect_gymv.py``.
   Each ``triples.jsonl`` row is one ``(frame, heuristic_schema,
   vision_schema)`` triple.
2. **BrowserGym triples** — output of
   ``labeling/grounding/collect_browser.py`` (same shape).
3. **Image-QA / Video-QA labels** — JSONL emitted by
   ``visual_reasoning_wrapper.benchmarks.<bench>.parse_*_batch`` (one row per QA
   sample, with ``schema`` populated by the SFT vision teacher (``gpt-5.5``).

Every loader yields a ``SchemaGenSample`` dataclass with a uniform
``(image, prompt, target_schema, domain, source)`` shape, plus the
``messages`` list ready for the Qwen3-VL processor's ``apply_chat_template``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class SchemaGenSample:
    """One labelled (input, schema) pair for SFT.

    ``images`` is a list because video-QA samples carry ``num_frames``
    images.  ``prompt`` is the user-side text portion (goal + extra
    context); ``target_schema`` is the assistant-side gold completion
    that the model is asked to produce verbatim.
    """

    sample_id: str
    domain: str           # gymv / browser / image_qa / video_qa
    source: str           # heuristic | vision (which schema is the target)
    images: list[str]     # absolute file paths
    prompt: str
    target_schema: str
    extra_context: dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, Any]]:
        """Return the multi-turn ``messages`` list for ``apply_chat_template``.

        The system prompt is filled in by ``train.py`` (which has access
        to the per-domain ``build_adaptive_system_prompt``) so this
        helper only emits the user + assistant turns.
        """
        content: list[dict[str, Any]] = []
        for path in self.images:
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": self.prompt})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": self.target_schema},
        ]


# ======================================================================
# Hard-case filter (from cross_validate.py)
# ======================================================================

def _load_hard_case_set(
    hard_cases_jsonl: Optional[Path],
) -> set[str]:
    """Build a set of ``frame_path`` strings flagged as hard cases."""
    if hard_cases_jsonl is None or not hard_cases_jsonl.exists():
        return set()
    out: set[str] = set()
    with hard_cases_jsonl.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            fp = row.get("frame_path")
            if fp:
                out.add(str(fp))
    return out


# ======================================================================
# Phase-0 triples loaders
# ======================================================================

def _select_target(
    triple: dict[str, Any], target_source: str,
) -> tuple[str, str] | None:
    """Return ``(target_schema, source_label)`` or ``None`` if invalid.

    ``target_source`` decides the priority:
      * ``"vision"``    → prefer the gpt-5.5 vision schema; skip if missing
      * ``"heuristic"`` → always use the deterministic heuristic schema
      * ``"auto"``      → prefer vision; fall back to heuristic
    """
    vision = triple.get("vision_schema")
    heuristic = triple.get("heuristic_schema")
    if target_source == "vision":
        return (vision, "vision") if vision else None
    if target_source == "heuristic":
        return (heuristic, "heuristic") if heuristic else None
    # auto
    if vision:
        return vision, "vision"
    if heuristic:
        return heuristic, "heuristic"
    return None


def iter_gymv_triples(
    triple_root: str | Path,
    *,
    target_source: str = "vision",
    drop_hard_cases: bool = True,
    hard_cases_jsonl: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> Iterator[SchemaGenSample]:
    """Yield ``SchemaGenSample`` rows from Gym-V triples on disk."""
    root = Path(triple_root)
    if not root.exists():
        logger.warning("Gym-V triple root %s does not exist; skipping.", root)
        return

    hard = _load_hard_case_set(
        Path(hard_cases_jsonl) if hard_cases_jsonl
        else root / "hard_cases.jsonl",
    ) if drop_hard_cases else set()

    count = 0
    for triples_path in root.rglob("triples.jsonl"):
        with triples_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    triple = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if triple.get("error"):
                    continue
                frame_path = triple.get("frame_path")
                if not frame_path:
                    continue
                if frame_path in hard:
                    continue
                target = _select_target(triple, target_source)
                if target is None:
                    continue
                target_schema, source_label = target

                env_id = triple.get("env_id", "")
                step = triple.get("step", 0)
                episode = triple.get("episode", 0)
                description = triple.get("description", "")
                obs_text = triple.get("obs_text", "")
                valid_actions = triple.get("valid_actions") or []

                prompt_lines = [
                    f"Domain: gymv ({env_id})",
                    f"Step: {step}",
                ]
                if description:
                    prompt_lines.append(f"Game rules:\n{description}")
                if obs_text:
                    prompt_lines.append(
                        f"Environment text state (for reference):\n{obs_text}"
                    )
                if valid_actions:
                    prompt_lines.append(
                        "Valid actions (copy verbatim into <actions>): "
                        + ", ".join(map(str, valid_actions))
                    )
                prompt = "\n\n".join(prompt_lines)

                yield SchemaGenSample(
                    sample_id=f"gymv.{env_id}.ep{episode}.s{step}",
                    domain="gymv",
                    source=source_label,
                    images=[frame_path],
                    prompt=prompt,
                    target_schema=target_schema,
                    extra_context={
                        "env_id": env_id,
                        "valid_actions": valid_actions,
                    },
                )
                count += 1
                if limit is not None and count >= limit:
                    return


def iter_env_wrappers_triples(
    triple_root: str | Path,
    *,
    target_source: str = "vision",
    drop_hard_cases: bool = True,
    hard_cases_jsonl: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> Iterator[SchemaGenSample]:
    """Yield ``SchemaGenSample`` rows from env_wrappers triples on disk.

    The on-disk shape is identical to gymv triples (one row per
    ``(frame_path, vision_schema, env_id, episode, step, …)``); we
    just label the domain ``env_wrappers`` and adapt the prompt header
    to use ``game`` instead of ``env_id``.
    """
    root = Path(triple_root)
    if not root.exists():
        logger.warning(
            "env_wrappers triple root %s does not exist; skipping.", root,
        )
        return

    hard = _load_hard_case_set(
        Path(hard_cases_jsonl) if hard_cases_jsonl
        else root / "hard_cases.jsonl",
    ) if drop_hard_cases else set()

    count = 0
    for triples_path in root.rglob("triples.jsonl"):
        with triples_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    triple = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if triple.get("error"):
                    continue
                frame_path = triple.get("frame_path")
                if not frame_path:
                    continue
                if frame_path in hard:
                    continue
                target = _select_target(triple, target_source)
                if target is None:
                    continue
                target_schema, source_label = target

                # ``env_id`` is repurposed to carry the game name in
                # the env_wrappers triples.
                game = triple.get("env_id") or triple.get("game") or ""
                step = triple.get("step", 0)
                episode = triple.get("episode", 0)
                description = triple.get("description", "")
                obs_text = triple.get("obs_text", "")
                valid_actions = triple.get("valid_actions") or []

                prompt_lines = [
                    f"Domain: env_wrappers ({game})",
                    f"Step: {step}",
                ]
                if description:
                    prompt_lines.append(f"Game rules:\n{description}")
                if obs_text:
                    prompt_lines.append(
                        f"Environment text state (for reference):\n{obs_text}"
                    )
                if valid_actions:
                    prompt_lines.append(
                        "Valid actions (copy verbatim into <actions>): "
                        + ", ".join(map(str, valid_actions))
                    )
                prompt = "\n\n".join(prompt_lines)

                yield SchemaGenSample(
                    sample_id=f"env_wrappers.{game}.ep{episode}.s{step}",
                    domain="env_wrappers",
                    source=source_label,
                    images=[frame_path],
                    prompt=prompt,
                    target_schema=target_schema,
                    extra_context={
                        "game": game,
                        "valid_actions": valid_actions,
                    },
                )
                count += 1
                if limit is not None and count >= limit:
                    return


def iter_browser_triples(
    triple_root: str | Path,
    *,
    target_source: str = "vision",
    drop_hard_cases: bool = True,
    hard_cases_jsonl: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> Iterator[SchemaGenSample]:
    """Yield ``SchemaGenSample`` rows from BrowserGym triples on disk."""
    root = Path(triple_root)
    if not root.exists():
        logger.warning(
            "Browser triple root %s does not exist; skipping.", root,
        )
        return

    hard = _load_hard_case_set(
        Path(hard_cases_jsonl) if hard_cases_jsonl
        else root / "hard_cases.jsonl",
    ) if drop_hard_cases else set()

    count = 0
    for triples_path in root.rglob("triples.jsonl"):
        with triples_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    triple = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if triple.get("error"):
                    continue
                frame_path = triple.get("frame_path")
                if not frame_path:
                    continue
                if frame_path in hard:
                    continue
                target = _select_target(triple, target_source)
                if target is None:
                    continue
                target_schema, source_label = target

                task_id = triple.get("task_id", "")
                step = triple.get("step", 0)
                episode = triple.get("episode", 0)
                url = triple.get("url", "")
                goal = triple.get("goal", "")
                axtree = triple.get("axtree_text", "")[:2500]

                prompt_lines = [
                    f"Domain: browser ({task_id})",
                    f"URL: {url}",
                    f"Step: {step}",
                    f"Goal: {goal}",
                ]
                if axtree:
                    prompt_lines.append(f"AXTree (truncated):\n{axtree}")
                prompt = "\n\n".join(prompt_lines)

                yield SchemaGenSample(
                    sample_id=f"browser.{task_id}.ep{episode}.s{step}",
                    domain="browser",
                    source=source_label,
                    images=[frame_path],
                    prompt=prompt,
                    target_schema=target_schema,
                    extra_context={"task_id": task_id, "url": url, "goal": goal},
                )
                count += 1
                if limit is not None and count >= limit:
                    return


# ======================================================================
# Image-QA / Video-QA loaders (from benchmark parse_*_batch JSONL)
# ======================================================================

def _iter_qa_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        logger.warning("QA labels JSONL %s does not exist; skipping.", path)
        return
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def iter_image_qa_triples(
    labels_jsonl: str | Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[SchemaGenSample]:
    """Yield ``SchemaGenSample`` rows from image-QA batch JSONL (e.g. TIR-Bench / VTB parsers).

    Required keys per row: ``schema`` (gold), ``sample.image_path``,
    and either ``sample.question`` or ``sample.goal``.
    """
    count = 0
    for row in _iter_qa_jsonl(Path(labels_jsonl)):
        if row.get("error"):
            continue
        schema = row.get("schema")
        sample = row.get("sample") or {}
        image_path = sample.get("image_path")
        question = sample.get("question") or sample.get("goal") or ""
        if not (schema and image_path and question):
            continue
        sid = (
            sample.get("question_id")
            or sample.get("image_filename")
            or sample.get("image_index")
            or f"image_qa_{count}"
        )
        prompt = (
            f"Domain: image_qa\n"
            f"Question: {question}"
        )
        yield SchemaGenSample(
            sample_id=f"image_qa.{sid}",
            domain="image_qa",
            source="vision",
            images=[image_path],
            prompt=prompt,
            target_schema=schema,
            extra_context={
                "question": question,
                "answer": sample.get("answer"),
            },
        )
        count += 1
        if limit is not None and count >= limit:
            return


def iter_video_qa_triples(
    labels_jsonl: str | Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[SchemaGenSample]:
    """Yield ``SchemaGenSample`` rows from a parse_video_holmes/siv_bench JSONL.

    Each row's ``sample.video_path`` resolves to a clip on disk; the
    trainer is responsible for sampling the configured number of frames
    at training time (so we don't have to materialise frames here).
    """
    count = 0
    for row in _iter_qa_jsonl(Path(labels_jsonl)):
        if row.get("error"):
            continue
        schema = row.get("schema")
        sample = row.get("sample") or {}
        video_path = sample.get("video_path")
        question = sample.get("question") or ""
        if not (schema and video_path and question):
            continue
        sid = (
            f"{sample.get('video_id', 'vid')}.Q"
            f"{sample.get('question_id', count)}"
        )
        options = sample.get("options") or {}
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        prompt_parts = [
            "Domain: video_qa",
            f"Question: {question}",
        ]
        if opts_text:
            prompt_parts.append(f"Options:\n{opts_text}")
        prompt = "\n\n".join(prompt_parts)
        yield SchemaGenSample(
            sample_id=f"video_qa.{sid}",
            domain="video_qa",
            source="vision",
            images=[video_path],  # trainer reads frames at train time
            prompt=prompt,
            target_schema=schema,
            extra_context={
                "question": question,
                "options": options,
                "answer": sample.get("answer"),
                "is_video": True,
            },
        )
        count += 1
        if limit is not None and count >= limit:
            return


# ======================================================================
# Top-level dataset assembly
# ======================================================================

def load_schema_gen_dataset(
    cfg,  # type: SchemaGenConfig — avoid circular import at type level
    *,
    domains: Optional[Iterable[str]] = None,
) -> list[SchemaGenSample]:
    """Build the full Phase-1 SFT dataset as a list of ``SchemaGenSample``.

    ``cfg`` is a ``SchemaGenConfig``; ``domains`` overrides
    ``cfg.domains`` (useful for quick smoke tests / ablations).

    The result is materialised into a list because the
    Qwen3-VL processor needs random-access for its multi-image batching;
    callers that need streaming should iterate the per-domain
    generators directly.
    """
    domains = list(domains) if domains else cfg.domains
    out: list[SchemaGenSample] = []
    cap = cfg.max_samples_per_domain

    if "gymv" in domains:
        out.extend(list(iter_gymv_triples(
            cfg.gymv_triple_root,
            target_source=cfg.target_source,
            drop_hard_cases=cfg.drop_hard_cases,
            hard_cases_jsonl=cfg.hard_cases_jsonl,
            limit=cap,
        )))
    if "env_wrappers" in domains:
        out.extend(list(iter_env_wrappers_triples(
            cfg.env_wrappers_triple_root,
            target_source=cfg.target_source,
            drop_hard_cases=cfg.drop_hard_cases,
            hard_cases_jsonl=cfg.hard_cases_jsonl,
            limit=cap,
        )))
    if "browser" in domains:
        out.extend(list(iter_browser_triples(
            cfg.browser_triple_root,
            target_source=cfg.target_source,
            drop_hard_cases=cfg.drop_hard_cases,
            hard_cases_jsonl=cfg.hard_cases_jsonl,
            limit=cap,
        )))
    if "image_qa" in domains:
        out.extend(list(iter_image_qa_triples(
            cfg.image_qa_jsonl, limit=cap,
        )))
    if "video_qa" in domains:
        out.extend(list(iter_video_qa_triples(
            cfg.video_qa_jsonl, limit=cap,
        )))

    logger.info(
        "Loaded %d schema_gen samples from domains=%s "
        "(target_source=%s, drop_hard_cases=%s)",
        len(out), domains, cfg.target_source, cfg.drop_hard_cases,
    )
    return out


__all__ = [
    "SchemaGenSample",
    "iter_gymv_triples",
    "iter_env_wrappers_triples",
    "iter_browser_triples",
    "iter_image_qa_triples",
    "iter_video_qa_triples",
    "load_schema_gen_dataset",
]
