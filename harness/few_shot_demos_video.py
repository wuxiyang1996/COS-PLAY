"""Build `FewShotDemo`s from cold-start video-VR samples.

Phase 5 / Stage 2 — Video-Holmes + SIV-Bench transfer cell.

The cold-start corpus lives at::

    Cold-start-out-visual-reasoning-video/
        video_holmes/sample_*.json
        siv_bench/sample_*.json

Each sample carries the typed schema block, the gold MCQ answer, and a
``video_meta`` dict with ``video_path`` / ``indices`` /
``sample_timestamps`` / ``num_frames`` / ``duration_s``. Stage 2's
deterministic executor doesn't actually decode the video, but a future
real executor will, so we surface ``video_meta`` through
``demo.expected`` for the binder to consume verbatim.

Public API mirrors :mod:`harness.few_shot_demos_gymv`::

    demos = build_demos_from_video_samples(
        Path("Cold-start-out-visual-reasoning-video"),
        sub_corpus="video_holmes", max_demos=4,
    )

A ``build_demos_from_video_corpus`` convenience helper loads from
both sub-corpora at once, useful for the dispatcher's
``--target=video`` aggregate path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from harness.few_shot_adapter import FewShotDemo
from labeling_supplement._harness_io_helpers import parse_schema_canonical

logger = logging.getLogger("harness.few_shot_demos_video")


_KNOWN_SUB_CORPORA: Sequence[str] = ("video_holmes", "siv_bench")


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def build_demos_from_video_sample_file(
    sample_path: Path,
    *,
    domain_tag: str = "video",
) -> Optional[FewShotDemo]:
    """Load a single ``sample_*.json`` and return one `FewShotDemo`.

    Returns ``None`` (and logs at DEBUG) when the sample is missing
    a parseable schema block or a non-empty gold answer — those rows
    are unusable for transfer scoring.
    """
    try:
        sample = json.loads(sample_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read %s: %r", sample_path, exc)
        return None

    schema_text = sample.get("schema") or ""
    if not _is_nonempty_string(schema_text):
        logger.debug("sample %s missing schema; skipping", sample_path.name)
        return None

    gold_answer = sample.get("gold_answer")
    if not _is_nonempty_string(gold_answer if isinstance(gold_answer, str) else str(gold_answer or "")):
        logger.debug(
            "sample %s missing gold_answer; skipping", sample_path.name,
        )
        return None

    try:
        state = parse_schema_canonical(schema_text, default_domain=domain_tag)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "parse_schema_canonical failed (%r) for %s",
            exc, sample_path.name,
        )
        return None

    state.domain = domain_tag
    state.task = state.task or str(sample.get("task_id") or "")

    expected: Dict[str, Any] = {
        "gold_answer": sample["gold_answer"],
        "is_mcq": bool(sample.get("is_mcq", False)),
        "valid_actions": list(sample.get("valid_actions") or []),
        "task_id": sample["task_id"],
        "benchmark": sample.get("benchmark"),
        "options_block": sample.get("options_block"),
        "video_meta": dict(sample.get("video_meta") or {}),
    }

    notes = (
        f"video_cold_start:{sample.get('benchmark')}:"
        f"{sample.get('sample_id')}"
    )

    return FewShotDemo(
        state=state,
        bindings={},
        expected=expected,
        notes=notes,
    )


def build_demos_from_video_samples(
    cold_start_root: Path,
    *,
    sub_corpus: str,
    max_demos: int = 8,
    skip_correct_only: bool = False,
    domain_tag: str = "video",
) -> List[FewShotDemo]:
    """Walk ``cold_start_root/<sub_corpus>/sample_*.json`` and harvest
    up to ``max_demos`` `FewShotDemo`s.

    Args:
      cold_start_root: Repo-relative path
        ``Cold-start-out-visual-reasoning-video`` (or absolute).
      sub_corpus: ``"video_holmes"`` or ``"siv_bench"``.
      max_demos: Cap on returned list length. Cold-start order is
        preserved; callers wanting randomness must shuffle the
        result themselves.
      skip_correct_only: When True, drop samples where the cold-start
        VLM already answered correctly (``sample.correct == True``).
        Defaults False — we want both correct and incorrect samples
        to feed the transfer probe so the score floor reflects
        adaptation difficulty, not lift quality.
      domain_tag: Domain to project the parsed `StateSchema` onto.
        Defaults to ``"video"`` — the canonical
        ``TRANSFER_TARGET_DOMAINS`` entry.
    """
    if sub_corpus not in _KNOWN_SUB_CORPORA:
        raise ValueError(
            f"sub_corpus={sub_corpus!r} not in {_KNOWN_SUB_CORPORA}"
        )

    src = cold_start_root / sub_corpus
    if not src.exists():
        logger.warning("cold-start sub-corpus missing: %s", src)
        return []

    out: List[FewShotDemo] = []
    sample_paths = sorted(src.glob("sample_*.json"))
    for sample_path in sample_paths:
        if len(out) >= max_demos:
            break
        if skip_correct_only:
            try:
                head = json.loads(sample_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if bool(head.get("correct")) or bool(head.get("correct_strmatch")):
                continue
        demo = build_demos_from_video_sample_file(
            sample_path, domain_tag=domain_tag,
        )
        if demo is None:
            continue
        out.append(demo)

    logger.info(
        "built %d video demo(s) from %s (cap=%d, scanned=%d)",
        len(out), src, max_demos, len(sample_paths),
    )
    return out


def build_demos_from_video_corpus(
    cold_start_root: Path,
    *,
    max_demos_per_sub_corpus: int = 4,
    skip_correct_only: bool = False,
    domain_tag: str = "video",
) -> List[FewShotDemo]:
    """Convenience: load demos from BOTH ``video_holmes`` and ``siv_bench``.

    Useful when the dispatcher's ``--target`` is a domain-level alias
    rather than a specific sub-corpus. Order: ``video_holmes`` first,
    then ``siv_bench`` — matches the section ordering in §11.5.
    """
    out: List[FewShotDemo] = []
    for sub_corpus in _KNOWN_SUB_CORPORA:
        out.extend(
            build_demos_from_video_samples(
                cold_start_root,
                sub_corpus=sub_corpus,
                max_demos=max_demos_per_sub_corpus,
                skip_correct_only=skip_correct_only,
                domain_tag=domain_tag,
            )
        )
    return out


__all__ = [
    "build_demos_from_video_corpus",
    "build_demos_from_video_sample_file",
    "build_demos_from_video_samples",
]
