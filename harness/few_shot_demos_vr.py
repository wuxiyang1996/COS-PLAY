"""Build `FewShotDemo`s from cold-start visual-reasoning samples.

Phase-5 Stage 1 (image-VR). Mirrors ``harness/few_shot_demos_gymv.py``
shape but reads ``Cold-start-out-visual-reasoning/{visual_toolbench,
tir_bench}/sample_*.json`` instead of episode files.

Note: cold-start ``<state>`` blocks are tagged ``domain=browser`` (VTB)
or ``domain=image_qa`` (TIR), neither of which matches
``VisualReasoningAdapter.name="visual_reasoning"``. We re-tag
``state.domain`` after parsing so ``can_handle`` accepts the demo.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.few_shot_adapter import FewShotDemo
from labeling_supplement._harness_io_helpers import parse_schema_canonical

logger = logging.getLogger("harness.few_shot_demos_vr")

__all__ = [
    "build_demos_from_vr_samples",
    "build_demos_from_vr_corpus",
    "build_demo_from_vr_sample",
]

_KNOWN_SUB_CORPORA = ("visual_toolbench", "tir_bench")


def build_demos_from_vr_samples(
    cold_start_root: Path,
    *,
    sub_corpus: str,
    max_demos: int = 8,
    skip_correct_only: bool = False,
    domain_tag: str = "visual_reasoning",
) -> List[FewShotDemo]:
    """Walk ``<cold_start_root>/<sub_corpus>/sample_*.json`` and harvest
    up to ``max_demos`` ``FewShotDemo``s in sorted-filename order.

    ``skip_correct_only=True`` keeps only ``sample.correct=True`` cases
    (useful when the demo's ``state`` block must be trustworthy).
    ``domain_tag`` is what we re-tag ``state.domain`` to after parsing.
    """
    src = cold_start_root / sub_corpus
    if not src.exists():
        logger.warning("VR demo source missing: %s", src)
        return []
    if sub_corpus not in _KNOWN_SUB_CORPORA:
        logger.warning("VR sub_corpus %r not in %s; loading anyway",
                       sub_corpus, _KNOWN_SUB_CORPORA)

    out: List[FewShotDemo] = []
    for sample_path in sorted(src.glob("sample_*.json")):
        if len(out) >= max_demos:
            break
        demo = build_demo_from_vr_sample(
            sample_path, domain_tag=domain_tag,
            skip_correct_only=skip_correct_only,
        )
        if demo is not None:
            out.append(demo)

    logger.info("built %d VR demo(s) from %s/%s (cap=%d)",
                len(out), cold_start_root, sub_corpus, max_demos)
    return out


def build_demos_from_vr_corpus(
    cold_start_root: Path,
    *,
    max_demos_per_subcorpus: int = 8,
    skip_correct_only: bool = False,
    domain_tag: str = "visual_reasoning",
) -> List[FewShotDemo]:
    """Concatenate demos from BOTH ``visual_toolbench`` and ``tir_bench``."""
    demos: List[FewShotDemo] = []
    for sub in _KNOWN_SUB_CORPORA:
        demos.extend(build_demos_from_vr_samples(
            cold_start_root, sub_corpus=sub,
            max_demos=max_demos_per_subcorpus,
            skip_correct_only=skip_correct_only,
            domain_tag=domain_tag,
        ))
    return demos


def build_demo_from_vr_sample(
    sample_path: Path,
    *,
    domain_tag: str = "visual_reasoning",
    skip_correct_only: bool = False,
) -> Optional[FewShotDemo]:
    """Load one ``sample_NNN.json`` and emit a ``FewShotDemo`` (or None
    when the sample is unusable — missing schema / gold / not correct
    when ``skip_correct_only``).
    """
    try:
        sample = json.loads(sample_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read %s: %r", sample_path, exc)
        return None

    schema_text = sample.get("schema") or ""
    gold = sample.get("gold_answer")
    if not schema_text or "<state>" not in schema_text:
        return None
    if gold is None or (isinstance(gold, str) and not gold.strip()):
        return None
    if skip_correct_only and not bool(sample.get("correct", False)):
        return None

    try:
        state = parse_schema_canonical(schema_text, default_domain=domain_tag)
    except Exception as exc:  # noqa: BLE001
        logger.debug("parse_schema_canonical failed (%r) for %s",
                     exc, sample_path.name)
        return None

    # Force re-tag: VTB schemas carry ``domain=browser`` (legal so the
    # parser kept it); TIR carries ``domain=image_qa`` (not in DOMAINS,
    # parser already fell back to default_domain). Either way, we want
    # the adapter dispatch to land on VisualReasoningAdapter.
    state.domain = domain_tag
    state.task = state.task or str(sample.get("task_id") or "")

    bindings: Dict[str, Any] = {}
    expected: Dict[str, Any] = {
        "gold_answer": gold,
        "is_mcq": bool(sample.get("is_mcq", False)),
        "valid_actions": list(sample.get("valid_actions") or []),
        "task_id": sample.get("task_id"),
        "benchmark": sample.get("benchmark"),
        "options_block": sample.get("options_block"),
    }
    notes = (
        f"vr_cold_start:{sample.get('benchmark', '?')}:"
        f"{sample.get('sample_id', sample_path.stem)}"
    )
    return FewShotDemo(
        state=state, bindings=bindings,
        expected=expected, notes=notes,
    )
