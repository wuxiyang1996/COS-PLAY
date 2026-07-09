"""CorpusSpec registry — one entry per corpus the extractor handles.

Each spec captures what the lift driver needs to know about a corpus
without hard-coding it inside the lift code itself: where the rollouts
live on disk, which lift architecture applies (sequence-segment vs
single-shot QA), and how to derive an archetype-cluster key from a
sample's metadata.

The registry is referenced from both :mod:`sequence_lift` and
:mod:`single_shot_lift` and lets :mod:`runner` pick the right driver
from a single ``--corpus <name>`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional


LiftKind = Literal["sequence", "single_shot"]
Modality = Literal["games", "desktop", "browser", "image", "video"]

CODEBASE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CorpusSpec:
    """How to locate, parse, and cluster one corpus's rollouts."""

    name: str
    lift_kind: LiftKind
    modality: Modality
    domain: str  # SkillRecord.feasible_domains[0]
    default_input_root: Path
    sample_glob: str  # e.g., "**/episode_*.json" or "**/sample_*.json"
    archetype_cluster_field: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve_input_root(self, override: Optional[Path] = None) -> Path:
        return override if override is not None else self.default_input_root


_SPECS: Dict[str, CorpusSpec] = {
    "browsergym": CorpusSpec(
        name="browsergym",
        lift_kind="sequence",
        modality="browser",
        domain="browser",
        default_input_root=CODEBASE_ROOT / "Cold-start-out-browsergym",
        sample_glob="**/episode_*.json",
    ),
    "osworld": CorpusSpec(
        name="osworld",
        lift_kind="sequence",
        modality="desktop",
        domain="desktop",
        default_input_root=CODEBASE_ROOT / "Cold-start-out-osworld" / "gpt5.4_3per_domain",
        sample_glob="**/episode_*.json",
    ),
    "visual_toolbench": CorpusSpec(
        name="visual_toolbench",
        lift_kind="single_shot",
        modality="image",
        domain="visual_reasoning",
        default_input_root=CODEBASE_ROOT / "Cold-start-out-visual-reasoning",
        sample_glob="**/visual_toolbench/**/sample_*.json",
        archetype_cluster_field="raw_sample.eval_focus",
        extra={"benchmark": "visual_toolbench"},
    ),
    "tir_bench": CorpusSpec(
        name="tir_bench",
        lift_kind="single_shot",
        modality="image",
        domain="visual_reasoning",
        default_input_root=CODEBASE_ROOT / "Cold-start-out-visual-reasoning",
        sample_glob="**/tir_bench/**/sample_*.json",
        # ``raw_sample.task`` is the TIR-Bench task family
        # (refcoco / maze / math / symbolic / color / ocr /
        # spot_difference / jigsaw / ...). Confirmed via audit-2026-05-01
        # over 200 samples: 8 distinct values, ~17 samples each.
        archetype_cluster_field="raw_sample.task",
        extra={"benchmark": "tir_bench"},
    ),
    "video_holmes": CorpusSpec(
        name="video_holmes",
        lift_kind="single_shot",
        modality="video",
        domain="video",
        default_input_root=CODEBASE_ROOT / "Cold-start-out-visual-reasoning-video",
        sample_glob="video_holmes/**/sample_*.json",
        archetype_cluster_field="raw_sample.question_type",
        extra={"benchmark": "video_holmes"},
    ),
    "siv_bench": CorpusSpec(
        name="siv_bench",
        lift_kind="single_shot",
        modality="video",
        domain="video",
        default_input_root=CODEBASE_ROOT / "Cold-start-out-visual-reasoning-video",
        sample_glob="siv_bench/**/sample_*.json",
        archetype_cluster_field="raw_sample.dimension",
        extra={"benchmark": "siv_bench"},
    ),
}


def get_spec(name: str) -> CorpusSpec:
    if name not in _SPECS:
        raise KeyError(
            f"unknown corpus {name!r} — registered: {sorted(_SPECS.keys())}"
        )
    return _SPECS[name]


def all_specs() -> List[CorpusSpec]:
    return list(_SPECS.values())


def all_names() -> List[str]:
    return sorted(_SPECS.keys())


__all__ = ["CorpusSpec", "LiftKind", "Modality", "get_spec", "all_specs", "all_names"]
