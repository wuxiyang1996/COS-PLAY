"""
Data schemas for Skill Evaluation.

Provides holistic quality assessment of extracted skills across six dimensions
that go beyond contract-level pass rates:

  1. **Coherence** — semantic consistency of state transitions within a skill.
  2. **Discriminability** — how well a skill can be distinguished from others.
  3. **Composability** — whether a skill chains naturally with neighbours.
  4. **Generalization** — consistency across different trajectories / contexts.
  5. **Utility** — contribution to downstream task completion.
  6. **Granularity** — whether the skill is at the right level of abstraction.

Core types:
  - ``DimensionScore``: score + evidence for one quality dimension.
  - ``SkillQualityReport``: per-skill report aggregating all dimensions.
  - ``EvaluationSummary``: bank-wide evaluation summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QualityDimension(str, Enum):
    COHERENCE = "coherence"
    DISCRIMINABILITY = "discriminability"
    COMPOSABILITY = "composability"
    GENERALIZATION = "generalization"
    UTILITY = "utility"
    GRANULARITY = "granularity"


class QualityGrade(str, Enum):
    """Ordinal grade derived from a numeric score."""
    EXCELLENT = "excellent"    # >= 0.8
    GOOD = "good"              # >= 0.6
    FAIR = "fair"              # >= 0.4
    POOR = "poor"              # >= 0.2
    FAILING = "failing"        # < 0.2

    @classmethod
    def from_score(cls, score: float) -> QualityGrade:
        if score >= 0.8:
            return cls.EXCELLENT
        if score >= 0.6:
            return cls.GOOD
        if score >= 0.4:
            return cls.FAIR
        if score >= 0.2:
            return cls.POOR
        return cls.FAILING


@dataclass
class DimensionScore:
    """Score and supporting evidence for one quality dimension."""

    dimension: QualityDimension
    score: float = 0.0  # normalised to [0, 1]
    grade: QualityGrade = QualityGrade.FAILING
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.grade = QualityGrade.from_score(self.score)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 4),
            "grade": self.grade.value,
            "details": self.details,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DimensionScore:
        return cls(
            dimension=QualityDimension(d["dimension"]),
            score=d["score"],
            details=d.get("details", {}),
            evidence=d.get("evidence", []),
        )


@dataclass
class SkillQualityReport:
    """Holistic quality report for one skill across all evaluation dimensions."""

    skill_id: str
    version: int = 0

    dimensions: Dict[str, DimensionScore] = field(default_factory=dict)

    overall_score: float = 0.0
    overall_grade: QualityGrade = QualityGrade.FAILING

    # Actionable flags for downstream consumers
    recommend_split: bool = False
    recommend_merge_with: List[str] = field(default_factory=list)
    recommend_discard: bool = False
    recommend_refine: bool = False

    warnings: List[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)

    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> None:
        """Compute weighted overall score from dimension scores."""
        if not self.dimensions:
            return
        default_weights = {d.value: 1.0 for d in QualityDimension}
        w = weights or default_weights
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_name, ds in self.dimensions.items():
            wt = w.get(dim_name, 1.0)
            weighted_sum += ds.score * wt
            total_weight += wt
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.overall_grade = QualityGrade.from_score(self.overall_score)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "overall_score": round(self.overall_score, 4),
            "overall_grade": self.overall_grade.value,
            "recommend_split": self.recommend_split,
            "recommend_merge_with": self.recommend_merge_with,
            "recommend_discard": self.recommend_discard,
            "recommend_refine": self.recommend_refine,
            "warnings": self.warnings,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillQualityReport:
        report = cls(
            skill_id=d["skill_id"],
            version=d.get("version", 0),
            overall_score=d.get("overall_score", 0.0),
            recommend_split=d.get("recommend_split", False),
            recommend_merge_with=d.get("recommend_merge_with", []),
            recommend_discard=d.get("recommend_discard", False),
            recommend_refine=d.get("recommend_refine", False),
            warnings=d.get("warnings", []),
            evaluated_at=d.get("evaluated_at", 0.0),
        )
        for k, v in d.get("dimensions", {}).items():
            report.dimensions[k] = DimensionScore.from_dict(v)
        report.overall_grade = QualityGrade.from_score(report.overall_score)
        return report

    def format_for_llm(self) -> str:
        """Render as a compact text block for LLM context injection."""
        lines = [f"=== Skill Quality: {self.skill_id} v{self.version} ==="]
        lines.append(f"  Overall: {self.overall_score:.2f} ({self.overall_grade.value})")
        for dim_name, ds in self.dimensions.items():
            lines.append(f"  {dim_name}: {ds.score:.2f} ({ds.grade.value})")
            for ev in ds.evidence[:3]:
                lines.append(f"    - {ev}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        actions = []
        if self.recommend_split:
            actions.append("SPLIT")
        if self.recommend_merge_with:
            actions.append(f"MERGE_WITH({','.join(self.recommend_merge_with)})")
        if self.recommend_discard:
            actions.append("DISCARD")
        if self.recommend_refine:
            actions.append("REFINE")
        if actions:
            lines.append(f"  Recommended actions: {', '.join(actions)}")
        return "\n".join(lines)


@dataclass
class EvaluationSummary:
    """Bank-wide evaluation summary across all skills."""

    skill_reports: Dict[str, SkillQualityReport] = field(default_factory=dict)

    mean_overall: float = 0.0
    n_excellent: int = 0
    n_good: int = 0
    n_fair: int = 0
    n_poor: int = 0
    n_failing: int = 0

    discard_candidates: List[str] = field(default_factory=list)
    split_candidates: List[str] = field(default_factory=list)
    merge_candidates: List[List[str]] = field(default_factory=list)
    refine_candidates: List[str] = field(default_factory=list)

    evaluated_at: float = field(default_factory=time.time)

    def compute_summary(self) -> None:
        """Aggregate per-skill reports into bank-level statistics."""
        if not self.skill_reports:
            return
        scores = [r.overall_score for r in self.skill_reports.values()]
        self.mean_overall = sum(scores) / len(scores)
        self.n_excellent = sum(1 for s in scores if s >= 0.8)
        self.n_good = sum(1 for s in scores if 0.6 <= s < 0.8)
        self.n_fair = sum(1 for s in scores if 0.4 <= s < 0.6)
        self.n_poor = sum(1 for s in scores if 0.2 <= s < 0.4)
        self.n_failing = sum(1 for s in scores if s < 0.2)

        self.discard_candidates = [
            sid for sid, r in self.skill_reports.items() if r.recommend_discard
        ]
        self.split_candidates = [
            sid for sid, r in self.skill_reports.items() if r.recommend_split
        ]
        self.refine_candidates = [
            sid for sid, r in self.skill_reports.items() if r.recommend_refine
        ]
        seen_merge = set()
        self.merge_candidates = []
        for sid, r in self.skill_reports.items():
            if r.recommend_merge_with:
                group = tuple(sorted([sid] + r.recommend_merge_with))
                if group not in seen_merge:
                    seen_merge.add(group)
                    self.merge_candidates.append(list(group))

    def to_dict(self) -> dict:
        return {
            "mean_overall": round(self.mean_overall, 4),
            "grade_distribution": {
                "excellent": self.n_excellent,
                "good": self.n_good,
                "fair": self.n_fair,
                "poor": self.n_poor,
                "failing": self.n_failing,
            },
            "discard_candidates": self.discard_candidates,
            "split_candidates": self.split_candidates,
            "merge_candidates": self.merge_candidates,
            "refine_candidates": self.refine_candidates,
            "n_skills": len(self.skill_reports),
            "per_skill": {
                sid: r.to_dict() for sid, r in self.skill_reports.items()
            },
            "evaluated_at": self.evaluated_at,
        }

    def format_for_llm(self) -> str:
        """Render bank-wide summary for LLM context injection."""
        lines = ["=== Skill Bank Quality Evaluation ==="]
        lines.append(f"  Skills evaluated: {len(self.skill_reports)}")
        lines.append(f"  Mean quality: {self.mean_overall:.2f}")
        lines.append(
            f"  Distribution: {self.n_excellent} excellent, {self.n_good} good, "
            f"{self.n_fair} fair, {self.n_poor} poor, {self.n_failing} failing"
        )
        if self.discard_candidates:
            lines.append(f"  Discard candidates: {', '.join(self.discard_candidates)}")
        if self.split_candidates:
            lines.append(f"  Split candidates: {', '.join(self.split_candidates)}")
        if self.merge_candidates:
            for group in self.merge_candidates:
                lines.append(f"  Merge group: {', '.join(group)}")
        if self.refine_candidates:
            lines.append(f"  Refine candidates: {', '.join(self.refine_candidates)}")
        lines.append("")
        for sid, r in sorted(self.skill_reports.items()):
            lines.append(f"  {sid}: {r.overall_score:.2f} ({r.overall_grade.value})")
        return "\n".join(lines)
