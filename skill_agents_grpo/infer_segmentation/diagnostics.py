"""
Diagnostic output from InferSegmentation.

For each decoded segment the diagnostics record:
  - top-K skill candidates with scores
  - margin (score gap between rank-1 and rank-2)
  - label entropy and compatibility margin
  - label confidence category (confident / uncertain / new)
  - boundary confidence (cut vs. no-cut score delta)
  - per-term score breakdown

Small margin → uncertain → good candidate for preference labeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Uncertain-label path thresholds
_DEFAULT_CONFIDENT_MARGIN = 2.0
_DEFAULT_UNCERTAIN_MARGIN = 1.0

# Label confidence categories
LABEL_CONFIDENT = "confident"
LABEL_UNCERTAIN = "uncertain"
LABEL_NEW = "new"


@dataclass
class SkillCandidate:
    """One candidate skill label for a segment, with full breakdown."""

    skill: str
    total_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentDiagnostic:
    """Diagnostics for a single decoded segment.

    Extended with Phase 3 diagnostics: label_entropy, compat_margin,
    boundary_confidence, and label_category (the uncertain-label path).
    """

    start: int
    end: int
    assigned_skill: str
    candidates: List[SkillCandidate] = field(default_factory=list)

    # Phase 3: boundary confidence at this segment's start boundary
    boundary_confidence: float = 0.0

    @property
    def margin(self) -> float:
        """Score gap between rank-1 and rank-2.  1e6 if < 2 candidates."""
        if len(self.candidates) < 2:
            return 1e6
        return self.candidates[0].total_score - self.candidates[1].total_score

    @property
    def label_entropy(self) -> float:
        """Shannon entropy over candidate scores (softmax-normalized).

        High entropy means the scorer is spread across many skills.
        Low entropy means one skill dominates.
        """
        if len(self.candidates) < 2:
            return 0.0
        scores = [c.total_score for c in self.candidates]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        if total <= 0:
            return 0.0
        probs = [e / total for e in exps]
        return -sum(p * math.log(p + 1e-12) for p in probs)

    @property
    def compat_margin(self) -> float:
        """Contract compatibility margin: difference between top-1 and top-2
        contract_compat scores (when available in breakdowns)."""
        if len(self.candidates) < 2:
            return float("inf")
        cc1 = self.candidates[0].breakdown.get("contract_compat", 0.0)
        cc2 = self.candidates[1].breakdown.get("contract_compat", 0.0)
        return cc1 - cc2

    @property
    def label_category(self) -> str:
        """Three-way label confidence: confident / uncertain / new.

        - **confident**: large margin, label is reliable.
        - **uncertain**: small margin; reconsider after bank updates
          or proto-skill formation.
        - **new**: assigned to __NEW__ (unknown skill).
        """
        if self.assigned_skill in ("__NEW__", "NEW"):
            return LABEL_NEW
        if self.margin >= _DEFAULT_CONFIDENT_MARGIN:
            return LABEL_CONFIDENT
        return LABEL_UNCERTAIN

    @property
    def is_uncertain(self) -> bool:
        """Heuristic: margin below 1.0 → worth querying."""
        return self.margin < _DEFAULT_UNCERTAIN_MARGIN


@dataclass
class BoundaryDiagnostic:
    """Diagnostics for a boundary decision (cut vs. no-cut).

    Extended with ``boundary_preference`` — a learned signal for
    "cut here" vs "do not cut here" (Phase 3 boundary preference learning).
    """

    time: int
    score_with_cut: float
    score_without_cut: float
    boundary_preference: float = 0.0  # positive = prefer cut

    @property
    def confidence(self) -> float:
        return self.score_with_cut - self.score_without_cut


@dataclass
class SegmentationDiagnostics:
    """Aggregate diagnostics for an entire segmentation decode.

    Phase 3 additions: mean_margin, mean_entropy, fraction_uncertain,
    fraction_new, boundary_confidence stats.
    """

    n_segments: int = 0
    n_confident: int = 0
    n_uncertain: int = 0
    n_new: int = 0
    mean_margin: float = 0.0
    mean_entropy: float = 0.0
    mean_boundary_confidence: float = 0.0

    @classmethod
    def from_result(cls, result: "SegmentationResult") -> "SegmentationDiagnostics":
        segs = result.segments
        n = len(segs) or 1
        margins = [s.margin for s in segs if s.margin < 1e6]
        entropies = [s.label_entropy for s in segs]
        bconfs = [s.boundary_confidence for s in segs]
        categories = [s.label_category for s in segs]

        return cls(
            n_segments=len(segs),
            n_confident=categories.count(LABEL_CONFIDENT),
            n_uncertain=categories.count(LABEL_UNCERTAIN),
            n_new=categories.count(LABEL_NEW),
            mean_margin=sum(margins) / max(len(margins), 1),
            mean_entropy=sum(entropies) / n,
            mean_boundary_confidence=sum(bconfs) / n,
        )

    def to_dict(self) -> dict:
        return {
            "n_segments": self.n_segments,
            "n_confident": self.n_confident,
            "n_uncertain": self.n_uncertain,
            "n_new": self.n_new,
            "fraction_uncertain": round(self.n_uncertain / max(self.n_segments, 1), 3),
            "fraction_new": round(self.n_new / max(self.n_segments, 1), 3),
            "mean_margin": round(self.mean_margin, 4),
            "mean_entropy": round(self.mean_entropy, 4),
            "mean_boundary_confidence": round(self.mean_boundary_confidence, 4),
        }


@dataclass
class SegmentationResult:
    """Full output of InferSegmentation, including the best path and diagnostics."""

    segments: List[SegmentDiagnostic] = field(default_factory=list)
    boundaries: List[BoundaryDiagnostic] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def skill_sequence(self) -> List[str]:
        return [s.assigned_skill for s in self.segments]

    @property
    def cut_points(self) -> List[int]:
        return [s.start for s in self.segments]

    def uncertain_segments(self, margin_threshold: float = 1.0) -> List[SegmentDiagnostic]:
        """Return segments where the margin is below threshold (best for preference labeling)."""
        return [s for s in self.segments if s.margin < margin_threshold]

    def confident_segments(self) -> List[SegmentDiagnostic]:
        """Return segments with confident label assignment."""
        return [s for s in self.segments if s.label_category == LABEL_CONFIDENT]

    def uncertain_known_segments(self) -> List[SegmentDiagnostic]:
        """Return low-confidence known-skill segments for reconsideration."""
        return [s for s in self.segments if s.label_category == LABEL_UNCERTAIN]

    def new_segments(self) -> List[SegmentDiagnostic]:
        """Return segments assigned to __NEW__."""
        return [s for s in self.segments if s.label_category == LABEL_NEW]

    @property
    def diagnostics(self) -> SegmentationDiagnostics:
        """Aggregate diagnostics for logging and debugging."""
        return SegmentationDiagnostics.from_result(self)

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "diagnostics": self.diagnostics.to_dict(),
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "assigned_skill": s.assigned_skill,
                    "margin": s.margin,
                    "label_entropy": round(s.label_entropy, 4),
                    "compat_margin": round(s.compat_margin, 4) if s.compat_margin != float("inf") else None,
                    "label_category": s.label_category,
                    "boundary_confidence": round(s.boundary_confidence, 4),
                    "candidates": [
                        {"skill": c.skill, "score": c.total_score, "breakdown": c.breakdown}
                        for c in s.candidates
                    ],
                }
                for s in self.segments
            ],
            "boundaries": [
                {
                    "time": b.time,
                    "score_with_cut": b.score_with_cut,
                    "score_without_cut": b.score_without_cut,
                    "boundary_preference": b.boundary_preference,
                    "confidence": b.confidence,
                }
                for b in self.boundaries
            ],
        }
