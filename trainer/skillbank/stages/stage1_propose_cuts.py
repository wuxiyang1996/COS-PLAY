"""
Stage 1: Propose boundary candidates for skill segmentation.

Detects potential skill boundaries using predicate changes and optional
surprisal signals, then merges nearby candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from trainer.skillbank.ingest_rollouts import TrajectoryForEM

logger = logging.getLogger(__name__)


@dataclass
class CutCandidate:
    """A proposed boundary cut between two timesteps."""

    t: int
    score: float = 0.0
    source: str = ""  # "predicate_change" | "surprisal" | "action_change" | "merged"
    predicates_changed: List[str] = field(default_factory=list)


@dataclass
class ProposeCutsConfig:
    """Configuration for boundary proposal."""

    min_segment_width: int = 5
    merge_radius: int = 5
    predicate_change_weight: float = 0.5
    surprisal_weight: float = 0.5
    min_cut_score: float = 0.1


def propose_cuts(
    trajectory: TrajectoryForEM,
    config: Optional[ProposeCutsConfig] = None,
    surprisal_scores: Optional[List[float]] = None,
) -> List[CutCandidate]:
    """Propose boundary candidates for a single trajectory.

    Combines:
      - Predicate change detection (when boolean predicates flip)
      - Surprisal-based detection (embedding/observation surprisal)
      - Action type transitions (primitive ↔ retrieval)

    Returns sorted list of CutCandidates.
    """
    cfg = config or ProposeCutsConfig()
    frames = trajectory.frames
    n = len(frames)
    if n < cfg.min_segment_width * 2:
        return []

    candidates: List[CutCandidate] = []

    from trainer.skillbank.stages.stage0_predicates import booleanize
    prev_bools: Set[str] = set()
    for i, frame in enumerate(frames):
        curr_bools = booleanize(frame.predicates)
        if i > 0:
            added = curr_bools - prev_bools
            removed = prev_bools - curr_bools
            n_changed = len(added) + len(removed)
            if n_changed > 0:
                score = n_changed * cfg.predicate_change_weight
                candidates.append(CutCandidate(
                    t=i,
                    score=score,
                    source="predicate_change",
                    predicates_changed=sorted(added | removed),
                ))
        prev_bools = curr_bools

    if surprisal_scores and len(surprisal_scores) == n:
        mean_s = sum(surprisal_scores) / n if n > 0 else 0
        std_s = (sum((s - mean_s) ** 2 for s in surprisal_scores) / max(n, 1)) ** 0.5
        threshold = mean_s + std_s
        for i, s in enumerate(surprisal_scores):
            if s > threshold and i > 0:
                candidates.append(CutCandidate(
                    t=i,
                    score=s * cfg.surprisal_weight,
                    source="surprisal",
                ))

    for i in range(1, n):
        prev_type = frames[i - 1].action_type
        curr_type = frames[i].action_type
        if prev_type != curr_type and curr_type in ("CALL_SKILL",):
            candidates.append(CutCandidate(
                t=i,
                score=0.3,
                source="action_change",
            ))

    candidates = _merge_nearby(candidates, cfg.merge_radius)
    candidates = [c for c in candidates if c.score >= cfg.min_cut_score]
    candidates = _enforce_min_width(candidates, cfg.min_segment_width, n)
    candidates.sort(key=lambda c: c.t)

    logger.debug("Proposed %d cuts for traj %s (len=%d)", len(candidates), trajectory.traj_id, n)
    return candidates


def _merge_nearby(candidates: List[CutCandidate], radius: int) -> List[CutCandidate]:
    """Merge cut candidates within `radius` timesteps, keeping the highest-scoring."""
    if not candidates:
        return []

    candidates.sort(key=lambda c: c.t)
    merged: List[CutCandidate] = []

    i = 0
    while i < len(candidates):
        group = [candidates[i]]
        j = i + 1
        while j < len(candidates) and candidates[j].t - candidates[i].t <= radius:
            group.append(candidates[j])
            j += 1

        best = max(group, key=lambda c: c.score)
        best.source = "merged" if len(group) > 1 else best.source
        merged.append(best)
        i = j

    return merged


def _enforce_min_width(
    candidates: List[CutCandidate],
    min_width: int,
    traj_len: int,
) -> List[CutCandidate]:
    """Remove cuts that would create segments shorter than min_width."""
    if not candidates:
        return []

    candidates.sort(key=lambda c: c.t)
    filtered: List[CutCandidate] = []
    last_t = 0

    for c in candidates:
        if c.t - last_t >= min_width and traj_len - c.t >= min_width:
            filtered.append(c)
            last_t = c.t

    return filtered


def cuts_to_segments(
    cuts: List[CutCandidate],
    traj_len: int,
) -> List[tuple]:
    """Convert cut positions to (start, end) segment tuples."""
    boundaries = sorted(set([0] + [c.t for c in cuts] + [traj_len]))
    segments = []
    for i in range(len(boundaries) - 1):
        segments.append((boundaries[i], boundaries[i + 1]))
    return segments
