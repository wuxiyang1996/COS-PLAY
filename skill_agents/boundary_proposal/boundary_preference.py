"""
Boundary plausibility scoring: lightweight quality evaluation for candidate
cut points.

Many segmentation failures come from wrong boundaries, not wrong labels.
This module provides a ``BoundaryPreferenceScorer`` that evaluates whether
a boundary is plausible based on:

  1. **Signal strength** — how many independent signals support this boundary
     (predicate flips, surprisal spikes, change-points, hard events).
  2. **Predicate discontinuity** — how much the predicate state changes across
     the boundary (larger change = more likely a real boundary).
  3. **Effect contrast** — whether the effects on either side of the boundary
     look different (computed from the segment scorer when available).
  4. **Learned preference** (optional) — pairwise preference scores from LLM
     or human feedback about boundary quality.

The scorer can be plugged in at two points:
  - **Stage 1 candidate filtering**: as a post-filter after
    ``propose_boundary_candidates()`` to prune implausible candidates.
  - **Stage 2 decoding**: as an additional boundary quality term in the
    segment score (additive bonus/penalty for choosing a cut at time t).

Usage::

    from skill_agents.boundary_proposal.boundary_preference import (
        BoundaryPreferenceScorer, BoundaryPreferenceConfig,
    )

    bp = BoundaryPreferenceScorer(config=BoundaryPreferenceConfig())
    bp.set_candidates(candidates)  # from Stage 1
    bp.set_predicates(predicates)  # per-timestep predicate dicts

    # Use as Stage 1 filter
    filtered = bp.filter_candidates(candidates, top_frac=0.7)

    # Use as Stage 2 boundary bonus
    score = bp.boundary_score(t=42)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class BoundaryPreferenceConfig:
    """Configuration for boundary plausibility scoring."""

    enabled: bool = False

    # Weight for each sub-signal in the composite plausibility score
    w_signal_strength: float = 1.0
    w_predicate_discontinuity: float = 1.0
    w_effect_contrast: float = 0.5

    # Minimum plausibility to keep a candidate (for filtering)
    min_plausibility: float = 0.1

    # Predicate discontinuity: window size for computing change magnitude
    pred_window: int = 3

    # Bonus for hard-event sources ("event", "predicate")
    hard_source_bonus: float = 0.5

    # Whether to use this as a Stage 2 term (boundary quality bias)
    use_in_decoding: bool = True
    decoding_weight: float = 0.2  # additive weight when used in Stage 2


@dataclass
class BoundaryScore:
    """Detailed plausibility breakdown for one candidate."""

    time: int
    signal_strength: float = 0.0
    predicate_discontinuity: float = 0.0
    effect_contrast: float = 0.0
    learned_preference: float = 0.0
    total: float = 0.0
    sources: List[str] = field(default_factory=list)


class BoundaryPreferenceScorer:
    """Scores boundary plausibility for candidate cut points.

    Combines rule-based features (signal support, predicate discontinuity)
    with optional learned preferences.  Designed to be lightweight and
    compatible with the existing pipeline.
    """

    def __init__(self, config: Optional[BoundaryPreferenceConfig] = None) -> None:
        self.config = config or BoundaryPreferenceConfig()
        self._candidate_info: Dict[int, Dict] = {}
        self._predicates: Optional[List[Optional[dict]]] = None
        self._scores_cache: Dict[int, BoundaryScore] = {}

        # Learned pairwise preferences: (t_win, t_lose) -> count
        self._pairwise_prefs: Dict[Tuple[int, int], int] = {}
        self._preference_scores: Dict[int, float] = {}

    # ── Setup ────────────────────────────────────────────────────────

    def set_candidates(self, candidates) -> None:
        """Register Stage 1 candidates with their source metadata.

        Parameters
        ----------
        candidates : list[BoundaryCandidate]
            From ``propose_boundary_candidates()``.
        """
        self._candidate_info.clear()
        self._scores_cache.clear()
        for c in candidates:
            sources = c.source.split("+") if hasattr(c, "source") else ["unknown"]
            self._candidate_info[c.center] = {
                "sources": sources,
                "half_window": getattr(c, "half_window", 0),
            }

    def set_predicates(self, predicates: List[Optional[dict]]) -> None:
        """Register per-timestep predicate dicts for discontinuity computation."""
        self._predicates = predicates
        self._scores_cache.clear()

    # ── Pairwise preference learning ─────────────────────────────────

    def add_preference(self, t_win: int, t_lose: int) -> None:
        """Record a pairwise preference: boundary at ``t_win`` is better than ``t_lose``."""
        key = (t_win, t_lose)
        self._pairwise_prefs[key] = self._pairwise_prefs.get(key, 0) + 1
        self._preference_scores.clear()  # invalidate

    def add_preference_batch(self, prefs: List[Tuple[int, int]]) -> None:
        for t_w, t_l in prefs:
            self.add_preference(t_w, t_l)

    def _compute_preference_scores(self) -> None:
        """Simple Bradley-Terry-style scoring from pairwise prefs."""
        if self._preference_scores:
            return

        all_times: Set[int] = set()
        for (tw, tl) in self._pairwise_prefs:
            all_times.add(tw)
            all_times.add(tl)

        scores = {t: 0.0 for t in all_times}
        lr = 0.1

        for _ in range(20):  # lightweight training
            for (tw, tl), count in self._pairwise_prefs.items():
                diff = scores[tw] - scores[tl]
                prob = 1.0 / (1.0 + math.exp(-diff))
                grad = (1.0 - prob) * count
                scores[tw] += lr * grad
                scores[tl] -= lr * grad

        self._preference_scores = scores

    # ── Scoring ──────────────────────────────────────────────────────

    def _signal_strength(self, t: int) -> Tuple[float, List[str]]:
        """Score based on how many independent signals support this boundary."""
        info = self._candidate_info.get(t)
        if info is None:
            return 0.0, []

        sources = info["sources"]
        n_sources = len(set(sources))

        hard_bonus = 0.0
        for s in sources:
            if s in ("event", "predicate"):
                hard_bonus += self.config.hard_source_bonus

        score = min(1.0, n_sources / 3.0) + hard_bonus
        return score, sources

    def _predicate_discontinuity(self, t: int) -> float:
        """Measure how much predicates change around time t."""
        if self._predicates is None:
            return 0.0

        T = len(self._predicates)
        w = self.config.pred_window
        left_start = max(0, t - w)
        right_end = min(T, t + w + 1)

        # Collect predicates before and after
        before_keys: Counter = Counter()
        after_keys: Counter = Counter()
        before_count = 0
        after_count = 0

        for i in range(left_start, t):
            preds = self._predicates[i]
            if preds is not None:
                for k, v in preds.items():
                    if v:
                        before_keys[k] += 1
                before_count += 1

        for i in range(t, right_end):
            preds = self._predicates[i]
            if preds is not None:
                for k, v in preds.items():
                    if v:
                        after_keys[k] += 1
                after_count += 1

        if before_count == 0 or after_count == 0:
            return 0.0

        # Normalize to frequencies
        before_freq = {k: c / before_count for k, c in before_keys.items()}
        after_freq = {k: c / after_count for k, c in after_keys.items()}

        # Compute change magnitude (symmetric difference of frequent predicates)
        all_keys = set(before_freq) | set(after_freq)
        if not all_keys:
            return 0.0

        change_sum = 0.0
        for k in all_keys:
            change_sum += abs(before_freq.get(k, 0.0) - after_freq.get(k, 0.0))

        return min(1.0, change_sum / max(len(all_keys), 1))

    def boundary_score(self, t: int) -> BoundaryScore:
        """Compute full plausibility score for a boundary at time t."""
        if t in self._scores_cache:
            return self._scores_cache[t]

        cfg = self.config
        sig_score, sources = self._signal_strength(t)
        disc_score = self._predicate_discontinuity(t)

        # Learned preference
        pref_score = 0.0
        if self._pairwise_prefs:
            self._compute_preference_scores()
            pref_score = self._preference_scores.get(t, 0.0)

        total = (
            cfg.w_signal_strength * sig_score
            + cfg.w_predicate_discontinuity * disc_score
            + pref_score
        )

        bs = BoundaryScore(
            time=t,
            signal_strength=sig_score,
            predicate_discontinuity=disc_score,
            learned_preference=pref_score,
            total=total,
            sources=sources,
        )
        self._scores_cache[t] = bs
        return bs

    def boundary_score_value(self, t: int) -> float:
        """Return just the scalar plausibility score for time t."""
        return self.boundary_score(t).total

    # ── Stage 1 filtering ────────────────────────────────────────────

    def filter_candidates(
        self,
        candidates,
        top_frac: float = 0.8,
        min_plausibility: Optional[float] = None,
    ):
        """Filter candidates by plausibility, keeping the top fraction.

        Parameters
        ----------
        candidates : list[BoundaryCandidate]
            Original candidates from Stage 1.
        top_frac : float
            Fraction of candidates to keep (e.g. 0.8 = keep top 80%).
        min_plausibility : float, optional
            Hard minimum threshold.  Defaults to ``config.min_plausibility``.

        Returns
        -------
        list[BoundaryCandidate]
            Filtered candidates sorted by center time.
        """
        if not self.config.enabled:
            return candidates

        thresh = min_plausibility if min_plausibility is not None else self.config.min_plausibility

        scored = []
        for c in candidates:
            bs = self.boundary_score(c.center)
            scored.append((bs.total, c))

        scored.sort(key=lambda x: -x[0])
        n_keep = max(1, int(len(scored) * top_frac))
        kept = [c for s, c in scored[:n_keep] if s >= thresh]

        # Also keep any candidate with a hard event source regardless of score
        hard_set = {c.center for c in kept}
        for s, c in scored[n_keep:]:
            info = self._candidate_info.get(c.center, {})
            sources = info.get("sources", [])
            if "event" in sources and c.center not in hard_set:
                kept.append(c)

        kept.sort(key=lambda c: c.center)
        return kept

    # ── Stage 2 integration ──────────────────────────────────────────

    def decoding_bonus(self, seg_start: int, seg_end: int) -> float:
        """Boundary quality bonus for Stage 2 decoding.

        Returns a score that rewards good boundaries at segment start/end.
        This can be added as an additional term in the segment score.
        """
        if not self.config.enabled or not self.config.use_in_decoding:
            return 0.0

        start_score = self.boundary_score_value(seg_start)
        end_score = self.boundary_score_value(seg_end)

        return self.config.decoding_weight * (start_score + end_score) / 2.0
