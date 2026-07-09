"""
Segment scoring: the interpretable decomposition.

Score(i, j, k | k_prev) =
    log p_k(a_{i:j} | x_{i:j})          [behavior fit]
  + IntentionFit(k, i, j)               [intention tag agreement]
  + log p(l = j-i+1 | k)                [duration prior]
  + log p(k | k_prev)                   [transition prior]
  + lambda * Compat(k, P_i, P_j)        [contract compatibility]

The **behavior fit** and **transition prior** are learned from LLM
preferences via ``PreferenceScorer`` (trained with Bradley-Terry).

The **duration prior** uses a simple Gaussian (or can be learned).

The **contract compatibility** term provides the Stage 3 → Stage 2
closed-loop feedback.  When ``ContractFeedbackConfig.mode`` is
``"weak"`` or ``"strong"``, the ``compat_fn`` from the skill bank
scores each segment against the skill's learned effects contract.
Set ``mode="off"`` (default) to disable and fall back to the LLM
teacher's implicit contract awareness via behavior_fit.
"""

from __future__ import annotations

import math
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

from skill_agents_grpo.infer_segmentation.config import (
    ContractFeedbackConfig,
    DurationPriorConfig,
    NewSkillConfig,
    ScorerWeights,
    SegmentationConfig,
)

_NEG_INF = float("-inf")

NEW_SKILL: str = "__NEW__"


# ── Protocol: pluggable behavior-fit scorer ─────────────────────────

class BehaviorFitScorer(Protocol):
    """Score how well actions a_{i:j} match skill k given observations x_{i:j}."""

    def __call__(
        self,
        observations: Sequence,
        actions: Sequence,
        skill: str,
    ) -> float: ...


# ── Default / fallback implementations ──────────────────────────────

def uniform_behavior_fit(
    observations: Sequence,
    actions: Sequence,
    skill: str,
) -> float:
    """Uniform prior: returns 0 for every segment-skill pair."""
    return 0.0


def gaussian_duration_log_prob(
    length: int,
    skill: str,
    per_skill_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    config: Optional[DurationPriorConfig] = None,
) -> float:
    """
    log p(l | k) as a truncated Gaussian.

    per_skill_stats maps skill -> (mean, std).
    Falls back to config defaults if skill is unknown.
    """
    cfg = config or DurationPriorConfig()
    if length < cfg.min_length or length > cfg.max_length:
        return _NEG_INF

    if per_skill_stats and skill in per_skill_stats:
        mu, sigma = per_skill_stats[skill]
    else:
        mu, sigma = cfg.default_mean, cfg.default_std
    sigma = max(sigma, 1e-6)

    return -0.5 * ((length - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


# ── Composite scorer ────────────────────────────────────────────────

class SegmentScorer:
    """
    Composite scorer: combines preference-learned terms with duration prior
    and (optionally) contract compatibility from the skill bank.

    In the preference-learning pipeline:
      - ``behavior_fit`` comes from ``PreferenceScorer.behavior_fit``
        (trained on LLM rankings).
      - ``transition_prior`` comes from ``PreferenceScorer.transition_prior``
        (trained on LLM transition rankings).
      - ``duration_prior`` uses a Gaussian (configurable).
      - ``contract_compat`` feeds back Stage 3 effects contracts.
        Controlled by ``ContractFeedbackConfig``.  Pass
        ``compat_fn=bank.compat_fn`` to enable the closed loop.

    For offline testing, you can pass a custom ``behavior_fit_fn`` instead.
    """

    def __init__(
        self,
        skill_names: List[str],
        config: Optional[SegmentationConfig] = None,
        *,
        behavior_fit_fn: Optional[BehaviorFitScorer] = None,
        transition_fn: Optional[Callable] = None,
        duration_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        compat_fn: Optional[Callable] = None,
        boundary_scorer: Optional[Callable] = None,
        intention_fit_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config or SegmentationConfig()
        self._weights = self.config.weights

        self._skills = list(skill_names)
        if self.config.new_skill.enabled and NEW_SKILL not in self._skills:
            self._skills.append(NEW_SKILL)

        self._behavior_fit = behavior_fit_fn or uniform_behavior_fit
        self._transition_fn = transition_fn
        self._duration_stats = duration_stats
        self._compat_fn = compat_fn
        self._boundary_scorer = boundary_scorer
        self._intention_fit_fn = intention_fit_fn
        self._new_cfg = self.config.new_skill

        import inspect
        sig = inspect.signature(self._behavior_fit)
        self._bf_has_seg_range = "_seg_start" in sig.parameters

    @property
    def skill_names(self) -> List[str]:
        return list(self._skills)

    @property
    def num_skills(self) -> int:
        return len(self._skills)

    def set_behavior_fit(self, fn: BehaviorFitScorer) -> None:
        """Hot-swap the behavior-fit scorer (e.g. after preference training)."""
        self._behavior_fit = fn

    def set_transition_fn(self, fn: Callable) -> None:
        """Hot-swap the transition scorer (e.g. from PreferenceScorer)."""
        self._transition_fn = fn

    # ── per-term methods ────────────────────────────────────────────

    def behavior_fit(
        self,
        observations: Sequence,
        actions: Sequence,
        skill: str,
        seg_start: int = -1,
        seg_end: int = -1,
    ) -> float:
        if skill == NEW_SKILL:
            return self._new_cfg.background_log_prob * len(observations)
        if self._bf_has_seg_range:
            return self._behavior_fit(
                observations, actions, skill,
                _seg_start=seg_start, _seg_end=seg_end,
            )
        return self._behavior_fit(observations, actions, skill)

    def duration_prior(self, length: int, skill: str) -> float:
        if skill == NEW_SKILL:
            return 0.0
        return gaussian_duration_log_prob(
            length, skill, self._duration_stats, self.config.duration
        )

    def transition_prior(self, skill: str, prev_skill: Optional[str]) -> float:
        if skill == NEW_SKILL or prev_skill == NEW_SKILL:
            return -self._new_cfg.penalty if skill == NEW_SKILL else 0.0
        if self._transition_fn is not None:
            return self._transition_fn(skill, prev_skill)
        return 0.0

    def contract_compat(
        self,
        skill: str,
        predicates_start: Optional[dict],
        predicates_end: Optional[dict],
    ) -> float:
        if self._compat_fn is not None and skill != NEW_SKILL:
            return self._compat_fn(skill, predicates_start, predicates_end)
        return 0.0

    def intention_fit(self, skill: str, i: int, j: int) -> float:
        """Score based on per-step intention tag agreement.

        Returns match_fraction * segment_length so the signal scales
        consistently with behavior_fit (which also scales by length).
        Returns 0.0 when no intention tags are available.
        """
        if self._intention_fit_fn is None or skill == NEW_SKILL:
            return 0.0
        return self._intention_fit_fn(skill, i, j)

    def boundary_quality(self, seg_start: int, seg_end: int) -> float:
        """Boundary plausibility bonus from BoundaryPreferenceScorer."""
        if self._boundary_scorer is not None:
            return self._boundary_scorer(seg_start, seg_end)
        return 0.0

    # ── composite score ─────────────────────────────────────────────

    def boundary_preference(self, seg_start: int, seg_end: int) -> float:
        """Boundary preference score: positive means "cut here is good".

        Combines the boundary scorer output (if any) with structural priors.
        This is the Phase 3 explicit boundary preference learning signal.
        """
        score = 0.0
        if self._boundary_scorer is not None:
            score += self._boundary_scorer(seg_start, seg_end)
        return score

    def score(
        self,
        i: int,
        j: int,
        skill: str,
        prev_skill: Optional[str],
        observations: Sequence,
        actions: Sequence,
        predicates_start: Optional[dict] = None,
        predicates_end: Optional[dict] = None,
    ) -> float:
        w = self._weights
        length = j - i + 1

        bf = w.behavior_fit * self.behavior_fit(observations, actions, skill, i, j)
        intf = w.intention_fit * self.intention_fit(skill, i, j)
        dp_ = w.duration_prior * self.duration_prior(length, skill)
        tp = w.transition_prior * self.transition_prior(skill, prev_skill)
        cc = w.contract_compat * self.contract_compat(skill, predicates_start, predicates_end)
        bq = self.boundary_quality(i, j)
        bp = w.boundary_preference * self.boundary_preference(i, j)

        return bf + intf + dp_ + tp + cc + bq + bp

    def score_breakdown(
        self,
        i: int,
        j: int,
        skill: str,
        prev_skill: Optional[str],
        observations: Sequence,
        actions: Sequence,
        predicates_start: Optional[dict] = None,
        predicates_end: Optional[dict] = None,
    ) -> Dict[str, float]:
        """Return per-term scores (un-weighted and weighted)."""
        w = self._weights
        length = j - i + 1

        bf_raw = self.behavior_fit(observations, actions, skill, i, j)
        if_raw = self.intention_fit(skill, i, j)
        dp_raw = self.duration_prior(length, skill)
        tp_raw = self.transition_prior(skill, prev_skill)
        cc_raw = self.contract_compat(skill, predicates_start, predicates_end)
        bq_raw = self.boundary_quality(i, j)
        bp_raw = self.boundary_preference(i, j)

        return {
            "behavior_fit": bf_raw,
            "behavior_fit_w": w.behavior_fit * bf_raw,
            "intention_fit": if_raw,
            "intention_fit_w": w.intention_fit * if_raw,
            "duration_prior": dp_raw,
            "duration_prior_w": w.duration_prior * dp_raw,
            "transition_prior": tp_raw,
            "transition_prior_w": w.transition_prior * tp_raw,
            "contract_compat": cc_raw,
            "contract_compat_w": w.contract_compat * cc_raw,
            "boundary_quality": bq_raw,
            "boundary_preference": bp_raw,
            "boundary_preference_w": w.boundary_preference * bp_raw,
            "total": (
                w.behavior_fit * bf_raw
                + w.intention_fit * if_raw
                + w.duration_prior * dp_raw
                + w.transition_prior * tp_raw
                + w.contract_compat * cc_raw
                + bq_raw
                + w.boundary_preference * bp_raw
            ),
        }

    def score_breakdown_batch(
        self,
        requests: List[Tuple[int, int, str, Optional[str], Sequence, Sequence, Optional[dict], Optional[dict]]],
    ) -> List[Dict[str, float]]:
        """
        Compute score breakdown for many (i, j, skill, prev_skill, obs, actions, pred_start, pred_end)
        in one call. Uses behavior_fit_batch when the behavior_fit_fn supports it (e.g. PreferenceScorer).
        """
        if not requests:
            return []
        w = self._weights
        # Batch behavior_fit when available
        bf_results: List[float] = [0.0] * len(requests)
        batch_indices: List[int] = []
        batch_reqs: List[Tuple[Sequence, Sequence, str, int, int]] = []
        for r, req in enumerate(requests):
            i, j, skill, prev_skill, obs, actions_slice, p_start, p_end = req
            length = j - i + 1
            if skill == NEW_SKILL:
                bf_results[r] = self._new_cfg.background_log_prob * len(obs)
            else:
                batch_indices.append(r)
                batch_reqs.append((obs, actions_slice, skill, i, j))
        if batch_reqs:
            bf_fn = getattr(self._behavior_fit, "behavior_fit_batch", None)
            if callable(bf_fn):
                bf_batch = bf_fn(batch_reqs)
                for idx, r in enumerate(batch_indices):
                    bf_results[r] = bf_batch[idx]
            else:
                for idx, r in enumerate(batch_indices):
                    obs, actions_slice, skill, i, j = batch_reqs[idx]
                    bf_results[r] = self.behavior_fit(obs, actions_slice, skill, i, j)
        # Build full breakdown per request
        out = []
        for r, req in enumerate(requests):
            i, j, skill, prev_skill, obs, actions_slice, p_start, p_end = req
            length = j - i + 1
            bf_raw = bf_results[r]
            if_raw = self.intention_fit(skill, i, j)
            dp_raw = self.duration_prior(length, skill)
            tp_raw = self.transition_prior(skill, prev_skill)
            cc_raw = self.contract_compat(skill, p_start, p_end)
            bq_raw = self.boundary_quality(i, j)
            bp_raw = self.boundary_preference(i, j)
            out.append({
                "behavior_fit": bf_raw,
                "behavior_fit_w": w.behavior_fit * bf_raw,
                "intention_fit": if_raw,
                "intention_fit_w": w.intention_fit * if_raw,
                "duration_prior": dp_raw,
                "duration_prior_w": w.duration_prior * dp_raw,
                "transition_prior": tp_raw,
                "transition_prior_w": w.transition_prior * tp_raw,
                "contract_compat": cc_raw,
                "contract_compat_w": w.contract_compat * cc_raw,
                "boundary_quality": bq_raw,
                "boundary_preference": bp_raw,
                "boundary_preference_w": w.boundary_preference * bp_raw,
                "total": (
                    w.behavior_fit * bf_raw
                    + w.intention_fit * if_raw
                    + w.duration_prior * dp_raw
                    + w.transition_prior * tp_raw
                    + w.contract_compat * cc_raw
                    + bq_raw
                    + w.boundary_preference * bp_raw
                ),
            })
        return out

    def rank_skills_for_segment(
        self,
        i: int,
        j: int,
        observations: Sequence,
        actions: Sequence,
        predicates_start: Optional[dict] = None,
        predicates_end: Optional[dict] = None,
        prev_skill: Optional[str] = None,
        *,
        include_breakdown: bool = False,
    ) -> List[Tuple[str, float, Optional[Dict[str, float]]]]:
        """
        Get a full ranking of skills for one segment using the current scorer.

        We only store pairwise preferences (A ≻ B); the ranking is derived by
        scoring every skill for this segment (behavior_fit + duration_prior +
        transition_prior + contract_compat) and sorting by total score descending.

        Parameters
        ----------
        i, j : int
            Segment range [i, j] (inclusive).
        observations, actions : Sequence
            Segment observations and actions (slices obs[i:j+1], act[i:j+1]).
        predicates_start, predicates_end : dict, optional
            State at segment start/end.
        prev_skill : str, optional
            Previous segment's skill (for transition prior).
        include_breakdown : bool
            If True, each element is (skill, total_score, breakdown); else breakdown is None.

        Returns
        -------
        list of (skill, total_score, breakdown or None)
            Sorted best-first (highest total_score first).
        """
        scored: List[Tuple[float, str, Optional[Dict[str, float]]]] = []
        for skill in self._skills:
            bd = self.score_breakdown(
                i, j, skill, prev_skill, observations, actions,
                predicates_start, predicates_end,
            )
            total = bd["total"]
            scored.append((total, skill, bd if include_breakdown else None))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(s, t, b) for t, s, b in scored]
