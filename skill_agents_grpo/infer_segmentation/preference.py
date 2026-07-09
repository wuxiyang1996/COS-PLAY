"""
Preference learning for InferSegmentation.

The LLM teacher provides **rankings** (not scores).  This module:
  1. Stores pairwise preferences collected from rankings.
  2. Trains a scorer f_θ from those preferences via Bradley-Terry.
  3. The trained ``PreferenceScorer`` plugs into ``SegmentScorer`` and provides
     the numeric scores used by DP/beam decoders.

This is the core of the preference-learning pipeline:
    LLM ranks skills → pairwise preferences → train f_θ → decode with f_θ.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from skill_agents_grpo.infer_segmentation.diagnostics import (
    SegmentationResult,
    SegmentDiagnostic,
)


# ── Data schema ─────────────────────────────────────────────────────

@dataclass
class PreferenceExample:
    """A single pairwise preference: skill_win ≻ skill_lose for a segment."""

    segment_start: int
    segment_end: int
    skill_win: str
    skill_lose: str
    score_win: float = 0.0
    score_lose: float = 0.0
    evidence: str = ""
    source: str = "llm"  # "llm" | "human" | "agent"
    # Omit from __repr__ so GRPO ``str([...PreferenceExample])`` reflects semantic
    # prefs only — otherwise every sample gets a unique repr (new time.time() per
    # object) while rewards correctly match identical rankings → confusing logs.
    timestamp: float = field(default_factory=time.time, repr=False)

    @property
    def is_transition_pref(self) -> bool:
        """Transition preferences use segment_start == segment_end == -1."""
        return self.segment_start == -1 and self.segment_end == -1

    def to_dict(self) -> dict:
        return {
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "skill_win": self.skill_win,
            "skill_lose": self.skill_lose,
            "score_win": self.score_win,
            "score_lose": self.score_lose,
            "evidence": self.evidence,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PreferenceExample:
        return cls(**d)


class PreferenceListWithRollouts(list):
    """Pairwise preferences plus raw LLM text (one entry per segment call).

    ``collect_segment_preferences`` may return this instead of a plain ``list``
    so GRPO rewards can distinguish **different JSON rollouts** that fuzzy-parse
    to identical ``PreferenceExample`` rows (same ranking after normalization).
    """

    def __init__(self, iterable=(), *, raw_rollouts: Optional[List[str]] = None):
        super().__init__(iterable)
        self.raw_rollouts: List[str] = list(raw_rollouts or [])


@dataclass
class PreferenceQuery:
    """Query to present to the teacher for a single uncertain segment."""

    segment_start: int
    segment_end: int
    candidate_a: str
    candidate_b: str
    score_a: float
    score_b: float
    margin: float
    context: str = ""
    breakdown_a: Dict[str, float] = field(default_factory=dict)
    breakdown_b: Dict[str, float] = field(default_factory=dict)


# ── Preference store ────────────────────────────────────────────────

class PreferenceStore:
    """Persistent storage for pairwise preferences."""

    def __init__(self, filepath: Optional[str] = None) -> None:
        self._examples: List[PreferenceExample] = []
        self._filepath = filepath

    def add(self, example: PreferenceExample) -> None:
        self._examples.append(example)

    def add_batch(self, examples: Optional[List[PreferenceExample]]) -> None:
        if examples:
            self._examples.extend(examples)

    @property
    def examples(self) -> List[PreferenceExample]:
        return list(self._examples)

    @property
    def segment_preferences(self) -> List[PreferenceExample]:
        """Preferences about segment-skill matching (not transitions)."""
        return [e for e in self._examples if not e.is_transition_pref]

    @property
    def transition_preferences(self) -> List[PreferenceExample]:
        """Preferences about skill transitions."""
        return [e for e in self._examples if e.is_transition_pref]

    def __len__(self) -> int:
        return len(self._examples)

    def known_skills(self) -> set:
        """Return the set of skill names that appear in stored preferences."""
        skills: set = set()
        for ex in self._examples:
            skills.add(ex.skill_win)
            skills.add(ex.skill_lose)
        return skills

    def save(self, filepath: Optional[str] = None) -> None:
        path = Path(filepath or self._filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._examples], f, indent=2)

    def load(self, filepath: Optional[str] = None) -> None:
        path = Path(filepath or self._filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._examples = [PreferenceExample.from_dict(d) for d in data]


# ── Query generation (active learning) ──────────────────────────────

def generate_preference_queries(
    result: SegmentationResult,
    margin_threshold: float = 1.0,
    max_queries: int = 10,
) -> List[PreferenceQuery]:
    """
    Extract uncertain segments and build preference queries.

    For each segment where margin < threshold, create a query comparing
    the top-2 skill candidates.
    """
    uncertain = result.uncertain_segments(margin_threshold)
    uncertain.sort(key=lambda s: s.margin)
    queries: List[PreferenceQuery] = []

    for seg in uncertain[:max_queries]:
        if len(seg.candidates) < 2:
            continue
        a, b = seg.candidates[0], seg.candidates[1]
        queries.append(
            PreferenceQuery(
                segment_start=seg.start,
                segment_end=seg.end,
                candidate_a=a.skill,
                candidate_b=b.skill,
                score_a=a.total_score,
                score_b=b.total_score,
                margin=seg.margin,
                breakdown_a=a.breakdown,
                breakdown_b=b.breakdown,
            )
        )

    return queries


# ── Preference-trained scorer ────────────────────────────────────────

class PreferenceScorer:
    """
    Scorer trained from pairwise preferences via Bradley-Terry.

    Learns three sets of parameters:
      - **Global skill affinity**: baseline per-skill scores (Bradley-Terry).
      - **Segment-specific win rates**: for a query segment, how often each
        skill was preferred in overlapping training segments.
      - **Transition scores**: per-(prev_skill → skill) affinity.

    The behavior_fit for a segment combines:
      (1) the global skill score, and
      (2) the segment-specific win rate from preferences covering that range.

    This lets the scorer differentiate "move fits [0-9]" from
    "attack fits [10-19]" even with a simple model.
    """

    def __init__(
        self,
        skill_names: List[str],
        lr: float = 0.1,
    ) -> None:
        self._skill_names = list(skill_names)
        self._global_scores: Dict[str, float] = {s: 0.0 for s in skill_names}
        self._transition_scores: Dict[str, float] = {}
        self._segment_prefs: List[PreferenceExample] = []
        self._lr = lr

    def _segment_win_rate(
        self, skill: str, seg_start: int, seg_end: int,
    ) -> float:
        """
        Compute win rate for ``skill`` among preferences that overlap
        with the query segment [seg_start, seg_end].

        Returns a score in [-1, +1]:  +1 = always preferred, -1 = never.
        """
        wins = 0
        losses = 0
        for pref in self._segment_prefs:
            p_start, p_end = pref.segment_start, pref.segment_end
            # Check for overlap
            if p_start > seg_end or p_end < seg_start:
                continue
            if pref.skill_win == skill:
                wins += 1
            if pref.skill_lose == skill:
                losses += 1

        total = wins + losses
        if total == 0:
            return 0.0
        return (wins - losses) / total

    def _segment_win_rate_batch(
        self,
        requests: List[Tuple[str, int, int]],
    ) -> List[float]:
        """
        Compute segment win rate for many (skill, seg_start, seg_end) at once.
        Single pass over prefs; for each pref, update counts for overlapping requests.
        """
        if not self._segment_prefs:
            return [0.0] * len(requests)
        skill_to_idx = {s: i for i, s in enumerate(self._skill_names)}
        R = len(requests)
        wins = [0] * R
        losses = [0] * R
        for pref in self._segment_prefs:
            p_start, p_end = pref.segment_start, pref.segment_end
            win_idx = skill_to_idx.get(pref.skill_win, -1)
            lose_idx = skill_to_idx.get(pref.skill_lose, -1)
            for r in range(R):
                skill, s, e = requests[r]
                if p_start > e or p_end < s:
                    continue
                r_skill_idx = skill_to_idx.get(skill, -2)
                if win_idx == r_skill_idx:
                    wins[r] += 1
                if lose_idx == r_skill_idx:
                    losses[r] += 1
        out = []
        for r in range(R):
            total = wins[r] + losses[r]
            out.append((wins[r] - losses[r]) / total if total > 0 else 0.0)
        return out

    def behavior_fit_batch(
        self,
        requests: List[Tuple[Sequence, Sequence, str, int, int]],
    ) -> List[float]:
        """
        Compute behavior_fit for many (observations, actions, skill, seg_start, seg_end)
        in one call. Uses batched segment win rate for speed at inference time.
        """
        if not requests:
            return []
        seg_reqs = [(skill, seg_start, seg_end) for (_, _, skill, seg_start, seg_end) in requests]
        seg_scores = self._segment_win_rate_batch(seg_reqs)
        out = []
        for r, ((obs, act, skill, s, e), seg_score) in enumerate(zip(requests, seg_scores)):
            global_score = self._global_scores.get(skill, 0.0)
            out.append((global_score + seg_score * 5.0) * len(obs))
        return out

    def behavior_fit(
        self,
        observations: Sequence,
        actions: Sequence,
        skill: str,
        _seg_start: int = -1,
        _seg_end: int = -1,
    ) -> float:
        """
        Learned behavior fit combining global affinity and segment-specific
        win rates from stored preferences.
        """
        global_score = self._global_scores.get(skill, 0.0)

        seg_score = 0.0
        if _seg_start >= 0 and _seg_end >= 0 and self._segment_prefs:
            seg_score = self._segment_win_rate(skill, _seg_start, _seg_end)

        return (global_score + seg_score * 5.0) * len(observations)

    def __call__(
        self,
        observations: Sequence,
        actions: Sequence,
        skill: str,
    ) -> float:
        """Callable interface for SegmentScorer (no segment range info)."""
        return self._global_scores.get(skill, 0.0) * len(observations)

    def transition_prior(self, skill: str, prev_skill: Optional[str]) -> float:
        """Learned transition score for (prev_skill -> skill)."""
        if prev_skill is None:
            return 0.0
        key = f"{prev_skill}->{skill}"
        return self._transition_scores.get(key, 0.0)

    def _bt_update(
        self,
        scores: Dict[str, float],
        key_win: str,
        key_lose: str,
    ) -> float:
        """One Bradley-Terry gradient step.  Returns loss."""
        sw = scores.get(key_win, 0.0)
        sl = scores.get(key_lose, 0.0)
        diff = sw - sl
        prob_correct = 1.0 / (1.0 + math.exp(-diff))
        loss = -math.log(max(prob_correct, 1e-10))

        grad = 1.0 - prob_correct
        if key_win in scores:
            scores[key_win] += self._lr * grad
        else:
            scores[key_win] = self._lr * grad
        if key_lose in scores:
            scores[key_lose] -= self._lr * grad
        else:
            scores[key_lose] = -self._lr * grad

        return loss

    def _bt_batch_update(
        self,
        scores: Dict[str, float],
        preferences: List[PreferenceExample],
    ) -> float:
        """
        One batch Bradley-Terry step: accumulate gradients over all preferences
        then apply once per key. Same objective as sequential _bt_update.
        """
        if not preferences:
            return 0.0
        grad_accum: Dict[str, float] = {}
        total_loss = 0.0
        for pref in preferences:
            key_win = pref.skill_win
            key_lose = pref.skill_lose
            sw = scores.get(key_win, 0.0)
            sl = scores.get(key_lose, 0.0)
            diff = sw - sl
            prob = 1.0 / (1.0 + math.exp(-diff))
            prob = max(min(prob, 1.0 - 1e-10), 1e-10)
            total_loss += -math.log(prob)
            g = 1.0 - prob
            grad_accum[key_win] = grad_accum.get(key_win, 0.0) + g
            grad_accum[key_lose] = grad_accum.get(key_lose, 0.0) - g
        for key, g in grad_accum.items():
            if key in scores:
                scores[key] += self._lr * g
            else:
                scores[key] = self._lr * g
        return total_loss / len(preferences)

    def update(self, preferences: List[PreferenceExample], batch: bool = True) -> float:
        """
        One gradient step on all preferences.

        If batch=True (default), accumulates gradients over all preferences
        and applies once per parameter (faster, same objective). If batch=False,
        updates after each preference (sequential, original behavior).
        """
        if not preferences:
            return 0.0
        if batch:
            segment_prefs = [p for p in preferences if not p.is_transition_pref]
            transition_prefs = [p for p in preferences if p.is_transition_pref]
            loss_seg = self._bt_batch_update(self._global_scores, segment_prefs)
            loss_trans = self._bt_batch_update(self._transition_scores, transition_prefs)
            n = len(preferences)
            return (loss_seg * len(segment_prefs) + loss_trans * len(transition_prefs)) / max(n, 1)
        total_loss = 0.0
        count = 0
        for pref in preferences:
            if pref.is_transition_pref:
                loss = self._bt_update(
                    self._transition_scores,
                    pref.skill_win,
                    pref.skill_lose,
                )
            else:
                loss = self._bt_update(
                    self._global_scores,
                    pref.skill_win,
                    pref.skill_lose,
                )
            total_loss += loss
            count += 1
        return total_loss / max(count, 1)

    def train(
        self,
        store: PreferenceStore,
        epochs: int = 10,
        batch: bool = True,
    ) -> List[float]:
        """Train for multiple epochs, return per-epoch losses. If batch=True (default), each epoch does one batch BT update; if False, one update per preference."""
        self._segment_prefs = store.segment_preferences
        losses = []
        for _ in range(epochs):
            loss = self.update(store.examples, batch=batch)
            losses.append(loss)
        return losses

    @property
    def skill_scores(self) -> Dict[str, float]:
        return dict(self._global_scores)

    @property
    def transition_scores(self) -> Dict[str, float]:
        return dict(self._transition_scores)
