"""
Step 1 — Segment summarization: compute smoothed predicate states.

Given a segment time-range and a per-frame predicate extractor, produce
booleanized P_start / P_end / B_start / B_end using a configurable window.
UI predicates use OR-aggregation (present if *any* frame shows them);
vision/HUD predicates use mean-probability aggregation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig
from skill_agents_grpo.stage3_mvp.schemas import SegmentRecord
from skill_agents_grpo.stage3_mvp.predicate_vocab import PredicateVocab, normalize_event


def _aggregate_window(
    per_step: List[Dict[str, float]],
    vocab: PredicateVocab,
    ui_or_mode: bool,
) -> Dict[str, float]:
    """Aggregate predicate probabilities over a window of frames.

    UI predicates: OR (max) if *ui_or_mode*, else average.
    All others: arithmetic mean.
    """
    if not per_step:
        return {}

    accum: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for frame in per_step:
        for k, v in frame.items():
            if ui_or_mode and vocab.is_ui(k):
                accum[k] = max(accum.get(k, 0.0), v)
                counts[k] = 1  # OR → single value
            else:
                accum[k] = accum.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

    return {k: accum[k] / counts[k] for k in accum}


def _booleanize(
    probs: Dict[str, float],
    vocab: PredicateVocab,
    p_thresh_vision: float,
) -> Set[str]:
    """Booleanize predicate probabilities.

    UI predicates: threshold at 0.5 (they are typically 0/1).
    Vision/HUD predicates: threshold at *p_thresh_vision*.
    """
    result: Set[str] = set()
    for k, v in probs.items():
        thresh = 0.5 if vocab.is_ui(k) else p_thresh_vision
        if v >= thresh:
            result.add(k)
    return result


def summarize_segment(
    seg_id: str,
    traj_id: str,
    t_start: int,
    t_end: int,
    skill_label: str,
    observations: Sequence[Any],
    extract_predicates: Callable[[Any], Dict[str, float]],
    config: Stage3MVPConfig,
    vocab: PredicateVocab,
    ui_events: Optional[List[str]] = None,
) -> SegmentRecord:
    """Build a ``SegmentRecord`` with smoothed, booleanized predicates.

    Parameters
    ----------
    seg_id, traj_id : str
        Identifiers for this segment and its parent trajectory.
    t_start, t_end : int
        Inclusive timestep range within *observations*.
    skill_label : str
        Skill id assigned by Stage 2 (or ``"NEW"``).
    observations : Sequence
        Full trajectory observations indexed by timestep.
    extract_predicates : callable
        ``(obs) -> {predicate: probability}``.
    config : Stage3MVPConfig
    vocab : PredicateVocab
        Predicate registry (auto-registers newly seen predicates).
    ui_events : list[str], optional
        Raw UI event strings that occurred within the segment.
    """
    per_step: List[Dict[str, float]] = []
    for t in range(t_start, min(t_end + 1, len(observations))):
        preds = extract_predicates(observations[t])
        if preds:
            for k in preds:
                vocab.register(k, reliability=0.9)
        per_step.append(preds if preds else {})

    seg_len = len(per_step)
    w = min(config.start_end_window, max(1, seg_len // 2))

    window_start = per_step[:w]
    window_end = per_step[-w:] if seg_len >= w else per_step

    P_start = _aggregate_window(window_start, vocab, config.ui_or_mode)
    P_end = _aggregate_window(window_end, vocab, config.ui_or_mode)

    B_start = _booleanize(P_start, vocab, config.p_thresh_vision)
    B_end = _booleanize(P_end, vocab, config.p_thresh_vision)

    events_normalized: List[str] = []
    if ui_events:
        events_normalized = [normalize_event(e) for e in ui_events]

    return SegmentRecord(
        seg_id=seg_id,
        traj_id=traj_id,
        t_start=t_start,
        t_end=t_end,
        skill_label=skill_label,
        P_start=P_start,
        P_end=P_end,
        events=events_normalized,
        B_start=B_start,
        B_end=B_end,
    )
