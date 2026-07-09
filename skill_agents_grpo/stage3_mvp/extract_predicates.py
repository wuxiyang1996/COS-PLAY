"""
Predicate extraction interface for Stage 3 MVP.

Provides a default no-op extractor and a composite extractor that
combines UI-state, HUD, and world-object predicate sources.
Users plug in domain-specific extractors (vision model, rule-based,
OCR, event logs) via the ``CompositePredicateExtractor``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from skill_agents_grpo.default_predicates import default_extract_predicates
from skill_agents_grpo.stage3_mvp.predicate_vocab import PredicateVocab


# Type alias for a single-frame predicate extractor
ExtractFn = Callable[[Any], Dict[str, float]]


class CompositePredicateExtractor:
    """Combines multiple predicate sources into a single extraction call.

    Each registered source is a callable ``(obs) -> {pred: prob}``.
    Results are merged; if two sources emit the same predicate the
    higher probability wins.

    All emitted predicates are auto-registered in the vocab.
    """

    def __init__(self, vocab: PredicateVocab) -> None:
        self._sources: List[ExtractFn] = []
        self._vocab = vocab

    def add_source(self, fn: ExtractFn) -> None:
        self._sources.append(fn)

    def __call__(self, obs: Any) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for src in self._sources:
            preds = src(obs)
            if not preds:
                continue
            for k, v in preds.items():
                self._vocab.register(k)
                if k not in merged or v > merged[k]:
                    merged[k] = v
        return merged


def extract_ui_events_from_log(
    log_entries: List[dict],
    t_start: int,
    t_end: int,
    timestamp_key: str = "t",
    event_key: str = "event",
) -> List[str]:
    """Pull raw UI event strings from a structured log within a time range.

    Parameters
    ----------
    log_entries : list[dict]
        Each entry should have at least a timestamp and event field.
    t_start, t_end : int
        Inclusive timestep range.
    timestamp_key : str
        Key for the timestep in each log entry.
    event_key : str
        Key for the event name in each log entry.
    """
    return [
        entry[event_key]
        for entry in log_entries
        if t_start <= entry.get(timestamp_key, -1) <= t_end
        and event_key in entry
    ]
