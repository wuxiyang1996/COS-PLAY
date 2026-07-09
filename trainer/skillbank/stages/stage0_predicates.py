"""
Stage 0: Extract and booleanize predicates from observations.

Converts raw observation text/embeddings into structured predicate
probabilities P(pred | obs) and boolean sets B_t for downstream stages.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from trainer.skillbank.ingest_rollouts import TrajectoryForEM, TrajectoryFrame

logger = logging.getLogger(__name__)


def extract_predicates_from_text(
    observation: str,
    predicate_vocabulary: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Extract predicate probabilities from observation text.

    Uses keyword matching as the default predicate extractor. Each predicate
    in the vocabulary is checked for presence in the observation.

    Args:
        observation: observation text
        predicate_vocabulary: list of predicate names to check for

    Returns:
        dict mapping predicate -> probability [0, 1]
    """
    if not observation:
        return {}

    obs_lower = observation.lower()
    predicates: Dict[str, float] = {}

    if predicate_vocabulary:
        for pred in predicate_vocabulary:
            tokens = pred.lower().replace("_", " ").split()
            matched = all(t in obs_lower for t in tokens if len(t) >= 2)
            predicates[pred] = 1.0 if matched else 0.0
    else:
        patterns = [
            (r"holding\s+(\w+)", "holding_{0}"),
            (r"near\s+(\w+)", "near_{0}"),
            (r"at\s+(\w+)", "at_{0}"),
            (r"has\s+(\w+)", "has_{0}"),
            (r"completed?\s+(\w+)", "completed_{0}"),
            (r"(\w+)\s+is\s+open", "{0}_open"),
            (r"(\w+)\s+is\s+closed", "{0}_closed"),
            (r"health\s*[:=]\s*(\d+)", "health_{0}"),
            (r"score\s*[:=]\s*(\d+)", "score_{0}"),
        ]
        for pattern, template in patterns:
            for match in re.finditer(pattern, obs_lower):
                pred_name = template.format(*match.groups())
                predicates[pred_name] = 1.0

    return predicates


def booleanize(
    predicates: Dict[str, float],
    threshold: float = 0.7,
) -> Set[str]:
    """Convert predicate probabilities to a boolean set."""
    return {pred for pred, prob in predicates.items() if prob >= threshold}


def smooth_predicates(
    frames: List[TrajectoryFrame],
    window: int = 3,
) -> List[Dict[str, float]]:
    """Temporally smooth predicate probabilities with a sliding average.

    Returns a list of smoothed predicate dicts (same length as frames).
    """
    n = len(frames)
    if n == 0:
        return []

    all_preds: Set[str] = set()
    for f in frames:
        all_preds.update(f.predicates.keys())

    smoothed: List[Dict[str, float]] = []
    half_w = window // 2

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        count = end - start
        avg: Dict[str, float] = {}
        for pred in all_preds:
            total = sum(frames[j].predicates.get(pred, 0.0) for j in range(start, end))
            avg[pred] = total / count
        smoothed.append(avg)

    return smoothed


_INTENTION_TAG_RE = re.compile(r"\[(\w+)\]")

_SUBGOAL_TAGS = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)


def extract_predicates_from_intentions(
    intentions: Optional[str],
    tags: tuple = _SUBGOAL_TAGS,
) -> Dict[str, float]:
    """Extract one-hot tag predicates from a ``[TAG] phrase`` intention string.

    Returns ``{tag_<lower>: 0.0/1.0}`` dict compatible with the predicate
    vocabulary used by ``IntentionSignalExtractor``.
    """
    preds: Dict[str, float] = {}
    tag = "UNKNOWN"
    if intentions:
        m = _INTENTION_TAG_RE.match(intentions.strip())
        if m:
            raw = m.group(1).upper()
            if raw in tags:
                tag = raw

    for t in tags:
        preds[f"tag_{t.lower()}"] = float(t == tag)
    return preds


def enrich_trajectory_predicates(
    trajectory: TrajectoryForEM,
    predicate_vocabulary: Optional[List[str]] = None,
    threshold: float = 0.7,
    smoothing_window: int = 3,
    extractor_fn: Optional[Callable] = None,
) -> TrajectoryForEM:
    """Enrich a trajectory's frames with extracted/smoothed predicates.

    If frames carry ``intentions`` (Strategy C), intention-tag predicates
    are merged into the predicate dict.  If frames already have predicates,
    they are kept.  Otherwise, predicates are extracted from observation text.

    Returns the trajectory (mutated in place).
    """
    extractor = extractor_fn or extract_predicates_from_text

    for frame in trajectory.frames:
        if not frame.predicates:
            frame.predicates = extractor(
                frame.observation_text,
                predicate_vocabulary=predicate_vocabulary,
            )
        # Merge intention-based tag predicates when intentions are present
        if getattr(frame, "intentions", None):
            tag_preds = extract_predicates_from_intentions(frame.intentions)
            frame.predicates.update(tag_preds)

    smoothed = smooth_predicates(trajectory.frames, window=smoothing_window)
    for frame, sp in zip(trajectory.frames, smoothed):
        frame.predicates = sp

    return trajectory
