"""QA-style success_fn factory for image-VR (and, later, video-VR).

Reference: ``harness/gymv_success.py:553`` is the registration pattern;
``implementation_notes/cross-domain-transfer-suite-rollout.md §11.5.5``
is the design note. The factory signature matches every other
registered factory (``pass_rate_threshold``, ``require_episode_success``)
so the registry can call it uniformly (`gymv_success.py:540-543`).
Stage 1 owns ``visual_reasoning``; Stage 2 will register ``video``
separately in ``harness/video_qa_success.py``.
"""
from __future__ import annotations

import logging
import string
from typing import Any, Callable, Optional

from data_structure.extensions.skill_episode import SkillEpisode

logger = logging.getLogger("harness.qa_success")

__all__ = ["make_qa_success_fn", "qa_answer_matches"]

_PUNCT_TO_STRIP: str = string.punctuation + "“”‘’`´"


def _normalise_freeform(text: str) -> str:
    """Lower-case, strip punctuation/quotes, collapse whitespace."""
    if not text:
        return ""
    s = text.strip().lower()
    s = s.translate(str.maketrans("", "", _PUNCT_TO_STRIP))
    return " ".join(s.split())


def _normalise_mcq_letter(text: str) -> str:
    """Pull a single A-Z letter out of MCQ-shaped strings ("A", " a ",
    "A.", "Answer: A", "(A)"). Returns "" when none recoverable."""
    if not text:
        return ""
    s = text.strip().strip(string.punctuation + "“”‘’ ").strip()
    for ch in s:
        if ch.isalpha():
            return ch.upper()
    return ""


def qa_answer_matches(
    predicted: Optional[str],
    gold: Optional[str],
    *,
    is_mcq: bool = False,
) -> bool:
    """Decide whether ``predicted`` matches ``gold``.

    MCQ: case-insensitive single-letter equivalence. Free-form:
    case+whitespace+punct-normalised exact match OR substring
    containment in either direction. Empty/None on either side ⇒
    ``False`` (silence is failure, matching gymv's missing-snapshot
    rule).
    """
    if predicted is None or gold is None:
        return False
    pred_s = str(predicted)
    gold_s = str(gold)
    if not pred_s.strip() or not gold_s.strip():
        return False
    if is_mcq:
        return _normalise_mcq_letter(pred_s) == _normalise_mcq_letter(gold_s)
    p = _normalise_freeform(pred_s)
    g = _normalise_freeform(gold_s)
    if not p or not g:
        return False
    if p == g:
        return True
    return (g in p) or (p in g)


def make_qa_success_fn(
    *,
    pass_rate_threshold: float = 1.0,
    require_episode_success: bool = True,
) -> Callable[[SkillEpisode, Any], float]:
    """Return a ``SuccessFn`` that scores by answer-match against
    ``demo.expected["gold_answer"]`` (with ``is_mcq``).

    ``pass_rate_threshold`` is unused for binary QA — accepted only to
    satisfy the registry contract (`gymv_success.py:540-543`).
    ``require_episode_success`` (default True, matches gymv) gates
    whether ``episode.outcome.success`` AND ``contract_satisfied``
    must also hold; when False, only the answer match is consulted.
    """
    _ = pass_rate_threshold  # registry-contract symmetry

    def _score(episode: SkillEpisode, demo: Any) -> float:
        out = episode.outcome
        if require_episode_success:
            if out is None or not out.success or not out.contract_satisfied:
                return 0.0
        if out is None:
            return 0.0
        predicted = getattr(out, "answer", None)
        expected = getattr(demo, "expected", None) or {}
        gold = expected.get("gold_answer")
        is_mcq = bool(expected.get("is_mcq", False))
        if predicted is None or gold is None:
            return 0.0
        return 1.0 if qa_answer_matches(
            str(predicted), str(gold), is_mcq=is_mcq
        ) else 0.0

    return _score


from harness.gymv_success import register_success_fn  # noqa: E402

register_success_fn("visual_reasoning", make_qa_success_fn)
