"""Per-step `SuccessFn` factory for ``target_domain="video"``.

Phase 5 / Stage 2 — Video-Holmes + SIV-Bench transfer cell. The
factory is registered with :func:`harness.gymv_success.register_success_fn`
at import time, so the ``FewShotAdapter`` automatically picks it up
when the orchestrator probes a transfer cell with
``target_domain="video"``.

Behaviour mirrors Stage 1's image-VR scorer (`harness.qa_success`)
with one extra fallback: the deterministic video executor in
:mod:`harness.video_executor` writes the COMMIT-time answer into
``ctx.state.facts["emitted_answer"]`` (in addition to the
``observation.answer`` payload). The harness's ``episode.final_state``
captures that mutation, so we use it as a fallback when
``episode.outcome.answer`` is not populated by the adapter.

The same :func:`harness.qa_success.qa_answer_matches` matcher is
reused — every QA target shares an MCQ-aware comparison rule, so
keeping the matcher centralised avoids drift.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from data_structure.extensions.skill_episode import SkillEpisode

logger = logging.getLogger("harness.video_qa_success")


def _get_emitted_answer_from_episode(episode: SkillEpisode) -> Optional[Any]:
    """Pull a usable answer off a finished `SkillEpisode`.

    Order of precedence (mirrors Stage 1's qa_success but adds the
    facts-fallback the video executor pushes through ``ctx.state``):

      1. ``episode.outcome.answer`` — the canonical channel; the
         adapter would set this if it returned an `AdapterRunResult`
         with ``answer`` populated.
      2. ``episode.final_state["facts"]["emitted_answer"]`` — what
         the deterministic video executor stashes when handling an
         ``EMIT_ANSWER`` hop. Falls through to
         ``initial_state["facts"]`` for the same key in case the
         executor mutated state but the harness didn't refresh
         ``final_state`` (e.g. if a later hop aborted).
      3. None when no channel surfaced an answer.
    """
    outcome = getattr(episode, "outcome", None)
    if outcome is not None:
        ans = getattr(outcome, "answer", None)
        if ans is not None and (not isinstance(ans, str) or ans.strip()):
            return ans

    for snap_attr in ("final_state", "initial_state"):
        snap = getattr(episode, snap_attr, None) or {}
        if not isinstance(snap, dict):
            continue
        facts = snap.get("facts") or {}
        ans = facts.get("emitted_answer")
        if ans is None:
            continue
        if isinstance(ans, str) and not ans.strip():
            continue
        return ans
    return None


def make_video_qa_success_fn(
    *,
    pass_rate_threshold: float = 1.0,
    require_episode_success: bool = True,
) -> Callable[[SkillEpisode, Any], float]:
    """Return a per-shot scorer for `FewShotAdapter`.

    Args:
      pass_rate_threshold: Unused for QA-shaped scoring (we score 1.0
        on a match, 0.0 otherwise) but accepted for signature
        compatibility with `make_per_step_success_fn`.
      require_episode_success: When True, the underlying
        ``episode.outcome.success`` must also hold; otherwise just
        the answer match is consulted.
    """
    from harness.qa_success import qa_answer_matches

    def _score(episode: SkillEpisode, demo: Any) -> float:
        if require_episode_success:
            out = getattr(episode, "outcome", None)
            if out is None or not getattr(out, "success", False):
                return 0.0

        expected = getattr(demo, "expected", None) or {}
        gold = expected.get("gold_answer")
        if gold is None:
            return 1.0 if (
                episode.outcome and episode.outcome.success
            ) else 0.0

        emitted = _get_emitted_answer_from_episode(episode)
        if emitted is None:
            return 0.0

        is_mcq = bool(expected.get("is_mcq"))
        try:
            ok = qa_answer_matches(
                predicted=str(emitted) if emitted is not None else None,
                gold=str(gold) if gold is not None else None,
                is_mcq=is_mcq,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "qa_answer_matches raised (%r); scoring 0.0", exc,
            )
            return 0.0
        return 1.0 if ok else 0.0

    return _score


# ----------------------------------------------------------------------
# Domain registration. Importing this module is the only side-effect
# the orchestrator needs to make ``success_fn_for_domain('video')``
# resolve to our factory.
# ----------------------------------------------------------------------

from harness.gymv_success import register_success_fn  # noqa: E402

register_success_fn("video", make_video_qa_success_fn)


__all__ = [
    "make_video_qa_success_fn",
]
