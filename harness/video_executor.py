"""Video hop executor — bind `VideoAdapter` to a video-QA tool surface.

Phase 5 / Stage 2 of the cross-domain transfer measurement plan
(``implementation_notes/legacy/phase5-cross-domain-measurement.md`` §11.5.2 /
§11.5.5). Stage 2's first cut keeps the executor *deterministic* —
it does not actually decode video frames or call a VLM yet — but it
emits typed evidence and propagates an answer the success_fn can
score against the cold-start ``gold_answer``.

The shape mirrors :mod:`visual_reasoning_wrapper.skill_executor`
(canonical reference for the image-VR path Stage 1 wires up). The
catch-all *observational* op set follows the gymv pattern in
:mod:`harness.gymv_executor` — verbs that *look* at the video without
moving the playhead never error out, they just emit a GATHER
``EvidenceRef`` so the harness's G0 evidence-driven invariant holds.

Recognised action types::

    SAMPLE_FRAME      — ask for a frame at index/timestamp           → GATHER
    INSPECT_FRAME     — describe the most-recently sampled frame     → GATHER
    LOCATE            — locate an entity/region in a frame           → GATHER
    OCR               — read text inside a region                    → GATHER
    CHECK_RELATION    — typed relation probe between two entities    → REASON
    RECALL_CONTEXT    — pull a prior clip / annotation               → GATHER
    EMIT_ANSWER       — commit the final MCQ choice / free-form ans  → COMMIT

Plus the union of :data:`OBSERVATIONAL_OPS` from the gymv path so a
cold-start protocol that uses generic verbs (INSPECT / OBSERVE /
VERIFY / …) still maps onto a useful evidence emission.

Public API::

    executor, holder = make_video_executor(
        video_meta=demo.expected.get("video_meta"),
        on_unresolved="skip",
    )
    adapter.set_executor(executor)
    # ...later: holder["emitted_answer"] holds the COMMIT-time answer
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from common.state_schema import EvidenceRef
from harness.adapters._stub_base import HopExecutor
from harness.skill_adapter import AdapterRunContext

logger = logging.getLogger("harness.video_executor")


VIDEO_GATHER_OPS: frozenset[str] = frozenset({
    "SAMPLE_FRAME", "INSPECT_FRAME", "LOCATE", "OCR", "RECALL_CONTEXT",
})

VIDEO_REASON_OPS: frozenset[str] = frozenset({
    "CHECK_RELATION", "CHECK", "COMPARE", "INFER", "REASON",
})

VIDEO_COMMIT_OPS: frozenset[str] = frozenset({
    "EMIT_ANSWER", "COMMIT", "ANSWER", "EMIT",
})

OBSERVATIONAL_OPS: frozenset[str] = frozenset({
    "INSPECT", "READ", "TRACK", "OBSERVE", "PERCEIVE", "SCAN",
    "VERIFY", "VALIDATE", "EVALUATE", "SIMULATE",
    "KEEP", "STOP", "CONTINUE", "GATHER", "GROUND", "RETRIEVE",
})


def _role_for_op(op: str) -> str:
    if op in VIDEO_COMMIT_OPS:
        return "COMMIT"
    if op in VIDEO_REASON_OPS:
        return "REASON"
    if op in VIDEO_GATHER_OPS or op in OBSERVATIONAL_OPS:
        return "GATHER"
    return "GATHER"


def _extract_answer(payload: Dict[str, Any]) -> Optional[Any]:
    """Pull the answer out of an EMIT_ANSWER payload.

    Tries the canonical keys in order, returning the first non-empty
    match. Falls through to ``payload`` as a whole if none match — a
    protocol that drops the answer at the top level still surfaces.
    """
    for key in ("answer", "text", "value", "claim", "selection"):
        v = payload.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
        return v
    return None


def _locator_for(action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext) -> str:
    """Build a short, deterministic locator string for the EvidenceRef.

    Mirrors the gymv executor's ``locator=f"step={...}"`` pattern but
    folds in the most discriminative payload field (frame index /
    timestamp / query) so transfer trace inspection picks up the
    "what was looked at" without us having to dump the full payload.
    """
    parts = [f"step={ctx.state.inner_step}"]
    for key in ("frame_index", "timestamp", "query", "target", "entity", "region"):
        v = payload.get(key)
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            parts.append(f"{key}={v}")
            break
    return ",".join(parts)


def make_video_executor(
    *,
    video_meta: Optional[Dict[str, Any]] = None,
    on_unresolved: str = "skip",
) -> Tuple[HopExecutor, Dict[str, Any]]:
    """Build a deterministic `HopExecutor` for the video-QA target.

    Args:
      video_meta: The ``demo.expected["video_meta"]`` dict from a
        cold-start sample (carries ``video_path`` / ``indices`` /
        ``num_frames`` / ``duration_s`` / etc.). Surfaced inside each
        evidence ``payload`` for traceability — the deterministic
        executor does *not* read frames, but a future real executor
        will.
      on_unresolved: ``"skip"`` (default) treats unknown ops as
        observational evidence and continues. ``"abort"`` returns
        ``ok=False`` so test fixtures can flush latent verb-set drift.

    Returns:
      ``(executor, holder)`` where ``holder`` is a mutable dict the
      executor writes the final ``emitted_answer`` into. Callers that
      need to read the answer back (e.g. ``video_qa_success``) reach
      into the holder, but the same answer is also propagated into
      ``ctx.state.facts["emitted_answer"]`` so the harness's
      ``episode.final_state`` carries it.
    """
    if on_unresolved not in {"skip", "abort"}:
        raise ValueError(
            f"on_unresolved={on_unresolved!r} must be 'skip' or 'abort'"
        )
    holder: Dict[str, Any] = {
        "emitted_answer": None,
        "video_meta": dict(video_meta or {}),
        "n_calls": 0,
    }

    def executor(
        action_type: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> Dict[str, Any]:
        op = (action_type or "STEP").upper()
        holder["n_calls"] = int(holder.get("n_calls", 0)) + 1
        role = _role_for_op(op)
        locator = _locator_for(op, payload, ctx)

        # COMMIT-time answer emission.
        if op in VIDEO_COMMIT_OPS:
            answer = _extract_answer(payload)
            holder["emitted_answer"] = answer
            try:
                ctx.state.facts["emitted_answer"] = answer
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "could not stash emitted_answer in state.facts: %r", exc,
                )
            ev = EvidenceRef(
                source=f"video:{op.lower()}",
                locator=locator,
                role="COMMIT",
                confidence=0.9,
                payload={
                    "answer": answer,
                    "video_path": holder["video_meta"].get("video_path"),
                },
            )
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload),
                    "answer": answer,
                },
                "evidence": [ev],
                "evidence_in": list(ctx.state.evidence or []),
                "evidence_out": [ev],
            }

        # GATHER / REASON / observational hops — no env step, just
        # emit a typed EvidenceRef so the G0 invariant holds.
        if op in VIDEO_GATHER_OPS or op in VIDEO_REASON_OPS or op in OBSERVATIONAL_OPS:
            ev = EvidenceRef(
                source=f"video:{op.lower()}",
                locator=locator,
                role=role,
                confidence=0.7,
                payload={
                    "video_path": holder["video_meta"].get("video_path"),
                    "num_frames": holder["video_meta"].get("num_frames"),
                },
            )
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload),
                    "role": role,
                },
                "evidence": [ev],
                "evidence_in": list(ctx.state.evidence or []),
                "evidence_out": [ev],
            }

        # Unknown verb.
        if on_unresolved == "abort":
            return {
                "ok": False,
                "reason": f"video_executor: unknown op={op!r}",
                "evidence": [],
            }
        # Soft skip: still emit GATHER evidence so the trace isn't
        # silently empty (mirrors gymv on_unresolved="skip").
        ev = EvidenceRef(
            source=f"video:{op.lower()}",
            locator=f"{locator},skip",
            role="GATHER",
            confidence=0.4,
            payload={"reason": "unknown_op"},
        )
        return {
            "ok": True,
            "observation": {
                "echo_action": op,
                "echo_payload": dict(payload),
                "no_env_step": True,
                "skip_reason": "unknown_op",
            },
            "evidence": [ev],
            "evidence_in": list(ctx.state.evidence or []),
            "evidence_out": [ev],
        }

    return executor, holder


__all__ = [
    "OBSERVATIONAL_OPS",
    "VIDEO_COMMIT_OPS",
    "VIDEO_GATHER_OPS",
    "VIDEO_REASON_OPS",
    "make_video_executor",
]
