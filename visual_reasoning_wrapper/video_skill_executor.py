"""Skill executor that binds the video-visual tool registries to the
harness ``VideoAdapter``.

Mirrors :mod:`visual_reasoning_wrapper.skill_executor` (which targets
the image-VR adapter) but dispatches against
:func:`tools_video_visual.build_video_visual_registry` -- the merged
registry that combines:

* Single-frame visual tools (``detect_objects``, ``grounded_detect``,
  ``describe_region``, ``read_text_region``, ``spatial_query``, ...) bound
  to the *current* frame.
* Multi-frame video navigation tools (``get_frame``, ``sample_frames``,
  ``compare_frames``, ``detect_changes``, ``temporal_navigate``, ...).
* Cross-frame analysis tools (``track_object``, ``summarize_clip``,
  ``find_moment``, ``detect_activity``, ``compare_elements``,
  ``detect_objects_at_frame``, ``describe_frame``).
* Reasoning tools (``count_value``, ``compute_ratio``, ``compare_values``,
  ``verify_claim``).

Inner-MDP action -> tool mapping (extends the image dispatch with
temporal-aware variants)::

    GROUND
        with frame_index           -> detect_objects_at_frame
        with query AND frame range -> track_object
        with query                 -> grounded_detect (current frame)
        default                    -> detect_objects (current frame)

    RETRIEVE
        with frame_index AND bbox  -> describe_frame (focused on bbox)
        with frame_index           -> describe_frame (whole frame)
        with bbox + use_ocr        -> read_text_region (current frame)
        with bbox                  -> describe_region  (current frame)
        with entity_index          -> describe_region on cached bbox
        otherwise                  -> raise _NoToolForAction

    CHECK
        kind=COUNT/RATIO/COMPARE   -> count_value / compute_ratio / compare_values
        with start_frame/end_frame -> summarize_clip
        with activity              -> detect_activity
        with moment                -> find_moment
        with element_a + element_b -> compare_elements (cross-frame) or
                                       spatial_query (current frame)

    VERIFY / COMMIT  -> verify_claim
    EXECUTE          -> no-op (video QA has no env effects)

The dispatch loop is deliberately the same shape as
:class:`VisualReasoningExecutor.__call__` so callers familiar with the
image executor see no surprises -- the only differences are (a) the
construction takes ``frames`` / ``video_path`` instead of a single
image, and (b) ``GROUND`` / ``RETRIEVE`` / ``CHECK`` learn a few extra
temporal payload fields.

Slot resolution
---------------
The harness has already substituted ``${slot}`` placeholders before the
payload reaches us (see :class:`harness.adapters._common.HopBindings`),
so the executor sees concrete strings/numbers. Anything left as
``${...}`` is reported as an unbound-slot error, mirroring
``VisualReasoningExecutor``.

Usage
-----
::

    from visual_reasoning_wrapper.video_skill_executor import bind_executor
    from harness.adapters.video_adapter import VideoAdapter

    adapter = VideoAdapter()
    executor = bind_executor(
        adapter,
        video_path="clips/holmes_001.mp4",
        num_frames=8,
    )
    # adapter.run(skill, ctx) now uses the real video + visual + reasoning
    # tools instead of the deterministic stub.

The returned executor exposes ``derivation_log`` (the
:class:`tools_reasoning._DerivationLog`) so callers can render the
``<derivations>`` block from the resulting trace, identical to the
image executor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from common.state_schema import EvidenceRef
from harness.skill_adapter import AdapterRunContext

from .tools_reasoning import _DerivationLog
from .tools_video import _decode_video
from .tools_video_visual import build_video_visual_registry
from vlm_wrapper.tools import ToolRegistry

logger = logging.getLogger(__name__)


# ── Action -> tool mapping ─────────────────────────────────────────────

#: ``CHECK`` payload key disambiguating which derivation tool to use.
_CHECK_KIND_TO_TOOL: Dict[str, str] = {
    "COUNT": "count_value",
    "RATIO": "compute_ratio",
    "COMPARE": "compare_values",
}

#: Inner-MDP action -> evidence role. Same mapping as
#: ``visual_reasoning_wrapper.skill_executor`` so cross-domain skill
#: replays produce comparable evidence chains.
_ACTION_TO_ROLE: Dict[str, str] = {
    "GROUND": "GATHER",
    "RETRIEVE": "GATHER",
    "CHECK": "REASON",
    "VERIFY": "VERIFY",
    "COMMIT": "COMMIT",
    "EXECUTE": "COMMIT",
}


# ── Helpers ────────────────────────────────────────────────────────────

def _coerce_frame(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame)
    if isinstance(frame, (str, bytes)):
        return Image.open(frame)  # type: ignore[arg-type]
    raise TypeError(f"unsupported frame type: {type(frame).__name__}")


def _coerce_frames(frames: Sequence[Any]) -> List[Image.Image]:
    return [_coerce_frame(f) for f in frames]


def _scrub_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out


def _has_unbound_slot(payload: Dict[str, Any]) -> Optional[str]:
    for k, v in payload.items():
        if isinstance(v, str) and "${" in v and "}" in v:
            return f"unbound slot in payload[{k!r}]: {v}"
    return None


# ── Executor ───────────────────────────────────────────────────────────

@dataclass
class VideoReasoningExecutor:
    """Concrete ``HopExecutor`` for the ``video`` adapter.

    Holds the merged video + visual + reasoning + cross-frame tool
    registry plus the typed derivation log, and dispatches each hop to
    the right tool. A "current frame" index gives single-frame visual
    tools (``detect_objects`` etc.) something concrete to operate on
    while temporal tools (``track_object`` etc.) walk the whole clip.
    """

    frames: List[Image.Image]
    registry: ToolRegistry
    derivation_log: _DerivationLog
    fps: float = 1.0
    confidence: float = 0.8
    current_index: int = 0
    _last_grounded_entities: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[Any],
        *,
        fps: float = 1.0,
        current_index: Optional[int] = None,
        prefer_gdino: bool = False,
        confidence: float = 0.8,
        vlm_describer: Optional[Callable[..., str]] = None,
        sample_timestamps: Optional[List[float]] = None,
    ) -> "VideoReasoningExecutor":
        pil_frames = _coerce_frames(frames)
        if not pil_frames:
            raise ValueError("frames is empty -- need at least one frame")
        ci = current_index if current_index is not None else len(pil_frames) // 2
        ci = max(0, min(ci, len(pil_frames) - 1))
        registry = build_video_visual_registry(
            frames=pil_frames,
            fps=fps,
            current_index=ci,
            prefer_gdino=prefer_gdino,
            vlm_describer=vlm_describer,
            sample_timestamps=sample_timestamps,
            include_reasoning=True,
        )
        log = getattr(registry, "derivation_log", None)
        if log is None:
            raise RuntimeError(
                "build_video_visual_registry must attach a derivation_log "
                "(include_reasoning=True). The reasoning sub-registry "
                "was not merged."
            )
        return cls(
            frames=pil_frames,
            registry=registry,
            derivation_log=log,
            fps=fps,
            confidence=confidence,
            current_index=ci,
        )

    @classmethod
    def from_video(
        cls,
        video_path: str,
        *,
        num_frames: int = 8,
        fps: Optional[float] = None,
        current_index: Optional[int] = None,
        prefer_gdino: bool = False,
        confidence: float = 0.8,
        vlm_describer: Optional[Callable[..., str]] = None,
    ) -> "VideoReasoningExecutor":
        """Decode ``video_path`` and uniformly sample ``num_frames``."""
        decoded = _decode_video(video_path)
        if not decoded:
            raise ValueError(f"failed to decode any frames from {video_path!r}")
        if num_frames >= len(decoded):
            sampled = decoded
        else:
            step = max(1, len(decoded) // num_frames)
            sampled = decoded[::step][:num_frames]
        # When the caller does not override fps, fall back to a value
        # that makes the sampled-frame timeline tractable (frame_idx /
        # fps yields seconds in the original clip *only* if we know the
        # native fps; the benchmark loaders supply sample_timestamps in
        # that case via from_frames). Here the safe fallback is the
        # number of sampled frames per second of original duration we
        # *think* we covered, but without a duration we just use 1.0.
        return cls.from_frames(
            sampled,
            fps=fps if fps is not None else 1.0,
            current_index=current_index,
            prefer_gdino=prefer_gdino,
            confidence=confidence,
            vlm_describer=vlm_describer,
        )

    # ------------------------------------------------------------------
    # HopExecutor protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        action_type: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> Dict[str, Any]:
        unbound = _has_unbound_slot(payload)
        if unbound:
            return {"ok": False, "reason": unbound, "evidence": []}

        action = action_type.upper()
        try:
            tool_name, tool_args = self._select_tool(action, payload, ctx)
        except _NoToolForAction as exc:
            return {"ok": False, "reason": str(exc), "evidence": []}

        if tool_name is None:
            # EXECUTE / commit-only hop with no tool call.
            return {
                "ok": True,
                "observation": {
                    "echo_action": action,
                    "echo_payload": _scrub_payload(payload),
                    "note": "no-op for video (no env side effects)",
                },
                "evidence": [
                    self._make_evidence(
                        action, action.lower(), step=ctx.state.inner_step,
                    )
                ],
            }

        result = self.registry.dispatch(tool_name, tool_args)
        if result.error is not None:
            return {
                "ok": False,
                "reason": f"tool {tool_name!r} failed: {result.error}",
                "evidence": [],
            }

        # Cache the most recent ground/track result so a follow-up
        # RETRIEVE with `entity_index` can resolve to a concrete bbox.
        if tool_name in ("detect_objects", "grounded_detect",
                         "detect_objects_at_frame"):
            elements = (
                (result.result or {}).get("elements", [])
                if isinstance(result.result, dict)
                else []
            )
            self._last_grounded_entities = list(elements)

        evidence = [
            self._make_evidence(
                action,
                tool_name,
                step=ctx.state.inner_step,
                payload_snippet=_scrub_payload(tool_args),
            )
        ]
        if action == "COMMIT" and tool_name == "verify_claim":
            evidence.insert(
                0,
                self._make_evidence(
                    "VERIFY", tool_name, step=ctx.state.inner_step,
                    payload_snippet=_scrub_payload(tool_args),
                ),
            )

        return {
            "ok": True,
            "observation": {"tool": tool_name, "result": result.result},
            "evidence": evidence,
        }

    # ------------------------------------------------------------------
    # Action -> tool selection
    # ------------------------------------------------------------------

    def _select_tool(
        self,
        action: str,
        payload: Dict[str, Any],
        ctx: AdapterRunContext,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        if action == "GROUND":
            # Frame-specific grounding takes priority.
            frame_index = payload.get("frame_index")
            query = payload.get("query")
            start_frame = payload.get("start_frame")
            end_frame = payload.get("end_frame")
            if query is not None and (start_frame is not None or end_frame is not None):
                args: Dict[str, Any] = {"query": str(query)}
                if start_frame is not None:
                    args["start_frame"] = int(start_frame)
                if end_frame is not None:
                    args["end_frame"] = int(end_frame)
                if "sample_every" in payload:
                    args["sample_every"] = int(payload["sample_every"])
                return "track_object", args
            if frame_index is not None:
                args = {"frame": int(frame_index)}
                if query is not None:
                    args["query"] = str(query)
                return "detect_objects_at_frame", args
            if query is not None:
                return "grounded_detect", {
                    "query": str(query),
                    "confidence_threshold": float(
                        payload.get("confidence_threshold", 0.20)
                    ),
                    "max_results": int(payload.get("max_results", 10)),
                }
            return "detect_objects", {
                "max_elements": int(payload.get("max_elements", 25)),
                "confidence_threshold": float(
                    payload.get("confidence_threshold", 0.20)
                ),
            }

        if action == "RETRIEVE":
            frame_index = payload.get("frame_index")
            has_bbox = all(k in payload for k in ("x", "y", "w", "h"))
            if frame_index is not None and not has_bbox:
                args = {"frame_index": int(frame_index)}
                if payload.get("focus"):
                    args["focus"] = str(payload["focus"])
                return "describe_frame", args
            if has_bbox:
                args = {
                    "x": int(payload["x"]),
                    "y": int(payload["y"]),
                    "w": int(payload["w"]),
                    "h": int(payload["h"]),
                }
                tool = "read_text_region" if payload.get("use_ocr") else "describe_region"
                return tool, args
            if "entity_index" in payload:
                idx = int(payload["entity_index"])
                if 0 <= idx < len(self._last_grounded_entities):
                    bbox = self._last_grounded_entities[idx].get("bbox", {})
                    args = {
                        "x": int(bbox.get("x", 0)),
                        "y": int(bbox.get("y", 0)),
                        "w": int(bbox.get("w", 0)),
                        "h": int(bbox.get("h", 0)),
                    }
                    return "describe_region", args
            raise _NoToolForAction(
                "RETRIEVE hop needs one of: frame_index, bbox (x,y,w,h), "
                "or entity_index (into the last GROUND result)."
            )

        if action == "CHECK":
            kind = str(payload.get("kind", "")).upper()
            tool = _CHECK_KIND_TO_TOOL.get(kind)
            if tool is not None:
                args = {k: v for k, v in payload.items() if k != "kind"}
                return tool, args
            # Temporal CHECK variants when no kind is specified.
            if "start_frame" in payload or "end_frame" in payload:
                args = {}
                if "start_frame" in payload:
                    args["start_frame"] = int(payload["start_frame"])
                if "end_frame" in payload:
                    args["end_frame"] = int(payload["end_frame"])
                if "num_samples" in payload:
                    args["num_samples"] = int(payload["num_samples"])
                return "summarize_clip", args
            if "activity" in payload:
                return "detect_activity", {
                    "activity": str(payload["activity"]),
                }
            if "moment" in payload:
                return "find_moment", {
                    "description": str(payload["moment"]),
                }
            if "element_a" in payload and "element_b" in payload:
                # Cross-frame compare when frames are explicitly given;
                # otherwise stay on the current frame's spatial query.
                if "frame_a" in payload or "frame_b" in payload:
                    return "compare_elements", {
                        "element_a": str(payload["element_a"]),
                        "element_b": str(payload["element_b"]),
                        "frame_a": int(payload.get("frame_a", self.current_index)),
                        "frame_b": int(payload.get("frame_b", self.current_index)),
                    }
                return "spatial_query", {
                    "element_a": str(payload["element_a"]),
                    "element_b": str(payload["element_b"]),
                }
            raise _NoToolForAction(
                "CHECK hop requires payload['kind'] in "
                f"{sorted(_CHECK_KIND_TO_TOOL)}, OR a temporal selector "
                "(start_frame/end_frame, activity, moment), OR "
                "(element_a, element_b)."
            )

        if action in ("VERIFY", "COMMIT"):
            claim = payload.get("claim") or payload.get("answer")
            if claim is None:
                raise _NoToolForAction(
                    f"{action} hop requires payload['claim'] or "
                    "payload['answer']."
                )
            evidence_refs = (
                payload.get("evidence_refs")
                or payload.get("evidence")
                or payload.get("refs")
            )
            if evidence_refs is None:
                raise _NoToolForAction(
                    f"{action} hop requires payload['evidence_refs'] "
                    "naming hops/entities/derivations to bind to."
                )
            if isinstance(evidence_refs, (list, tuple)):
                evidence_refs = ",".join(str(r) for r in evidence_refs)
            return "verify_claim", {
                "claim": str(claim),
                "evidence_refs": str(evidence_refs),
                "confidence": str(payload.get("confidence", "medium")),
            }

        if action == "EXECUTE":
            return None, {}

        raise _NoToolForAction(
            f"action {action!r} has no video tool mapping. Supported: "
            "GROUND, CHECK, RETRIEVE, VERIFY, COMMIT, EXECUTE."
        )

    # ------------------------------------------------------------------
    # Evidence construction
    # ------------------------------------------------------------------

    def _make_evidence(
        self,
        action: str,
        tool_name: str,
        *,
        step: int,
        payload_snippet: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRef:
        role = _ACTION_TO_ROLE.get(action, "GATHER")
        return EvidenceRef(
            source=f"video:{tool_name}",
            locator=f"step={step}",
            role=role,
            confidence=self.confidence,
            payload=payload_snippet,
        )


class _NoToolForAction(ValueError):
    """Raised when an action_type cannot be mapped to a tool."""


# ── Wiring helpers ─────────────────────────────────────────────────────

def bind_executor(
    adapter: Any,
    *,
    frames: Optional[Sequence[Any]] = None,
    video_path: Optional[str] = None,
    num_frames: int = 8,
    fps: float = 1.0,
    current_index: Optional[int] = None,
    prefer_gdino: bool = False,
    confidence: float = 0.8,
    vlm_describer: Optional[Callable[..., str]] = None,
    sample_timestamps: Optional[List[float]] = None,
) -> VideoReasoningExecutor:
    """Build a :class:`VideoReasoningExecutor` and attach it to ``adapter``.

    Exactly one of ``frames`` or ``video_path`` must be provided. The
    function returns the executor so callers can inspect
    ``executor.derivation_log`` after the run to render the
    ``<derivations>`` block.
    """
    if not hasattr(adapter, "set_executor"):
        raise TypeError(
            f"adapter {type(adapter).__name__} has no set_executor -- "
            "is it a StubTransferTargetAdapter?"
        )
    if (frames is None) == (video_path is None):
        raise ValueError(
            "exactly one of frames= or video_path= must be provided"
        )
    if frames is not None:
        executor = VideoReasoningExecutor.from_frames(
            frames,
            fps=fps,
            current_index=current_index,
            prefer_gdino=prefer_gdino,
            confidence=confidence,
            vlm_describer=vlm_describer,
            sample_timestamps=sample_timestamps,
        )
    else:
        assert video_path is not None
        executor = VideoReasoningExecutor.from_video(
            video_path,
            num_frames=num_frames,
            fps=fps,
            current_index=current_index,
            prefer_gdino=prefer_gdino,
            confidence=confidence,
            vlm_describer=vlm_describer,
        )
    adapter.set_executor(executor)
    return executor


def make_video_reasoning_executor(
    *,
    frames: Optional[Sequence[Any]] = None,
    video_path: Optional[str] = None,
    num_frames: int = 8,
    fps: float = 1.0,
    current_index: Optional[int] = None,
    prefer_gdino: bool = False,
    confidence: float = 0.8,
    vlm_describer: Optional[Callable[..., str]] = None,
    sample_timestamps: Optional[List[float]] = None,
) -> VideoReasoningExecutor:
    """Functional alias for :meth:`VideoReasoningExecutor.from_frames` /
    :meth:`VideoReasoningExecutor.from_video`."""
    if (frames is None) == (video_path is None):
        raise ValueError(
            "exactly one of frames= or video_path= must be provided"
        )
    if frames is not None:
        return VideoReasoningExecutor.from_frames(
            frames,
            fps=fps,
            current_index=current_index,
            prefer_gdino=prefer_gdino,
            confidence=confidence,
            vlm_describer=vlm_describer,
            sample_timestamps=sample_timestamps,
        )
    assert video_path is not None
    return VideoReasoningExecutor.from_video(
        video_path,
        num_frames=num_frames,
        fps=fps,
        current_index=current_index,
        prefer_gdino=prefer_gdino,
        confidence=confidence,
        vlm_describer=vlm_describer,
    )


__all__ = [
    "VideoReasoningExecutor",
    "bind_executor",
    "make_video_reasoning_executor",
]
