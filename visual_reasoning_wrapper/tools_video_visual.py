"""Cross-frame visual understanding tools for video + vision reasoning.

Bridges the temporal navigation of tools_video with the vision-model
analysis of tools_visual.  These tools enable multi-hop chains like:

  detect_objects(frame=0) → track_object("submit button", frames=0..10)
  → find_moment("dialog closes") → describe_region(frame=15, bbox=...)

Each tool has access to both the video state (frame sequence) and the
visual analysis pipeline (OmniParser, Florence-2, OCR) so it can run
detections across multiple frames.

Usage::

    from visual_reasoning_wrapper.tools_video_visual import build_video_visual_registry

    # From pre-decoded frames
    registry = build_video_visual_registry(frames=pil_frames, fps=2.0)

    # From a video file
    registry = build_video_visual_registry(video_path="demo.mp4")

    # The registry includes ALL tools: video nav + visual + cross-frame
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Callable

import numpy as np
from PIL import Image

from vlm_wrapper.tools import ToolDef, ToolRegistry
from .tools_video import _VideoState, _decode_video, build_video_registry
from .tools_visual import _VisualState, build_visual_registry

logger = logging.getLogger(__name__)


# ── Cross-frame tool definitions ─────────────────────────────────────

TOOL_TRACK_OBJECT = ToolDef(
    name="track_object",
    description=(
        "Track a specific element across multiple frames. Given an "
        "element label or description, runs detection on sampled frames "
        "in the specified range and reports where the element appears, "
        "how it moves, and when it disappears. Returns per-frame bbox "
        "and a motion summary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Label or description of the element to track.",
            },
            "start_frame": {
                "type": "integer",
                "description": "Start of frame range. Default 0.",
            },
            "end_frame": {
                "type": "integer",
                "description": "End of frame range. Default last frame.",
            },
            "sample_every": {
                "type": "integer",
                "description": "Check every N-th frame. Default: auto (5-10 samples).",
            },
        },
        "required": ["query"],
    },
    domain="video_visual",
)

TOOL_SUMMARIZE_CLIP = ToolDef(
    name="summarize_clip",
    description=(
        "Summarize visual changes across a range of frames. Samples "
        "frames, runs detection + scene classification on each, and "
        "reports what elements appear/disappear, layout changes, and "
        "scene transitions. Returns a structured timeline."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_frame": {
                "type": "integer",
                "description": "Start frame index. Default 0.",
            },
            "end_frame": {
                "type": "integer",
                "description": "End frame index. Default last.",
            },
            "num_samples": {
                "type": "integer",
                "description": "Number of frames to sample. Default 5.",
            },
        },
        "required": [],
    },
    domain="video_visual",
)

TOOL_FIND_MOMENT = ToolDef(
    name="find_moment",
    description=(
        "Find the frame where a specific visual event occurs. Searches "
        "through the video by running detection on sampled frames and "
        "checking for the presence/absence of elements matching the "
        "query. Returns the frame index and timestamp of the best match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Description of what to find (e.g. 'dialog closes', 'score changes', 'new element appears').",
            },
            "event_type": {
                "type": "string",
                "enum": ["appears", "disappears", "changes", "any"],
                "description": "Type of event. 'appears': element first shows up. 'disappears': element gone. 'changes': element changes position/label. Default 'any'.",
            },
            "start_frame": {
                "type": "integer",
                "description": "Start of search range. Default 0.",
            },
            "end_frame": {
                "type": "integer",
                "description": "End of search range. Default last.",
            },
        },
        "required": ["query"],
    },
    domain="video_visual",
)

TOOL_DETECT_ACTIVITY = ToolDef(
    name="detect_activity",
    description=(
        "Detect the type of visual activity happening in a frame range. "
        "Analyses element changes, motion patterns, and scene structure "
        "across sampled frames to classify the activity (scrolling, "
        "typing, navigation, game action, idle, etc.)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_frame": {
                "type": "integer",
                "description": "Start frame. Default 0.",
            },
            "end_frame": {
                "type": "integer",
                "description": "End frame. Default last.",
            },
        },
        "required": [],
    },
    domain="video_visual",
)

TOOL_COMPARE_ELEMENTS = ToolDef(
    name="compare_elements",
    description=(
        "Compare detected elements between two specific frames. Shows "
        "which elements are added, removed, moved, or changed label. "
        "More semantic than pixel-level compare_frames — this compares "
        "objects rather than raw pixels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "frame_a": {
                "type": "integer",
                "description": "First frame index.",
            },
            "frame_b": {
                "type": "integer",
                "description": "Second frame index.",
            },
        },
        "required": ["frame_a", "frame_b"],
    },
    domain="video_visual",
)

TOOL_DETECT_OBJECTS_AT_FRAME = ToolDef(
    name="detect_objects_at_frame",
    description=(
        "Run object detection on a specific frame (not just the current "
        "one). Returns detected elements with bboxes and labels. "
        "Combines temporal access with visual grounding.  NOTE: on "
        "natural video (movies, vlogs, documentaries) this routes to an "
        "open-vocabulary natural-image detector, not the UI icon "
        "captioner — for pure narrative questions (relationships, "
        "motives, plot) prefer `describe_frame` which gives a coherent "
        "caption of the whole frame."
    ),
    parameters={
        "type": "object",
        "properties": {
            "frame_index": {
                "type": "integer",
                "description": "Frame to run detection on.",
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Min confidence. Default 0.3 for video "
                               "(subtitle-style hallucinations cluster "
                               "below this).",
            },
            "max_elements": {
                "type": "integer",
                "description": "Max elements. Default 30.",
            },
        },
        "required": ["frame_index"],
    },
    domain="video_visual",
)


TOOL_DESCRIBE_FRAME = ToolDef(
    name="describe_frame",
    description=(
        "Ask the vision-LM to describe what is happening in a specific "
        "frame: the setting, the visible people (age/appearance/role), "
        "their actions, and any notable objects.  Prefer this over "
        "`detect_objects_at_frame` for narrative video-QA (social "
        "relationships, motives, causality, plot) where a coherent "
        "caption is more useful than pixel-accurate boxes.  You can pass "
        "an optional `focus` question to get an answer tailored to what "
        "you care about."
    ),
    parameters={
        "type": "object",
        "properties": {
            "frame_index": {
                "type": "integer",
                "description": "Frame to describe (0-based).",
            },
            "focus": {
                "type": "string",
                "description": "Optional guiding question (e.g. "
                               "\"how many people are visible and what "
                               "are they doing?\"). Keep short.",
            },
        },
        "required": ["frame_index"],
    },
    domain="video_visual",
)


# ── Combined state ───────────────────────────────────────────────────

class _VideoVisualState:
    """Holds video frames and per-frame visual analysis caches."""

    def __init__(
        self,
        frames: list[Image.Image | np.ndarray] | None = None,
        video_path: str | None = None,
        fps: float = 1.0,
        current_index: int = 0,
        *,
        prefer_gdino: bool = False,
        vlm_describer: Callable[..., str] | None = None,
        sample_timestamps: list[float] | None = None,
    ):
        self._video = _VideoState(
            frames=frames, video_path=video_path,
            fps=fps, current_index=current_index,
        )
        self._visual_cache: dict[int, _VisualState] = {}
        # When True, _VisualState routes detection through GroundingDINO
        # (open-vocab natural-image detector) instead of the default
        # OmniParser-v2 pipeline whose Florence-2 icon captioner
        # hallucinates subtitle-style captions on cinematic frames.
        self._prefer_gdino = prefer_gdino
        # Optional callable: (pil_frame, focus=None) -> str.  Wired by
        # ground.py so `describe_frame` can call the same model the
        # outer tool loop is using.
        self._vlm_describer = vlm_describer
        # Optional per-sampled-frame wallclock timestamps in the
        # original video.  When present, takes precedence over
        # `i / fps` math and is surfaced to the model as the true
        # timestamp so `sample_frames(start_sec, end_sec)` lines up
        # with the actual video timeline.
        self._sample_timestamps = sample_timestamps or []

    @property
    def video(self) -> _VideoState:
        return self._video

    @property
    def total_frames(self) -> int:
        return self._video.total_frames

    @property
    def fps(self) -> float:
        return self._video._fps

    def get_visual(self, frame_idx: int) -> _VisualState | None:
        """Get (or create) visual state for a specific frame."""
        if frame_idx in self._visual_cache:
            return self._visual_cache[frame_idx]
        frame = self._video.get_frame(frame_idx)
        if frame is None:
            return None
        vs = _VisualState(frame, prefer_gdino=self._prefer_gdino)
        self._visual_cache[frame_idx] = vs
        return vs

    def frame_timestamp(self, frame_idx: int) -> float:
        """True wallclock timestamp for a sampled frame.

        Prefers the per-sample map provided by the benchmark loader
        (the real seconds in the original video) and falls back to
        `frame_idx / fps` for in-memory sequences (game replays,
        browser recordings) that were not pre-sampled.
        """
        if 0 <= frame_idx < len(self._sample_timestamps):
            return float(self._sample_timestamps[frame_idx])
        fps = self._video._fps
        return round(frame_idx / fps, 3) if fps > 0 else 0.0

    def sample_indices(
        self,
        start: int = 0,
        end: int | None = None,
        n: int = 5,
    ) -> list[int]:
        """Uniformly sample frame indices from a range."""
        end = end if end is not None else self.total_frames - 1
        end = min(end, self.total_frames - 1)
        start = max(0, start)
        if end <= start:
            return [start] if start < self.total_frames else []
        n = min(n, end - start + 1)
        step = max(1, (end - start) // (n - 1)) if n > 1 else 1
        indices = []
        for i in range(start, end + 1, step):
            indices.append(i)
            if len(indices) >= n:
                break
        if indices and indices[-1] != end and len(indices) < n:
            indices.append(end)
        return indices


# ── Handler implementations ──────────────────────────────────────────

def _h_track_object(
    state: _VideoVisualState,
    *,
    query: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    sample_every: int | None = None,
) -> dict:
    end = end_frame if end_frame is not None else state.total_frames - 1
    end = min(end, state.total_frames - 1)
    start_frame = max(0, start_frame)

    if sample_every is None:
        total_range = end - start_frame + 1
        sample_every = max(1, total_range // 8)

    query_lower = query.lower()
    track_points: list[dict] = []
    last_bbox = None

    for idx in range(start_frame, end + 1, sample_every):
        vs = state.get_visual(idx)
        if vs is None:
            continue
        dets = vs.detect()

        best_match = None
        best_score = 0.0
        for d in dets:
            label_lower = d.label.lower()
            score = 0.0
            if query_lower in label_lower:
                score = 1.0
            elif query_lower == label_lower:
                score = 1.5
            else:
                query_words = set(query_lower.split())
                label_words = set(label_lower.split())
                overlap = len(query_words & label_words)
                if query_words:
                    score = 0.5 * overlap / len(query_words)
            if score > best_score:
                best_score = score
                best_match = d

        if best_match and best_score > 0.2:
            bx, by, bw, bh = best_match.bbox
            motion = None
            if last_bbox:
                dx = (bx + bw // 2) - (last_bbox[0] + last_bbox[2] // 2)
                dy = (by + bh // 2) - (last_bbox[1] + last_bbox[3] // 2)
                motion = {"dx": dx, "dy": dy, "dist": round(math.sqrt(dx*dx + dy*dy), 1)}

            track_points.append({
                "frame": idx,
                "timestamp": round(idx / state.fps, 2) if state.fps > 0 else 0,
                "found": True,
                "label": best_match.label,
                "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
                "confidence": round(best_match.confidence, 3),
                "motion_from_prev": motion,
            })
            last_bbox = best_match.bbox
        else:
            track_points.append({
                "frame": idx,
                "timestamp": round(idx / state.fps, 2) if state.fps > 0 else 0,
                "found": False,
            })

    found_count = sum(1 for tp in track_points if tp["found"])
    total_motion = 0.0
    for tp in track_points:
        if tp.get("found") and tp.get("motion_from_prev"):
            total_motion += tp["motion_from_prev"]["dist"]

    return {
        "query": query,
        "track_points": track_points,
        "frames_checked": len(track_points),
        "frames_found": found_count,
        "total_motion_px": round(total_motion, 1),
        "motion_summary": (
            "stationary" if total_motion < 5
            else "minor movement" if total_motion < 50
            else "significant movement" if total_motion < 200
            else "large movement"
        ),
    }


def _h_summarize_clip(
    state: _VideoVisualState,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    num_samples: int = 5,
) -> dict:
    indices = state.sample_indices(start_frame, end_frame, num_samples)
    timeline: list[dict] = []
    prev_labels: set[str] | None = None

    for idx in indices:
        vs = state.get_visual(idx)
        if vs is None:
            continue
        dets = vs.detect()

        labels = {d.label for d in dets}
        type_counts = {}
        for d in dets:
            type_counts[d.element_type] = type_counts.get(d.element_type, 0) + 1

        entry: dict[str, Any] = {
            "frame": idx,
            "timestamp": round(idx / state.fps, 2) if state.fps > 0 else 0,
            "element_count": len(dets),
            "type_counts": type_counts,
            "top_labels": [d.label for d in dets[:8]],
        }

        if prev_labels is not None:
            added = labels - prev_labels
            removed = prev_labels - labels
            if added:
                entry["added"] = list(added)[:5]
            if removed:
                entry["removed"] = list(removed)[:5]
            entry["change_magnitude"] = (
                "none" if not added and not removed
                else "minor" if len(added) + len(removed) <= 3
                else "moderate" if len(added) + len(removed) <= 8
                else "major"
            )

        timeline.append(entry)
        prev_labels = labels

    scene_changes = sum(
        1 for t in timeline
        if t.get("change_magnitude") in ("moderate", "major")
    )

    return {
        "frame_range": [start_frame, end_frame or state.total_frames - 1],
        "samples": len(timeline),
        "timeline": timeline,
        "scene_changes": scene_changes,
    }


def _h_find_moment(
    state: _VideoVisualState,
    *,
    query: str,
    event_type: str = "any",
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    end = end_frame if end_frame is not None else state.total_frames - 1
    indices = state.sample_indices(start_frame, end, n=12)
    query_lower = query.lower()

    frame_matches: list[dict] = []
    prev_found = None

    for idx in indices:
        vs = state.get_visual(idx)
        if vs is None:
            continue
        dets = vs.detect()

        matched = [
            d for d in dets
            if query_lower in d.label.lower()
            or any(w in d.label.lower() for w in query_lower.split() if len(w) > 2)
        ]
        is_present = len(matched) > 0

        event_detected = False
        if event_type == "appears" and is_present and prev_found is False:
            event_detected = True
        elif event_type == "disappears" and not is_present and prev_found is True:
            event_detected = True
        elif event_type == "changes" and prev_found is not None and is_present != prev_found:
            event_detected = True
        elif event_type == "any" and is_present:
            event_detected = True

        if event_detected or is_present:
            frame_matches.append({
                "frame": idx,
                "timestamp": round(idx / state.fps, 2) if state.fps > 0 else 0,
                "event_detected": event_detected,
                "matches_found": len(matched),
                "match_labels": [m.label for m in matched[:5]],
            })

        prev_found = is_present

    best = None
    for fm in frame_matches:
        if fm["event_detected"]:
            best = fm
            break
    if best is None and frame_matches:
        best = frame_matches[0]

    return {
        "query": query,
        "event_type": event_type,
        "best_match": best,
        "all_matches": frame_matches,
        "found": best is not None,
    }


def _h_detect_activity(
    state: _VideoVisualState,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    indices = state.sample_indices(start_frame, end_frame, n=6)

    element_counts = []
    all_labels: list[set[str]] = []
    type_distributions: list[dict[str, int]] = []

    for idx in indices:
        vs = state.get_visual(idx)
        if vs is None:
            continue
        dets = vs.detect()
        element_counts.append(len(dets))
        all_labels.append({d.label for d in dets})
        tc: dict[str, int] = {}
        for d in dets:
            tc[d.element_type] = tc.get(d.element_type, 0) + 1
        type_distributions.append(tc)

    if len(all_labels) < 2:
        return {
            "activity": "insufficient_data",
            "confidence": 0.0,
            "frames_analysed": len(indices),
        }

    churn_scores = []
    for i in range(1, len(all_labels)):
        added = len(all_labels[i] - all_labels[i-1])
        removed = len(all_labels[i-1] - all_labels[i])
        total = max(1, len(all_labels[i] | all_labels[i-1]))
        churn_scores.append((added + removed) / total)

    mean_churn = sum(churn_scores) / len(churn_scores) if churn_scores else 0
    count_variance = (
        np.var(element_counts).item() if len(element_counts) > 1 else 0
    )

    stable_labels = set.intersection(*all_labels) if all_labels else set()
    total_unique = set.union(*all_labels) if all_labels else set()

    if mean_churn < 0.05:
        activity = "idle"
    elif mean_churn < 0.15:
        text_heavy = all(
            td.get("text", 0) > td.get("icon", 0) for td in type_distributions
        )
        activity = "reading/typing" if text_heavy else "minor_interaction"
    elif mean_churn < 0.4:
        activity = "navigation" if count_variance > 20 else "scrolling"
    else:
        activity = "rapid_change"

    combined_label_text = " ".join(
        " ".join(labels) for labels in all_labels
    ).lower()
    if any(w in combined_label_text for w in ("score", "level", "game", "lives")):
        activity = "game_action"

    return {
        "activity": activity,
        "confidence": round(1.0 - mean_churn * 0.5, 3),
        "mean_churn": round(mean_churn, 4),
        "element_count_range": [
            min(element_counts) if element_counts else 0,
            max(element_counts) if element_counts else 0,
        ],
        "stable_elements": len(stable_labels),
        "total_unique_elements": len(total_unique),
        "frames_analysed": len(indices),
    }


def _h_compare_elements(
    state: _VideoVisualState,
    *,
    frame_a: int,
    frame_b: int,
) -> dict:
    vs_a = state.get_visual(frame_a)
    vs_b = state.get_visual(frame_b)
    if vs_a is None:
        return {"error": f"Frame {frame_a} out of range"}
    if vs_b is None:
        return {"error": f"Frame {frame_b} out of range"}

    dets_a = vs_a.detect()
    dets_b = vs_b.detect()

    labels_a = {d.label: d for d in dets_a}
    labels_b = {d.label: d for d in dets_b}

    set_a = set(labels_a.keys())
    set_b = set(labels_b.keys())

    added = set_b - set_a
    removed = set_a - set_b
    common = set_a & set_b

    moved = []
    for label in common:
        da = labels_a[label]
        db = labels_b[label]
        ax, ay = da.bbox[0] + da.bbox[2] // 2, da.bbox[1] + da.bbox[3] // 2
        bx, by = db.bbox[0] + db.bbox[2] // 2, db.bbox[1] + db.bbox[3] // 2
        dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        if dist > 10:
            moved.append({
                "label": label,
                "bbox_a": {"x": da.bbox[0], "y": da.bbox[1], "w": da.bbox[2], "h": da.bbox[3]},
                "bbox_b": {"x": db.bbox[0], "y": db.bbox[1], "w": db.bbox[2], "h": db.bbox[3]},
                "distance_px": round(dist, 1),
            })

    return {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "elements_in_a": len(dets_a),
        "elements_in_b": len(dets_b),
        "added": list(added)[:10],
        "removed": list(removed)[:10],
        "moved": moved[:10],
        "unchanged": len(common) - len(moved),
        "summary": (
            "identical" if not added and not removed and not moved
            else "minor changes" if len(added) + len(removed) + len(moved) <= 3
            else "moderate changes" if len(added) + len(removed) + len(moved) <= 8
            else "major changes"
        ),
    }


def _h_detect_objects_at_frame(
    state: _VideoVisualState,
    *,
    frame_index: int,
    confidence_threshold: float = 0.3,
    max_elements: int = 30,
) -> dict:
    vs = state.get_visual(frame_index)
    if vs is None:
        return {"error": f"Frame {frame_index} out of range [0, {state.total_frames})"}

    # On cinematic/natural video the UI-icon captioner fires 9-17%
    # "subtitle"-style hallucinations on frames with no text overlays
    # at all.  Clamp the floor to 0.3 so those false positives get
    # dropped unless the caller explicitly asks to see them.
    effective_conf = max(confidence_threshold, 0.3) \
        if not state._prefer_gdino and confidence_threshold < 0.3 \
        else confidence_threshold

    dets = vs.detect(
        confidence_threshold=effective_conf,
        max_elements=max_elements,
    )

    elements = []
    for d in dets:
        elements.append({
            "index": d.index,
            "type": d.element_type,
            "label": d.label,
            "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
            "interactable": d.interactable,
            "confidence": round(d.confidence, 3),
        })

    frame = state.video.get_frame(frame_index)
    img_w, img_h = frame.size if frame else (0, 0)

    return {
        "frame_index": frame_index,
        "timestamp": round(state.frame_timestamp(frame_index), 2),
        "elements": elements,
        "count": len(elements),
        "image_size": {"w": img_w, "h": img_h},
        "backend": "gdino" if state._prefer_gdino else "omniparser",
    }


def _h_describe_frame(
    state: _VideoVisualState,
    *,
    frame_index: int,
    focus: str | None = None,
) -> dict:
    """Return a natural-language description of a specific frame.

    Calls the same VLM the outer tool loop is using so we don't need
    a second model.  When no describer is wired in (e.g. unit tests
    or offline replay), falls back to a structural summary built from
    `detect_objects_at_frame`.
    """
    frame = state.video.get_frame(frame_index)
    if frame is None:
        return {
            "error": f"Frame {frame_index} out of range "
                     f"[0, {state.total_frames})",
        }
    ts = round(state.frame_timestamp(frame_index), 2)

    if state._vlm_describer is not None:
        try:
            description = state._vlm_describer(frame, focus=focus)
        except Exception as exc:
            logger.warning(
                "describe_frame: VLM describer failed (%s); "
                "falling back to structural summary", exc,
            )
            description = _structural_summary(state, frame_index)
            return {
                "frame_index": frame_index,
                "timestamp": ts,
                "description": description,
                "backend": "fallback",
                "warning": str(exc),
            }
        return {
            "frame_index": frame_index,
            "timestamp": ts,
            "description": description,
            "backend": "vlm",
        }

    return {
        "frame_index": frame_index,
        "timestamp": ts,
        "description": _structural_summary(state, frame_index),
        "backend": "fallback",
    }


def _structural_summary(state: _VideoVisualState, frame_index: int) -> str:
    """Offline fallback description when no VLM describer is configured."""
    vs = state.get_visual(frame_index)
    if vs is None:
        return "(frame out of range)"
    dets = vs.detect(confidence_threshold=0.4, max_elements=10)
    if not dets:
        return "(no high-confidence detections)"
    parts = [f"{d.element_type}:{d.label}" for d in dets[:6]]
    return "frame contains " + ", ".join(parts)


def _frame_to_data_url(frame: Image.Image, max_side: int = 640) -> str:
    """Encode a PIL frame as a PNG data: URL suitable for OpenAI vision."""
    img = frame.convert("RGB")
    w, h = img.size
    long = max(w, h)
    if max_side and long > max_side:
        scale = max_side / long
        img = img.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_openai_describer(
    *,
    client,
    model: str,
    max_side: int = 640,
    max_tokens: int = 220,
) -> Callable[..., str]:
    """Build a `describe_frame` handler that calls an OpenAI vision model.

    The returned callable has signature ``(frame, focus=None) -> str``
    and asks the model for a concise factual description (no subtitle
    OCR, no role-playing).  Designed to be cheap: one API call per
    invocation, ~200 tokens back.
    """
    def _describe(frame: Image.Image, focus: str | None = None) -> str:
        prompt = (
            "You are a careful scene annotator for video-QA.  Describe "
            "this frame in at most 5 short sentences.\n\n"
            "RULES:\n"
            " - Focus on PEOPLE (age/gender/clothing/visible facial "
            "features), their ACTIONS, the SETTING, and any salient "
            "OBJECTS.\n"
            " - If text is visible in the frame, quote it ONLY if you "
            "are sure it is really there.  Never invent subtitles, "
            "chat bubbles, or captions that are not clearly present.\n"
            " - Say 'unclear' if something is off-screen, blurry, or "
            "you can't tell.  Do not guess identities.\n"
            " - Do not describe camera / cinematography choices."
        )
        if focus:
            prompt += f"\n\nEXTRA QUESTION (answer in the last sentence): {focus}"

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _frame_to_data_url(frame, max_side),
                                "detail": "high",
                            },
                        },
                    ],
                }],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"describe_frame VLM call failed: {exc}"
            ) from exc

        txt = (resp.choices[0].message.content or "").strip()
        return txt or "(empty description)"

    return _describe


# ── Public: build combined registry ──────────────────────────────────

def build_video_visual_registry(
    frames: list[Image.Image | np.ndarray] | None = None,
    video_path: str | None = None,
    fps: float = 1.0,
    current_index: int = 0,
    *,
    prefer_gdino: bool = False,
    vlm_describer: Callable[..., str] | None = None,
    sample_timestamps: list[float] | None = None,
    include_reasoning: bool = True,
) -> ToolRegistry:
    """Create a ToolRegistry with video nav + visual + cross-frame tools.

    This is the most complete registry, combining:
      - Video navigation tools (get_frame, sample_frames, compare_frames,
        detect_scene_changes, temporal_navigate, etc.)
      - Single-frame visual tools (detect_objects, describe_region,
        visual_search, etc.) bound to the current frame
      - Cross-frame visual tools (track_object, summarize_clip,
        find_moment, detect_activity, compare_elements,
        detect_objects_at_frame)

    Parameters
    ----------
    frames : list[Image | ndarray], optional
        Pre-decoded frames.
    video_path : str, optional
        Path to video file (decoded lazily).
    fps : float
        Frames per second.
    current_index : int
        Initial frame position.

    Returns
    -------
    ToolRegistry
        Combined registry with all tool types.
    """
    vv_state = _VideoVisualState(
        frames=frames, video_path=video_path,
        fps=fps, current_index=current_index,
        prefer_gdino=prefer_gdino,
        vlm_describer=vlm_describer,
        sample_timestamps=sample_timestamps,
    )

    video_reg = build_video_registry(
        frames=frames, video_path=video_path,
        fps=fps, current_index=current_index,
    )

    current_frame = None
    if frames and 0 <= current_index < len(frames):
        f = frames[current_index]
        current_frame = Image.fromarray(f) if isinstance(f, np.ndarray) else f
    elif video_path:
        decoded = _decode_video(video_path)
        if decoded and 0 <= current_index < len(decoded):
            current_frame = decoded[current_index]

    if current_frame is not None:
        visual_reg = build_visual_registry(
            current_frame,
            prefer_gdino=prefer_gdino,
            include_reasoning=False,
        )
    else:
        visual_reg = ToolRegistry(domain="visual")

    combined = video_reg.merge(visual_reg)

    cross_reg = ToolRegistry(domain="video_visual")
    cross_reg.register(TOOL_TRACK_OBJECT, lambda **kw: _h_track_object(vv_state, **kw))
    cross_reg.register(TOOL_SUMMARIZE_CLIP, lambda **kw: _h_summarize_clip(vv_state, **kw))
    cross_reg.register(TOOL_FIND_MOMENT, lambda **kw: _h_find_moment(vv_state, **kw))
    cross_reg.register(TOOL_DETECT_ACTIVITY, lambda **kw: _h_detect_activity(vv_state, **kw))
    cross_reg.register(TOOL_COMPARE_ELEMENTS, lambda **kw: _h_compare_elements(vv_state, **kw))
    cross_reg.register(TOOL_DETECT_OBJECTS_AT_FRAME, lambda **kw: _h_detect_objects_at_frame(vv_state, **kw))
    cross_reg.register(TOOL_DESCRIBE_FRAME, lambda **kw: _h_describe_frame(vv_state, **kw))

    final = combined.merge(cross_reg)

    if include_reasoning:
        from .tools_reasoning import build_reasoning_registry

        reasoning_reg, derivation_log = build_reasoning_registry()
        final = final.merge(reasoning_reg)
        final.derivation_log = derivation_log  # type: ignore[attr-defined]

    return final
