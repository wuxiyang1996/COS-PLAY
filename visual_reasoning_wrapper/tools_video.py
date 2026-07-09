"""Video-domain tool implementations for multi-frame visual reasoning.

Provides tools that let the VLM navigate and query video temporally
instead of receiving a single frame.  The VLM sees the current frame,
then can request other frames, temporal diffs, text extraction, or
scene-change boundaries to build multi-hop reasoning chains.

Designed for two use cases:
  1. Interactive environments recorded as frame sequences (game replays,
     browser session recordings) — frames come from the trajectory.
  2. Passive videos (.mp4, .avi) loaded and decoded on demand.

Usage::

    from visual_reasoning_wrapper.tools_video import build_video_registry

    # From a list of PIL frames (game replay, BrowserGym recording)
    registry = build_video_registry(frames=pil_frames, fps=2.0)

    # From a video file (if cv2 / decord available)
    registry = build_video_registry(video_path="demo.mp4")
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from vlm_wrapper.tools import (
    TOOL_GET_STATE_FLAGS,
    TOOL_LIST_ENTITIES,
    TOOL_LIST_VALID_ACTIONS,
    TOOL_QUERY_ENTITY_POS,
    ToolDef,
    ToolRegistry,
)

logger = logging.getLogger(__name__)


# ── Video-specific tool definitions ───────────────────────────────────

TOOL_GET_FRAME = ToolDef(
    name="get_frame",
    description=(
        "Retrieve a specific frame from the video by index or timestamp. "
        "Returns the frame as a base64 image plus metadata (dimensions, "
        "timestamp, frame index). Use to inspect moments before/after "
        "the current observation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "Frame index (0-based). Mutually exclusive with 'timestamp'.",
            },
            "timestamp": {
                "type": "number",
                "description": "Timestamp in seconds. Mutually exclusive with 'index'.",
            },
        },
        "required": [],
    },
    domain="video",
)

TOOL_SAMPLE_FRAMES = ToolDef(
    name="sample_frames",
    description=(
        "Uniformly sample N frames across the video (or a time range). "
        "Returns frame indices and timestamps. Use for getting an "
        "overview of the full video before drilling into specifics."
    ),
    parameters={
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "Number of frames to sample. Default 8.",
            },
            "start_sec": {
                "type": "number",
                "description": "Start of range in seconds. Default 0.",
            },
            "end_sec": {
                "type": "number",
                "description": "End of range in seconds. Default end of video.",
            },
        },
        "required": [],
    },
    domain="video",
)

TOOL_COMPARE_FRAMES = ToolDef(
    name="compare_frames",
    description=(
        "Compare two frames and return a structural diff: which regions "
        "changed, pixel difference magnitude, and a brief description. "
        "Useful for detecting what happened between two moments."
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
    domain="video",
)

TOOL_DETECT_CHANGES = ToolDef(
    name="detect_scene_changes",
    description=(
        "Find scene change boundaries in the video (or a range). Returns "
        "a list of frame indices where significant visual changes occur. "
        "Useful for segmenting the video into distinct scenes/events."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_idx": {
                "type": "integer",
                "description": "Start frame index. Default 0.",
            },
            "end_idx": {
                "type": "integer",
                "description": "End frame index. Default last frame.",
            },
            "threshold": {
                "type": "number",
                "description": "Sensitivity threshold (0-1). Lower = more changes detected. Default 0.15.",
            },
        },
        "required": [],
    },
    domain="video",
)

TOOL_GET_VIDEO_INFO = ToolDef(
    name="get_video_info",
    description=(
        "Get video metadata: total frames, duration, FPS, resolution, "
        "and current playback position."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="video",
)

TOOL_READ_TEXT_IN_FRAME = ToolDef(
    name="read_text_in_frame",
    description=(
        "Extract visible text from a specific frame using OCR. Returns "
        "detected text regions with their bounding boxes and content. "
        "Useful for reading UI text, subtitles, game scores, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "frame_index": {
                "type": "integer",
                "description": "Frame to run OCR on. Default: current frame.",
            },
            "region": {
                "type": "object",
                "description": "Optional region of interest: {x, y, w, h} in pixels.",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "w": {"type": "integer"},
                    "h": {"type": "integer"},
                },
            },
        },
        "required": [],
    },
    domain="video",
)

TOOL_TEMPORAL_NAVIGATE = ToolDef(
    name="temporal_navigate",
    description=(
        "Move the current observation to a different point in time. "
        "Supports absolute index, relative offset (+N/-N frames), "
        "or named positions ('start', 'end', 'middle')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Where to go: frame index (e.g. '42'), relative offset ('+5', '-3'), or named ('start', 'end', 'middle').",
            },
        },
        "required": ["target"],
    },
    domain="video",
)


# ── Video state ──────────────────────────────────────────────────────

class _VideoState:
    """Holds the video frames (decoded or as a list) and current position."""

    def __init__(
        self,
        frames: list[Image.Image | np.ndarray] | None = None,
        video_path: str | None = None,
        fps: float = 1.0,
        current_index: int = 0,
    ):
        self._raw_frames = frames
        self._video_path = video_path
        self._fps = fps
        self.current_index = current_index
        self._decoded: list[Image.Image] | None = None

    @property
    def frames(self) -> list[Image.Image]:
        if self._decoded is not None:
            return self._decoded
        if self._raw_frames is not None:
            self._decoded = [
                Image.fromarray(f) if isinstance(f, np.ndarray) else f
                for f in self._raw_frames
            ]
            return self._decoded
        if self._video_path:
            self._decoded = _decode_video(self._video_path)
            return self._decoded
        self._decoded = []
        return self._decoded

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.total_frames / self._fps if self._fps > 0 else 0.0

    def get_frame(self, idx: int) -> Image.Image | None:
        if 0 <= idx < self.total_frames:
            return self.frames[idx]
        return None

    def idx_from_timestamp(self, ts: float) -> int:
        return min(int(ts * self._fps), self.total_frames - 1)


def _decode_video(path: str) -> list[Image.Image]:
    """Attempt to decode a video file. Tries cv2 first, then imageio."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames
    except ImportError:
        pass

    try:
        import imageio.v3 as iio
        raw = iio.imread(path, plugin="pyav")
        return [Image.fromarray(f) for f in raw]
    except ImportError:
        pass

    logger.warning("No video decoder available (install opencv-python or imageio[pyav])")
    return []


# ── Handler implementations ──────────────────────────────────────────

def _h_get_frame(state: _VideoState, *, index: int | None = None, timestamp: float | None = None) -> dict:
    if timestamp is not None:
        idx = state.idx_from_timestamp(timestamp)
    elif index is not None:
        idx = index
    else:
        idx = state.current_index

    frame = state.get_frame(idx)
    if frame is None:
        return {"found": False, "message": f"Frame {idx} out of range [0, {state.total_frames})"}
    w, h = frame.size
    return {
        "found": True,
        "index": idx,
        "timestamp": round(idx / state._fps, 2) if state._fps > 0 else 0,
        "width": w,
        "height": h,
        "frame_hash": hashlib.md5(_frame_bytes(frame)).hexdigest()[:12],
    }


def _h_sample_frames(
    state: _VideoState,
    *,
    n: int = 8,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> dict:
    start_idx = state.idx_from_timestamp(start_sec)
    end_idx = state.idx_from_timestamp(end_sec) if end_sec else state.total_frames - 1
    if end_idx <= start_idx:
        end_idx = start_idx + 1

    n = min(n, end_idx - start_idx + 1)
    step = max(1, (end_idx - start_idx) // n)
    sampled = []
    for i in range(start_idx, end_idx + 1, step):
        if len(sampled) >= n:
            break
        frame = state.get_frame(i)
        if frame:
            sampled.append({
                "index": i,
                "timestamp": round(i / state._fps, 2) if state._fps > 0 else 0,
            })
    return {"sampled": sampled, "count": len(sampled), "total_frames": state.total_frames}


def _h_compare_frames(state: _VideoState, *, frame_a: int, frame_b: int) -> dict:
    fa = state.get_frame(frame_a)
    fb = state.get_frame(frame_b)
    if fa is None or fb is None:
        return {"error": "One or both frame indices out of range"}

    arr_a = np.array(fa.resize((256, 256)))
    arr_b = np.array(fb.resize((256, 256)))

    diff = np.abs(arr_a.astype(float) - arr_b.astype(float))
    mean_diff = float(diff.mean()) / 255.0
    max_diff = float(diff.max()) / 255.0

    h, w = 256, 256
    quadrants = {}
    for qname, (rs, re, cs, ce) in [
        ("top_left", (0, h // 2, 0, w // 2)),
        ("top_right", (0, h // 2, w // 2, w)),
        ("bottom_left", (h // 2, h, 0, w // 2)),
        ("bottom_right", (h // 2, h, w // 2, w)),
    ]:
        q_diff = diff[rs:re, cs:ce].mean() / 255.0
        quadrants[qname] = round(q_diff, 4)

    changed_pct = float((diff.max(axis=-1) > 25).mean())

    return {
        "mean_difference": round(mean_diff, 4),
        "max_difference": round(max_diff, 4),
        "changed_pixel_pct": round(changed_pct, 4),
        "quadrant_diffs": quadrants,
        "time_delta": round(abs(frame_b - frame_a) / state._fps, 2) if state._fps > 0 else 0,
        "summary": _diff_summary(mean_diff, changed_pct, quadrants),
    }


def _h_detect_changes(
    state: _VideoState,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    threshold: float = 0.15,
) -> dict:
    end = end_idx if end_idx is not None else state.total_frames - 1
    changes: list[dict] = []

    prev_arr = None
    for i in range(start_idx, min(end + 1, state.total_frames)):
        frame = state.get_frame(i)
        if frame is None:
            continue
        arr = np.array(frame.resize((128, 128))).astype(float)
        if prev_arr is not None:
            diff_score = np.abs(arr - prev_arr).mean() / 255.0
            if diff_score > threshold:
                changes.append({
                    "frame_index": i,
                    "timestamp": round(i / state._fps, 2) if state._fps > 0 else 0,
                    "diff_score": round(diff_score, 4),
                })
        prev_arr = arr

    return {"changes": changes, "count": len(changes)}


def _h_get_video_info(state: _VideoState) -> dict:
    frame = state.get_frame(0)
    w, h = frame.size if frame else (0, 0)
    return {
        "total_frames": state.total_frames,
        "fps": state._fps,
        "duration_seconds": round(state.duration, 2),
        "width": w,
        "height": h,
        "current_index": state.current_index,
        "current_timestamp": round(state.current_index / state._fps, 2) if state._fps > 0 else 0,
    }


def _h_read_text_in_frame(
    state: _VideoState,
    *,
    frame_index: int | None = None,
    region: dict | None = None,
) -> dict:
    idx = frame_index if frame_index is not None else state.current_index
    frame = state.get_frame(idx)
    if frame is None:
        return {"error": f"Frame {idx} out of range"}

    if region:
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", frame.size[0]), region.get("h", frame.size[1])
        frame = frame.crop((x, y, x + w, y + h))

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(frame))
        texts = []
        for bbox_pts, text, conf in results:
            flat = [int(p) for pt in bbox_pts for p in pt]
            x_min, y_min = min(flat[::2]), min(flat[1::2])
            x_max, y_max = max(flat[::2]), max(flat[1::2])
            texts.append({
                "text": text,
                "confidence": round(conf, 3),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            })
        return {"frame_index": idx, "texts": texts, "count": len(texts)}
    except ImportError:
        pass

    try:
        import pytesseract
        text = pytesseract.image_to_string(frame)
        return {
            "frame_index": idx,
            "full_text": text.strip(),
            "engine": "tesseract",
            "note": "Install easyocr for bbox-level results.",
        }
    except ImportError:
        pass

    return {
        "frame_index": idx,
        "error": "No OCR engine available. Install easyocr or pytesseract.",
    }


def _h_temporal_navigate(state: _VideoState, *, target: str) -> dict:
    target = target.strip()
    if target == "start":
        state.current_index = 0
    elif target == "end":
        state.current_index = max(0, state.total_frames - 1)
    elif target == "middle":
        state.current_index = state.total_frames // 2
    elif target.startswith("+") or target.startswith("-"):
        offset = int(target)
        state.current_index = max(0, min(state.current_index + offset, state.total_frames - 1))
    else:
        state.current_index = max(0, min(int(target), state.total_frames - 1))

    return {
        "current_index": state.current_index,
        "timestamp": round(state.current_index / state._fps, 2) if state._fps > 0 else 0,
        "total_frames": state.total_frames,
    }


def _h_list_valid_actions_video(state: _VideoState) -> dict:
    actions = [
        {"action": "temporal_navigate('+1')", "description": "Next frame"},
        {"action": "temporal_navigate('-1')", "description": "Previous frame"},
        {"action": "sample_frames(n=8)", "description": "Overview of video"},
        {"action": "detect_scene_changes()", "description": "Find key moments"},
        {"action": "read_text_in_frame()", "description": "OCR current frame"},
        {"action": "compare_frames(a, b)", "description": "Diff two frames"},
    ]
    return {"actions": actions, "count": len(actions)}


# ── Helpers ──────────────────────────────────────────────────────────

def _frame_bytes(frame: Image.Image) -> bytes:
    buf = io.BytesIO()
    frame.resize((64, 64)).save(buf, format="PNG")
    return buf.getvalue()


def _diff_summary(mean_diff: float, changed_pct: float, quadrants: dict) -> str:
    if mean_diff < 0.02:
        return "Frames are nearly identical"
    if mean_diff < 0.08:
        most_changed = max(quadrants, key=quadrants.get)  # type: ignore[arg-type]
        return f"Minor changes, mostly in {most_changed} ({changed_pct:.0%} pixels changed)"
    if mean_diff < 0.25:
        return f"Moderate changes across {changed_pct:.0%} of pixels"
    return f"Major scene change — {changed_pct:.0%} of pixels differ significantly"


# ── Public: build a registry ─────────────────────────────────────────

def build_video_registry(
    frames: list[Image.Image | np.ndarray] | None = None,
    video_path: str | None = None,
    fps: float = 1.0,
    current_index: int = 0,
) -> ToolRegistry:
    """Create a ToolRegistry with all video tools.

    Parameters
    ----------
    frames : list[Image | ndarray], optional
        Pre-decoded frames (from game replay, browser recording, etc.).
    video_path : str, optional
        Path to a video file. Decoded lazily on first tool call.
    fps : float
        Frames per second (for timestamp calculation).
    current_index : int
        Initial frame position.

    Returns
    -------
    ToolRegistry
    """
    state = _VideoState(frames=frames, video_path=video_path, fps=fps, current_index=current_index)
    reg = ToolRegistry(domain="video")

    reg.register(TOOL_GET_FRAME, lambda **kw: _h_get_frame(state, **kw))
    reg.register(TOOL_SAMPLE_FRAMES, lambda **kw: _h_sample_frames(state, **kw))
    reg.register(TOOL_COMPARE_FRAMES, lambda **kw: _h_compare_frames(state, **kw))
    reg.register(TOOL_DETECT_CHANGES, lambda **kw: _h_detect_changes(state, **kw))
    reg.register(TOOL_GET_VIDEO_INFO, lambda **kw: _h_get_video_info(state))
    reg.register(TOOL_READ_TEXT_IN_FRAME, lambda **kw: _h_read_text_in_frame(state, **kw))
    reg.register(TOOL_TEMPORAL_NAVIGATE, lambda **kw: _h_temporal_navigate(state, **kw))
    reg.register(TOOL_LIST_VALID_ACTIONS, lambda **kw: _h_list_valid_actions_video(state))

    return reg
