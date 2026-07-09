"""Vision-model-backed tool implementations for visual understanding.

Provides tools that let the VLM delegate to specialised vision models
(OmniParser-v2, Florence-2, CLIP, OCR) via tool calling instead of
relying on its own pixel-level reasoning.  Each tool wraps a real
model invocation and returns structured results.

Two registries are available:

  * ``build_visual_registry(image)`` — tools that operate on a single frame.
  * Tools can also be merged with the video registry for temporal+visual
    understanding (see ``tools_video_visual`` in this package).

Designed for multi-hop reasoning: the VLM sees a screenshot, calls
``detect_objects`` to get precise elements, then ``describe_region`` to
caption an ambiguous area, then ``spatial_query`` to verify a relation —
building a chain of grounded evidence before producing the final schema.

Usage::

    from visual_reasoning_wrapper.tools_visual import build_visual_registry

    registry = build_visual_registry(pil_image)
    # merge with video tools if needed:
    # combined = video_registry.merge(visual_registry)
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

from vlm_wrapper.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


# ── Tool definitions (OpenAI function-calling schema) ─────────────────

TOOL_DETECT_OBJECTS = ToolDef(
    name="detect_objects",
    description=(
        "Run an object/UI-element detector on the current frame. Returns "
        "a list of detected elements with bounding boxes, labels, types "
        "(icon/text), confidence scores, and interactability flags. "
        "Uses OmniParser-v2 (YOLO + OCR + Florence-2) for precise "
        "grounding. Much more accurate than eyeballing from pixels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum detection confidence (0-1). Default 0.05.",
            },
            "max_elements": {
                "type": "integer",
                "description": "Max elements to return. Default 30.",
            },
        },
        "required": [],
    },
    domain="visual",
)

TOOL_DESCRIBE_REGION = ToolDef(
    name="describe_region",
    description=(
        "Generate a natural-language caption for a specific rectangular "
        "region of the frame. Crops the region and runs Florence-2 for "
        "a semantic description. Use when you can see something but "
        "cannot identify what it is."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge in pixels."},
            "y": {"type": "integer", "description": "Top edge in pixels."},
            "w": {"type": "integer", "description": "Width in pixels."},
            "h": {"type": "integer", "description": "Height in pixels."},
        },
        "required": ["x", "y", "w", "h"],
    },
    domain="visual",
)

TOOL_VISUAL_SEARCH = ToolDef(
    name="visual_search",
    description=(
        "Search the frame for elements matching a text query. Runs the "
        "detector then filters results by text similarity to the query. "
        "Returns matching elements ranked by relevance. Example queries: "
        "'submit button', 'red icon', 'price label'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what to find.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results. Default 5.",
            },
        },
        "required": ["query"],
    },
    domain="visual",
)

TOOL_COUNT_OBJECTS = ToolDef(
    name="count_objects",
    description=(
        "Count elements of a specific type or matching a description in "
        "the current frame. Runs the detector and counts matches. "
        "Supports filtering by type ('icon', 'text', 'all') and/or "
        "a text query for more specific counting."
    ),
    parameters={
        "type": "object",
        "properties": {
            "element_type": {
                "type": "string",
                "enum": ["all", "icon", "text"],
                "description": "Filter by element type. Default 'all'.",
            },
            "query": {
                "type": "string",
                "description": "Optional text filter — only count elements whose label matches.",
            },
        },
        "required": [],
    },
    domain="visual",
)

TOOL_CLASSIFY_SCENE = ToolDef(
    name="classify_scene",
    description=(
        "Classify the current frame's scene type and content. Returns "
        "scene categories (game, browser, desktop, dialog, menu, etc.), "
        "a brief scene description, dominant visual properties, and "
        "counts of detected element types."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="visual",
)

TOOL_SPATIAL_QUERY = ToolDef(
    name="spatial_query",
    description=(
        "Query the spatial relationship between two detected elements "
        "using their ground-truth bounding boxes. Returns distance, "
        "direction, overlap, containment, and alignment information. "
        "More precise than estimating from the screenshot."
    ),
    parameters={
        "type": "object",
        "properties": {
            "element_a": {
                "type": "string",
                "description": "Label or index (e.g. '0', '3') of the first element.",
            },
            "element_b": {
                "type": "string",
                "description": "Label or index of the second element.",
            },
        },
        "required": ["element_a", "element_b"],
    },
    domain="visual",
)

TOOL_MEASURE_DISTANCE = ToolDef(
    name="measure_distance",
    description=(
        "Measure the pixel distance between two points or element "
        "centres. Returns Euclidean distance, horizontal/vertical "
        "offsets, and direction. Use for precise spatial reasoning."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x1": {"type": "integer", "description": "X of first point (pixels)."},
            "y1": {"type": "integer", "description": "Y of first point (pixels)."},
            "x2": {"type": "integer", "description": "X of second point (pixels)."},
            "y2": {"type": "integer", "description": "Y of second point (pixels)."},
        },
        "required": ["x1", "y1", "x2", "y2"],
    },
    domain="visual",
)

TOOL_EXTRACT_COLORS = ToolDef(
    name="extract_colors",
    description=(
        "Extract dominant colours from a region of the frame. Returns "
        "top-K colours as hex codes with their coverage percentages. "
        "Useful for identifying colour-coded elements, status indicators, "
        "or matching visual themes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge (pixels). Default: full frame."},
            "y": {"type": "integer", "description": "Top edge (pixels)."},
            "w": {"type": "integer", "description": "Width (pixels)."},
            "h": {"type": "integer", "description": "Height (pixels)."},
            "top_k": {"type": "integer", "description": "Number of colours. Default 5."},
        },
        "required": [],
    },
    domain="visual",
)

TOOL_READ_TEXT_REGION = ToolDef(
    name="read_text_region",
    description=(
        "Run OCR on a specific region (or the whole frame) and return "
        "structured text with bounding boxes. More precise than the "
        "video-level read_text_in_frame because it also returns "
        "line structure and reading order."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge (pixels). Omit for full frame."},
            "y": {"type": "integer", "description": "Top edge (pixels)."},
            "w": {"type": "integer", "description": "Width (pixels)."},
            "h": {"type": "integer", "description": "Height (pixels)."},
        },
        "required": [],
    },
    domain="visual",
)

TOOL_ZOOM_REGION = ToolDef(
    name="zoom_region",
    description=(
        "Re-observation (PLAN-VISUAL-GROUNDING §4 Option B): crop the "
        "current frame to a region and resend the zoomed crop to yourself "
        "on the next turn.  Use this when ``describe_region`` output is "
        "ambiguous and you need to look at the region with fresh visual "
        "focus.  Returns a base64 PNG of the crop and the crop geometry; "
        "the harness attaches the image to the next user message."
    ),
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Left edge (pixels)."},
            "y": {"type": "integer", "description": "Top edge (pixels)."},
            "w": {"type": "integer", "description": "Width (pixels)."},
            "h": {"type": "integer", "description": "Height (pixels)."},
            "zoom": {
                "type": "number",
                "description": (
                    "Optional upscale factor (1.0–4.0).  Default 2.0.  "
                    "Higher values help with tiny text / fine details."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "Short note on what you are trying to see (logged in "
                    "the evidence chain)."
                ),
            },
        },
        "required": ["x", "y", "w", "h"],
    },
    domain="visual",
)


TOOL_GROUNDED_DETECT = ToolDef(
    name="grounded_detect",
    description=(
        "Detect objects matching a natural-language query in the image. "
        "Uses open-vocabulary grounding (GroundingDINO) — works on ANY "
        "image content: natural scenes, photos, diagrams, synthetic "
        "scenes (synthetic QA, photos), video frames, etc.  Returns bounding boxes, "
        "labels, and confidence scores for every match.  Unlike "
        "detect_objects (which finds UI elements), this finds arbitrary "
        "objects described in plain English.  Example queries: 'red "
        "sphere', 'person sitting on chair', 'largest car', 'blue cube'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to detect.  Use '.' to separate multiple "
                    "categories, e.g. 'red sphere . blue cube . cylinder'."
                ),
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence (0-1). Default 0.15.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum detections to return. Default 20.",
            },
        },
        "required": ["query"],
    },
    domain="visual",
)


# ── Internal state ────────────────────────────────────────────────────

@dataclass
class _DetectedElement:
    """Cached detection result for a single element."""
    index: int
    element_type: str
    label: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    interactable: bool
    confidence: float
    source: str


def _check_gdino() -> bool:
    """Check if GroundingDINO (or groundingdino via autodistill) is available."""
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import groundingdino  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _check_omniparser() -> bool:
    try:
        from .grounding import parse_screen  # noqa: F401
        return True
    except ImportError:
        return False


class _VisualState:
    """Holds the image and lazily-cached detection results.

    Supports two detection backends:
    - **OmniParser-v2** (YOLO + Florence-2 + OCR): accurate for GUI
      screenshots (buttons, icons, text fields).  The default for
      ``domain in ("gymv", "browser", "desktop")``.
    - **GroundingDINO**: open-vocabulary object detection — handles
      natural images (benchmark photos, UI crops, video
      frames with people/cars/animals).  Used for ``image_qa``,
      ``video_qa``, and any domain where OmniParser would fail.

    Both backends produce the same ``_DetectedElement`` list, so every
    downstream tool (spatial_query, visual_search, count_objects, etc.)
    works identically regardless of which backend ran.
    """

    def __init__(self, image: Image.Image, *, prefer_gdino: bool = False):
        self.image = image.convert("RGB")
        self._prefer_gdino = prefer_gdino
        self._detections: list[_DetectedElement] | None = None
        self._gdino_cache: dict[str, list[_DetectedElement]] = {}

    def detect(
        self,
        confidence_threshold: float = 0.05,
        max_elements: int = 30,
    ) -> list[_DetectedElement]:
        """Run detection (cached after first call with default params)."""
        if self._detections is not None:
            filtered = [
                d for d in self._detections
                if d.confidence >= confidence_threshold
            ]
            return filtered[:max_elements]

        elements = self._run_detection(
            confidence_threshold=confidence_threshold,
            max_elements=max_elements,
        )
        self._detections = elements
        return elements

    def _run_detection(
        self,
        confidence_threshold: float,
        max_elements: int,
    ) -> list[_DetectedElement]:
        if self._prefer_gdino and _check_gdino():
            return self._detect_gdino_open(
                confidence_threshold=confidence_threshold,
                max_elements=max_elements,
            )
        if _check_omniparser():
            return self._detect_omniparser(confidence_threshold, max_elements)
        if _check_gdino():
            return self._detect_gdino_open(
                confidence_threshold=confidence_threshold,
                max_elements=max_elements,
            )
        return self._detect_ocr_only()

    def _detect_omniparser(
        self,
        confidence_threshold: float,
        max_elements: int,
    ) -> list[_DetectedElement]:
        from .grounding import parse_screen

        screen_elements = parse_screen(
            self.image,
            box_threshold=confidence_threshold,
            max_elements=max_elements,
        )
        results = []
        for i, el in enumerate(screen_elements):
            results.append(_DetectedElement(
                index=i,
                element_type=el.element_type,
                label=el.label,
                bbox=(el.bbox.x, el.bbox.y, el.bbox.w, el.bbox.h),
                interactable=el.interactable,
                confidence=el.confidence,
                source=el.source,
            ))
        return results

    def _detect_gdino_open(
        self,
        *,
        query: str = "",
        confidence_threshold: float = 0.15,
        max_elements: int = 30,
    ) -> list[_DetectedElement]:
        """Open-vocabulary detection via GroundingDINO.

        When *query* is empty, uses a generic prompt that detects
        salient objects in the image.  When *query* is given, detects
        specifically those objects.
        """
        if not query:
            query = "object . item . thing . person . shape . element"

        cache_key = f"{query}|{confidence_threshold}"
        if cache_key in self._gdino_cache:
            return self._gdino_cache[cache_key][:max_elements]

        results = _run_gdino(
            self.image,
            query=query,
            box_threshold=confidence_threshold,
            max_elements=max_elements,
        )
        self._gdino_cache[cache_key] = results
        return results[:max_elements]

    def grounded_detect(
        self,
        query: str,
        confidence_threshold: float = 0.15,
        max_elements: int = 20,
    ) -> list[_DetectedElement]:
        """Query-driven detection: find objects matching a text description.

        Unlike ``detect()`` (which does an open sweep), this targets
        specific objects — the key capability for multi-hop reasoning
        on natural images.
        """
        cache_key = f"{query}|{confidence_threshold}"
        if cache_key in self._gdino_cache:
            return self._gdino_cache[cache_key][:max_elements]

        results = _run_gdino(
            self.image,
            query=query,
            box_threshold=confidence_threshold,
            max_elements=max_elements,
        )
        self._gdino_cache[cache_key] = results
        return results[:max_elements]

    def _detect_ocr_only(self) -> list[_DetectedElement]:
        """Fallback: OCR-only detection when no detector is available."""
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            raw = reader.readtext(np.array(self.image))
            results = []
            for i, (bbox_pts, text, conf) in enumerate(raw):
                flat = [int(p) for pt in bbox_pts for p in pt]
                x_min = min(flat[::2])
                y_min = min(flat[1::2])
                x_max = max(flat[::2])
                y_max = max(flat[1::2])
                results.append(_DetectedElement(
                    index=i,
                    element_type="text",
                    label=text,
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    interactable=False,
                    confidence=conf,
                    source="easyocr",
                ))
            return results
        except ImportError:
            logger.warning("No detection backend available")
            return []

    def find_by_ref(self, ref: str) -> _DetectedElement | None:
        """Find element by index string ('0', '3') or label substring."""
        dets = self.detect()
        if ref.isdigit():
            idx = int(ref)
            if 0 <= idx < len(dets):
                return dets[idx]
            return None
        ref_lower = ref.lower()
        for d in dets:
            if ref_lower in d.label.lower():
                return d
        return None


# ── GroundingDINO runner ─────────────────────────────────────────────

_gdino_model = None
_gdino_processor = None


def _run_gdino(
    image: Image.Image,
    query: str,
    box_threshold: float = 0.15,
    max_elements: int = 30,
) -> list[_DetectedElement]:
    """Run GroundingDINO on an image with a text query.

    Tries HuggingFace transformers API first (Grounding DINO via
    ``AutoModelForZeroShotObjectDetection``), which supports the
    ``IDEA-Research/grounding-dino-base`` checkpoint.
    """
    global _gdino_model, _gdino_processor

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        if _gdino_model is None:
            model_id = "IDEA-Research/grounding-dino-base"
            logger.info("Loading GroundingDINO from %s ...", model_id)
            _gdino_processor = AutoProcessor.from_pretrained(model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(device)
            logger.info("GroundingDINO loaded on %s", device)

        inputs = _gdino_processor(
            images=image, text=query, return_tensors="pt"
        ).to(_gdino_model.device)

        with torch.no_grad():
            outputs = _gdino_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(_gdino_model.device)

        # transformers renamed this kwarg between 4.40 and 4.50:
        #   4.40–4.49: post_process_grounded_object_detection(..., box_threshold=, text_threshold=)
        #   4.50+   : post_process_grounded_object_detection(..., threshold=, text_threshold=)
        # Detect which signature we have so the adapter works on both.
        import inspect as _inspect
        _sig_params = _inspect.signature(
            _gdino_processor.post_process_grounded_object_detection
        ).parameters
        _thresh_kw = "threshold" if "threshold" in _sig_params else "box_threshold"
        _post_kwargs = {
            _thresh_kw: box_threshold,
            "text_threshold": box_threshold,
            "target_sizes": target_sizes,
        }
        results = _gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            **_post_kwargs,
        )[0]

        elements: list[_DetectedElement] = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        # 4.50+ returns "text_labels"; older versions return "labels".
        labels = results.get("text_labels",
                             results.get("labels",
                                         results.get("text", [])))

        for i, (box, score) in enumerate(zip(boxes, scores)):
            if i >= max_elements:
                break
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            label = labels[i] if i < len(labels) else query
            elements.append(_DetectedElement(
                index=i,
                element_type="object",
                label=str(label).strip(),
                bbox=(int(x1), int(y1), int(w), int(h)),
                interactable=False,
                confidence=float(score),
                source="grounding_dino",
            ))
        return elements

    except ImportError:
        logger.debug("GroundingDINO not available via transformers")
    except Exception as exc:
        logger.warning("GroundingDINO inference failed: %s", exc)

    return []


# ── Handler implementations ──────────────────────────────────────────

def _h_detect_objects(
    state: _VisualState,
    *,
    confidence_threshold: float = 0.05,
    max_elements: int = 30,
) -> dict:
    dets = state.detect(
        confidence_threshold=confidence_threshold,
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
            "source": d.source,
        })
    return {
        "elements": elements,
        "count": len(elements),
        "image_size": {"w": state.image.size[0], "h": state.image.size[1]},
    }


def _h_describe_region(
    state: _VisualState,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> dict:
    img_w, img_h = state.image.size
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    crop = state.image.crop((x, y, x + w, y + h))

    caption = _caption_crop(crop)
    ocr_text = _ocr_crop(crop)

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "caption": caption,
        "text_content": ocr_text,
    }


def _h_zoom_region(
    state: _VisualState,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    zoom: float = 2.0,
    reason: str = "",
) -> dict:
    """Crop + upscale for Option-B re-observation.

    Returns the crop both as a caption (like ``describe_region``) and as
    a base64-encoded PNG under the magic key ``_reobserve_image_b64``.
    ``run_tool_loop`` looks for that key and attaches the image to the
    next user message so the VLM re-perceives the zoomed region directly.
    """
    import base64 as _b64
    import io as _io

    img_w, img_h = state.image.size
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    zoom = max(1.0, min(float(zoom), 4.0))

    crop = state.image.crop((x, y, x + w, y + h))
    if zoom > 1.0:
        crop = crop.resize(
            (int(crop.width * zoom), int(crop.height * zoom)),
            Image.LANCZOS,
        )

    buf = _io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = _b64.b64encode(buf.getvalue()).decode("ascii")

    caption = _caption_crop(crop)
    ocr_text = _ocr_crop(crop)

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "zoom": zoom,
        "crop_size": [crop.width, crop.height],
        "caption": caption,
        "text_content": ocr_text,
        "reason": reason,
        # Magic key — consumed by tool_loop.run_tool_loop to re-feed the
        # crop as a user-side image message.  Keep the key name in sync
        # with ``tool_loop._REOBSERVE_KEY``.
        "_reobserve_image_b64": b64,
    }


def _h_visual_search(
    state: _VisualState,
    *,
    query: str,
    max_results: int = 5,
) -> dict:
    dets = state.detect()
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored: list[tuple[float, _DetectedElement]] = []
    for d in dets:
        label_lower = d.label.lower()
        label_words = set(label_lower.split())

        score = 0.0
        if query_lower in label_lower:
            score += 1.0
        if query_lower == label_lower:
            score += 0.5
        word_overlap = len(query_words & label_words)
        if query_words:
            score += 0.4 * word_overlap / len(query_words)
        if d.element_type.lower() in query_lower:
            score += 0.2
        if d.interactable and any(
            kw in query_lower for kw in ("button", "link", "input", "click")
        ):
            score += 0.15

        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: -x[0])
    top = scored[:max_results]

    results = []
    for score, d in top:
        results.append({
            "index": d.index,
            "label": d.label,
            "type": d.element_type,
            "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
            "relevance": round(score, 3),
        })

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "total_elements": len(dets),
    }


def _h_count_objects(
    state: _VisualState,
    *,
    element_type: str = "all",
    query: str = "",
) -> dict:
    dets = state.detect()

    if element_type != "all":
        dets = [d for d in dets if d.element_type == element_type]

    if query:
        query_lower = query.lower()
        dets = [d for d in dets if query_lower in d.label.lower()]

    return {
        "count": len(dets),
        "element_type_filter": element_type,
        "query_filter": query or None,
        "labels": [d.label for d in dets[:20]],
    }


def _h_classify_scene(state: _VisualState) -> dict:
    dets = state.detect()
    img_w, img_h = state.image.size

    type_counts = {}
    for d in dets:
        type_counts[d.element_type] = type_counts.get(d.element_type, 0) + 1

    interactable_count = sum(1 for d in dets if d.interactable)
    text_count = type_counts.get("text", 0)
    icon_count = type_counts.get("icon", 0)

    scene_hints = []
    if icon_count > 10:
        scene_hints.append("icon-heavy (toolbar / app grid)")
    if text_count > 15:
        scene_hints.append("text-heavy (article / form)")
    if interactable_count > 8:
        scene_hints.append("highly interactive (form / game / control panel)")
    if len(dets) < 5:
        scene_hints.append("sparse (dialog / splash / loading)")

    label_text = " ".join(d.label.lower() for d in dets)
    category_clues = {
        "game": any(w in label_text for w in ("score", "level", "lives", "play", "game")),
        "form": any(w in label_text for w in ("submit", "name", "email", "password", "login")),
        "menu": any(w in label_text for w in ("file", "edit", "view", "settings", "menu")),
        "dialog": any(w in label_text for w in ("ok", "cancel", "close", "confirm", "yes", "no")),
        "browser": any(w in label_text for w in ("http", "www", "url", "search", "tab")),
    }
    categories = [k for k, v in category_clues.items() if v] or ["unknown"]

    arr = np.array(state.image)
    mean_brightness = float(arr.mean()) / 255.0
    color_variety = float(arr.reshape(-1, 3).std(axis=0).mean()) / 255.0

    return {
        "categories": categories,
        "scene_hints": scene_hints or ["no strong signals"],
        "element_counts": type_counts,
        "total_elements": len(dets),
        "interactable_elements": interactable_count,
        "image_size": {"w": img_w, "h": img_h},
        "visual_properties": {
            "brightness": round(mean_brightness, 3),
            "color_variety": round(color_variety, 3),
            "theme": "dark" if mean_brightness < 0.35 else "light" if mean_brightness > 0.65 else "medium",
        },
    }


def _h_spatial_query(
    state: _VisualState,
    *,
    element_a: str,
    element_b: str,
) -> dict:
    a = state.find_by_ref(element_a)
    b = state.find_by_ref(element_b)

    if a is None:
        return {"error": f"Element '{element_a}' not found. Run detect_objects first."}
    if b is None:
        return {"error": f"Element '{element_b}' not found. Run detect_objects first."}

    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    a_cx, a_cy = ax + aw // 2, ay + ah // 2
    b_cx, b_cy = bx + bw // 2, by + bh // 2

    dx, dy = b_cx - a_cx, b_cy - a_cy
    dist = math.sqrt(dx * dx + dy * dy)

    # Overlap / containment
    overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap_area = overlap_x * overlap_y
    a_area = max(1, aw * ah)
    b_area = max(1, bw * bh)

    contains_a_b = overlap_area / b_area > 0.85 if b_area > 0 else False
    contains_b_a = overlap_area / a_area > 0.85 if a_area > 0 else False
    overlaps = overlap_area > 0

    h_aligned = abs(a_cy - b_cy) < min(ah, bh) * 0.5
    v_aligned = abs(a_cx - b_cx) < min(aw, bw) * 0.5

    if abs(dx) > abs(dy) * 2:
        direction = "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx) * 2:
        direction = "below" if dy > 0 else "above"
    elif dx > 0 and dy > 0:
        direction = "below-right"
    elif dx > 0:
        direction = "above-right"
    elif dy > 0:
        direction = "below-left"
    else:
        direction = "above-left"

    return {
        "element_a": {"label": a.label, "bbox": {"x": ax, "y": ay, "w": aw, "h": ah}},
        "element_b": {"label": b.label, "bbox": {"x": bx, "y": by, "w": bw, "h": bh}},
        "center_distance_px": round(dist, 1),
        "horizontal_offset": dx,
        "vertical_offset": dy,
        "direction_a_to_b": direction,
        "overlaps": overlaps,
        "overlap_area_px": overlap_area,
        "a_contains_b": contains_a_b,
        "b_contains_a": contains_b_a,
        "horizontally_aligned": h_aligned,
        "vertically_aligned": v_aligned,
    }


def _h_measure_distance(
    state: _VisualState,
    *,
    x1: int, y1: int, x2: int, y2: int,
) -> dict:
    dx, dy = x2 - x1, y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    img_w, img_h = state.image.size
    diag = math.sqrt(img_w ** 2 + img_h ** 2)

    if abs(dx) > abs(dy) * 2:
        direction = "right" if dx > 0 else "left"
    elif abs(dy) > abs(dx) * 2:
        direction = "down" if dy > 0 else "up"
    else:
        direction = f"{'down' if dy > 0 else 'up'}-{'right' if dx > 0 else 'left'}"

    return {
        "euclidean_px": round(dist, 1),
        "horizontal_px": dx,
        "vertical_px": dy,
        "direction": direction,
        "relative_to_diagonal": round(dist / diag, 4) if diag > 0 else 0,
    }


def _h_extract_colors(
    state: _VisualState,
    *,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    top_k: int = 5,
) -> dict:
    if x is not None and y is not None and w is not None and h is not None:
        img_w, img_h = state.image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        region = state.image.crop((x, y, x + w, y + h))
    else:
        region = state.image
        x, y = 0, 0
        w, h = region.size

    small = region.resize((64, 64))
    arr = np.array(small).reshape(-1, 3)

    try:
        from sklearn.cluster import KMeans
        n_clusters = min(top_k, len(arr))
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        km.fit(arr)
        centres = km.cluster_centers_.astype(int)
        labels = km.labels_
        counts = np.bincount(labels, minlength=n_clusters)
        total = counts.sum()
        order = np.argsort(-counts)
    except ImportError:
        colours_unique, counts = np.unique(arr, axis=0, return_counts=True)
        order = np.argsort(-counts)[:top_k]
        centres = colours_unique[order]
        counts = counts[order]
        total = counts.sum()
        order = range(len(centres))

    colors = []
    for i in order[:top_k]:
        r, g, b = int(centres[i][0]), int(centres[i][1]), int(centres[i][2])
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        pct = float(counts[i]) / total if total > 0 else 0
        colors.append({"hex": hex_code, "rgb": [r, g, b], "coverage": round(pct, 3)})

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "colors": colors,
    }


def _h_read_text_region(
    state: _VisualState,
    *,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
) -> dict:
    if x is not None and y is not None and w is not None and h is not None:
        img_w, img_h = state.image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        region = state.image.crop((x, y, x + w, y + h))
    else:
        region = state.image
        x, y = 0, 0
        w, h = region.size

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(region))
        lines: list[dict] = []
        full_texts: list[str] = []
        for bbox_pts, text, conf in results:
            flat = [int(p) for pt in bbox_pts for p in pt]
            x_min, y_min = min(flat[::2]), min(flat[1::2])
            x_max, y_max = max(flat[::2]), max(flat[1::2])
            lines.append({
                "text": text,
                "confidence": round(conf, 3),
                "bbox": {"x": x_min + x, "y": y_min + y,
                         "w": x_max - x_min, "h": y_max - y_min},
            })
            full_texts.append(text)

        lines.sort(key=lambda l: (l["bbox"]["y"], l["bbox"]["x"]))
        return {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "lines": lines,
            "full_text": " ".join(full_texts),
            "count": len(lines),
            "engine": "easyocr",
        }
    except ImportError:
        pass

    try:
        import pytesseract
        text = pytesseract.image_to_string(region)
        return {
            "region": {"x": x, "y": y, "w": w, "h": h},
            "full_text": text.strip(),
            "lines": [],
            "count": 0,
            "engine": "tesseract",
            "note": "Install easyocr for bbox-level results.",
        }
    except ImportError:
        pass

    return {
        "region": {"x": x, "y": y, "w": w, "h": h},
        "error": "No OCR engine available. Install easyocr or pytesseract.",
    }


def _h_grounded_detect(
    state: _VisualState,
    *,
    query: str,
    confidence_threshold: float = 0.15,
    max_results: int = 20,
) -> dict:
    if not _check_gdino():
        return {
            "error": (
                "GroundingDINO not available.  Install: "
                "pip install transformers torch"
            ),
            "elements": [],
            "count": 0,
        }

    dets = state.grounded_detect(
        query=query,
        confidence_threshold=confidence_threshold,
        max_elements=max_results,
    )
    elements = []
    for d in dets:
        elements.append({
            "index": d.index,
            "label": d.label,
            "bbox": {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
            "confidence": round(d.confidence, 3),
            "source": d.source,
        })

    return {
        "query": query,
        "elements": elements,
        "count": len(elements),
        "image_size": {"w": state.image.size[0], "h": state.image.size[1]},
    }


# ── Vision model helpers ─────────────────────────────────────────────

def _caption_crop(crop: Image.Image) -> str:
    """Caption a cropped region using Florence-2 (if available)."""
    try:
        from .grounding import _load_caption
        import torch

        cap = _load_caption()
        model, processor = cap["model"], cap["processor"]
        device = model.device

        small = crop.resize((224, 224))
        prompt = "<CAPTION>"

        if device.type == "cuda":
            inputs = processor(
                images=[small], text=[prompt],
                return_tensors="pt", do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=[small], text=[prompt],
                return_tensors="pt",
            ).to(device=device)

        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50, num_beams=1, do_sample=False,
            use_cache=False,  # Florence-2 + transformers ≥4.55 cache shim bug
        )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()
    except (ImportError, Exception) as exc:
        logger.debug("Florence-2 captioning unavailable: %s", exc)
        return "(captioning model not available)"


def _ocr_crop(crop: Image.Image) -> str:
    """Run OCR on a crop and return concatenated text."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.array(crop))
        return " ".join(text for _, text, _ in results)
    except ImportError:
        pass
    try:
        import pytesseract
        return pytesseract.image_to_string(crop).strip()
    except ImportError:
        pass
    return ""


# ── Public: build registry ───────────────────────────────────────────

def build_visual_registry(
    image: Image.Image | np.ndarray,
    *,
    prefer_gdino: bool = False,
    include_reasoning: bool = True,
) -> ToolRegistry:
    """Create a ToolRegistry with all vision-model-backed tools.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The frame / screenshot to analyse.
    prefer_gdino : bool
        If True, ``detect_objects`` uses GroundingDINO (open-vocabulary)
        instead of OmniParser (GUI-specific).  Set to True for natural
        images (image QA, video QA).  Regardless of this flag,
        ``grounded_detect`` is always available as an explicit
        query-driven tool.
    include_reasoning : bool
        If True (default), also register the symbolic reasoning tools
        (``count_value``, ``compute_ratio``, ``compare_values``,
        ``verify_claim``) defined in :mod:`.tools_reasoning`.  These
        produce typed derivation rows the schema can cite inside the
        ``<derivations>`` block.  The associated ``_DerivationLog`` is
        attached to the registry as ``reg.derivation_log`` so callers
        (e.g. the cascading grounding head) can render it onto the
        final schema.

    Returns
    -------
    ToolRegistry
        Registry with observation tools (detect_objects,
        grounded_detect, describe_region, visual_search, count_objects,
        classify_scene, spatial_query, measure_distance,
        extract_colors, read_text_region) and — when
        ``include_reasoning=True`` — derivation tools (count_value,
        compute_ratio, compare_values, verify_claim).
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    state = _VisualState(image, prefer_gdino=prefer_gdino)
    reg = ToolRegistry(domain="visual")

    reg.register(TOOL_DETECT_OBJECTS, lambda **kw: _h_detect_objects(state, **kw))
    reg.register(TOOL_GROUNDED_DETECT, lambda **kw: _h_grounded_detect(state, **kw))
    reg.register(TOOL_DESCRIBE_REGION, lambda **kw: _h_describe_region(state, **kw))
    reg.register(TOOL_ZOOM_REGION, lambda **kw: _h_zoom_region(state, **kw))
    reg.register(TOOL_VISUAL_SEARCH, lambda **kw: _h_visual_search(state, **kw))
    reg.register(TOOL_COUNT_OBJECTS, lambda **kw: _h_count_objects(state, **kw))
    reg.register(TOOL_CLASSIFY_SCENE, lambda **kw: _h_classify_scene(state))
    reg.register(TOOL_SPATIAL_QUERY, lambda **kw: _h_spatial_query(state, **kw))
    reg.register(TOOL_MEASURE_DISTANCE, lambda **kw: _h_measure_distance(state, **kw))
    reg.register(TOOL_EXTRACT_COLORS, lambda **kw: _h_extract_colors(state, **kw))
    reg.register(TOOL_READ_TEXT_REGION, lambda **kw: _h_read_text_region(state, **kw))

    if include_reasoning:
        from .tools_reasoning import build_reasoning_registry

        reasoning_reg, derivation_log = build_reasoning_registry()
        reg = reg.merge(reasoning_reg)
        reg.derivation_log = derivation_log  # type: ignore[attr-defined]

    return reg
