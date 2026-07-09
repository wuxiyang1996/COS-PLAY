"""Result dataclasses + small helpers shared by the perception protocols.

Kept dependency-free on purpose: importing this module never pulls
``torch`` / ``PIL`` / ``transformers``.  Backends that need image
manipulation import Pillow lazily inside their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# BBox convention
# ──────────────────────────────────────────────────────────────────────

# Throughout the perception package a ``BBox`` is ``(x1, y1, x2, y2)``
# in **absolute pixel coordinates**, top-left origin, ints.  Same
# convention as ``Entity.pos`` in :mod:`decision_agents.schema_parser`.
# Backends that natively emit normalised ``cxcywh`` (Grounding-DINO,
# DETR family) convert in their own adapter.
BBox = Tuple[int, int, int, int]


# ──────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detection:
    """One bbox + confidence + label produced by a :class:`RegionDetector`.

    The label is whatever phrase the caller passed in (``"red cup"``,
    ``"close button"``) — detectors don't invent categories, they
    localise the caller's text query.  ``score`` is the detector's
    confidence in [0, 1] (post-sigmoid for Grounding-DINO).
    """

    bbox: BBox
    label: str
    score: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Segmentation:
    """One pixel mask + bbox produced by a :class:`Segmenter`.

    ``mask_rle`` is COCO-style run-length encoding so the cache can
    keep many masks cheaply.  ``area_px`` is precomputed because most
    callers want it and recomputing from RLE every time is wasteful.
    """

    bbox: BBox
    label: str
    score: float
    area_px: int
    mask_rle: Optional[str] = None  # None when backend is bbox-only
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRResult:
    """One text span + bbox + per-character confidence from an :class:`OCREngine`.

    ``text`` is the recognised string; ``score`` the engine's overall
    confidence in [0, 1]; ``bbox`` the region that produced the text
    (the input bbox when the caller passed one, else the engine's own
    detected text region).
    """

    text: str
    bbox: BBox
    score: float
    extra: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────


def bbox_iou(a: BBox, b: BBox) -> float:
    """Standard axis-aligned IoU.

    Returns ``0.0`` for disjoint or zero-area boxes; tolerant of
    swapped corners (callers occasionally hand in ``(x2, y2, x1, y1)``
    by accident).
    """
    ax1, ay1, ax2, ay2 = _normalise_bbox(a)
    bx1, by1, bx2, by2 = _normalise_bbox(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _normalise_bbox(b: BBox) -> BBox:
    x1, y1, x2, y2 = b
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def crop_image_bytes(image_bytes: bytes, bbox: BBox) -> bytes:
    """Return PNG-encoded crop of ``image_bytes`` at ``bbox``.

    Pillow is imported lazily so this module stays import-cheap.
    Falls back to returning the input bytes verbatim when Pillow is
    unavailable (e.g. minimal CI environment) — the caller can detect
    the no-op by comparing identity.
    """
    try:
        from io import BytesIO

        from PIL import Image  # type: ignore[import-not-found]
    except Exception:
        return image_bytes

    try:
        img = Image.open(BytesIO(image_bytes))
        x1, y1, x2, y2 = _normalise_bbox(bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return image_bytes
        cropped = img.crop((x1, y1, x2, y2))
        out = BytesIO()
        cropped.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return image_bytes


__all__ = [
    "BBox",
    "Detection",
    "Segmentation",
    "OCRResult",
    "bbox_iou",
    "crop_image_bytes",
]
