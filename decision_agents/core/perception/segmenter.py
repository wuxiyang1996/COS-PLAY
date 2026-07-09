"""``Segmenter`` Protocol + ``MockSegmenter``.

Segmentation differs from detection in two ways: callers usually
already have a bbox (from a prior :class:`RegionDetector` call) and
want a *mask* refined inside it; and the result carries pixel-level
shape information (``area_px``, optional ``mask_rle``) that downstream
ops like ``COUNT`` and ``MEASURE`` use for accurate geometry.

Real backends (Phase 8.1): ``SAM2Segmenter`` (Meta SAM-2, point or
bbox prompts).  Mock ships now for tests.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Protocol, runtime_checkable

from decision_agents.core.perception.types import BBox, Segmentation

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class Segmenter(Protocol):
    """Mask-refinement segmenter — bbox/point prompt → mask + tighter bbox.

    Backends accept either a bbox prompt (``prompt_bbox``) or a point
    prompt (``prompt_point``); at least one must be provided.  They
    return at most ``top_k`` :class:`Segmentation` rows sorted by
    score.

    Like :class:`RegionDetector`, implementations must be deterministic
    for ``EvidenceCache`` to behave correctly.
    """

    def segment(
        self,
        image_bytes: bytes,
        *,
        prompt_bbox: Optional[BBox] = None,
        prompt_point: Optional[tuple] = None,
        label: str = "",
        top_k: int = 1,
    ) -> List[Segmentation]:
        """Return up to ``top_k`` masks refining the prompt.

        Parameters
        ----------
        image_bytes
            Raw image bytes.
        prompt_bbox
            Optional bbox prompt.  When provided, the segmenter
            refines the mask inside this region.
        prompt_point
            Optional ``(x, y)`` point in absolute pixels.  Used when
            ``prompt_bbox`` is ``None``.
        label
            Free-form label preserved on the result for downstream
            entity creation.  Not used by the segmenter itself.
        top_k
            Hard cap on returned segmentations.  Most prompts produce
            one mask; SAM-2 can produce up to 3 ranked alternatives.
        """
        ...


# ──────────────────────────────────────────────────────────────────────
# MockSegmenter
# ──────────────────────────────────────────────────────────────────────


class MockSegmenter:
    """Deterministic stand-in segmenter for tests / CI.

    When ``prompt_bbox`` is provided, returns it verbatim with a
    deterministic score and an inferred ``area_px``.  When only
    ``prompt_point`` is given, fabricates a small bbox around the
    point.  No mask RLE is produced (``mask_rle=None``) — downstream
    code must tolerate that, mirroring the contract real backends will
    follow when configured for bbox-only mode.
    """

    def __init__(
        self,
        *,
        default_size: tuple = (640, 480),
        default_score: float = 0.85,
        point_bbox_radius: int = 32,
    ) -> None:
        self.default_size = default_size
        self.default_score = float(default_score)
        self.point_bbox_radius = max(1, int(point_bbox_radius))

    def segment(
        self,
        image_bytes: bytes,
        *,
        prompt_bbox: Optional[BBox] = None,
        prompt_point: Optional[tuple] = None,
        label: str = "",
        top_k: int = 1,
    ) -> List[Segmentation]:
        if prompt_bbox is None and prompt_point is None:
            return []

        if prompt_bbox is not None:
            bbox = self._normalise_bbox(prompt_bbox)
        else:
            bbox = self._point_to_bbox(prompt_point, image_bytes)

        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area == 0:
            return []

        # Deterministic score perturbation so cache hits are stable
        # but two different prompts on the same image differ.
        seed = int(
            hashlib.sha1(
                image_bytes + b"::seg::" + repr(bbox).encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        score_jitter = ((seed & 0xFF) / 255.0 - 0.5) * 0.1  # ±0.05
        score = max(0.0, min(1.0, self.default_score + score_jitter))

        return [
            Segmentation(
                bbox=bbox,
                label=label,
                score=score,
                area_px=int(area),
                mask_rle=None,
                extra={"backend": "mock"},
            )
        ][: max(1, top_k)]

    # ── private ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_bbox(b: BBox) -> BBox:
        x1, y1, x2, y2 = b
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def _point_to_bbox(
        self, point: tuple, image_bytes: bytes
    ) -> BBox:
        x, y = int(point[0]), int(point[1])
        r = self.point_bbox_radius
        width, height = self._infer_size(image_bytes)
        return (
            max(0, x - r),
            max(0, y - r),
            min(width, x + r),
            min(height, y + r),
        )

    def _infer_size(self, image_bytes: bytes) -> tuple:
        try:
            from io import BytesIO

            from PIL import Image  # type: ignore[import-not-found]

            img = Image.open(BytesIO(image_bytes))
            return img.width, img.height
        except Exception:
            return self.default_size


__all__ = [
    "Segmenter",
    "MockSegmenter",
]
