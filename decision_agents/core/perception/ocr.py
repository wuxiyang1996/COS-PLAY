"""``OCREngine`` Protocol + ``MockOCR``.

OCR closes the third leg of the cross-cutting perception trio.  It's
the highest-leverage of the three for the *web* and *OS* harnesses:
DOM accessibility trees lose alt text for icon buttons, and OSWorld
desktop screenshots routinely have read-only labels (file names,
dialog text, terminal output) that no a11y API exposes.

Real backends (Phase 8.1): ``PaddleOCREngine`` (multilingual, fast),
``TesseractOCREngine`` (CPU-friendly fallback).  Mock ships now.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Protocol, runtime_checkable

from decision_agents.core.perception.types import BBox, OCRResult

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class OCREngine(Protocol):
    """Optical character recognition over a region (or a whole image).

    Backends accept an optional ``bbox``: when present, OCR is
    restricted to that region (useful after a :class:`RegionDetector`
    pinned an icon-with-label); when absent, the engine scans the
    whole image and may return many spans.
    """

    def read(
        self,
        image_bytes: bytes,
        *,
        bbox: Optional[BBox] = None,
        languages: Optional[List[str]] = None,
        max_spans: int = 32,
    ) -> List[OCRResult]:
        """Return at most ``max_spans`` text spans found in the region.

        Parameters
        ----------
        image_bytes
            Raw image bytes.
        bbox
            Optional crop region (absolute pixel coords).
        languages
            ISO 639-1 codes (``["en", "zh"]``).  When ``None`` the
            backend's configured default is used.
        max_spans
            Hard cap on returned spans, sorted by reading order
            (top-to-bottom, left-to-right) then truncated.
        """
        ...


# ──────────────────────────────────────────────────────────────────────
# MockOCR
# ──────────────────────────────────────────────────────────────────────


class MockOCR:
    """Deterministic stand-in OCR engine for tests / CI.

    Produces a fixed phrase per ``(image_hash, bbox)`` pair so cache
    behaviour is testable.  When ``bbox`` is provided, the result has
    that bbox; otherwise a small bbox in the upper-left is fabricated.

    The fixed phrase rotates through a small bank so two different
    queries don't trivially collide.
    """

    _PHRASE_BANK = (
        "Submit",
        "Cancel",
        "OK",
        "File",
        "Edit",
        "Sign in",
        "Search",
        "Close",
        "Login",
        "Save",
    )

    def __init__(
        self,
        *,
        default_score: float = 0.92,
        default_size: tuple = (640, 480),
    ) -> None:
        self.default_score = float(default_score)
        self.default_size = default_size

    def read(
        self,
        image_bytes: bytes,
        *,
        bbox: Optional[BBox] = None,
        languages: Optional[List[str]] = None,
        max_spans: int = 32,
    ) -> List[OCRResult]:
        if not image_bytes:
            return []

        target_bbox = bbox if bbox is not None else self._default_bbox(image_bytes)
        seed = int(
            hashlib.sha1(
                image_bytes + b"::ocr::" + repr(target_bbox).encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        phrase = self._PHRASE_BANK[seed % len(self._PHRASE_BANK)]

        score_jitter = ((seed & 0xFF) / 255.0 - 0.5) * 0.06  # ±0.03
        score = max(0.0, min(1.0, self.default_score + score_jitter))

        return [
            OCRResult(
                text=phrase,
                bbox=tuple(target_bbox),  # ensure plain tuple
                score=score,
                extra={"backend": "mock"},
            )
        ][: max(1, max_spans)]

    # ── private ──────────────────────────────────────────────────────

    def _default_bbox(self, image_bytes: bytes) -> BBox:
        try:
            from io import BytesIO

            from PIL import Image  # type: ignore[import-not-found]

            img = Image.open(BytesIO(image_bytes))
            w, h = img.width, img.height
        except Exception:
            w, h = self.default_size
        return (
            max(0, w // 20),
            max(0, h // 20),
            min(w, w // 20 + max(1, w // 6)),
            min(h, h // 20 + max(1, h // 12)),
        )


__all__ = [
    "OCREngine",
    "MockOCR",
]
