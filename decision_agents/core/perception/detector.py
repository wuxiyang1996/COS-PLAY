"""``RegionDetector`` Protocol + ``MockRegionDetector``.

The contract every detector backend implements: take an image (as
bytes, the harness owns the loading) plus a free-form text query
(``"close button"``, ``"red cup"``, ``"the king on e4"``) and return
zero or more :class:`Detection` rows.  Real backends (Phase 8.1)
include ``GroundingDINODetector`` and ``OWLv2Detector``; the mock
ships now so ``VRHarness`` / tests run without GPU.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Protocol, runtime_checkable

from decision_agents.core.perception.types import BBox, Detection

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class RegionDetector(Protocol):
    """Open-vocabulary region detector — ground a text query in an image.

    Implementations must be deterministic for a given (image, query,
    threshold) so the :class:`~decision_agents.core.perception.cache.EvidenceCache`
    works correctly.  Backends that wrap stochastic models should set
    a fixed seed in their constructor.
    """

    def detect(
        self,
        image_bytes: bytes,
        query: str,
        *,
        threshold: float = 0.30,
        top_k: int = 5,
    ) -> List[Detection]:
        """Return up to ``top_k`` detections for ``query`` in ``image_bytes``.

        Parameters
        ----------
        image_bytes
            Raw image (PNG / JPEG / whatever Pillow can decode).
        query
            Free-form text query.  No category restriction.
        threshold
            Minimum score in [0, 1].  Detections below are dropped
            *before* ``top_k`` truncation.
        top_k
            Hard cap on returned detections.  Backends should sort by
            score (descending) before truncating.

        Returns
        -------
        Empty list when nothing scored above ``threshold`` — never
        ``None``, so callers can iterate without a None-check.
        """
        ...


# ──────────────────────────────────────────────────────────────────────
# MockRegionDetector
# ──────────────────────────────────────────────────────────────────────


class MockRegionDetector:
    """Deterministic stand-in detector for tests / CI.

    Generates a synthetic bbox derived from ``hash(image_bytes, query)``
    so the same query on the same image always returns the same box.
    The score is also derived from the hash but lives in [0.4, 0.95]
    so most queries pass realistic ``threshold=0.30`` filtering.

    Image dimensions are inferred from the bytes when Pillow is
    available, else default to ``(640, 480)``.  Generated bboxes
    cover ~10-30% of the image area in a plausible position.

    Usage in tests::

        det = MockRegionDetector(default_size=(800, 600))
        hits = det.detect(image_bytes, query="close button")
        assert len(hits) == 1
        assert hits[0].label == "close button"
    """

    def __init__(
        self,
        *,
        default_size: tuple = (640, 480),
        min_score: float = 0.40,
        max_score: float = 0.95,
        return_count: int = 1,
    ) -> None:
        if not (0.0 <= min_score <= max_score <= 1.0):
            raise ValueError(
                f"min_score / max_score out of range: {min_score} / {max_score}"
            )
        self.default_size = default_size
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.return_count = max(1, int(return_count))

    def detect(
        self,
        image_bytes: bytes,
        query: str,
        *,
        threshold: float = 0.30,
        top_k: int = 5,
    ) -> List[Detection]:
        if not image_bytes or not query:
            return []

        width, height = self._infer_size(image_bytes)
        # Stable hash — same query on same image → same bboxes.
        seed = int(
            hashlib.sha1(image_bytes + b"::" + query.encode("utf-8")).hexdigest()[:12],
            16,
        )

        out: List[Detection] = []
        for i in range(min(self.return_count, top_k)):
            local = seed ^ (i * 0x9E37_79B1)  # golden-ratio mixing
            bbox = self._synthesise_bbox(local, width, height)
            score = self._synthesise_score(local)
            if score < threshold:
                continue
            out.append(
                Detection(
                    bbox=bbox,
                    label=query,
                    score=score,
                    extra={"backend": "mock", "rank": i},
                )
            )
        return out

    # ── private ──────────────────────────────────────────────────────

    def _infer_size(self, image_bytes: bytes) -> tuple:
        """Best-effort image-size detection; falls back to ``default_size``."""
        try:
            from io import BytesIO

            from PIL import Image  # type: ignore[import-not-found]

            img = Image.open(BytesIO(image_bytes))
            return img.width, img.height
        except Exception:
            return self.default_size

    def _synthesise_bbox(
        self, seed: int, width: int, height: int
    ) -> BBox:
        """Map ``seed`` to a deterministic bbox covering ~15% of the image."""
        # 6 bytes from the seed → 6 quasi-random numbers in [0, 1).
        rng = [(seed >> (i * 8)) & 0xFF for i in range(6)]
        cx_frac = 0.15 + (rng[0] / 255.0) * 0.7  # centre in [0.15, 0.85]
        cy_frac = 0.15 + (rng[1] / 255.0) * 0.7
        w_frac = 0.15 + (rng[2] / 255.0) * 0.20  # width in [0.15, 0.35]
        h_frac = 0.15 + (rng[3] / 255.0) * 0.20

        cx = int(cx_frac * width)
        cy = int(cy_frac * height)
        w = max(1, int(w_frac * width))
        h = max(1, int(h_frac * height))

        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(width, x1 + w)
        y2 = min(height, y1 + h)
        return (x1, y1, x2, y2)

    def _synthesise_score(self, seed: int) -> float:
        """Map ``seed`` to a score in ``[min_score, max_score]``."""
        frac = ((seed >> 24) & 0xFF) / 255.0
        return self.min_score + frac * (self.max_score - self.min_score)


__all__ = [
    "RegionDetector",
    "MockRegionDetector",
]
