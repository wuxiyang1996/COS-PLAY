"""Perception backends for the cross-cutting harness ops.

Phase 8.0 ships the *plumbing*: Protocol definitions, lightweight
result dataclasses, an episode-scoped ``EvidenceCache``, and Mock
implementations that let the harness ops (``LOOK / CROP / READ_TEXT /
COUNT / SEGMENT``) run end-to-end in tests without GPU.

Real backends (Phase 8.1) plug in as Protocol implementations and
are loaded lazily so plain ``import decision_agents`` never pulls
``transformers`` / ``segment_anything_2`` / ``paddleocr``:

    from decision_agents.core.perception import RegionDetector
    from decision_agents.core.perception.real import GroundingDINODetector  # lazy

The harness owns one detector + one segmenter + one OCR engine + one
cache per episode.  Constructor injection means tests pass mocks and
production code passes real backends without touching ``ActorAgent``.
"""

from __future__ import annotations

from decision_agents.core.perception.cache import EvidenceCache
from decision_agents.core.perception.types import (
    Detection,
    OCRResult,
    Segmentation,
    bbox_iou,
    crop_image_bytes,
)
from decision_agents.core.perception.detector import (
    MockRegionDetector,
    RegionDetector,
)
from decision_agents.core.perception.segmenter import (
    MockSegmenter,
    Segmenter,
)
from decision_agents.core.perception.ocr import (
    MockOCR,
    OCREngine,
)

__all__ = [
    # Protocols.
    "RegionDetector",
    "Segmenter",
    "OCREngine",
    # Mock implementations (deterministic, GPU-free).
    "MockRegionDetector",
    "MockSegmenter",
    "MockOCR",
    # Result dataclasses.
    "Detection",
    "Segmentation",
    "OCRResult",
    # Cache.
    "EvidenceCache",
    # Helpers.
    "bbox_iou",
    "crop_image_bytes",
]
