"""Unit tests for the perception sub-package (Phase 8.0).

Covers:

* Each Protocol implements ``isinstance(...)`` against its
  ``runtime_checkable`` Protocol.
* Mock backends are deterministic — same image + query → same output.
* :class:`EvidenceCache` round-trips, evicts LRU entries, and
  reports hit/miss stats.
* Geometry helpers (``bbox_iou``).
"""

from __future__ import annotations

import pytest

from decision_agents.core.perception import (
    Detection,
    EvidenceCache,
    MockOCR,
    MockRegionDetector,
    MockSegmenter,
    OCREngine,
    OCRResult,
    RegionDetector,
    Segmentation,
    Segmenter,
    bbox_iou,
)
from decision_agents.core.perception.cache import (
    DEFAULT_CACHE_SIZE,
    hash_image_bytes,
    serialise_args,
)


# ──────────────────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────────────────


def test_mock_backends_satisfy_protocols() -> None:
    """``runtime_checkable`` Protocols accept the matching mock."""
    assert isinstance(MockRegionDetector(), RegionDetector)
    assert isinstance(MockSegmenter(), Segmenter)
    assert isinstance(MockOCR(), OCREngine)


# ──────────────────────────────────────────────────────────────────────
# MockRegionDetector
# ──────────────────────────────────────────────────────────────────────


def test_detector_returns_one_hit_for_nonempty_query() -> None:
    det = MockRegionDetector()
    hits = det.detect(b"img-bytes", "close button")
    assert len(hits) == 1
    h = hits[0]
    assert isinstance(h, Detection)
    assert h.label == "close button"
    assert 0.0 <= h.score <= 1.0
    x1, y1, x2, y2 = h.bbox
    assert x2 > x1 and y2 > y1


def test_detector_is_deterministic() -> None:
    det = MockRegionDetector()
    a = det.detect(b"same-image", "red cup")
    b = det.detect(b"same-image", "red cup")
    assert a == b


def test_detector_threshold_filters_low_scores() -> None:
    det = MockRegionDetector(min_score=0.1, max_score=0.2)
    hits = det.detect(b"img", "anything", threshold=0.5)
    assert hits == []


def test_detector_empty_inputs_yield_empty_list() -> None:
    det = MockRegionDetector()
    assert det.detect(b"", "q") == []
    assert det.detect(b"img", "") == []


# ──────────────────────────────────────────────────────────────────────
# MockSegmenter
# ──────────────────────────────────────────────────────────────────────


def test_segmenter_returns_input_bbox_for_bbox_prompt() -> None:
    seg = MockSegmenter()
    masks = seg.segment(b"img", prompt_bbox=(10, 20, 100, 200), label="x")
    assert len(masks) == 1
    m = masks[0]
    assert isinstance(m, Segmentation)
    assert m.bbox == (10, 20, 100, 200)
    assert m.area_px == 90 * 180
    assert m.label == "x"


def test_segmenter_normalises_swapped_bbox() -> None:
    seg = MockSegmenter()
    masks = seg.segment(b"img", prompt_bbox=(100, 200, 10, 20))
    assert masks[0].bbox == (10, 20, 100, 200)


def test_segmenter_handles_point_prompt() -> None:
    seg = MockSegmenter(point_bbox_radius=5)
    masks = seg.segment(b"img", prompt_point=(50, 60))
    assert len(masks) == 1
    x1, y1, x2, y2 = masks[0].bbox
    assert (x2 - x1) > 0 and (y2 - y1) > 0


def test_segmenter_no_prompt_returns_empty() -> None:
    seg = MockSegmenter()
    assert seg.segment(b"img") == []


# ──────────────────────────────────────────────────────────────────────
# MockOCR
# ──────────────────────────────────────────────────────────────────────


def test_ocr_with_bbox_returns_one_span() -> None:
    ocr = MockOCR()
    spans = ocr.read(b"img", bbox=(0, 0, 50, 50))
    assert len(spans) == 1
    s = spans[0]
    assert isinstance(s, OCRResult)
    assert s.text  # non-empty phrase
    assert s.bbox == (0, 0, 50, 50)


def test_ocr_is_deterministic() -> None:
    ocr = MockOCR()
    a = ocr.read(b"img", bbox=(1, 2, 3, 4))
    b = ocr.read(b"img", bbox=(1, 2, 3, 4))
    assert a == b


def test_ocr_max_spans_clamps() -> None:
    ocr = MockOCR()
    spans = ocr.read(b"img", bbox=(0, 0, 10, 10), max_spans=0)
    assert len(spans) == 1  # clamp lifts 0 → 1


def test_ocr_empty_image_returns_empty() -> None:
    ocr = MockOCR()
    assert ocr.read(b"") == []


# ──────────────────────────────────────────────────────────────────────
# EvidenceCache
# ──────────────────────────────────────────────────────────────────────


def test_cache_get_then_put_records_hit_and_miss() -> None:
    cache = EvidenceCache()
    assert cache.get("h1", "detect", {"q": "foo"}) is None
    assert cache.misses == 1 and cache.hits == 0

    cache.put("h1", "detect", {"q": "foo"}, "value")
    out = cache.get("h1", "detect", {"q": "foo"})
    assert out == "value"
    assert cache.hits == 1 and cache.misses == 1


def test_cache_keys_are_arg_aware() -> None:
    cache = EvidenceCache()
    cache.put("h1", "detect", {"q": "foo"}, "A")
    cache.put("h1", "detect", {"q": "bar"}, "B")
    assert cache.get("h1", "detect", {"q": "foo"}) == "A"
    assert cache.get("h1", "detect", {"q": "bar"}) == "B"


def test_cache_keys_are_image_aware() -> None:
    cache = EvidenceCache()
    cache.put("h1", "detect", {}, "A")
    cache.put("h2", "detect", {}, "B")
    assert cache.get("h1", "detect", {}) == "A"
    assert cache.get("h2", "detect", {}) == "B"


def test_cache_lru_eviction() -> None:
    cache = EvidenceCache(max_entries=2)
    cache.put("h", "op", "a", "A")
    cache.put("h", "op", "b", "B")
    # Touch "a" so "b" becomes LRU.
    cache.get("h", "op", "a")
    cache.put("h", "op", "c", "C")
    # "b" should have been evicted.
    assert cache.get("h", "op", "b") is None
    assert cache.get("h", "op", "a") == "A"
    assert cache.get("h", "op", "c") == "C"


def test_cache_clear_resets_state() -> None:
    cache = EvidenceCache()
    cache.put("h", "op", None, "X")
    cache.get("h", "op", None)
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0 and cache.misses == 0
    assert cache.get("h", "op", None) is None


def test_cache_default_size_constant() -> None:
    """If someone bumps the default, they must update the docstring."""
    assert DEFAULT_CACHE_SIZE >= 64  # any sane minimum


def test_cache_stats_round_trip() -> None:
    cache = EvidenceCache()
    cache.get("h", "op", None)
    cache.put("h", "op", None, "v")
    cache.get("h", "op", None)
    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert pytest.approx(stats["hit_rate"], abs=1e-3) == 0.5


def test_cache_make_key_is_deterministic() -> None:
    k1 = EvidenceCache.make_key("h", "op", {"a": 1, "b": 2})
    k2 = EvidenceCache.make_key("h", "op", {"b": 2, "a": 1})
    assert k1 == k2  # sort_keys ensures ordering insensitivity


# ──────────────────────────────────────────────────────────────────────
# Hash + serialise helpers
# ──────────────────────────────────────────────────────────────────────


def test_hash_image_bytes_short_hex() -> None:
    h = hash_image_bytes(b"hello")
    assert isinstance(h, str)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_serialise_args_handles_primitives_and_dicts() -> None:
    assert serialise_args(None) == ""
    assert serialise_args(3) == "3"
    assert serialise_args("foo") == "foo"
    assert serialise_args({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


# ──────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────


def test_bbox_iou_basic() -> None:
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    iou = bbox_iou(a, b)
    # intersection = 5×5 = 25; union = 100 + 100 − 25 = 175.
    assert pytest.approx(iou, abs=1e-3) == 25 / 175


def test_bbox_iou_disjoint_zero() -> None:
    assert bbox_iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0


def test_bbox_iou_identical_one() -> None:
    a = (1, 2, 3, 4)
    assert bbox_iou(a, a) == 1.0


def test_bbox_iou_swapped_corners() -> None:
    a = (10, 10, 0, 0)  # caller mistake; helper should normalise
    b = (0, 0, 10, 10)
    assert bbox_iou(a, b) == 1.0
