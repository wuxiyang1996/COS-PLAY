"""End-to-end VR harness + perception package tests (Phase 8.0).

The harness should:

1. Load image bytes from a :class:`VisualInput` once per episode and
   cache the hash for ``EvidenceCache`` keys.
2. ``LOOK("close button")`` → mint an :class:`Entity` and surface it
   on ``info["schema_delta"]`` for the actor to merge.
3. ``READ_TEXT(eid)`` after a prior ``LOOK`` → reuses the bbox via
   the detector cache (one detector call only) and writes the OCR
   string into ``entity.value`` / ``entity.attributes["text"]``.
4. ``CROP(eid)`` → emits both a ``schema_delta`` entry and an
   ``info["images"]`` entry with the cropped region.
5. The cache backs every backend hit so two identical ``LOOK`` ops
   collapse to one detector call (cache stats verifiable).
6. ``ActorAgent._merge_schema_delta`` folds the harness-emitted
   entity into the next step's :class:`StateSchema`.
"""

from __future__ import annotations

import base64

import pytest

from decision_agents.actor_agent import ActorAgent
from decision_agents.core.harness_vr import VRHarness
from decision_agents.core.multimodal import VisualInput
from decision_agents.core.perception import (
    EvidenceCache,
    MockOCR,
    MockRegionDetector,
    MockSegmenter,
)
from decision_agents.schema_parser import Entity, StateSchema


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_image() -> VisualInput:
    """Produce a tiny in-memory PNG-ish bytes stream as a VisualInput.

    The mocks don't actually decode the bytes (they ``hash`` them and
    use ``default_size``), so any non-empty payload works for tests.
    """
    payload = b"\x89PNG\r\n\x1a\n" + b"vr-harness-test-bytes" * 4
    return VisualInput(image_b64=base64.b64encode(payload).decode("ascii"))


@pytest.fixture
def harness(fake_image: VisualInput) -> VRHarness:
    return VRHarness(
        image=fake_image,
        question="What is the close button label?",
        gold_answer="OK",
        max_steps=10,
        detector=MockRegionDetector(),
        segmenter=MockSegmenter(),
        ocr=MockOCR(),
    )


# ──────────────────────────────────────────────────────────────────────
# Image-byte loading + cache
# ──────────────────────────────────────────────────────────────────────


def test_constructor_auto_creates_cache_when_backends_present(
    fake_image: VisualInput,
) -> None:
    h = VRHarness(image=fake_image, detector=MockRegionDetector())
    assert isinstance(h.cache, EvidenceCache)


def test_constructor_no_cache_when_no_backends(
    fake_image: VisualInput,
) -> None:
    h = VRHarness(image=fake_image)
    assert h.cache is None


def test_reset_clears_cache(harness: VRHarness) -> None:
    harness.reset()
    harness.step('LOOK("close button")')
    assert harness.cache is not None and harness.cache.size > 0
    harness.reset()
    assert harness.cache.size == 0
    assert harness.cache.hits == 0 and harness.cache.misses == 0


# ──────────────────────────────────────────────────────────────────────
# LOOK
# ──────────────────────────────────────────────────────────────────────


def test_look_emits_schema_delta_with_entity(harness: VRHarness) -> None:
    harness.reset()
    obs, reward, done, info = harness.step('LOOK("close button")')
    assert not done
    delta = info.get("schema_delta")
    assert delta is not None and len(delta) == 1
    ent = delta[0]
    assert isinstance(ent, Entity)
    assert ent.label == "close button"
    assert ent.pos is not None
    assert ent.extra.get("source_op") == "LOOK"


def test_look_also_marks_scratchpad(harness: VRHarness) -> None:
    harness.reset()
    harness.step('LOOK("scene")')
    sp = harness.scratchpad
    assert "scene" in sp.grounded_slots


def test_look_without_detector_returns_no_delta(
    fake_image: VisualInput,
) -> None:
    h = VRHarness(image=fake_image)
    h.reset()
    obs, reward, done, info = h.step('LOOK("close button")')
    assert "schema_delta" not in info
    # Scratchpad behaviour preserved.
    assert "close button" in h.scratchpad.grounded_slots


def test_look_without_image_returns_no_delta() -> None:
    h = VRHarness(image=None, detector=MockRegionDetector())
    h.reset()
    _, _, _, info = h.step('LOOK("anything")')
    assert "schema_delta" not in info


def test_repeated_look_uses_cache(harness: VRHarness) -> None:
    harness.reset()
    harness.step('LOOK("close button")')
    misses_after_first = harness.cache.misses  # type: ignore[union-attr]
    harness.step('LOOK("close button")')
    # Second call should hit the cache → no new misses for the
    # ``detect`` op.
    assert harness.cache.misses == misses_after_first  # type: ignore[union-attr]
    assert harness.cache.hits >= 1  # type: ignore[union-attr]


# ──────────────────────────────────────────────────────────────────────
# READ_TEXT
# ──────────────────────────────────────────────────────────────────────


def test_read_text_writes_value_and_attributes(harness: VRHarness) -> None:
    harness.reset()
    _, _, _, info = harness.step('READ_TEXT("close button")')
    delta = info.get("schema_delta")
    assert delta is not None and len(delta) == 1
    ent = delta[0]
    assert ent.value  # MockOCR returns a non-empty phrase
    assert ent.attributes.get("text") == ent.value
    assert "ocr_score" in ent.attributes


def test_read_text_without_ocr_returns_no_delta(
    fake_image: VisualInput,
) -> None:
    h = VRHarness(image=fake_image, detector=MockRegionDetector())
    h.reset()
    _, _, _, info = h.step('READ_TEXT("close button")')
    assert "schema_delta" not in info


# ──────────────────────────────────────────────────────────────────────
# CROP
# ──────────────────────────────────────────────────────────────────────


def test_crop_emits_image_and_schema_delta(harness: VRHarness) -> None:
    harness.reset()
    _, _, _, info = harness.step('CROP("submit")')
    delta = info.get("schema_delta")
    images = info.get("images")
    assert delta and len(delta) == 1
    assert images and len(images) == 1
    img = images[0]
    assert isinstance(img, VisualInput)
    assert img.image_b64  # crop bytes inlined
    assert img.caption  # caption refers to query


def test_crop_without_image_no_op(
    fake_image: VisualInput,
) -> None:
    h = VRHarness(image=None, detector=MockRegionDetector())
    h.reset()
    _, _, _, info = h.step('CROP("anything")')
    assert "schema_delta" not in info
    assert "images" not in info


# ──────────────────────────────────────────────────────────────────────
# COUNT
# ──────────────────────────────────────────────────────────────────────


def test_count_returns_at_least_one_and_summary_entity(
    harness: VRHarness,
) -> None:
    harness.reset()
    _, _, _, info = harness.step('COUNT("button")')
    assert info["count"] >= 1
    delta = info.get("schema_delta")
    assert delta and delta[0].attributes["count"] == str(info["count"])
    assert delta[0].attributes["query"] == "button"


# ──────────────────────────────────────────────────────────────────────
# SEGMENT
# ──────────────────────────────────────────────────────────────────────


def test_segment_writes_area_and_score(harness: VRHarness) -> None:
    harness.reset()
    _, _, _, info = harness.step('SEGMENT("close button")')
    delta = info.get("schema_delta")
    assert delta and "area_px" in delta[0].attributes
    assert "seg_score" in delta[0].attributes
    assert int(delta[0].attributes["area_px"]) > 0


# ──────────────────────────────────────────────────────────────────────
# ANSWER
# ──────────────────────────────────────────────────────────────────────


def test_answer_terminates_episode(harness: VRHarness) -> None:
    harness.reset()
    _, reward, done, info = harness.step('ANSWER("OK")')
    assert done
    assert reward == 1.0
    assert info["correct"] is True


def test_answer_wrong_returns_zero_reward(harness: VRHarness) -> None:
    harness.reset()
    _, reward, done, _ = harness.step('ANSWER("nope")')
    assert done
    assert reward == 0.0


def test_max_steps_truncation(fake_image: VisualInput) -> None:
    h = VRHarness(
        image=fake_image,
        question="q",
        max_steps=2,
        detector=MockRegionDetector(),
    )
    h.reset()
    h.step('LOOK("a")')
    _, _, done, info = h.step('LOOK("b")')
    assert done
    assert info.get("truncated") is True


# ──────────────────────────────────────────────────────────────────────
# action_kind / valid_actions sanity
# ──────────────────────────────────────────────────────────────────────


def test_action_kind_includes_segment(harness: VRHarness) -> None:
    assert harness.action_kind('SEGMENT(e1)') == harness.action_kind('LOOK(e1)')


def test_valid_actions_includes_segment_when_schema_has_entities(
    harness: VRHarness,
) -> None:
    from decision_agents.core.harness import HarnessState

    sch = StateSchema(
        entities={"e1": Entity(eid="e1", label="x")},
        entity_order=["e1"],
    )
    state = HarnessState(schema=sch)
    valid = harness.valid_actions(state)
    assert "SEGMENT(e1)" in valid


# ──────────────────────────────────────────────────────────────────────
# ActorAgent._merge_schema_delta integration
# ──────────────────────────────────────────────────────────────────────


def test_merge_schema_delta_appends_new_entity() -> None:
    out = ActorAgent._merge_schema_delta(
        StateSchema(),
        [Entity(eid="e_new_1", label="cup", pos=(0, 0, 10, 10))],
    )
    assert "e_new_1" in out.entities
    assert out.entity_order == ["e_new_1"]
    assert out.entities["e_new_1"].label == "cup"


def test_merge_schema_delta_updates_existing_entity_in_place() -> None:
    sch = StateSchema(
        entities={"e1": Entity(eid="e1", label="cup")},
        entity_order=["e1"],
    )
    out = ActorAgent._merge_schema_delta(
        sch,
        [Entity(eid="e1", label="red cup", pos=(1, 2, 3, 4))],
    )
    assert out.entities["e1"].label == "red cup"
    assert out.entities["e1"].pos == (1, 2, 3, 4)
    assert out.entity_order == ["e1"]


def test_merge_schema_delta_accepts_dicts() -> None:
    out = ActorAgent._merge_schema_delta(
        None,
        [{"eid": "e_d_1", "label": "x", "extra": {"source": "test"}}],
    )
    assert "e_d_1" in out.entities
    assert out.entities["e_d_1"].extra["source"] == "test"


def test_merge_schema_delta_skips_malformed_dicts() -> None:
    out = ActorAgent._merge_schema_delta(
        StateSchema(), [{"label": "no-eid"}, "not-a-dict", None],
    )
    assert out.entities == {}


def test_merge_schema_delta_dict_unknown_keys_go_to_extra() -> None:
    out = ActorAgent._merge_schema_delta(
        None,
        [{"eid": "e1", "novel_field": "v"}],
    )
    assert out.entities["e1"].extra["novel_field"] == "v"


def test_merge_schema_delta_merges_attributes_dict() -> None:
    sch = StateSchema(
        entities={"e1": Entity(eid="e1", attributes={"a": "1"})},
        entity_order=["e1"],
    )
    out = ActorAgent._merge_schema_delta(
        sch, [Entity(eid="e1", attributes={"b": "2"})],
    )
    assert out.entities["e1"].attributes == {"a": "1", "b": "2"}


def test_merge_schema_delta_dedups_affords() -> None:
    sch = StateSchema(
        entities={"e1": Entity(eid="e1", affords=["click"])},
        entity_order=["e1"],
    )
    out = ActorAgent._merge_schema_delta(
        sch, [Entity(eid="e1", affords=["click", "type"])],
    )
    assert out.entities["e1"].affords == ["click", "type"]


def test_merge_schema_delta_passthrough_when_empty() -> None:
    sch = StateSchema(entities={"e1": Entity(eid="e1")}, entity_order=["e1"])
    out = ActorAgent._merge_schema_delta(sch, [])
    assert out is sch
    out2 = ActorAgent._merge_schema_delta(sch, None)
    assert out2 is sch
