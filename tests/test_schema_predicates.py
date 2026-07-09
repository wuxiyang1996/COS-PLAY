"""
Unit tests for ``skill_agents.schema_predicates`` — the bridge between
the unified visual grounding ``<state>`` schema (``vlm_wrapper.schema``)
and the Stage 3 contract learner's predicate-bag input.

Covers:
* parsing a realistic gymv-style ``<state>`` block,
* uncertainty attenuation (``high`` → < 0.5 so booleanisation flips),
* tolerating dict-form per-step records produced by the visual
  grounding driver scripts,
* tolerating Experience-like objects (``summary_state`` / ``state``),
* graceful handling of malformed / missing schemas (returns ``{}``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from skill_agents.schema_predicates import (
    register_with,
    schema_to_predicates,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


_GYMV_SCHEMA = """\
<state>
domain=gymv
task=2048
goal=merge_high_tile
step=12

<entities>
e1[type=object, label=2-tile, bid=null, pos=0,1,1,1, ontology=selectable_entity]
e2[type=object, label=4-tile, bid=null, pos=1,1,1,1, ontology=selectable_entity]
e3[type=region, label=board, bid=null, pos=null, ontology=container_entity]
e4[type=object, label=goal_tile, bid=null, pos=2,2,1,1, ontology=goal_indicator]

<attributes>
e1.state=visible
e1.value=2
e2.state=visible
e2.value=4
e4.state=focused

<affordances>
e1.affords=[select, track, compare]
e2.affords=[select, track]
e4.affords=[approach, compare]

<relations>
contains(e3,e1)
contains(e3,e2)
adjacent(e1,e2)
grouped(e1,e2,e4)

<state_flags>
progress=0.42
phase=mid
scene_type=game_play
error=null
dialog_open=false
input_pending=true

<targets>
target=e2
blocker=null
constraint=avoid_overflow
candidate_set=[e2,e1,e4]
history_anchor=e1

<uncertainty>
e4.state=high
e4.value=high

<actions>
a1=Up
a2=Right
</state>
"""


_BROWSER_SCHEMA = """\
<state>
domain=browser
task=login
goal=submit_form
step=3

<entities>
e1[type=element, label=username_input, bid=42, ontology=interactive_entity]
e2[type=element, label=password_input, bid=43, ontology=interactive_entity]
e3[type=element, label=submit_button, bid=44, ontology=selectable_entity]

<attributes>
e1.state=focused
e2.state=visible
e3.state=visible

<affordances>
e1.affords=[focus, enter_text]
e3.affords=[select]

<relations>
adjacent(e1,e2)

<state_flags>
phase=early
scene_type=form_entry
dialog_open=false
input_pending=true

<targets>
target=e1
candidate_set=[e1,e2,e3]
</state>
"""


# ─────────────────────────────────────────────────────────────────────
# Core parsing
# ─────────────────────────────────────────────────────────────────────


class TestEntityPredicates:
    """Entity-level predicates: existence, type, ontology."""

    def test_entity_existence(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["entity:e1:exists"] == 1.0
        assert preds["entity:e2:exists"] == 1.0
        assert preds["entity:e3:exists"] == 1.0
        assert preds["entity:e4:exists"] < 1.0  # high uncertainty

    def test_entity_type(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["entity:e1:type:object"] == 1.0
        assert preds["entity:e3:type:region"] == 1.0

    def test_entity_ontology(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["entity:e1:ontology:selectable_entity"] == 1.0
        assert preds["entity:e3:ontology:container_entity"] == 1.0
        assert preds["entity:e4:ontology:goal_indicator"] < 1.0  # attenuated


class TestAttributePredicates:
    def test_attributes_emit_keyed_predicates(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["attr:e1:state=visible"] == 1.0
        assert preds["attr:e1:value=2"] == 1.0
        assert preds["attr:e2:value=4"] == 1.0

    def test_attributes_skip_null_values(self):
        schema = _GYMV_SCHEMA.replace("e1.value=2", "e1.value=null")
        preds = schema_to_predicates(schema)
        assert "attr:e1:value=null" not in preds


class TestAffordancePredicates:
    def test_affordances_split_per_verb(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["afford:e1:select"] == 1.0
        assert preds["afford:e1:track"] == 1.0
        assert preds["afford:e1:compare"] == 1.0
        assert preds["afford:e2:select"] == 1.0
        # absent verbs do not appear
        assert "afford:e2:compare" not in preds

    def test_affordances_not_in_attributes(self):
        """``e1.affords=[…]`` must not also appear as an ``attr:`` key."""
        preds = schema_to_predicates(_GYMV_SCHEMA)
        for key in preds:
            assert "affords=[" not in key


class TestRelationPredicates:
    def test_binary_relation(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["rel:contains:e3:e1"] == 1.0
        assert preds["rel:adjacent:e1:e2"] == 1.0

    def test_arity_three_relation(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["rel:grouped:e1:e2:e4"] == 1.0


class TestStateFlagPredicates:
    def test_progress_bucketing(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        # 0.42 falls into the mid bucket (0.33 ≤ p < 0.66)
        assert preds["flag:progress=mid"] == 1.0
        assert "flag:progress=low" not in preds
        assert "flag:progress=high" not in preds

    def test_phase_passthrough(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["flag:phase=mid"] == 1.0

    def test_scene_type_passthrough(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["flag:scene_type=game_play"] == 1.0

    def test_boolean_flags(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["flag:input_pending"] == 1.0
        # false-valued booleans are NOT emitted (closed-world predicate set)
        assert "flag:dialog_open" not in preds

    def test_null_flag_is_skipped(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert "flag:error" not in preds
        assert not any(k.startswith("flag:error") for k in preds)


class TestTargetPredicates:
    def test_target_eid(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["target:eid=e2"] == 1.0

    def test_blocker_null_omitted(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert not any(k.startswith("target:blocker") for k in preds)

    def test_candidate_set(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["target:candidate:e1"] == 1.0
        assert preds["target:candidate:e2"] == 1.0
        assert preds["target:candidate:e4"] == 1.0

    def test_history_anchor(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["target:history_anchor=e1"] == 1.0


class TestUncertaintyAttenuation:
    def test_high_uncertainty_below_boolean_threshold(self):
        """``high`` uncertainty must drop probability below 0.5 so that
        Stage 3's booleanisation flips the predicate to ``False``."""
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["entity:e4:exists"] < 0.5
        assert preds["entity:e4:ontology:goal_indicator"] < 0.5

    def test_certain_entity_unattenuated(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["entity:e1:exists"] == 1.0

    def test_uncertainty_only_affects_named_entity(self):
        preds = schema_to_predicates(_GYMV_SCHEMA)
        assert preds["entity:e2:ontology:selectable_entity"] == 1.0
        assert preds["entity:e2:exists"] == 1.0


# ─────────────────────────────────────────────────────────────────────
# Input coercion
# ─────────────────────────────────────────────────────────────────────


class TestInputCoercion:
    def test_dict_form_record(self):
        record = {
            "step": 12,
            "image_path": "frame_012.png",
            "schema_image_llm": _BROWSER_SCHEMA,
            "head": "image",
        }
        preds = schema_to_predicates(record)
        assert preds["entity:e1:exists"] == 1.0
        assert preds["target:eid=e1"] == 1.0

    def test_dict_form_with_nested_schema(self):
        record = {
            "schema_image_llm": {
                "schema": _BROWSER_SCHEMA,
                "warnings": [],
            },
        }
        preds = schema_to_predicates(record)
        assert preds["entity:e1:exists"] == 1.0

    def test_experience_like_object(self):
        @dataclass
        class FakeExperience:
            summary_state: Optional[str] = None
            state: Optional[str] = None

        exp = FakeExperience(summary_state=_BROWSER_SCHEMA)
        preds = schema_to_predicates(exp)
        assert preds["entity:e1:exists"] == 1.0

    def test_state_attribute_fallback(self):
        @dataclass
        class FakeExperience:
            summary_state: Optional[str] = None
            state: Optional[str] = None

        exp = FakeExperience(state=_BROWSER_SCHEMA)
        preds = schema_to_predicates(exp)
        assert preds["entity:e1:exists"] == 1.0

    def test_record_key_preference_image_over_text(self):
        record = {
            "schema_image_llm": _BROWSER_SCHEMA,
            "schema_text_llm": _GYMV_SCHEMA,
        }
        preds = schema_to_predicates(record)
        # If image schema wins, we should see browser entities (e1=username)
        assert preds["entity:e1:type:element"] == 1.0


class TestRobustness:
    @pytest.mark.parametrize("obs", [None, "", "no schema here", 42, [], {}])
    def test_no_schema_returns_empty_dict(self, obs):
        assert schema_to_predicates(obs) == {}

    def test_missing_close_tag_returns_empty(self):
        broken = _BROWSER_SCHEMA.replace("</state>", "")
        assert schema_to_predicates(broken) == {}

    def test_extra_whitespace_tolerated(self):
        spacey = _BROWSER_SCHEMA.replace("\n", "\n   ")
        preds = schema_to_predicates(spacey)
        assert preds.get("entity:e1:exists") == 1.0

    def test_partial_schema_no_optional_sections(self):
        minimal = (
            "<state>\n"
            "domain=gymv\n"
            "task=t\n"
            "goal=g\n"
            "step=0\n\n"
            "<entities>\n"
            "e1[type=object, label=tile, ontology=selectable_entity]\n"
            "</state>\n"
        )
        preds = schema_to_predicates(minimal)
        assert preds == {
            "entity:e1:exists": 1.0,
            "entity:e1:type:object": 1.0,
            "entity:e1:ontology:selectable_entity": 1.0,
        }

    def test_state_block_extracted_from_surrounding_text(self):
        wrapped = (
            "Here is the parsed output:\n"
            f"{_BROWSER_SCHEMA}\n"
            "(end of model output)"
        )
        preds = schema_to_predicates(wrapped)
        assert preds["entity:e1:exists"] == 1.0


# ─────────────────────────────────────────────────────────────────────
# Composite extractor wiring
# ─────────────────────────────────────────────────────────────────────


class TestRegisterWith:
    def test_register_with_calls_add_source(self):
        captured = []

        class FakeExtractor:
            def add_source(self, fn):
                captured.append(fn)

        register_with(FakeExtractor())
        assert captured == [schema_to_predicates]

    def test_composite_extractor_integration(self):
        """End-to-end: plug into the real CompositePredicateExtractor and
        verify schema-derived predicates land in the merged bag."""
        try:
            from skill_agents.stage3_mvp.extract_predicates import (
                CompositePredicateExtractor,
            )
            from skill_agents.stage3_mvp.predicate_vocab import PredicateVocab
        except Exception:
            pytest.skip("stage3_mvp predicate plumbing not importable here")

        vocab = PredicateVocab()
        extractor = CompositePredicateExtractor(vocab)
        register_with(extractor)

        merged = extractor(_BROWSER_SCHEMA)
        assert merged["entity:e1:exists"] == 1.0
        assert merged["target:eid=e1"] == 1.0
