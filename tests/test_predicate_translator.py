"""Tests for ``harness.predicate_translator``.

Coverage:

* :func:`translate_predicates` -- identity (diagonal cells, unregistered
  cells, unmapped predicates), drop (empty target list), one-to-many
  fan-out, dedupe across fan-outs, empty input.
* :func:`translate_skill_contract` -- never mutates input, deep-copies,
  diagonal identity, cross-modal rewrite, idempotent ``notes`` tag.
* :func:`with_predicate_translation` -- factory pass-through, source
  resolution from ``skill.source_domains[0]``, default fallback,
  translated success_fn delegates to inner with translated skill.
* Table sanity -- every target predicate in the table appears in
  :data:`skill_transfer_test.extract.audits._target_vocabularies.TARGET_PREDICATE_VOCAB`
  for that target (otherwise translation just shifts the static-vocab
  miss without unblocking the cell).

Tests are pure / hermetic -- no on-disk data required.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from harness.predicate_translator import (
    PREDICATE_TRANSLATIONS,
    translate_predicates,
    translate_skill_contract,
    with_predicate_translation,
)


# ---------------------------------------------------------------------------
# Helpers -- minimal stubs that satisfy the translator's getattr surface
# ---------------------------------------------------------------------------

def _stub_contract(eff_add=None, eff_del=None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        effects_add=list(eff_add or []),
        effects_del=list(eff_del or []),
    )


def _stub_skill(
    eff_add=None,
    eff_del=None,
    source_domains=None,
    notes: str = "",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        contract=_stub_contract(eff_add, eff_del),
        source_domains=list(source_domains or []),
        notes=notes,
    )


# ===========================================================================
# translate_predicates
# ===========================================================================


class TestTranslatePredicates:

    def test_empty_input_returns_empty(self):
        assert translate_predicates([], source="gymv", target="visual_reasoning") == []

    def test_diagonal_is_identity(self):
        preds = ["entity_value_increased", "phase_transitioned"]
        assert translate_predicates(preds, source="gymv", target="gymv") == preds

    def test_unregistered_cell_is_identity(self):
        # No (visual_reasoning, browser) cell registered -> identity.
        assert ("visual_reasoning", "browser") not in PREDICATE_TRANSLATIONS
        preds = ["answer_emitted", "answer_matches_gold"]
        out = translate_predicates(preds, source="visual_reasoning", target="browser")
        assert out == preds

    def test_unmapped_predicate_passes_through(self):
        # gymv->visual_reasoning is registered but doesn't list this
        # synthetic predicate, so it should pass through.
        out = translate_predicates(
            ["unmapped_synthetic_predicate"],
            source="gymv", target="visual_reasoning",
        )
        assert out == ["unmapped_synthetic_predicate"]

    def test_one_to_one_rename(self):
        # phase_transitioned -> phase_transitioned (passthrough mapping).
        out = translate_predicates(
            ["phase_transitioned"],
            source="gymv", target="visual_reasoning",
        )
        assert out == ["phase_transitioned"]

    def test_one_to_many_fanout(self):
        # cumulative_reward_increased -> [answer_emitted, answer_matches_gold]
        out = translate_predicates(
            ["cumulative_reward_increased"],
            source="gymv", target="visual_reasoning",
        )
        assert out == ["answer_emitted", "answer_matches_gold"]

    def test_drop_predicate(self):
        # entity_count_changed has no analogue in image-VR -> drop.
        out = translate_predicates(
            ["entity_count_changed"],
            source="gymv", target="visual_reasoning",
        )
        assert out == []

    def test_dedupe_across_fanouts(self):
        # entity_appeared -> [entity_appeared, entity_grounded] AND
        # cumulative_reward_increased -> [answer_emitted, answer_matches_gold]
        # If we add a duplicate `entity_appeared` after the first, we
        # should not see entity_appeared twice in the output.
        out = translate_predicates(
            ["entity_appeared", "cumulative_reward_increased", "entity_appeared"],
            source="gymv", target="visual_reasoning",
        )
        # entity_appeared (first occurrence) -> [entity_appeared, entity_grounded]
        # cumulative_reward_increased -> [answer_emitted, answer_matches_gold]
        # entity_appeared (second occurrence) -> dedup'd out
        assert out == [
            "entity_appeared", "entity_grounded",
            "answer_emitted", "answer_matches_gold",
        ]

    def test_mix_of_drop_and_keep(self):
        out = translate_predicates(
            [
                "phase_transitioned",       # passthrough
                "entity_count_changed",     # drop
                "cumulative_reward_increased",  # fan-out
            ],
            source="gymv", target="visual_reasoning",
        )
        assert out == [
            "phase_transitioned",
            "answer_emitted", "answer_matches_gold",
        ]

    def test_video_target_has_temporal_extension(self):
        # gymv -> video maps entity_disappeared to temporal_ordering_correct
        # (whereas gymv -> visual_reasoning drops it).
        assert translate_predicates(
            ["entity_disappeared"], source="gymv", target="visual_reasoning",
        ) == []
        assert translate_predicates(
            ["entity_disappeared"], source="gymv", target="video",
        ) == ["temporal_ordering_correct"]

    def test_osworld_keeps_count_and_attribute(self):
        # gymv -> osworld preserves entity_count_changed + attribute_changed
        # because osworld's vocab has both (per TARGET_PREDICATE_VOCAB).
        out = translate_predicates(
            ["entity_count_changed", "attribute_changed"],
            source="gymv", target="osworld",
        )
        assert out == ["entity_count_changed", "attribute_changed"]

    def test_browser_remaps_disappeared(self):
        # browser has no entity_disappeared in its vocab -> remapped.
        out = translate_predicates(
            ["entity_disappeared"], source="gymv", target="browser",
        )
        assert out == ["attribute_changed"]


# ===========================================================================
# translate_skill_contract
# ===========================================================================


class TestTranslateSkillContract:

    def test_returns_none_for_none_input(self):
        assert translate_skill_contract(None, source="gymv", target="video") is None

    def test_does_not_mutate_input(self):
        skill = _stub_skill(eff_add=["cumulative_reward_increased"])
        original_eff_add = list(skill.contract.effects_add)
        translate_skill_contract(skill, source="gymv", target="visual_reasoning")
        assert skill.contract.effects_add == original_eff_add

    def test_returns_deep_copy(self):
        skill = _stub_skill(eff_add=["phase_transitioned"])
        out = translate_skill_contract(skill, source="gymv", target="osworld")
        assert out is not skill
        assert out.contract is not skill.contract
        # Mutating the copy must not touch the original.
        out.contract.effects_add.append("hacked")
        assert "hacked" not in skill.contract.effects_add

    def test_diagonal_returns_unchanged_copy(self):
        skill = _stub_skill(
            eff_add=["entity_value_increased", "phase_transitioned"],
            eff_del=["entity_appeared"],
        )
        out = translate_skill_contract(skill, source="gymv", target="gymv")
        assert out.contract.effects_add == skill.contract.effects_add
        assert out.contract.effects_del == skill.contract.effects_del

    def test_cross_modal_rewrites_and_tags_notes(self):
        skill = _stub_skill(
            eff_add=["cumulative_reward_increased", "entity_count_changed"],
            eff_del=["phase_transitioned"],
        )
        out = translate_skill_contract(
            skill, source="gymv", target="visual_reasoning",
        )
        # cumulative_reward_increased -> [answer_emitted, answer_matches_gold]
        # entity_count_changed -> dropped
        assert out.contract.effects_add == ["answer_emitted", "answer_matches_gold"]
        # phase_transitioned passthrough on the del side too.
        assert out.contract.effects_del == ["phase_transitioned"]
        assert "[predicate_translator: gymv->visual_reasoning]" in out.notes

    def test_notes_tag_idempotent(self):
        skill = _stub_skill(
            eff_add=["cumulative_reward_increased"],
            notes="existing note",
        )
        out = translate_skill_contract(
            skill, source="gymv", target="visual_reasoning",
        )
        # Re-translate -- the marker must not appear twice.
        out2 = translate_skill_contract(
            out, source="gymv", target="visual_reasoning",
        )
        assert out2.notes.count("[predicate_translator: gymv->visual_reasoning]") == 1
        assert "existing note" in out2.notes

    def test_no_translation_when_unchanged(self):
        # Diagonal => no notes tag should appear.
        skill = _stub_skill(eff_add=["phase_transitioned"], notes="x")
        out = translate_skill_contract(skill, source="gymv", target="gymv")
        assert "predicate_translator" not in out.notes

    def test_handles_skill_without_contract(self):
        skill = types.SimpleNamespace(notes="")
        out = translate_skill_contract(skill, source="gymv", target="video")
        assert out is not skill
        # No-op: nothing to translate, no contract attribute either way.


# ===========================================================================
# with_predicate_translation
# ===========================================================================


class TestWithPredicateTranslation:

    def test_factory_forwards_args_kwargs(self):
        captured = {}

        def factory(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return lambda skill, *a, **kw: 1.0

        wrapped = with_predicate_translation(
            factory, target_domain="visual_reasoning",
        )
        wrapped(7, 8, name="qa", threshold=0.5)
        assert captured["args"] == (7, 8)
        assert captured["kwargs"] == {"name": "qa", "threshold": 0.5}

    def test_translated_success_fn_sees_translated_skill(self):
        seen_skills = []

        def factory(**kwargs):
            def inner(skill, *a, **kw):
                seen_skills.append(skill)
                return 1.0
            return inner

        wrapped = with_predicate_translation(
            factory, target_domain="visual_reasoning",
        )
        success_fn = wrapped()
        skill = _stub_skill(
            eff_add=["cumulative_reward_increased"],
            source_domains=["gymv"],
        )
        success_fn(skill, "episode_arg", demo="demo_arg")
        assert len(seen_skills) == 1
        translated = seen_skills[0]
        # The inner sees a translated copy, not the original.
        assert translated is not skill
        assert translated.contract.effects_add == [
            "answer_emitted", "answer_matches_gold",
        ]

    def test_translated_success_fn_uses_default_when_source_empty(self):
        seen_targets = []

        def factory():
            def inner(skill, *a, **kw):
                # Read the marker out of the notes to assert the inferred
                # source.
                seen_targets.append(skill.notes)
                return 0.0
            return inner

        wrapped = with_predicate_translation(
            factory, target_domain="visual_reasoning", default_source="gymv",
        )
        success_fn = wrapped()
        skill = _stub_skill(
            eff_add=["cumulative_reward_increased"],
            source_domains=[],  # empty -> falls back to default_source
        )
        success_fn(skill)
        assert any("gymv->visual_reasoning" in n for n in seen_targets)

    def test_diagonal_translation_is_noop(self):
        seen_skills = []

        def factory():
            def inner(skill, *a, **kw):
                seen_skills.append(skill)
                return 1.0
            return inner

        wrapped = with_predicate_translation(factory, target_domain="gymv")
        success_fn = wrapped()
        skill = _stub_skill(
            eff_add=["entity_value_increased"], source_domains=["gymv"],
        )
        success_fn(skill)
        # Diagonal call -> notes should NOT carry a translator marker.
        assert "predicate_translator" not in seen_skills[0].notes

    def test_wrapped_factory_preserves_inner_contract(self):
        # Forwarding kwargs should reach the inner success_fn unchanged.
        captured_calls = []

        def factory():
            def inner(skill, *a, **kw):
                captured_calls.append((a, kw))
                return 0.5
            return inner

        wrapped = with_predicate_translation(
            factory, target_domain="visual_reasoning",
        )
        success_fn = wrapped()
        skill = _stub_skill(eff_add=[], source_domains=["gymv"])
        result = success_fn(skill, "ep", demo="d")
        assert result == 0.5
        assert captured_calls == [(("ep",), {"demo": "d"})]


# ===========================================================================
# Table sanity -- every target predicate must be in the target's vocab
# ===========================================================================


class TestTableSanity:

    def test_every_target_predicate_is_in_target_vocab(self):
        """Translation must not shift static-vocab miss rather than fix it."""
        from skill_transfer_test.extract.audits._target_vocabularies import (
            TARGET_PREDICATE_VOCAB,
        )
        for (source, target), table in PREDICATE_TRANSLATIONS.items():
            target_vocab = TARGET_PREDICATE_VOCAB.get(target)
            assert target_vocab is not None, (
                f"({source}, {target}): target {target!r} not in TARGET_PREDICATE_VOCAB"
            )
            for source_pred, target_preds in table.items():
                for tp in target_preds:
                    assert tp in target_vocab, (
                        f"({source}, {target}): {source_pred!r} -> {tp!r} but "
                        f"{tp!r} is not in {target}'s vocab "
                        f"({sorted(target_vocab)})"
                    )

    def test_all_source_targets_are_canonical_domains(self):
        from common.enums import DOMAINS
        for source, target in PREDICATE_TRANSLATIONS.keys():
            assert source in DOMAINS, f"unknown source domain {source!r}"
            assert target in DOMAINS, f"unknown target domain {target!r}"

    def test_no_diagonal_cells_in_table(self):
        for source, target in PREDICATE_TRANSLATIONS.keys():
            assert source != target, (
                f"diagonal cell ({source}, {target}) registered -- "
                "diagonals are handled by translate_predicates' identity path"
            )
