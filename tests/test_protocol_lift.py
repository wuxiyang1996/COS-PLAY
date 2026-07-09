"""Tests for the protocol lift (`labeling/_protocol_lift.py`).

Validates the design locked in
`implementation_notes/legacy/protocol-lift-design.md` and the integration into
`labeling/_decorate_skill_records.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from labeling._protocol_lift import (
    EFFECT_PREDICATE_TYPES,
    GameSchemaIndex,
    LiftStats,
    VERB_TAXONOMY,
    classify_prose_step,
    extract_payload_slots,
    lift_protocol_to_typed_hops,
    mine_effects,
)


# ───────────── verb classifier ─────────────


def test_taxonomy_is_complete() -> None:
    """All 21 verbs from the §4.1 design table are present."""

    expected = {
        "SELECT", "SWAP", "SLIDE", "MOVE", "ROTATE", "DROP", "PLACE",
        "APPROACH", "EXECUTE",
        "READ", "INSPECT", "TRACK",
        "COMPARE", "EVALUATE", "SIMULATE", "PREFER", "PENALIZE", "VERIFY",
        "STOP", "CONTINUE", "KEEP",
    }
    assert VERB_TAXONOMY == expected


@pytest.mark.parametrize(
    "prose,expected_verb,expected_mode",
    [
        # First-word match
        ("Select the highest-value tile.", "SELECT", "first"),
        ("Slide the board left.", "SLIDE", "first"),
        ("Inspect the empty cells region.", "INSPECT", "first"),
        ("Read the highest tile indicator.", "READ", "first"),
        ("Compare the two candidate moves.", "COMPARE", "first"),
        ("Stop and treat the state as game over.", "STOP", "first"),
        ("Verify again that the direction is legal.", "VERIFY", "first"),
        # Subordinator-stripped first-word match. After stripping
        # `during/the` the head becomes `slide`, which is the SLIDE
        # head verb — so this lands as SLIDE/first, not EXECUTE.
        ("During the slide, compress tiles toward the chosen side.", "SLIDE", "first"),
        # Downstream walk rescue (head is not in lemma table). After
        # stripping `if/no` the head is `legal`, which is not a verb;
        # we walk downstream and find `stop`.
        ("If no legal direction exists, stop and treat as game over.", "STOP", "rescued"),
        ("The agent should select an adjacent candy.", "SELECT", "rescued"),
        # `each` is now a stripped quantifier; head becomes `turn`,
        # which is the ROTATE head verb. We deliberately don't override
        # because in tetris/2048 prose `turn` more often means rotate
        # than narrative-English "each turn". So this prose lands as
        # ROTATE/first — the test is a guard against the classifier
        # reverting to a more naive "look for English `each turn`".
        ("Each turn, rotate the active piece by 90 degrees.", "ROTATE", "first"),
        # Fallback
        ("Beep boop quux nonsense.", "EXEC", "fallback_exec"),
    ],
)
def test_classify_prose_modes(prose: str, expected_verb: str, expected_mode: str) -> None:
    verb, _bucket, _role, _slots, mode = classify_prose_step(prose)
    assert verb == expected_verb, f"{prose!r} → {verb} (expected {expected_verb})"
    assert mode == expected_mode


def test_empty_prose_falls_back() -> None:
    verb, _, _, _, mode = classify_prose_step("")
    assert verb == "EXEC"
    assert mode == "fallback_exec"


# ───────────── slot population ─────────────


def _make_schema_index(label_to_ontology: Dict[str, str]) -> GameSchemaIndex:
    return GameSchemaIndex(
        game="test_game",
        entity_labels=frozenset(label_to_ontology.keys()),
        label_to_ontology=dict(label_to_ontology),
    )


def test_slot_population_direction_enum() -> None:
    schema = _make_schema_index({})
    payload, types = extract_payload_slots(
        "Slide the board left.",
        slot_signature={"direction": "enum"},
        schema_index=schema,
    )
    assert payload == {"direction": "left"}
    assert types == {"direction": "enum"}


def test_slot_population_entity_match() -> None:
    schema = _make_schema_index({
        "highest_tile": "goal_indicator",
        "board": "container_entity",
    })
    payload, types = extract_payload_slots(
        "Inspect the board for empty cells.",
        slot_signature={"target": "container_entity"},
        schema_index=schema,
    )
    assert payload == {"target": "board"}
    assert types == {"target": "container_entity"}


def test_slot_population_unbound_emits_placeholder() -> None:
    schema = _make_schema_index({"score": "goal_indicator"})
    payload, _ = extract_payload_slots(
        "Move the piece downward.",
        slot_signature={"target": "tracked_entity", "direction": "enum"},
        schema_index=schema,
    )
    # No tracked_entity in the schema → ${target}; direction matches.
    assert payload == {"target": "${target}", "direction": "down"}


# ───────────── effect mining ─────────────


def test_mine_effects_recognises_value_increase() -> None:
    schema = _make_schema_index({"highest_tile": "goal_indicator"})
    add, dele = mine_effects(
        success_criteria=["The highest tile increases by 1."],
        abort_criteria=[],
        schema_index=schema,
    )
    types = [e["type"] for e in add]
    assert "entity_value_increased" in types
    eff = next(e for e in add if e["type"] == "entity_value_increased")
    assert eff["args"].get("entity_label") == "highest_tile"
    assert dele == []


def test_mine_effects_recognises_phase_transition() -> None:
    add, dele = mine_effects(
        success_criteria=[],
        abort_criteria=["The state reaches game over."],
        schema_index=_make_schema_index({}),
    )
    assert add == []
    types = [e["type"] for e in dele]
    assert "phase_transitioned" in types


def test_mine_effects_recognises_reward_increase() -> None:
    add, _ = mine_effects(
        success_criteria=["The cumulative reward goes up after the slide."],
        abort_criteria=[],
        schema_index=_make_schema_index({}),
    )
    types = [e["type"] for e in add]
    assert "cumulative_reward_increased" in types


def test_mine_effects_dedupes_identical_predicates() -> None:
    add, _ = mine_effects(
        success_criteria=[
            "The score increases.",
            "The score also increases over time.",
        ],
        abort_criteria=[],
        schema_index=_make_schema_index({}),
    )
    # Both phrases mine to cumulative_reward_increased with no entity arg → dedup.
    reward_hits = [e for e in add if e["type"] == "cumulative_reward_increased"]
    assert len(reward_hits) == 1


def test_all_predicate_types_are_in_taxonomy() -> None:
    """The mined `type` strings are always in the canonical taxonomy."""

    add, dele = mine_effects(
        success_criteria=[
            "The highest tile increases.",
            "The cumulative reward goes up.",
            "A new piece appears in the playfield.",
            "Lines clear from the bottom.",
        ],
        abort_criteria=[
            "The game ends.",
            "Tile vanishes from the board.",
            "Position changes after a fall.",
            "Value drops below threshold.",
        ],
        schema_index=_make_schema_index({}),
    )
    for e in add + dele:
        assert e["type"] in EFFECT_PREDICATE_TYPES


# ───────────── Day-4 trigger expansion (lift v2) ─────────────
#
# Phase-2 surfaced two miss patterns that block the success_fn from
# producing a strong verdict:
#
#   1. 2048 `Commit/Merge`'s success_criteria include "Any valid merges
#      were applied correctly", which should imply a score increase but
#      under v1 only fired the catch-all `attribute_changed`.
#
#   2. tetris `Commit/Optimize`'s "small line clear or queue advancement
#      … completed" should fire `cumulative_reward_increased` (lines
#      clear award score) on top of the existing `entity_count_changed`.
#
# These tests pin the Day-4 fix so v2 can't regress on real-bank prose.


def test_mine_effects_2048_valid_merges_fires_reward() -> None:
    """The 2048 success criterion 'Any valid merges were applied
    correctly according to 2048 rules' should mine
    `cumulative_reward_increased` — the Day-3 Phase-2 smoke saw only
    `attribute_changed` here, leaving the reward signal undecidable."""

    add, _ = mine_effects(
        success_criteria=[
            "Any valid merges were applied correctly according to 2048 rules, "
            "with no tile merged more than once in the same move.",
        ],
        abort_criteria=[],
        schema_index=_make_schema_index({}),
    )
    types = [e["type"] for e in add]
    assert "cumulative_reward_increased" in types


def test_mine_effects_does_not_overfire_on_disjunctive_merge() -> None:
    """The 2048 phrase 'differs from the previous board by at least one
    tile movement or one merge' is disjunctive — a movement without a
    merge means no score, so we must NOT fire
    `cumulative_reward_increased` here. The catch-all `attribute_changed`
    is the right verdict."""

    add, _ = mine_effects(
        success_criteria=[
            "The resulting board differs from the previous board by at "
            "least one tile movement or one merge.",
        ],
        abort_criteria=[],
        schema_index=_make_schema_index({}),
    )
    types = [e["type"] for e in add]
    assert "cumulative_reward_increased" not in types
    assert "attribute_changed" in types


def test_mine_effects_candy_crush_score_higher_fires_reward() -> None:
    add, _ = mine_effects(
        success_criteria=["The score is higher than before the move resolved."],
        abort_criteria=[],
        schema_index=_make_schema_index({}),
    )
    types = [e["type"] for e in add]
    assert "cumulative_reward_increased" in types


def test_mine_effects_tetris_topout_fires_phase_transition() -> None:
    add, dele = mine_effects(
        success_criteria=[],
        abort_criteria=[
            "Placing vertically on the far left would cause immediate top-out.",
        ],
        schema_index=_make_schema_index({}),
    )
    assert add == []
    types = [e["type"] for e in dele]
    assert "phase_transitioned" in types


def test_mine_effects_candy_crush_moves_decreased_fires_value_decrease() -> None:
    add, _ = mine_effects(
        success_criteria=[
            "The moves remaining count has decreased appropriately for the committed move.",
        ],
        abort_criteria=[],
        schema_index=_make_schema_index({"moves_remaining": "goal_indicator"}),
    )
    types = [e["type"] for e in add]
    assert "entity_value_decreased" in types


def test_mine_effects_does_not_overfire_on_movement_only_phrase() -> None:
    """'The selected slide direction was executed without error' must
    NOT fire any predicate — it's an execution acknowledgement, not a
    state-delta claim."""

    add, _ = mine_effects(
        success_criteria=[
            "The selected slide direction was executed without error.",
        ],
        abort_criteria=[],
        schema_index=_make_schema_index({}),
    )
    assert add == []


# ───────────── Day-5 schema-index whitelist ─────────────


def test_schema_index_whitelist_binds_tetris_holes() -> None:
    """The Day-5 per-game whitelist registers ``holes`` /
    ``stack_height`` / ``filled_cells`` even when cold-start
    schema_canonical didn't enumerate them — so prose like
    ``"Hole count increases from 3 to 4"`` mines an entity_label that
    matches the producer's canonical emission."""

    from labeling._protocol_lift import build_schema_index_for_game

    idx = build_schema_index_for_game(
        actions_root=None,            # no cold-start data → whitelist only
        corpus="env_wrappers",
        game="tetris",
    )
    assert "holes" in idx.entity_labels
    assert "stack_height" in idx.entity_labels
    assert "filled_cells" in idx.entity_labels
    assert "lines_cleared" in idx.entity_labels


def test_first_entity_label_singular_to_plural_fold() -> None:
    """`"Hole count"` (singular) must bind to label `holes` (plural,
    the producer's canonical emission)."""

    from labeling._protocol_lift import _first_entity_label

    idx = _make_schema_index({"holes": "goal_indicator", "hole": "goal_indicator"})
    # Longest form wins → 'holes' (5 > 4 chars), aligned with producer.
    assert _first_entity_label("Hole count increases from 3 to 4", idx) == "holes"


def test_first_entity_label_underscore_to_space_fold() -> None:
    from labeling._protocol_lift import _first_entity_label

    idx = _make_schema_index({"lines_cleared": "goal_indicator"})
    # Both `lines_cleared` and `lines cleared` substrings should bind.
    assert _first_entity_label(
        "lines cleared by the placement: 2", idx
    ) == "lines_cleared"


def test_mine_effects_tetris_hole_count_binds_label() -> None:
    """End-to-end: the Day-4 prose phrase ``"Hole count increases from
    3 to 4"`` was mined as ``entity_count_changed`` with ``args={}``
    under Day-3/4. Day-5 whitelist + plural fold makes the lift bind
    ``entity_label="holes"`` so the runtime predicate evaluator can
    look up ``entity_label_count["holes"]`` on the producer's
    output."""

    from labeling._protocol_lift import build_schema_index_for_game

    idx = build_schema_index_for_game(
        actions_root=None,
        corpus="env_wrappers",
        game="tetris",
    )
    add, _ = mine_effects(
        success_criteria=[
            "Hole count increases from 3 to 4, with no worse unintended "
            "damage elsewhere.",
        ],
        abort_criteria=[],
        schema_index=idx,
    )
    types_to_args = {e["type"]: e["args"] for e in add}
    # Trigger order is most-specific-first: 'hole count' fires
    # entity_count_changed before the catch-all attribute_changed.
    assert "entity_count_changed" in types_to_args
    assert types_to_args["entity_count_changed"].get("entity_label") == "holes"


# ───────────── lift orchestrator ─────────────


@pytest.fixture
def schema_2048() -> GameSchemaIndex:
    return _make_schema_index({
        "board": "container_entity",
        "tile_2": "selectable_entity",
        "empty_cells": "navigable_region",
        "highest_tile": "goal_indicator",
        "score": "goal_indicator",
    })


def test_lift_smoke_2048_commit_merge(schema_2048: GameSchemaIndex) -> None:
    """End-to-end lift on a real-shape 2048 COMMIT/MERGE skill body."""

    skill = {
        "skill_id": "COMMIT/MERGE",
        "evidence_role": "COMMIT",
        "protocol": {
            "preconditions": ["Board has at least one legal move."],
            "steps": [
                "Read the current board and enumerate the four possible slide directions: up, down, left, right.",
                "Determine which directions are legal.",
                "If no legal direction exists, stop and treat as game over.",
                "Select the intended direction according to the policy.",
                "Verify again that the selected direction is legal.",
                "Execute a single slide of the full board in the selected direction.",
                "During the slide, compress tiles and merge each pair of adjacent equal tiles at most once.",
            ],
            "success_criteria": [
                "The selected slide direction was executed without error.",
                "The resulting board differs from the previous board.",
                "Highest tile may increase.",
            ],
            "abort_criteria": [
                "No legal slide direction exists because no move would change the board.",
                "The game ends.",
            ],
        },
    }
    stats = LiftStats()
    typed, contract_add, contract_del = lift_protocol_to_typed_hops(
        skill, schema_index=schema_2048, stats=stats,
    )
    assert typed is not None and len(typed) == 7

    # Every hop has an op in the taxonomy ∪ {EXEC}.
    for hop in typed:
        assert hop["op"] in (VERB_TAXONOMY | {"EXEC"})

    # Verbs we expect to land:
    ops = [h["op"] for h in typed]
    assert "READ" in ops or "INSPECT" in ops    # step 1
    assert "EVALUATE" in ops or "INSPECT" in ops  # step 2
    assert "STOP" in ops                         # step 3
    assert "SELECT" in ops                       # step 4
    assert "VERIFY" in ops                       # step 5
    assert "EXECUTE" in ops or "SLIDE" in ops    # step 6 / 7

    # Fallback rate is 0 on this hand-curated body.
    assert stats.n_hops == 7
    assert stats.n_fallback_exec == 0

    # Effects rolled up onto env-mutating hops only.
    env_hops = [h for h in typed if h["op"] in {
        "SELECT", "SWAP", "SLIDE", "MOVE", "ROTATE", "DROP", "PLACE",
        "APPROACH", "EXECUTE",
    }]
    if env_hops:
        non_empty = [h for h in env_hops if h["effects_add"]]
        assert non_empty, "expected at least one env hop to carry effects"

    # Contract roll-up matches the union of per-hop predicates.
    assert "phase_transitioned" in contract_del


def test_lift_idempotent_on_already_lifted(schema_2048: GameSchemaIndex) -> None:
    """Re-running the lift on a list-of-dicts protocol whose ops are
    already in-taxonomy must be a no-op (return None)."""

    skill = {
        "evidence_role": "COMMIT",
        "protocol": [
            {"op": "SLIDE", "payload": {"direction": "up"}, "notes": "already lifted"},
            {"op": "EXECUTE", "payload": {}, "notes": "already lifted 2"},
        ],
    }
    typed, _, _ = lift_protocol_to_typed_hops(skill, schema_index=schema_2048)
    assert typed is None


def test_lift_relifts_shape_only_workaround(schema_2048: GameSchemaIndex) -> None:
    """The dump-driver `_wrap_protocol_steps` shape-lift produces hops whose
    op is `"EXEC"` but whose `notes` carry the original prose. A subsequent
    decorator pass should re-classify those hops semantically."""

    skill = {
        "evidence_role": "COMMIT",
        "protocol": [
            {"action": "EXEC", "payload": {}, "notes": "Slide the board up."},
            {"action": "EXEC", "payload": {}, "notes": "Inspect the board."},
        ],
    }
    typed, _, _ = lift_protocol_to_typed_hops(skill, schema_index=schema_2048)
    assert typed is not None
    assert [h["op"] for h in typed] == ["SLIDE", "INSPECT"]


def test_lift_handles_empty_protocol(schema_2048: GameSchemaIndex) -> None:
    skill = {"protocol": {"steps": []}}
    typed, add, dele = lift_protocol_to_typed_hops(skill, schema_index=schema_2048)
    assert typed is None
    assert add == [] and dele == []


# ───────────── coverage smoke against the live cold-start corpus ─────────────


def _real_bank_root() -> Path:
    return Path(__file__).resolve().parents[1] / (
        "labeling/skill_bank_out/run_20260430_030637/env_wrappers"
    )


@pytest.mark.skipif(
    not _real_bank_root().exists(),
    reason="cold-start bank not present in this checkout",
)
def test_real_bank_fallback_exec_under_threshold() -> None:
    """`implementation_notes/legacy/protocol-lift-design.md` §8 acceptance gate:
    `lift_fallback_exec_pct ≤ 10 %` over the real cold-start corpus
    (env_wrappers slice — gym_v arcade ROMs need an extended verb set,
    out of scope for v0 per design §9).

    We sweep all 4 env_wrappers games and assert the aggregate fallback
    rate clears the gate. Re-runs against an already-lifted bank read
    from `skill["protocol_raw"]` (the prose body the lift preserved
    when it ran in-place) so the test is idempotent against decorator
    invocations between test runs.
    """

    bank_root = _real_bank_root()
    actions_root = bank_root.parent.parent.parent / (
        "skill_actions_out/run_20260430_064325"
    )
    if not actions_root.exists():
        pytest.skip("skill_actions_out/run_20260430_064325 not present")

    from labeling._protocol_lift import build_schema_index_for_game

    total = LiftStats()
    for game_dir in sorted(p for p in bank_root.iterdir() if p.is_dir()):
        bank_jsonl = game_dir / "skill_bank.jsonl"
        if not bank_jsonl.exists():
            continue
        idx = build_schema_index_for_game(
            actions_root, corpus="env_wrappers", game=game_dir.name,
        )
        for line in bank_jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            skill = entry.get("skill") if isinstance(entry.get("skill"), dict) else entry
            # If the bank has already been v2-lifted in-place by the
            # decorator, `skill["protocol"]` is the typed-hops form and
            # `skill["protocol_raw"]` is the original prose dict. Re-lift
            # the prose to recover an apples-to-apples coverage number.
            raw = skill.get("protocol_raw")
            if isinstance(raw, dict):
                skill = dict(skill)
                skill["protocol"] = raw
            lift_protocol_to_typed_hops(skill, schema_index=idx, stats=total)

    assert total.n_hops > 0, "no hops lifted from real bank"
    pct = 100 * total.fallback_exec_pct
    # §8 gate is ≤ 10 %; we expect ~2.5 % after the v0 lemma table.
    assert pct <= 10.0, (
        f"fallback_exec_pct={pct:.1f}% exceeds 10% gate; "
        f"verb counts={total.verbs}"
    )
