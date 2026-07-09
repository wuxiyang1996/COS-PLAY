"""Tests for the deterministic ``<state>``-block producer
(``harness/gym_schema_producer.py``).

The producer's contract is round-trip with
``labeling_supplement._harness_io_helpers.parse_schema_canonical``: every
hop the lift mines becomes evaluable when the producer is wired into
``make_gymv_executor``. These tests pin:

  1. The block parses cleanly via ``parse_schema_canonical``.
  2. ``StateSchema.facts`` carries the fields the predicate evaluator
     keys on (``score``, ``highest_tile``, ``phase``, ``entity_attrs``,
     ``entity_label_count``).
  3. Numeric predicates (``cumulative_reward_increased``,
     ``entity_value_increased``) are decidable on the producer's
     output — i.e. Day-3's "entity_attrs missing on both sides" failure
     mode is gone.
  4. The registry returns the right producer for known games and
     ``None`` for unsupported ones.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from harness.gym_schema_producer import (
    candy_crush_producer,
    make_gaming_env_producer,
    render_state_block,
    super_mario_producer,
    tetris_producer,
    twenty_forty_eight_producer,
)
from harness.gymv_success import evaluate_predicate
from labeling_supplement._harness_io_helpers import parse_schema_canonical


# ---------------------------------------------------------------------------
# Round-trip: producer → parse_schema_canonical → StateSchema
# ---------------------------------------------------------------------------


def test_2048_producer_parses_and_surfaces_facts() -> None:
    info: Dict[str, Any] = {
        "board": [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
        ],
        "total_score": 0,
        "step_score": 0,
        "max_tile_power": 1,           # 2 ** 1 = 2
        "is_legal_move": True,
        "action_names": ["up", "right", "down", "left"],
    }
    block = twenty_forty_eight_producer(info, {"text": "ignored"}, step=0)

    assert "<state>" in block and "</state>" in block
    assert "<entities>" in block
    assert "<attributes>" in block
    assert "<state_flags>" in block

    state = parse_schema_canonical(block, default_domain="gymv")

    # Hot-path scalars promoted into `facts`.
    assert state.facts.get("score") == 0
    assert state.facts.get("highest_tile") == 2
    assert state.facts.get("phase") == "play"

    # Per-label entity attribute table (used by `entity_value_*`).
    eattrs = state.facts.get("entity_attrs") or {}
    assert "tile_2" in eattrs
    # tile_2.value should round-trip as the integer 2.
    assert int(eattrs["tile_2"]["value"]) == 2

    # Per-label count (used by `entity_count_changed`, `entity_appeared`).
    ecount = state.facts.get("entity_label_count") or {}
    assert ecount.get("tile_2") == 2  # two tiles of value 2 on the board


def test_2048_producer_phase_is_gameover_when_full_and_illegal() -> None:
    info: Dict[str, Any] = {
        "board": [
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ],
        "total_score": 100,
        "max_tile_power": 4,           # 16
        # Last move was illegal AND every cell is filled → gameover.
        "is_legal_move": False,
    }
    block = twenty_forty_eight_producer(info, None, step=5)
    state = parse_schema_canonical(block, default_domain="gymv")
    assert state.facts.get("phase") == "gameover"
    assert state.facts.get("highest_tile") == 16


def test_2048_producer_handles_numpy_like_board() -> None:
    """The 2048 env emits ``np.uint8`` cells; coercion must survive."""
    import numpy as np
    info: Dict[str, Any] = {
        "board": np.asarray([
            [np.uint8(2), 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [np.uint8(2), 0, 0, 0],
        ]),
        "total_score": np.int64(0),
        "max_tile_power": np.int64(1),
    }
    block = twenty_forty_eight_producer(info, None, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    assert state.facts.get("score") == 0
    assert state.facts.get("highest_tile") == 2


# ---------------------------------------------------------------------------
# Predicate decidability (the headline Day-4B win)
# ---------------------------------------------------------------------------


def test_cumulative_reward_increased_is_decidable_on_producer_output() -> None:
    """Day-3's failure mode was 'score → 0.0 → 0.0' or 'undecidable'
    when the env returned a plain text obs. Day-4B fixes the latter:
    when the producer surfaces a real score, the predicate evaluator
    must compare numerics, not flag missing facts."""

    pre_block = twenty_forty_eight_producer(
        {"board": [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
         "total_score": 0, "max_tile_power": 1},
        None, step=0,
    )
    post_block = twenty_forty_eight_producer(
        {"board": [[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         "total_score": 4, "max_tile_power": 2},   # one merge, score=4
        None, step=1,
    )
    pre = parse_schema_canonical(pre_block, default_domain="gymv")
    post = parse_schema_canonical(post_block, default_domain="gymv")

    result = evaluate_predicate(
        {"type": "cumulative_reward_increased", "args": {}},
        pre, post,
    )
    assert result.passed is True, result.detail


def test_entity_value_increased_is_decidable_on_producer_output() -> None:
    pre_block = twenty_forty_eight_producer(
        {"board": [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]],
         "total_score": 0, "max_tile_power": 1},
        None, step=0,
    )
    post_block = twenty_forty_eight_producer(
        {"board": [[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
         "total_score": 4, "max_tile_power": 2},
        None, step=1,
    )
    pre = parse_schema_canonical(pre_block, default_domain="gymv")
    post = parse_schema_canonical(post_block, default_domain="gymv")

    result = evaluate_predicate(
        {"type": "entity_value_increased",
         "args": {"entity_label": "highest_tile"}},
        pre, post,
    )
    assert result.passed is True, result.detail


def test_entity_count_changed_is_decidable_for_tetris_holes() -> None:
    """Day-3 left ``entity_count_changed`` undecidable for tetris's
    'Hole count increases from 3 to 4' criterion because the
    GamingAgent text obs didn't surface holes. Day-4B's tetris
    producer counts holes itself, making the predicate decide."""

    pre_obs = {"text": (
        "Board:\n"
        ".....T....\n"
        "....TTT...\n"
        "..........\n"
        "..........\n"
        + "..........\n" * 16
        + "( '.' = empty, IOTSZJL = tetrominoes. Active piece is rendered on board.)"
    )}
    post_obs = {"text": (
        "Board:\n"
        ".....T....\n"
        "....TTT...\n"
        "..........\n"
        "X.X.X.X.X.\n"          # row with gaps over empty space
        ".X.X.X.X.X\n"          # creates holes (empty under filled cells)
        + "..........\n" * 15
        + "( '.' = empty, IOTSZJL = tetrominoes. Active piece is rendered on board.)"
    )}
    pre_block = tetris_producer(
        {"score": 0, "lines": 0, "level": 1, "next_piece_ids": [1, 2, 3, 4]},
        pre_obs, step=0,
    )
    post_block = tetris_producer(
        {"score": 0, "lines": 0, "level": 1, "next_piece_ids": [1, 2, 3, 4]},
        post_obs, step=1,
    )
    pre = parse_schema_canonical(pre_block, default_domain="gymv")
    post = parse_schema_canonical(post_block, default_domain="gymv")

    result = evaluate_predicate(
        {"type": "entity_value_increased",
         "args": {"entity_label": "holes"}},
        pre, post,
    )
    assert result.passed is True, result.detail


# ---------------------------------------------------------------------------
# Tetris producer
# ---------------------------------------------------------------------------


def test_tetris_producer_surfaces_score_lines_level() -> None:
    info: Dict[str, Any] = {
        "score": 100.0, "lines": 5, "level": 2,
        "next_piece_ids": [1, 2, 3, 4],
        "action_names": ["no_op", "left", "right",
                         "rotate_left", "rotate_right",
                         "soft_drop", "hard_drop"],
    }
    obs = {"text": (
        "Board:\n"
        ".....T....\n"
        "....TTT...\n"
        + "..........\n" * 18
        + "( '.' = empty, IOTSZJL = tetrominoes. Active piece is rendered on board.)"
    )}
    block = tetris_producer(info, obs, step=10)
    state = parse_schema_canonical(block, default_domain="gymv")

    assert state.facts.get("score") == 100
    assert state.facts.get("phase") == "play"
    eattrs = state.facts.get("entity_attrs") or {}
    assert "score" in eattrs
    assert "lines_cleared" in eattrs and int(eattrs["lines_cleared"]["value"]) == 5
    assert "level" in eattrs and int(eattrs["level"]["value"]) == 2
    # Active piece detected from the text obs.
    assert "active_piece_T" in eattrs


def test_tetris_producer_handles_empty_text_obs() -> None:
    """When obs has no 'Board:' marker the producer still emits a
    well-formed block; the active piece simply isn't surfaced."""
    info = {"score": 0, "lines": 0, "level": 1, "next_piece_ids": []}
    block = tetris_producer(info, {"text": ""}, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    assert state.facts.get("phase") == "play"
    # No active piece entry — but `score`, `lines_cleared`, etc. exist.
    eattrs = state.facts.get("entity_attrs") or {}
    assert "score" in eattrs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_make_gaming_env_producer_known_games() -> None:
    assert make_gaming_env_producer("twenty_forty_eight") is twenty_forty_eight_producer
    assert make_gaming_env_producer("tetris") is tetris_producer
    # Day-6 producers.
    assert make_gaming_env_producer("candy_crush") is candy_crush_producer
    assert make_gaming_env_producer("super_mario") is super_mario_producer


def test_make_gaming_env_producer_unknown_returns_none() -> None:
    assert make_gaming_env_producer("nonsense") is None


# ---------------------------------------------------------------------------
# candy_crush producer (Day-6)
# ---------------------------------------------------------------------------


_CANDY_OBS = (
    "Board:\n"
    "0| R C G C P P C R\n"
    "1| P C R R C G P P\n"
    "2| R G C G P R G C\n"
    "3| G R C P C C R G\n"
    "4| P C R G C G C R\n"
    "5| R R G G R G R C\n"
    "6| R C C P G P P R\n"
    "7| P G P P R P C R\n"
    "Score: 0\n"
    "Moves Left: 50"
)


def test_candy_crush_producer_round_trips() -> None:
    info = {"action_names": ["((0,0),(1,0))", "((0,1),(1,1))"]}
    block = candy_crush_producer(info, {"text": _CANDY_OBS}, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    assert state.facts.get("phase") == "play"
    assert state.facts.get("score") == 0
    assert state.facts.get("moves_remaining") == 50
    eattrs = state.facts.get("entity_attrs") or {}
    # Per-color aggregate counts.
    assert "candy_red" in eattrs
    assert "candy_green" in eattrs
    assert "candy_purple" in eattrs
    assert "candy_cyan" in eattrs
    # Goal indicators.
    assert "score" in eattrs
    assert "moves_remaining" in eattrs


def test_candy_crush_producer_prefers_info_over_text() -> None:
    """When ``info`` carries explicit ``score`` / ``moves_remaining``,
    they override the text-parsed values."""
    info = {"score": 100, "moves_remaining": 12}
    block = candy_crush_producer(info, {"text": _CANDY_OBS}, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    assert state.facts.get("score") == 100
    assert state.facts.get("moves_remaining") == 12


def test_candy_crush_producer_phase_gameover_when_no_moves() -> None:
    obs = _CANDY_OBS.replace("Moves Left: 50", "Moves Left: 0")
    block = candy_crush_producer({}, {"text": obs}, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    assert state.facts.get("phase") == "gameover"


def test_candy_crush_score_increase_is_decidable() -> None:
    pre = parse_schema_canonical(
        candy_crush_producer({"score": 0, "moves_remaining": 50},
                             {"text": _CANDY_OBS}, step=0),
        default_domain="gymv")
    post = parse_schema_canonical(
        candy_crush_producer({"score": 30, "moves_remaining": 49},
                             {"text": _CANDY_OBS}, step=1),
        default_domain="gymv")
    r = evaluate_predicate(
        {"type": "cumulative_reward_increased", "args": {}}, pre, post,
    )
    assert r.passed is True, r.detail


# ---------------------------------------------------------------------------
# super_mario producer (Day-6)
# ---------------------------------------------------------------------------


_MARIO_OBS = (
    "Position of Mario: (122, 45)\n"
    "Positions of all objects\n"
    "- Bricks: None\n"
    "- Question Blocks: (214, 96)\n"
    "- Inactivated Blocks: None\n"
    "- Monster Goomba: (300, 100)\n"
    "- Monster Koopas: None\n"
    "- Pit: None\n"
    "- Warp Pipe: None\n"
    "- Item Mushrooms: None\n"
    "- Stair Blocks: None\n"
    "- Flag: None\n"
)


def test_super_mario_producer_round_trips() -> None:
    info = {"score": 200, "lives": 3,
            "action_names": [f"Jump Level: {i}" for i in range(7)]}
    block = super_mario_producer(info, {"text": _MARIO_OBS}, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    eattrs = state.facts.get("entity_attrs") or {}
    assert "mario" in eattrs
    # Visible world objects from the text obs.
    assert "question_blocks" in eattrs
    assert "monster_goomba" in eattrs
    # Goal-indicator scalars.
    assert "score" in eattrs and int(eattrs["score"]["value"]) == 200
    assert "lives" in eattrs and int(eattrs["lives"]["value"]) == 3
    assert "scroll_x" in eattrs and int(eattrs["scroll_x"]["value"]) == 122


def test_super_mario_progress_advances_with_scroll_x() -> None:
    pre_obs = _MARIO_OBS
    post_obs = _MARIO_OBS.replace(
        "Position of Mario: (122, 45)",
        "Position of Mario: (500, 45)",
    )
    pre = parse_schema_canonical(
        super_mario_producer({"score": 100}, {"text": pre_obs}, step=0),
        default_domain="gymv")
    post = parse_schema_canonical(
        super_mario_producer({"score": 100}, {"text": post_obs}, step=1),
        default_domain="gymv")
    r = evaluate_predicate(
        {"type": "entity_value_increased",
         "args": {"entity_label": "scroll_x"}}, pre, post,
    )
    assert r.passed is True, r.detail


def test_super_mario_producer_handles_no_objects() -> None:
    obs = "Position of Mario: (50, 100)\nPositions of all objects\n"
    block = super_mario_producer({"score": 0, "lives": 3}, {"text": obs}, step=0)
    state = parse_schema_canonical(block, default_domain="gymv")
    eattrs = state.facts.get("entity_attrs") or {}
    assert "mario" in eattrs
    assert "score" in eattrs


# ---------------------------------------------------------------------------
# render_state_block scaffolding
# ---------------------------------------------------------------------------


def test_render_state_block_minimum_sections_parse() -> None:
    block = render_state_block(
        domain="gymv", task="x", goal="y", step=3,
        entities=[{
            "id": "e1", "type": "object", "label": "tile_4",
            "bid": "null", "pos": "0,0,1,1",
            "ontology": "selectable_entity",
        }],
        attributes={"e1": {"value": 4, "state": "visible"}},
        state_flags={"phase": "play"},
    )
    state = parse_schema_canonical(block, default_domain="gymv")
    eattrs = state.facts.get("entity_attrs") or {}
    assert int(eattrs["tile_4"]["value"]) == 4
    assert state.facts.get("phase") == "play"
