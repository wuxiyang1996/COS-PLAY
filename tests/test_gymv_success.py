"""Unit tests for `harness.gymv_success` predicate evaluators.

PLAN-HARNESS §22 (Day-3). These cover the eight predicate types from the
protocol-lift taxonomy in isolation, plus the per-hop / per-episode
roll-ups that `GymvAdapter.run` and the orchestrator's gate path consume.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from common.state_schema import StateSchema
from harness.gymv_success import (
    EFFECT_PREDICATE_TYPES,
    HopEffectResult,
    PredicateResult,
    evaluate_hop_effects,
    evaluate_predicate,
    make_per_step_success_fn,
)


def _state(facts: Dict[str, Any], *, task: str = "twenty_forty_eight") -> StateSchema:
    return StateSchema(task=task, domain="gymv", facts=dict(facts))


# ───────────── per-predicate evaluators ─────────────


def test_taxonomy_completeness() -> None:
    """The runtime taxonomy must agree with the labeling-side one. If
    the lift adds a new predicate type, this test pulls in the symbol so
    we get a single failure pointing at the missing evaluator branch."""

    from labeling._protocol_lift import EFFECT_PREDICATE_TYPES as LIFT_TYPES
    assert set(EFFECT_PREDICATE_TYPES) == set(LIFT_TYPES)


def test_cumulative_reward_increased_passes_when_score_grows() -> None:
    pre = _state({"score": 0})
    post = _state({"score": 4})
    res = evaluate_predicate({"type": "cumulative_reward_increased", "args": {}}, pre, post)
    assert res.passed is True
    assert "0" in res.detail


def test_cumulative_reward_undecidable_when_score_missing() -> None:
    pre = _state({})
    post = _state({"score": 4})
    res = evaluate_predicate({"type": "cumulative_reward_increased", "args": {}}, pre, post)
    assert res.passed is None


def test_entity_value_increased_uses_entity_attrs_fallback() -> None:
    """When the parser didn't promote a hot-path scalar, the evaluator
    should still find the value via `facts["entity_attrs"][label]["value"]`."""

    pre = _state({"entity_attrs": {"power_up": {"value": 1}}})
    post = _state({"entity_attrs": {"power_up": {"value": 3}}})
    res = evaluate_predicate(
        {"type": "entity_value_increased", "args": {"entity_label": "power_up"}},
        pre, post,
    )
    assert res.passed is True


def test_entity_value_decreased_passes_when_value_shrinks() -> None:
    pre = _state({"highest_tile": 8})
    post = _state({"highest_tile": 4})
    res = evaluate_predicate(
        {"type": "entity_value_decreased", "args": {"entity_label": "highest_tile"}},
        pre, post,
    )
    assert res.passed is True


def test_entity_count_changed_detects_count_delta() -> None:
    pre = _state({"entity_label_count": {"tile_2": 4}})
    post = _state({"entity_label_count": {"tile_2": 2}})
    res = evaluate_predicate(
        {"type": "entity_count_changed", "args": {"entity_label": "tile_2"}},
        pre, post,
    )
    assert res.passed is True


def test_entity_appeared_passes_when_count_grows() -> None:
    pre = _state({"entity_label_count": {"tile_4": 0}})
    post = _state({"entity_label_count": {"tile_4": 1}})
    res = evaluate_predicate(
        {"type": "entity_appeared", "args": {"entity_label": "tile_4"}},
        pre, post,
    )
    assert res.passed is True


def test_entity_disappeared_passes_when_count_shrinks() -> None:
    pre = _state({"entity_label_count": {"tile_2": 2}})
    post = _state({"entity_label_count": {"tile_2": 0}})
    res = evaluate_predicate(
        {"type": "entity_disappeared", "args": {"entity_label": "tile_2"}},
        pre, post,
    )
    assert res.passed is True


def test_phase_transitioned_to_specific_target() -> None:
    pre = _state({"phase": "play"})
    post = _state({"phase": "gameover"})
    res = evaluate_predicate(
        {"type": "phase_transitioned", "args": {"to": "gameover"}},
        pre, post,
    )
    assert res.passed is True

    # Same target on both sides → not a transition.
    res2 = evaluate_predicate(
        {"type": "phase_transitioned", "args": {"to": "gameover"}},
        post, post,
    )
    assert res2.passed is False


def test_attribute_changed_catch_all() -> None:
    pre = _state({"entity_attrs": {"piece": {"pos": "0,0,1,1"}}})
    post = _state({"entity_attrs": {"piece": {"pos": "0,1,1,1"}}})
    res = evaluate_predicate({"type": "attribute_changed", "args": {}}, pre, post)
    assert res.passed is True


def test_unknown_predicate_is_undecidable() -> None:
    res = evaluate_predicate({"type": "totally_made_up", "args": {}},
                             _state({}), _state({}))
    assert res.passed is None
    assert "unknown" in res.detail


# ───────────── per-hop and per-episode roll-up ─────────────


def test_evaluate_hop_effects_passes_when_all_predicates_pass() -> None:
    pre = _state({"score": 0, "highest_tile": 2})
    post = _state({"score": 4, "highest_tile": 4})
    hop = {
        "op": "SLIDE",
        "effects_add": [
            {"type": "cumulative_reward_increased", "args": {}},
            {"type": "entity_value_increased", "args": {"entity_label": "highest_tile"}},
        ],
    }
    res = evaluate_hop_effects(hop, pre, post)
    assert isinstance(res, HopEffectResult)
    assert res.passed is True
    assert res.n_required == 2
    assert res.n_passed == 2
    assert res.n_violated == 0


def test_evaluate_hop_effects_fails_on_any_violation() -> None:
    pre = _state({"score": 0, "highest_tile": 4})
    post = _state({"score": 4, "highest_tile": 2})  # highest_tile shrank
    hop = {
        "op": "SLIDE",
        "effects_add": [
            {"type": "cumulative_reward_increased", "args": {}},
            {"type": "entity_value_increased", "args": {"entity_label": "highest_tile"}},
        ],
    }
    res = evaluate_hop_effects(hop, pre, post)
    assert res.passed is False
    assert res.n_violated == 1


def test_evaluate_hop_effects_undecidable_does_not_block() -> None:
    """A hop whose predicates all skip (e.g. unsurfaced labels) is
    treated as `passed=True` so transfer probes against partial parsers
    don't blanket-fail."""

    pre = _state({})
    post = _state({})
    hop = {
        "op": "SLIDE",
        "effects_add": [
            {"type": "entity_value_increased", "args": {"entity_label": "ghost"}},
        ],
    }
    res = evaluate_hop_effects(hop, pre, post)
    assert res.passed is True
    assert res.n_passed == 0
    assert res.n_undecidable == 1


# ───────────── make_per_step_success_fn ─────────────


def test_per_step_success_fn_returns_one_when_all_hops_pass() -> None:
    from data_structure.extensions.skill_episode import (
        SkillEpisode, SkillEpisodeOutcome,
    )
    from common.enums import SkillType

    episode = SkillEpisode(
        episode_id="ep-1",
        skill_id="sk-1",
        skill_version="v0",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
    )
    episode.outcome = SkillEpisodeOutcome(
        success=True,
        contract_satisfied=True,
        evidence_role=["COMMIT"],
        extra={
            "per_hop_effects": {
                "n_hops_evaluated": 2,
                "n_hops_passed": 2,
                "pass_rate": 1.0,
            }
        },
    )

    score_fn = make_per_step_success_fn(pass_rate_threshold=1.0)
    assert score_fn(episode, None) == 1.0


def test_per_step_success_fn_returns_zero_when_below_threshold() -> None:
    from data_structure.extensions.skill_episode import (
        SkillEpisode, SkillEpisodeOutcome,
    )
    from common.enums import SkillType

    episode = SkillEpisode(
        episode_id="ep-2",
        skill_id="sk-2",
        skill_version="v0",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
    )
    episode.outcome = SkillEpisodeOutcome(
        success=True,
        contract_satisfied=True,
        evidence_role=["COMMIT"],
        extra={
            "per_hop_effects": {
                "n_hops_evaluated": 4,
                "n_hops_passed": 1,
                "pass_rate": 0.25,
            }
        },
    )

    score_fn = make_per_step_success_fn(pass_rate_threshold=0.5)
    assert score_fn(episode, None) == 0.0


def test_per_step_success_fn_falls_back_to_outcome_when_no_predicates() -> None:
    from data_structure.extensions.skill_episode import (
        SkillEpisode, SkillEpisodeOutcome,
    )
    from common.enums import SkillType

    episode = SkillEpisode(
        episode_id="ep-3",
        skill_id="sk-3",
        skill_version="v0",
        skill_type=SkillType.ACTION,
        domain="gymv",
        parent_run_id=None,
    )
    episode.outcome = SkillEpisodeOutcome(
        success=True,
        contract_satisfied=True,
        evidence_role=["COMMIT"],
        extra={},  # no per_hop_effects
    )
    score_fn = make_per_step_success_fn()
    assert score_fn(episode, None) == 1.0
