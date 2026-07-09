"""Tests for the game-specific critical-action prior.

Locks in two behaviors added after the TF3-phase-1-collapse post-mortem:

1. ``_critical_actions_for`` returns the GAME_CRITICAL_ACTIONS subset
   that's actually exposed by this episode's ``valid_actions`` list.
   (Gymv adapters expose a curated subset; we silently drop critical
   actions that aren't present rather than crashing.)

2. ``_apply_anti_repetition`` now has two new behaviors on top of the
   pre-existing single-action loop break:
     - When breaking a stuck-on-one-action loop, prefer a critical
       action over a random alternative.
     - During a "critical-action dry spell" (8 consecutive
       zero-reward decisions with no critical action picked), force-
       substitute the critical action even if the policy isn't stuck
       on one specific action.

The third effect (in-context prompt hint) is exercised in the
end-to-end episode tests via the full run_episode fixture; the unit
test here just verifies that the helper returns the expected list so
the prompt assembly has something to work with.
"""

from __future__ import annotations

from trainer.coevolution.episode_runner import (
    _apply_anti_repetition,
    _critical_actions_for,
    _CRITICAL_ACTION_DRY_SPELL,
)


# ---------------------------------------------------------------------------
# _critical_actions_for
# ---------------------------------------------------------------------------


def test_critical_actions_for_known_shooter():
    actions = _critical_actions_for(
        "gymv_thunder_force_iii",
        ["UP", "DOWN", "LEFT", "RIGHT", "B", "A", "START"],
    )
    assert actions == ["B"]


def test_critical_actions_for_skips_missing_actions():
    """B isn't in this game's exposed action list ⇒ silently dropped."""
    actions = _critical_actions_for(
        "gymv_thunder_force_iii",
        ["UP", "DOWN", "LEFT", "RIGHT"],   # no B
    )
    assert actions == []


def test_critical_actions_for_unknown_game():
    actions = _critical_actions_for("twenty_forty_eight", ["UP", "DOWN", "LEFT", "RIGHT"])
    assert actions == []


def test_critical_actions_for_brawler_subset():
    actions = _critical_actions_for(
        "gymv_streets_of_rage_2",
        ["UP", "DOWN", "B", "C", "A"],
    )
    assert "B" in actions


# ---------------------------------------------------------------------------
# _apply_anti_repetition — loop break with critical preference
# ---------------------------------------------------------------------------


def test_loop_break_prefers_critical_action_for_shooter():
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "B", "A", "START"]
    out = _apply_anti_repetition(
        "UP", valid_actions=valid,
        recent_actions=["UP", "UP"],
        recent_rewards=[0.0, 0.0],
        game="gymv_thunder_force_iii",
    )
    # B (fire) should win over a random alternative.
    assert out == "B"


def test_loop_break_falls_back_to_random_when_no_critical_action(monkeypatch):
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "B"]
    # Force `random.choice` to be deterministic for the test.
    import random
    monkeypatch.setattr(random, "choice", lambda xs: xs[0])
    out = _apply_anti_repetition(
        "UP", valid_actions=valid,
        recent_actions=["UP", "UP"],
        recent_rewards=[0.0, 0.0],
        game="twenty_forty_eight",   # no critical actions configured
    )
    # First non-`UP` alternative wins (random.choice stubbed).
    assert out == "DOWN"


def test_loop_break_does_not_replace_critical_with_critical():
    valid = ["UP", "DOWN", "B"]
    out = _apply_anti_repetition(
        "B", valid_actions=valid,
        recent_actions=["B", "B"],
        recent_rewards=[0.0, 0.0],
        game="gymv_thunder_force_iii",
    )
    # The only critical action IS the stuck action; nothing better to
    # substitute, so the loop break still has to pick a non-critical
    # alternative (UP or DOWN).
    assert out in {"UP", "DOWN"}


# ---------------------------------------------------------------------------
# Critical-action dry spell
# ---------------------------------------------------------------------------


def test_dry_spell_substitutes_critical_after_window():
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "B"]
    # 8 consecutive non-B picks with zero reward
    recent = ["UP", "RIGHT", "LEFT", "UP", "DOWN", "RIGHT", "UP", "LEFT"]
    rewards = [0.0] * 8
    assert len(recent) == _CRITICAL_ACTION_DRY_SPELL

    out = _apply_anti_repetition(
        "RIGHT", valid_actions=valid,
        recent_actions=recent, recent_rewards=rewards,
        game="gymv_thunder_force_iii",
    )
    assert out == "B", "expected dry-spell substitution of critical action"


def test_dry_spell_does_not_fire_with_recent_critical_action():
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "B"]
    # Picked B once in the window ⇒ not a dry spell.
    recent = ["UP", "RIGHT", "B", "UP", "DOWN", "RIGHT", "UP", "LEFT"]
    rewards = [0.0] * 8
    out = _apply_anti_repetition(
        "RIGHT", valid_actions=valid,
        recent_actions=recent, recent_rewards=rewards,
        game="gymv_thunder_force_iii",
    )
    # No substitution — RIGHT was the LLM's pick.
    assert out == "RIGHT"


def test_dry_spell_does_not_fire_when_reward_is_positive():
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "B"]
    recent = ["UP", "RIGHT", "LEFT", "UP", "DOWN", "RIGHT", "UP", "LEFT"]
    rewards = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    out = _apply_anti_repetition(
        "RIGHT", valid_actions=valid,
        recent_actions=recent, recent_rewards=rewards,
        game="gymv_thunder_force_iii",
    )
    # Positive reward in window ⇒ trust the policy, no substitution.
    assert out == "RIGHT"


def test_dry_spell_does_not_fire_for_unconfigured_game():
    """No GAME_CRITICAL_ACTIONS entry ⇒ the dry-spell branch is a no-op."""
    valid = ["UP", "DOWN", "LEFT", "RIGHT"]
    recent = ["UP"] * _CRITICAL_ACTION_DRY_SPELL
    # Pre-existing loop-break still applies (8 of the same action,
    # all zero reward), but it shouldn't fall into the dry-spell branch.
    out = _apply_anti_repetition(
        "UP", valid_actions=valid,
        recent_actions=recent, recent_rewards=[0.0] * 8,
        game="tetris",
    )
    # Still gets broken out of the loop (UP repeated), but via the
    # generic random-alternative path; we just need it to NOT raise
    # and to NOT pick UP again.
    assert out != "UP"


def test_dry_spell_skips_when_action_already_critical():
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "B"]
    recent = ["UP", "RIGHT", "LEFT", "UP", "DOWN", "RIGHT", "UP", "LEFT"]
    rewards = [0.0] * 8
    # The LLM already picked B — we should NOT substitute (it's the
    # critical action!).
    out = _apply_anti_repetition(
        "B", valid_actions=valid,
        recent_actions=recent, recent_rewards=rewards,
        game="gymv_thunder_force_iii",
    )
    assert out == "B"
