"""Tests for per-game episode-count overrides.

Locks in the fix that bumps high-variance gymv games (TF3, Altered
Beast, etc.) from the default 8 to 16 episodes per step, dropping the
sampling-noise floor on bimodal-success games from ~22% (P(zero-mean)
at n=8) to ~4% at n=16.  See the post-mortem in chat for the full
bootstrap calculation; the relevant production fact is that under the
old code the override dict was only consulted when
``unified_role_rollouts=True``, so the gymv shooters silently kept
n=8 forever.
"""

from __future__ import annotations

from trainer.coevolution.config import (
    CoEvolutionConfig,
    EPISODES_PER_GAME_MULTIROLE,
    HIGH_VARIANCE_GYMV_EPISODES,
)
from trainer.coevolution.rollout_collector import build_lpt_schedule


# ---------------------------------------------------------------------------
# Default factory now bakes in the high-variance gymv games
# ---------------------------------------------------------------------------


def test_default_overrides_bake_in_high_variance_gymv_games():
    cfg = CoEvolutionConfig()
    for g, expected_n in HIGH_VARIANCE_GYMV_EPISODES.items():
        assert cfg.episodes_per_game_overrides.get(g) == expected_n, (
            f"{g} should default to {expected_n} eps/step "
            "(see HIGH_VARIANCE_GYMV_EPISODES)"
        )


def test_default_overrides_include_multirole_constants():
    cfg = CoEvolutionConfig()
    for g, n in EPISODES_PER_GAME_MULTIROLE.items():
        # MULTIROLE constants are merged FIRST so any HIGH_VARIANCE
        # entry with the same key wins (intentional precedence).
        if g in HIGH_VARIANCE_GYMV_EPISODES:
            continue
        assert cfg.episodes_per_game_overrides.get(g) == n


# ---------------------------------------------------------------------------
# get_episodes_for_game honors overrides regardless of mode
# ---------------------------------------------------------------------------


def test_get_episodes_for_game_uses_override_in_legacy_mode():
    cfg = CoEvolutionConfig(
        episodes_per_game=8,
        unified_role_rollouts=False,
        episodes_per_game_overrides={"gymv_thunder_force_iii": 16},
    )
    assert cfg.get_episodes_for_game("gymv_thunder_force_iii") == 16
    # Games not in the override dict fall back to the global.
    assert cfg.get_episodes_for_game("tetris") == 8


def test_get_episodes_for_game_uses_override_in_unified_mode():
    cfg = CoEvolutionConfig(
        episodes_per_game=4,
        unified_role_rollouts=True,
        episodes_per_game_overrides={"avalon": 5, "diplomacy": 7},
    )
    assert cfg.get_episodes_for_game("avalon") == 5
    assert cfg.get_episodes_for_game("diplomacy") == 7
    assert cfg.get_episodes_for_game("tetris") == 4   # unrelated game ⇒ global


# ---------------------------------------------------------------------------
# build_lpt_schedule honors overrides in BOTH modes (this is the bug fix)
# ---------------------------------------------------------------------------


def _count_episodes_per_game(specs):
    counts: dict = {}
    for s in specs:
        counts[s.game] = counts.get(s.game, 0) + 1
    return counts


def test_lpt_schedule_legacy_mode_honors_overrides():
    """The bug: under the old code this assertion failed because
    ``build_lpt_schedule`` only looked at overrides when
    ``unified_role_rollouts=True``.  Pin the post-fix behaviour."""
    games = ["tetris", "gymv_thunder_force_iii"]
    schedule = build_lpt_schedule(
        games,
        episodes_per_game=8,
        episodes_per_game_overrides={"gymv_thunder_force_iii": 16},
        unified_role_rollouts=False,    # ← legacy mode
    )
    counts = _count_episodes_per_game(schedule)
    assert counts == {"tetris": 8, "gymv_thunder_force_iii": 16}


def test_lpt_schedule_legacy_mode_with_full_high_variance_default():
    """End-to-end: hand build_lpt_schedule the same overrides dict
    CoEvolutionConfig produces by default, and confirm every gymv
    high-variance game gets 16 while non-gymv games stay at 8."""
    cfg = CoEvolutionConfig(episodes_per_game=8)
    games = ["tetris", "candy_crush", "gymv_thunder_force_iii", "gymv_altered_beast"]
    schedule = build_lpt_schedule(
        games,
        episodes_per_game=cfg.episodes_per_game,
        episodes_per_game_overrides=cfg.episodes_per_game_overrides,
        unified_role_rollouts=False,
    )
    counts = _count_episodes_per_game(schedule)
    assert counts["tetris"] == 8
    assert counts["candy_crush"] == 8
    assert counts["gymv_thunder_force_iii"] == 16
    assert counts["gymv_altered_beast"] == 16


def test_lpt_schedule_unified_mode_still_honors_overrides():
    """Unified-roles parity test (this path always worked)."""
    schedule = build_lpt_schedule(
        ["avalon"],
        episodes_per_game=4,
        episodes_per_game_overrides={"avalon": 5},
        unified_role_rollouts=True,
    )
    counts = _count_episodes_per_game(schedule)
    assert counts == {"avalon": 5}


def test_lpt_schedule_no_overrides_uses_global():
    schedule = build_lpt_schedule(
        ["tetris", "gymv_thunder_force_iii"],
        episodes_per_game=8,
        episodes_per_game_overrides=None,
        unified_role_rollouts=False,
    )
    counts = _count_episodes_per_game(schedule)
    assert counts == {"tetris": 8, "gymv_thunder_force_iii": 8}
