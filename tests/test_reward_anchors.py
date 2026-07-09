"""Tests for ``common.reward_anchors`` + ``RewardLogger`` normalization plumbing.

Covers training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.5:

  1. Static fallback table is well-formed (positive floats for the
     8 gymv games; ``None`` for the 4 paper Table-3 games).
  2. ``normalize_reward`` honors the §4.5 contract:
       - ``raw=None`` → ``None``
       - anchor ``None`` / 0 / negative → ``None``
       - ``r_norm = clip(raw/anchor, 0.0, 2.0)`` for valid anchors.
  3. ``auto_derive_anchors`` reads ``mean_reward`` from
     ``rollout_summary.json`` and skips zero/missing entries.
  4. ``resolve_anchors`` layers fallback → auto → overrides correctly,
     never letting a zero auto value clobber a positive fallback.
  5. ``RewardLogger`` round-trips ``reward_normalized`` through both
     ``RewardLogEntry`` (legacy `log_episode` path) and
     ``GRPOStepLogEntry`` (T2.4 `log_grpo_record` path).
  6. ``RewardLogger.set_reward_anchors`` upgrades anchors mid-run.
"""

from __future__ import annotations

import json

import pytest

from common.reward_anchors import (
    TEACHER_REWARD_ANCHORS,
    anchor_for,
    auto_derive_anchors,
    known_anchored_games,
    normalize_per_game,
    normalize_reward,
    resolve_anchors,
)


# ── Static fallback table ────────────────────────────────────────────


_GYMV_SLUGS = (
    "gymv_thunder_force_iii",
    "gymv_altered_beast",
    "gymv_columns",
    "gymv_dynamite_headdy",
    "gymv_space_harrier_ii",
    "gymv_streets_of_rage_2",
    "gymv_airstriker",
    "gymv_strider",
)

_PAPER_SLUGS = ("tetris", "candy_crush", "twenty_forty_eight", "super_mario")


def test_static_fallback_table_covers_all_phase1_phase2_slugs() -> None:
    """All 8 gymv games + 4 paper games must appear (anchor or None)."""
    for slug in _GYMV_SLUGS:
        assert slug in TEACHER_REWARD_ANCHORS, slug
        anchor = TEACHER_REWARD_ANCHORS[slug]
        assert anchor is not None and anchor > 0.0, (
            f"gymv slug {slug} must have a positive static anchor "
            f"(got {anchor})"
        )
    for slug in _PAPER_SLUGS:
        assert slug in TEACHER_REWARD_ANCHORS, slug
        # Paper games must be ``None`` (placeholder) — see §4.5 last row.
        assert TEACHER_REWARD_ANCHORS[slug] is None, (
            f"paper slug {slug} should be None pending baselines run"
        )


def test_known_anchored_games_returns_full_registry() -> None:
    games = set(known_anchored_games())
    expected = set(_GYMV_SLUGS) | set(_PAPER_SLUGS)
    assert games == expected


def test_anchor_for_returns_static_when_no_override() -> None:
    assert anchor_for("gymv_altered_beast") == 425.0
    assert anchor_for("tetris") is None
    assert anchor_for("not_a_game") is None


# ── normalize_reward contract ────────────────────────────────────────


def test_normalize_reward_basic() -> None:
    norm = normalize_reward(50.0, "gymv_altered_beast")
    # 50 / 425 ≈ 0.1176
    assert norm == pytest.approx(50.0 / 425.0)


def test_normalize_reward_at_anchor_returns_one() -> None:
    norm = normalize_reward(425.0, "gymv_altered_beast")
    assert norm == pytest.approx(1.0)


def test_normalize_reward_clipped_at_ceiling() -> None:
    # 100x the anchor must clip to 2.0
    norm = normalize_reward(42500.0, "gymv_altered_beast")
    assert norm == 2.0


def test_normalize_reward_clipped_at_floor() -> None:
    norm = normalize_reward(-50.0, "gymv_altered_beast")
    assert norm == 0.0


def test_normalize_reward_none_inputs() -> None:
    assert normalize_reward(None, "gymv_altered_beast") is None


def test_normalize_reward_no_anchor_returns_none() -> None:
    # Anchor missing → None
    assert normalize_reward(50.0, "tetris") is None
    # Game not in table → None
    assert normalize_reward(50.0, "not_a_game") is None
    # Anchor explicitly None → None
    assert normalize_reward(50.0, "x", anchors={"x": None}) is None
    # Anchor zero → None (no /0 poisoning)
    assert normalize_reward(50.0, "x", anchors={"x": 0.0}) is None
    # Anchor negative → None
    assert normalize_reward(50.0, "x", anchors={"x": -1.0}) is None


def test_normalize_per_game_vector() -> None:
    out = normalize_per_game(
        {"gymv_altered_beast": 425.0, "tetris": 100.0, "gymv_columns": 80.4},
    )
    assert out["gymv_altered_beast"] == pytest.approx(1.0)
    assert out["tetris"] is None  # no anchor
    assert out["gymv_columns"] == pytest.approx(80.4 / 160.8)


# ── auto_derive_anchors + resolve_anchors ────────────────────────────


def _write_summary(root, env_dir, *, mean_reward, max_reward=None):
    """Write a rollout_summary.json fixture.

    The default ``anchor_field`` is ``"max_reward"`` (post-decision-2);
    callers that want to test the ``mean_reward`` codepath should pass
    ``anchor_field="mean_reward"`` to the function under test.
    """
    sub = root / env_dir
    sub.mkdir(parents=True)
    payload = {"mean_reward": mean_reward}
    if max_reward is not None:
        payload["max_reward"] = max_reward
    sub.joinpath("rollout_summary.json").write_text(json.dumps(payload))


def test_auto_derive_skips_missing_root(tmp_path) -> None:
    # Non-existent root → empty dict (graceful degradation).
    out = auto_derive_anchors(cold_start_root=str(tmp_path / "nope"))
    assert out == {}


def test_auto_derive_reads_max_reward_by_default(tmp_path) -> None:
    """Default anchor_field is ``"max_reward"`` (matches static fallback)."""
    _write_summary(tmp_path, "Temporal_AlteredBeast-v0",
                   mean_reward=118.75, max_reward=300.0)
    _write_summary(tmp_path, "Temporal_Columns-v0",
                   mean_reward=153.625, max_reward=174.0)
    out = auto_derive_anchors(cold_start_root=str(tmp_path))
    assert out == {
        "gymv_altered_beast": 300.0,
        "gymv_columns": 174.0,
    }


def test_auto_derive_anchor_field_kwarg_override(tmp_path) -> None:
    """``anchor_field='mean_reward'`` re-reads the lenient field."""
    _write_summary(tmp_path, "Temporal_AlteredBeast-v0",
                   mean_reward=118.75, max_reward=300.0)
    out = auto_derive_anchors(
        cold_start_root=str(tmp_path), anchor_field="mean_reward",
    )
    assert out == {"gymv_altered_beast": 118.75}


def test_auto_derive_anchor_field_env_var_override(tmp_path, monkeypatch) -> None:
    """``COEVO_REWARD_ANCHOR_FIELD`` env var flips auto-derive globally."""
    _write_summary(tmp_path, "Temporal_Columns-v0",
                   mean_reward=153.625, max_reward=174.0)
    monkeypatch.setenv("COEVO_REWARD_ANCHOR_FIELD", "mean_reward")
    out = auto_derive_anchors(cold_start_root=str(tmp_path))
    assert out == {"gymv_columns": 153.625}


def test_auto_derive_unknown_field_falls_back_to_default(tmp_path) -> None:
    """Unknown field name → warning + falls back to ``"max_reward"``."""
    _write_summary(tmp_path, "Temporal_AlteredBeast-v0",
                   mean_reward=118.75, max_reward=300.0)
    out = auto_derive_anchors(
        cold_start_root=str(tmp_path), anchor_field="nonsense",
    )
    assert out == {"gymv_altered_beast": 300.0}


def test_auto_derive_skips_zero_reward(tmp_path) -> None:
    """Zero/negative ``max_reward`` → omitted (don't shadow static fallback)."""
    _write_summary(tmp_path, "Temporal_Strider-v0",
                   mean_reward=0.0, max_reward=0.0)
    _write_summary(tmp_path, "Temporal_AlteredBeast-v0",
                   mean_reward=50.0, max_reward=100.0)
    out = auto_derive_anchors(cold_start_root=str(tmp_path))
    assert "gymv_strider" not in out  # zero max → skipped
    assert out["gymv_altered_beast"] == 100.0


def test_auto_derive_skips_malformed(tmp_path) -> None:
    sub = tmp_path / "Temporal_Columns-v0"
    sub.mkdir()
    (sub / "rollout_summary.json").write_text("{not json")
    out = auto_derive_anchors(cold_start_root=str(tmp_path))
    assert out == {}


def test_resolve_layers_fallback_then_auto_then_overrides(tmp_path) -> None:
    _write_summary(tmp_path, "Temporal_AlteredBeast-v0",
                   mean_reward=100.0, max_reward=200.0)  # auto
    table = resolve_anchors(
        cold_start_root=str(tmp_path),
        overrides={"gymv_columns": 999.0, "tetris": 50.0},
    )
    # Auto wins over static fallback for AlteredBeast (uses max_reward).
    assert table["gymv_altered_beast"] == 200.0
    # Override wins over both auto and fallback for Columns.
    assert table["gymv_columns"] == 999.0
    # Override populates a previously-None paper slug.
    assert table["tetris"] == 50.0
    # Static fallback survives where auto absent (Strider).
    assert table["gymv_strider"] == 112.5


def test_resolve_anchor_field_kwarg_threaded_through(tmp_path) -> None:
    """``resolve_anchors(anchor_field='mean_reward')`` must propagate."""
    _write_summary(tmp_path, "Temporal_AlteredBeast-v0",
                   mean_reward=100.0, max_reward=200.0)
    table = resolve_anchors(
        cold_start_root=str(tmp_path), anchor_field="mean_reward",
    )
    assert table["gymv_altered_beast"] == 100.0


def test_resolve_zero_auto_does_not_clobber_static(tmp_path) -> None:
    """A zero/missing auto value must not zero-out a positive static."""
    # Strider has anchor=112.5 statically; cold-start records max=0.0 → skip.
    _write_summary(tmp_path, "Temporal_Strider-v0",
                   mean_reward=0.0, max_reward=0.0)
    table = resolve_anchors(cold_start_root=str(tmp_path))
    assert table["gymv_strider"] == 112.5


def test_resolve_explicit_none_override_marks_no_anchor() -> None:
    # Caller passes ``None`` for a game → downstream sees "no anchor".
    table = resolve_anchors(overrides={"gymv_altered_beast": None})
    assert table["gymv_altered_beast"] is None
    assert normalize_reward(100.0, "gymv_altered_beast", anchors=table) is None


# ── RewardLogger round-trip ──────────────────────────────────────────


def test_grpo_log_writes_reward_normalized(tmp_path) -> None:
    from harness.reward_logger import RewardLogger

    log_path = tmp_path / "reward_log.jsonl"
    rl = RewardLogger(
        log_path=str(log_path),
        reward_anchors={"gymv_columns": 100.0},
    )
    entry = rl.log_grpo_record(
        episode_id="ep1",
        adapter="action_taking",
        step=0,
        reward=50.0,
        game="gymv_columns",
    )
    assert entry.reward_normalized == pytest.approx(0.5)

    # File round-trip
    payload = json.loads(log_path.read_text().splitlines()[0])
    assert payload["reward"] == 50.0
    assert payload["reward_normalized"] == pytest.approx(0.5)
    assert payload["kind"] == "grpo_step"


def test_grpo_log_clips_at_ceiling(tmp_path) -> None:
    from harness.reward_logger import RewardLogger

    rl = RewardLogger(reward_anchors={"gymv_columns": 100.0})
    entry = rl.log_grpo_record(
        episode_id="ep1",
        adapter="action_taking",
        step=0,
        reward=10000.0,
        game="gymv_columns",
    )
    assert entry.reward_normalized == 2.0


def test_grpo_log_no_anchor_yields_none(tmp_path) -> None:
    from harness.reward_logger import RewardLogger

    rl = RewardLogger(reward_anchors={})
    entry = rl.log_grpo_record(
        episode_id="ep1",
        adapter="action_taking",
        step=0,
        reward=50.0,
        game="not_anchored",
    )
    assert entry.reward_normalized is None


def test_episode_log_writes_reward_normalized(tmp_path) -> None:
    from common.enums import SkillType
    from data_structure.extensions.skill_episode import (
        SkillEpisode,
        SkillEpisodeOutcome,
    )
    from harness.reward_logger import RewardLogger

    log_path = tmp_path / "reward_log.jsonl"
    rl = RewardLogger(
        log_path=str(log_path),
        reward_anchors={"gymv_altered_beast": 100.0},
    )

    ep = SkillEpisode(
        episode_id="ep1",
        skill_id="sk1",
        skill_version="v1",
        skill_type=SkillType.ACTION,
        domain="gymv_altered_beast",
        parent_run_id=None,
        outcome=SkillEpisodeOutcome(success=True, contract_satisfied=True, score=75.0),
    )
    entry = rl.log_episode(ep)
    assert entry.reward_normalized == pytest.approx(0.75)
    assert entry.score == 75.0  # raw unchanged
    payload = json.loads(log_path.read_text().splitlines()[0])
    assert payload["reward_normalized"] == pytest.approx(0.75)
    assert payload["score"] == 75.0
    assert payload["kind"] == "skill_episode"


def test_set_reward_anchors_upgrades_anchors_mid_run(tmp_path) -> None:
    """Late-bind via ``set_reward_anchors`` (orchestrator's auto-derive
    upgrade path)."""
    from harness.reward_logger import RewardLogger

    rl = RewardLogger()  # starts with no override → uses static fallback
    # gymv_altered_beast has static anchor=425.0
    e1 = rl.log_grpo_record(
        episode_id="ep1", adapter="action_taking", step=0,
        reward=425.0, game="gymv_altered_beast",
    )
    assert e1.reward_normalized == pytest.approx(1.0)

    # Upgrade the anchor (e.g. cold-start summary read) → re-normalize.
    rl.set_reward_anchors({"gymv_altered_beast": 100.0})
    e2 = rl.log_grpo_record(
        episode_id="ep2", adapter="action_taking", step=1,
        reward=425.0, game="gymv_altered_beast",
    )
    # 425/100 → clipped to 2.0
    assert e2.reward_normalized == 2.0


def test_legacy_reward_log_entry_to_json_includes_field() -> None:
    """Backward-compat: existing readers won't break; new field is just
    an extra key (``None`` by default)."""
    from harness.reward_logger import GRPOStepLogEntry, RewardLogEntry

    e = RewardLogEntry(
        episode_id="ep1", skill_id="sk1", skill_version="v1",
        domain="game", success=True, score=42.0,
    )
    assert e.reward_normalized is None
    payload = e.to_json()
    assert payload["reward_normalized"] is None
    assert payload["score"] == 42.0

    g = GRPOStepLogEntry(episode_id="ep1", reward=42.0)
    assert g.reward_normalized is None
    assert g.to_json()["reward_normalized"] is None
