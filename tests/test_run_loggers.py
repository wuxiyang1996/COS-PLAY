"""Unit tests for ``trainer.coevolution._run_loggers``.

Verifies the 5 reviewer-facing JSONL streams (block A1-A5) write in the
correct shape and gracefully no-op when ``set_run_dir`` was not called.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trainer.coevolution import _run_loggers as rl


@pytest.fixture(autouse=True)
def _isolated_run_dir(tmp_path):
    """Each test gets a fresh run dir + clean module state."""
    rl.set_run_dir(None)  # reset
    rl._component_agg.clear()  # reset aggregator
    rl._shaping_agg.clear()   # reset shaping aggregator
    rl.set_run_dir(tmp_path)
    yield tmp_path
    rl.set_run_dir(None)


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_log_harness_rejection_writes_jsonl(_isolated_run_dir):
    rl.log_harness_rejection(
        step=42,
        episode_id="ep-1",
        game="gymv_columns",
        domain="gymv",
        task="cl_match_3",
        skill_id="skill-abc",
        veto="domain_mismatch",
        veto_reason="state.domain='gymv' not in feasible_domains=['web']",
    )
    rows = _read_lines(_isolated_run_dir / "harness_log" / "rejections.jsonl")
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "harness_rejection"
    assert r["step"] == 42
    assert r["episode_id"] == "ep-1"
    assert r["veto"] == "domain_mismatch"
    assert r["skill_id"] == "skill-abc"
    assert "ts" in r


def test_log_harness_validate_writes_diagnostic(_isolated_run_dir):
    rl.log_harness_validate(
        step=5,
        episode_id="ep-2",
        game="gymv_thunder_force_iii",
        inner_step=12,
        skill_id="skill-xyz",
        ok=False,
        binding_ok=True,
        precondition_ok=False,
        evidence_ok=True,
        adapter_ok=True,
        veto_reasons=["precondition_failed"],
        missing_bindings=["target_enemy"],
        failed_preconditions=["enemy_in_range"],
    )
    rows = _read_lines(_isolated_run_dir / "harness_log" / "validate.jsonl")
    assert len(rows) == 1
    r = rows[0]
    assert r["ok"] is False
    assert r["binding_ok"] is True
    assert r["precondition_ok"] is False
    assert r["missing_bindings"] == ["target_enemy"]
    assert r["failed_preconditions"] == ["enemy_in_range"]


def test_log_lifecycle_transition(_isolated_run_dir):
    rl.log_lifecycle_transition(
        skill_id="skill-1",
        from_status="DRAFT",
        to_status="PROVISIONAL",
        reason="cold_start_seed",
    )
    rl.log_lifecycle_transition(
        skill_id="skill-1",
        from_status="PROVISIONAL",
        to_status="ACTIVE",
        reason="promotion:limited_pass",
    )
    rows = _read_lines(_isolated_run_dir / "lifecycle_log" / "transitions.jsonl")
    assert len(rows) == 2
    assert rows[0]["from_status"] == "DRAFT"
    assert rows[1]["to_status"] == "ACTIVE"
    assert all("ts" in r for r in rows)


def test_log_intention_switch(_isolated_run_dir):
    rl.log_intention_switch(
        step=3,
        episode_id="ep-x",
        game="g",
        inner_step=0,
        prev_intention="",
        new_intention="[SETUP] play",
        switched=False,
        sharp_shift=False,
    )
    rl.log_intention_switch(
        step=3,
        episode_id="ep-x",
        game="g",
        inner_step=1,
        prev_intention="[SETUP] play",
        new_intention="[ATTACK] target enemy",
        switched=True,
        sharp_shift=True,
        urgency="high",
    )
    rows = _read_lines(_isolated_run_dir / "intention_log" / "switches.jsonl")
    assert len(rows) == 2
    assert rows[1]["switched"] is True
    assert rows[1]["sharp_shift"] is True
    assert rows[1]["urgency"] == "high"


def test_component_timing_aggregator_flush(_isolated_run_dir):
    rl.record_component_call("crafter.llm", latency_ms=1234.5)
    rl.record_component_call("crafter.llm", latency_ms=2000.0)
    rl.record_component_call("promotion.judge", latency_ms=500.0)
    snap = rl.flush_component_timings(step=7)
    assert snap["crafter.llm"]["n_calls"] == 2
    assert snap["crafter.llm"]["total_ms"] == pytest.approx(3234.5)
    assert snap["promotion.judge"]["n_calls"] == 1

    # Aggregator is reset after flush.
    snap2 = rl.flush_component_timings(step=8)
    assert snap2 == {}

    rows = _read_lines(_isolated_run_dir / "runtime_log" / "component_timings.jsonl")
    # 2 components on step 7, none on step 8 (empty flush).
    by_component = {(r["step"], r["component"]): r for r in rows}
    assert (7, "crafter.llm") in by_component
    assert (7, "promotion.judge") in by_component
    assert by_component[(7, "crafter.llm")]["n_calls"] == 2


def test_disabled_run_dir_is_noop(tmp_path):
    rl.set_run_dir(None)
    rl.log_harness_rejection(
        step=0, episode_id="x", game="g", domain="d", task="",
        skill_id="s", veto="v", veto_reason="",
    )
    # No file written anywhere.
    assert not (tmp_path / "harness_log").exists()


def test_measure_component_context_manager(_isolated_run_dir):
    import time as _t
    with rl.measure_component("schema.profile"):
        _t.sleep(0.005)  # 5ms
    snap = rl.flush_component_timings(step=1)
    assert snap["schema.profile"]["n_calls"] == 1
    assert snap["schema.profile"]["total_ms"] >= 4.0  # sleep is best-effort

    rows = _read_lines(_isolated_run_dir / "runtime_log" / "component_timings.jsonl")
    assert any(r["component"] == "schema.profile" for r in rows)


# ── Shaping ratio aggregator (block 3 of the post-collapse fix) ────────────


def test_shaping_ratio_aggregator_emits_per_game_rows(_isolated_run_dir):
    # Simulate one TF3 batch where most decisions saw zero env reward
    # but the +1.0 survival constant + small intrinsic still landed in
    # GRPO. Mirror what episode_runner records per-decision.
    for _ in range(40):
        rl.record_shaping_signal(
            game="gymv_thunder_force_iii",
            raw_env=0.0, intrinsic=0.0, constant_offset=1.0,
        )
    rl.record_shaping_signal(
        game="gymv_thunder_force_iii",
        raw_env=100.0, intrinsic=0.3, constant_offset=1.0,
    )
    snap = rl.flush_shaping_ratio(step=9)
    assert "gymv_thunder_force_iii" in snap
    bucket = snap["gymv_thunder_force_iii"]
    assert bucket["n_decisions"] == 41
    # 40 decisions at raw=0, 1 decision at raw=100 ⇒ raw_abs_sum=100
    assert bucket["raw_env_abs_sum"] == pytest.approx(100.0)
    assert bucket["n_zero_raw"] == 40
    # 41 decisions × 1.0 survival = 41 const sum, plus 0.3 intrinsic
    assert bucket["constant_sum"] == pytest.approx(41.0)

    rows = _read_lines(_isolated_run_dir / "reward_shaping_log" / "ratio.jsonl")
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == "shaping_ratio"
    assert r["step"] == 9
    assert r["zero_raw_frac"] == pytest.approx(40.0 / 41.0)
    # ratio = (intrinsic_abs + const) / max(raw_abs, eps) =
    #         (0.3 + 41.0) / 100.0 ≈ 0.413
    assert r["shaping_ratio"] == pytest.approx((0.3 + 41.0) / 100.0)

    # Aggregator is drained.
    assert rl.flush_shaping_ratio(step=10) == {}


def test_shaping_ratio_warns_when_threshold_crossed(_isolated_run_dir, caplog):
    # All-zero raw env across 40 decisions ⇒ ratio is enormous.
    for _ in range(40):
        rl.record_shaping_signal(
            game="gymv_thunder_force_iii",
            raw_env=0.0, intrinsic=0.0, constant_offset=1.0,
        )
    with caplog.at_level("WARNING", logger="trainer.coevolution._run_loggers"):
        rl.flush_shaping_ratio(step=11)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("Shaping ratio high" in m for m in msgs), (
        "Expected a WARN row for the high shaping ratio, got: " + str(msgs)
    )


def test_shaping_ratio_skips_warn_on_small_samples(_isolated_run_dir, caplog):
    # Only 10 decisions ⇒ warn is suppressed even with infinite ratio.
    for _ in range(10):
        rl.record_shaping_signal(
            game="gymv_thunder_force_iii",
            raw_env=0.0, intrinsic=0.0, constant_offset=1.0,
        )
    with caplog.at_level("WARNING", logger="trainer.coevolution._run_loggers"):
        rl.flush_shaping_ratio(step=12)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("Shaping ratio high" in m for m in msgs)


def test_shaping_ratio_no_warn_when_raw_dominates(_isolated_run_dir, caplog):
    # Healthy regime: raw env reward dwarfs the constant.
    for _ in range(40):
        rl.record_shaping_signal(
            game="tetris", raw_env=50.0, intrinsic=0.1, constant_offset=1.0,
        )
    with caplog.at_level("WARNING", logger="trainer.coevolution._run_loggers"):
        snap = rl.flush_shaping_ratio(step=15)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("Shaping ratio high" in m for m in msgs), (
        f"Should not warn when raw env reward is dominant; ratio={snap['tetris']['shaping_ratio']}"
    )
