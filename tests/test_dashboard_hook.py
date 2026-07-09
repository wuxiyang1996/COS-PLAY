"""Unit tests for ``trainer.coevolution._dashboard_hook``.

Covers:

* The cadence helper (``should_run_dashboard``) under all the
  on/off / every-K-steps combinations the orchestrator can throw
  at it.
* The pure summary helpers (``_mean_admit``,
  ``_per_target_cluster_means``, ``_compute_gate_verdicts``) on
  hand-crafted ``cells.json`` payloads — diagonals, within-cluster
  off-diagonal, cross-cluster bands, QA->game floor.
* End-to-end ``run_dashboard_step`` flow with the matrix driver
  mocked at the subprocess boundary, asserting the dashboard
  produces a structured ``DashboardReport`` whose
  ``to_metrics()`` output is suitable for the trainer's wandb
  sink (string verdicts encoded as scalars).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from trainer.coevolution._dashboard_hook import (
    DashboardReport,
    _compute_gate_verdicts,
    _mean_admit,
    _per_target_cluster_means,
    _verdict_to_scalar,
    run_dashboard_step,
    should_run_dashboard,
)


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step,every_k,enabled,expected",
    [
        (0,   100, False, False),
        (0,   100, True,  True),
        (50,  100, True,  False),
        (100, 100, True,  True),
        (200, 100, True,  True),
        (0,     0, True,  False),                     # cadence-disabled
        (0,    -5, True,  False),                     # negative ignored
        (5,     1, True,  True),                      # every step
    ],
)
def test_should_run_dashboard_cadence(step, every_k, enabled, expected):
    assert (
        should_run_dashboard(
            step=step, every_k_steps=every_k, enabled=enabled,
        )
        is expected
    )


# ---------------------------------------------------------------------------
# _verdict_to_scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict,scalar",
    [
        ("PASS",      1.0),
        ("FAIL",      0.0),
        ("soft-FAIL", 0.5),
        ("SOFT-FAIL", 0.5),
        ("N-A",      -1.0),
        ("",         -1.0),
        ("garbage",  -1.0),
    ],
)
def test_verdict_to_scalar(verdict, scalar):
    assert _verdict_to_scalar(verdict) == scalar


# ---------------------------------------------------------------------------
# _mean_admit / _per_target_cluster_means
# ---------------------------------------------------------------------------


def test_mean_admit_empty():
    assert _mean_admit([]) == 0.0


def test_mean_admit_basic():
    cells = [
        {"admit_rate": 0.4}, {"admit_rate": 0.6}, {"admit_rate": 0.0},
    ]
    assert _mean_admit(cells) == pytest.approx(1.0 / 3)


def test_per_cluster_means_groups_by_target_cluster():
    cells = [
        {"target_cluster": "image", "admit_rate": 0.2},
        {"target_cluster": "image", "admit_rate": 0.4},
        {"target_cluster": "video", "admit_rate": 0.5},
        {"admit_rate": 0.9},                            # missing cluster
    ]
    out = _per_target_cluster_means(cells)
    assert out["image"] == pytest.approx(0.3)
    assert out["video"] == 0.5
    assert out["unknown"] == 0.9


# ---------------------------------------------------------------------------
# _compute_gate_verdicts
# ---------------------------------------------------------------------------


def _cell(
    *,
    src_corpus: str, tgt_corpus: str,
    src_cluster: str, tgt_cluster: str,
    admit_rate: float,
):
    return {
        "source_corpus": src_corpus, "target_corpus": tgt_corpus,
        "source_cluster": src_cluster, "target_cluster": tgt_cluster,
        "admit_rate": admit_rate,
    }


def test_gate_verdicts_all_na_on_empty():
    g = _compute_gate_verdicts([])
    assert g == {"G1": "N-A", "G2": "N-A", "G3": "N-A", "G4": "N-A", "G5": "N-A"}


def test_gate_verdicts_g1_diagonal_pass():
    cells = [
        _cell(src_corpus="tetris", tgt_corpus="tetris",
              src_cluster="game", tgt_cluster="game",
              admit_rate=0.85),
        _cell(src_corpus="2048", tgt_corpus="2048",
              src_cluster="game", tgt_cluster="game",
              admit_rate=0.92),
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G1"] == "PASS"


def test_gate_verdicts_g1_diagonal_fail_when_below_floor():
    cells = [
        _cell(src_corpus="tetris", tgt_corpus="tetris",
              src_cluster="game", tgt_cluster="game",
              admit_rate=0.75),                          # <0.80
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G1"] == "FAIL"


def test_gate_verdicts_g2_within_cluster_off_diag():
    cells = [
        _cell(src_corpus="tetris", tgt_corpus="2048",
              src_cluster="game", tgt_cluster="game",
              admit_rate=0.40),
        _cell(src_corpus="2048", tgt_corpus="tetris",
              src_cluster="game", tgt_cluster="game",
              admit_rate=0.35),
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G2"] == "PASS"


def test_gate_verdicts_g3_band():
    cells = [
        _cell(src_corpus="tetris", tgt_corpus="vqa",
              src_cluster="game", tgt_cluster="image",
              admit_rate=0.20),
        _cell(src_corpus="vqa", tgt_corpus="tetris",
              src_cluster="image", tgt_cluster="game",
              admit_rate=0.30),
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G3"] == "PASS"


def test_gate_verdicts_g3_fail_outside_band():
    cells = [
        _cell(src_corpus="tetris", tgt_corpus="vqa",
              src_cluster="game", tgt_cluster="image",
              admit_rate=0.50),                          # > 0.35 ceiling
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G3"] == "FAIL"


def test_gate_verdicts_g4_band():
    cells = [
        _cell(src_corpus="tetris", tgt_corpus="vid",
              src_cluster="game", tgt_cluster="video",
              admit_rate=0.18),
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G4"] == "PASS"


def test_gate_verdicts_g5_pass_when_qa_to_game_near_zero():
    cells = [
        _cell(src_corpus="vqa", tgt_corpus="tetris",
              src_cluster="image", tgt_cluster="game",
              admit_rate=0.02),
        _cell(src_corpus="vid", tgt_corpus="tetris",
              src_cluster="video", tgt_cluster="game",
              admit_rate=0.04),
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G5"] == "PASS"


def test_gate_verdicts_g5_soft_fail_when_qa_to_game_above_floor():
    cells = [
        _cell(src_corpus="vqa", tgt_corpus="tetris",
              src_cluster="image", tgt_cluster="game",
              admit_rate=0.10),                          # >= 0.05 ⇒ soft-FAIL
    ]
    g = _compute_gate_verdicts(cells)
    assert g["G5"] == "soft-FAIL"


# ---------------------------------------------------------------------------
# DashboardReport
# ---------------------------------------------------------------------------


def test_dashboard_report_to_metrics_skipped_returns_empty():
    r = DashboardReport(
        step=0, run_dir=Path("/x"), dashboard_run_dir=Path("/x/y"),
        skipped=True, skipped_reason="no targets",
    )
    assert r.to_metrics() == {}


def test_dashboard_report_to_metrics_emits_scalars():
    r = DashboardReport(
        step=10, run_dir=Path("/x"), dashboard_run_dir=Path("/x/y"),
        n_cells_evaluated=8, n_cells_errored=1,
        mean_admit_rate=0.42,
        mean_diagonal_admit_rate=0.85,
        mean_off_diagonal_admit_rate=0.31,
        gate_verdicts={"G1": "PASS", "G3": "FAIL", "G5": "soft-FAIL"},
        per_cluster_admit_rate={"image": 0.20, "video": 0.30},
        driver_wall_time_s=12.5,
    )
    m = r.to_metrics()
    assert m["cross_domain/n_cells_evaluated"] == 8.0
    assert m["cross_domain/mean_admit_rate"] == pytest.approx(0.42)
    assert m["cross_domain/mean_diagonal_admit_rate"] == 0.85
    assert m["cross_domain/gates/G1"] == 1.0
    assert m["cross_domain/gates/G3"] == 0.0
    assert m["cross_domain/gates/G5"] == 0.5
    assert m["cross_domain/per_cluster/image"] == 0.20
    assert m["cross_domain/driver_wall_time_s"] == 12.5


def test_dashboard_report_to_metrics_custom_prefix():
    r = DashboardReport(
        step=0, run_dir=Path("/x"), dashboard_run_dir=Path("/x/y"),
        n_cells_evaluated=2, mean_admit_rate=0.1,
        gate_verdicts={"G1": "N-A"},
    )
    m = r.to_metrics(prefix="xd_dash")
    assert "xd_dash/mean_admit_rate" in m
    assert m["xd_dash/gates/G1"] == -1.0


# ---------------------------------------------------------------------------
# run_dashboard_step — end-to-end with mocked subprocess
# ---------------------------------------------------------------------------


def _make_mock_driver(cells_payload: Dict[str, Any]) -> List[str]:
    """Tiny ``python -c`` shim that pretends to be the Stage-6
    matrix driver: parses ``--out-dir`` from argv and writes a
    fixed ``cells.json`` payload."""
    payload_json = json.dumps(cells_payload)
    code = f"""
import json, sys
from pathlib import Path
argv = sys.argv[1:]
out_dir = None
for i, a in enumerate(argv):
    if a == "--out-dir" and i + 1 < len(argv):
        out_dir = Path(argv[i + 1])
        break
assert out_dir is not None, "out-dir not in argv"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "cells.json").write_text(json.dumps(json.loads({payload_json!r})))
sys.exit(0)
"""
    return [sys.executable, "-c", code]


def _write_bank(path: Path, skill_ids: List[str]) -> None:
    rows = [
        {"skill": {"skill_id": sid, "name": sid, "applicable_domains": ["gymv"]}}
        for sid in skill_ids
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )


def test_run_dashboard_no_banks_returns_skipped(tmp_path: Path):
    r = run_dashboard_step(
        step=0,
        run_dir=tmp_path,
        legacy_bank_paths={},
        dashboard_targets=("video",),
    )
    assert r.skipped is True
    assert "no legacy_bank_paths" in r.skipped_reason


def test_run_dashboard_no_targets_returns_skipped(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])
    r = run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=(),
    )
    assert r.skipped is True
    assert "no dashboard_targets" in r.skipped_reason


def test_run_dashboard_missing_bank_files_returns_skipped(tmp_path: Path):
    r = run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": tmp_path / "absent.jsonl"},
        dashboard_targets=("video",),
    )
    assert r.skipped is True
    assert "no per-game banks" in r.skipped_reason


def test_run_dashboard_end_to_end_writes_summary(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1", "s2"])
    payload = {
        "run_id": "run_xx", "n_cells": 2,
        "cells": [
            {
                "source_corpus": "g", "target_corpus": "g",
                "source_cluster": "game", "target_cluster": "game",
                "admit_rate": 0.85, "n_admit": 4, "n_total": 5,
                "verdicts": [],
            },
            {
                "source_corpus": "g", "target_corpus": "vqa",
                "source_cluster": "game", "target_cluster": "image",
                "admit_rate": 0.20, "n_admit": 1, "n_total": 5,
                "verdicts": [],
            },
        ],
    }
    driver = _make_mock_driver(payload)

    r = run_dashboard_step(
        step=42,
        run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=("video", "visual_reasoning"),
        driver_executable=driver,
    )

    assert r.skipped is False
    assert r.driver_returncode == 0
    assert r.n_cells_evaluated == 2
    assert r.n_cells_errored == 0
    assert r.mean_admit_rate == pytest.approx((0.85 + 0.20) / 2)
    assert r.mean_diagonal_admit_rate == 0.85
    assert r.mean_off_diagonal_admit_rate == 0.20
    assert r.gate_verdicts.get("G1") == "PASS"
    assert r.gate_verdicts.get("G3") == "PASS"
    # Step summary file produced.
    summary_path = (
        tmp_path / "cross_domain_dashboard_out" / "step_0042" / "_step_summary.json"
    )
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["step"] == 42
    assert summary["gate_verdicts"]["G1"] == "PASS"


def test_run_dashboard_end_to_end_metrics_dict_shape(tmp_path: Path):
    """The to_metrics() output must be flat ``{str: float}`` —
    nothing else (no nested dicts, no Nones) so wandb / TB don't choke."""
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])
    payload = {
        "cells": [
            {
                "source_corpus": "g", "target_corpus": "g",
                "source_cluster": "game", "target_cluster": "game",
                "admit_rate": 0.9, "verdicts": [],
            },
        ],
    }
    driver = _make_mock_driver(payload)
    r = run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=("video",),
        driver_executable=driver,
    )
    m = r.to_metrics()
    for k, v in m.items():
        assert isinstance(k, str), f"non-str key: {k!r}"
        assert isinstance(v, float), f"non-float value at {k!r}: {v!r}"


def test_run_dashboard_subprocess_failure_skipped(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])
    fail_driver = [sys.executable, "-c", "import sys; sys.exit(9)"]
    r = run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=("video",),
        driver_executable=fail_driver,
    )
    assert r.skipped is True
    assert r.driver_returncode == 9


def test_run_dashboard_cells_json_missing_returns_skipped(tmp_path: Path):
    """Driver returncode==0 but no cells.json produced ⇒ skipped."""
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])
    quiet_driver = [sys.executable, "-c", "import sys; sys.exit(0)"]
    r = run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=("video",),
        driver_executable=quiet_driver,
    )
    assert r.skipped is True
    assert "cells.json missing" in r.skipped_reason


def test_run_dashboard_subprocess_timeout_skipped(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])
    slow_driver = [sys.executable, "-c", "import time; time.sleep(60)"]
    r = run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=("video",),
        driver_executable=slow_driver,
        dashboard_driver_timeout_s=0.5,
    )
    assert r.skipped is True
    assert r.driver_returncode == 124


def test_run_dashboard_does_not_mutate_trainer_bank(tmp_path: Path):
    """Snapshot is a copy, not a symlink — the trainer's bank stays
    pristine even after the matrix runs."""
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1", "s2"])
    before = bank.read_text(encoding="utf-8")
    payload = {"cells": []}
    driver = _make_mock_driver(payload)
    run_dashboard_step(
        step=0, run_dir=tmp_path,
        legacy_bank_paths={"g": bank},
        dashboard_targets=("video",),
        driver_executable=driver,
    )
    assert bank.read_text(encoding="utf-8") == before
