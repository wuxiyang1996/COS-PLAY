"""Unit tests for ``trainer.coevolution._transfer_hook``.

Covers the two stages of the cross-domain transfer gate:

* The pure decision logic (band scoring + per-skill verdict) — no
  subprocess, no filesystem, no I/O.
* The end-to-end ``run_transfer_gate_step`` flow with the matrix
  driver mocked at the subprocess boundary. The mock writes a
  hand-crafted ``per_skill.jsonl`` so we can assert the gate's
  rollback behaviour on the legacy ``skill_bank.jsonl`` files.

We don't exercise the real ``_phase4_transfer_matrix.py`` driver — it
needs cross-domain bank data the test fixture doesn't have. The
contract under test is *the gate*, not the underlying measurement;
the matrix's own correctness is covered by
``tests/test_phase5_transfer_matrix*.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from trainer.coevolution._transfer_hook import (
    DEFAULT_TRANSFER_ADMIT_BAND,
    TransferGateReport,
    TransferSkillVerdict,
    _collect_just_promoted_skill_ids,
    _decide_for_skill,
    _drop_skill_ids_from_bank,
    _materialise_filtered_bank_run,
    _parse_per_skill_results,
    run_transfer_gate_step,
)


# ---------------------------------------------------------------------------
# _collect_just_promoted_skill_ids
# ---------------------------------------------------------------------------


def test_collect_promoted_handles_empty_input():
    assert _collect_just_promoted_skill_ids({}) == {}
    assert _collect_just_promoted_skill_ids(None) == {}


def test_collect_promoted_lifts_inserted_and_updated():
    wb = {
        "tetris": {
            "n_inserted": 2, "n_updated": 1,
            "inserted_skill_ids": ["s1", "s2"],
            "updated_skill_ids": ["s_old"],
        },
        "twenty_forty_eight": {
            "n_inserted": 0, "n_updated": 0,
            "inserted_skill_ids": [], "updated_skill_ids": [],
        },
    }
    out = _collect_just_promoted_skill_ids(wb)
    assert out == {"tetris": ["s1", "s2", "s_old"]}


def test_collect_promoted_tolerates_missing_keys():
    wb = {"g": {"n_inserted": 0}}                     # no skill_id lists
    assert _collect_just_promoted_skill_ids(wb) == {}


# ---------------------------------------------------------------------------
# _decide_for_skill — band scoring (pure)
# ---------------------------------------------------------------------------


def _v(decision_args, **kwargs):
    """Tiny verdict-builder shim for assertion clarity."""
    return TransferSkillVerdict(**decision_args, **kwargs)


def test_decide_keep_when_one_target_clears_band():
    v = _decide_for_skill(
        skill_id="s1", game="g",
        target_admits={"video": 0.42, "browser": 0.05},
        band_lo=0.15, min_in_band=1,
        requested_targets=("video", "browser"),
    )
    assert v.decision == "KEEP"
    assert v.n_targets_in_band == 1
    assert v.failure_class is None


def test_decide_demote_when_every_requested_target_below_band():
    v = _decide_for_skill(
        skill_id="s1", game="g",
        target_admits={"video": 0.05, "browser": 0.10},
        band_lo=0.15, min_in_band=1,
        requested_targets=("video", "browser"),
    )
    assert v.decision == "DEMOTE"
    assert v.n_targets_in_band == 0
    assert v.failure_class == "CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION"


def test_decide_insufficient_data_when_no_target_ran():
    v = _decide_for_skill(
        skill_id="s1", game="g",
        target_admits={},
        band_lo=0.15, min_in_band=1,
        requested_targets=("video",),
    )
    assert v.decision == "INSUFFICIENT_DATA"
    assert v.failure_class is None


def test_decide_ignores_extra_targets_outside_request():
    """Cells the matrix surfaced for non-requested targets must not
    sway the band decision (informational only). The verdict's
    ``admit_rate_per_target`` still includes them for the dashboard,
    but the DEMOTE/KEEP decision is computed against the requested
    set only."""
    v = _decide_for_skill(
        skill_id="s1", game="g",
        target_admits={"video": 0.05, "osworld": 0.99},   # osworld in admit, not requested
        band_lo=0.15, min_in_band=1,
        requested_targets=("video",),
    )
    assert v.decision == "DEMOTE"
    # The osworld pass rate is preserved in the verdict for the
    # downstream dashboard, but it doesn't count toward the band.
    assert "osworld" in v.admit_rate_per_target
    # n_targets_in_band on the DEMOTE branch reflects requested-only.
    assert v.n_targets_in_band == 0


def test_decide_min_in_band_gate():
    """min_in_band=2 is stricter — one target alone is no longer enough."""
    v = _decide_for_skill(
        skill_id="s1", game="g",
        target_admits={"video": 0.42, "browser": 0.05},
        band_lo=0.15, min_in_band=2,
        requested_targets=("video", "browser"),
    )
    assert v.decision == "DEMOTE"


# ---------------------------------------------------------------------------
# _parse_per_skill_results
# ---------------------------------------------------------------------------


def test_parse_per_skill_collapses_repeats_to_mean(tmp_path: Path):
    p = tmp_path / "per_skill.jsonl"
    rows = [
        {"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.4, "success": True},
        {"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.6, "success": True},
        {"skill_id": "s1", "target_corpus": "browser", "pass_rate": 0.0, "success": False},
        {"skill_id": "s2", "target_corpus": "video", "pass_rate": 0.8, "success": True},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = _parse_per_skill_results(tmp_path)
    assert pytest.approx(out["s1"]["video"]) == 0.5
    assert out["s1"]["browser"] == 0.0
    assert out["s2"]["video"] == 0.8


def test_parse_per_skill_falls_back_to_success_when_no_pass_rate(tmp_path: Path):
    p = tmp_path / "per_skill.jsonl"
    rows = [{"skill_id": "s1", "target_corpus": "video", "success": True}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = _parse_per_skill_results(tmp_path)
    assert out["s1"]["video"] == 1.0


def test_parse_per_skill_missing_file_is_empty(tmp_path: Path):
    out = _parse_per_skill_results(tmp_path)
    assert out == {}


def test_parse_per_skill_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "per_skill.jsonl"
    p.write_text(
        "not-json\n"
        + json.dumps({"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.5}) + "\n"
        + "{\"skill_id\": \"s2\"}\n"  # missing target_corpus
        + "\n",
        encoding="utf-8",
    )
    out = _parse_per_skill_results(tmp_path)
    assert out == {"s1": {"video": 0.5}}


# ---------------------------------------------------------------------------
# _materialise_filtered_bank_run
# ---------------------------------------------------------------------------


def _write_bank(path: Path, skill_ids: List[str]) -> None:
    rows = [
        {"skill": {"skill_id": sid, "name": sid, "applicable_domains": ["gymv"]}}
        for sid in skill_ids
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )


def test_materialise_filtered_keeps_only_requested(tmp_path: Path):
    bank = tmp_path / "tetris.jsonl"
    _write_bank(bank, ["s1", "s2", "s3"])
    out = tmp_path / "synth"
    _materialise_filtered_bank_run(
        bank_root=out,
        legacy_bank_paths={"tetris": bank},
        eligible_skill_ids_per_game={"tetris": ["s1", "s3"]},
    )
    written = (out / "env_wrappers" / "tetris" / "skill_bank.jsonl").read_text(
        encoding="utf-8",
    ).strip().splitlines()
    written_sids = {json.loads(r)["skill"]["skill_id"] for r in written}
    assert written_sids == {"s1", "s3"}


def test_materialise_filtered_skips_missing_files(tmp_path: Path):
    out = tmp_path / "synth"
    _materialise_filtered_bank_run(
        bank_root=out,
        legacy_bank_paths={"tetris": tmp_path / "missing.jsonl"},
        eligible_skill_ids_per_game={"tetris": ["s1"]},
    )
    # No file produced; layout was tolerant to the missing source.
    assert not (out / "env_wrappers" / "tetris").exists()


# ---------------------------------------------------------------------------
# _drop_skill_ids_from_bank
# ---------------------------------------------------------------------------


def test_drop_skill_ids_removes_only_targeted(tmp_path: Path):
    bank = tmp_path / "tetris.jsonl"
    _write_bank(bank, ["s1", "s2", "s3"])
    n_dropped = _drop_skill_ids_from_bank(bank, {"s2"})
    assert n_dropped == 1
    remaining = [
        json.loads(line)["skill"]["skill_id"]
        for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert remaining == ["s1", "s3"]


def test_drop_skill_ids_noop_on_empty_set(tmp_path: Path):
    bank = tmp_path / "tetris.jsonl"
    _write_bank(bank, ["s1", "s2"])
    before = bank.read_text(encoding="utf-8")
    n = _drop_skill_ids_from_bank(bank, set())
    assert n == 0
    assert bank.read_text(encoding="utf-8") == before


def test_drop_skill_ids_noop_when_no_match(tmp_path: Path):
    bank = tmp_path / "tetris.jsonl"
    _write_bank(bank, ["s1", "s2"])
    before = bank.read_text(encoding="utf-8")
    n = _drop_skill_ids_from_bank(bank, {"nonexistent"})
    assert n == 0
    assert bank.read_text(encoding="utf-8") == before


def test_drop_skill_ids_preserves_malformed_lines(tmp_path: Path):
    bank = tmp_path / "tetris.jsonl"
    bank.write_text(
        "not-json\n"
        + json.dumps({"skill": {"skill_id": "s1"}}) + "\n"
        + json.dumps({"skill": {"skill_id": "s2"}}) + "\n",
        encoding="utf-8",
    )
    n = _drop_skill_ids_from_bank(bank, {"s1"})
    assert n == 1
    out = bank.read_text(encoding="utf-8").splitlines()
    assert out[0] == "not-json"
    assert json.loads(out[1])["skill"]["skill_id"] == "s2"


# ---------------------------------------------------------------------------
# run_transfer_gate_step — end-to-end with mocked subprocess
# ---------------------------------------------------------------------------


def _make_mock_driver(per_skill_rows: List[Dict[str, Any]]) -> List[str]:
    """Return a [python, -c, '...'] executable that pretends to be the
    matrix driver: parses ``--out-dir`` from argv and writes a stub
    ``per_skill.jsonl``. Used as ``driver_executable`` in the tests
    so we don't need the real cross-domain bank fixture."""
    rows_json = json.dumps(per_skill_rows)
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
rows = json.loads({rows_json!r})
with (out_dir / "per_skill.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\\n")
sys.exit(0)
"""
    return [sys.executable, "-c", code]


def test_run_transfer_gate_no_skills_in_returns_skipped(tmp_path: Path):
    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={},
        legacy_bank_paths={"g": tmp_path / "g.jsonl"},
        transfer_targets=("video",),
    )
    assert report.skipped is True
    assert "no skills promoted" in report.skipped_reason
    assert report.n_keep == 0 and report.n_demote == 0


def test_run_transfer_gate_no_targets_returns_skipped(tmp_path: Path):
    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": tmp_path / "g.jsonl"},
        transfer_targets=(),
    )
    assert report.skipped is True
    assert "no transfer_targets" in report.skipped_reason
    assert report.n_keep == 1                       # synthesised KEEP verdict
    assert len(report.verdicts) == 1


def test_run_transfer_gate_keeps_skill_when_one_target_clears_band(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1", "s2"])

    driver = _make_mock_driver([
        {"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.42, "success": True},
        {"skill_id": "s1", "target_corpus": "browser", "pass_rate": 0.05, "success": False},
    ])

    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": bank},
        transfer_targets=("video", "browser"),
        transfer_admit_band=(0.15, 0.60),
        transfer_min_targets_in_band=1,
        driver_executable=driver,
    )
    assert report.skipped is False
    assert report.n_skills_in == 1
    assert report.n_keep == 1
    assert report.n_demote == 0
    # Bank file unchanged.
    remaining = [
        json.loads(line)["skill"]["skill_id"]
        for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert sorted(remaining) == ["s1", "s2"]


def test_run_transfer_gate_demotes_skill_when_band_floor_violated(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1", "s2", "keeper"])

    driver = _make_mock_driver([
        {"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.05, "success": False},
        {"skill_id": "s1", "target_corpus": "browser", "pass_rate": 0.00, "success": False},
        {"skill_id": "s2", "target_corpus": "video", "pass_rate": 0.80, "success": True},
    ])

    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1", "s2"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": bank},
        transfer_targets=("video", "browser"),
        transfer_admit_band=(0.15, 0.60),
        transfer_min_targets_in_band=1,
        driver_executable=driver,
    )
    assert report.skipped is False
    assert report.n_demote == 1
    assert report.n_keep == 1
    assert report.demotions_per_game == {"g": ["s1"]}
    # The DEMOTE verdict carries the canonical failure class.
    demote_v = next(v for v in report.verdicts if v.skill_id == "s1")
    assert demote_v.failure_class == "CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION"
    # Bank file: s1 is gone, s2 + keeper remain.
    remaining = [
        json.loads(line)["skill"]["skill_id"]
        for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert sorted(remaining) == ["keeper", "s2"]


def test_run_transfer_gate_dry_run_does_not_mutate_bank(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1", "keeper"])

    driver = _make_mock_driver([
        {"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.05, "success": False},
    ])

    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": bank},
        transfer_targets=("video",),
        driver_executable=driver,
        apply_demotions=False,
    )
    assert report.n_demote == 1                   # verdict still says DEMOTE
    assert report.demotions_per_game == {"g": ["s1"]}
    # But the bank wasn't mutated.
    remaining = [
        json.loads(line)["skill"]["skill_id"]
        for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert sorted(remaining) == ["keeper", "s1"]


def test_run_transfer_gate_subprocess_failure_keeps_all_skills(tmp_path: Path):
    """A non-zero subprocess returncode must not demote anything."""
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1", "s2"])

    fail_driver = [sys.executable, "-c", "import sys; sys.exit(7)"]

    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": bank},
        transfer_targets=("video",),
        driver_executable=fail_driver,
    )
    assert report.skipped is True
    assert report.driver_returncode == 7
    assert report.n_demote == 0
    # Bank untouched.
    remaining = [
        json.loads(line)["skill"]["skill_id"]
        for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert sorted(remaining) == ["s1", "s2"]


def test_run_transfer_gate_subprocess_timeout_keeps_all_skills(tmp_path: Path):
    """A subprocess timeout is handled like other failures: no demotion."""
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])

    slow_driver = [sys.executable, "-c", "import time; time.sleep(60)"]

    report = run_transfer_gate_step(
        step=0,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": bank},
        transfer_targets=("video",),
        driver_executable=slow_driver,
        transfer_driver_timeout_s=0.5,
    )
    assert report.skipped is True
    assert report.driver_returncode == 124
    assert report.n_demote == 0


def test_run_transfer_gate_writes_step_summary(tmp_path: Path):
    bank = tmp_path / "g.jsonl"
    _write_bank(bank, ["s1"])

    driver = _make_mock_driver([
        {"skill_id": "s1", "target_corpus": "video", "pass_rate": 0.5, "success": True},
    ])

    report = run_transfer_gate_step(
        step=42,
        run_dir=tmp_path,
        promotion_writeback_per_game={
            "g": {"inserted_skill_ids": ["s1"], "updated_skill_ids": []},
        },
        legacy_bank_paths={"g": bank},
        transfer_targets=("video",),
        driver_executable=driver,
    )
    summary_path = tmp_path / "transfer_gate_out" / "step_0042" / "_step_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["step"] == 42
    assert summary["n_skills_in"] == 1
    assert summary["params"]["transfer_targets"] == ["video"]
    assert summary["params"]["transfer_admit_band"] == [0.15, 0.6]


def test_default_admit_band_constant():
    assert DEFAULT_TRANSFER_ADMIT_BAND == (0.15, 0.60)


def test_transfer_gate_report_to_dict_round_trip():
    r = TransferGateReport(
        step=1,
        run_dir=Path("/tmp/run"),
        transfer_run_dir=Path("/tmp/run/transfer_gate_out/step_0001"),
        n_skills_in=3, n_keep=1, n_demote=1, n_insufficient_data=1,
        verdicts=[
            TransferSkillVerdict(
                skill_id="a", game="g", decision="KEEP",
                admit_rate_per_target={"video": 0.4}, n_targets_in_band=1,
            ),
        ],
        demotions_per_game={"g": ["b"]},
    )
    d = r.to_dict()
    assert d["n_keep"] == 1
    assert d["verdicts"][0]["decision"] == "KEEP"
    assert d["demotions_per_game"] == {"g": ["b"]}
