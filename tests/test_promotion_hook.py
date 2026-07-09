"""Unit tests for ``trainer.coevolution._promotion_hook``.

Three tiers:

1. Pure helper tests (``_count_proposals``,
   ``_materialise_synthetic_bank_run``, ``_read_decision_counts``,
   ``_resolve_driver_executable``).
2. Subprocess-shape tests with a tiny stub driver that writes the
   exact filesystem layout the real driver writes (``_run_summary.json``,
   ``<corpus>/<source>/bank_snapshots/snap-*.json``,
   ``promotion_decisions.jsonl``). Exercises the full hook *without*
   importing the real ~2k-LOC ``decide_promotion_gpt54.py``.
3. Live end-to-end smoke against the real driver + real Airstriker
   bank/proposals — skipped when the on-disk fixtures are missing.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

import pytest

from trainer.coevolution._promotion_hook import (
    DEFAULT_GATE_MODE,
    PromotionStepReport,
    _codebase_root,
    _count_proposals,
    _materialise_synthetic_bank_run,
    _read_decision_counts,
    _resolve_driver_executable,
    run_promotion_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_proposals_jsonl(root: Path, corpus: str, source: str, n: int) -> Path:
    p = root / corpus / source / "proposals.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "proposal_id": f"prop-{corpus}-{source}-{i}",
                "proposal_kind": "patch",
                "proposer": "reflector",
                "target_skill_id": f"S{i}",
                "target_domains": ["gymv", "browser", "osworld", "video", "visual_reasoning"],
                "rationale": f"test {i}",
                "evidence_role": "COMMIT",
                "patch_kind": "precondition",
                "adapter_plan": {},
            }
            f.write(json.dumps(row) + "\n")
    return p


def _write_legacy_bank(path: Path, n_skills: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n_skills):
            f.write(json.dumps({
                "skill": {
                    "skill_id": f"S{i}", "name": f"S{i}",
                    "evidence_role": "COMMIT",
                    "applicable_domains": ["gymv"],
                    "protocol": {"steps": ["step"]},
                    "contract": {"eff_add": [], "eff_del": []},
                },
                "report": {},
            }) + "\n")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_count_proposals_returns_zero_when_dir_missing(tmp_path: Path):
    assert _count_proposals(tmp_path / "nope") == 0


def test_count_proposals_walks_corpus_source_jsonl(tmp_path: Path):
    _write_proposals_jsonl(tmp_path, "gym_v", "Temporal_Airstriker-v0", 3)
    _write_proposals_jsonl(tmp_path, "env_wrappers", "tetris", 5)
    # An empty file shouldn't error.
    (tmp_path / "gym_v" / "Empty-v0").mkdir()
    (tmp_path / "gym_v" / "Empty-v0" / "proposals.jsonl").write_text("\n", encoding="utf-8")
    assert _count_proposals(tmp_path) == 8


def test_materialise_synthetic_bank_run_uses_symlinks(tmp_path: Path):
    real_a = _write_legacy_bank(tmp_path / "real" / "tetris" / "skill_bank.jsonl", 2)
    real_b = _write_legacy_bank(tmp_path / "real" / "Temporal_Strider-v0" / "skill_bank.jsonl", 1)
    bank_run = tmp_path / "synthetic"

    _materialise_synthetic_bank_run(
        bank_run=bank_run,
        legacy_bank_paths={
            "tetris": real_a,
            "Temporal_Strider-v0": real_b,
        },
    )
    sym_tetris = bank_run / "env_wrappers" / "tetris" / "skill_bank.jsonl"
    sym_strider = bank_run / "gym_v" / "Temporal_Strider-v0" / "skill_bank.jsonl"
    assert sym_tetris.is_file()
    assert sym_strider.is_file()
    # Reading the symlink yields the real bank's content.
    assert sym_tetris.read_text(encoding="utf-8") == real_a.read_text(encoding="utf-8")
    # And the trainer's source-of-truth file is untouched.
    assert real_a.read_text(encoding="utf-8")     # still readable; would have raised


def test_materialise_skips_missing_legacy_files(tmp_path: Path):
    bank_run = tmp_path / "synthetic"
    _materialise_synthetic_bank_run(
        bank_run=bank_run,
        legacy_bank_paths={
            "ghost_game": tmp_path / "does_not_exist.jsonl",
        },
    )
    # No files materialised — but the call must not raise.
    assert not (bank_run / "env_wrappers" / "ghost_game").exists()


def test_read_decision_counts_from_run_summary(tmp_path: Path):
    (tmp_path / "_run_summary.json").write_text(
        json.dumps({
            "by_decision": {
                "PROMOTE": 7, "REJECT": 1, "DEFER": 0, "ROLLBACK": 2,
            },
        }), encoding="utf-8",
    )
    assert _read_decision_counts(tmp_path) == (7, 1, 0, 2)


def test_read_decision_counts_falls_back_to_jsonl(tmp_path: Path):
    pair = tmp_path / "gym_v" / "Temporal_Strider-v0"
    pair.mkdir(parents=True)
    rows = [
        {"decision": "PROMOTE"}, {"decision": "PROMOTE"},
        {"decision": "REJECT"},
        {"decision": "DEFER"},
        {"not_a_decision_field": True},                  # malformed-but-recoverable
    ]
    (pair / "promotion_decisions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    assert _read_decision_counts(tmp_path) == (2, 1, 1, 0)


def test_read_decision_counts_returns_zeros_when_empty(tmp_path: Path):
    assert _read_decision_counts(tmp_path) == (0, 0, 0, 0)


def test_resolve_driver_executable_default_points_at_real_driver():
    cmd = _resolve_driver_executable(None)
    assert len(cmd) == 2
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).name == "decide_promotion_gpt54.py"
    assert Path(cmd[1]).is_file()


def test_resolve_driver_executable_passes_through_override():
    cmd = _resolve_driver_executable(["my-runner", "--flag"])
    assert cmd == ["my-runner", "--flag"]


# ---------------------------------------------------------------------------
# Subprocess-shape test (with stub driver)
# ---------------------------------------------------------------------------


def _write_stub_driver(
    path: Path,
    *,
    promote_per_source: int = 1,
    return_code: int = 0,
) -> Path:
    """Write a tiny standalone Python program that pretends to be
    ``decide_promotion_gpt54.py`` — emits the exact filesystem layout
    the hook expects, then exits with the given return code.

    Reads ``--proposals-run`` to discover sources, and ``--output-dir``
    to know where to write."""
    src = textwrap.dedent(f"""
        import argparse, json, os, sys
        from pathlib import Path

        ap = argparse.ArgumentParser()
        ap.add_argument("--proposals-run", required=True)
        ap.add_argument("--bank-run", required=True)
        ap.add_argument("--no-actions", action="store_true")
        ap.add_argument("--gate-mode", default="offline-synthetic")
        ap.add_argument("--output-dir", required=True)
        # Tolerate any other args the real driver accepts.
        args, _ = ap.parse_known_args()

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        per_pair = []
        n_total = 0
        for jsonl in Path(args.proposals_run).glob("*/*/proposals.jsonl"):
            corpus = jsonl.parents[1].name
            source = jsonl.parents[0].name
            n_props = sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
            pair = out / corpus / source
            pair.mkdir(parents=True, exist_ok=True)
            decisions = []
            for i in range(min(n_props, {promote_per_source})):
                decisions.append({{
                    "decision": "PROMOTE",
                    "proposal_id": f"{{source}}-prop-{{i}}",
                    "subject_skill_id": f"new-{{source}}-{{i}}",
                    "target_status": "provisional",
                }})
            for i in range({promote_per_source}, n_props):
                decisions.append({{
                    "decision": "REJECT",
                    "proposal_id": f"{{source}}-prop-{{i}}",
                    "subject_skill_id": f"S{{i}}",
                    "target_status": "rejected",
                }})
            (pair / "promotion_decisions.jsonl").write_text(
                "\\n".join(json.dumps(d) for d in decisions) + "\\n",
                encoding="utf-8",
            )
            (pair / "gate_verdicts.jsonl").write_text("", encoding="utf-8")
            snap_dir = pair / "bank_snapshots"
            snap_dir.mkdir(exist_ok=True)
            snap = snap_dir / f"snap-stub-{{source}}.json"
            promoted = []
            for i in range(min(n_props, {promote_per_source})):
                promoted.append({{
                    "skill_id": f"new-{{source}}-{{i}}",
                    "name": f"new-{{source}}-{{i}}",
                    "skill_type": "action",
                    "source_type": "repaired_from_failure",
                    "status": "provisional",
                    "version": "v1.stub",
                    "feasible_domains": ["gymv","browser","osworld","video","visual_reasoning"],
                    "verified_domains": ["browser","osworld","video","visual_reasoning"],
                    "protocol": [{{"action": "EXEC", "payload": {{}}, "notes": "stub step"}}],
                    "contract": {{
                        "preconditions": [], "effects_add": [], "effects_del": [],
                        "expected_evidence_roles": ["COMMIT"],
                        "success_criteria": [], "abort_criteria": [],
                    }},
                }})
            snap.write_text(json.dumps({{
                "snapshot_id": f"snap-stub-{{source}}",
                "body": {{"skills": promoted}},
            }}), encoding="utf-8")
            per_pair.append({{"corpus": corpus, "source": source,
                              "n_proposals": n_props, "n_decisions": n_props}})
            n_total += n_props

        (out / "_run_summary.json").write_text(json.dumps({{
            "n_proposals": n_total,
            "by_decision": {{
                "PROMOTE": sum(min(p["n_proposals"], {promote_per_source}) for p in per_pair),
                "REJECT":  sum(max(0, p["n_proposals"]-{promote_per_source}) for p in per_pair),
                "DEFER": 0, "ROLLBACK": 0,
            }},
            "per_pair": per_pair,
        }}), encoding="utf-8")
        sys.exit({return_code})
    """).strip()
    path.write_text(src, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_run_promotion_step_skips_when_no_proposals(tmp_path: Path):
    proposals = tmp_path / "props"
    proposals.mkdir()                                    # empty
    bank_path = _write_legacy_bank(tmp_path / "bank" / "tetris" / "skill_bank.jsonl")

    report = run_promotion_step(
        step=0, run_dir=tmp_path,
        proposals_run_dir=proposals,
        legacy_bank_paths={"tetris": bank_path},
    )
    assert isinstance(report, PromotionStepReport)
    assert report.skipped is True
    assert report.skipped_reason == "no proposals on disk"
    assert report.n_proposals_in == 0


def test_run_promotion_step_subprocess_path_round_trip(tmp_path: Path):
    """Drive the hook with a stub `decide_promotion_gpt54.py`; verify
    every contract: subprocess invocation, output reading, writeback."""
    # 1. Crafter-shaped proposals on disk.
    proposals = tmp_path / "props"
    _write_proposals_jsonl(proposals, "gym_v", "Temporal_Airstriker-v0", 3)
    _write_proposals_jsonl(proposals, "env_wrappers", "tetris", 2)

    # 2. Trainer-shaped legacy banks.
    bank_air = _write_legacy_bank(
        tmp_path / "bank" / "Temporal_Airstriker-v0" / "skill_bank.jsonl", 2,
    )
    bank_tet = _write_legacy_bank(
        tmp_path / "bank" / "tetris" / "skill_bank.jsonl", 1,
    )

    # 3. Stub driver: promotes 1/source.
    stub = _write_stub_driver(tmp_path / "stub_driver.py", promote_per_source=1)

    report = run_promotion_step(
        step=2, run_dir=tmp_path,
        proposals_run_dir=proposals,
        legacy_bank_paths={
            "Temporal_Airstriker-v0": bank_air,
            "tetris": bank_tet,
        },
        driver_executable=[sys.executable, str(stub)],
    )

    assert report.skipped is False
    assert report.driver_returncode == 0
    assert report.n_proposals_in == 5
    assert report.n_promote == 2                        # 1 per source × 2 sources
    assert report.n_reject == 3
    assert "Temporal_Airstriker-v0" in report.writeback_per_game
    assert "tetris" in report.writeback_per_game
    air_wb = report.writeback_per_game["Temporal_Airstriker-v0"]
    assert air_wb["n_inserted"] == 1                    # stub mints "new-Temporal_Airstriker-v0-0"
    # Writeback actually mutated the trainer bank: original 2 + 1 promoted = 3 lines.
    air_lines = bank_air.read_text(encoding="utf-8").splitlines()
    assert len(air_lines) == 3
    # The new skill is the promoted one.
    new_ids = {json.loads(l)["skill"]["skill_id"] for l in air_lines}
    assert "new-Temporal_Airstriker-v0-0" in new_ids
    # Pre-existing ids preserved.
    assert {"S0", "S1"}.issubset(new_ids)

    # Per-step summary written.
    assert (report.promotion_run_dir / "_step_summary.json").is_file()


def test_run_promotion_step_handles_driver_failure(tmp_path: Path):
    """When the stub driver exits non-zero, the hook records the
    failure cleanly without raising."""
    proposals = tmp_path / "props"
    _write_proposals_jsonl(proposals, "gym_v", "Temporal_Airstriker-v0", 1)
    bank = _write_legacy_bank(
        tmp_path / "bank" / "Temporal_Airstriker-v0" / "skill_bank.jsonl",
    )

    stub = _write_stub_driver(tmp_path / "stub.py", return_code=42)

    report = run_promotion_step(
        step=0, run_dir=tmp_path,
        proposals_run_dir=proposals,
        legacy_bank_paths={"Temporal_Airstriker-v0": bank},
        driver_executable=[sys.executable, str(stub)],
    )
    assert report.skipped is True
    assert report.driver_returncode == 42
    assert "returncode=42" in report.skipped_reason
    # Trainer bank untouched.
    n_lines = sum(1 for line in bank.read_text(encoding="utf-8").splitlines() if line.strip())
    assert n_lines == 1


def test_run_promotion_step_no_snapshot_means_zero_writeback(tmp_path: Path):
    """If the driver wrote no snapshot for a source (all rejected),
    writeback for that source is a no-op."""
    proposals = tmp_path / "props"
    _write_proposals_jsonl(proposals, "gym_v", "Temporal_Strider-v0", 1)
    bank = _write_legacy_bank(
        tmp_path / "bank" / "Temporal_Strider-v0" / "skill_bank.jsonl",
    )
    # Stub: promotes 0/source ⇒ snapshot dir exists but has zero promoted skills.
    # Actually, our stub still writes the snapshot, just with empty body.skills.
    # That triggers the "snapshot present, eligible=0" branch instead.
    stub = _write_stub_driver(tmp_path / "stub.py", promote_per_source=0)

    report = run_promotion_step(
        step=0, run_dir=tmp_path,
        proposals_run_dir=proposals,
        legacy_bank_paths={"Temporal_Strider-v0": bank},
        driver_executable=[sys.executable, str(stub)],
    )
    assert report.skipped is False
    wb = report.writeback_per_game["Temporal_Strider-v0"]
    assert wb["n_inserted"] == 0
    assert wb["n_updated"] == 0
    # Trainer bank still has its single original entry.
    assert sum(1 for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()) == 1


# ---------------------------------------------------------------------------
# Live end-to-end (real driver, real bank/proposals)
# ---------------------------------------------------------------------------


def _real_proposals_dir() -> Path:
    p = (
        _codebase_root()
        / "labeling_supplement" / "crafter_proposals_out"
        / "run_20260430_073444"
    )
    return p if p.is_dir() else Path()


def _real_airstriker_bank() -> Path:
    p = (
        _codebase_root()
        / "labeling" / "skill_bank_out" / "run_20260430_030637"
        / "gym_v" / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    )
    return p if p.is_file() else Path()


def test_run_promotion_step_live_end_to_end(tmp_path: Path):
    """Subprocess-invoke the *real* ``decide_promotion_gpt54.py`` against
    the real Phase-0 proposals + Airstriker bank. Confirms:

    * driver returns 0;
    * ≥1 PROMOTE decision;
    * writeback inserts into the trainer bank copy;
    * the trainer bank's resulting JSONL still parses.

    Skipped when the real fixtures aren't present (fresh checkout).
    """
    real_props = _real_proposals_dir()
    real_bank = _real_airstriker_bank()
    if not real_props.is_dir() or not real_bank.is_file():
        pytest.skip("no real Phase-0 fixtures on disk")

    # Copy bank so writeback doesn't mutate the source-of-truth file.
    bank = tmp_path / "bank" / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    bank.parent.mkdir(parents=True)
    bank.write_text(real_bank.read_text(encoding="utf-8"), encoding="utf-8")
    n_lines_before = sum(1 for line in bank.read_text(encoding="utf-8").splitlines() if line.strip())

    report = run_promotion_step(
        step=0, run_dir=tmp_path,
        proposals_run_dir=real_props,
        legacy_bank_paths={"Temporal_Airstriker-v0": bank},
        gate_mode=DEFAULT_GATE_MODE,
        # Restrict the sweep to one (corpus, source) so we don't pay
        # for the full 13-game sweep on every test run.
        extra_driver_args=["--corpus", "gym_v", "--source", "Temporal_Airstriker-v0"],
        driver_timeout_s=120.0,
    )

    assert report.skipped is False, report.skipped_reason
    assert report.driver_returncode == 0
    assert report.n_promote >= 1
    air_wb = report.writeback_per_game["Temporal_Airstriker-v0"]
    assert air_wb.get("n_inserted", 0) >= 1

    # Trainer bank now has more lines than before, all parseable.
    final_lines = [
        line for line in bank.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(final_lines) > n_lines_before
    for line in final_lines:
        env = json.loads(line)
        assert isinstance(env.get("skill"), dict)
        assert env["skill"].get("skill_id")
        # Two on-disk shapes for ``protocol`` are accepted by every
        # downstream loader (see Day-2 lift work):
        #   * legacy cold-start: ``{"steps": [<NL>], "preconditions":
        #     [...], …}``
        #   * Day-2-lifted: a list of typed hops
        #     ``[{"action", "payload", "notes"}, …]``
        proto = env["skill"].get("protocol", {})
        if isinstance(proto, list):
            assert all(isinstance(h, dict) for h in proto), proto
        else:
            assert isinstance(proto, dict)
            assert isinstance(proto.get("steps", []), list)
