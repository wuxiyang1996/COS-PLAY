"""Unit tests for ``skill_bank.legacy_writeback``.

Covers:

* per-helper unit tests (``_typed_protocol_to_nl_steps``, ``_semver_to_int``,
  ``_project_to_legacy_envelope``);
* end-to-end ``writeback_promotion`` on synthetic snapshots — insert /
  upsert / status filter / dry-run / atomic write;
* a smoke check against the *real* snapshot
  ``decide_promotion_gpt54.py --gate-mode offline-synthetic`` produces
  in Phase 0, when present (skipped otherwise so the test is hermetic
  on a fresh checkout).

These tests intentionally do **not** import ``skill_bank.SkillRepository``
or ``skill_agents.*`` — the projector is on-disk-only by design (see
module docstring on D8 Option A).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from skill_bank.legacy_writeback import (
    DEFAULT_ELIGIBLE_STATUSES,
    WritebackReport,
    _empty_report,
    _project_to_legacy_envelope,
    _semver_to_int,
    _typed_protocol_to_nl_steps,
    find_latest_snapshot,
    writeback_promotion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_skill(
    *,
    skill_id: str = "skill-test-01",
    status: str = "provisional",
    name: str = "Patch/Attack",
    notes: str = "Press the fire button",
    feasible: list = None,
    expected_role: str = "COMMIT",
    eff_add: list = None,
    eff_del: list = None,
    version: str = "v1.offline.recond_0",
) -> Dict[str, Any]:
    return {
        "skill_id": skill_id,
        "name": name,
        "status": status,
        "skill_type": "action",
        "source_type": "repaired_from_failure",
        "version": version,
        "feasible_domains": feasible or ["gymv", "browser", "osworld"],
        "verified_domains": ["browser", "osworld"],
        "protocol": [
            {"action": "EXEC", "payload": {}, "notes": notes},
            {"action": "EXEC", "payload": {}, "notes": "Wait for animation"},
        ],
        "contract": {
            "preconditions": ["enemy_visible"],
            "effects_add": list(eff_add or ["enemy_volley_complete"]),
            "effects_del": list(eff_del or []),
            "expected_evidence_roles": [expected_role],
            "success_criteria": ["volley_executed"],
            "abort_criteria": ["player_destroyed"],
            "name": name,
            "description": "fixture skill",
        },
    }


def _make_snapshot(skills: list, snapshot_id: str = "snap-test-01") -> Dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "created_at": 1.0,
        "notes": "test fixture",
        "body_hash": "deadbeef",
        "body": {"skills": skills},
    }


# ---------------------------------------------------------------------------
# Helper: _typed_protocol_to_nl_steps
# ---------------------------------------------------------------------------


def test_typed_protocol_prefers_notes_over_action():
    out = _typed_protocol_to_nl_steps([
        {"action": "EXEC", "payload": {}, "notes": "Press fire"},
    ])
    assert out == ["Press fire"]


def test_typed_protocol_falls_back_to_action_then_payload():
    out = _typed_protocol_to_nl_steps([
        {"action": "EXEC", "payload": {"k": "v"}, "notes": ""},
        {"action": "MOVE", "payload": {}},
    ])
    assert out == ['EXEC {"k": "v"}', "MOVE"]


def test_typed_protocol_handles_non_mapping_hops():
    out = _typed_protocol_to_nl_steps(["already a string", 42])
    assert out == ["already a string", "42"]


def test_typed_protocol_serialises_unknown_shape_without_loss():
    out = _typed_protocol_to_nl_steps([{"weird": "hop"}])
    parsed = json.loads(out[0])
    assert parsed == {"weird": "hop"}


# ---------------------------------------------------------------------------
# Helper: _semver_to_int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inp,expected", [
    (3, 3),
    ("3", 3),
    ("v1.offline.recond_0", 1),
    ("v17", 17),
    ("0.1.2", 1),                       # falls back to 1, not 0
    ("v0", 1),                          # ditto — 0 means "uninitialised" in legacy bank
    (None, 1),
    ("", 1),
    ("garbage", 1),
])
def test_semver_to_int(inp, expected):
    assert _semver_to_int(inp) == expected


# ---------------------------------------------------------------------------
# Helper: _project_to_legacy_envelope
# ---------------------------------------------------------------------------


def test_project_returns_none_for_missing_skill_id():
    skill = _make_skill()
    skill.pop("skill_id")
    assert _project_to_legacy_envelope(skill) is None


def test_project_emits_legacy_envelope_shape():
    env = _project_to_legacy_envelope(_make_skill())
    assert env is not None
    assert set(env) == {"skill", "report"}
    s = env["skill"]
    assert s["skill_id"] == "skill-test-01"
    assert s["name"] == "Patch/Attack"
    # protocol.steps must be NL strings, not typed hops
    assert s["protocol"]["steps"] == ["Press the fire button", "Wait for animation"]
    # contract: effects_add → eff_add
    assert s["contract"]["eff_add"] == ["enemy_volley_complete"]
    assert s["contract"]["eff_del"] == []
    # evidence_role lifted from expected_evidence_roles[0]
    assert s["evidence_role"] == "COMMIT"
    # feasible_domains → applicable_domains
    assert s["applicable_domains"] == ["gymv", "browser", "osworld"]
    # version coerced to leading int
    assert s["version"] == 1
    # writeback markers
    assert s["_writeback_status"] == "provisional"
    assert s["_writeback_verified_domains"] == ["browser", "osworld"]
    # Report block: must be VerificationReport-compatible (the real
    # SkillBankMVP.load() round-trips it via VerificationReport.from_dict
    # → cls(**d), which raises on unexpected keys).  The canonical empty
    # shape is documented in skill_agents/stage3_mvp/schemas.py.
    rep = env["report"]
    assert rep["skill_id"] == "skill-test-01"
    assert rep["n_instances"] == 0
    assert rep["overall_pass_rate"] == 0.0
    assert rep["eff_add_success_rate"] == {}
    assert rep["eff_del_success_rate"] == {}
    assert rep["eff_event_rate"] == {}
    assert rep["worst_segments"] == []
    assert rep["failure_signatures"] == {}
    # No spurious keys leak through (would break SkillBankMVP.load()).
    assert set(rep) == {
        "skill_id", "n_instances",
        "eff_add_success_rate", "eff_del_success_rate", "eff_event_rate",
        "overall_pass_rate", "worst_segments", "failure_signatures",
    }


def test_project_preserves_canonical_prior_report_fields():
    """Carrying over a prior report block: the canonical
    ``VerificationReport`` fields are preserved verbatim, *non*-canonical
    fields are dropped (keeping the output loadable by the real
    ``SkillBankMVP.load()``)."""
    prior = {
        "skill": {"skill_id": "skill-test-01"},
        "report": {
            "skill_id": "skill-test-01",
            "n_instances": 17,
            "overall_pass_rate": 0.53,
            "eff_add_success_rate": {"world.e1[type=text]": 0.7},
            "worst_segments": ["seg-42"],
            # These three are NOT canonical — must be dropped.
            "selection_count": 99,
            "pass_rate": 0.42,
            "source": "promotion-writeback",
        },
    }
    env = _project_to_legacy_envelope(_make_skill(), prior_envelope=prior)
    assert env is not None
    rep = env["report"]
    # Canonical fields preserved.
    assert rep["n_instances"] == 17
    assert rep["overall_pass_rate"] == 0.53
    assert rep["eff_add_success_rate"] == {"world.e1[type=text]": 0.7}
    assert rep["worst_segments"] == ["seg-42"]
    # Non-canonical fields dropped (would crash VerificationReport.from_dict).
    assert "selection_count" not in rep
    assert "pass_rate" not in rep
    assert "source" not in rep


def test_project_carries_over_predicate_lists_from_prior():
    prior = {
        "skill": {
            "skill_id": "skill-test-01",
            "protocol": {
                "predicate_success": ["world.e1"],
                "predicate_abort": ["player_dead"],
                "step_checks": ["fire_started"],
            },
        },
        "report": {},
    }
    env = _project_to_legacy_envelope(_make_skill(), prior_envelope=prior)
    p = env["skill"]["protocol"]
    assert p["predicate_success"] == ["world.e1"]
    assert p["predicate_abort"] == ["player_dead"]
    assert p["step_checks"] == ["fire_started"]


# ---------------------------------------------------------------------------
# End-to-end: writeback_promotion
# ---------------------------------------------------------------------------


def _write_snapshot(tmp_path: Path, snap: Dict[str, Any]) -> Path:
    p = tmp_path / "bank_snapshots" / f"{snap['snapshot_id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap), encoding="utf-8")
    return p


def test_writeback_inserts_into_empty_legacy_bank(tmp_path: Path):
    snap = _make_snapshot([
        _make_skill(skill_id="skill-A"),
        _make_skill(skill_id="skill-B"),
    ])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )

    assert isinstance(report, WritebackReport)
    assert report.snapshot_id == "snap-test-01"
    assert report.n_total_in_snapshot == 2
    assert report.n_eligible == 2
    assert report.n_inserted == 2
    assert report.n_updated == 0
    assert report.n_skipped_status == 0
    assert report.inserted_skill_ids == ["skill-A", "skill-B"]

    lines = legacy_bank.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {p["skill"]["skill_id"] for p in parsed} == {"skill-A", "skill-B"}


def test_writeback_upserts_existing_and_preserves_canonical_report(tmp_path: Path):
    """Upsert path: the ``skill`` block updates from the snapshot, the
    canonical ``VerificationReport`` fields in the prior ``report`` block
    are preserved.  Bogus fields are stripped (would break ``load()``)."""
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"
    legacy_bank.parent.mkdir(parents=True)
    legacy_bank.write_text(
        json.dumps({
            "skill": {
                "skill_id": "skill-A",
                "name": "OldName",
                "version": 7,
                "protocol": {"steps": ["old step"]},
                "contract": {"eff_add": ["old_eff"]},
                "evidence_role": "GATHER",
                "applicable_domains": ["gymv"],
            },
            "report": {
                # Canonical fields — preserved.
                "skill_id": "skill-A",
                "n_instances": 42,
                "overall_pass_rate": 0.5,
                # Non-canonical — must be dropped.
                "selection_count": 99,
                "pass_rate_legacy_field": 0.5,
            },
        }) + "\n",
        encoding="utf-8",
    )

    snap = _make_snapshot([_make_skill(skill_id="skill-A", name="NewName")])
    snapshot_path = _write_snapshot(tmp_path, snap)

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    assert report.n_inserted == 0
    assert report.n_updated == 1
    assert report.updated_skill_ids == ["skill-A"]

    lines = legacy_bank.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    env = json.loads(lines[0])
    # New name takes effect.
    assert env["skill"]["name"] == "NewName"
    # Canonical report fields preserved.
    assert env["report"]["n_instances"] == 42
    assert env["report"]["overall_pass_rate"] == 0.5
    # Non-canonical fields stripped.
    assert "selection_count" not in env["report"]
    assert "pass_rate_legacy_field" not in env["report"]


def test_writeback_filters_by_status(tmp_path: Path):
    snap = _make_snapshot([
        _make_skill(skill_id="skill-prov", status="provisional"),
        _make_skill(skill_id="skill-cand", status="candidate"),
        _make_skill(skill_id="skill-rej", status="rejected"),
        _make_skill(skill_id="skill-active", status="active"),
        _make_skill(skill_id="skill-no-status", status=""),
    ])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    # Default eligible set is {active, provisional, shadow}
    assert report.eligible_statuses == DEFAULT_ELIGIBLE_STATUSES
    assert report.n_eligible == 2
    assert set(report.inserted_skill_ids) == {"skill-prov", "skill-active"}
    # candidate / rejected / "" are skipped via status filter
    assert report.n_skipped_status == 3


def test_writeback_custom_status_filter(tmp_path: Path):
    snap = _make_snapshot([
        _make_skill(skill_id="skill-A", status="provisional"),
        _make_skill(skill_id="skill-B", status="candidate"),
    ])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
        eligible_statuses={"candidate", "provisional"},
    )
    assert report.n_eligible == 2
    assert set(report.inserted_skill_ids) == {"skill-A", "skill-B"}


def test_writeback_skips_invalid_skill_ids(tmp_path: Path):
    bad_a = _make_skill(skill_id="")
    bad_a.pop("skill_id", None)
    bad_a["skill_id"] = ""  # empty string
    bad_b = _make_skill(skill_id="will-be-removed")
    bad_b.pop("skill_id")
    snap = _make_snapshot([bad_a, bad_b, _make_skill(skill_id="ok")])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    assert report.n_skipped_invalid == 2
    assert report.inserted_skill_ids == ["ok"]


def test_writeback_dry_run_does_not_write(tmp_path: Path):
    snap = _make_snapshot([_make_skill(skill_id="skill-A")])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
        dry_run=True,
    )
    assert report.n_inserted == 1
    assert not legacy_bank.exists()


def test_writeback_creates_parent_dirs(tmp_path: Path):
    snap = _make_snapshot([_make_skill(skill_id="skill-A")])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "deep" / "nested" / "path" / "skill_bank.jsonl"

    writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    assert legacy_bank.is_file()


def test_writeback_preserves_existing_skill_order(tmp_path: Path):
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"
    legacy_bank.parent.mkdir(parents=True)
    # Pre-existing skills X, Y, Z in that order
    with legacy_bank.open("w", encoding="utf-8") as f:
        for sid in ("X", "Y", "Z"):
            f.write(json.dumps({
                "skill": {"skill_id": sid, "protocol": {}, "contract": {}},
                "report": {"selection_count": 0},
            }) + "\n")

    # Snapshot upserts Y and inserts new W (in that order)
    snap = _make_snapshot([
        _make_skill(skill_id="Y"),
        _make_skill(skill_id="W"),
    ])
    snapshot_path = _write_snapshot(tmp_path, snap)

    writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    final_ids = [
        json.loads(line)["skill"]["skill_id"]
        for line in legacy_bank.read_text(encoding="utf-8").splitlines()
    ]
    # X, Y, Z preserved in original order; W appended at the tail.
    # X and Z are NOT touched (they weren't in the snapshot at all).
    assert final_ids == ["X", "Y", "Z", "W"]


def test_writeback_atomic_no_partial_file_on_error(tmp_path: Path, monkeypatch):
    """Force ``os.replace`` to fail; the legacy bank's prior contents
    must still be readable (no partial write)."""
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"
    legacy_bank.parent.mkdir(parents=True)
    legacy_bank.write_text(
        json.dumps({
            "skill": {"skill_id": "X", "protocol": {}, "contract": {}},
            "report": {"selection_count": 99},
        }) + "\n",
        encoding="utf-8",
    )

    snap = _make_snapshot([_make_skill(skill_id="A")])
    snapshot_path = _write_snapshot(tmp_path, snap)

    import skill_bank.legacy_writeback as mod
    orig_replace = mod.os.replace

    def boom(*a, **kw):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(mod.os, "replace", boom)

    with pytest.raises(OSError, match="simulated replace failure"):
        writeback_promotion(
            snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
        )

    monkeypatch.setattr(mod.os, "replace", orig_replace)
    # Original content still intact
    line = legacy_bank.read_text(encoding="utf-8").strip()
    assert json.loads(line)["skill"]["skill_id"] == "X"
    assert json.loads(line)["report"]["selection_count"] == 99


def test_writeback_handles_missing_snapshot(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        writeback_promotion(
            snapshot_path=tmp_path / "nope.json",
            legacy_bank_path=tmp_path / "skill_bank.jsonl",
        )


def test_writeback_handles_empty_body(tmp_path: Path):
    snap = _make_snapshot([])
    snapshot_path = _write_snapshot(tmp_path, snap)
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    assert report.n_total_in_snapshot == 0
    assert report.n_inserted == 0
    # Empty file is fine — the legacy bank exists but has zero lines.
    assert legacy_bank.is_file()
    assert legacy_bank.read_text(encoding="utf-8") == ""


def test_writeback_skips_malformed_legacy_lines(tmp_path: Path):
    legacy_bank = tmp_path / "legacy" / "skill_bank.jsonl"
    legacy_bank.parent.mkdir(parents=True)
    legacy_bank.write_text(
        '{"skill": {"skill_id": "valid"}, "report": {}}\n'
        'not-json-at-all\n'
        '{"skill": "string-not-dict"}\n',
        encoding="utf-8",
    )
    snap = _make_snapshot([_make_skill(skill_id="new-one")])
    snapshot_path = _write_snapshot(tmp_path, snap)

    report = writeback_promotion(
        snapshot_path=snapshot_path, legacy_bank_path=legacy_bank,
    )
    assert report.n_inserted == 1
    final = legacy_bank.read_text(encoding="utf-8").splitlines()
    # Valid entry preserved + new entry appended; malformed lines dropped.
    ids = [json.loads(line)["skill"]["skill_id"] for line in final]
    assert ids == ["valid", "new-one"]


# ---------------------------------------------------------------------------
# find_latest_snapshot
# ---------------------------------------------------------------------------


def test_find_latest_snapshot_returns_none_when_dir_missing(tmp_path: Path):
    assert find_latest_snapshot(tmp_path / "nope") is None


def test_find_latest_snapshot_returns_none_when_empty(tmp_path: Path):
    (tmp_path / "bank_snapshots").mkdir()
    assert find_latest_snapshot(tmp_path) is None


def test_find_latest_snapshot_picks_most_recent(tmp_path: Path):
    snap_dir = tmp_path / "bank_snapshots"
    snap_dir.mkdir()
    older = snap_dir / "snap-001.json"
    newer = snap_dir / "snap-002.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    import os as _os
    # Make `older` older by 10s
    _os.utime(older, (1.0, 1.0))
    _os.utime(newer, (2.0, 2.0))

    picked = find_latest_snapshot(tmp_path)
    assert picked == newer


def test_find_latest_snapshot_ignores_non_snap_files(tmp_path: Path):
    snap_dir = tmp_path / "bank_snapshots"
    snap_dir.mkdir()
    (snap_dir / "snap-001.json").write_text("{}", encoding="utf-8")
    (snap_dir / "README.md").write_text("docs", encoding="utf-8")
    (snap_dir / "snap-002.txt").write_text("not json", encoding="utf-8")

    picked = find_latest_snapshot(tmp_path)
    assert picked is not None
    assert picked.name == "snap-001.json"


# ---------------------------------------------------------------------------
# Smoke test against the Phase-0 real snapshot (when present)
# ---------------------------------------------------------------------------


def _real_phase_0_snapshot() -> Path:
    """Locate the Phase-0 Airstriker snapshot if it exists on disk."""
    root = (
        Path(__file__).resolve().parent.parent
        / "labeling_supplement" / "promotion_decisions_out"
    )
    if not root.is_dir():
        return Path()
    pair_dirs = sorted(root.glob("run_*/gym_v/Temporal_Airstriker-v0"))
    for pair in reversed(pair_dirs):                   # newest first
        snap = find_latest_snapshot(pair)
        if snap is not None and snap.is_file():
            return snap
    return Path()


def test_smoke_against_real_phase_0_snapshot(tmp_path: Path):
    """Runs the writeback end-to-end on the live snapshot
    ``decide_promotion_gpt54.py`` produced. Skipped on a fresh checkout."""
    snap = _real_phase_0_snapshot()
    if not snap.is_file():
        pytest.skip("no real Phase-0 snapshot on disk")

    legacy_bank = tmp_path / "skill_bank.jsonl"
    report = writeback_promotion(
        snapshot_path=snap, legacy_bank_path=legacy_bank,
    )
    # Phase-0 should produce ≥1 PROVISIONAL skill.
    assert report.n_total_in_snapshot >= 1
    assert report.n_eligible >= 1
    assert report.n_inserted >= 1
    # Every emitted line must parse and have the legacy envelope shape.
    for line in legacy_bank.read_text(encoding="utf-8").splitlines():
        env = json.loads(line)
        assert isinstance(env.get("skill"), dict)
        assert isinstance(env.get("report"), dict)
        s = env["skill"]
        assert s["skill_id"]
        assert "protocol" in s
        assert isinstance(s["protocol"].get("steps"), list)
        # Every step must be a string (the legacy reader's contract).
        for step in s["protocol"]["steps"]:
            assert isinstance(step, str)
