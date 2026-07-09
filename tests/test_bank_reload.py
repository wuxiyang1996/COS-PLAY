"""Tests for ``AsyncSkillBankPipeline.reload_bank_from_disk`` and
``PerGameSkillBankManager.reload_banks_from_disk``.

These cover the Phase B′ correctness requirement that the live actor's
read path observes writeback-promoted skills on the *next* step. Without
the reload, ``SkillBankMVP._skills`` and ``SkillQueryEngine._skill_id_order``
are both stale and the actor never sees the new skills — see
``implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md``
§"actor read path" for the failure trace these tests guard against.

We avoid initialising the heavyweight real ``SkillBankAgent`` (which
pulls in vLLM-style deps) by injecting a tiny stub agent on
``AsyncSkillBankPipeline._agent`` directly.  This lets us drive
``reload_bank_from_disk()`` on the same logical plumbing the live
trainer uses, without paying for the full pipeline import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from skill_bank.legacy_writeback import writeback_promotion
from trainer.coevolution.skillbank_pipeline import (
    AsyncSkillBankPipeline,
    PerGameSkillBankManager,
)


# ---------------------------------------------------------------------------
# Lightweight stub agent — exposes the same surface
# ``reload_bank_from_disk`` reads (``bank``, ``load()``,
# ``_invalidate_query_engine()``).
# ---------------------------------------------------------------------------


@dataclass
class _StubBank:
    bank_path: Path
    skill_ids: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.skill_ids)

    def load(self, path: str = "") -> None:
        target = Path(path) if path else self.bank_path
        if not target.is_file():
            self.skill_ids = []
            return
        ids: List[str] = []
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = ((obj.get("skill") or {}).get("skill_id"))
                if sid:
                    ids.append(sid)
        self.skill_ids = ids


@dataclass
class _StubAgent:
    bank: _StubBank
    invalidations: int = 0
    n_loads: int = 0

    def load(self) -> None:
        self.bank.load()
        self.n_loads += 1

    def _invalidate_query_engine(self) -> None:
        self.invalidations += 1


def _seed_pipeline(tmp_path: Path, *, initial_ids: List[str]) -> tuple:
    """Build a real `AsyncSkillBankPipeline`, force-inject a stub agent
    so we don't pay for the heavy real init, and pre-populate the on-disk
    bank with the given ids.

    Returns ``(pipeline, bank_path, stub_agent)``.
    """
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    bank_path = bank_dir / "skill_bank.jsonl"
    with bank_path.open("w", encoding="utf-8") as f:
        for sid in initial_ids:
            f.write(json.dumps({
                "skill": {
                    "skill_id": sid, "name": sid,
                    "evidence_role": "COMMIT",
                    "applicable_domains": ["gymv"],
                    "protocol": {"steps": [f"{sid}-step"]},
                    "contract": {"eff_add": [], "eff_del": []},
                },
                "report": {},
            }) + "\n")

    pipe = AsyncSkillBankPipeline(bank_dir=str(bank_dir), game_name="testgame")
    stub_bank = _StubBank(bank_path=bank_path, skill_ids=list(initial_ids))
    stub_agent = _StubAgent(bank=stub_bank)
    pipe._agent = stub_agent                                  # bypass real init
    return pipe, bank_path, stub_agent


# ---------------------------------------------------------------------------
# AsyncSkillBankPipeline.reload_bank_from_disk
# ---------------------------------------------------------------------------


def test_reload_returns_skipped_when_agent_not_initialised(tmp_path: Path):
    pipe = AsyncSkillBankPipeline(bank_dir=str(tmp_path), game_name="g")
    rep = pipe.reload_bank_from_disk()
    assert rep["reloaded"] is False
    assert rep["agent_initialised"] is False
    assert "skipped_reason" in rep


def test_reload_returns_skipped_when_bank_file_missing(tmp_path: Path):
    pipe = AsyncSkillBankPipeline(bank_dir=str(tmp_path), game_name="g")
    pipe._agent = _StubAgent(bank=_StubBank(bank_path=Path("/nonexistent")))
    rep = pipe.reload_bank_from_disk()
    assert rep["reloaded"] is False
    assert rep["agent_initialised"] is True
    assert "missing on disk" in rep["skipped_reason"]


def test_reload_picks_up_new_skill_ids_and_invalidates_caches(tmp_path: Path):
    pipe, bank_path, agent = _seed_pipeline(tmp_path, initial_ids=["A"])
    pipe._query_engine = "stale_cache"                       # any non-None sentinel

    # Mutate the file on disk (simulating the writeback hook).
    with bank_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "skill": {
                "skill_id": "B-promoted", "name": "B-promoted",
                "evidence_role": "COMMIT",
                "applicable_domains": ["gymv"],
                "protocol": {"steps": ["B step"]},
                "contract": {"eff_add": [], "eff_del": []},
            },
            "report": {},
        }) + "\n")

    rep = pipe.reload_bank_from_disk()
    assert rep["reloaded"] is True
    assert rep["n_before"] == 1
    assert rep["n_after"] == 2
    assert rep["query_engine_invalidated"] is True
    # In-memory bank actually saw the new entry.
    assert "B-promoted" in agent.bank.skill_ids
    # Pipeline-level query engine cache cleared.
    assert pipe._query_engine is None
    # Agent-level invalidator was called exactly once.
    assert agent.invalidations == 1
    assert agent.n_loads == 1


def test_reload_handles_agent_load_exception(tmp_path: Path):
    pipe, bank_path, agent = _seed_pipeline(tmp_path, initial_ids=["A"])

    def _boom() -> None:
        raise RuntimeError("simulated agent.load failure")
    agent.load = _boom                                       # type: ignore[assignment]

    rep = pipe.reload_bank_from_disk()
    assert rep["reloaded"] is False
    assert rep["query_engine_invalidated"] is False
    assert "agent.load() failed" in rep["skipped_reason"]
    # On failure we deliberately don't clear the pipeline's cache —
    # we'd rather serve a known-stable snapshot than an empty one.
    # (Document this behaviour with the assertion.)
    pipe._query_engine = "kept"
    rep2 = pipe.reload_bank_from_disk()
    assert pipe._query_engine == "kept"


# ---------------------------------------------------------------------------
# PerGameSkillBankManager.reload_banks_from_disk
# ---------------------------------------------------------------------------


def test_manager_reload_dispatches_to_each_pipeline(tmp_path: Path):
    mgr = PerGameSkillBankManager(
        games=["tetris", "Temporal_Airstriker-v0"],
        bank_dir=str(tmp_path / "skillbank"),
        unified_role_rollouts=False,
    )
    # Replace each pipeline's agent with a stub that knows its bank file.
    for key, pipe in mgr._pipelines.items():
        bank_path = Path(pipe.bank_dir) / "skill_bank.jsonl"
        bank_path.write_text(json.dumps({
            "skill": {
                "skill_id": f"S_{key}", "name": f"S_{key}",
                "evidence_role": "COMMIT",
                "applicable_domains": ["gymv"],
                "protocol": {"steps": ["a"]},
                "contract": {"eff_add": [], "eff_del": []},
            }, "report": {},
        }) + "\n", encoding="utf-8")
        pipe._agent = _StubAgent(bank=_StubBank(bank_path=bank_path, skill_ids=[]))

    out = mgr.reload_banks_from_disk()
    assert set(out.keys()) == {"tetris", "Temporal_Airstriker-v0"}
    for key, rep in out.items():
        assert rep["reloaded"] is True
        assert rep["n_after"] == 1
        # In-memory state actually mutated.
        assert mgr._pipelines[key]._agent.bank.skill_ids == [f"S_{key}"]


def test_manager_reload_keys_filter(tmp_path: Path):
    mgr = PerGameSkillBankManager(
        games=["tetris", "Temporal_Airstriker-v0"],
        bank_dir=str(tmp_path / "skillbank"),
        unified_role_rollouts=False,
    )
    for key, pipe in mgr._pipelines.items():
        bank_path = Path(pipe.bank_dir) / "skill_bank.jsonl"
        bank_path.write_text(json.dumps({
            "skill": {
                "skill_id": f"S_{key}", "name": f"S_{key}",
                "evidence_role": "COMMIT", "applicable_domains": ["gymv"],
                "protocol": {"steps": []}, "contract": {"eff_add": [], "eff_del": []},
            }, "report": {},
        }) + "\n", encoding="utf-8")
        pipe._agent = _StubAgent(bank=_StubBank(bank_path=bank_path, skill_ids=[]))

    out = mgr.reload_banks_from_disk(keys=["tetris"])
    assert set(out.keys()) == {"tetris"}
    # The other pipeline's agent was untouched.
    assert mgr._pipelines["Temporal_Airstriker-v0"]._agent.n_loads == 0


def test_manager_reload_unknown_keys_silently_skipped(tmp_path: Path):
    mgr = PerGameSkillBankManager(
        games=["tetris"],
        bank_dir=str(tmp_path / "skillbank"),
        unified_role_rollouts=False,
    )
    out = mgr.reload_banks_from_disk(keys=["nonexistent", "tetris"])
    # Only the existing key shows up; unknown is silently dropped.
    assert "nonexistent" not in out
    assert "tetris" in out


# ---------------------------------------------------------------------------
# Integration: writeback → reload → in-memory bank actually sees new skill
# ---------------------------------------------------------------------------


def test_writeback_then_reload_propagates_to_in_memory_bank(tmp_path: Path):
    """The whole point of this module: simulate Phase B′
    writeback (via the production projector) then reload, and confirm
    the in-memory bank actually sees the new skill_id the writeback
    inserted."""
    pipe, bank_path, agent = _seed_pipeline(tmp_path, initial_ids=["EXISTING"])

    # Drive the production legacy_writeback against a synthetic snapshot.
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps({
        "snapshot_id": "snap-test",
        "body": {"skills": [{
            "skill_id": "PROMOTED-via-writeback",
            "name": "promoted",
            "skill_type": "action",
            "source_type": "repaired_from_failure",
            "status": "provisional",
            "version": "v1.test",
            "feasible_domains": ["gymv"],
            "verified_domains": [],
            "protocol": [{"action": "EXEC", "payload": {}, "notes": "step"}],
            "contract": {
                "preconditions": [], "effects_add": [], "effects_del": [],
                "expected_evidence_roles": ["COMMIT"],
                "success_criteria": [], "abort_criteria": [],
            },
        }]},
    }), encoding="utf-8")

    wb = writeback_promotion(snapshot_path=snap_path, legacy_bank_path=bank_path)
    assert wb.n_inserted == 1

    # Pre-reload sanity: in-memory bank still has only the old skill.
    assert "PROMOTED-via-writeback" not in agent.bank.skill_ids

    # Reload + verify.
    rep = pipe.reload_bank_from_disk()
    assert rep["reloaded"] is True
    assert rep["n_after"] == 2
    assert "PROMOTED-via-writeback" in agent.bank.skill_ids
    # Stable on a second reload (idempotent).
    rep2 = pipe.reload_bank_from_disk()
    assert rep2["reloaded"] is True
    assert rep2["n_after"] == 2
