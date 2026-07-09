"""Regression tests for the resume-time skill-bank restore path.

Guards against the failure mode reproduced in run
``Qwen3.5-9B_20260504_144712`` where a mid-step trainer crash caused
the curriculum launcher to respawn a fresh trainer process which:

1. Successfully restored the 5 LoRA adapters from the latest
   checkpoint (step 9, 20 skills on disk).
2. Then silently failed to restore the in-memory skill bank because
   :meth:`PerGameSkillBankManager.get_agents` returned ``{game: None}``
   for the still-lazy pipelines, and
   :func:`trainer.coevolution.checkpoint.load_checkpoint` no-op'd the
   bank load via its ``if agent is None: continue`` clause.
3. The orchestrator's next step then read ``n_total_skills = 0`` from
   ``sb_manager.skill_counts()`` (also lazy-gated), flipped into
   spurious cold-start mode, and re-bootstrapped the bank from the
   cold-start labels — losing the 20 skills the prior trainer had
   accumulated.

The fix is a new :meth:`PerGameSkillBankManager.ensure_agents_initialized`
method that eagerly drives lazy ``_ensure_agent`` on every pipeline
*before* the orchestrator calls ``load_checkpoint``, so the
``bank_agents`` dict the loader iterates is fully populated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass, field

import pytest

from trainer.coevolution.checkpoint import load_checkpoint, save_checkpoint
from trainer.coevolution.skillbank_pipeline import (
    PerGameSkillBankManager,
    SharedSkillBankManager,
)


# ---------------------------------------------------------------------------
# Stubs (mirror tests/test_bank_reload.py — keeps test cost low and avoids
# pulling the full SkillBankAgent which transitively imports vllm-style deps)
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

    def save(self, path: str = "") -> None:                  # pragma: no cover
        # Not exercised by the resume path; defined so the public surface
        # matches SkillBankMVP for future tests that may need it.
        return


@dataclass
class _StubAgent:
    bank: _StubBank
    n_ensure_calls: int = 0
    invalidations: int = 0

    def load(self) -> None:
        self.bank.load()

    def _invalidate_query_engine(self) -> None:
        self.invalidations += 1


def _write_bank_jsonl(path: Path, skill_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in skill_ids:
            f.write(json.dumps({
                "skill": {
                    "skill_id": sid,
                    "name": sid,
                    "evidence_role": "COMMIT",
                    "applicable_domains": ["gymv"],
                    "protocol": {"steps": []},
                    "contract": {"eff_add": [], "eff_del": []},
                },
                "report": {},
            }) + "\n")


# ---------------------------------------------------------------------------
# PerGameSkillBankManager.ensure_agents_initialized
# ---------------------------------------------------------------------------


def test_ensure_agents_initialized_returns_concrete_agents_per_game(monkeypatch, tmp_path: Path):
    mgr = PerGameSkillBankManager(
        games=["tetris", "Temporal_Airstriker-v0"],
        bank_dir=str(tmp_path / "skillbank"),
        unified_role_rollouts=False,
    )
    # Sanity: lazy state — get_agents returns {game: None} on a fresh manager.
    pre = mgr.get_agents()
    assert pre == {"tetris": None, "Temporal_Airstriker-v0": None}

    # Stub out the heavyweight `_ensure_agent` so the test doesn't import
    # the real SkillBankAgent (vllm-style deps).
    def _make_stub(self) -> _StubAgent:                       # noqa: ANN001
        bank_file = Path(self.bank_dir) / "skill_bank.jsonl"
        agent = _StubAgent(bank=_StubBank(bank_path=bank_file))
        agent.n_ensure_calls += 1
        self._agent = agent
        return agent
    monkeypatch.setattr(
        "trainer.coevolution.skillbank_pipeline.AsyncSkillBankPipeline._ensure_agent",
        _make_stub, raising=True,
    )

    out = mgr.ensure_agents_initialized()
    assert set(out.keys()) == {"tetris", "Temporal_Airstriker-v0"}
    assert all(isinstance(a, _StubAgent) for a in out.values())
    # And get_agents now sees them, too.
    post = mgr.get_agents()
    assert all(a is not None for a in post.values())


def test_ensure_agents_initialized_handles_per_pipeline_failures(monkeypatch, tmp_path: Path):
    """One pipeline raising shouldn't crash the whole manager."""
    mgr = PerGameSkillBankManager(
        games=["good", "bad"],
        bank_dir=str(tmp_path / "skillbank"),
        unified_role_rollouts=False,
    )

    def _selective_init(self) -> _StubAgent:                  # noqa: ANN001
        if Path(self.bank_dir).name == "bad":
            raise RuntimeError("simulated lazy-init failure")
        bank_file = Path(self.bank_dir) / "skill_bank.jsonl"
        agent = _StubAgent(bank=_StubBank(bank_path=bank_file))
        self._agent = agent
        return agent
    monkeypatch.setattr(
        "trainer.coevolution.skillbank_pipeline.AsyncSkillBankPipeline._ensure_agent",
        _selective_init, raising=True,
    )

    out = mgr.ensure_agents_initialized()
    assert out["good"] is not None
    assert out["bad"] is None      # graceful: failed key returns None, not raised


# ---------------------------------------------------------------------------
# Resume integration: ensure_agents_initialized + load_checkpoint really
# does restore per-game banks (this is the bug that bit v12-instrumented).
# ---------------------------------------------------------------------------


def test_resume_restores_per_game_banks_from_checkpoint(monkeypatch, tmp_path: Path):
    """End-to-end: simulate a saved checkpoint + a freshly-respawned
    trainer with lazy pipelines, then run the new resume sequence and
    assert per-game banks actually got loaded."""
    games = ["tetris", "Temporal_Airstriker-v0"]
    bank_dir = tmp_path / "skillbank"

    mgr = PerGameSkillBankManager(
        games=games, bank_dir=str(bank_dir), unified_role_rollouts=False,
    )

    # Wire a stub `_ensure_agent` so we don't pull the heavy real impl.
    def _make_stub(self) -> _StubAgent:                       # noqa: ANN001
        if self._agent is not None:
            return self._agent
        bank_file = Path(self.bank_dir) / "skill_bank.jsonl"
        agent = _StubAgent(bank=_StubBank(bank_path=bank_file))
        if bank_file.is_file():
            agent.bank.load()
        self._agent = agent
        return agent
    monkeypatch.setattr(
        "trainer.coevolution.skillbank_pipeline.AsyncSkillBankPipeline._ensure_agent",
        _make_stub, raising=True,
    )

    # ── Set up a saved checkpoint with 3 skills per game ─────────────
    ckpt_dir = tmp_path / "checkpoints"
    snap_step = 9
    snap_path = ckpt_dir / f"step_{snap_step:04d}"
    for g in games:
        _write_bank_jsonl(
            snap_path / "banks" / g / "skill_bank.jsonl",
            [f"{g}_skill_{i}" for i in range(3)],
        )
    # Minimal metadata so find_latest_checkpoint sees this snapshot.
    (snap_path / "metadata.json").write_text(json.dumps({"step": snap_step}))

    # Simulate a freshly-respawned trainer: pipelines are lazy,
    # in-memory banks are empty.
    assert mgr.get_agents() == {g: None for g in games}, "fresh manager must be lazy"

    # ── Apply the new resume sequence (mirrors orchestrator) ─────────
    mgr.ensure_agents_initialized()
    load_checkpoint(
        checkpoint_dir=str(ckpt_dir),
        step=snap_step,
        adapter_dir=str(tmp_path / "adapters"),     # empty dir; no adapters exercised here
        bank_agents=mgr.get_agents(),
    )

    # ── Assert per-game banks actually got restored ──────────────────
    for g in games:
        agent = mgr.get_agents()[g]
        assert agent is not None, f"{g}: agent stayed lazy/None — resume failed"
        ids = sorted(agent.bank.skill_ids)
        assert ids == [f"{g}_skill_0", f"{g}_skill_1", f"{g}_skill_2"], \
            f"{g}: bank not restored from checkpoint, got {ids}"


def test_resume_without_ensure_init_is_a_silent_no_op(monkeypatch, tmp_path: Path):
    """Negative test that documents the original bug — without the
    eager init step, ``load_checkpoint`` silently skips the bank
    restore.  Pin this so a future refactor that breaks the eager-init
    contract surfaces here."""
    games = ["tetris"]
    bank_dir = tmp_path / "skillbank"
    mgr = PerGameSkillBankManager(
        games=games, bank_dir=str(bank_dir), unified_role_rollouts=False,
    )
    monkeypatch.setattr(
        "trainer.coevolution.skillbank_pipeline.AsyncSkillBankPipeline._ensure_agent",
        lambda self: setattr(self, "_agent", _StubAgent(
            bank=_StubBank(bank_path=Path(self.bank_dir) / "skill_bank.jsonl")
        )) or self._agent, raising=True,
    )

    ckpt_dir = tmp_path / "checkpoints"
    snap_path = ckpt_dir / "step_0009"
    _write_bank_jsonl(
        snap_path / "banks" / "tetris" / "skill_bank.jsonl",
        ["should_not_load_without_eager_init"],
    )
    (snap_path / "metadata.json").write_text(json.dumps({"step": 9}))

    # Skip ensure_agents_initialized — exactly the old buggy resume path.
    load_checkpoint(
        checkpoint_dir=str(ckpt_dir), step=9,
        adapter_dir=str(tmp_path / "adapters"),
        bank_agents=mgr.get_agents(),     # ← {game: None}
    )
    # Bug confirmed: nothing restored, agent still lazy.
    assert mgr.get_agents()["tetris"] is None


# ---------------------------------------------------------------------------
# SharedSkillBankManager parity
# ---------------------------------------------------------------------------


def test_shared_manager_ensure_agents_initialized(monkeypatch, tmp_path: Path):
    mgr = SharedSkillBankManager(
        games=["tetris", "Temporal_Airstriker-v0"],
        bank_dir=str(tmp_path / "shared"),
    )
    pre = mgr.get_agents()
    assert pre == {"tetris": None, "Temporal_Airstriker-v0": None}

    def _make_stub(self) -> _StubAgent:                       # noqa: ANN001
        if self._agent is not None:
            return self._agent
        bank_file = Path(self.bank_dir) / "skill_bank.jsonl"
        agent = _StubAgent(bank=_StubBank(bank_path=bank_file))
        self._agent = agent
        return agent
    monkeypatch.setattr(
        "trainer.coevolution.skillbank_pipeline.AsyncSkillBankPipeline._ensure_agent",
        _make_stub, raising=True,
    )

    out = mgr.ensure_agents_initialized()
    # Same agent instance shared across both game keys.
    assert out["tetris"] is out["Temporal_Airstriker-v0"]
    assert isinstance(out["tetris"], _StubAgent)
