"""Empirical gate tests for the writeback ↔ live-trainer boundary.

These tests guard against the class of bug the original test suite
missed by being tautological — they exercise the writeback's *output*
through the **production code paths** the trainer actually uses
(``skill_agents.skill_bank.bank.SkillBankMVP.load()``,
``skill_agents.query.SkillQueryEngine``, real
``trainer.coevolution.skillbank_pipeline.AsyncSkillBankPipeline``),
*not* through stubs.

Concrete failure modes these gates catch (one of them was real and
landed in this very commit; see the audit thread for ``selection_count``
breaking ``VerificationReport.from_dict``):

* The legacy ``report`` block schema drifts from
  ``VerificationReport`` and ``SkillBankMVP.load()`` raises ``TypeError``.
* The legacy ``skill`` block has a field shape that ``Skill.from_dict``
  rejects (e.g. ``protocol.steps`` containing non-strings).
* ``SkillQueryEngine`` silently filters skills missing some required
  predicate / contract field.
* ``reload_bank_from_disk()`` works on the stub but not on the real
  ``AsyncSkillBankPipeline`` plumbing (e.g. ``_query_engine`` cleared
  but the actor reads through some other cached path).

Skipped cleanly when the on-disk Phase-0 fixtures aren't present so a
fresh checkout still passes the suite.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Set

import pytest

# These imports MUST be the production targets — no stubs.
from skill_agents.skill_bank.bank import SkillBankMVP
from skill_bank.legacy_writeback import (
    _empty_report,
    _project_to_verification_report,
    find_latest_snapshot,
    writeback_promotion,
)
from trainer.coevolution.skillbank_pipeline import AsyncSkillBankPipeline


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _real_airstriker_bank() -> Path:
    p = (
        _repo_root()
        / "labeling" / "skill_bank_out" / "run_20260430_030637"
        / "gym_v" / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    )
    return p if p.is_file() else Path()


def _real_phase_0_snapshot() -> Path:
    pair = (
        _repo_root()
        / "labeling_supplement" / "promotion_decisions_out"
    )
    if not pair.is_dir():
        return Path()
    for run in sorted(pair.glob("run_*"), reverse=True):
        ar = run / "gym_v" / "Temporal_Airstriker-v0"
        snap = find_latest_snapshot(ar) if ar.is_dir() else None
        if snap is not None:
            return snap
    return Path()


# ---------------------------------------------------------------------------
# Pure-helper unit gates (catch the regression at the projector level)
# ---------------------------------------------------------------------------


def test_empty_report_only_contains_verification_report_keys():
    """The bug that motivated this whole audit: ``_empty_report`` used
    to emit ``selection_count`` / ``pass_rate`` / ``source`` keys that
    ``VerificationReport.__init__`` rejects. This test pins the
    canonical schema and any future drift fails here loudly."""
    from skill_agents.stage3_mvp.schemas import VerificationReport

    rep = _empty_report(skill_id="x")
    # Round-trips through the real loader without raising.
    obj = VerificationReport.from_dict(rep)
    assert obj.skill_id == "x"
    assert obj.n_instances == 0
    assert obj.overall_pass_rate == 0.0


def test_project_report_drops_unknown_keys():
    """The legacy reader ``cls(**d)`` raises on any unknown key, so the
    projector MUST drop them — even when the prior on-disk envelope had
    them."""
    legacy_blob = {
        "selection_count": 99,                        # bogus key from old writeback
        "pass_rate": 0.42,                            # also bogus
        "source": "promotion-writeback",              # also bogus
        "skill_id": "stale-id",                       # gets overwritten
        "n_instances": 7,                             # canonical — preserved
        "overall_pass_rate": 0.5,                     # canonical — preserved
    }
    out = _project_to_verification_report(legacy_blob, skill_id="actual-id")
    # No extra keys leak through.
    extras = set(out) - {
        "skill_id", "n_instances",
        "eff_add_success_rate", "eff_del_success_rate", "eff_event_rate",
        "overall_pass_rate", "worst_segments", "failure_signatures",
    }
    assert not extras, f"unexpected keys would crash SkillBankMVP.load(): {extras}"
    # Canonical keys survived.
    assert out["n_instances"] == 7
    assert out["overall_pass_rate"] == 0.5
    # skill_id forced to caller value (defensive).
    assert out["skill_id"] == "actual-id"


def test_project_report_handles_non_mapping():
    out = _project_to_verification_report(None, skill_id="x")  # type: ignore[arg-type]
    assert out["skill_id"] == "x"


# ---------------------------------------------------------------------------
# Gate #1: writeback output must load through real SkillBankMVP.load()
# ---------------------------------------------------------------------------


def test_writeback_output_loads_through_real_skillbank_mvp(tmp_path: Path):
    """The bug-finder. Drive the production
    ``writeback_promotion`` against the real Phase-0 snapshot and the
    real Airstriker bank, then load the resulting JSONL through the
    *real* ``SkillBankMVP.load()`` (not a stub).

    Any field shape the projector emits that the loader rejects fails
    this test. Caught the original ``selection_count`` breakage."""
    real_bank = _real_airstriker_bank()
    real_snap = _real_phase_0_snapshot()
    if not real_bank.is_file() or not real_snap.is_file():
        pytest.skip("no Phase-0 fixtures on disk")

    bank_path = tmp_path / "skill_bank.jsonl"
    shutil.copy(real_bank, bank_path)

    pre_load = SkillBankMVP(str(bank_path))
    pre_load.load()
    n_before = len(pre_load)
    ids_before = set(pre_load.skill_ids)

    rep = writeback_promotion(
        snapshot_path=real_snap, legacy_bank_path=bank_path,
    )
    assert rep.n_inserted >= 1, "Phase-0 should produce ≥1 insertion"

    post_load = SkillBankMVP(str(bank_path))
    post_load.load()                                  # MUST NOT RAISE
    n_after = len(post_load)
    ids_after = set(post_load.skill_ids)

    assert n_after == n_before + rep.n_inserted
    # Every promoted skill_id must be visible to the real loader.
    promoted = set(rep.inserted_skill_ids)
    missing = promoted - ids_after
    assert not missing, (
        f"{len(missing)} promoted skill_ids invisible to real loader: "
        f"{sorted(missing)}"
    )
    # Every promoted skill must have its contract + protocol + report
    # round-trip through Skill.from_dict / VerificationReport.from_dict.
    for sid in sorted(promoted)[:5]:
        skill = post_load.get_skill(sid)
        assert skill is not None
        assert skill.contract is not None, f"{sid}: contract dropped on load"
        assert skill.protocol is not None
        assert isinstance(skill.protocol.steps, list)
        for step in skill.protocol.steps:
            # Step shape contract: NL strings (legacy reader expects this).
            assert isinstance(step, str), (
                f"{sid}: protocol step is {type(step).__name__}, expected str"
            )
        # Report block round-tripped through VerificationReport.
        rep_obj = post_load.get_report(sid)
        assert rep_obj is not None, f"{sid}: report dropped on load"
        assert rep_obj.skill_id == sid


# ---------------------------------------------------------------------------
# Gate #2: SkillQueryEngine must actually index the new skills
# ---------------------------------------------------------------------------


def test_writeback_output_visible_to_real_skill_query_engine(tmp_path: Path):
    """Build a real ``SkillQueryEngine`` on top of the writeback-augmented
    bank and assert every promoted ``skill_id`` appears in
    ``_skill_id_order``. Without this, the actor's ``select()`` could
    silently filter out promoted skills (e.g. if they lacked a required
    contract field)."""
    real_bank = _real_airstriker_bank()
    real_snap = _real_phase_0_snapshot()
    if not real_bank.is_file() or not real_snap.is_file():
        pytest.skip("no Phase-0 fixtures on disk")

    bank_path = tmp_path / "skill_bank.jsonl"
    shutil.copy(real_bank, bank_path)
    pre = SkillBankMVP(str(bank_path)); pre.load()
    ids_before = set(pre.skill_ids)

    rep = writeback_promotion(snapshot_path=real_snap, legacy_bank_path=bank_path)
    promoted = set(rep.inserted_skill_ids)
    assert len(promoted) >= 1

    bank2 = SkillBankMVP(str(bank_path)); bank2.load()
    from skill_agents.query import SkillQueryEngine
    eng = SkillQueryEngine(bank2)

    indexed: Set[str] = set(eng._skill_id_order)
    missing_from_index = promoted - indexed
    assert not missing_from_index, (
        f"{len(missing_from_index)} promoted skill_ids missing from "
        f"SkillQueryEngine index: {sorted(missing_from_index)}"
    )
    assert len(indexed) == len(bank2.skill_ids), (
        "engine indexed fewer skills than bank holds — silent filter?"
    )


# ---------------------------------------------------------------------------
# Gate #3: real-driver status case lines up with our eligible filter
# ---------------------------------------------------------------------------


def test_real_driver_status_lowercase_matches_eligible_filter():
    """The eligible-statuses default is ``{"active", "provisional", "shadow"}``
    (lowercase). The driver may, in principle, emit them in any case;
    pin the case from the actual on-disk snapshot to detect drift."""
    real_snap = _real_phase_0_snapshot()
    if not real_snap.is_file():
        pytest.skip("no Phase-0 snapshot on disk")
    body = json.loads(real_snap.read_text(encoding="utf-8"))["body"]
    statuses = {s.get("status") for s in (body.get("skills") or [])}
    assert statuses, "Phase-0 snapshot was empty — fixture broken"
    for st in statuses:
        assert isinstance(st, str)
        assert st == st.lower(), (
            f"driver emitted non-lowercase status {st!r}; "
            f"writeback's eligible filter would silently drop it"
        )
    # Phase-0 specifically: synthetic gate caps at PROVISIONAL.
    assert statuses <= {"provisional", "active", "shadow"}, (
        f"unexpected statuses in synthetic-mode snapshot: {statuses}"
    )


# ---------------------------------------------------------------------------
# Gate #4: real AsyncSkillBankPipeline reload — end-to-end
# ---------------------------------------------------------------------------


def test_real_pipeline_reload_picks_up_writeback(tmp_path: Path):
    """End-to-end gate: real ``AsyncSkillBankPipeline`` →
    real ``SkillBankAgent.load()`` → real ``SkillQueryEngine``. No
    stubs. The exact path the live trainer takes between Phase B′ and
    Phase A of the next step.

    Sequence (mirrors the orchestrator splice exactly):

      1. Pipeline auto-inits, loads the cold-start bank (5 skills).
      2. ``pipe.get_bank()`` builds a ``SkillQueryEngine`` over those 5.
      3. ``writeback_promotion()`` mutates the on-disk JSONL (+13).
      4. ``pipe.reload_bank_from_disk()`` reloads.
      5. ``pipe.get_bank()`` returns a *fresh* engine indexing all 18.

    Catches: any layer in ``AsyncSkillBankPipeline`` that holds a stale
    reference (e.g. a not-yet-known cache, an internal buffer in
    ``SkillBankAgent``)."""
    real_bank = _real_airstriker_bank()
    real_snap = _real_phase_0_snapshot()
    if not real_bank.is_file() or not real_snap.is_file():
        pytest.skip("no Phase-0 fixtures on disk")

    bank_dir = tmp_path / "Temporal_Airstriker-v0"
    bank_dir.mkdir()
    bank_path = bank_dir / "skill_bank.jsonl"
    shutil.copy(real_bank, bank_path)

    pipe = AsyncSkillBankPipeline(
        bank_dir=str(bank_dir),
        model_name="Qwen/Qwen3.5-9B",
        game_name="Temporal_Airstriker-v0",
    )
    agent = pipe._ensure_agent()                      # real SkillBankAgent
    n_before = len(agent.bank)
    assert n_before >= 1, "fixture bank should be non-empty"
    ids_before = set(agent.bank.skill_ids)

    pre_engine = pipe.get_bank()
    assert pre_engine is not None
    pre_indexed = set(getattr(pre_engine, "_skill_id_order", []))
    assert pre_indexed == ids_before

    rep = writeback_promotion(snapshot_path=real_snap, legacy_bank_path=bank_path)
    assert rep.n_inserted >= 1

    # In-memory bank is still stale until we explicitly reload.
    assert len(agent.bank) == n_before

    reload_rep = pipe.reload_bank_from_disk()
    assert reload_rep["reloaded"] is True
    assert reload_rep["n_after"] > reload_rep["n_before"]
    assert reload_rep["query_engine_invalidated"] is True

    # In-memory bank now reflects disk.
    assert len(agent.bank) == n_before + rep.n_inserted

    post_engine = pipe.get_bank()
    assert post_engine is not pre_engine, (
        "reload_bank_from_disk must invalidate the cached engine, "
        "but get_bank() returned the same instance"
    )
    post_indexed = set(getattr(post_engine, "_skill_id_order", []))
    promoted = set(rep.inserted_skill_ids)
    missing = promoted - post_indexed
    assert not missing, (
        f"{len(missing)}/{len(promoted)} promoted skill_ids invisible to "
        f"the freshly-built SkillQueryEngine: {sorted(missing)}"
    )
    # Engine size matches the bank, no silent filter.
    assert len(post_indexed) == len(agent.bank.skill_ids)
