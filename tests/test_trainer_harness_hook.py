"""Unit tests for ``trainer.coevolution._harness_hook.SkillHarnessHook``.

Covers:

* ``filter_candidates`` admits known PROVISIONAL skills with the
  ``gymv`` adapter and vetoes status / domain / task / adapter
  mismatches.
* The ``RejectedSkillSink`` accumulates the per-step vetoes and
  ``flush_to_lifecycle`` lands them on
  ``SkillRecord.false_binding_patterns`` via
  ``SkillLifecycleManager.record_false_binding_pattern``.
* ``validate_choice`` returns the structured ``ok / veto`` verdict and
  degrades gracefully (admit) for skills the cache doesn't know.
* The hook is a no-op pass-through when the candidate list is empty
  or when every candidate is unknown to the cache.
* ``SkillHarnessHook.for_game`` hydrates a per-game bank.jsonl into
  the cache and is robust to missing or malformed lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.enums import SkillSourceType, SkillStatus, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.adapters.gymv_adapter import GymvAdapter
from harness.eligibility import RejectedSkill
from harness.rejected_skill_sink import RejectedSkillSink
from skill_bank.lifecycle import SkillLifecycleManager
from skill_bank.repository import SkillRepository
from skill_bank.stores import SkillStore, StoreName

from trainer.coevolution._harness_hook import (
    HarnessStepStats,
    SkillHarnessHook,
    _hydrate_records_from_bank,
    _state_for_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_record(
    *,
    skill_id: str,
    name: str = "skill",
    skill_type: SkillType = SkillType.ACTION,
    domain: str = "gymv",
    feasible_tasks=None,
    status: SkillStatus = SkillStatus.PROVISIONAL,
    protocol=None,
    contract: SkillContract = None,
    source_domains=None,
) -> SkillRecord:
    rec = SkillRecord.new(
        name=name,
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=[domain],
        feasible_tasks=list(feasible_tasks or []),
        protocol=list(protocol or [{"action": "EXEC", "payload": {}, "notes": "step"}]),
        contract=contract or SkillContract(),
    )
    object.__setattr__(rec, "skill_id", skill_id)
    object.__setattr__(rec, "status", status)
    # Backdoor `source_domains` past the SOURCE_DOMAINS=('gymv',)
    # canonical-tuple validator. Mirrors what the orchestrator's
    # bank-hydration path does for legacy entries that name a foundry
    # corpus outside the trainer-facing canonical set.
    if source_domains is not None:
        object.__setattr__(rec, "source_domains", list(source_domains))
    return rec


def _state(domain: str = "gymv", task: str = "twenty_forty_eight") -> StateSchema:
    return StateSchema(domain=domain, task=task, evidence=[])


def _cand(skill_id: str, **extra) -> dict:
    """Cold-start RAG candidate dict shape."""
    base = {
        "skill_id": skill_id,
        "skill_name": skill_id,
        "execution_hint": "",
        "protocol": {},
        "confidence": 0.5,
        "relevance": 0.5,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# _state_for_step / state_for_step
# ---------------------------------------------------------------------------


def test_state_for_step_minimal():
    s = _state_for_step(
        game="tetris",
        summary_state="empty=10",
        intention="[CLEAR] tetris",
    )
    assert s.domain == "gymv"
    assert s.task == "tetris"
    assert s.extra["summary_state"].startswith("empty=")
    assert s.extra["intention"].startswith("[CLEAR]")


def test_state_for_step_truncates_long_summary():
    long = "x" * 5000
    s = _state_for_step(game="g", summary_state=long, intention="i")
    assert len(s.extra["summary_state"]) == 1024


def test_hook_state_for_step_uses_hook_domain():
    h = SkillHarnessHook(domain="gymv", records={})
    s = h.state_for_step(game="tetris")
    assert s.domain == "gymv"
    assert s.task == "tetris"


# ---------------------------------------------------------------------------
# filter_candidates — eligibility filter
# ---------------------------------------------------------------------------


def test_filter_admits_runnable_gymv_skill():
    rec = _mk_record(skill_id="s1", status=SkillStatus.PROVISIONAL)
    h = SkillHarnessHook(records={"s1": rec})

    state = _state()
    cands = [_cand("s1")]
    out, diag = h.filter_candidates(cands, state)

    assert len(out) == 1
    assert out[0]["skill_id"] == "s1"
    assert out[0]["_harness_eligible"]["adapter_name"] == "gymv"
    assert diag["n_in"] == 1
    assert diag["n_admitted"] == 1
    assert diag["n_rejected"] == 0
    assert diag["eligible_ids"] == ["s1"]


def test_filter_vetos_draft_status():
    """Skill with status=DRAFT is rejected by F1."""
    rec = _mk_record(skill_id="s_draft")
    object.__setattr__(rec, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"s_draft": rec})

    out, diag = h.filter_candidates([_cand("s_draft")], _state())

    assert out == []
    assert diag["n_admitted"] == 0
    assert diag["n_rejected"] == 1
    assert diag["rejected"][0]["veto"] == "status_not_runnable"
    # Sink absorbed the rejection.
    assert len(h.sink) == 1


def test_filter_vetos_domain_mismatch():
    """Skill whose feasible_domains doesn't include the state domain."""
    rec = _mk_record(skill_id="s_browser", domain="browser")
    h = SkillHarnessHook(records={"s_browser": rec})

    state = _state(domain="gymv")
    out, diag = h.filter_candidates([_cand("s_browser")], state)

    assert out == []
    assert diag["rejected"][0]["veto"] in ("domain_mismatch", "no_adapter")


def test_filter_vetos_task_mismatch():
    """Skill with feasible_tasks=[X] is rejected when state.task != X (F2′)."""
    rec = _mk_record(
        skill_id="s_task",
        feasible_tasks=["tetris"],
    )
    h = SkillHarnessHook(records={"s_task": rec})

    out, diag = h.filter_candidates([_cand("s_task")], _state(task="candy_crush"))
    assert out == []
    assert diag["rejected"][0]["veto"] == "task_mismatch"

    # Same skill on the matching task is admitted.
    out2, diag2 = h.filter_candidates([_cand("s_task")], _state(task="tetris"))
    assert len(out2) == 1
    assert diag2["task_match_distribution"].get("same_task", 0) >= 1


def test_filter_passes_through_unknown_skill_ids():
    h = SkillHarnessHook(records={})
    out, diag = h.filter_candidates([_cand("unknown")], _state())
    # Unknown skills bypass the harness — degrade gracefully.
    assert len(out) == 1
    assert out[0]["skill_id"] == "unknown"
    assert "_harness_eligible" not in out[0]
    assert diag["n_unknown"] == 1


def test_filter_empty_candidates_is_noop():
    h = SkillHarnessHook(records={})
    out, diag = h.filter_candidates([], _state())
    assert out == []
    assert diag["n_in"] == 0


# ---------------------------------------------------------------------------
# validate_choice
# ---------------------------------------------------------------------------


def test_validate_choice_admits_runnable_skill():
    rec = _mk_record(skill_id="s1")
    h = SkillHarnessHook(records={"s1": rec})

    ok, d = h.validate_choice("s1", _state())
    assert ok is True
    assert d["status"] == "ok"
    assert d["skill_id"] == "s1"


def test_validate_choice_vetos_when_adapter_missing():
    """A skill whose domain has no registered adapter is vetoed by validate_invocation."""
    rec = _mk_record(skill_id="s_browser", domain="browser")
    # No browser adapter registered in the default hook.
    h = SkillHarnessHook(records={"s_browser": rec})

    state = StateSchema(domain="browser", task="t", evidence=[])
    ok, d = h.validate_choice("s_browser", state)
    assert ok is False
    assert any("missing_adapter" in s for s in d.get("veto_reasons", []))


def test_validate_choice_unknown_skill_admits():
    h = SkillHarnessHook(records={})
    ok, d = h.validate_choice("not_in_cache", _state())
    assert ok is True
    assert d["status"] == "skill_not_in_cache"


def test_validate_choice_no_skill_id_admits():
    h = SkillHarnessHook(records={})
    ok, d = h.validate_choice(None, _state())
    assert ok is True
    assert d["status"] == "no_skill_id_supplied"


# ---------------------------------------------------------------------------
# Sink → lifecycle drainage (Day-9c integration)
# ---------------------------------------------------------------------------


def test_flush_to_lifecycle_writes_false_binding_patterns(tmp_path: Path):
    """End-to-end: filter rejects a skill, flush_to_lifecycle lands the
    veto on `SkillRecord.false_binding_patterns`."""
    # Build a real lifecycle so the write actually round-trips.
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "cand")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "arch")),
    )
    lifecycle = SkillLifecycleManager(repo)

    rec_draft = _mk_record(skill_id="s_draft")
    object.__setattr__(rec_draft, "status", SkillStatus.DRAFT)
    # Seed the lifecycle with the same skill_id as the hook will see,
    # so flush_to_lifecycle finds it.
    seed = _mk_record(skill_id="s_draft")
    object.__setattr__(seed, "status", SkillStatus.DRAFT)
    lifecycle.ingest_draft(seed)
    lifecycle.transition(
        "s_draft",
        to_status=SkillStatus.CANDIDATE,
        rationale="seed",
    )

    h = SkillHarnessHook(records={"s_draft": rec_draft})
    h.filter_candidates([_cand("s_draft")], _state())
    assert len(h.sink) == 1

    report = h.flush_to_lifecycle(lifecycle)
    assert report.n_patterns_written == 1
    assert report.n_skills_touched == 1

    after = lifecycle.get("s_draft")
    assert after is not None
    assert len(after.false_binding_patterns) == 1
    fbp = after.false_binding_patterns[0]
    assert fbp["veto"] == "status_not_runnable"
    assert fbp["domain"] == "gymv"
    assert fbp["task"] == "twenty_forty_eight"


def test_flush_skips_skill_ids_not_in_lifecycle(tmp_path: Path):
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "cand")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "arch")),
    )
    lifecycle = SkillLifecycleManager(repo)

    rec_draft = _mk_record(skill_id="s_orphan")
    object.__setattr__(rec_draft, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"s_orphan": rec_draft})
    h.filter_candidates([_cand("s_orphan")], _state())

    report = h.flush_to_lifecycle(lifecycle)
    assert report.n_patterns_written == 0
    assert "s_orphan" in report.skipped_unknown_skill_ids


# ---------------------------------------------------------------------------
# _hydrate_records_from_bank
# ---------------------------------------------------------------------------


def test_hydrate_records_from_bank_promotes_to_provisional(tmp_path: Path):
    """Skills hydrated from a legacy bank.jsonl come out as PROVISIONAL
    so the eligibility filter admits them."""
    bank_path = tmp_path / "skill_bank.jsonl"
    payload = {
        "skill": {
            "skill_id": "EXEC/CLEAR",
            "name": "exec_clear",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {
                "steps": ["clear the bottom row"],
                "preconditions": [],
                "success_criteria": [],
                "abort_criteria": [],
            },
            "contract": {"eff_add": [], "eff_del": []},
        }
    }
    bank_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    out = _hydrate_records_from_bank(bank_path, default_domain="gymv")
    assert len(out) == 1
    rec = next(iter(out.values()))
    assert rec.status == SkillStatus.PROVISIONAL
    assert "gymv" in rec.feasible_domains


def test_hydrate_records_handles_missing_bank(tmp_path: Path):
    out = _hydrate_records_from_bank(
        tmp_path / "nope.jsonl", default_domain="gymv",
    )
    assert out == {}


def test_hydrate_records_skips_malformed_lines(tmp_path: Path):
    bank_path = tmp_path / "skill_bank.jsonl"
    bank_path.write_text("not-json\n", encoding="utf-8")
    out = _hydrate_records_from_bank(bank_path, default_domain="gymv")
    assert out == {}


# ---------------------------------------------------------------------------
# for_game
# ---------------------------------------------------------------------------


def test_for_game_with_missing_bank_returns_empty_hook(tmp_path: Path):
    h = SkillHarnessHook.for_game(
        game="tetris", bank_path=tmp_path / "absent.jsonl",
    )
    assert h.n_records() == 0
    # Filtering with an empty cache is a graceful no-op.
    out, diag = h.filter_candidates([_cand("anything")], _state())
    assert len(out) == 1
    assert diag["n_unknown"] == 1


def test_for_game_with_real_bank_admits_candidates(tmp_path: Path):
    bank = tmp_path / "skill_bank.jsonl"
    payload = {
        "skill": {
            "skill_id": "STRAT/STACK",
            "name": "stack",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["stack tiles"]},
            "contract": {"eff_add": [], "eff_del": []},
        }
    }
    bank.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    h = SkillHarnessHook.for_game(game="twenty_forty_eight", bank_path=bank)
    assert h.n_records() == 1

    sid = next(iter(h._records))
    out, diag = h.filter_candidates([_cand(sid)], _state(task="twenty_forty_eight"))
    assert len(out) == 1
    assert diag["n_admitted"] == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_aggregate_across_calls():
    rec_ok = _mk_record(skill_id="ok")
    rec_bad = _mk_record(skill_id="bad")
    object.__setattr__(rec_bad, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"ok": rec_ok, "bad": rec_bad})

    h.filter_candidates([_cand("ok"), _cand("bad")], _state())
    h.validate_choice("ok", _state())
    h.validate_choice("bad", _state())  # bad is DRAFT — but validate_invocation
                                          # checks adapter/binding/preconditions,
                                          # not status, so admits
    s = h.stats.to_json()
    assert s["n_candidates_in"] == 2
    assert s["n_candidates_admitted"] == 1
    assert s["n_candidates_rejected"] == 1
    assert "status_not_runnable" in s["veto_class_distribution"]
    assert s["n_validate_ok"] >= 1


# ---------------------------------------------------------------------------
# Layer C — predicate translator splice
# ---------------------------------------------------------------------------
#
# Asserts the trainer hook's filter_candidates picks up
# `harness.predicate_translator.translate_skill_contract` and:
#
#   * Diagonal cells (source == target) skip the rewrite (fast path).
#   * Cross-domain cells (e.g. source=visual_reasoning, target=gymv)
#     rewrite contract.effects_{add,del} when the (src, tgt) pair has a
#     registered translation table entry, and bump
#     n_predicate_translations_applied.
#   * The cached SkillRecord is NEVER mutated -- the translator returns
#     a deep copy, so the bank-hydrated cache is preserved across calls.
#   * A translator crash degrades to identity (the original record),
#     bumps n_predicate_translations_failed, and keeps the eligibility
#     filter running.


def test_predicate_translation_diagonal_is_noop():
    """Diagonal cell (source_domains == target == 'gymv'): no rewrite."""
    contract = SkillContract(
        effects_add=["cumulative_reward_increased"],
        effects_del=[],
    )
    rec = _mk_record(
        skill_id="diag",
        contract=contract,
        source_domains=["gymv"],
    )
    h = SkillHarnessHook(records={"diag": rec})

    out, diag = h.filter_candidates([_cand("diag")], _state())

    assert diag["n_predicate_translations_applied"] == 0
    assert diag["n_predicate_translations_failed"] == 0
    # Cached record's contract is untouched.
    assert h._records["diag"].contract.effects_add == ["cumulative_reward_increased"]
    assert h.stats.n_predicate_translations_applied == 0


def test_predicate_translation_cross_domain_rewrites_contract():
    """Cross-domain cell (source=gymv, target=visual_reasoning): the
    PREDICATE_TRANSLATIONS table fires and the diagnostic counter
    increments. The cached SkillRecord stays unchanged.

    This exercises the forward direction of the translator's table
    (gymv-mined skills retargeted onto a non-gymv adapter). The
    eligibility filter will veto on ``no_adapter`` -- we don't have a
    visual_reasoning adapter registered in the default hook -- but
    Layer C runs *before* eligibility, so the translation counter
    still bumps. That is exactly the behaviour the dashboard layer
    (Layer D) needs to surface 'translator was exercised, executor
    veto came after'."""
    from harness.predicate_translator import PREDICATE_TRANSLATIONS

    cell = PREDICATE_TRANSLATIONS.get(("gymv", "visual_reasoning"), {})
    if not cell:
        pytest.skip("(gymv, visual_reasoning) cell not registered")

    # Find a source predicate whose target list is non-trivial
    # (i.e. != [src_pred]). At least one such mapping exists in the
    # canonical table -- 'cumulative_reward_increased' fans out to
    # ['answer_emitted', 'answer_matches_gold'] in this cell.
    src_pred = next(
        (sp for sp, tgts in cell.items() if list(tgts) != [sp]),
        None,
    )
    assert src_pred is not None, (
        "(gymv, visual_reasoning) cell has only identity mappings -- "
        "test needs a non-trivial rewrite to assert the counter bumps"
    )

    contract = SkillContract(
        effects_add=[src_pred],
        effects_del=[],
    )
    # Note: feasible_domains kept = [gymv] because backdoor-setting
    # `feasible_domains` would also need to bypass the validator,
    # and the translator runs off `source_domains` not feasible_domains.
    rec = _mk_record(
        skill_id="xd",
        domain="gymv",
        source_domains=["gymv"],
        contract=contract,
    )
    # Override source_domains to the mined-foundry value.
    object.__setattr__(rec, "source_domains", ["gymv"])
    h = SkillHarnessHook(records={"xd": rec})

    # Probe with a state whose domain is the *target* (visual_reasoning).
    state = StateSchema(domain="visual_reasoning", task="xd_task", evidence=[])
    out, diag = h.filter_candidates([_cand("xd")], state)

    # The cached record's contract is preserved -- the translator
    # returned a deep copy, never mutated the bank entry.
    assert h._records["xd"].contract.effects_add == [src_pred]
    # The translation fired and the counter incremented.
    assert diag["n_predicate_translations_applied"] == 1
    assert diag["n_predicate_translations_failed"] == 0
    assert h.stats.n_predicate_translations_applied == 1


def test_predicate_translation_failure_falls_back_to_identity(monkeypatch):
    """A buggy translator must not break the trainer rollout."""
    from trainer.coevolution import _harness_hook as hh_mod

    def _boom(skill, *, source, target):
        raise RuntimeError("synthetic translator bug")

    monkeypatch.setattr(hh_mod, "translate_skill_contract", _boom)

    rec = _mk_record(
        skill_id="bug",
        domain="gymv",
        source_domains=["visual_reasoning"],   # forces the translator path
        contract=SkillContract(effects_add=["any_predicate"]),
    )
    h = SkillHarnessHook(records={"bug": rec})

    out, diag = h.filter_candidates([_cand("bug")], _state())

    # Filter still ran on the original record (gymv adapter accepts it).
    assert diag["n_predicate_translations_applied"] == 0
    assert diag["n_predicate_translations_failed"] == 1
    assert h.stats.n_predicate_translations_failed == 1
    # The hook didn't crash and the rollout sees a sensible result.
    assert isinstance(out, list)


def test_predicate_translation_diag_includes_counters_on_filter_error(monkeypatch):
    """Even when the eligibility filter itself raises, the per-step
    diagnostic surfaces the translation counters (so dashboards can
    still attribute the cell)."""
    from trainer.coevolution import _harness_hook as hh_mod

    rec = _mk_record(
        skill_id="ok",
        domain="gymv",
        source_domains=["gymv"],   # diagonal -- translation skipped
        contract=SkillContract(),
    )
    h = SkillHarnessHook(records={"ok": rec})

    # Patch the eligibility filter to crash.
    def _crash(*args, **kwargs):
        raise RuntimeError("synthetic eligibility crash")

    monkeypatch.setattr(
        h._harness._eligibility, "filter_with_rejections", _crash,
    )

    out, diag = h.filter_candidates([_cand("ok")], _state())
    # Pass-through degrades but the counters are still stamped.
    assert "n_predicate_translations_applied" in diag
    assert "n_predicate_translations_failed" in diag
    assert "harness_error" in diag
    assert len(out) == 1


def test_predicate_translation_counters_in_step_stats_to_json():
    """to_json() must surface the new counters so the orchestrator's
    `experiences[].harness` payload exposes them to wandb / TB."""
    h = SkillHarnessHook(records={})
    j = h.stats.to_json()
    assert j["n_predicate_translations_applied"] == 0
    assert j["n_predicate_translations_failed"] == 0


# ---------------------------------------------------------------------------
# Refinement B — adaptation score on admitted candidates
# ---------------------------------------------------------------------------
#
# Asserts every admitted candidate carries a numeric
# ``_harness_adaptation_score`` in [0,1] composed from the task-axis
# match, the adapter native-vs-bridged signal, and the predicate
# translation provenance. The diagnostic dict additionally surfaces the
# adaptation_score min/max/mean so the orchestrator can log moments to
# wandb / TB without re-iterating the candidate list.
#
# Vetoed candidates do NOT receive the score (they're absent from the
# filtered list); unknown-to-cache candidates pass through unchanged
# and explicitly do not get the score (we have no opinion).


def test_adaptation_score_diagonal_runnable_is_one():
    """Same-domain skill mined from gymv, no rewrite needed, native
    adapter, agnostic task — score is dominated by task-match's
    'agnostic' weight (0.60) but adapter and translation are both
    1.0; expect ((0.60 + 1.0 + 1.0) / 3) = 0.867."""
    rec = _mk_record(
        skill_id="diag",
        domain="gymv",
        source_domains=["gymv"],
    )
    h = SkillHarnessHook(records={"diag": rec})
    out, diag = h.filter_candidates([_cand("diag")], _state())
    assert len(out) == 1
    score = out[0]["_harness_adaptation_score"]
    assert 0.85 < score < 0.90
    breakdown = out[0]["_harness_adaptation_breakdown"]
    assert breakdown["adapter"] == 1.0
    assert breakdown["translation"] == 1.0
    assert breakdown["translation_status"] == "diagonal"


def test_adaptation_score_same_task_higher_than_agnostic():
    """When the skill names the current task in feasible_tasks the
    eligibility filter stamps task_match='same_task' -- which weights
    higher than 'agnostic' -- so the headline score should rise."""
    rec_agnostic = _mk_record(skill_id="agn", domain="gymv", source_domains=["gymv"])
    rec_same = _mk_record(
        skill_id="same",
        domain="gymv",
        source_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
    )
    h = SkillHarnessHook(records={"agn": rec_agnostic, "same": rec_same})
    out, _ = h.filter_candidates(
        [_cand("agn"), _cand("same")],
        _state(task="twenty_forty_eight"),
    )
    by_id = {c["skill_id"]: c for c in out}
    assert by_id["same"]["_harness_adaptation_score"] > by_id["agn"]["_harness_adaptation_score"]


def test_adaptation_score_dropped_when_skill_vetoed():
    """A vetoed skill doesn't appear in the filtered list at all, so
    the prompt never sees an adaptation_score for it -- the LLM only
    picks among admitted candidates."""
    rec = _mk_record(skill_id="bad", domain="gymv", source_domains=["gymv"])
    object.__setattr__(rec, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"bad": rec})
    out, _ = h.filter_candidates([_cand("bad")], _state())
    assert out == []


def test_adaptation_score_omitted_for_unknown_skill():
    """Skills the cache doesn't know are passed through unchanged --
    we have no opinion, so no score."""
    h = SkillHarnessHook(records={})
    out, _ = h.filter_candidates([_cand("unknown")], _state())
    assert len(out) == 1
    assert "_harness_adaptation_score" not in out[0]
    assert "_harness_adaptation_breakdown" not in out[0]


def test_adaptation_score_summary_in_diag():
    """The diagnostic dict surfaces min/max/mean of the per-candidate
    adaptation_scores so the orchestrator can log moments to
    wandb / TB without scanning the filtered list."""
    rec_a = _mk_record(skill_id="a", domain="gymv", source_domains=["gymv"])
    rec_b = _mk_record(
        skill_id="b",
        domain="gymv",
        source_domains=["gymv"],
        feasible_tasks=["twenty_forty_eight"],
    )
    h = SkillHarnessHook(records={"a": rec_a, "b": rec_b})
    _, diag = h.filter_candidates(
        [_cand("a"), _cand("b")],
        _state(task="twenty_forty_eight"),
    )
    assert diag["adaptation_score_min"] is not None
    assert diag["adaptation_score_max"] is not None
    assert diag["adaptation_score_mean"] is not None
    assert (
        diag["adaptation_score_min"]
        <= diag["adaptation_score_mean"]
        <= diag["adaptation_score_max"]
    )


def test_adaptation_score_summary_none_when_no_admitted():
    """When every candidate is vetoed the summary fields are None
    (rather than e.g. ``0.0`` which would imply 'admitted but bad')."""
    rec = _mk_record(skill_id="bad", domain="gymv", source_domains=["gymv"])
    object.__setattr__(rec, "status", SkillStatus.DRAFT)
    h = SkillHarnessHook(records={"bad": rec})
    _, diag = h.filter_candidates([_cand("bad")], _state())
    assert diag["adaptation_score_min"] is None
    assert diag["adaptation_score_max"] is None
    assert diag["adaptation_score_mean"] is None


def test_adaptation_score_translation_failure_lowers_score():
    """Translator crash → identity fallback. The skill is still
    admitted (gymv adapter handles the original record) but the
    translation slot drops to the failed-fallback weight, pulling
    the headline score below the diagonal baseline."""
    from trainer.coevolution import _harness_hook as hh_mod

    def _boom(skill, *, source, target):
        raise RuntimeError("synthetic translator bug")

    rec_clean = _mk_record(
        skill_id="clean",
        domain="gymv",
        source_domains=["gymv"],   # diagonal
    )
    rec_failed = _mk_record(
        skill_id="failed",
        domain="gymv",
        source_domains=["visual_reasoning"],   # forces translator path
        contract=SkillContract(effects_add=["any_predicate"]),
    )
    h = SkillHarnessHook(records={"clean": rec_clean, "failed": rec_failed})

    import pytest as _pytest
    monkeypatch = _pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(hh_mod, "translate_skill_contract", _boom)
        out, _ = h.filter_candidates(
            [_cand("clean"), _cand("failed")], _state(),
        )
    finally:
        monkeypatch.undo()

    by_id = {c["skill_id"]: c for c in out}
    assert by_id["failed"]["_harness_adaptation_breakdown"]["translation_status"] == "failed"
    assert by_id["clean"]["_harness_adaptation_score"] > by_id["failed"]["_harness_adaptation_score"]


def test_adaptation_score_in_range():
    """All emitted scores must lie in [0, 1] regardless of input."""
    rec = _mk_record(skill_id="a", domain="gymv", source_domains=["gymv"])
    h = SkillHarnessHook(records={"a": rec})
    out, _ = h.filter_candidates([_cand("a")], _state())
    score = out[0]["_harness_adaptation_score"]
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Prompt formatter — surfaces the score and the deboost rate
# ---------------------------------------------------------------------------
#
# Two parallel `_format_candidates_for_selection` definitions:
# * scripts/qwen3_decision_agent.py — standalone agent
# * trainer/coevolution/episode_runner.py — trainer-side mirror
# Both must surface the harness signals when present, and degrade
# silently when they're absent (callers without the harness path
# should still be able to render candidate menus).


def test_prompt_formatter_includes_adaptation_when_present():
    from scripts.qwen3_decision_agent import _format_candidates_for_selection

    out = _format_candidates_for_selection(
        [{
            "skill_id": "x", "skill_name": "x",
            "execution_hint": "do x", "protocol": {},
            "confidence": 0.5,
            "_harness_adaptation_score": 0.83,
        }],
    )
    assert "Adaptation: 0.83" in out


def test_prompt_formatter_omits_adaptation_when_absent():
    from scripts.qwen3_decision_agent import _format_candidates_for_selection

    out = _format_candidates_for_selection(
        [{"skill_id": "x", "skill_name": "x", "execution_hint": "h",
          "protocol": {}, "confidence": 0.5}],
    )
    assert "Adaptation" not in out


def test_prompt_formatter_renders_recent_veto_rate_only_when_meaningful():
    from scripts.qwen3_decision_agent import _format_candidates_for_selection

    # Below the 0.95 threshold ⇒ rendered.
    rendered = _format_candidates_for_selection(
        [{"skill_id": "x", "skill_name": "x", "execution_hint": "h",
          "protocol": {}, "confidence": 0.5,
          "_harness_deboost": 0.5}],
    )
    assert "Recent veto rate" in rendered

    # Above the threshold ⇒ skipped (clean skill).
    rendered_clean = _format_candidates_for_selection(
        [{"skill_id": "x", "skill_name": "x", "execution_hint": "h",
          "protocol": {}, "confidence": 0.5,
          "_harness_deboost": 1.0}],
    )
    assert "Recent veto rate" not in rendered_clean


def test_trainer_prompt_formatter_mirrors_standalone():
    """The trainer's mirror in `episode_runner._format_candidates_for_selection`
    must surface the same signals as the standalone agent."""
    from trainer.coevolution.episode_runner import _format_candidates_for_selection

    rendered = _format_candidates_for_selection(
        [{"skill_id": "x", "skill_name": "x", "execution_hint": "h",
          "protocol": {}, "confidence": 0.5,
          "_harness_adaptation_score": 0.42,
          "_harness_deboost": 0.6}],
    )
    assert "Adaptation: 0.42" in rendered
    assert "Recent veto rate" in rendered
