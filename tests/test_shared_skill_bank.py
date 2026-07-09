"""Regression tests for shared-bank mode + cross-game translator.

Locks in three classes of invariant:

1. **Backward compat** — pre-shared-mode ``Skill.to_dict()`` /
   ``Skill.from_dict()`` round-trip survives without the four new
   fields. A bank file written by an older trainer must still load
   into a current ``SkillBankMVP``.

2. **Shared-bank manager interface parity** —
   :class:`SharedSkillBankManager` exposes the same external surface as
   :class:`PerGameSkillBankManager` (so ``orchestrator.py`` can branch
   on ``config.bank_mode`` without further changes).

3. **Translator hard invariants** — the function
   :func:`translate_skill_for_target` either returns ``None`` or a
   :class:`Skill` whose
   ``feasible_tasks == [target_game]``,
   ``derived_from`` is non-empty,
   ``confidence_tag == "translated"``,
   and every ``protocol.steps`` token is in the supplied
   ``target_actions`` list.

The translator's LLM call is monkey-patched per test case to make the
suite deterministic and offline-runnable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_agents.skill_bank import translate_for_target as tft
from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.stage3_mvp.schemas import (
    Protocol,
    Skill,
    SkillEffectsContract,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def candy_crush_skill() -> Skill:
    """A representative source skill that *can* be translated to a
    target game whose action vocabulary overlaps."""
    contract = SkillEffectsContract(
        skill_id="cc_match3",
        version=1,
        name="match3_setup",
        description="Set up a 3-in-a-row in the upper-left quadrant",
        eff_add={"cumulative_reward_increased", "entity_value_increased"},
        eff_del=set(),
        eff_event=set(),
    )
    return Skill(
        skill_id="cc_match3",
        version=1,
        name="match3_setup",
        strategic_description="Place pieces to chain a 3-in-a-row",
        tags=["match3", "setup"],
        protocol=Protocol(steps=["UP", "LEFT", "DOWN"]),
        contract=contract,
        feasible_tasks=["candy_crush"],
        verified_tasks=["candy_crush"],
    )


# ---------------------------------------------------------------------------
# 1. Backward compat — legacy Skill round-trip
# ---------------------------------------------------------------------------

def test_legacy_skill_loads_without_new_fields():
    """A skill_bank.jsonl produced before the four new fields landed
    must still load — the new fields must default to safe values."""
    legacy_payload = {
        "skill_id": "old_skill",
        "version": 1,
        "name": "old_skill",
        "strategic_description": "pre-shared-bank record",
        "tags": [],
        "protocol": {"preconditions": [], "steps": ["UP"], "success_criteria": [], "abort_criteria": [], "expected_duration": 1},
        "contract": None,
        "sub_episodes": [],
        "expected_tag_pattern": [],
        "execution_hint": None,
        "protocol_history": [],
        "n_instances": 0,
        "retired": False,
        "created_at": 1.0,
        "updated_at": 1.0,
        # NOTE: deliberately omits feasible_tasks / verified_tasks / derived_from / confidence_tag
    }
    skill = Skill.from_dict(legacy_payload)
    assert skill.skill_id == "old_skill"
    assert skill.feasible_tasks == []
    assert skill.verified_tasks == []
    assert skill.derived_from is None
    assert skill.confidence_tag == "stable"


def test_skill_round_trip_preserves_new_fields(tmp_path: Path):
    """Writing a Skill with the four new fields populated and reading
    it back yields identical values."""
    bank_path = tmp_path / "skill_bank.jsonl"
    contract = SkillEffectsContract(
        skill_id="s1", version=1,
        name="s1", description="d",
        eff_add={"cumulative_reward_increased"},
    )
    s = Skill(
        skill_id="s1",
        name="s1",
        contract=contract,
        feasible_tasks=["candy_crush"],
        verified_tasks=["candy_crush"],
        derived_from="parent_skill",
        confidence_tag="translated",
    )
    bank = SkillBankMVP(str(bank_path))
    bank.add_or_update_skill(s)
    bank.save()

    bank2 = SkillBankMVP(str(bank_path))
    bank2.load(str(bank_path))
    loaded = bank2.get_skill("s1")
    assert loaded is not None
    assert loaded.feasible_tasks == ["candy_crush"]
    assert loaded.verified_tasks == ["candy_crush"]
    assert loaded.derived_from == "parent_skill"
    assert loaded.confidence_tag == "translated"


# ---------------------------------------------------------------------------
# 2. Shared-bank manager interface parity
# ---------------------------------------------------------------------------

def test_shared_bank_manager_exposes_per_game_interface(tmp_path: Path):
    """``SharedSkillBankManager`` must expose every method the
    orchestrator + per-step hooks call on
    :class:`PerGameSkillBankManager`. We're not running the pipeline
    end-to-end here — just confirming the interface is structurally
    compatible so a config swap doesn't break the orchestrator at
    import / construction time.
    """
    from trainer.coevolution.skillbank_pipeline import (
        PerGameSkillBankManager,
        SharedSkillBankManager,
    )

    games = ["candy_crush", "Columns", "tetris"]

    # Both managers must accept the same constructor kwargs.
    for cls in (PerGameSkillBankManager, SharedSkillBankManager):
        mgr = cls(
            games=games,
            bank_dir=str(tmp_path / cls.__name__),
            seed_bank_dir=None,
            unified_role_rollouts=False,
        )

        # External-interface methods (read-only — no LLM calls).
        assert hasattr(mgr, "bank_paths")
        assert hasattr(mgr, "get_banks")
        assert hasattr(mgr, "get_agents")
        assert hasattr(mgr, "process_batch_async")
        assert hasattr(mgr, "finalize_all")
        assert hasattr(mgr, "reload_banks_from_disk")
        assert hasattr(mgr, "reset_for_step")
        assert hasattr(mgr, "total_skills")
        assert hasattr(mgr, "skill_counts")

    shared_mgr = SharedSkillBankManager(
        games=games,
        bank_dir=str(tmp_path / "shared_layout"),
    )
    paths = shared_mgr.bank_paths(simple_only=True)
    assert set(paths.keys()) == set(games)
    # All games must point at exactly the same on-disk file in shared mode.
    distinct = {str(p) for p in paths.values()}
    assert len(distinct) == 1, f"Shared mode must use one file; got {distinct!r}"


def test_shared_bank_manager_rejects_unified_role_rollouts(tmp_path: Path):
    """Avalon/Diplomacy per-side splits aren't supported in shared mode
    (they're tied to the per-game directory layout). The constructor
    must fail fast rather than silently degrade."""
    from trainer.coevolution.skillbank_pipeline import SharedSkillBankManager

    with pytest.raises(ValueError, match="unified_role_rollouts"):
        SharedSkillBankManager(
            games=["avalon", "diplomacy"],
            bank_dir=str(tmp_path / "x"),
            unified_role_rollouts=True,
        )


# ---------------------------------------------------------------------------
# 3. Translator hard invariants
# ---------------------------------------------------------------------------

def _stub_judge_call(monkeypatch, response: str | None) -> None:
    """Replace the translator's judge call with a deterministic stub."""
    def _fake_call(*args, **kwargs):
        return response
    monkeypatch.setattr(tft, "_call_judge", _fake_call)


def test_translator_enforces_feasible_tasks_invariant(monkeypatch, candy_crush_skill):
    """Successful translation produces a Skill whose
    ``feasible_tasks == [target_game]`` (single-element list,
    target-only). The §22 cross-contamination guard rests on this."""
    response = json.dumps({
        "transferable": True,
        "name": "match3_setup",
        "strategic_description": "Set up matched groups in target context.",
        "protocol": {"steps": ["UP", "LEFT", "DOWN"]},
        "contract": {
            "eff_add": ["cumulative_reward_increased"],
            "eff_del": [],
            "eff_event": [],
        },
        "rationale": "Both games reward grouping pieces.",
    })
    _stub_judge_call(monkeypatch, response)

    out = tft.translate_skill_for_target(
        candy_crush_skill,
        source_game="candy_crush",
        target_game="Columns",
        target_actions=["UP", "DOWN", "LEFT", "RIGHT", "ROTATE"],
        judge_model="stub",
    )
    assert out is not None
    assert out.feasible_tasks == ["Columns"]
    assert out.verified_tasks == []
    assert out.derived_from == "cc_match3"
    assert out.confidence_tag == tft.CONFIDENCE_TAG_TRANSLATED
    # Skill ID lineage encoded in the new ID.
    assert out.skill_id == "cc_match3__translated_to__Columns"


def test_translator_drops_invalid_actions(monkeypatch, candy_crush_skill):
    """Steps that aren't in target_actions must be filtered, not retained."""
    response = json.dumps({
        "transferable": True,
        "name": "x", "strategic_description": "x",
        "protocol": {"steps": ["UP", "JUMP_INVALID", "LEFT", "ATTACK_FAKE"]},
        "contract": {"eff_add": [], "eff_del": [], "eff_event": []},
        "rationale": "x",
    })
    _stub_judge_call(monkeypatch, response)

    out = tft.translate_skill_for_target(
        candy_crush_skill,
        source_game="candy_crush",
        target_game="Columns",
        target_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        judge_model="stub",
    )
    assert out is not None
    assert out.protocol.steps == ["UP", "LEFT"]


def test_translator_returns_none_on_judge_rejection(monkeypatch, candy_crush_skill):
    """When the judge marks the skill as non-transferable we return
    ``None`` — never a fake-mapped record."""
    response = json.dumps({
        "transferable": False,
        "rationale": "Match-3 setup has no analogue in beat-em-up combat",
    })
    _stub_judge_call(monkeypatch, response)

    out = tft.translate_skill_for_target(
        candy_crush_skill,
        source_game="candy_crush",
        target_game="gymv_streets_of_rage_2",
        target_actions=["B", "A", "UP", "DOWN", "LEFT", "RIGHT"],
        judge_model="stub",
    )
    assert out is None


def test_translator_returns_none_on_judge_failure(monkeypatch, candy_crush_skill):
    """Transport / parse errors yield ``None`` rather than raising —
    the curriculum boundary script can survive a flaky vLLM."""
    _stub_judge_call(monkeypatch, None)

    out = tft.translate_skill_for_target(
        candy_crush_skill,
        source_game="candy_crush",
        target_game="Columns",
        target_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        judge_model="stub",
    )
    assert out is None


def test_translator_returns_none_on_no_valid_steps(monkeypatch, candy_crush_skill):
    """If the LLM emits only invalid actions we should reject the
    translation — silently retaining an empty protocol would let the
    skill be admitted but never executable."""
    response = json.dumps({
        "transferable": True,
        "name": "x", "strategic_description": "x",
        "protocol": {"steps": ["NONSENSE_1", "NONSENSE_2"]},
        "contract": {"eff_add": [], "eff_del": [], "eff_event": []},
        "rationale": "x",
    })
    _stub_judge_call(monkeypatch, response)

    out = tft.translate_skill_for_target(
        candy_crush_skill,
        source_game="candy_crush",
        target_game="Columns",
        target_actions=["UP", "DOWN"],
        judge_model="stub",
    )
    assert out is None


def test_translator_predicate_filter_keeps_shared_vocabulary(monkeypatch, candy_crush_skill):
    """Predicates in the shared gymv vocabulary survive unchanged;
    unknown ones are dropped (target-grounded vocabulary discipline)."""
    response = json.dumps({
        "transferable": True,
        "name": "x", "strategic_description": "x",
        "protocol": {"steps": ["UP"]},
        "contract": {
            "eff_add": ["cumulative_reward_increased", "candy_specific_predicate"],
            "eff_del": ["entity_value_decreased", "match3_specific"],
            "eff_event": ["phase_transitioned", "ad_hoc_event"],
        },
        "rationale": "x",
    })
    _stub_judge_call(monkeypatch, response)

    out = tft.translate_skill_for_target(
        candy_crush_skill,
        source_game="candy_crush",
        target_game="Columns",
        target_actions=["UP"],
        judge_model="stub",
    )
    assert out is not None
    assert out.contract is not None
    assert out.contract.eff_add == {"cumulative_reward_increased"}
    assert out.contract.eff_del == {"entity_value_decreased"}
    assert out.contract.eff_event == {"phase_transitioned"}


def test_translator_bank_round_trip_seeds_source(monkeypatch, candy_crush_skill, tmp_path: Path):
    """The bank-level helper writes both the source and the translated
    record into the output bank, with disjoint ``feasible_tasks``."""
    response = json.dumps({
        "transferable": True,
        "name": "match3_setup",
        "strategic_description": "Translated to Columns.",
        "protocol": {"steps": ["UP", "LEFT"]},
        "contract": {"eff_add": ["cumulative_reward_increased"], "eff_del": [], "eff_event": []},
        "rationale": "x",
    })
    _stub_judge_call(monkeypatch, response)

    src_path = tmp_path / "source_bank.jsonl"
    src_bank = SkillBankMVP(str(src_path))
    src_bank.add_or_update_skill(candy_crush_skill)
    src_bank.save()

    out_path = tmp_path / "out_bank.jsonl"
    summary = tft.translate_bank_for_target(
        source_bank_path=src_path,
        target_game="Columns",
        target_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        output_bank_path=out_path,
        source_game="candy_crush",
        judge_model="stub",
        seed_with_source=True,
    )
    assert summary["n_source"] == 1
    assert summary["n_translated"] == 1
    assert summary["n_rejected"] == 0

    out_bank = SkillBankMVP(str(out_path))
    out_bank.load(str(out_path))
    skills = {sid: out_bank.get_skill(sid) for sid in out_bank.skill_ids}
    assert "cc_match3" in skills
    assert "cc_match3__translated_to__Columns" in skills

    # Disjoint feasible_tasks — the §22 invariant.
    assert skills["cc_match3"].feasible_tasks == ["candy_crush"]
    assert skills["cc_match3__translated_to__Columns"].feasible_tasks == ["Columns"]
