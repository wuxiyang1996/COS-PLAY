"""Unit tests for ``trainer.coevolution._crafter_hook``.

Covers:

* helpers: ``corpus_for_game`` mapping; ``_synthesize_failures`` for both
  trainer signals (OUTCOME_FAILURE, NO_SKILL_BOUND); ``_to_offline_row``
  shape for each ``BankMutationProposal`` subclass; bank-seeding from
  legacy JSONL.
* end-to-end: ``run_crafter_step`` over a synthetic ``EpisodeResult``
  fixture against a per-game legacy ``skill_bank.jsonl`` written from
  the real Airstriker bank. Confirms the JSONL output is parseable by
  ``decide_promotion_gpt54.py::_OfflineProposal.from_json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

from data_structure.extensions.bank_mutation_proposal import (
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.skill_record import SkillContract
from trainer.coevolution._crafter_hook import (
    ALL_FIVE_DOMAINS,
    DEFAULT_OUTCOME_FAILURE_THRESHOLD,
    _record_from_bank_entry,
    _seed_repo_from_legacy_jsonl,
    _synthesize_failures,
    _to_offline_row,
    corpus_for_game,
    run_crafter_step,
)


# ---------------------------------------------------------------------------
# Lightweight EpisodeResult fixture (avoids importing the heavy
# trainer.coevolution.episode_runner module which pulls in env_wrappers).
# ---------------------------------------------------------------------------


@dataclass
class FakeEpisodeResult:
    game: str
    episode_id: str
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    experiences: List[Dict[str, Any]] = field(default_factory=list)


def _exp(step: int, *, reward: float = 0.0, skill_id: Any = None) -> Dict[str, Any]:
    return {
        "step": step,
        "state": f"s{step}",
        "action": "0",
        "reward": reward,
        "raw_env_reward": reward,
        "next_state": f"s{step + 1}",
        "done": False,
        "intention": "",
        "summary_state": "",
        "skill_id": skill_id,
    }


# ---------------------------------------------------------------------------
# corpus_for_game
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game,expected", [
    ("Temporal_Airstriker-v0", "gym_v"),
    ("Temporal_Strider-v0", "gym_v"),
    ("Temporal_MortalKombatII-v0", "gym_v"),
    ("tetris", "env_wrappers"),
    ("twenty_forty_eight", "env_wrappers"),
    ("candy_crush", "env_wrappers"),
    ("super_mario", "env_wrappers"),
    ("Avalon", "env_wrappers"),  # unknown is not gym_v unless prefixed
])
def test_corpus_for_game(game, expected):
    assert corpus_for_game(game) == expected


# ---------------------------------------------------------------------------
# _synthesize_failures
# ---------------------------------------------------------------------------


def test_synthesize_outcome_failure_fires_when_total_reward_zero():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=3, total_reward=0.0,
        experiences=[_exp(0), _exp(1), _exp(2)],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=DEFAULT_OUTCOME_FAILURE_THRESHOLD,
        max_failures=8, bank_was_available=False,
    )
    assert len(out) == 1
    t = out[0]
    assert t.failure_class == "INVARIANT_VIOLATION"
    assert t.failed_step_index == 2
    assert t.extra["synthesis_signal"] == "OUTCOME_FAILURE"
    assert t.extra["total_reward"] == 0.0
    assert t.skill_id == ""        # nothing was bound


def test_synthesize_outcome_failure_picks_most_frequent_bound_skill():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=4, total_reward=0.0,
        experiences=[
            _exp(0, skill_id="ATTACK"),
            _exp(1, skill_id="ATTACK"),
            _exp(2, skill_id="DEFEND"),
            _exp(3, skill_id="ATTACK"),
        ],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=True,
    )
    outcome = next(o for o in out if o.extra["synthesis_signal"] == "OUTCOME_FAILURE")
    assert outcome.skill_id == "ATTACK"


def test_synthesize_skips_outcome_failure_above_threshold():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=2, total_reward=5.0,
        experiences=[_exp(0), _exp(1)],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=False,
    )
    assert out == []


def test_synthesize_no_skill_bound_only_when_bank_available():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=3, total_reward=10.0,
        experiences=[
            _exp(0, skill_id="ATTACK"),
            _exp(1, skill_id=None),
            _exp(2, skill_id=""),
        ],
    )
    # bank not available: no NO_SKILL_BOUND at all
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=False,
    )
    assert out == []

    # bank available: 2 NO_SKILL_BOUND traces (steps 1, 2)
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=True,
    )
    assert len(out) == 2
    for t in out:
        assert t.failure_class == "MISSING_ADAPTER"
        assert t.extra["synthesis_signal"] == "NO_SKILL_BOUND"


def test_synthesize_caps_at_max_failures():
    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=20, total_reward=0.0,                 # 1 OUTCOME_FAILURE
        experiences=[_exp(i, skill_id=None) for i in range(20)],   # 20 NO_SKILL_BOUND
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=5, bank_was_available=True,
    )
    assert len(out) == 5
    # OUTCOME_FAILURE comes first (severity ordering).
    assert out[0].extra["synthesis_signal"] == "OUTCOME_FAILURE"


def test_synthesize_records_truncated_flag():
    ep = FakeEpisodeResult(
        game="tetris", episode_id="ep1", steps=10,
        total_reward=-1.0, truncated=True,
        experiences=[_exp(i) for i in range(10)],
    )
    out = _synthesize_failures(
        episode=ep, domain="gymv",
        outcome_failure_threshold=0.0, max_failures=8, bank_was_available=False,
    )
    assert len(out) == 1
    assert out[0].extra["truncated"] is True
    assert "(truncated)" in out[0].abort_reason


# ---------------------------------------------------------------------------
# _to_offline_row — every BankMutationProposal subclass
# ---------------------------------------------------------------------------


def test_to_offline_row_patch():
    p = PatchProposal(
        proposal_id="prop-1", rationale="test patch",
        target_domains=list(ALL_FIVE_DOMAINS),
        base_skill_id="COMMIT__ATTACK",
        recovery_strategy="precondition",
        patched_contract=SkillContract(expected_evidence_roles=["COMMIT"]),
        seed_failure_ids=["fail-a", "fail-b"],
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_id"] == "prop-1"
    assert row["proposal_kind"] == "patch"
    assert row["proposer"] == "reflector"
    assert row["target_skill_id"] == "COMMIT__ATTACK"
    assert row["patch_kind"] == "precondition"
    assert row["evidence_role"] == "COMMIT"
    assert row["seed_failure_ids"] == ["fail-a", "fail-b"]
    assert sorted(row["adapter_plan"].keys()) == sorted(ALL_FIVE_DOMAINS)
    assert row["adapter_plan"]["gymv"]["strategy"] == "reuse"
    assert row["adapter_plan"]["browser"]["strategy"] == "synthesize_from_slot_ontology"


def test_to_offline_row_retire():
    p = RetireProposal(
        proposal_id="prop-2", rationale="evidence-starved",
        target_domains=list(ALL_FIVE_DOMAINS),
        target_skill_id="STALE__SKILL", reason="evidence-starved",
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "retire"
    assert row["proposer"] == "reflector"
    assert row["target_skill_id"] == "STALE__SKILL"
    assert row["retire_reason"] == "evidence-starved"


def test_to_offline_row_compose():
    p = ComposeProposal(
        proposal_id="prop-3", rationale="hot pair",
        target_domains=list(ALL_FIVE_DOMAINS),
        component_skill_ids=["A", "B", "C"],
        contract=SkillContract(expected_evidence_roles=["GATHER"]),
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "compose"
    assert row["proposer"] == "composer"
    assert row["components"] == ["A", "B", "C"]
    assert row["compose_op"] == "sequence"
    assert row["evidence_role"] == "GATHER"


def test_to_offline_row_generalize():
    p = GeneralizeProposal(
        proposal_id="prop-4", rationale="single-domain mature",
        target_domains=list(ALL_FIVE_DOMAINS),
        base_skill_id="MATURE__SKILL",
        source_domain="gymv", target_domain="browser",
        slot_remap={"tile": "node"},
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "transfer"
    assert row["proposer"] == "generalizer"
    assert row["source_skill_id"] == "MATURE__SKILL"
    assert row["source_domain"] == "gymv"
    assert row["new_adapter_per_target"] == {"browser": True}
    assert row["slot_remap_per_target"] == {"browser": {"tile": "node"}}


def test_to_offline_row_hypothesis():
    p = HypothesisProposal(
        proposal_id="prop-5", rationale="gap",
        target_domains=list(ALL_FIVE_DOMAINS),
        name="hypothesis-skill",
        contract=SkillContract(expected_evidence_roles=["REASON"]),
    )
    row = _to_offline_row(p, domain="gymv")
    assert row["proposal_kind"] == "hypothesize"
    assert row["proposer"] == "hypothesizer"
    assert row["new_skill_name"] == "hypothesis-skill"
    assert row["evidence_role"] == "REASON"


# ---------------------------------------------------------------------------
# Bank seeding
# ---------------------------------------------------------------------------


def test_record_from_bank_entry_returns_none_for_malformed():
    assert _record_from_bank_entry({}, "gymv") is None
    assert _record_from_bank_entry({"skill": "string"}, "gymv") is None
    assert _record_from_bank_entry({"skill": {"name": "no-id"}}, "gymv") is None


def test_record_from_bank_entry_round_trip(tmp_path: Path):
    entry = {
        "skill": {
            "skill_id": "COMMIT/ATTACK",
            "name": "Attack",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {
                "preconditions": ["enemy_visible"],
                "steps": ["press fire", "wait"],
                "success_criteria": ["volley_complete"],
                "abort_criteria": ["player_dead"],
            },
            "contract": {
                "eff_add": ["volley_complete"],
                "eff_del": [],
            },
        },
        "report": {},
    }
    rec = _record_from_bank_entry(entry, "gymv")
    assert rec is not None
    # `/` was rewritten to `__` for filesystem safety.
    assert rec.skill_id == "COMMIT__ATTACK"
    assert rec.feasible_domains == ["gymv"]
    assert len(rec.protocol) == 2
    # NL strings were lifted into typed hops.
    assert rec.protocol[0]["notes"] == "press fire"
    assert rec.contract.preconditions == ["enemy_visible"]
    assert rec.contract.effects_add == ["volley_complete"]
    assert rec.contract.expected_evidence_roles == ["COMMIT"]


def test_seed_repo_from_legacy_jsonl_seeds_only_valid_lines(tmp_path: Path):
    from skill_bank.lifecycle import SkillLifecycleManager
    from skill_bank.repository import SkillRepository
    from skill_bank.stores import SkillStore, StoreName

    bank_path = tmp_path / "skill_bank.jsonl"
    valid_a = {
        "skill": {
            "skill_id": "A",
            "name": "A",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a step"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }
    valid_b = {
        "skill": {
            "skill_id": "B",
            "name": "B",
            "evidence_role": "GATHER",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["b step"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }
    with bank_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(valid_a) + "\n")
        f.write("not-json\n")
        f.write("\n")
        f.write(json.dumps(valid_b) + "\n")
        f.write(json.dumps({"skill": {}}) + "\n")            # malformed: no skill_id

    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)

    n = _seed_repo_from_legacy_jsonl(
        lifecycle=lifecycle, bank_path=bank_path, default_domain="gymv",
    )
    assert n == 2
    assert {r.skill_id for r in repo.candidates()} == {"A", "B"}


def test_seed_repo_returns_zero_for_missing_file(tmp_path: Path):
    from skill_bank.lifecycle import SkillLifecycleManager
    from skill_bank.repository import SkillRepository
    from skill_bank.stores import SkillStore, StoreName

    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(tmp_path / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_path / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(tmp_path / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_path / "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)
    n = _seed_repo_from_legacy_jsonl(
        lifecycle=lifecycle, bank_path=tmp_path / "nope.jsonl",
        default_domain="gymv",
    )
    assert n == 0


# ---------------------------------------------------------------------------
# End-to-end run_crafter_step
# ---------------------------------------------------------------------------


def _real_airstriker_bank() -> Path:
    """Locate the live cold-start Airstriker bank if present."""
    root = (
        Path(__file__).resolve().parent.parent
        / "labeling" / "skill_bank_out"
    )
    if not root.is_dir():
        return Path()
    pair = root / "run_20260430_030637" / "gym_v" / "Temporal_Airstriker-v0"
    bank_path = pair / "skill_bank.jsonl"
    return bank_path if bank_path.is_file() else Path()


def test_run_crafter_step_no_signal_short_circuits(tmp_path: Path):
    """Healthy episodes (positive reward, all skills bound) produce zero
    proposals and an empty proposals.jsonl file."""
    bank_path = tmp_path / "bank" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({
        "skill": {
            "skill_id": "S1",
            "name": "S1",
            "evidence_role": "COMMIT",
            "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }) + "\n", encoding="utf-8")

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep1",
        steps=2, total_reward=10.0,
        experiences=[_exp(0, skill_id="S1"), _exp(1, skill_id="S1")],
    )

    report = run_crafter_step(
        step=0,
        run_dir=tmp_path,
        rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
    )
    assert report.n_episodes_reflected == 1
    assert report.n_failure_traces == 0
    assert report.n_proposals == 0
    out = (tmp_path / "crafter_proposals_out" / "step_0000"
           / "gym_v" / "Temporal_Airstriker-v0" / "proposals.jsonl")
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == ""


def test_run_crafter_step_outcome_failure_emits_proposals(tmp_path: Path):
    """A failed episode against a real seeded bank should produce ≥1
    PatchProposal that round-trips through the offline schema parser."""
    real_bank = _real_airstriker_bank()
    if not real_bank.is_file():
        pytest.skip("no real Airstriker bank on disk")

    # Copy bank into tmp (test stays read-only against the input).
    bank_path = tmp_path / "bank" / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(real_bank.read_text(encoding="utf-8"), encoding="utf-8")

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-real",
        steps=4, total_reward=0.0,                       # OUTCOME_FAILURE
        experiences=[
            _exp(0, skill_id="COMMIT__ATTACK"),
            _exp(1, skill_id="COMMIT__ATTACK"),
            _exp(2, skill_id=None),                      # NO_SKILL_BOUND
            _exp(3, skill_id="COMMIT__ATTACK"),
        ],
    )

    report = run_crafter_step(
        step=1,
        run_dir=tmp_path,
        rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=True,
        # Lane-(b) opt-in: this test asserts a PatchProposal lands;
        # under the lane-(a) default (T1.3a) the dispatch falls through
        # to the Hypothesizer instead. Both paths are exercised: see
        # test_run_crafter_step_lane_a_default_routes_to_hypothesizer.
        enable_protocol_patching=True,
    )
    assert report.n_episodes_reflected == 1
    assert report.n_failure_traces >= 1                  # at least OUTCOME_FAILURE
    assert "Temporal_Airstriker-v0" in report.proposals_per_game
    out = report.proposals_jsonl_paths["Temporal_Airstriker-v0"]
    assert out.is_file()

    # Validate every emitted row parses through the consumer schema.
    from labeling_supplement.decide_promotion_gpt54 import _OfflineProposal
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == report.n_proposals
    for row in rows:
        op = _OfflineProposal.from_json(row)
        assert op.proposal_id
        assert op.proposal_kind in {"patch", "retire", "compose", "transfer", "hypothesize"}
        assert op.target_domains == list(ALL_FIVE_DOMAINS)


def test_run_crafter_step_writes_step_summary(tmp_path: Path):
    bank_path = tmp_path / "bank" / "skill_bank.jsonl"
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({
        "skill": {
            "skill_id": "S1", "name": "S1",
            "evidence_role": "COMMIT", "applicable_domains": ["gymv"],
            "protocol": {"steps": ["a"]},
            "contract": {"eff_add": [], "eff_del": []},
        },
        "report": {},
    }) + "\n", encoding="utf-8")

    ep = FakeEpisodeResult(
        game="Temporal_Airstriker-v0", episode_id="ep-s",
        steps=1, total_reward=0.0, experiences=[_exp(0)],
    )
    run_crafter_step(
        step=42, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={"Temporal_Airstriker-v0": bank_path},
        bank_was_available=False,
    )
    summary_path = (
        tmp_path / "crafter_proposals_out" / "step_0042" / "_step_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["step"] == 42
    assert summary["n_games"] == 1
    assert summary["n_episodes_reflected"] == 1
    assert "params" in summary


def test_run_crafter_step_skips_unknown_games_gracefully(tmp_path: Path):
    """Episodes for a game not in legacy_bank_paths are skipped, not crashed."""
    ep = FakeEpisodeResult(
        game="unknown_game", episode_id="ep-x",
        steps=1, total_reward=-1.0, experiences=[_exp(0)],
    )
    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[ep],
        legacy_bank_paths={},                         # no banks
        bank_was_available=False,
    )
    assert report.n_proposals == 0
    assert report.n_episodes_reflected == 0


def test_run_crafter_step_skips_sentinel_results(tmp_path: Path):
    sentinel = FakeEpisodeResult(game="__SENTINEL__", episode_id="__SENTINEL__")
    bank_path = tmp_path / "skill_bank.jsonl"
    bank_path.write_text("", encoding="utf-8")

    report = run_crafter_step(
        step=0, run_dir=tmp_path, rollout_results=[sentinel],
        legacy_bank_paths={"__SENTINEL__": bank_path},
        bank_was_available=False,
    )
    assert report.n_episodes_reflected == 0
    assert report.n_proposals == 0
