from __future__ import annotations

from types import SimpleNamespace

from data_structure.experience import Episode, Experience
from skill_agents.pipeline import (
    PipelineConfig,
    SkillBankAgent,
    _heuristic_only_segment_episode,
)
from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.stage3_mvp.run_stage3_mvp import _raw_delta_contract
from skill_agents.stage3_mvp.schemas import (
    Protocol,
    SegmentRecord,
    Skill,
    SkillEffectsContract,
)
from trainer.coevolution.skillbank_pipeline import (
    _candy_crush_ablation_modes,
    _strip_effect_contracts_for_ablation,
)


def test_candy_crush_ablation_env_modes(monkeypatch) -> None:
    monkeypatch.setenv("COSPLAY_CC_EFFECT_CONTRACT_MODE", "raw_delta")
    monkeypatch.setenv("COSPLAY_CC_SEGMENTATION_MODE", "heuristic_only")

    modes = _candy_crush_ablation_modes("candy_crush")

    assert modes == {
        "effect_contract_mode": "raw_delta",
        "segmentation_mode": "heuristic_only",
    }
    assert _candy_crush_ablation_modes("tetris") == {
        "effect_contract_mode": "consensus_verified",
        "segmentation_mode": "learned_decode",
    }


def test_no_effect_contract_mode_strips_existing_contract_but_keeps_protocol() -> None:
    bank = SkillBankMVP()
    skill = Skill(
        skill_id="early:CLEAR",
        protocol=Protocol(steps=["Match candies to clear blockers"]),
        contract=SkillEffectsContract(
            skill_id="early:CLEAR",
            eff_add={"world.score_increased"},
        ),
    )
    bank.add_or_update_skill(skill)
    agent = SimpleNamespace(bank=bank)

    removed = _strip_effect_contracts_for_ablation(agent)

    assert removed == 1
    assert bank.get_skill("early:CLEAR").protocol.steps
    assert bank.get_contract("early:CLEAR") is None


def test_run_contract_learning_none_mode_skips_persisting_effect_contract() -> None:
    agent = SkillBankAgent(
        PipelineConfig(
            game_name="candy_crush",
            effect_contract_mode="none",
            min_instances_per_skill=1,
        )
    )
    agent._all_segments = [
        SegmentRecord(
            seg_id="s0",
            traj_id="t0",
            t_start=0,
            t_end=1,
            skill_label="early:CLEAR",
        )
    ]

    summary = agent.run_contract_learning()

    assert summary.n_segments == 1
    assert agent.bank.get_contract("early:CLEAR") is None


def test_raw_delta_contract_uses_single_segment_delta_without_consensus() -> None:
    first = SegmentRecord(
        seg_id="s0",
        traj_id="t0",
        t_start=0,
        t_end=1,
        skill_label="early:CLEAR",
        eff_add={"world.score_increased"},
        eff_del={"world.blocker_present"},
        eff_event={"event.swap"},
    )
    second = SegmentRecord(
        seg_id="s1",
        traj_id="t1",
        t_start=0,
        t_end=1,
        skill_label="early:CLEAR",
        eff_add={"world.combo_ready"},
    )

    contract = _raw_delta_contract("early:CLEAR", [first, second])

    assert contract.eff_add == {"world.score_increased"}
    assert contract.eff_del == {"world.blocker_present"}
    assert contract.eff_event == {"event.swap"}
    assert contract.n_instances == 1


def test_heuristic_only_segmentation_uses_boundary_candidates(monkeypatch) -> None:
    import skill_agents.boundary_proposal as boundary_proposal
    from skill_agents.boundary_proposal import BoundaryCandidate

    monkeypatch.setattr(
        boundary_proposal,
        "propose_from_episode",
        lambda *args, **kwargs: [BoundaryCandidate(center=2, source="test")],
    )

    experiences = [
        Experience(
            state=f"s{i}",
            action="swap",
            reward=0.0,
            next_state=f"s{i+1}",
            done=False,
            intentions="CLEAR",
        )
        for i in range(5)
    ]
    episode = Episode(experiences=experiences, task="candy_crush")

    result, sub_episodes = _heuristic_only_segment_episode(
        episode,
        skill_names=["early:CLEAR", "__NEW__"],
        env_name="llm",
        game_name="candy_crush",
    )

    assert [seg.start for seg in result.segments] == [0, 2]
    assert [seg.end for seg in result.segments] == [2, 4]
    assert len(sub_episodes) == 2
