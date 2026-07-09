"""Minimal tests for the redesigned quality scoring functions.

Covers:
  - score_sub_episode: episode credit, local progress, seg validity,
    contract validity, novelty gating
  - compute_skill_score: mean segment quality, reuse success, contract pass,
    cross-episode consistency, exploration value
  - Backward compatibility: old callers without new args still work
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from skill_agents_grpo.stage3_mvp.schemas import (
    Skill,
    SkillEffectsContract,
    SubEpisodeRef,
    Protocol,
)
from skill_agents_grpo.quality.sub_episode_evaluator import (
    score_sub_episode,
    score_all_sub_episodes,
    SegmentQualityWeights,
)


def _make_sub_ep(
    cumulative_reward: float = 1.0,
    outcome: str = "success",
    length: int = 5,
    episode_id: str = "ep0",
    quality_score: float = 0.0,
) -> SubEpisodeRef:
    return SubEpisodeRef(
        episode_id=episode_id,
        seg_start=0,
        seg_end=length - 1,
        outcome=outcome,
        cumulative_reward=cumulative_reward,
        quality_score=quality_score,
    )


def _make_skill(
    sub_episodes=None,
    has_contract=True,
    pass_rate=None,
) -> Skill:
    contract = None
    if has_contract:
        contract = SkillEffectsContract(
            skill_id="test_skill",
            eff_add={"p1", "p2"},
            eff_del={"q1"},
        )
    return Skill(
        skill_id="test_skill",
        contract=contract,
        sub_episodes=sub_episodes or [],
        protocol=Protocol(expected_duration=5),
    )


# ── score_sub_episode tests ──────────────────────────────────────────

class TestScoreSubEpisode:
    def test_high_reward_success_scores_high(self):
        skill = _make_skill()
        se = _make_sub_ep(cumulative_reward=10.0, outcome="success", length=5)
        score = score_sub_episode(se, skill, reward_range=(0.0, 10.0), contract_pass_rate=0.9)
        assert score > 0.7, f"High reward + success should score > 0.7, got {score:.3f}"

    def test_zero_reward_failure_scores_low(self):
        skill = _make_skill()
        se = _make_sub_ep(cumulative_reward=0.0, outcome="failure", length=5)
        score = score_sub_episode(se, skill, reward_range=(0.0, 10.0), contract_pass_rate=0.1)
        assert score < 0.35, f"Zero reward + failure should score < 0.35, got {score:.3f}"

    def test_novelty_only_with_validity_gate(self):
        skill = _make_skill()
        se = _make_sub_ep(cumulative_reward=5.0, outcome="success", length=5)
        score_novel = score_sub_episode(
            se, skill, reward_range=(0.0, 10.0),
            contract_pass_rate=0.8, is_novel=True,
        )
        score_not_novel = score_sub_episode(
            se, skill, reward_range=(0.0, 10.0),
            contract_pass_rate=0.8, is_novel=False,
        )
        assert score_novel > score_not_novel, "Novelty should boost score when gates pass"

    def test_novelty_blocked_without_contract(self):
        skill = _make_skill(has_contract=False)
        se = _make_sub_ep(cumulative_reward=5.0, outcome="success", length=5)
        score_novel = score_sub_episode(
            se, skill, reward_range=(0.0, 10.0),
            contract_pass_rate=None, is_novel=True,
        )
        score_not_novel = score_sub_episode(
            se, skill, reward_range=(0.0, 10.0),
            contract_pass_rate=None, is_novel=False,
        )
        assert score_novel == score_not_novel, "Novelty should not boost without contract"

    def test_backward_compat_no_new_args(self):
        """Old callers that don't pass contract_pass_rate/is_novel still work."""
        skill = _make_skill()
        se = _make_sub_ep(cumulative_reward=5.0, outcome="partial", length=5)
        score = score_sub_episode(se, skill, reward_range=(0.0, 10.0))
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        skill = _make_skill()
        se = _make_sub_ep(cumulative_reward=10.0, outcome="success", length=5)
        heavy_reward = SegmentQualityWeights(
            episode_credit=0.80, local_progress=0.05, seg_validity=0.05,
            contract_validity=0.05, novelty_bonus=0.05,
        )
        score = score_sub_episode(
            se, skill, reward_range=(0.0, 10.0),
            contract_pass_rate=0.9, weights=heavy_reward,
        )
        assert score > 0.75, f"With heavy reward weight, score should be > 0.75, got {score:.3f}"


# ── compute_skill_score tests ────────────────────────────────────────

class TestComputeSkillScore:
    def test_no_sub_episodes_returns_default(self):
        skill = _make_skill(sub_episodes=[])
        assert skill.compute_skill_score() == 0.5

    def test_high_quality_high_score(self):
        subs = [
            _make_sub_ep(cumulative_reward=10.0, outcome="success", episode_id=f"ep{i}")
            for i in range(5)
        ]
        for s in subs:
            s.quality_score = 0.9
        skill = _make_skill(sub_episodes=subs)
        score = skill.compute_skill_score(contract_pass_rate=0.85)
        assert score > 0.6, f"High quality should give > 0.6, got {score:.3f}"

    def test_low_quality_low_score(self):
        subs = [
            _make_sub_ep(cumulative_reward=0.0, outcome="failure", episode_id="ep0")
            for _ in range(5)
        ]
        for s in subs:
            s.quality_score = 0.1
        skill = _make_skill(sub_episodes=subs)
        score = skill.compute_skill_score(contract_pass_rate=0.1)
        assert score < 0.25, f"Low quality should give < 0.25, got {score:.3f}"

    def test_usage_frequency_no_longer_dominant(self):
        """Even with high n_selections, low quality should still mean low score."""
        subs = [_make_sub_ep(outcome="failure") for _ in range(5)]
        for s in subs:
            s.quality_score = 0.1
        skill = _make_skill(sub_episodes=subs)
        score = skill.compute_skill_score(
            n_selections=100, n_total_selections=100, n_bank_skills=10,
        )
        assert score < 0.35, f"Usage alone should not inflate score, got {score:.3f}"

    def test_cross_episode_consistency(self):
        subs_consistent = []
        for i in range(4):
            se = _make_sub_ep(cumulative_reward=5.0, outcome="success", episode_id=f"ep{i}")
            se.quality_score = 0.7
            subs_consistent.append(se)
        skill_c = _make_skill(sub_episodes=subs_consistent)
        score_c = skill_c.compute_skill_score(contract_pass_rate=0.8)

        subs_inconsistent = []
        for i, r in enumerate([1.0, 10.0, 0.1, 8.0]):
            se = _make_sub_ep(cumulative_reward=r, outcome="success", episode_id=f"ep{i}")
            se.quality_score = 0.7
            subs_inconsistent.append(se)
        skill_i = _make_skill(sub_episodes=subs_inconsistent)
        score_i = skill_i.compute_skill_score(contract_pass_rate=0.8)

        assert score_c > score_i, (
            f"Consistent skill ({score_c:.3f}) should score higher than "
            f"inconsistent ({score_i:.3f})"
        )

    def test_backward_compat_old_signature(self):
        """Old callers with just n_selections/n_total_selections still work."""
        subs = [_make_sub_ep(outcome="partial") for _ in range(3)]
        for s in subs:
            s.quality_score = 0.5
        skill = _make_skill(sub_episodes=subs)
        score = skill.compute_skill_score()
        assert 0.0 <= score <= 1.0

    def test_exploration_bonus_for_young_skills(self):
        subs = [
            _make_sub_ep(cumulative_reward=5.0, outcome="success", episode_id=f"ep{i}")
            for i in range(3)
        ]
        for s in subs:
            s.quality_score = 0.6
        skill = _make_skill(sub_episodes=subs, has_contract=True)
        score_with_contract = skill.compute_skill_score(contract_pass_rate=0.7)

        skill_no_contract = _make_skill(sub_episodes=subs, has_contract=False)
        score_no_contract = skill_no_contract.compute_skill_score()

        assert score_with_contract > score_no_contract, (
            "Young skill with contract should score higher (exploration bonus)"
        )


# ── GRPO reward integration tests ─────────────────────────────────────

from skill_agents_grpo.grpo.rewards import (
    contract_reward,
    curator_reward,
    segmentation_reward,
)


class TestContractRewardStartEnd:
    """Contract reward should be solid on start/end conditions."""

    def test_good_start_end_coverage(self):
        llm_output = {
            "eff_add": ["has_item", "near_goal"],
            "eff_del": ["has_obstacle", "far_from_goal"],
        }
        score = contract_reward(
            llm_output, "skill_0", [],
            predicates_start={"has_obstacle", "far_from_goal", "alive"},
            predicates_end={"has_item", "near_goal", "alive"},
        )
        assert score > 0.5, f"Good start/end coverage should score > 0.5, got {score:.3f}"

    def test_empty_effects_scores_low(self):
        score = contract_reward(
            {"eff_add": [], "eff_del": []}, "skill_0", [],
            predicates_start={"a"}, predicates_end={"b"},
        )
        assert score < 0.1

    def test_add_only_misses_del_coverage(self):
        llm_output = {"eff_add": ["has_item"], "eff_del": []}
        score_add_only = contract_reward(
            llm_output, "skill_0", [],
            predicates_start={"obstacle"}, predicates_end={"has_item"},
        )
        llm_both = {"eff_add": ["has_item"], "eff_del": ["obstacle"]}
        score_both = contract_reward(
            llm_both, "skill_0", [],
            predicates_start={"obstacle"}, predicates_end={"has_item"},
        )
        assert score_both > score_add_only, "Both add+del should score higher than add only"


class TestCuratorRewardQualityBased:
    """Curator decisions should be based on skill quality scores."""

    def test_approve_high_quality_scores_high(self):
        decisions = {"decisions": [
            {"idx": 0, "verdict": "approve", "reason": "skill_score 0.85, pass_rate 0.90, 10 instances"},
        ]}
        candidates = [{"type": "refine", "skill_id": "s1", "skill_score": 0.85, "pass_rate": 0.9, "n_instances": 10}]
        score = curator_reward(decisions, candidates, None)
        assert score > 0.5, f"Approving high-quality skill should score > 0.5, got {score:.3f}"

    def test_veto_low_quality_scores_high(self):
        decisions = {"decisions": [
            {"idx": 0, "verdict": "veto", "reason": "skill_score 0.15, poor quality"},
        ]}
        candidates = [{"type": "refine", "skill_id": "s1", "skill_score": 0.15, "n_instances": 3}]
        score = curator_reward(decisions, candidates, None)
        assert score > 0.4, f"Vetoing low-quality skill should score > 0.4, got {score:.3f}"

    def test_approve_low_quality_scores_low(self):
        decisions = {"decisions": [
            {"idx": 0, "verdict": "approve", "reason": "some reason"},
        ]}
        candidates = [{"type": "refine", "skill_id": "s1", "skill_score": 0.1, "n_instances": 3}]
        score = curator_reward(decisions, candidates, None)
        # Low quality approve → low quality_align, neutral exploration, weak reason
        assert score < 0.5, f"Approving low-quality skill should score < 0.5, got {score:.3f}"


class TestCuratorExploration:
    """Curator should encourage new skill exploration."""

    def test_approve_new_skill_with_evidence(self):
        decisions = {"decisions": [
            {"idx": 0, "verdict": "approve", "reason": "promising new skill, pass_rate 0.60"},
        ]}
        candidates = [{"type": "materialize", "skill_id": "new_s", "skill_score": 0.5, "pass_rate": 0.6, "n_instances": 3}]
        score = curator_reward(decisions, candidates, None)
        assert score > 0.4, f"Approving new skill with evidence should be rewarded, got {score:.3f}"

    def test_veto_new_skill_penalized(self):
        decisions_approve = {"decisions": [
            {"idx": 0, "verdict": "approve", "reason": "pass_rate 0.60, exploring"},
        ]}
        decisions_veto = {"decisions": [
            {"idx": 0, "verdict": "veto", "reason": "not enough data"},
        ]}
        candidates = [{"type": "promote", "skill_id": "proto_s", "skill_score": 0.5, "pass_rate": 0.6, "n_instances": 3}]
        score_approve = curator_reward(decisions_approve, candidates, None)
        score_veto = curator_reward(decisions_veto, candidates, None)
        assert score_approve > score_veto, (
            f"Approving promising new skill ({score_approve:.3f}) should score higher "
            f"than vetoing it ({score_veto:.3f})"
        )


class TestSegmentationValueMatch:
    """Segmentation should find the most valuable skills."""

    def test_bank_scores_boost_fallback(self):
        from types import SimpleNamespace
        prefs = [
            SimpleNamespace(segment_start=0, segment_end=5, skill_win="existing_a", skill_lose="__NEW__"),
            SimpleNamespace(segment_start=5, segment_end=10, skill_win="existing_a", skill_lose="existing_b"),
        ]
        segments = [(0, 5), (5, 10)]
        skill_names = ["existing_a", "existing_b", "__NEW__"]

        score_no_bank = segmentation_reward(
            prefs, segments, [], [], skill_names,
        )
        score_with_bank = segmentation_reward(
            prefs, segments, [], [], skill_names,
            bank_skill_scores={"existing_a": 0.9, "existing_b": 0.3},
        )
        assert score_with_bank >= score_no_bank * 0.8, (
            "Bank scores should provide useful signal in fallback"
        )

    def test_none_preference_returns_zero(self):
        score = segmentation_reward([], [(0, 5)], [], [], ["a"])
        assert score == 0.0


# ── Run tests ─────────────────────────────────────────────────────────

_ALL_TEST_CLASSES = [
    TestScoreSubEpisode,
    TestComputeSkillScore,
    TestContractRewardStartEnd,
    TestCuratorRewardQualityBased,
    TestCuratorExploration,
    TestSegmentationValueMatch,
]


def _run_tests():
    import traceback
    passed = 0
    failed = 0
    for cls in _ALL_TEST_CLASSES:
        obj = cls()
        for name in dir(obj):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  PASS: {cls.__name__}.{name}")
            except Exception:
                failed += 1
                print(f"  FAIL: {cls.__name__}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_tests()
    sys.exit(0 if ok else 1)
