"""GRPO advantage computation when group rewards are identical."""

from skill_agents_grpo.grpo.advantage_utils import compute_grpo_group_advantages


def test_identical_rewards_zero_advantage_without_completions():
    adv = compute_grpo_group_advantages([0.9, 0.9, 0.9, 0.9], completions=None)
    assert len(adv) == 4
    assert all(abs(a) < 1e-9 for a in adv)


def test_identical_rewards_nonzero_advantage_with_completions():
    adv = compute_grpo_group_advantages(
        [0.9, 0.9, 0.9, 0.9],
        completions=["resp_a", "resp_b", "resp_c", "resp_d"],
    )
    assert len(adv) == 4
    assert abs(sum(adv)) < 1e-5
    assert max(abs(x) for x in adv) > 0.05


def test_different_rewards_still_spread():
    adv = compute_grpo_group_advantages(
        [0.2, 0.8, 0.5, 0.4],
        completions=["a", "b", "c", "d"],
    )
    assert max(adv) - min(adv) > 0.5
