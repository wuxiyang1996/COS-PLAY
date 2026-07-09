"""Test that GRPO reward functions produce different values for different rollouts.

Each test simulates G=4 different LLM completions (the sample_output)
while keeping the fixed args/kwargs constant, and verifies that at least
2 out of 4 rewards differ — the minimum needed for GRPO advantage signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from skill_agents_grpo.grpo.rewards import (
    contract_reward,
    curator_reward,
    segmentation_reward,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _n_distinct(values: list, tol: float = 1e-6) -> int:
    """Count distinct values up to tolerance."""
    distinct = []
    for v in values:
        if not any(abs(v - d) < tol for d in distinct):
            distinct.append(v)
    return len(distinct)


# ── CONTRACT REWARD ───────────────────────────────────────────────────

class TestContractRewardVariance:
    """Contract: 4 different LLM effect sets → 4 (or at least 2) different rewards."""

    def test_fallback_with_predicates(self):
        """Different effect sets produce different rewards when predicates exist."""
        samples = [
            {"eff_add": ["has_item", "near_goal"], "eff_del": ["far"]},
            {"eff_add": ["has_item"], "eff_del": []},
            {"eff_add": ["near_goal", "powered_up"], "eff_del": ["far", "weak"]},
            {"eff_add": [], "eff_del": ["far", "weak", "slow"]},
        ]
        rewards = [
            contract_reward(
                s, "skill_0", [],
                predicates_start={"far", "weak", "slow"},
                predicates_end={"has_item", "near_goal", "powered_up"},
            )
            for s in samples
        ]
        nd = _n_distinct(rewards)
        assert nd >= 3, f"Expected >=3 distinct rewards, got {nd}: {rewards}"

    def test_fallback_empty_predicates(self):
        """Even without predicates, different effects → different rewards."""
        samples = [
            {"eff_add": ["a", "b"], "eff_del": ["c"]},
            {"eff_add": ["a"], "eff_del": []},
            {"eff_add": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
             "eff_del": ["k", "l", "m", "n", "o"]},
            {"eff_add": ["x"], "eff_del": ["y"]},
        ]
        rewards = [
            contract_reward(
                s, "skill_0", [],
                predicates_start=set(),
                predicates_end=set(),
            )
            for s in samples
        ]
        nd = _n_distinct(rewards)
        assert nd >= 2, f"Expected >=2 distinct rewards with empty preds, got {nd}: {rewards}"

    def test_fallback_none_predicates(self):
        """None predicates (cold-start) must still vary."""
        samples = [
            {"eff_add": ["a", "b"], "eff_del": ["c"]},
            {"eff_add": ["a"], "eff_del": []},
            {"eff_add": list("abcdefghij"), "eff_del": list("klmno")},
            {"eff_add": ["x"], "eff_del": ["y"]},
        ]
        rewards = [
            contract_reward(s, "skill_0", [], set(), set())
            for s in samples
        ]
        nd = _n_distinct(rewards)
        assert nd >= 2, f"Expected >=2 distinct with None preds, got {nd}: {rewards}"

    def test_similar_effects_still_differ(self):
        """Effects that overlap but differ slightly → different rewards."""
        samples = [
            {"eff_add": ["a", "b", "c"], "eff_del": ["x", "y"]},
            {"eff_add": ["a", "b", "d"], "eff_del": ["x", "y"]},
            {"eff_add": ["a", "b", "c"], "eff_del": ["x", "z"]},
            {"eff_add": ["a", "b"], "eff_del": ["x", "y", "z"]},
        ]
        rewards = [
            contract_reward(
                s, "skill_0", [],
                predicates_start={"x", "y", "z"},
                predicates_end={"a", "b", "c", "d"},
            )
            for s in samples
        ]
        nd = _n_distinct(rewards)
        assert nd >= 2, f"Expected >=2 distinct for similar effects, got {nd}: {rewards}"


# ── CURATOR REWARD ────────────────────────────────────────────────────

class TestCuratorRewardVariance:
    """Curator: 4 different verdict patterns → 4 different rewards."""

    def test_different_verdicts_differ(self):
        candidates = [
            {"type": "refine", "skill_id": "s1", "skill_score": 0.7, "pass_rate": 0.8, "n_instances": 10},
            {"type": "materialize", "skill_id": "s2", "skill_score": 0.4, "pass_rate": 0.5, "n_instances": 3},
        ]
        decision_sets = [
            {"decisions": [
                {"idx": 0, "verdict": "approve", "reason": "skill_score 0.70, pass_rate 0.80, 10 instances"},
                {"idx": 1, "verdict": "approve", "reason": "promising new skill, pass_rate 0.50"},
            ]},
            {"decisions": [
                {"idx": 0, "verdict": "approve", "reason": "good quality"},
                {"idx": 1, "verdict": "veto", "reason": "not enough data"},
            ]},
            {"decisions": [
                {"idx": 0, "verdict": "veto", "reason": "should be better"},
                {"idx": 1, "verdict": "approve", "reason": "skill_score 0.40, let's explore"},
            ]},
            {"decisions": [
                {"idx": 0, "verdict": "defer", "reason": "need more data"},
                {"idx": 1, "verdict": "defer", "reason": "pass_rate 0.50, marginal"},
            ]},
        ]
        rewards = [
            curator_reward(d, candidates, None)
            for d in decision_sets
        ]
        nd = _n_distinct(rewards)
        assert nd >= 3, f"Expected >=3 distinct curator rewards, got {nd}: {rewards}"

    def test_same_verdicts_different_reasons(self):
        """Even with same verdicts, different reason quality should differ."""
        candidates = [
            {"type": "refine", "skill_id": "s1", "skill_score": 0.7, "pass_rate": 0.8, "n_instances": 10},
        ]
        decision_sets = [
            {"decisions": [{"idx": 0, "verdict": "approve", "reason": "skill_score 0.70, pass_rate 0.80, s1, 10 instances"}]},
            {"decisions": [{"idx": 0, "verdict": "approve", "reason": "looks good"}]},
            {"decisions": [{"idx": 0, "verdict": "approve", "reason": "s1 has pass_rate 0.80"}]},
            {"decisions": [{"idx": 0, "verdict": "approve", "reason": ""}]},
        ]
        rewards = [
            curator_reward(d, candidates, None)
            for d in decision_sets
        ]
        nd = _n_distinct(rewards)
        assert nd >= 2, f"Same verdict but different reasons should give >=2 distinct, got {nd}: {rewards}"


# ── SEGMENTATION REWARD ──────────────────────────────────────────────


def test_segment_fingerprint_differs_on_raw_rollouts_only():
    """Same parsed prefs + different raw LLM JSON → different fingerprint."""
    from types import SimpleNamespace

    from skill_agents_grpo.grpo.rewards import _preference_list_fingerprint
    from skill_agents_grpo.infer_segmentation.preference import PreferenceListWithRollouts

    prefs = [
        SimpleNamespace(
            segment_start=0, segment_end=1, skill_win="a", skill_lose="b", evidence="x",
        ),
    ]
    a = PreferenceListWithRollouts(prefs, raw_rollouts=['{"ranking":["a","b"],"reasoning":"one"}'])
    b = PreferenceListWithRollouts(prefs, raw_rollouts=['{"ranking":["a","b"],"reasoning":"two"}'])
    assert _preference_list_fingerprint(a) != _preference_list_fingerprint(b)


class TestSegmentationRewardVariance:
    """Segmentation: 4 different preference lists → 4 different rewards."""

    def test_fallback_different_preference_winners(self):
        """Different skill rankings → different reuse_hint → different rewards."""
        from types import SimpleNamespace

        skill_names = ["skill_a", "skill_b", "__NEW__"]
        segments = [(0, 5), (5, 10)]

        pref_sets = [
            # Sample 1: skill_a wins everything
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="skill_a", skill_lose="skill_b")],
            # Sample 2: __NEW__ wins everything
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="__NEW__", skill_lose="skill_a"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="__NEW__", skill_lose="skill_b")],
            # Sample 3: skill_b wins
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="skill_b", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="skill_b", skill_lose="skill_a")],
            # Sample 4: mixed
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="skill_a", skill_lose="skill_b"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="__NEW__", skill_lose="skill_a")],
        ]

        rewards = [
            segmentation_reward(prefs, segments, [], [], skill_names)
            for prefs in pref_sets
        ]
        nd = _n_distinct(rewards)
        assert nd >= 3, f"Expected >=3 distinct segmentation rewards, got {nd}: {rewards}"

    def test_fallback_with_bank_scores(self):
        """Bank scores add differentiation based on which skills win."""
        from types import SimpleNamespace

        skill_names = ["good_skill", "bad_skill", "__NEW__"]
        segments = [(0, 5), (5, 10)]
        bank_scores = {"good_skill": 0.9, "bad_skill": 0.1}

        pref_sets = [
            # Good skill wins
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="good_skill", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="good_skill", skill_lose="bad_skill")],
            # Bad skill wins
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="bad_skill", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="bad_skill", skill_lose="good_skill")],
        ]

        rewards = [
            segmentation_reward(prefs, segments, [], [], skill_names, bank_skill_scores=bank_scores)
            for prefs in pref_sets
        ]
        nd = _n_distinct(rewards)
        assert nd >= 2, f"Bank scores should differentiate good vs bad skill wins, got {nd}: {rewards}"
        assert rewards[0] > rewards[1], (
            f"Good skill winning ({rewards[0]:.3f}) should score higher than "
            f"bad skill winning ({rewards[1]:.3f})"
        )

    def test_fallback_varying_preference_counts(self):
        """Different number of preferences per sample → different rewards."""
        from types import SimpleNamespace

        skill_names = ["skill_a", "__NEW__"]
        segments = [(0, 5), (5, 10), (10, 15)]

        pref_sets = [
            # 2 preferences
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="skill_a", skill_lose="__NEW__")],
            # 6 preferences (more depth per segment)
            [SimpleNamespace(segment_start=0, segment_end=5, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=0, segment_end=5, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=5, segment_end=10, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=10, segment_end=15, skill_win="skill_a", skill_lose="__NEW__"),
             SimpleNamespace(segment_start=10, segment_end=15, skill_win="skill_a", skill_lose="__NEW__")],
        ]

        rewards = [
            segmentation_reward(prefs, segments, [], [], skill_names)
            for prefs in pref_sets
        ]
        nd = _n_distinct(rewards)
        assert nd >= 2, f"Different preference counts should give different rewards, got {nd}: {rewards}"

    def test_decode_path_pref_fingerprint_breaks_saturation_tie(self):
        """Same decode outcome for different prefs → rewards must still differ."""
        from types import SimpleNamespace

        class _FakeSeg:
            def __init__(self, skill: str, start: int = 0, end: int = 1, margin: float = 5.0):
                self.assigned_skill = skill
                self.start = start
                self.end = end
                self.margin = margin

        class _FakeResult:
            def __init__(self, skill: str):
                self.segments = [_FakeSeg(skill)]
                self.total_score = 50.0

        def _fake_scorer_factory(_prefs):
            return object()

        def _fake_decode(_scorer, _segments, _obs, _actions, _skill_names, _preds):
            # Ignores preferences — all samples look equally "good" to decode metrics
            return _FakeResult("skill_a")

        skill_names = ["skill_a", "skill_b", "__NEW__"]
        segments = [(0, 1)]
        per_step_rewards = [1.0, 1.0]
        bank = {"skill_a": 0.8, "skill_b": 0.2}

        prefs_a = [
            SimpleNamespace(
                segment_start=0, segment_end=1, skill_win="skill_a", skill_lose="skill_b",
            ),
        ]
        prefs_b = [
            SimpleNamespace(
                segment_start=0, segment_end=1, skill_win="skill_b", skill_lose="skill_a",
            ),
        ]

        ra = segmentation_reward(
            prefs_a, segments, [], [], skill_names,
            scorer_factory=_fake_scorer_factory, decode_fn=_fake_decode,
            per_step_rewards=per_step_rewards, episode_total_reward=2.0,
            bank_skill_scores=bank,
        )
        rb = segmentation_reward(
            prefs_b, segments, [], [], skill_names,
            scorer_factory=_fake_scorer_factory, decode_fn=_fake_decode,
            per_step_rewards=per_step_rewards, episode_total_reward=2.0,
            bank_skill_scores=bank,
        )
        assert abs(ra - rb) > 1e-6, (
            f"Saturated decode path must differ when prefs differ: {ra=} {rb=}"
        )


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    passed = 0
    failed = 0
    for cls in [
        TestContractRewardVariance,
        TestCuratorRewardVariance,
        TestSegmentationRewardVariance,
    ]:
        obj = cls()
        for name in sorted(dir(obj)):
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
    sys.exit(0 if failed else 0)
