"""Unit tests for :mod:`harness.rejection_deboost`.

Covers Refinement A of the "RAG-vs-harness" design loop:

* Pure scoring (:func:`compute_deboost`) — empty input, on-axis count,
  off-axis discount, recency decay, malformed entries, floor clamp.
* Convenience helper (:func:`apply_deboost_to_candidates`) — score
  multiplication, annotation key, re-sort, fetcher errors, missing
  skill_id, no-op on empty candidates.
* End-to-end through
  :func:`scripts.qwen3_decision_agent.get_top_k_skill_candidates`
  with a stub bank — verifies that a heavily-vetoed skill drops in
  rank below a fresh skill, and that the deboost can be opted out.
"""

from __future__ import annotations

import math
import time
import types
from typing import Any, Dict, List, Optional

import pytest

from harness.rejection_deboost import (
    DEFAULT_DEBOOST_FLOOR,
    DEFAULT_DEBOOST_HALF_LIFE_COUNT,
    apply_deboost_to_candidates,
    compute_deboost,
)


# ---------------------------------------------------------------------------
# compute_deboost
# ---------------------------------------------------------------------------


class TestComputeDeboost:
    def test_empty_patterns_returns_one(self):
        assert compute_deboost([]) == 1.0

    def test_none_patterns_returns_one(self):
        assert compute_deboost(None) == 1.0          # type: ignore[arg-type]

    def test_zero_count_pattern_is_ignored(self):
        patterns = [{"veto": "x", "domain": "d", "task": "t", "count": 0}]
        assert compute_deboost(patterns, domain="d", task="t") == 1.0

    def test_on_axis_count_decays_to_half_at_half_life(self):
        n = int(round(DEFAULT_DEBOOST_HALF_LIFE_COUNT))
        patterns = [{"veto": "v", "domain": "d", "task": "t", "count": n}]
        factor = compute_deboost(
            patterns, domain="d", task="t", recency_half_life_s=None,
        )
        # floor + (1-floor)*0.5  ≈ 0.10 + 0.45 = 0.55
        expected = DEFAULT_DEBOOST_FLOOR + (1.0 - DEFAULT_DEBOOST_FLOOR) * 0.5
        assert math.isclose(factor, expected, rel_tol=1e-6)

    def test_large_count_approaches_floor(self):
        patterns = [{"veto": "v", "domain": "d", "task": "t", "count": 10_000}]
        factor = compute_deboost(
            patterns, domain="d", task="t", recency_half_life_s=None,
        )
        assert math.isclose(factor, DEFAULT_DEBOOST_FLOOR, abs_tol=1e-6)

    def test_off_axis_uses_discount(self):
        # Same count, but off-axis: should be much milder than on-axis.
        on = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t", "count": 6}],
            domain="d", task="t", recency_half_life_s=None,
        )
        off = compute_deboost(
            [{"veto": "v", "domain": "OTHER", "task": "t", "count": 6}],
            domain="d", task="t", recency_half_life_s=None,
            off_axis_discount=0.25,
        )
        assert on < off < 1.0

    def test_off_axis_discount_zero_disables(self):
        factor = compute_deboost(
            [{"veto": "v", "domain": "OTHER", "task": "OTHER", "count": 100}],
            domain="d", task="t",
            off_axis_discount=0.0,
            recency_half_life_s=None,
        )
        assert factor == 1.0

    def test_empty_axis_strings_are_wildcards(self):
        # When the request domain/task are blank, every entry is on-axis.
        factor = compute_deboost(
            [{"veto": "v", "domain": "any", "task": "any", "count": 6}],
            domain="", task="",
            recency_half_life_s=None,
        )
        assert factor < 1.0

    def test_recency_old_pattern_weighs_less_than_new(self):
        now = 1_000_000.0
        recent = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t",
              "count": 6, "last_observed_at": now - 1.0}],
            domain="d", task="t",
            recency_half_life_s=86400.0, now_s=now,
        )
        old = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t",
              "count": 6, "last_observed_at": now - 10 * 86400.0}],
            domain="d", task="t",
            recency_half_life_s=86400.0, now_s=now,
        )
        assert recent < old < 1.0     # older patterns deboost less

    def test_recency_disabled_when_half_life_none(self):
        now = 1_000_000.0
        factor_a = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t",
              "count": 6, "last_observed_at": now - 1e9}],
            domain="d", task="t",
            recency_half_life_s=None, now_s=now,
        )
        factor_b = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t", "count": 6}],
            domain="d", task="t",
            recency_half_life_s=None, now_s=now,
        )
        assert math.isclose(factor_a, factor_b, rel_tol=1e-6)

    def test_floor_is_respected(self):
        factor = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t", "count": 10_000}],
            domain="d", task="t",
            deboost_floor=0.5,
            recency_half_life_s=None,
        )
        assert factor >= 0.5

    def test_floor_zero_allows_arbitrary_low(self):
        factor = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t", "count": 1_000_000}],
            domain="d", task="t",
            deboost_floor=0.0,
            recency_half_life_s=None,
        )
        assert factor < 1e-3

    def test_malformed_entry_is_silently_ignored(self):
        # Mixed garbage list shouldn't raise.
        patterns = [
            "not-a-dict",
            None,
            {"count": "not-an-int"},
            {"count": -3, "domain": "d", "task": "t"},
            {"veto": "v", "domain": "d", "task": "t", "count": 3,
             "last_observed_at": "garbage"},
        ]
        factor = compute_deboost(
            patterns, domain="d", task="t", recency_half_life_s=None,
        )
        # Only the last (count=3, malformed-but-recoverable timestamp) contributes.
        assert factor < 1.0

    def test_multiple_patterns_aggregate(self):
        # Two on-axis count=2 entries should equal one count=4 entry.
        a = compute_deboost(
            [
                {"veto": "v1", "domain": "d", "task": "t", "count": 2},
                {"veto": "v2", "domain": "d", "task": "t", "count": 2},
            ],
            domain="d", task="t", recency_half_life_s=None,
        )
        b = compute_deboost(
            [{"veto": "v", "domain": "d", "task": "t", "count": 4}],
            domain="d", task="t", recency_half_life_s=None,
        )
        assert math.isclose(a, b, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# apply_deboost_to_candidates
# ---------------------------------------------------------------------------


class _FakeRecord:
    def __init__(self, false_binding_patterns: List[Dict[str, Any]]):
        self.false_binding_patterns = list(false_binding_patterns)


class TestApplyDeboost:
    def _fetcher(self, mapping: Dict[str, _FakeRecord]):
        def _f(sid: str) -> Optional[_FakeRecord]:
            return mapping.get(sid)
        return _f

    def test_empty_input_no_op(self):
        out = apply_deboost_to_candidates(
            [], fetch_record=lambda sid: None,
        )
        assert out == []

    def test_no_skill_id_passes_through(self):
        cand = [{"confidence": 0.9}]
        out = apply_deboost_to_candidates(
            cand, fetch_record=lambda sid: None, sort_by=None,
        )
        assert out == cand
        assert "_harness_deboost" not in out[0]

    def test_unknown_skill_id_passes_through(self):
        cand = [{"skill_id": "missing", "confidence": 0.9}]
        out = apply_deboost_to_candidates(
            cand, fetch_record=lambda sid: None, sort_by=None,
        )
        assert out[0]["confidence"] == 0.9

    def test_clean_skill_gets_deboost_one(self):
        recs = {"clean": _FakeRecord([])}
        out = apply_deboost_to_candidates(
            [{"skill_id": "clean", "confidence": 0.9, "relevance": 0.8}],
            fetch_record=self._fetcher(recs),
            sort_by=None,
        )
        assert math.isclose(out[0]["confidence"], 0.9, rel_tol=1e-6)
        assert math.isclose(out[0]["relevance"], 0.8, rel_tol=1e-6)
        assert out[0]["_harness_deboost"] == 1.0

    def test_vetoed_skill_gets_lower_score(self):
        recs = {
            "fresh": _FakeRecord([]),
            "vetoed": _FakeRecord([
                {"veto": "v", "domain": "d", "task": "t", "count": 6},
            ]),
        }
        cands = [
            {"skill_id": "fresh", "confidence": 0.5, "relevance": 0.5},
            {"skill_id": "vetoed", "confidence": 0.9, "relevance": 0.9},
        ]
        out = apply_deboost_to_candidates(
            cands,
            fetch_record=self._fetcher(recs),
            domain="d", task="t",
            sort_by="confidence",
            recency_half_life_s=None,
        )
        # After deboost, even though "vetoed" started at 0.9, it should
        # drop below "fresh"'s 0.5 multiplied by 1.0.
        assert out[0]["skill_id"] == "fresh"
        assert out[1]["skill_id"] == "vetoed"
        assert out[1]["confidence"] < 0.9
        assert out[1]["_harness_deboost"] < 1.0

    def test_fetcher_exception_skips_candidate(self):
        def _bad_fetcher(sid: str):
            raise RuntimeError("bank offline")
        cand = [{"skill_id": "x", "confidence": 0.9}]
        out = apply_deboost_to_candidates(
            cand, fetch_record=_bad_fetcher, sort_by=None,
        )
        assert out[0]["confidence"] == 0.9
        assert "_harness_deboost" not in out[0]

    def test_annotation_key_can_be_disabled(self):
        recs = {"x": _FakeRecord([])}
        out = apply_deboost_to_candidates(
            [{"skill_id": "x", "confidence": 0.5}],
            fetch_record=self._fetcher(recs),
            annotation_key="",
            sort_by=None,
        )
        assert "_harness_deboost" not in out[0]

    def test_sort_by_none_preserves_order(self):
        recs = {
            "a": _FakeRecord([
                {"veto": "v", "domain": "d", "task": "t", "count": 6},
            ]),
            "b": _FakeRecord([]),
        }
        cands = [
            {"skill_id": "a", "confidence": 0.9},
            {"skill_id": "b", "confidence": 0.5},
        ]
        out = apply_deboost_to_candidates(
            cands,
            fetch_record=self._fetcher(recs),
            domain="d", task="t",
            sort_by=None,
            recency_half_life_s=None,
        )
        assert [c["skill_id"] for c in out] == ["a", "b"]

    def test_score_keys_filter_only_listed(self):
        recs = {"x": _FakeRecord([
            {"veto": "v", "domain": "d", "task": "t", "count": 10},
        ])}
        cands = [{"skill_id": "x", "confidence": 0.9, "relevance": 0.9, "extra": 0.9}]
        out = apply_deboost_to_candidates(
            cands,
            fetch_record=self._fetcher(recs),
            domain="d", task="t",
            score_keys=("confidence",),
            sort_by=None,
            recency_half_life_s=None,
        )
        assert out[0]["confidence"] < 0.9
        assert out[0]["relevance"] == 0.9
        assert out[0]["extra"] == 0.9


# ---------------------------------------------------------------------------
# End-to-end through get_top_k_skill_candidates
# ---------------------------------------------------------------------------


class _StubSelectResult:
    def __init__(self, sid: str, conf: float):
        # Provide skill_name AND execution_hint so `_enrich_candidate`
        # short-circuits cleanly (it would otherwise try to read
        # `skill_obj.strategic_description` off our fake record).
        self._d = {
            "skill_id": sid, "confidence": conf, "relevance": conf,
            "skill_name": sid, "execution_hint": f"hint for {sid}",
        }

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._d)


class _StubBank:
    """Mimics the SkillQueryEngine `.select()` + `.get_skill()` shape."""

    def __init__(self, results: List[_StubSelectResult],
                 records: Dict[str, _FakeRecord]):
        self._results = results
        self._records = records

    def select(self, key, current_state=None, current_predicates=None, top_k=3):
        return list(self._results)

    def get_skill(self, sid: str):
        return self._records.get(sid)


def _patch_protocol_lookup(monkeypatch):
    # `_get_protocol_for_skill` is imported lazily inside
    # get_top_k_skill_candidates; stub it so we don't need the real
    # decision_agents helper.
    import decision_agents.agent_helper as helper
    monkeypatch.setattr(
        helper, "_get_protocol_for_skill",
        lambda bank, sid: {"steps": []},
        raising=False,
    )


class TestGetTopKDeboostIntegration:
    def test_deboost_changes_rank(self, monkeypatch):
        _patch_protocol_lookup(monkeypatch)
        from scripts.qwen3_decision_agent import get_top_k_skill_candidates

        bank = _StubBank(
            results=[
                _StubSelectResult("vetoed", 0.95),
                _StubSelectResult("fresh", 0.30),
            ],
            records={
                "vetoed": _FakeRecord([
                    {"veto": "v", "domain": "crafter",
                     "task": "wake_up", "count": 8,
                     "last_observed_at": time.time()},
                ]),
                "fresh": _FakeRecord([]),
            },
        )

        # With deboost, vetoed should drop below fresh.
        deboosted = get_top_k_skill_candidates(
            bank, "state-text", game_name="crafter",
            intention="wake_up", top_k=3,
        )
        assert deboosted[0]["skill_id"] == "fresh"
        assert deboosted[1]["skill_id"] == "vetoed"
        assert deboosted[1]["_harness_deboost"] < 1.0
        assert deboosted[0].get("_harness_deboost") in (None, 1.0)

    def test_opt_out_preserves_rank(self, monkeypatch):
        _patch_protocol_lookup(monkeypatch)
        from scripts.qwen3_decision_agent import get_top_k_skill_candidates

        bank = _StubBank(
            results=[
                _StubSelectResult("vetoed", 0.95),
                _StubSelectResult("fresh", 0.30),
            ],
            records={
                "vetoed": _FakeRecord([
                    {"veto": "v", "domain": "crafter",
                     "task": "wake_up", "count": 8},
                ]),
                "fresh": _FakeRecord([]),
            },
        )

        raw = get_top_k_skill_candidates(
            bank, "state-text", game_name="crafter",
            intention="wake_up", top_k=3,
            apply_rejection_deboost=False,
        )
        assert raw[0]["skill_id"] == "vetoed"
        assert "_harness_deboost" not in raw[0]

    def test_bank_without_get_skill_passes_through(self, monkeypatch):
        _patch_protocol_lookup(monkeypatch)
        from scripts.qwen3_decision_agent import get_top_k_skill_candidates

        # Bank exposes `.select()` but no `.get_skill()` — the
        # _resolve_skill_record_fetcher path should return None and
        # the deboost step should silently no-op.
        class _Bank:
            def select(self, *a, **kw):
                return [_StubSelectResult("a", 0.5), _StubSelectResult("b", 0.4)]

        out = get_top_k_skill_candidates(
            _Bank(), "state-text", game_name="g", intention="i",
        )
        # Order preserved (no deboost applied).
        assert [c["skill_id"] for c in out] == ["a", "b"]
        for c in out:
            assert "_harness_deboost" not in c

    def test_none_bank_returns_empty(self):
        from scripts.qwen3_decision_agent import get_top_k_skill_candidates

        assert get_top_k_skill_candidates(None, "state") == []
