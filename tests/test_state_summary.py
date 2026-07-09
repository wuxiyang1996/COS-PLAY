"""Tests for the compact state summary pipeline.

Run:  python -m pytest tests/test_state_summary.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from decision_agents.agent_helper import (
    DEFAULT_SUMMARY_CHAR_BUDGET,
    HARD_SUMMARY_CHAR_LIMIT,
    _safe_str,
    _remove_boilerplate,
    _truncate_keep_important,
    _join_kv,
    compact_structured_state,
    compact_text_observation,
    get_state_summary,
)


# ---------------------------------------------------------------------------
# Budget constant sanity checks
# ---------------------------------------------------------------------------

class TestBudgetConstants:
    def test_default_budget_is_400(self):
        assert DEFAULT_SUMMARY_CHAR_BUDGET == 400

    def test_hard_limit_is_400(self):
        assert HARD_SUMMARY_CHAR_LIMIT == 400


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestSafeStr:
    def test_none(self):
        assert _safe_str(None) == ""

    def test_string(self):
        assert _safe_str("  hello  ") == "hello"

    def test_short_list(self):
        assert _safe_str([1, 2, 3]) == "1,2,3"

    def test_long_list(self):
        result = _safe_str(list(range(10)))
        assert "..+4" in result

    def test_dict(self):
        result = _safe_str({"a": 1, "b": 2})
        assert "a:1" in result


class TestRemoveBoilerplate:
    def test_removes_choose_action(self):
        text = "You are at (1,2).\n\nChoose one action: north, south, east, west."
        cleaned = _remove_boilerplate(text)
        assert "choose one action" not in cleaned.lower()
        assert "You are at (1,2)" in cleaned

    def test_removes_valid_actions(self):
        text = "State info.\nValid actions: a, b, c\nMore info."
        cleaned = _remove_boilerplate(text)
        assert "valid actions" not in cleaned.lower()

    def test_removes_respond_with(self):
        text = "Phase: team vote.\nReply with: approve / reject"
        cleaned = _remove_boilerplate(text)
        assert "reply with" not in cleaned.lower()

    def test_removes_order_format(self):
        text = "Your units: A PAR.\n--- Order Format ---\n  Hold:         A PAR H\n  Move:         A PAR - BUR"
        cleaned = _remove_boilerplate(text)
        assert "order format" not in cleaned.lower()


class TestTruncate:
    def test_short_text(self):
        assert _truncate_keep_important("hello", 100) == "hello"

    def test_cut_at_sentence(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate_keep_important(text, 35)
        assert len(result) <= 35
        assert "First sentence." in result


class TestJoinKV:
    def test_simple(self):
        parts = [("a", "1"), ("b", "2"), ("c", "3")]
        result = _join_kv(parts, 100)
        assert result == "a=1 | b=2 | c=3"

    def test_budget_respected(self):
        parts = [("game", "overcooked"), ("self", "hold:onion pos:1,2"),
                 ("ally", "hold:dish"), ("critical", "pot:2ing:cooking")]
        result = _join_kv(parts, 50)
        assert len(result) <= 50

    def test_skips_empty_values(self):
        parts = [("a", "1"), ("b", ""), ("c", "3")]
        result = _join_kv(parts, 100)
        assert "b=" not in result


# ---------------------------------------------------------------------------
# Structured state compression
# ---------------------------------------------------------------------------

class TestCompactStructuredState:
    def test_overcooked_example(self):
        state = {
            "game": "overcooked",
            "self": "hold:onion pos:1,2",
            "ally": "hold:dish",
            "critical": "pot:2ing:cooking",
            "orders": "onion_soup",
            "time_left": "47",
        }
        result = compact_structured_state(state)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        assert "game=overcooked" in result
        assert "self=" in result
        print(f"  Overcooked structured: [{len(result)}] {result}")

    def test_avalon_example(self):
        state = {
            "game": "avalon",
            "phase": "team_vote",
            "self": "role:Percival(G)",
            "progress": "quest:1/5 good:1 evil:0",
            "critical": "leader:p3 team_sz:3 rd:2/5",
            "objective": "approve_or_reject_team",
        }
        result = compact_structured_state(state)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        assert "game=avalon" in result
        assert "phase=team_vote" in result
        print(f"  Avalon structured:    [{len(result)}] {result}")

    def test_diplomacy_example(self):
        state = {
            "game": "diplomacy",
            "phase": "S1902M",
            "self": "power:FRANCE centers:5",
            "critical": "locs:PAR,BRE,MAR",
            "resources": "units:A PAR,F BRE,A MAR",
            "objective": "issue_move_orders",
        }
        result = compact_structured_state(state)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        assert "game=diplomacy" in result
        assert "phase=S1902M" in result
        print(f"  Diplomacy structured: [{len(result)}] {result}")

    def test_gamingagent_example(self):
        from env_wrappers.gamingagent_nl_wrapper import build_structured_state_summary
        state = build_structured_state_summary(
            "Player at (2,3). Box at (3,3). Goal at (4,3). Push box right.",
            step=14,
            action_names=["push_left", "push_right", "push_up", "push_down"],
        )
        result = compact_structured_state(state)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        assert "game=" in result
        print(f"  GamingAgent structured: [{len(result)}] {result}")

    def test_videogamebench_example(self):
        from env_wrappers.videogamebench_nl_wrapper import build_structured_state_summary
        state = build_structured_state_summary(
            step=37,
            last_action=5,
            last_reward=0.0,
            recent_actions=[5, 5, 5, 5, 5, 5, 5, 5],
            recent_rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            game_name="kirby",
        )
        result = compact_structured_state(state)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        assert "game=kirby" in result
        assert "stall" in result.lower() or "repeat" in result.lower()
        print(f"  VideoGameBench structured: [{len(result)}] {result}")

    def test_respects_hard_limit(self):
        state = {f"key{i}": f"value{'x' * 50}{i}" for i in range(20)}
        result = compact_structured_state(state, max_chars=999)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT

    def test_empty_dict(self):
        assert compact_structured_state({}) == ""

    def test_priority_ordering(self):
        state = {"zzz": "last", "game": "test", "phase": "p1"}
        result = compact_structured_state(state)
        assert result.startswith("game=test")


# ---------------------------------------------------------------------------
# Text observation compression
# ---------------------------------------------------------------------------

class TestCompactTextObservation:
    def test_short_observation_still_compressed(self):
        obs = "You are at (1,2) holding onion."
        result = compact_text_observation(obs)
        assert result
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        print(f"  Short obs:  [{len(result)}] {result}")

    def test_long_observation_under_budget(self):
        obs = (
            "=== Diplomacy — Phase: S1901M ===\n"
            "You are: FRANCE.\n\n"
            "Your units: ['A PAR', 'F BRE', 'A MAR']\n"
            "Your supply centers: ['PAR', 'BRE', 'MAR'] (3 total)\n"
            "Your home centers: ['PAR', 'BRE', 'MAR']\n\n"
            "--- All Powers Status ---\n"
            "  FRANCE (you): 3 centers, units=['A PAR', 'F BRE', 'A MAR']\n"
            "  ENGLAND: 3 centers, units=['F LON', 'F EDI', 'A LVP']\n\n"
            "--- Your Possible Orders ---\n"
            "  PAR: A PAR H, A PAR - BUR, A PAR - PIC\n"
            "  BRE: F BRE H, F BRE - ENG, F BRE - MAO\n\n"
            "--- Order Format ---\n"
            "Movement phase. Issue orders for each unit:\n"
            "  Hold:         A PAR H\n"
            "  Move:         A PAR - BUR\n\n"
            "Submit your orders as a list of order strings.\n"
            "Example: [\"A PAR - BUR\", \"F BRE H\", \"A MAR S A PAR - BUR\"]"
        )
        result = compact_text_observation(obs)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT
        assert "submit your orders" not in result.lower()
        assert "order format" not in result.lower()
        print(f"  Long obs:   [{len(result)}] {result}")

    def test_empty_observation(self):
        assert compact_text_observation("") == ""
        assert compact_text_observation(None) == ""

    def test_respects_hard_limit(self):
        obs = "word " * 500
        result = compact_text_observation(obs, max_chars=999)
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT


# ---------------------------------------------------------------------------
# Main entry point: get_state_summary
# ---------------------------------------------------------------------------

class TestGetStateSummary:
    def test_never_returns_raw_verbatim(self):
        obs = "You are at (1,2), facing north, holding nothing."
        result = get_state_summary(obs)
        assert result
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT

    def test_structured_takes_priority(self):
        obs = "Long observation text " * 20
        structured = {"game": "test", "phase": "p1", "self": "hero"}
        result = get_state_summary(obs, structured_state=structured)
        assert "game=test" in result
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT

    def test_text_fallback_when_no_structured(self):
        obs = "Player at (3,4). Enemy nearby. Health low."
        result = get_state_summary(obs)
        assert result
        assert "Player at (3,4)" in result

    def test_empty_observation_empty_result(self):
        assert get_state_summary("") == ""
        assert get_state_summary(None) == ""

    def test_llm_fallback_disabled_by_default(self):
        obs = "x " * 2000
        result = get_state_summary(obs)
        assert result
        assert len(result) <= HARD_SUMMARY_CHAR_LIMIT

    def test_backward_compat_kwargs(self):
        """Legacy game= and model= kwargs should not raise."""
        obs = "Game state info here."
        result = get_state_summary(obs, game="overcooked", model="gpt-4o-mini")
        assert result

    def test_max_chars_clamped_to_400(self):
        """Even if caller passes max_chars > 400, output must be <= 400."""
        obs = "word " * 200
        result = get_state_summary(obs, max_chars=999)
        assert len(result) <= 400


# ---------------------------------------------------------------------------
# Demo: print example summaries
# ---------------------------------------------------------------------------

def demo_examples():
    """Print example summaries for each supported environment."""
    print("\n" + "=" * 60)
    print("STATE SUMMARY PIPELINE — DEMO EXAMPLES")
    print(f"Budget: DEFAULT={DEFAULT_SUMMARY_CHAR_BUDGET}  HARD={HARD_SUMMARY_CHAR_LIMIT}")
    print("=" * 60)

    # --- Overcooked ---
    oc_struct = {
        "game": "overcooked",
        "self": "hold:onion pos:1,2",
        "ally": "hold:dish",
        "critical": "pot:2ing:cooking",
        "orders": "onion_soup",
        "time_left": "47",
    }
    oc_text = (
        "You are at position (1, 2), facing north, holding onion.\n"
        "Your teammate is at position (3, 1), facing east, holding dish.\n"
        "Objects on the grid:\n  - At (2, 0): soup (onion, onion; cooking)\n"
        "Current timestep: 53.\nSteps remaining in episode: 47.\n"
        "Choose one action: north, south, east, west, stay, or interact."
    )
    print("\n--- Overcooked (structured) ---")
    s = compact_structured_state(oc_struct)
    print(f"  [{len(s)} chars] {s}")
    print("--- Overcooked (text fallback) ---")
    s = compact_text_observation(oc_text)
    print(f"  [{len(s)} chars] {s}")
    print("--- Overcooked (get_state_summary, structured) ---")
    s = get_state_summary(oc_text, structured_state=oc_struct)
    print(f"  [{len(s)} chars] {s}")

    # --- Avalon ---
    av_struct = {
        "game": "avalon",
        "phase": "team_vote",
        "self": "role:Percival(G)",
        "progress": "quest:1/5 good:1 evil:0",
        "critical": "leader:p3 team_sz:3 rd:2/5",
        "objective": "approve_or_reject_team",
    }
    av_text = (
        "=== Avalon Game — Team Voting ===\n"
        "You are Player 2 (of 5 players).\nYour role: Percival (Good side).\n"
        "Quest results so far: [Success] (Good 1 - Evil 0).\n"
        "Current quest: 2 of 5.\nCurrent round: 2 of 5.\n"
        "Proposed team: [0, 2, 3].\n\n"
        "Vote to APPROVE or REJECT this team.\nReply with: approve / reject"
    )
    print("\n--- Avalon (structured) ---")
    s = compact_structured_state(av_struct)
    print(f"  [{len(s)} chars] {s}")
    print("--- Avalon (text fallback) ---")
    s = compact_text_observation(av_text)
    print(f"  [{len(s)} chars] {s}")

    # --- Diplomacy ---
    dip_struct = {
        "game": "diplomacy",
        "phase": "S1902M",
        "self": "power:FRANCE centers:5",
        "critical": "locs:PAR,BRE,MAR,BUR,PIC",
        "resources": "units:A PAR,F BRE,A MAR,A BUR,F PIC",
        "objective": "issue_move_orders",
    }
    print("\n--- Diplomacy (structured) ---")
    s = compact_structured_state(dip_struct)
    print(f"  [{len(s)} chars] {s}")

    # --- GamingAgent ---
    from env_wrappers.gamingagent_nl_wrapper import build_structured_state_summary as ga_build
    ga_struct = ga_build(
        "Player at (2,3). Box at (3,3). Goal at (4,3). Push box right to reach the goal.",
        step=14,
        action_names=["push_left", "push_right", "push_up", "push_down"],
    )
    print("\n--- GamingAgent (structured) ---")
    s = compact_structured_state(ga_struct)
    print(f"  [{len(s)} chars] {s}")

    # --- VideoGameBench ---
    from env_wrappers.videogamebench_nl_wrapper import build_structured_state_summary as vgb_build
    vgb_struct = vgb_build(
        step=37,
        last_action=5,
        last_reward=0.0,
        recent_actions=[5, 5, 5, 5, 5, 5, 5, 5],
        recent_rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        game_name="kirby",
    )
    print("\n--- VideoGameBench (structured) ---")
    s = compact_structured_state(vgb_struct)
    print(f"  [{len(s)} chars] {s}")

    vgb_progress = vgb_build(
        step=12,
        last_action=1,
        last_reward=10.0,
        recent_actions=[5, 5, 7, 1, 1],
        recent_rewards=[0.0, 0.0, 5.0, 10.0, 10.0],
        game_name="kirby",
    )
    print("--- VideoGameBench (with progress) ---")
    s = compact_structured_state(vgb_progress)
    print(f"  [{len(s)} chars] {s}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_examples()
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
