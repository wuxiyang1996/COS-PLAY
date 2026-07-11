"""Shared utilities for protocol-aware skill lifecycle management.

Provides predicate checking against parsed ``summary_state`` dicts,
progress tracking helpers, and the canonical effect-tag registry used
by the skill-selection LoRA.  Used by ``_SkillTracker`` in both
``scripts/qwen3_decision_agent.py`` and
``trainer/coevolution/episode_runner.py``.

Text-environment (WebShop, ALFWorld) effect computation and structured-
state helpers are also defined here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_CMP_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*([<>=!]+)\s*(.+)$"
)


def parse_summary_state(state_str: str) -> Dict[str, str]:
    """Parse a ``key=value | key=value`` summary_state string into a dict."""
    result: Dict[str, str] = {}
    if not state_str:
        return result
    for part in state_str.split("|"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.strip()
    return result


def check_predicate(pred: str, state: Dict[str, str]) -> bool:
    """Check a single predicate against a parsed summary_state dict.

    Supported formats:
      ``key=value``      — exact match
      ``key!=value``     — not equal
      ``key>N``          — numeric greater-than
      ``key<N``          — numeric less-than
      ``key>=N``         — numeric greater-or-equal
      ``key<=N``         — numeric less-or-equal

    Returns False if the key is missing from state or parsing fails.
    """
    m = _CMP_RE.match(pred.strip())
    if not m:
        return False
    key, op, expected = m.group(1), m.group(2), m.group(3).strip()
    actual = state.get(key)
    if actual is None:
        return False

    if op == "==" or op == "=":
        return actual == expected
    if op == "!=":
        return actual != expected

    try:
        a_num = float(actual)
        e_num = float(expected)
    except (ValueError, TypeError):
        return False

    if op == ">":
        return a_num > e_num
    if op == "<":
        return a_num < e_num
    if op == ">=":
        return a_num >= e_num
    if op == "<=":
        return a_num <= e_num
    return False


def check_predicates(preds: List[str], state: Dict[str, str]) -> bool:
    """Return True if ALL predicates pass (AND semantics)."""
    if not preds:
        return False
    return all(check_predicate(p, state) for p in preds)


def check_any_predicate(preds: List[str], state: Dict[str, str]) -> bool:
    """Return True if ANY predicate passes (OR semantics)."""
    if not preds:
        return False
    return any(check_predicate(p, state) for p in preds)


def keyword_match(criteria_text: str, state_text: str) -> bool:
    """Legacy keyword matching (fallback when no predicates available).

    Checks if at least 3-char tokens from *criteria_text* all appear in
    *state_text*.  This is the old behavior from ``_SkillTracker``.
    """
    if not criteria_text or not state_text:
        return False
    state_lower = state_text.lower()
    tokens = [t for t in criteria_text.lower().split() if len(t) >= 3]
    return bool(tokens) and all(tok in state_lower for tok in tokens[:3])


def compute_step_advancement(
    current_idx: int,
    step_checks: List[str],
    state: Dict[str, str],
    total_steps: int,
) -> int:
    """Determine the protocol step index after one timestep.

    If ``step_checks`` are available and the current step's check passes,
    advance.  If no checks are defined, advance by one (legacy behavior).
    Returns the new step index (clamped to ``total_steps - 1``).
    """
    if total_steps <= 0:
        return 0

    if not step_checks or current_idx >= len(step_checks):
        return min(current_idx + 1, total_steps - 1)

    check = step_checks[current_idx]
    if not check:
        return min(current_idx + 1, total_steps - 1)

    if check_predicate(check, state):
        return min(current_idx + 1, total_steps - 1)

    return current_idx


def build_progress_summary(
    steps: List[str],
    step_checks: List[str],
    current_idx: int,
    state: Dict[str, str],
) -> str:
    """Build a short progress summary for prompt injection.

    Returns a string like:
      ``Steps 1-2 done. Current: step 3 — Shift piece to target column.``
    """
    if not steps:
        return ""

    completed = []
    for i in range(min(current_idx, len(steps))):
        completed.append(i + 1)

    parts = []
    if completed:
        if len(completed) == 1:
            parts.append(f"Step {completed[0]} done.")
        else:
            parts.append(f"Steps {completed[0]}-{completed[-1]} done.")

    if current_idx < len(steps):
        parts.append(f"Current: step {current_idx + 1} — {steps[current_idx][:80]}")

    return " ".join(parts)


def compute_expected_duration(
    sub_episode_lengths: List[int],
    protocol_steps: int = 0,
) -> int:
    """Compute a reasonable expected_duration from sub-episode statistics.

    Uses the median length (robust to outliers), capped between
    ``max(protocol_steps, 3)`` and 30.  Falls back to ``protocol_steps``
    or 10 if no data.
    """
    min_dur = max(protocol_steps, 3) if protocol_steps > 0 else 3
    if not sub_episode_lengths:
        return max(min_dur, protocol_steps) if protocol_steps > 0 else 10

    sorted_lens = sorted(sub_episode_lengths)
    n = len(sorted_lens)
    if n % 2 == 0:
        median = (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2
    else:
        median = sorted_lens[n // 2]

    return max(min_dur, min(int(median), 30))


# ══════════════════════════════════════════════════════════════════════
# CANONICAL EFFECT TAG REGISTRY
# ──────────────────────────────────────────────────────────────────────
# Closed-set vocabulary for the skill-selection LoRA's EFFECTS output.
# Each tag must be independently observable from env state/actions.
# ══════════════════════════════════════════════════════════════════════

EFFECT_REGISTRY: Dict[str, str] = {
    # ── Universal ─────────────────────────────────────────────────────
    "state_observed":            "Agent has perceived / inspected current state",
    "action_taken":              "Agent executed at least one action this turn",
    "action_executed":           "A domain-specific action was performed",
    "reward_positive":           "Positive reward received this step",
    "cumulative_reward_positive":"Sum of rewards across skill so far > 0",
    "score_increased":           "Numeric score went up",

    # ── Reasoning / QA ────────────────────────────────────────────────
    "evidence_cited":            "Relevant visual or textual evidence extracted",
    "hypothesis_formed":         "A candidate hypothesis / interpretation stated",
    "context_retrieved":         "External or temporal context recalled / fetched",
    "options_compared":          "Multiple candidate answers compared",
    "candidates_eliminated":     "At least one wrong option ruled out",
    "answer_selected":           "A single best answer chosen",
    "answer_emitted":            "Final answer committed / output produced",
    "answer_confirmed":          "Answer cross-checked against evidence",

    # ── Board / puzzle ────────────────────────────────────────────────
    "board_transformed":         "Board layout changed from previous state",
    "board_crowded":             "Board is near capacity / few open cells",
    "board_reshuffled":          "Board was reshuffled or cascaded",
    "tile_promoted":             "Highest tile value increased (2048)",
    "merge_executed":            "A merge / combine occurred (2048)",
    "direction_applied":         "A directional move was applied (2048)",
    "piece_placed":              "A piece was placed on the board (tetris/columns)",
    "piece_changed":             "Active piece changed / new piece spawned (tetris)",
    "piece_rotated":             "A piece was rotated (columns)",
    "line_cleared":              "One or more lines cleared (tetris)",
    "holes_reduced":             "Board holes decreased (tetris)",
    "holes_created":             "Board holes increased (tetris, negative signal)",
    "move_applied":              "A move/shift/rotate was executed (tetris)",
    "match_scored":              "A match / cascade scored points (candy/columns)",
    "swap_applied":              "A swap action performed (candy crush)",
    "move_spent":                "A move resource was consumed (candy crush)",

    # ── Platformer / action ───────────────────────────────────────────
    "position_changed":          "Agent position moved to a new location",
    "mario_moved":               "Mario character moved (super_mario specific)",
    "progress_made":             "Forward progress toward goal",
    "damage_taken":              "Agent took damage / lost health",
    "obstacle_cleared":          "An obstacle or hazard was successfully avoided",
    "collectible_obtained":      "A coin / power-up / item was collected",

    # ── Shooter / combat ──────────────────────────────────────────────
    "enemy_hit":                 "An enemy was hit / destroyed",
    "projectile_fired":          "Agent fired a projectile",
    "attack_landed":             "A melee / ranged attack connected",

    # ── Web interaction (webshop) ─────────────────────────────────────
    "page_navigated":            "Browser navigated to a new URL / page",
    "form_filled":               "A text input / form field was filled",
    "element_clicked":           "A UI element was clicked",
    "dom_changed":               "Page DOM changed meaningfully after action",
    "item_found":                "Target item / element located on page",
    "search_performed":          "A search query was submitted",
    "product_selected":          "A product / option was chosen from results",
    "cart_updated":              "Shopping cart was modified (webshop)",

    # ── Embodied household (alfworld) ─────────────────────────────────
    "location_changed":          "Agent moved to a different room / receptacle",
    "object_picked_up":          "Agent picked up an object",
    "object_put_down":           "Agent placed an object in a receptacle",
    "receptacle_opened":         "Agent opened a container (cabinet, fridge, …)",
    "receptacle_closed":         "Agent closed a container",
    "object_examined":           "Agent examined / looked at an object",
    "object_cleaned":            "An object was cleaned (sink, bathtub)",
    "object_heated":             "An object was heated (microwave, stove)",
    "object_cooled":             "An object was cooled (fridge)",
    "object_toggled":            "A device was turned on / off (lamp, faucet)",
    "task_subtask_completed":    "A sub-goal of the task was completed",
}

EFFECT_TAGS: List[str] = sorted(EFFECT_REGISTRY.keys())

TASK_EFFECT_SUBSET: Dict[str, List[str]] = {
    # ── Classic games ─────────────────────────────────────────────────
    "twenty_forty_eight": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "board_transformed", "board_crowded",
        "tile_promoted", "merge_executed", "direction_applied",
    ],
    "tetris": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "board_transformed", "piece_placed", "piece_changed",
        "line_cleared", "holes_reduced", "holes_created", "move_applied",
    ],
    "candy_crush": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "board_transformed", "board_reshuffled",
        "match_scored", "swap_applied", "move_spent",
    ],
    "sokoban": [
        "state_observed", "action_taken", "reward_positive",
        "cumulative_reward_positive", "score_increased",
        "position_changed", "progress_made",
        "board_transformed",
    ],
    "super_mario": [
        "state_observed", "action_taken", "action_executed",
        "reward_positive", "cumulative_reward_positive", "score_increased",
        "position_changed", "mario_moved", "progress_made",
        "damage_taken", "obstacle_cleared", "collectible_obtained",
    ],
    # ── Text environments ─────────────────────────────────────────────
    "webshop": [
        "state_observed", "action_taken",
        "evidence_cited", "candidates_eliminated",
        "page_navigated", "form_filled", "element_clicked",
        "dom_changed", "item_found", "answer_emitted",
        "search_performed", "product_selected", "cart_updated",
    ],
    "alfworld": [
        "state_observed", "action_taken",
        "reward_positive", "progress_made",
        "location_changed", "object_picked_up", "object_put_down",
        "receptacle_opened", "receptacle_closed", "object_examined",
        "object_cleaned", "object_heated", "object_cooled",
        "object_toggled", "task_subtask_completed",
    ],
}


def get_valid_effects(task_name: str) -> List[str]:
    """Return the closed set of valid effect tags for a given task.

    Falls back to ``EFFECT_TAGS`` (the full global set) if no
    game-specific subset is defined.
    """
    name = task_name.lower().replace("-", "_")
    if name in TASK_EFFECT_SUBSET:
        return TASK_EFFECT_SUBSET[name]
    return EFFECT_TAGS


# ══════════════════════════════════════════════════════════════════════
# TEXT-ENV EFFECT COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_webshop_effects(
    action: str,
    prev_obs: str,
    curr_obs: str,
    reward: float,
) -> Dict[str, str]:
    """Compute effect tags for one WebShop step."""
    effects: Dict[str, str] = {"state_observed": "true", "action_taken": "true"}
    act_l = action.lower()

    if act_l.startswith("search["):
        effects["search_performed"] = "true"
        effects["form_filled"] = "true"
    if act_l.startswith("click["):
        effects["element_clicked"] = "true"

    if prev_obs != curr_obs:
        effects["dom_changed"] = "true"
        effects["page_navigated"] = "true"

    if "buy now" in act_l:
        effects["cart_updated"] = "true"
        effects["answer_emitted"] = "true"

    if any(kw in act_l for kw in ["b0", "asin", "product"]):
        effects["product_selected"] = "true"

    if "back to search" in act_l:
        effects["page_navigated"] = "true"

    if reward > 0:
        effects["reward_positive"] = "true"
        effects["item_found"] = "true"

    return effects


def compute_webshop_structured_state(
    goal: str,
    page_type: str,
    step: int,
    action_history: List[str],
    cumulative_reward: float,
) -> Dict[str, str]:
    """Build a structured state dict for WebShop (predicate-checkable)."""
    search_count = sum(1 for a in action_history if a.lower().startswith("search["))
    click_count = sum(1 for a in action_history if a.lower().startswith("click["))
    return {
        "goal": goal,
        "page": page_type,
        "step": str(step),
        "searches": str(search_count),
        "clicks": str(click_count),
        "actions_taken": str(len(action_history)),
        "cumulative_reward": str(cumulative_reward),
    }


def compute_alfworld_effects(
    action: str,
    prev_obs: str,
    curr_obs: str,
    reward: float,
) -> Dict[str, str]:
    """Compute effect tags for one ALFWorld step."""
    effects: Dict[str, str] = {"state_observed": "true", "action_taken": "true"}
    act_l = action.lower()

    if act_l.startswith("go to "):
        effects["location_changed"] = "true"
    if act_l.startswith("take ") or act_l.startswith("pick up "):
        effects["object_picked_up"] = "true"
    if act_l.startswith("put "):
        effects["object_put_down"] = "true"
    if act_l.startswith("open "):
        effects["receptacle_opened"] = "true"
    if act_l.startswith("close "):
        effects["receptacle_closed"] = "true"
    if act_l.startswith("examine ") or act_l.startswith("look "):
        effects["object_examined"] = "true"
    if act_l.startswith("clean "):
        effects["object_cleaned"] = "true"
    if act_l.startswith("heat ") or act_l.startswith("cook "):
        effects["object_heated"] = "true"
    if act_l.startswith("cool "):
        effects["object_cooled"] = "true"
    if act_l.startswith("use ") or act_l.startswith("toggle ") or act_l.startswith("turn "):
        effects["object_toggled"] = "true"

    if reward > 0:
        effects["reward_positive"] = "true"
        effects["task_subtask_completed"] = "true"
        effects["progress_made"] = "true"

    return effects


def compute_alfworld_structured_state(
    goal: str,
    location: str,
    step: int,
    action_history: List[str],
    inventory: List[str],
    cumulative_reward: float,
) -> Dict[str, str]:
    """Build a structured state dict for ALFWorld (predicate-checkable)."""
    nav_count = sum(1 for a in action_history if a.lower().startswith("go to "))
    pick_count = sum(1 for a in action_history if a.lower().startswith("take ") or a.lower().startswith("pick up "))
    put_count = sum(1 for a in action_history if a.lower().startswith("put "))
    return {
        "goal": goal,
        "location": location,
        "step": str(step),
        "navigations": str(nav_count),
        "pickups": str(pick_count),
        "placements": str(put_count),
        "actions_taken": str(len(action_history)),
        "inventory_count": str(len(inventory)),
        "cumulative_reward": str(cumulative_reward),
    }
