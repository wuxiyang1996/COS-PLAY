"""Shared utilities for protocol-aware skill lifecycle management.

Provides predicate checking against parsed ``summary_state`` dicts and
progress tracking helpers.  Used by ``_SkillTracker`` in both
``scripts/qwen3_decision_agent.py`` and
``trainer/coevolution/episode_runner.py``.
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
