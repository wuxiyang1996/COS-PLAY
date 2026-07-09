"""
Phase detector — identifies game phases from state trajectories.

Per-step intention tags (e.g. [MERGE], [POSITION]) capture *tactical* intent
but not *strategic* context.  Two MERGE actions in different game phases
(opening board vs. endgame survival) represent fundamentally different skills.

This module produces per-step **phase labels** that are combined with intention
tags to create compound skill identifiers: ``"opening:MERGE"`` vs
``"endgame:MERGE"``.

Design:
  - Game-specific extractors parse structured state to produce numeric features
    (board occupancy, progress, etc.) and map them to phase labels.
  - A generic fallback uses sliding-window tag-distribution shifts and temporal
    position to detect phases when state parsing is unavailable.

Usage::

    from skill_agents_grpo.infer_segmentation.phase_detector import detect_phases

    phases = detect_phases(episode.experiences, game_name="twenty_forty_eight")
    # phases[t] = "opening", "midgame", "endgame", etc.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def detect_phases(
    experiences: Sequence,
    game_name: str = "generic",
) -> List[str]:
    """Return a per-step phase label for each experience.

    Tries a game-specific extractor first; falls back to the generic
    temporal/tag-distribution method.

    Parameters
    ----------
    experiences : list
        Episode experiences, each with ``.state`` and ``.intentions``.
    game_name : str
        Canonical game name (e.g. ``"twenty_forty_eight"``).

    Returns
    -------
    list[str]
        Phase label per timestep (e.g. ``"opening"``, ``"midgame"``).
    """
    T = len(experiences)
    if T == 0:
        return []

    extractor = _GAME_EXTRACTORS.get(game_name)
    if extractor is not None:
        try:
            phases = extractor(experiences)
            if phases and len(phases) == T:
                n_unique = len(set(phases))
                if n_unique > 1:
                    return phases
                # Single-phase result from game-specific extractor is
                # uninformative — fall through to generic.
                logger.debug(
                    "%s: game-specific detector found only 1 phase (%s), "
                    "falling back to generic", game_name, phases[0],
                )
        except Exception:
            logger.debug("Game-specific phase detection failed for %s, "
                         "falling back to generic", game_name, exc_info=True)

    return _generic_phase_detector(experiences)


# ═════════════════════════════════════════════════════════════════════
# Game-specific extractors
# ═════════════════════════════════════════════════════════════════════

def _extract_2048_phases(experiences: Sequence) -> List[str]:
    """2048: phase by board occupancy and highest tile."""
    phases = []
    for exp in experiences:
        state = _get_state_dict(exp)
        if state is None:
            phases.append("mid")
            continue

        board = state.get("board")
        highest = state.get("highest_tile", 0)

        empty = 16
        if board:
            empty = sum(1 for row in board for cell in row if cell == 0)
        occupancy = 1.0 - (empty / 16.0)

        if occupancy < 0.35 and highest <= 32:
            phases.append("opening")
        elif occupancy > 0.7 or highest >= 256:
            phases.append("endgame")
        else:
            phases.append("midgame")

    return phases


def _extract_tetris_phases(experiences: Sequence) -> List[str]:
    """Tetris: phase by board height and density.

    Uses ``summary_state`` (``stack_h=N``) when the raw state is a text
    board (not a parseable dict).  Assumes a 20-row board for thresholds.
    """
    phases = []
    for exp in experiences:
        # Try structured dict first (e.g. JSON state with "board" key)
        state = _get_state_dict(exp)
        if state is not None:
            board = state.get("board", state.get("grid"))
            if board and isinstance(board, list):
                n_rows = len(board)
                filled_rows = sum(
                    1 for row in board
                    if any(cell not in (0, None, "", " ") for cell in row)
                )
                fill_ratio = filled_rows / max(n_rows, 1)
                if fill_ratio < 0.25:
                    phases.append("opening")
                elif fill_ratio > 0.65:
                    phases.append("endgame")
                else:
                    phases.append("midgame")
                continue

        # Fall back to summary_state which has stack_h=N
        summary = _get_summary_str(exp)
        m = re.search(r'stack_h=(\d+)', summary)
        if m:
            stack_h = int(m.group(1))
            if stack_h <= 4:
                phases.append("opening")
            elif stack_h >= 14:
                phases.append("endgame")
            else:
                phases.append("midgame")
        else:
            phases.append("midgame")

    return phases


def _extract_mario_phases(experiences: Sequence) -> List[str]:
    """Super Mario: phase by position progress (x-coordinate).

    Tries both ``summary_state`` (``mario=(x,y)``) and raw state
    (``Position of Mario: (x, y)``).  If x-positions don't vary
    (short episodes), falls back to temporal thirds.
    """
    positions = []
    for exp in experiences:
        x = None
        summary = _get_summary_str(exp)
        m = re.search(r'mario=\((\d+)', summary)
        if m:
            x = int(m.group(1))
        else:
            state_str = _get_state_str(exp)
            m = re.search(r'Position of Mario:\s*\((\d+)', state_str)
            if m:
                x = int(m.group(1))
        positions.append(x)

    valid = [p for p in positions if p is not None]
    if not valid:
        return _generic_phase_detector(experiences)

    min_x, max_x = min(valid), max(valid)
    span = max_x - min_x
    if span < 20:
        return _temporal_thirds(len(experiences))

    phases = []
    for p in positions:
        if p is None:
            phases.append("mid")
        else:
            progress = (p - min_x) / span
            if progress < 0.33:
                phases.append("early_level")
            elif progress < 0.66:
                phases.append("mid_level")
            else:
                phases.append("late_level")
    return phases


def _extract_sokoban_phases(experiences: Sequence) -> List[str]:
    """Sokoban: phase by boxes-on-goals progress.

    Uses ``summary_state`` (``solved=M/N | boxes=K``) for reliable
    progress tracking.  The raw state is an item table that lacks
    "dark_storage"/"box_on_goal" labels.
    """
    phases = []
    for exp in experiences:
        summary = _get_summary_str(exp)
        solved_m = re.search(r'solved=(\d+)/(\d+)', summary)
        if solved_m:
            solved, total = int(solved_m.group(1)), int(solved_m.group(2))
            if total == 0:
                phases.append("explore")
            elif solved == 0:
                phases.append("setup")
            elif solved < total:
                phases.append("solving")
            else:
                phases.append("finishing")
            continue

        state_str = _get_state_str(exp).lower()
        n_dark = state_str.count("dark_storage") + state_str.count("box_on_goal")
        n_box = state_str.count("box")
        if n_box == 0:
            phases.append("explore")
        elif n_dark == 0:
            phases.append("setup")
        elif n_dark < n_box:
            phases.append("solving")
        else:
            phases.append("finishing")
    return phases


def _extract_candy_phases(experiences: Sequence) -> List[str]:
    """Candy Crush: phase by temporal progress (no strong state signal)."""
    T = len(experiences)
    return _temporal_thirds(T)


def _extract_avalon_phases(experiences: Sequence) -> List[str]:
    """Avalon: phase from quest number in summary_state.

    Uses ``summary_state`` (``quest=N``) to avoid false positives from
    the raw state which always contains keywords like "team" and "quest".
    """
    phases = []
    for exp in experiences:
        summary = _get_summary_str(exp)
        m = re.search(r'quest=(\d+)', summary)
        if m:
            quest = int(m.group(1))
            if quest <= 2:
                phases.append("early_quests")
            elif quest <= 4:
                phases.append("mid_quests")
            else:
                phases.append("final_quest")
        else:
            low = _get_state_str(exp).lower()
            if "assassin" in low or "accuse" in low or "reveal" in low:
                phases.append("endgame")
            elif "vote" in low or "proposal" in low:
                phases.append("team_building")
            elif "quest" in low or "mission" in low:
                phases.append("quest")
            else:
                phases.append("discussion")
    return phases


def _extract_diplomacy_phases(experiences: Sequence) -> List[str]:
    """Diplomacy: phase from abbreviated phase codes (S1901M, F1902R, W1903A).

    Uses ``summary_state`` which has ``phase=S1901M`` format.  Falls back
    to keyword matching on the raw state for non-standard formats.
    S=Spring, F=Fall, W=Winter; M=Movement, R=Retreat, A=Adjustment.
    """
    phases = []
    for exp in experiences:
        summary = _get_summary_str(exp)
        m = re.search(r'phase=([SFW])(\d{4})([MRA])', summary)
        if not m:
            m = re.search(r'Phase:\s*([SFW])(\d{4})([MRA])', _get_state_str(exp))
        if m:
            season, year, phase_type = m.group(1), int(m.group(2)), m.group(3)
            if phase_type == "R":
                phases.append("retreat")
            elif phase_type == "A":
                phases.append("adjustment")
            elif year <= 1902 and season == "S":
                phases.append("opening")
            elif year >= 1905:
                phases.append("late_orders")
            else:
                phases.append("orders")
        else:
            low = _get_state_str(exp).lower()
            if "spring" in low and ("1901" in low or "1902" in low):
                phases.append("opening")
            elif "retreat" in low or "disband" in low:
                phases.append("retreat")
            elif "build" in low or "adjust" in low:
                phases.append("adjustment")
            else:
                phases.append("orders")
    return phases


def _extract_pokemon_phases(experiences: Sequence) -> List[str]:
    """Pokemon Red: phase from context (battle, overworld, menu).

    Checks both ``summary_state`` and raw ``state`` for ``State: X``
    (Battle, Field, Dialog, Menu).  The raw state always starts with
    ``State: <context>`` while the summary may omit it after early steps.
    Uses ``\\bState:\\s*(\\w+)`` to avoid matching "Not in battle".
    """
    phases = []
    for exp in experiences:
        ctx = None
        for source in (_get_summary_str(exp), _get_state_str(exp)):
            m = re.search(r'\bState:\s*(\w+)', source)
            if m:
                ctx = m.group(1).lower()
                break

        if ctx:
            if ctx == "battle":
                phases.append("battle")
            elif ctx in ("menu", "shop"):
                phases.append("menu")
            elif ctx == "dialog":
                phases.append("dialog")
            else:
                phases.append("overworld")
        else:
            phases.append("overworld")
    return phases


# ═════════════════════════════════════════════════════════════════════
# Generic fallback
# ═════════════════════════════════════════════════════════════════════

def _generic_phase_detector(experiences: Sequence) -> List[str]:
    """Fallback: combine temporal thirds with tag-distribution windows.

    Splits episode into temporal thirds (early/mid/late).  If the
    tag distribution in each third differs significantly, uses temporal
    phase labels; otherwise returns uniform "mid" (no phase signal).
    """
    T = len(experiences)
    if T < 6:
        return _temporal_thirds(T)

    tags = []
    for exp in experiences:
        if isinstance(exp, dict):
            intent = exp.get("intentions")
        else:
            intent = getattr(exp, "intentions", None)
        tag = "UNKNOWN"
        if intent:
            m = re.match(r"\[(\w+)\]", str(intent))
            if m:
                tag = m.group(1)
        tags.append(tag)

    thirds = [
        tags[:T // 3],
        tags[T // 3: 2 * T // 3],
        tags[2 * T // 3:],
    ]
    counters = [Counter(t) for t in thirds]

    if _distributions_differ(counters):
        return _temporal_thirds(T)

    return ["mid"] * T


def _distributions_differ(counters: List[Counter]) -> bool:
    """Check if tag distributions across thirds are meaningfully different."""
    all_tags = set()
    for c in counters:
        all_tags.update(k for k in c if k != "UNKNOWN")
    if len(all_tags) < 2:
        return False

    proportions = []
    for c in counters:
        total = sum(v for k, v in c.items() if k != "UNKNOWN") or 1
        proportions.append({t: c.get(t, 0) / total for t in all_tags})

    for tag in all_tags:
        vals = [p.get(tag, 0) for p in proportions]
        if max(vals) - min(vals) > 0.20:
            return True
    return False


def _temporal_thirds(T: int) -> List[str]:
    """Simple temporal third labels."""
    phases = []
    for t in range(T):
        progress = t / max(T - 1, 1)
        if progress < 0.33:
            phases.append("early")
        elif progress < 0.66:
            phases.append("mid")
        else:
            phases.append("late")
    return phases


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _get_state_dict(exp) -> Optional[dict]:
    """Try to get state as a dict (for games with structured state).

    Handles dicts, JSON strings, and Python-repr strings (single quotes).
    """
    state = getattr(exp, "state", None)
    if state is None and isinstance(exp, dict):
        state = exp.get("state")
    if isinstance(state, dict):
        return state
    if isinstance(state, str):
        try:
            return json.loads(state)
        except (json.JSONDecodeError, ValueError):
            pass
        import ast
        try:
            val = ast.literal_eval(state)
            if isinstance(val, dict):
                return val
        except (ValueError, SyntaxError):
            pass
    return None


def _get_state_str(exp) -> str:
    """Get state as a string for text-based matching."""
    if isinstance(exp, dict):
        state = exp.get("state") or exp.get("summary_state")
    else:
        state = getattr(exp, "state", None)
        if state is None:
            state = getattr(exp, "summary_state", None)
    if isinstance(state, dict):
        return json.dumps(state, default=str)
    return str(state) if state else ""


def _get_summary_str(exp) -> str:
    """Get the structured ``key=value`` summary_state string.

    Prefers ``summary_state`` (parsed, deterministic) over the raw
    ``state`` (verbose game text).  Phase detectors that match on
    parsed fields (phase codes, quest numbers, stack_h, etc.) should
    use this instead of ``_get_state_str``.
    """
    if isinstance(exp, dict):
        s = exp.get("summary_state") or exp.get("state")
    else:
        s = getattr(exp, "summary_state", None)
        if s is None:
            s = getattr(exp, "state", None)
    if isinstance(s, dict):
        return json.dumps(s, default=str)
    return str(s) if s else ""


def make_compound_label(phase: str, tag: str) -> str:
    """Combine phase and intention tag into a compound skill label.

    Returns ``"phase:tag"`` or just ``tag`` when phase is uninformative.
    """
    if not phase or phase == "mid" or phase == "UNKNOWN":
        return tag
    if not tag or tag == "UNKNOWN":
        return phase
    return f"{phase}:{tag}"


_GAME_EXTRACTORS: Dict[str, Any] = {
    "twenty_forty_eight": _extract_2048_phases,
    "2048": _extract_2048_phases,
    "tetris": _extract_tetris_phases,
    "super_mario": _extract_mario_phases,
    "sokoban": _extract_sokoban_phases,
    "candy_crush": _extract_candy_phases,
    "avalon": _extract_avalon_phases,
    "diplomacy": _extract_diplomacy_phases,
    "pokemon_red": _extract_pokemon_phases,
    "pokemon": _extract_pokemon_phases,
}
