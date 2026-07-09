"""Game-domain tool implementations for Gym-V environments.

Each tool handler takes arguments from the VLM's function call and
returns structured data by querying the environment's ground-truth state
(obs.text, grid parsing, env metadata).  The VLM never needs to
hallucinate coordinates or relations -- it calls these tools instead.

Usage::

    from gymv_wrapper.tools import build_gymv_registry

    registry = build_gymv_registry(obs_text=obs.text, description=env.description)
"""

from __future__ import annotations

import re
from typing import Any

from vlm_wrapper.tools import (
    TOOL_CHECK_RELATION,
    TOOL_GET_STATE_FLAGS,
    TOOL_LIST_ENTITIES,
    TOOL_LIST_VALID_ACTIONS,
    TOOL_QUERY_ENTITY_POS,
    ToolDef,
    ToolRegistry,
)
from gymv_wrapper.heuristic import (
    _entities_from_grid,
    _extract_state_flags,
    _infer_actions,
    _try_parse_grid,
)

TOOL_GET_GRID = ToolDef(
    name="get_grid_state",
    description=(
        "Return the full game grid as a structured 2D array. Each cell "
        "contains the entity type or value (e.g. '2', '16', '#', '@', '$'). "
        "Only available for grid-based games (2048, Sokoban, Minesweeper, etc.)."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="gymv",
)

TOOL_CHECK_DEADLOCK = ToolDef(
    name="check_deadlock",
    description=(
        "Check if the current game state has any deadlocked entities "
        "(e.g. a Sokoban box stuck in a corner). Returns a list of "
        "deadlocked entity positions with explanations."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="gymv",
)

TOOL_SPATIAL_ANALYSIS = ToolDef(
    name="spatial_analysis",
    description=(
        "Compute spatial relationships between entities: Manhattan "
        "distances, nearest neighbors, alignment (same row/column). "
        "Useful for planning moves in grid games."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity_label": {
                "type": "string",
                "description": "Focus entity label (e.g. 'player', 'box'). If omitted, analyzes all.",
            },
        },
        "required": [],
    },
    domain="gymv",
)

TOOL_COUNT_MERGE_CANDIDATES = ToolDef(
    name="count_merge_candidates",
    description=(
        "For merge-based games (2048, Threes), count how many adjacent "
        "pairs of tiles can merge. Returns pairs with their values and "
        "the direction to merge."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="gymv",
)


class _GymVState:
    """Lazy-parsed game state backing all tool handlers."""

    def __init__(self, obs_text: str, description: str, step: int = 0):
        self.obs_text = obs_text
        self.description = description
        self.step = step
        self._grid: list[list[str]] | None = ...  # sentinel
        self._entities: list[dict] | None = None
        self._attributes: list[dict] | None = None
        self._relations: list[str] | None = None

    @property
    def grid(self) -> list[list[str]] | None:
        if self._grid is ...:
            self._grid = _try_parse_grid(self.obs_text)
        return self._grid

    def _ensure_parsed(self) -> None:
        if self._entities is not None:
            return
        if self.grid is not None:
            self._entities, self._attributes, self._relations = (
                _entities_from_grid(self.grid, max_entities=30)
            )
        else:
            self._entities, self._attributes, self._relations = [], [], []

    @property
    def entities(self) -> list[dict]:
        self._ensure_parsed()
        return self._entities  # type: ignore[return-value]

    @property
    def attributes(self) -> list[dict]:
        self._ensure_parsed()
        return self._attributes  # type: ignore[return-value]

    @property
    def relations(self) -> list[str]:
        self._ensure_parsed()
        return self._relations  # type: ignore[return-value]


def _h_query_entity_pos(state: _GymVState, *, entity_label: str) -> dict:
    label_lower = entity_label.lower().strip()
    matches = []
    for e in state.entities:
        el = e["label"].lower()
        if label_lower in el or el in label_lower:
            matches.append({
                "eid": e["eid"],
                "label": e["label"],
                "pos": e.get("pos"),
                "type": e["type"],
            })
    if not matches:
        return {"found": False, "message": f"No entity matching '{entity_label}'"}
    return {"found": True, "matches": matches}


def _h_list_entities(
    state: _GymVState, *, filter_type: str = "all", max_results: int = 25,
) -> dict:
    ents = state.entities
    if filter_type != "all":
        ents = [e for e in ents if e["type"] == filter_type]
    ents = ents[:max_results]
    result = []
    for e in ents:
        entry: dict[str, Any] = {
            "eid": e["eid"],
            "type": e["type"],
            "label": e["label"],
        }
        if e.get("pos"):
            entry["pos"] = e["pos"]
        for a in state.attributes:
            if a["eid"] == e["eid"]:
                entry[a["key"]] = a["val"]
        result.append(entry)
    return {"count": len(result), "entities": result}


def _h_check_relation(
    state: _GymVState, *, entity_a: str, entity_b: str, relation: str,
) -> dict:
    ea = _resolve_entity(state, entity_a)
    eb = _resolve_entity(state, entity_b)
    if ea is None or eb is None:
        return {
            "holds": False,
            "reason": f"Could not resolve {'entity_a' if ea is None else 'entity_b'}",
        }

    pos_a = _parse_pos(ea.get("pos"))
    pos_b = _parse_pos(eb.get("pos"))

    if relation == "adjacent":
        if pos_a and pos_b:
            dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
            return {"holds": dist == 1, "manhattan_distance": dist}
        return {"holds": False, "reason": "positions unknown"}

    if relation == "same_row":
        if pos_a and pos_b:
            return {"holds": pos_a[0] == pos_b[0], "row_a": pos_a[0], "row_b": pos_b[0]}
        return {"holds": False, "reason": "positions unknown"}

    if relation == "same_column":
        if pos_a and pos_b:
            return {"holds": pos_a[1] == pos_b[1], "col_a": pos_a[1], "col_b": pos_b[1]}
        return {"holds": False, "reason": "positions unknown"}

    if relation == "blocks":
        return _check_blocks(state, ea, eb)

    if relation == "merge_candidate":
        return _check_merge(state, ea, eb)

    for rel_str in state.relations:
        if ea["eid"] in rel_str and eb["eid"] in rel_str and relation in rel_str:
            return {"holds": True, "source": "parsed_relations"}

    return {"holds": False, "reason": f"Relation '{relation}' not determinable"}


def _h_get_state_flags(state: _GymVState) -> dict:
    progress, phase, error = _extract_state_flags(state.obs_text, state.step)
    return {
        "progress": progress,
        "phase": phase,
        "error": error,
        "dialog_open": False,
        "input_pending": False,
    }


def _h_list_valid_actions(state: _GymVState) -> dict:
    actions = _infer_actions(state.description)
    return {"actions": actions, "count": len(actions)}


def _h_get_grid(state: _GymVState) -> dict:
    if state.grid is None:
        return {"available": False, "message": "Not a grid-based game"}
    return {
        "available": True,
        "rows": len(state.grid),
        "cols": max(len(r) for r in state.grid) if state.grid else 0,
        "grid": state.grid,
    }


def _h_check_deadlock(state: _GymVState) -> dict:
    if state.grid is None:
        return {"available": False, "message": "Not a grid-based game"}

    deadlocked: list[dict] = []
    rows, cols = len(state.grid), max(len(r) for r in state.grid)

    for r, row in enumerate(state.grid):
        for c, cell in enumerate(row):
            if cell not in ("$", "B", "*"):
                continue
            is_corner = _is_corner_deadlock(state.grid, r, c, rows, cols)
            if is_corner:
                deadlocked.append({
                    "pos": f"{r},{c}",
                    "label": cell,
                    "reason": "corner deadlock — walls on two perpendicular sides",
                })

    return {"deadlocked": deadlocked, "count": len(deadlocked)}


def _h_spatial_analysis(state: _GymVState, *, entity_label: str = "") -> dict:
    positioned = []
    for e in state.entities:
        pos = _parse_pos(e.get("pos"))
        if pos:
            positioned.append({"eid": e["eid"], "label": e["label"], "r": pos[0], "c": pos[1]})

    if entity_label:
        focus = [p for p in positioned if entity_label.lower() in p["label"].lower()]
    else:
        focus = positioned

    result: list[dict] = []
    for f in focus:
        neighbors = []
        for other in positioned:
            if other["eid"] == f["eid"]:
                continue
            dist = abs(f["r"] - other["r"]) + abs(f["c"] - other["c"])
            neighbors.append({
                "eid": other["eid"],
                "label": other["label"],
                "manhattan_distance": dist,
                "same_row": f["r"] == other["r"],
                "same_col": f["c"] == other["c"],
            })
        neighbors.sort(key=lambda x: x["manhattan_distance"])
        result.append({
            "eid": f["eid"],
            "label": f["label"],
            "pos": f"{f['r']},{f['c']}",
            "nearest_3": neighbors[:3],
        })

    return {"entities": result, "count": len(result)}


def _h_count_merge_candidates(state: _GymVState) -> dict:
    if state.grid is None:
        return {"available": False, "message": "Not a grid-based game"}

    pairs: list[dict] = []
    rows = len(state.grid)
    for r in range(rows):
        cols = len(state.grid[r])
        for c in range(cols):
            val = state.grid[r][c]
            if not re.fullmatch(r"\d+", val) or val == "0":
                continue
            if c + 1 < cols and state.grid[r][c + 1] == val:
                pairs.append({"a": f"{r},{c}", "b": f"{r},{c+1}", "value": val, "direction": "right"})
            if r + 1 < rows and c < len(state.grid[r + 1]) and state.grid[r + 1][c] == val:
                pairs.append({"a": f"{r},{c}", "b": f"{r+1},{c}", "value": val, "direction": "down"})

    return {"pairs": pairs, "count": len(pairs)}


def _resolve_entity(state: _GymVState, ref: str) -> dict | None:
    ref_lower = ref.lower().strip()
    if ref_lower.startswith("e") and ref_lower[1:].isdigit():
        for e in state.entities:
            if e["eid"] == ref_lower:
                return e
    for e in state.entities:
        if ref_lower in e["label"].lower():
            return e
    return None


def _parse_pos(pos_str: str | None) -> tuple[int, int] | None:
    if not pos_str:
        return None
    parts = pos_str.split(",")
    if len(parts) >= 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return None
    return None


def _is_corner_deadlock(
    grid: list[list[str]], r: int, c: int, rows: int, cols: int,
) -> bool:
    def is_wall(rr: int, cc: int) -> bool:
        if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
            return True
        return grid[rr][cc] in ("#", "X")

    up = is_wall(r - 1, c)
    down = is_wall(r + 1, c)
    left = is_wall(r, c - 1)
    right = is_wall(r, c + 1)
    return (up and left) or (up and right) or (down and left) or (down and right)


def _check_blocks(state: _GymVState, ea: dict, eb: dict) -> dict:
    pos_a = _parse_pos(ea.get("pos"))
    pos_b = _parse_pos(eb.get("pos"))
    if not (pos_a and pos_b):
        return {"holds": False, "reason": "positions unknown"}
    if abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1:
        if ea["label"] in ("wall", "#", "X"):
            return {"holds": True, "reason": f"{ea['label']} is adjacent wall"}
    return {"holds": False, "reason": "no blocking relationship detected"}


def _check_merge(state: _GymVState, ea: dict, eb: dict) -> dict:
    pos_a = _parse_pos(ea.get("pos"))
    pos_b = _parse_pos(eb.get("pos"))
    if not (pos_a and pos_b):
        return {"holds": False, "reason": "positions unknown"}
    dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
    val_a = _get_attr(state, ea["eid"], "value")
    val_b = _get_attr(state, eb["eid"], "value")
    if val_a and val_b and val_a == val_b and dist == 1:
        return {"holds": True, "value": val_a, "manhattan_distance": dist}
    return {
        "holds": False,
        "values": [val_a, val_b],
        "manhattan_distance": dist,
        "reason": "different values or not adjacent",
    }


def _get_attr(state: _GymVState, eid: str, key: str) -> str | None:
    for a in state.attributes:
        if a["eid"] == eid and a["key"] == key:
            return a["val"]
    return None


def build_gymv_registry(
    obs_text: str = "",
    description: str = "",
    step: int = 0,
) -> ToolRegistry:
    state = _GymVState(obs_text, description, step)
    reg = ToolRegistry(domain="gymv")

    reg.register(TOOL_QUERY_ENTITY_POS, lambda **kw: _h_query_entity_pos(state, **kw))
    reg.register(TOOL_LIST_ENTITIES, lambda **kw: _h_list_entities(state, **kw))
    reg.register(TOOL_CHECK_RELATION, lambda **kw: _h_check_relation(state, **kw))
    reg.register(TOOL_GET_STATE_FLAGS, lambda **kw: _h_get_state_flags(state))
    reg.register(TOOL_LIST_VALID_ACTIONS, lambda **kw: _h_list_valid_actions(state))
    reg.register(TOOL_GET_GRID, lambda **kw: _h_get_grid(state))
    reg.register(TOOL_CHECK_DEADLOCK, lambda **kw: _h_check_deadlock(state))
    reg.register(TOOL_SPATIAL_ANALYSIS, lambda **kw: _h_spatial_analysis(state, **kw))
    reg.register(TOOL_COUNT_MERGE_CANDIDATES, lambda **kw: _h_count_merge_candidates(state))

    return reg
