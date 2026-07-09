"""Deterministic per-game canonical schema generators.

Why this exists
---------------
The LLM-generated ``schema_text_llm`` and ``schema_image_llm`` blocks carry
free-form labels, free-form abstraction granularity (e.g. "the 14 empty
cells" vs. one-entity-per-empty-cell), and pick up rendering HUD overlays
("Perf 0.00", "score text", "world text") only on the visual side. As a
result the two schemas overlap by 0 – 30 % entity-IoU on
2048 / Candy Crush / Tetris / Super Mario — they cannot be used as a stable
state representation for skill learning.

This module produces a **deterministic, modality-invariant** ``<state>``
block per game, derived directly from the env's structured state
(``info`` and ``info['raw_obs']['text']``). Two key invariants:

* **Stable IDs.** Entities are named by (canonical_type, row, col) or
  (canonical_type, x, y). The same world state always yields the same
  schema string, byte-for-byte.
* **Stable labels & units.** Each game has a fixed label vocabulary
  (``CANONICAL_LABELS``) and a fixed positional unit
  (``CANONICAL_POSITION_UNITS``). HUD overlays are excluded.

The output is a string in the same ``<state>``…``</state>`` format as
``vlm_wrapper.schema`` so existing parsers/validators apply unchanged.

Usage
-----
>>> from visual_grounding_tests.canonical_schema import make_canonical_schema
>>> sch = make_canonical_schema(
...     game="twenty_forty_eight",
...     info=info,
...     task_id="make_gaming_env/twenty_forty_eight",
...     goal="Play 2048…",
...     step=0,
... )
>>> print(sch)
<state>
domain=gymv
task=make_gaming_env/twenty_forty_eight
goal=Play 2048…
step=0
…
</state>
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Public spec: canonical labels and position units per game.
#
# These are the ground-truth label sets that BOTH the canonical
# generator AND the LLM/VLM prompts should use. They are documented
# here so a future skill learner has one authoritative reference.
# ─────────────────────────────────────────────────────────────────────

CANONICAL_LABELS: Dict[str, Dict[str, str]] = {
    "twenty_forty_eight": {
        "board": "board",
        "tile": "tile_{value}",
        "empty_cells": "empty_cells",
        "score": "score",
        "highest_tile": "highest_tile",
    },
    "candy_crush": {
        "board": "board",
        "candy": "candy_{color}",
        "score": "score",
        "moves_left": "moves_left",
    },
    "tetris": {
        "playfield": "playfield",
        "active_piece": "active_piece_{kind}",
        "stack_block": "stack_block_{kind}",
        "next_piece": "next_piece_{kind}",
        "score": "score",
        "level": "level",
        "lines": "lines",
    },
    "super_mario": {
        "mario": "mario",
        "brick": "brick",
        "question_block": "question_block",
        "inactivated_block": "inactivated_block",
        "goomba": "goomba",
        "koopa": "koopa",
        "pit_start": "pit_start",
        "pit_end": "pit_end",
        "warp_pipe": "warp_pipe",
        "item_mushroom": "item_mushroom",
        "stair_block": "stair_block",
        "flag": "flag",
    },
}

CANONICAL_POSITION_UNITS: Dict[str, str] = {
    "twenty_forty_eight": "grid (row, col, h, w) where row ∈ [0,4), col ∈ [0,4)",
    "candy_crush":        "grid (row, col, h, w) where row ∈ [0,8), col ∈ [0,8)",
    "tetris":             "grid (row, col, h, w) where row ∈ [0,20), col ∈ [0,10)",
    # Orak's super_mario env emits (x, y) with the y-axis flipped: y=0 is
    # the BOTTOM of the screen (the ground), y=240 is the top. This is why
    # the text obs reports Mario at y≈45 even though Mario is visually
    # near the bottom of the rendered image. Both modalities must use
    # this convention so the canonical and LLM schemas line up.
    "super_mario":        ("Orak pixels (x, y, w, h); x ∈ [0,256] from left, "
                            "y ∈ [0,240] from BOTTOM of screen (y=0 is ground "
                            "level, y=240 is top of sky). Top-left corner of "
                            "each object."),
}

# Exclude these entity-label tokens from VLM/LLM output: HUD overlays
# rendered on top of the game frame are not part of the game state.
HUD_BLOCKLIST: Dict[str, List[str]] = {
    "twenty_forty_eight": ["perf", "perf score", "perf 0.00"],
    "candy_crush":        ["perf"],
    "tetris":             ["next label", "stats text", "perf"],
    "super_mario":        ["score text", "world text", "time text", "coin text",
                           "score:", "world", "time", "coin"],
}


# ─────────────────────────────────────────────────────────────────────
# Tetromino / piece id mapping (from GamingAgent tetris env)
# ─────────────────────────────────────────────────────────────────────

_TETRIS_PIECE_BY_ID = {0: "I", 1: "O", 2: "T", 3: "S", 4: "Z", 5: "J", 6: "L", 7: "I"}


# ─────────────────────────────────────────────────────────────────────
# Parsers — each returns plain dicts of "world facts" the schema
# builder can consume. Kept independent of the schema serialiser so
# they're easy to unit-test.
# ─────────────────────────────────────────────────────────────────────


def _parse_2048(info: Dict[str, Any]) -> Dict[str, Any]:
    """2048 board, normalised to **displayed tile values** (2, 4, 8 …).

    Two encodings exist in this repo: ``info['board']`` is the GamingAgent
    log2-power form (1 → tile 2, 2 → tile 4 …) while ``raw_obs['text']``
    holds the value form (2, 4, 8 …). The user-facing schema uses the
    value form, so we prefer the text-parsed dict when available.
    """
    board: Optional[List[List[int]]] = None
    raw_text = ((info.get("raw_obs") or {}).get("text") or "").strip()
    if raw_text.startswith("{"):
        try:
            parsed = ast.literal_eval(raw_text)
            cand = parsed.get("board")
            if isinstance(cand, list):
                board = [list(row) for row in cand]
        except Exception:
            board = None
    if board is None:
        b = info.get("board")
        if b is not None:
            try:
                arr = [list(row) for row in b]
                # Heuristic: GamingAgent's info['board'] is log2-power form,
                # so values are small (≤ ~16). Convert by 2**v when v > 0.
                board = [[(1 << int(v)) if int(v) > 0 else 0 for v in row]
                         for row in arr]
            except Exception:
                board = None
    if not isinstance(board, list) or not board:
        return {"rows": 0, "cols": 0, "cells": [],
                "highest": None, "score": None, "step_score": None}
    h = len(board)
    w = len(board[0]) if h else 0
    cells: List[Tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            try:
                v = int(board[r][c])
            except (TypeError, ValueError):
                v = 0
            cells.append((r, c, v))
    highest = max((v for _, _, v in cells), default=0) or None
    return {
        "rows": h,
        "cols": w,
        "cells": cells,
        "highest": highest,
        "score": info.get("total_score"),
        "step_score": info.get("step_score"),
    }


_CC_ROW_RE = re.compile(r"^\s*(\d+)\s*\|\s*([A-Z\s]+)\s*$")
_CC_SCORE_RE = re.compile(r"^Score:\s*(\d+)", re.M)
_CC_MOVES_RE = re.compile(r"^Moves\s*Left:\s*(\d+)", re.M)
_CC_COLOR_NAMES = {"R": "red", "G": "green", "C": "cyan", "P": "purple",
                   "Y": "yellow", "O": "orange", "B": "blue"}


def _parse_candy_crush(info: Dict[str, Any]) -> Dict[str, Any]:
    text = (info.get("raw_obs") or {}).get("text") or info.get("textual_representation") or ""
    cells: List[Tuple[int, int, str]] = []
    rows = 0
    cols = 0
    for line in text.splitlines():
        m = _CC_ROW_RE.match(line)
        if not m:
            continue
        r = int(m.group(1))
        letters = m.group(2).split()
        rows = max(rows, r + 1)
        cols = max(cols, len(letters))
        for c, ltr in enumerate(letters):
            cells.append((r, c, ltr.upper()))
    score_match = _CC_SCORE_RE.search(text)
    moves_match = _CC_MOVES_RE.search(text)
    return {
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "score": float(score_match.group(1)) if score_match else info.get("score"),
        "moves_left": int(moves_match.group(1)) if moves_match else info.get("num_moves_left"),
    }


_TETRIS_ROW_RE = re.compile(r"^([\.IOTSZJL]{10})$")
_TETRIS_NEXT_RE = re.compile(r"^Next Pieces:\s*(.+)$", re.M)
_TETRIS_STATS_RE = re.compile(
    r"PerfS:\s*([\-\d\.]+)\s+S:\s*([\-\d\.]+)\s+L:\s*(\d+)\s+Lv:\s*(\d+)"
)


def _parse_tetris(info: Dict[str, Any]) -> Dict[str, Any]:
    text = (info.get("raw_obs") or {}).get("text") or info.get("textual_representation") or ""
    rows: List[str] = []
    started = False
    for line in text.splitlines():
        if line.strip().lower().startswith("board:"):
            started = True
            continue
        if not started:
            continue
        if _TETRIS_ROW_RE.match(line.strip()):
            rows.append(line.strip())
            if len(rows) == 20:
                break
        else:
            if rows:
                break

    cells: List[Tuple[int, int, str]] = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != ".":
                cells.append((r, c, ch))

    active_cells: List[Tuple[int, int, str]] = []
    if cells:
        topmost_r = min(r for r, _, _ in cells)
        active_kind = next(ch for r, _, ch in cells if r == topmost_r)
        active_cells = [(r, c, ch) for r, c, ch in cells
                        if ch == active_kind and r <= topmost_r + 3]
        active_set = set((r, c) for r, c, _ in active_cells)
    else:
        active_set = set()
    stack_cells = [(r, c, ch) for r, c, ch in cells if (r, c) not in active_set]

    next_ids = info.get("next_piece_ids") or []
    next_kinds = [_TETRIS_PIECE_BY_ID.get(int(i), "?") for i in next_ids]
    if not next_kinds:
        m = _TETRIS_NEXT_RE.search(text)
        if m:
            next_kinds = [s.strip() for s in m.group(1).split(",")]

    score = info.get("score")
    level = info.get("level")
    lines = info.get("lines")
    if score is None or level is None or lines is None:
        m = _TETRIS_STATS_RE.search(text)
        if m:
            score = score or float(m.group(2))
            level = level or int(m.group(4))
            lines = lines or int(m.group(3))

    return {
        "rows": 20,
        "cols": 10,
        "active_kind": active_cells[0][2] if active_cells else None,
        "active_cells": active_cells,
        "stack_cells": stack_cells,
        "next_kinds": next_kinds,
        "score": score,
        "level": level,
        "lines": lines,
    }


_MARIO_MARIO_RE = re.compile(r"Position of Mario:\s*\((-?\d+),\s*(-?\d+)\)")
_MARIO_OBJ_RE = re.compile(r"^-\s*([^:]+):\s*(.+)$", re.M)
_MARIO_COORD_RE = re.compile(r"\((-?\d+),\s*(-?\d+)\)")


def _parse_mario(info: Dict[str, Any]) -> Dict[str, Any]:
    text = (info.get("raw_obs") or {}).get("text") or info.get("textual_representation") or ""
    mario_xy: Optional[Tuple[int, int]] = None
    m = _MARIO_MARIO_RE.search(text)
    if m:
        mario_xy = (int(m.group(1)), int(m.group(2)))
    objects: Dict[str, List[Tuple[int, int]]] = {}
    section = text.split("Positions of all objects", 1)
    body = section[1] if len(section) == 2 else text
    label_to_canonical = {
        "bricks": "brick",
        "question blocks": "question_block",
        "inactivated blocks": "inactivated_block",
        "monster goomba": "goomba",
        "monster koopas": "koopa",
        "pit": "pit",
        "warp pipe": "warp_pipe",
        "item mushrooms": "item_mushroom",
        "stair blocks": "stair_block",
        "flag": "flag",
    }
    for mm in _MARIO_OBJ_RE.finditer(body):
        raw_label = mm.group(1).strip().lower()
        rest = mm.group(2).strip()
        if "none" in rest.lower() and not _MARIO_COORD_RE.search(rest):
            continue
        if raw_label == "pit":
            objects.setdefault("pit_start", [])
            objects.setdefault("pit_end", [])
            sm = re.search(r"start at\s*\((-?\d+),\s*(-?\d+)\)", rest)
            em = re.search(r"end at\s*\((-?\d+),\s*(-?\d+)\)", rest)
            if sm:
                objects["pit_start"].append((int(sm.group(1)), int(sm.group(2))))
            if em:
                objects["pit_end"].append((int(em.group(1)), int(em.group(2))))
            continue
        canon = label_to_canonical.get(raw_label)
        if not canon:
            continue
        coords = _MARIO_COORD_RE.findall(rest)
        objects.setdefault(canon, []).extend((int(x), int(y)) for x, y in coords)
    return {
        "mario": mario_xy,
        "objects": objects,
        "screen_w": 256,
        "screen_h": 240,
    }


# ─────────────────────────────────────────────────────────────────────
# Schema serialiser
# ─────────────────────────────────────────────────────────────────────


def _emit(
    *,
    domain: str,
    task: str,
    goal: str,
    step: int,
    entities: List[Dict[str, Any]],
    attributes: List[str],
    affordances: List[str],
    relations: List[str],
    state_flags: Dict[str, str],
    target: Optional[str],
    candidate_set: List[str],
    actions: List[str],
) -> str:
    """Render the canonical entity dicts into the standard <state> block.

    Entity IDs are renumbered to the validator-compatible ``e1, e2, …``
    form, but the numbering is **deterministic** (driven by the order in
    which the per-game generator appends entities, which is itself
    determined by sort order over (r, c) or canonical label).
    The original "logical" ID supplied by the generator (e.g.
    ``e_tile_1_1``) is mapped via ``id_map`` so that ``attributes``,
    ``affordances``, ``relations``, ``target`` and ``candidate_set``
    automatically follow the renumbering.
    """
    id_map: Dict[str, str] = {}
    for i, e in enumerate(entities, start=1):
        id_map[e["id"]] = f"e{i}"

    def remap(s: str) -> str:
        if not s:
            return s
        # Replace any occurrence of a logical ID with its eN form. Sort
        # keys by length descending so longer matches win first
        # (avoids e_tile_1 swallowed by e_tile substring).
        for old in sorted(id_map, key=len, reverse=True):
            new = id_map[old]
            s = s.replace(old, new)
        return s

    lines: List[str] = []
    lines.append("<state>")
    lines.append(f"domain={domain}")
    lines.append(f"task={task}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")
    lines.append("<entities>")
    for e in entities:
        pos = e.get("pos") or "null"
        new_id = id_map[e["id"]]
        lines.append(
            f"{new_id}[type={e['type']}, label={e['label']}, "
            f"bid=null, pos={pos}, ontology={e['ontology']}]"
        )
    lines.append("")
    lines.append("<attributes>")
    lines.extend(remap(a) for a in attributes)
    lines.append("")
    lines.append("<affordances>")
    lines.extend(remap(a) for a in affordances)
    lines.append("")
    lines.append("<relations>")
    lines.extend(remap(r) for r in relations)
    lines.append("")
    lines.append("<state_flags>")
    for k, v in state_flags.items():
        lines.append(f"{k}={v}")
    lines.append("")
    lines.append("<targets>")
    lines.append(f"target={remap(target) if target else 'null'}")
    lines.append("blocker=null")
    lines.append("constraint=null")
    lines.append(f"candidate_set=[{','.join(remap(c) for c in candidate_set)}]")
    lines.append("history_anchor=null")
    lines.append("")
    lines.append("<actions>")
    for i, a in enumerate(actions, start=1):
        lines.append(f"a{i}={a}")
    lines.append("</state>")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Per-game schema generators
# ─────────────────────────────────────────────────────────────────────


def _schema_2048(info: Dict[str, Any], *, task: str, goal: str, step: int,
                 actions: List[str]) -> str:
    facts = _parse_2048(info)
    rows, cols = facts["rows"], facts["cols"]
    cells = facts["cells"]

    entities: List[Dict[str, Any]] = []
    attributes: List[str] = []
    affordances: List[str] = []
    relations: List[str] = []

    entities.append({"id": "e1", "type": "region", "label": "board",
                     "pos": f"0,0,{rows},{cols}", "ontology": "container_entity"})
    attributes.append("e1.state=visible")
    affordances.append("e1.affords=[inspect]")

    nonzero = [(r, c, v) for r, c, v in cells if v > 0]
    empty_count = len(cells) - len(nonzero)
    tile_ids: List[str] = []
    for r, c, v in nonzero:
        eid = f"e_tile_{r}_{c}"
        tile_ids.append(eid)
        entities.append({"id": eid, "type": "object", "label": f"tile_{v}",
                         "pos": f"{r},{c},1,1", "ontology": "selectable_entity"})
        attributes.append(f"{eid}.state=visible")
        attributes.append(f"{eid}.value={v}")
        affordances.append(f"{eid}.affords=[select, track, compare]")
        relations.append(f"contains(e1,{eid})")

    if empty_count > 0:
        entities.append({"id": "e_empty", "type": "region", "label": "empty_cells",
                         "pos": "null", "ontology": "navigable_region"})
        attributes.append(f"e_empty.value={empty_count}")
        affordances.append("e_empty.affords=[inspect]")
        relations.append("contains(e1,e_empty)")

    if facts["highest"] is not None:
        entities.append({"id": "e_highest", "type": "text", "label": "highest_tile",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_highest.value={facts['highest']}")
        affordances.append("e_highest.affords=[read]")
    if facts["score"] is not None:
        entities.append({"id": "e_score", "type": "text", "label": "score",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_score.value={facts['score']}")
        affordances.append("e_score.affords=[read]")

    target = tile_ids[-1] if tile_ids else None
    return _emit(
        domain="gymv", task=task, goal=goal, step=step,
        entities=entities, attributes=attributes,
        affordances=affordances, relations=relations,
        state_flags={"progress": "0.0", "phase": "early",
                     "scene_type": "game_play", "error": "null",
                     "dialog_open": "false", "input_pending": "true"},
        target=target, candidate_set=tile_ids,
        actions=actions,
    )


def _schema_candy(info: Dict[str, Any], *, task: str, goal: str, step: int,
                  actions: List[str]) -> str:
    facts = _parse_candy_crush(info)
    rows, cols = facts["rows"], facts["cols"]
    entities: List[Dict[str, Any]] = []
    attributes: List[str] = []
    affordances: List[str] = []
    relations: List[str] = []

    entities.append({"id": "e1", "type": "region", "label": "board",
                     "pos": f"0,0,{max(rows,1)},{max(cols,1)}",
                     "ontology": "container_entity"})
    attributes.append("e1.state=visible")
    affordances.append("e1.affords=[inspect]")

    cand_ids: List[str] = []
    for r, c, ltr in facts["cells"]:
        color = _CC_COLOR_NAMES.get(ltr, ltr.lower())
        eid = f"e_cell_{r}_{c}"
        cand_ids.append(eid)
        entities.append({"id": eid, "type": "object",
                         "label": f"candy_{color}",
                         "pos": f"{r},{c},1,1",
                         "ontology": "selectable_entity"})
        attributes.append(f"{eid}.state=visible")
        attributes.append(f"{eid}.value={ltr}")
        affordances.append(f"{eid}.affords=[select, swap, compare]")
        relations.append(f"contains(e1,{eid})")

    if facts["score"] is not None:
        entities.append({"id": "e_score", "type": "text", "label": "score",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_score.value={facts['score']}")
        affordances.append("e_score.affords=[read]")
    if facts["moves_left"] is not None:
        entities.append({"id": "e_moves", "type": "text", "label": "moves_left",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_moves.value={facts['moves_left']}")
        affordances.append("e_moves.affords=[read]")

    return _emit(
        domain="gymv", task=task, goal=goal, step=step,
        entities=entities, attributes=attributes,
        affordances=affordances, relations=relations,
        state_flags={"progress": "0.0", "phase": "early",
                     "scene_type": "game_play", "error": "null",
                     "dialog_open": "false", "input_pending": "true"},
        target=None, candidate_set=[],
        actions=actions,
    )


def _schema_tetris(info: Dict[str, Any], *, task: str, goal: str, step: int,
                   actions: List[str]) -> str:
    facts = _parse_tetris(info)
    entities: List[Dict[str, Any]] = []
    attributes: List[str] = []
    affordances: List[str] = []
    relations: List[str] = []

    entities.append({"id": "e1", "type": "region", "label": "playfield",
                     "pos": f"0,0,{facts['rows']},{facts['cols']}",
                     "ontology": "container_entity"})
    attributes.append("e1.state=visible")
    affordances.append("e1.affords=[inspect]")

    active_ids: List[str] = []
    if facts["active_cells"]:
        active_label = f"active_piece_{facts['active_kind']}"
        rs = [r for r, _, _ in facts["active_cells"]]
        cs = [c for _, c, _ in facts["active_cells"]]
        bbox = (min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
        entities.append({"id": "e_active", "type": "object", "label": active_label,
                         "pos": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                         "ontology": "tracked_entity"})
        attributes.append("e_active.state=visible")
        attributes.append(f"e_active.value={facts['active_kind']}")
        affordances.append("e_active.affords=[move, rotate, drop]")
        relations.append("contains(e1,e_active)")
        active_ids.append("e_active")
        for r, c, ch in facts["active_cells"]:
            cid = f"e_active_cell_{r}_{c}"
            entities.append({"id": cid, "type": "object",
                             "label": f"active_piece_{ch}",
                             "pos": f"{r},{c},1,1",
                             "ontology": "tracked_entity"})
            relations.append("part_of(e_active," + cid + ")")

    for r, c, ch in facts["stack_cells"]:
        sid = f"e_stack_{r}_{c}"
        entities.append({"id": sid, "type": "object",
                         "label": f"stack_block_{ch}",
                         "pos": f"{r},{c},1,1",
                         "ontology": "blocking_entity"})
        attributes.append(f"{sid}.state=visible")
        attributes.append(f"{sid}.value={ch}")
        affordances.append(f"{sid}.affords=[track]")
        relations.append(f"contains(e1,{sid})")

    for i, kind in enumerate(facts["next_kinds"]):
        eid = f"e_next_{i}"
        entities.append({"id": eid, "type": "text",
                         "label": f"next_piece_{kind}",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"{eid}.value={kind}")
        affordances.append(f"{eid}.affords=[read]")

    if facts["score"] is not None:
        entities.append({"id": "e_score", "type": "text", "label": "score",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_score.value={facts['score']}")
        affordances.append("e_score.affords=[read]")
    if facts["lines"] is not None:
        entities.append({"id": "e_lines", "type": "text", "label": "lines",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_lines.value={facts['lines']}")
        affordances.append("e_lines.affords=[read]")
    if facts["level"] is not None:
        entities.append({"id": "e_level", "type": "text", "label": "level",
                         "pos": "null", "ontology": "goal_indicator"})
        attributes.append(f"e_level.value={facts['level']}")
        affordances.append("e_level.affords=[read]")

    target = "e_active" if active_ids else None
    return _emit(
        domain="gymv", task=task, goal=goal, step=step,
        entities=entities, attributes=attributes,
        affordances=affordances, relations=relations,
        state_flags={"progress": "0.0", "phase": "early",
                     "scene_type": "game_play", "error": "null",
                     "dialog_open": "false", "input_pending": "true"},
        target=target, candidate_set=active_ids,
        actions=actions,
    )


def _schema_mario(info: Dict[str, Any], *, task: str, goal: str, step: int,
                  actions: List[str]) -> str:
    facts = _parse_mario(info)
    entities: List[Dict[str, Any]] = []
    attributes: List[str] = []
    affordances: List[str] = []
    relations: List[str] = []

    if facts["mario"] is not None:
        x, y = facts["mario"]
        entities.append({"id": "e_mario", "type": "object", "label": "mario",
                         "pos": f"{x},{y},16,16",
                         "ontology": "tracked_entity"})
        attributes.append("e_mario.state=visible")
        attributes.append(f"e_mario.value={x},{y}")
        affordances.append("e_mario.affords=[move_left, move_right, jump]")

    onto_by_label = {
        "brick": "blocking_entity",
        "question_block": "selectable_entity",
        "inactivated_block": "blocking_entity",
        "goomba": "blocking_entity",
        "koopa": "blocking_entity",
        "pit_start": "navigable_region",
        "pit_end": "navigable_region",
        "warp_pipe": "blocking_entity",
        "item_mushroom": "selectable_entity",
        "stair_block": "blocking_entity",
        "flag": "goal_indicator",
    }
    candidates: List[str] = []
    for canon_label, coords in sorted(facts["objects"].items()):
        for i, (x, y) in enumerate(sorted(coords)):
            eid = f"e_{canon_label}_{i}"
            entities.append({"id": eid, "type": "object",
                             "label": canon_label,
                             "pos": f"{x},{y},16,16",
                             "ontology": onto_by_label.get(canon_label, "object")})
            attributes.append(f"{eid}.state=visible")
            attributes.append(f"{eid}.value={x},{y}")
            affordances.append(f"{eid}.affords=[avoid, track]")
            candidates.append(eid)

    target = "e_flag_0" if "flag" in facts["objects"] and facts["objects"]["flag"] else None
    if not target and candidates:
        target = candidates[0]

    return _emit(
        domain="gymv", task=task, goal=goal, step=step,
        entities=entities, attributes=attributes,
        affordances=affordances, relations=relations,
        state_flags={"progress": "0.0", "phase": "early",
                     "scene_type": "game_play", "error": "null",
                     "dialog_open": "false", "input_pending": "true"},
        target=target, candidate_set=candidates,
        actions=actions,
    )


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


_DISPATCH = {
    "twenty_forty_eight": _schema_2048,
    "candy_crush":        _schema_candy,
    "tetris":             _schema_tetris,
    "super_mario":        _schema_mario,
}


def make_canonical_schema(
    *,
    game: str,
    info: Dict[str, Any],
    task_id: str,
    goal: str,
    step: int,
    actions: Optional[List[str]] = None,
) -> Optional[str]:
    """Return the deterministic canonical ``<state>`` schema for ``game``.

    Returns ``None`` if ``game`` has no canonical generator.
    """
    fn = _DISPATCH.get(game)
    if fn is None:
        return None
    raw_actions = actions or info.get("action_names") or []
    actions_list = [str(a) for a in raw_actions][:25]
    return fn(info, task=task_id, goal=goal, step=step, actions=actions_list)


def canonical_label_hint(game: str) -> str:
    """Return a human-readable spec block to inject into LLM/VLM prompts.

    Lists the allowed entity labels and the position unit the model
    must use, plus the HUD blocklist. For ``super_mario`` we additionally
    instruct the model to copy ``(x, y)`` values from the auxiliary text
    observation rather than estimate pixels from the image — Orak's
    coordinate convention (y from bottom) cannot be inferred from a
    screenshot alone.
    """
    labels = CANONICAL_LABELS.get(game, {})
    units = CANONICAL_POSITION_UNITS.get(game, "grid (row, col, h, w)")
    blocked = HUD_BLOCKLIST.get(game, [])
    label_lines = [f"  - {slot:18} → {fmt}" for slot, fmt in labels.items()]
    blocked_str = ", ".join(repr(b) for b in blocked) if blocked else "(none)"

    extra = ""
    if game == "super_mario":
        extra = (
            "\nIMPORTANT — Mario coordinates: copy the `(x, y)` values "
            "verbatim from the 'Position of Mario:' and 'Positions of "
            "all objects' lines in the auxiliary text observation. "
            "Do NOT estimate pixel offsets from the image; Orak's y-axis "
            "is flipped relative to standard image coords, so visual "
            "estimation will be wrong.\n"
        )
    if game == "candy_crush":
        extra = (
            "\nIMPORTANT — emit ONE entity per board cell (8x8 = 64 cells), "
            "even when colors repeat. Skill learning depends on the full "
            "grid state, not a summary.\n"
        )
    if game == "twenty_forty_eight":
        extra = (
            "\nIMPORTANT — emit ONE entity per non-empty tile (label "
            "tile_<value>, pos=row,col,1,1). Aggregate ALL empty cells "
            "into a SINGLE `empty_cells` region entity with pos=null and "
            "value=<count>. Do NOT enumerate empty cells one-by-one — "
            "this breaks alignment with the canonical schema.\n"
        )
    if game == "tetris":
        extra = (
            "\nIMPORTANT — emit ONE entity for the active piece "
            "(label active_piece_<KIND>, pos=top-left bbox). Then emit "
            "ONE entity per individual stack block already locked on the "
            "board (label stack_block_<KIND>, pos=row,col,1,1). Do not "
            "merge active-piece cells into the active-piece entity itself; "
            "emit them as separate cells too (label active_piece_<KIND>, "
            "pos=row,col,1,1).\n"
        )
    return (
        "CANONICAL LABEL VOCABULARY (you MUST use these exact label strings):\n"
        + "\n".join(label_lines)
        + f"\n\nPOSITION UNITS: {units}\n"
        + "All `pos=` values MUST be in those units. Do NOT mix pixel and "
          "grid coordinates.\n\n"
        + f"DO NOT include HUD overlay text rendered on the frame as game "
          f"entities — these are NOT part of the game state: {blocked_str}.\n"
        + "Entity IDs MUST be of the form e1, e2, e3, … (sequential "
          "integers starting at e1). Iterate entities in a stable order "
          "(typically the board container first, then per-cell entries "
          "sorted by (row, col) or by (x, y), then HUD entries) so the "
          "schema is reproducible across calls.\n"
        + extra
    )


# Per-game entity caps for the LLM/VLM prompts. Default is 20 (the
# value the cross-domain schema spec uses), but candy_crush has 64
# board cells and we want them all enumerated for skill learning.
MAX_ENTITIES_BY_GAME: Dict[str, int] = {
    "twenty_forty_eight": 25,
    "candy_crush":        80,
    "tetris":             60,
    "super_mario":        25,
}
