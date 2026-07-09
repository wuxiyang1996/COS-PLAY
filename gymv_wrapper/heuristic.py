"""Gym-V heuristic adapter: obs.text → structured schema (no LLM call).

Parses the native textual observation from Gym-V environments into the
canonical <state>…</state> schema using regex and heuristics.  This is
the **fast/free** head — no API cost, sub-millisecond latency.

Also exposes :func:`parse_retro_pipe_text` for stable-retro / Temporal
environments whose ``obs.text`` uses ``Key: value | …`` layout.

Usage::

    from gymv_wrapper.heuristic import text_to_schema

    schema_str = text_to_schema(
        obs_text="...",
        description="...",
        task_id="Game2048-v0",
        step=14,
    )
"""

from __future__ import annotations

import re
from typing import Any

_GRID_LINE_RE = re.compile(r"^[\s|+\-\d.#@$*XO_]+$")
_NUMBER_RE = re.compile(r"\d+")
_COORD_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")

_DIRECTION_ACTIONS = ["[Up]", "[Down]", "[Left]", "[Right]"]
_WASD_ACTIONS = ["[w]", "[a]", "[s]", "[d]"]


_LABEL_TO_ONTOLOGY: dict[str, str] = {
    "wall": "blocking_entity",
    "player": "tracked_entity",
    "box": "selectable_entity",
    "box_on_target": "tracked_entity",
    "player_on_target": "tracked_entity",
    "target": "goal_indicator",
    "empty": "navigable_region",
}

_ONTOLOGY_TO_AFFORDANCES: dict[str, list[str]] = {
    "blocking_entity":    [],
    "tracked_entity":     ["track", "compare"],
    "selectable_entity":  ["select", "compare"],
    "goal_indicator":     ["approach", "inspect"],
    "navigable_region":   ["approach"],
}


def _ontology_for(label: str) -> str:
    if label in _LABEL_TO_ONTOLOGY:
        return _LABEL_TO_ONTOLOGY[label]
    if label.startswith("tile_"):
        return "selectable_entity"
    return "tracked_entity"


def text_to_schema(
    obs_text: str = "",
    *,
    description: str = "",
    task_id: str = "",
    step: int = 0,
    max_entities: int = 20,
    include_actions: bool = True,
) -> str:
    """Convert Gym-V native text observation into the canonical schema."""
    goal = _extract_goal(description)
    entities, attributes, relations = _parse_game_text(obs_text, max_entities)
    progress, phase, error = _extract_state_flags(obs_text, step)
    target_eid, blocker_eid = _pick_targets(entities, relations)
    actions = _infer_actions(description) if include_actions else []

    lines: list[str] = ["<state>"]
    lines.append("domain=gymv")
    lines.append(f"task={task_id}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")

    lines.append("<entities>")
    for e in entities:
        ontology = _ontology_for(e.get("label", ""))
        e["_ontology"] = ontology
        pos_val = e.get("pos") or "null"
        lines.append(
            f"{e['eid']}[type={e['type']}, label={e['label']}, "
            f"bid=null, pos={pos_val}, ontology={ontology}]"
        )
    lines.append("")

    lines.append("<attributes>")
    for a in attributes:
        lines.append(f"{a['eid']}.{a['key']}={a['val']}")
    lines.append("")

    lines.append("<affordances>")
    for e in entities:
        ops = _ONTOLOGY_TO_AFFORDANCES.get(e.get("_ontology", ""), [])
        if ops:
            lines.append(f"{e['eid']}.affords=[{', '.join(ops)}]")
    lines.append("")

    lines.append("<relations>")
    for r in relations:
        lines.append(r)
    lines.append("")

    lines.append("<state_flags>")
    lines.append(f"progress={progress}")
    lines.append(f"phase={phase}")
    lines.append("scene_type=game_play")
    lines.append(f"error={error}")
    lines.append("dialog_open=false")
    lines.append("input_pending=false")
    lines.append("")

    lines.append("<targets>")
    lines.append(f"target={target_eid}")
    lines.append(f"blocker={blocker_eid}")
    lines.append("constraint=null")
    eids = [e["eid"] for e in entities[:5]]
    lines.append(f"candidate_set=[{','.join(eids)}]")
    lines.append("history_anchor=null")
    lines.append("")

    if actions:
        lines.append("<actions>")
        for i, act in enumerate(actions, 1):
            lines.append(f"a{i}={act}")
        lines.append("")

    lines.append("</state>")
    return "\n".join(lines)


def parse_retro_pipe_text(obs_text: str) -> dict[str, str]:
    """Parse RetroGymVEnv-style ``obs.text`` (pipe-separated ``Key: value``)."""
    out: dict[str, str] = {}
    if not obs_text:
        return out
    for part in obs_text.split("|"):
        p = part.strip()
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        key = k.strip().lower().replace(" ", "_")
        out[key] = v.strip()
    return out


def _extract_goal(description: str) -> str:
    if not description:
        return "unknown"
    first_line = description.strip().split("\n")[0]
    words = first_line.split()
    if len(words) > 60:
        words = words[:60]
    return " ".join(words)


def _parse_game_text(
    raw_text: str, max_entities: int,
) -> tuple[list[dict], list[dict], list[str]]:
    if not raw_text:
        return [], [], []
    grid = _try_parse_grid(raw_text)
    if grid is not None:
        return _entities_from_grid(grid, max_entities)
    return _entities_from_prose(raw_text, max_entities)


def _try_parse_grid(text: str) -> list[list[str]] | None:
    lines = text.strip().splitlines()
    grid_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _GRID_LINE_RE.match(stripped) and len(stripped) > 2:
            grid_lines.append(stripped)
    if len(grid_lines) < 2:
        return None
    grid: list[list[str]] = []
    for line in grid_lines:
        cleaned = line.replace("|", " ").replace("+", " ").replace("-", " ")
        cells = cleaned.split()
        if cells:
            grid.append(cells)
    return grid or None


def _entities_from_grid(
    grid: list[list[str]], max_entities: int,
) -> tuple[list[dict], list[dict], list[str]]:
    entities: list[dict] = []
    attributes: list[dict] = []
    relations: list[str] = []
    empty_count = 0
    eid_counter = 1

    cell_type_map = {
        "#": "wall", "X": "wall",
        "@": "player", "P": "player",
        "$": "box", "B": "box",
        ".": "target", "T": "target",
        "*": "box_on_target",
        "+": "player_on_target",
        "0": "empty",
    }

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if eid_counter > max_entities:
                break
            label = cell_type_map.get(cell)
            if label == "empty" or cell == "0":
                empty_count += 1
                continue
            if label is None:
                if _NUMBER_RE.fullmatch(cell):
                    label = f"tile_{cell}"
                else:
                    label = cell
            eid = f"e{eid_counter}"
            entities.append({
                "eid": eid, "type": "object", "label": label,
                "pos": f"{r},{c},1,1",
            })
            if _NUMBER_RE.fullmatch(cell) and cell != "0":
                attributes.append({"eid": eid, "key": "value", "val": cell})
            eid_counter += 1

    if empty_count > 0:
        eid = f"e{eid_counter}"
        entities.append({
            "eid": eid, "type": "region", "label": "empty", "pos": None,
        })
        attributes.append({"eid": eid, "key": "cells", "val": str(empty_count)})

    positioned = [(e, *map(int, e["pos"].split(","))) for e in entities if e.get("pos")]
    for i, (ei, ri, ci, _, _) in enumerate(positioned):
        for ej, rj, cj, _, _ in positioned[i + 1:]:
            if abs(ri - rj) + abs(ci - cj) == 1:
                relations.append(f"adjacent({ei['eid']},{ej['eid']})")

    return entities, attributes, relations


def _entities_from_prose(
    text: str, max_entities: int,
) -> tuple[list[dict], list[dict], list[str]]:
    entities: list[dict] = []
    attributes: list[dict] = []
    relations: list[str] = []
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
    eid_counter = 1
    for sent in sentences:
        if eid_counter > max_entities:
            break
        label = sent[:80].replace("\n", " ")
        eid = f"e{eid_counter}"
        pos = None
        coord_match = _COORD_RE.search(sent)
        if coord_match:
            pos = f"{coord_match.group(1)},{coord_match.group(2)},1,1"
        entities.append({"eid": eid, "type": "text", "label": label, "pos": pos})
        eid_counter += 1
    return entities, attributes, relations


def _extract_state_flags(raw_text: str, step: int) -> tuple[str, str, str]:
    error = "null"
    lower = raw_text.lower()
    if "invalid" in lower or "illegal" in lower:
        first_err = re.search(r"(?:invalid|illegal)[^\n.]*", lower)
        error = first_err.group(0).strip() if first_err else "invalid_action"
    progress_match = re.search(r"(\d+(?:\.\d+)?)\s*%", raw_text)
    if progress_match:
        progress = str(round(float(progress_match.group(1)) / 100, 2))
    else:
        progress = "null"
    if step <= 3:
        phase = "early"
    elif step <= 15:
        phase = "mid"
    else:
        phase = "late"
    return progress, phase, error


def _pick_targets(
    entities: list[dict], relations: list[str],
) -> tuple[str, str]:
    target = "null"
    blocker = "null"
    for e in entities:
        label = e["label"].lower()
        if any(kw in label for kw in ("player", "agent", "cursor")):
            target = e["eid"]
            break
    if target == "null" and entities:
        target = entities[0]["eid"]
    for e in entities:
        label = e["label"].lower()
        if any(kw in label for kw in ("wall", "block", "obstacle", "mine")):
            blocker = e["eid"]
            break
    return target, blocker


def _infer_actions(description: str) -> list[str]:
    lower = description.lower()
    if "[up]" in lower or "up, down, left, right" in lower:
        return _DIRECTION_ACTIONS
    if "[w]" in lower or "wasd" in lower or "[w] for up" in lower:
        return _WASD_ACTIONS
    bracket_actions = re.findall(r"\[([A-Za-z0-9_ ]+)\]", description)
    if bracket_actions:
        return [f"[{a}]" for a in bracket_actions[:6]]
    return []
