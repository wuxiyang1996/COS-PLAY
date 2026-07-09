"""BrowserGym heuristic adapter: AXTree/DOM → structured schema (no LLM call).

Parses the native AXTree and element properties from a BrowserGym
observation dict into the canonical <state>…</state> schema using
deterministic tree-walking.  This is the **fast/free** head.

Use cases:
  - Cheap baseline labels for comparison against the vision head.
  - Real-time schema generation during RL rollouts.
  - Validator: compare heuristic output against GPT vision output to
    flag hallucinations or missed entities.

Usage::

    from browsergym_wrapper.heuristic import obs_to_schema

    schema_str = obs_to_schema(obs, step=3, task_id="webarena.shopping.143")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


IGNORED_ROLES = frozenset({
    "LineBreak", "none", "generic", "InlineTextBox",
    "StaticText", "paragraph",
})

INTERACTIVE_ROLES = frozenset({
    "link", "button", "textbox", "combobox", "checkbox", "radio",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "switch",
    "searchbox", "spinbutton", "slider", "option", "treeitem",
})

CONTAINER_ROLES = frozenset({
    "navigation", "banner", "main", "complementary", "contentinfo",
    "form", "dialog", "alertdialog", "menu", "menubar", "toolbar",
    "tablist", "list", "listbox", "tree", "grid", "table", "group",
    "region",
})


@dataclass
class _Entity:
    eid: str
    etype: str
    label: str
    bid: str | None = None
    pos: str | None = None
    role: str = ""
    value: str | None = None
    states: list[str] = field(default_factory=list)
    children_eids: list[str] = field(default_factory=list)


def obs_to_schema(
    obs: dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    max_entities: int = 25,
) -> str:
    """Convert a BrowserGym observation dict into the canonical schema string.

    Parameters
    ----------
    obs : dict
        The observation dict from ``BrowserEnv._get_obs()``.
    step : int
        Current step number.
    task_id : str
        Human-readable task identifier.
    max_entities : int
        Hard cap on emitted entities.

    Returns
    -------
    str
        The ``<state>…</state>`` tagged-text summary.
    """
    axtree = obs.get("axtree_object")
    extra_props = obs.get("extra_element_properties", {})
    focused_bid = obs.get("focused_element_bid", "")
    goal = _extract_goal(obs)
    url = obs.get("url", "")
    last_error = obs.get("last_action_error", "")
    open_urls = obs.get("open_pages_urls", ())

    entities = _extract_entities(axtree, extra_props, focused_bid, max_entities)
    relations = _build_relations(entities)

    error_str = last_error.strip()[:120] if last_error else "null"
    dialog_open = any(e.role in ("dialog", "alertdialog") for e in entities)
    input_pending = any(
        "focused" in e.states and e.role in ("textbox", "searchbox", "combobox")
        for e in entities
    )

    target_eid, blocker_eid = _pick_targets(entities, goal, error_str)
    actions = _suggest_actions(entities, target_eid)

    lines: list[str] = ["<state>"]
    lines.append("domain=browser")
    lines.append(f"task={task_id}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")

    lines.append("<entities>")
    for e in entities:
        parts = [f"type={e.etype}", f"label={e.role} '{e.label}'"]
        if e.bid:
            parts.append(f"bid={e.bid}")
        if e.pos:
            parts.append(f"pos={e.pos}")
        lines.append(f"{e.eid}[{', '.join(parts)}]")
    lines.append("")

    lines.append("<attributes>")
    for e in entities:
        if e.value is not None:
            lines.append(f"{e.eid}.value={e.value}")
        if e.states:
            lines.append(f"{e.eid}.state={','.join(e.states)}")
    lines.append("")

    lines.append("<relations>")
    for r in relations:
        lines.append(r)
    lines.append("")

    lines.append("<state_flags>")
    lines.append("progress=null")
    lines.append("phase=null")
    lines.append(f"error={error_str}")
    lines.append(f"dialog_open={'true' if dialog_open else 'false'}")
    lines.append(f"input_pending={'true' if input_pending else 'false'}")
    lines.append(f"num_tabs={len(open_urls)}")
    lines.append(f"url={url}")
    lines.append("")

    lines.append("<targets>")
    lines.append(f"target={target_eid}")
    lines.append(f"blocker={blocker_eid}")
    clickable_eids = [e.eid for e in entities if "clickable" in e.states][:8]
    lines.append(f"candidate_set=[{','.join(clickable_eids)}]")
    lines.append("")

    if actions:
        lines.append("<actions>")
        for i, act in enumerate(actions, 1):
            lines.append(f"a{i}={act}")
        lines.append("")

    lines.append("</state>")
    return "\n".join(lines)


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_goal(obs: dict) -> str:
    goal = obs.get("goal", "")
    if not goal:
        goal_obj = obs.get("goal_object", ())
        texts = [m.get("text", "") for m in goal_obj if m.get("type") == "text"]
        goal = " ".join(texts)
    words = goal.strip().split()
    if len(words) > 60:
        words = words[:60]
    return " ".join(words) if words else "unknown"


def _extract_entities(
    axtree: dict | None,
    extra_props: dict,
    focused_bid: str,
    max_entities: int,
) -> list[_Entity]:
    if axtree is None:
        return []
    nodes = axtree.get("nodes", [])
    if not nodes:
        return []

    node_id_to_idx: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        node_id_to_idx[node["nodeId"]] = idx

    raw_entities: list[_Entity] = []
    bid_to_eid: dict[str, str] = {}
    eid_counter = 1

    for node in nodes:
        role = node.get("role", {}).get("value", "")
        if role in IGNORED_ROLES:
            continue
        name = node.get("name", {}).get("value", "")
        if not name and role not in CONTAINER_ROLES:
            continue

        bid = node.get("browsergym_id")
        props = extra_props.get(bid, {}) if bid else {}

        visibility = props.get("visibility", 0)
        if bid and visibility < 0.5:
            continue

        is_interactive = role.lower() in INTERACTIVE_ROLES
        is_container = role.lower() in CONTAINER_ROLES
        is_clickable = props.get("clickable", False)
        is_in_som = props.get("set_of_marks", False)

        if not (is_interactive or is_container or is_clickable or is_in_som):
            continue

        eid = f"e{eid_counter}"
        eid_counter += 1

        bbox = props.get("bbox")
        pos = None
        if bbox:
            pos = ",".join(str(int(round(v))) for v in bbox[:4])

        value = None
        if "value" in node and "value" in node.get("value", {}):
            value = node["value"]["value"]

        states: list[str] = ["visible"]
        if is_clickable:
            states.append("clickable")
        if bid and bid == focused_bid:
            states.append("focused")
        for prop in node.get("properties", []):
            pname = prop.get("name", "")
            pval = prop.get("value", {}).get("value")
            if pname == "checked" and pval:
                states.append("checked")
            if pname == "disabled" and pval:
                states.append("disabled")
            if pname == "required" and pval:
                states.append("required")
            if pname == "expanded" and pval is not None:
                states.append(f"expanded={pval}")

        ent = _Entity(
            eid=eid,
            etype="element",
            label=name.strip()[:80],
            bid=bid,
            pos=pos,
            role=role,
            value=value,
            states=states,
        )

        if bid:
            bid_to_eid[bid] = eid

        for cid in node.get("childIds", []):
            cidx = node_id_to_idx.get(cid)
            if cidx is not None:
                cbid = nodes[cidx].get("browsergym_id")
                if cbid:
                    ent.children_eids.append(cbid)

        raw_entities.append(ent)
        if eid_counter > max_entities + 1:
            break

    for ent in raw_entities:
        ent.children_eids = [bid_to_eid[b] for b in ent.children_eids if b in bid_to_eid]

    return raw_entities[:max_entities]


def _build_relations(entities: list[_Entity]) -> list[str]:
    relations: list[str] = []
    eid_set = {e.eid for e in entities}

    for e in entities:
        valid_children = [c for c in e.children_eids if c in eid_set]
        if len(valid_children) == 1:
            relations.append(f"contains({e.eid},{valid_children[0]})")
        elif len(valid_children) > 1:
            relations.append(f"grouped({','.join(valid_children)})")

    positioned = []
    for e in entities:
        if e.pos:
            parts = list(map(int, e.pos.split(",")))
            if len(parts) >= 4:
                cx = parts[0] + parts[2] / 2
                cy = parts[1] + parts[3] / 2
                positioned.append((e.eid, cx, cy))

    for i, (eid_a, ax, ay) in enumerate(positioned):
        for eid_b, bx, by in positioned[i + 1:]:
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist < 60:
                relations.append(f"adjacent({eid_a},{eid_b})")

    return relations


def _pick_targets(
    entities: list[_Entity], goal: str, error: str,
) -> tuple[str, str]:
    target = "null"
    blocker = "null"
    goal_words = set(goal.lower().split())

    best_score = 0
    for e in entities:
        label_words = set(e.label.lower().split())
        overlap = len(goal_words & label_words)
        if overlap > best_score and "clickable" in e.states:
            best_score = overlap
            target = e.eid

    if target == "null":
        for e in entities:
            if "clickable" in e.states:
                target = e.eid
                break

    if error != "null":
        for e in entities:
            if "focused" in e.states:
                blocker = e.eid
                break

    return target, blocker


def _suggest_actions(
    entities: list[_Entity], target_eid: str,
) -> list[str]:
    actions: list[str] = []
    target_ent = None
    for e in entities:
        if e.eid == target_eid:
            target_ent = e
            break

    if target_ent and target_ent.bid:
        bid = target_ent.bid
        role = target_ent.role.lower()
        if role in ("textbox", "searchbox", "combobox"):
            actions.append(f'fill("{bid}", "...")')
        else:
            actions.append(f'click("{bid}")')

    clickable = [e for e in entities if "clickable" in e.states and e.eid != target_eid]
    for e in clickable[:4]:
        if e.bid:
            actions.append(f'click("{e.bid}")')

    if not actions:
        actions.append("scroll(down)")

    return actions[:5]
