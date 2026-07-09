"""BrowserGym tool implementations for multi-turn visual grounding.

Each tool handler queries the AXTree, ``extra_element_properties``, or
focused-element data on the BrowserGym observation dict for ground-truth
element information. The VLM identifies elements visually, then calls
these tools to get exact bounding boxes, properties, and relationships.

Usage::

    from browsergym_wrapper.tools import build_browser_registry

    registry = build_browser_registry(obs=browsergym_obs_dict)
    # pass registry to tool_loop for multi-turn grounding
"""

from __future__ import annotations

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
from browsergym_wrapper.heuristic import (
    CONTAINER_ROLES,
    IGNORED_ROLES,
    INTERACTIVE_ROLES,
    _extract_goal,
)


# ── Browser-specific tool definitions ─────────────────────────────────

TOOL_QUERY_ELEMENT_BBOX = ToolDef(
    name="query_element_bbox",
    description=(
        "Get the exact bounding box [x, y, width, height] for a browser "
        "element by its bid (element ID visible in the screenshot or AXTree). "
        "Also returns visibility ratio, clickability, and accessible name."
    ),
    parameters={
        "type": "object",
        "properties": {
            "bid": {
                "type": "string",
                "description": "The browser element ID (bid) to look up.",
            },
        },
        "required": ["bid"],
    },
    domain="browser",
)

TOOL_SEARCH_ELEMENTS = ToolDef(
    name="search_elements",
    description=(
        "Search the page's accessibility tree for elements matching a "
        "query (by role, name/label text, or value). Returns matching "
        "elements with their bids, roles, bounding boxes, and states."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for in element names, values, or roles.",
            },
            "role_filter": {
                "type": "string",
                "description": "Restrict to elements with this accessibility role (e.g. 'button', 'link', 'textbox').",
            },
            "interactive_only": {
                "type": "boolean",
                "description": "If true, only return interactive elements (buttons, links, inputs, etc.). Default false.",
            },
        },
        "required": ["query"],
    },
    domain="browser",
)

TOOL_GET_PAGE_INFO = ToolDef(
    name="get_page_info",
    description=(
        "Get current page metadata: URL, page title, number of open tabs, "
        "focused element, and last action result."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="browser",
)

TOOL_GET_ELEMENT_TREE = ToolDef(
    name="get_element_tree",
    description=(
        "Get the accessibility tree rooted at a specific element bid. "
        "Shows the element's children with their roles, names, and bids. "
        "Useful for understanding complex UI structures (menus, forms, tables)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "bid": {
                "type": "string",
                "description": "Root element bid. Omit for the full page tree (truncated).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum tree depth to return. Default 3.",
            },
        },
        "required": [],
    },
    domain="browser",
)

TOOL_GET_SOM_ELEMENTS = ToolDef(
    name="get_som_elements",
    description=(
        "Get all elements currently marked in the Set-of-Mark overlay. "
        "These are the interactive, unobscured elements that the agent "
        "can meaningfully click/type into."
    ),
    parameters={
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Maximum elements to return. Default 25.",
            },
        },
        "required": [],
    },
    domain="browser",
)


# ── Cached browser state ─────────────────────────────────────────────

class _BrowserState:
    """Lazy-parsed browser observation backing all tool handlers."""

    def __init__(self, obs: dict[str, Any]):
        self.obs = obs
        self._axtree = obs.get("axtree_object")
        self._extra = obs.get("extra_element_properties", {})
        self._focused = obs.get("focused_element_bid", "")
        self._goal = _extract_goal(obs)
        self._url = obs.get("url", "")
        self._titles = obs.get("open_pages_titles", ())
        self._urls = obs.get("open_pages_urls", ())
        self._last_action = obs.get("last_action", "")
        self._last_error = obs.get("last_action_error", "")
        self._nodes: list[dict] | None = None
        self._bid_to_node: dict[str, dict] | None = None

    def _ensure_nodes(self) -> None:
        if self._nodes is not None:
            return
        self._nodes = []
        self._bid_to_node = {}
        if self._axtree is None:
            return
        for node in self._axtree.get("nodes", []):
            self._nodes.append(node)
            bid = node.get("browsergym_id")
            if bid:
                self._bid_to_node[bid] = node

    @property
    def nodes(self) -> list[dict]:
        self._ensure_nodes()
        return self._nodes  # type: ignore[return-value]

    @property
    def bid_to_node(self) -> dict[str, dict]:
        self._ensure_nodes()
        return self._bid_to_node  # type: ignore[return-value]

    def get_element_info(self, bid: str) -> dict[str, Any] | None:
        node = self.bid_to_node.get(bid)
        if node is None:
            return None
        props = self._extra.get(bid, {})
        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "")
        value = None
        if "value" in node and "value" in node.get("value", {}):
            value = node["value"]["value"]

        states = []
        if props.get("visibility", 0) > 0.5:
            states.append("visible")
        if props.get("clickable"):
            states.append("clickable")
        if bid == self._focused:
            states.append("focused")
        if props.get("set_of_marks"):
            states.append("set_of_marks")
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

        return {
            "bid": bid,
            "role": role,
            "name": name,
            "value": value,
            "bbox": props.get("bbox"),
            "visibility": props.get("visibility", 0),
            "clickable": props.get("clickable", False),
            "set_of_marks": props.get("set_of_marks", False),
            "states": states,
        }


# ── Handler implementations ──────────────────────────────────────────

def _h_query_entity_pos(state: _BrowserState, *, entity_label: str) -> dict:
    label_lower = entity_label.lower().strip()

    if state.bid_to_node.get(label_lower) or state.bid_to_node.get(entity_label):
        bid = entity_label if entity_label in state.bid_to_node else label_lower
        info = state.get_element_info(bid)
        if info:
            return {"found": True, "match": info}

    matches = []
    for node in state.nodes:
        bid = node.get("browsergym_id")
        if not bid:
            continue
        name = node.get("name", {}).get("value", "")
        role = node.get("role", {}).get("value", "")
        if label_lower in name.lower() or label_lower in role.lower():
            info = state.get_element_info(bid)
            if info and info.get("visibility", 0) > 0.3:
                matches.append(info)
                if len(matches) >= 5:
                    break

    if not matches:
        return {"found": False, "message": f"No element matching '{entity_label}'"}
    return {"found": True, "matches": matches}


def _h_query_element_bbox(state: _BrowserState, *, bid: str) -> dict:
    info = state.get_element_info(bid)
    if info is None:
        return {"found": False, "message": f"No element with bid '{bid}'"}
    return {"found": True, **info}


def _h_list_entities(
    state: _BrowserState, *, filter_type: str = "all", max_results: int = 25,
) -> dict:
    results = []
    for node in state.nodes:
        if len(results) >= max_results:
            break
        role = node.get("role", {}).get("value", "")
        if role in IGNORED_ROLES:
            continue
        bid = node.get("browsergym_id")
        if not bid:
            continue
        props = state._extra.get(bid, {})
        if props.get("visibility", 0) < 0.5:
            continue

        is_interactive = role.lower() in INTERACTIVE_ROLES
        is_container = role.lower() in CONTAINER_ROLES

        if filter_type == "interactive" and not is_interactive:
            continue
        if filter_type == "container" and not is_container:
            continue

        name = node.get("name", {}).get("value", "")
        results.append({
            "bid": bid,
            "role": role,
            "name": name[:80],
            "bbox": props.get("bbox"),
            "clickable": props.get("clickable", False),
            "set_of_marks": props.get("set_of_marks", False),
        })

    return {"count": len(results), "entities": results}


def _h_search_elements(
    state: _BrowserState,
    *,
    query: str,
    role_filter: str = "",
    interactive_only: bool = False,
) -> dict:
    query_lower = query.lower()
    matches = []
    for node in state.nodes:
        bid = node.get("browsergym_id")
        if not bid:
            continue
        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "")
        value_str = ""
        if "value" in node and "value" in node.get("value", {}):
            value_str = str(node["value"]["value"])

        if role_filter and role.lower() != role_filter.lower():
            continue
        if interactive_only and role.lower() not in INTERACTIVE_ROLES:
            continue

        text_blob = f"{name} {value_str} {role}".lower()
        if query_lower not in text_blob:
            continue

        props = state._extra.get(bid, {})
        if props.get("visibility", 0) < 0.3:
            continue

        matches.append({
            "bid": bid,
            "role": role,
            "name": name[:80],
            "value": value_str or None,
            "bbox": props.get("bbox"),
            "clickable": props.get("clickable", False),
        })
        if len(matches) >= 15:
            break

    return {"count": len(matches), "matches": matches}


def _h_check_relation(
    state: _BrowserState, *, entity_a: str, entity_b: str, relation: str,
) -> dict:
    info_a = _resolve_browser_element(state, entity_a)
    info_b = _resolve_browser_element(state, entity_b)
    if info_a is None or info_b is None:
        return {
            "holds": False,
            "reason": f"Could not resolve {'entity_a' if info_a is None else 'entity_b'}",
        }

    bbox_a = info_a.get("bbox")
    bbox_b = info_b.get("bbox")

    if relation == "adjacent":
        if bbox_a and bbox_b:
            dist = _bbox_distance(bbox_a, bbox_b)
            return {"holds": dist < 30, "pixel_distance": round(dist, 1)}
        return {"holds": False, "reason": "bboxes unknown"}

    if relation == "overlaps":
        if bbox_a and bbox_b:
            overlap = _bbox_overlap(bbox_a, bbox_b)
            return {"holds": overlap > 0, "overlap_area": round(overlap, 1)}
        return {"holds": False, "reason": "bboxes unknown"}

    if relation == "contains":
        if bbox_a and bbox_b:
            contained = _bbox_contains(bbox_a, bbox_b)
            return {"holds": contained}
        return {"holds": False, "reason": "bboxes unknown"}

    if relation == "parent_of":
        return _check_parent(state, info_a, info_b)

    if relation == "sibling":
        return _check_sibling(state, info_a, info_b)

    return {"holds": False, "reason": f"Relation '{relation}' not implemented for browser"}


def _h_get_state_flags(state: _BrowserState) -> dict:
    error = state._last_error.strip()[:120] if state._last_error else None
    nodes = state.nodes
    dialog_open = any(
        n.get("role", {}).get("value", "") in ("dialog", "alertdialog")
        for n in nodes
    )
    input_pending = False
    for n in nodes:
        bid = n.get("browsergym_id")
        role = n.get("role", {}).get("value", "")
        if bid and bid == state._focused and role in ("textbox", "searchbox", "combobox"):
            input_pending = True
            break

    return {
        "url": state._url,
        "num_tabs": len(state._urls),
        "focused_bid": state._focused or None,
        "last_action": state._last_action or None,
        "error": error,
        "dialog_open": dialog_open,
        "input_pending": input_pending,
    }


_INPUT_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}
_TOGGLE_ROLES = {"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"}
_PRESS_ROLES = {
    "button", "link", "menuitem", "menuitemcheckbox", "menuitemradio",
    "tab", "option", "treeitem", "row", "listitem", "cell",
}


def _h_list_valid_actions(state: _BrowserState) -> dict:
    """Surface the per-page valid action shortlist for the actor.

    Bug fix (2026-05-03 — VWA diagnostic on visualwebarena.92): the prior
    filter required ``props.get("set_of_marks") or
    props.get("clickable")`` to even consider a node, which silently
    dropped EVERY ``fill(...)`` candidate on pages where BrowserGym had
    not populated those flags (about-blank fallbacks, dynamically
    rendered pages, classifieds search pages where ``set_of_marks=True``
    is only set on a small subset of obviously-clickable nodes). The
    actor then only saw 5 navigation-only actions and thrashed via
    ``scroll/go_back/go_forward``. Fix: surface ``fill(...)`` for any
    visible textbox/searchbox/combobox role regardless of the SoM/
    clickable flags, and only require those flags for click/check
    candidates (which truly need them to disambiguate from background
    spans). See ``legacy/visualwebarena/vwa-improvement-plan.md`` §3
    Tier-1 change B.
    """
    actions: list[dict] = []
    n_input = 0
    n_toggle = 0
    n_click = 0
    for node in state.nodes:
        bid = node.get("browsergym_id")
        if not bid:
            continue
        props = state._extra.get(bid, {})
        if props.get("visibility", 0) < 0.5:
            continue

        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "")[:40]

        if role in _INPUT_ROLES:
            # Always surface ``fill(...)`` for visible input-like elements.
            # ``set_of_marks`` / ``clickable`` flags are unreliable for
            # ``<input>`` / ``<textarea>`` / ``<select>`` nodes.
            actions.append({"action": f'fill("{bid}", "...")', "role": role, "name": name})
            n_input += 1
        elif role in _TOGGLE_ROLES:
            if not (props.get("set_of_marks") or props.get("clickable")):
                continue
            actions.append({"action": f'check("{bid}")', "role": role, "name": name})
            n_toggle += 1
        else:
            # Click candidates: require either SoM flag or clickable signal
            # to avoid surfacing the entire page DOM.
            if not (props.get("set_of_marks") or props.get("clickable")):
                continue
            actions.append({"action": f'click("{bid}")', "role": role, "name": name})
            n_click += 1

        if len(actions) >= 25:
            break

    actions.append({"action": "scroll(down)", "role": "page", "name": "scroll"})
    actions.append({"action": "go_back()", "role": "navigation", "name": "back"})
    return {
        "actions": actions,
        "count": len(actions),
        "fill_candidates": n_input,
        "click_candidates": n_click,
        "toggle_candidates": n_toggle,
    }


def _h_get_page_info(state: _BrowserState) -> dict:
    return {
        "url": state._url,
        "goal": state._goal,
        "open_tabs": [
            {"url": u, "title": t}
            for u, t in zip(state._urls, state._titles)
        ],
        "active_tab_index": 0,
        "focused_element_bid": state._focused or None,
        "last_action": state._last_action or None,
        "last_action_error": state._last_error or None,
    }


def _h_get_element_tree(state: _BrowserState, *, bid: str = "", max_depth: int = 3) -> dict:
    if not bid:
        top_level = []
        for node in state.nodes[:30]:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            nbid = node.get("browsergym_id", "")
            if role not in IGNORED_ROLES and (name or role in CONTAINER_ROLES):
                top_level.append({"bid": nbid, "role": role, "name": name[:60]})
        return {"root": "page", "children": top_level[:20], "truncated": True}

    node = state.bid_to_node.get(bid)
    if node is None:
        return {"error": f"No element with bid '{bid}'"}

    return {"root": _build_subtree(state, node, max_depth, 0)}


def _h_get_som_elements(state: _BrowserState, *, max_results: int = 25) -> dict:
    results = []
    for bid, props in state._extra.items():
        if not props.get("set_of_marks"):
            continue
        if props.get("visibility", 0) < 0.5:
            continue
        node = state.bid_to_node.get(bid)
        role = node.get("role", {}).get("value", "") if node else ""
        name = node.get("name", {}).get("value", "") if node else ""
        results.append({
            "bid": bid,
            "role": role,
            "name": name[:80],
            "bbox": props.get("bbox"),
            "clickable": props.get("clickable", False),
        })
        if len(results) >= max_results:
            break
    return {"count": len(results), "elements": results}


# ── Geometry helpers ─────────────────────────────────────────────────

def _bbox_distance(a: list, b: list) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _bbox_overlap(a: list, b: list) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _bbox_contains(outer: list, inner: list) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[0] + outer[2] >= inner[0] + inner[2]
        and outer[1] + outer[3] >= inner[1] + inner[3]
    )


def _resolve_browser_element(state: _BrowserState, ref: str) -> dict | None:
    if ref in state.bid_to_node:
        return state.get_element_info(ref)
    ref_lower = ref.lower()
    for node in state.nodes:
        bid = node.get("browsergym_id")
        if not bid:
            continue
        name = node.get("name", {}).get("value", "")
        if ref_lower in name.lower():
            return state.get_element_info(bid)
    return None


def _build_subtree(state: _BrowserState, node: dict, max_depth: int, depth: int) -> dict:
    role = node.get("role", {}).get("value", "")
    name = node.get("name", {}).get("value", "")
    bid = node.get("browsergym_id", "")
    entry: dict[str, Any] = {"bid": bid, "role": role, "name": name[:60]}

    if depth < max_depth:
        children = []
        for cid in node.get("childIds", []):
            for n in state.nodes:
                if n["nodeId"] == cid:
                    child_role = n.get("role", {}).get("value", "")
                    if child_role not in IGNORED_ROLES:
                        children.append(_build_subtree(state, n, max_depth, depth + 1))
                    break
        if children:
            entry["children"] = children

    return entry


def _check_parent(state: _BrowserState, info_a: dict, info_b: dict) -> dict:
    bid_a = info_a.get("bid", "")
    bid_b = info_b.get("bid", "")
    node_a = state.bid_to_node.get(bid_a)
    if node_a is None:
        return {"holds": False, "reason": "parent node not found"}
    for cid in node_a.get("childIds", []):
        for n in state.nodes:
            if n["nodeId"] == cid and n.get("browsergym_id") == bid_b:
                return {"holds": True}
    return {"holds": False}


def _check_sibling(state: _BrowserState, info_a: dict, info_b: dict) -> dict:
    bid_a = info_a.get("bid", "")
    bid_b = info_b.get("bid", "")
    node_id_a = None
    node_id_b = None
    for n in state.nodes:
        if n.get("browsergym_id") == bid_a:
            node_id_a = n["nodeId"]
        if n.get("browsergym_id") == bid_b:
            node_id_b = n["nodeId"]
    if not (node_id_a and node_id_b):
        return {"holds": False, "reason": "nodes not found"}
    for n in state.nodes:
        children = n.get("childIds", [])
        if node_id_a in children and node_id_b in children:
            return {"holds": True, "parent_bid": n.get("browsergym_id", "")}
    return {"holds": False}


# ── Public: build registry ───────────────────────────────────────────

def build_browser_registry(obs: dict[str, Any]) -> ToolRegistry:
    """Create a ToolRegistry with all browser tools bound to a BrowserGym obs.

    Parameters
    ----------
    obs : dict
        The observation dict from ``BrowserEnv._get_obs()``.

    Returns
    -------
    ToolRegistry
    """
    state = _BrowserState(obs)
    reg = ToolRegistry(domain="browser")

    reg.register(TOOL_QUERY_ENTITY_POS, lambda **kw: _h_query_entity_pos(state, **kw))
    reg.register(TOOL_QUERY_ELEMENT_BBOX, lambda **kw: _h_query_element_bbox(state, **kw))
    reg.register(TOOL_LIST_ENTITIES, lambda **kw: _h_list_entities(state, **kw))
    reg.register(TOOL_SEARCH_ELEMENTS, lambda **kw: _h_search_elements(state, **kw))
    reg.register(TOOL_CHECK_RELATION, lambda **kw: _h_check_relation(state, **kw))
    reg.register(TOOL_GET_STATE_FLAGS, lambda **kw: _h_get_state_flags(state))
    reg.register(TOOL_LIST_VALID_ACTIONS, lambda **kw: _h_list_valid_actions(state))
    reg.register(TOOL_GET_PAGE_INFO, lambda **kw: _h_get_page_info(state))
    reg.register(TOOL_GET_ELEMENT_TREE, lambda **kw: _h_get_element_tree(state, **kw))
    reg.register(TOOL_GET_SOM_ELEMENTS, lambda **kw: _h_get_som_elements(state, **kw))

    return reg


__all__ = [
    "TOOL_QUERY_ELEMENT_BBOX",
    "TOOL_SEARCH_ELEMENTS",
    "TOOL_GET_PAGE_INFO",
    "TOOL_GET_ELEMENT_TREE",
    "TOOL_GET_SOM_ELEMENTS",
    "build_browser_registry",
]
