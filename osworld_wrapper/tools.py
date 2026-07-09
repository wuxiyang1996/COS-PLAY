"""OSWorld tool implementations for multi-turn visual grounding.

Tool handlers that query the OS-level accessibility tree (AT-SPI on
Linux, UI Automation on Windows) to give the VLM ground-truth element
positions, sizes, and state flags. The VLM identifies elements visually,
then calls these tools to convert the visual identification into pixel
coordinates suitable for ``pyautogui.click(x, y)``.

Usage::

    from osworld_wrapper.tools import build_osworld_registry

    registry = build_osworld_registry(
        a11y_tree_xml=obs["accessibility_tree"],
        instruction=obs["instruction"],
        terminal_output=obs.get("terminal", ""),
    )
    # pass registry to tool_loop for multi-turn grounding
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, List

from vlm_wrapper.tools import (
    TOOL_GET_STATE_FLAGS,
    TOOL_QUERY_ENTITY_POS,
    ToolDef,
    ToolRegistry,
)

from osworld_wrapper.heuristic import (
    _FLAG_NAMES,
    _NS_STATE,
    _bbox_from,
    _ns_attr,
    _strip_ns,
)


# ── OSWorld-specific tool definitions ────────────────────────────────

TOOL_QUERY_OS_ELEMENT = ToolDef(
    name="query_os_element",
    description=(
        "Look up a desktop UI element by name or role from the OS "
        "accessibility tree (AT-SPI on Linux, UI Automation on Windows). "
        "Returns screen coordinates, size, and state flags."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Element name or text to search for.",
            },
            "role": {
                "type": "string",
                "description": "Accessibility role filter (e.g. 'push button', 'text', 'menu item').",
            },
        },
        "required": ["name"],
    },
    domain="osworld",
)


# ── Handler implementation ───────────────────────────────────────────


def _element_states(el: ET.Element) -> List[str]:
    """Collect ``st:flag="true"`` boolean states (lowercase names)."""
    out: List[str] = []
    for flag in _FLAG_NAMES:
        v = _ns_attr(el, _NS_STATE, flag)
        if v is not None and v.strip().lower() == "true":
            out.append(flag)
    return out


def _h_query_os_element(
    a11y_tree_xml: str,
    *,
    name: str,
    role: str = "",
    max_results: int = 10,
) -> dict:
    """Search the OS accessibility tree XML for matching elements.

    Walks the namespaced AT-SPI / UI-Automation tree (the same XML
    OSWorld returns from ``GET /accessibility``) and returns the first
    ``max_results`` elements whose ``name`` or text contains *name*
    case-insensitively, optionally filtered to a given role (= element
    tag).  Each match carries its pixel ``x, y, width, height`` (read
    from ``cp:screencoord`` + ``cp:size``) plus a list of boolean state
    flags so the VLM can convert a visual identification into a
    pyautogui click target.
    """
    name_lower = (name or "").lower()
    role_lower = (role or "").lower()

    if not a11y_tree_xml or not a11y_tree_xml.strip():
        return {"found": False, "message": "empty accessibility tree"}

    try:
        root = ET.fromstring(a11y_tree_xml)
    except ET.ParseError as exc:
        return {"found": False, "message": f"xml_parse_failed: {exc}"}

    matches: List[dict[str, Any]] = []
    for el in root.iter():
        el_role = _strip_ns(el.tag).lower()
        if not el_role or el_role == "root":
            continue

        el_name = (el.get("name") or "").strip()
        el_text = (el.text or "").strip()

        if name_lower:
            if (
                name_lower not in el_name.lower()
                and name_lower not in el_text.lower()
            ):
                continue
        if role_lower and role_lower not in el_role:
            continue

        entry: dict[str, Any] = {
            "name": el_name,
            "role": el_role,
            "text": el_text[:80] if el_text else None,
        }
        bbox = _bbox_from(el)
        if bbox is not None:
            entry["x"], entry["y"], entry["width"], entry["height"] = bbox
            entry["center"] = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
        entry["states"] = _element_states(el)
        matches.append(entry)
        if len(matches) >= max_results:
            break

    if not matches:
        return {"found": False, "message": f"No OS element matching name='{name}'"}
    return {"found": True, "matches": matches}


# ── Public: build registry ───────────────────────────────────────────

def build_osworld_registry(
    a11y_tree_xml: str = "",
    instruction: str = "",
    terminal_output: str = "",
) -> ToolRegistry:
    """Create a ToolRegistry with OSWorld tools bound to the current state.

    Parameters
    ----------
    a11y_tree_xml : str
        Raw XML accessibility tree from the OS.
    instruction : str
        Task instruction.
    terminal_output : str
        Latest terminal output.

    Returns
    -------
    ToolRegistry
    """
    reg = ToolRegistry(domain="osworld")

    reg.register(TOOL_QUERY_OS_ELEMENT, lambda **kw: _h_query_os_element(a11y_tree_xml, **kw))

    reg.register(TOOL_QUERY_ENTITY_POS, lambda **kw: _h_query_os_element(a11y_tree_xml, **kw))

    def _os_state_flags(**kw: Any) -> dict:
        return {
            "instruction": instruction,
            "has_terminal": bool(terminal_output),
            "terminal_last_line": terminal_output.strip().splitlines()[-1] if terminal_output.strip() else None,
        }
    reg.register(TOOL_GET_STATE_FLAGS, _os_state_flags)

    return reg


__all__ = [
    "TOOL_QUERY_OS_ELEMENT",
    "build_osworld_registry",
]
