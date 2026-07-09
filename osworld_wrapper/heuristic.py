"""OSWorld heuristic adapter: AT-SPI / UI-Automation XML → ``<state>`` schema.

Parses the namespaced accessibility-tree XML that
:meth:`desktop_env.controllers.python.PythonController.get_accessibility_tree`
returns from a real OSWorld VM into the canonical ``<state>…</state>``
schema using deterministic tree-walking — **no LLM call**.

This is the OSWorld analog of
:func:`browsergym_wrapper.heuristic.obs_to_schema` and the desktop
equivalent of :func:`gymv_wrapper.heuristic.text_to_schema`. It is the
fast / free / reproducible head used as a baseline by
:mod:`visual_grounding_tests.generate_osworld_text_schema`.

Design notes
------------
The XML emitted by ``OSWorld/desktop_env/server/main.py:_create_atspi_node``
uses **the role as the element tag** (e.g. ``<push-button>``,
``<menu-item>``, ``<frame>``) and namespaces every attribute, e.g.::

    <push-button
        name="Files"
        st:visible="true" st:showing="true" st:enabled="true"
        cp:screencoord="(60, 80)" cp:size="(64, 64)" />

with namespace map ``{st: …/state, cp: …/component, attr: …/attributes,
txt: …/text, val: …/value, act: …/action}``. We accept all three
flavours emitted by OSWorld (Ubuntu / Windows / macOS) by mapping any
known namespace to a single canonical short prefix. Roles use the
element ``tag`` (with hyphens normalised to spaces, matching pyatspi's
``getRoleName`` output).

Usage::

    from osworld_wrapper.heuristic import obs_to_schema

    schema_str = obs_to_schema(obs, step=3, task_id="osworld.install-spotify")
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple


# ── Namespace handling ────────────────────────────────────────────────

_NS_STATE = (
    "https://accessibility.ubuntu.example.org/ns/state",
    "https://accessibility.windows.example.org/ns/state",
    "https://accessibility.macos.example.org/ns/state",
)
_NS_COMPONENT = (
    "https://accessibility.ubuntu.example.org/ns/component",
    "https://accessibility.windows.example.org/ns/component",
    "https://accessibility.macos.example.org/ns/component",
)
_NS_VALUE = (
    "https://accessibility.ubuntu.example.org/ns/value",
    "https://accessibility.windows.example.org/ns/value",
    "https://accessibility.macos.example.org/ns/value",
)
_NS_TEXT = (
    "https://accessibility.ubuntu.example.org/ns/text",
    "https://accessibility.windows.example.org/ns/text",
    "https://accessibility.macos.example.org/ns/text",
)


def _ns_attr(el: ET.Element, namespaces: Iterable[str], local: str) -> Optional[str]:
    """Read ``el[{ns}local]`` for any ns in *namespaces* (first hit wins)."""
    for ns in namespaces:
        v = el.get(f"{{{ns}}}{local}")
        if v is not None:
            return v
    return None


def _strip_ns(tag: str) -> str:
    """Drop ``{uri}`` Clark notation from an Element tag."""
    if not tag or tag[0] != "{":
        return tag
    return tag.split("}", 1)[1]


# ── Role taxonomy ─────────────────────────────────────────────────────

# Roles we always skip (noise / non-interactive containers we don't
# want to enumerate as entities).  Tags from pyatspi.getRoleName() with
# spaces normalised to hyphens by the OSWorld serializer.
IGNORED_ROLES = frozenset({
    "filler", "separator", "redundant-object", "unknown",
    "panel",          # too generic / always present
    "section",
    "image",          # decorative; promote only when accompanied by name
    "label",          # capture text via parent's name instead
})

INTERACTIVE_ROLES = frozenset({
    "push-button", "toggle-button", "radio-button", "check-box",
    "menu-item", "check-menu-item", "radio-menu-item", "tearoff-menu-item",
    "link", "hyperlink",
    "text", "entry", "password-text", "edit-bar", "spin-button",
    "combo-box", "list-item", "table-cell", "tree-item",
    "slider", "scroll-bar", "tab", "page-tab",
    "icon",
})

CONTAINER_ROLES = frozenset({
    "application", "frame", "window", "dialog", "alert",
    "menu", "menu-bar", "tool-bar", "status-bar",
    "tab-list", "page-tab-list", "tree", "tree-table", "table",
    "list-box", "list", "scroll-pane", "split-pane", "viewport",
    "document-frame", "document-web", "form",
})

# Roles that are clickable from the user's POV but are classified as
# CONTAINER_ROLES above (because their *children* are usually what we
# enumerate as entities). The SoM extractor needs to include these so
# that classic-desktop-app menu bars (LibreOffice File/Edit, GIMP
# Filters, VLC Tools, …) get a numbered click target — without them,
# every step's SoM collapses onto the GNOME dock + window decorations
# and the agent has no way to open a menu at all.
#
# Empirically (Cold-start-out-osworld/2026-05-01 run): with the old
# filter every step had ~12 SoM elements, all dock/window-deco. After
# adding ``menu`` / ``menu-button`` the same a11y trees expand to
# 25-40 elements covering the actual app's menu bar.
SOM_CLICKABLE_ROLES = INTERACTIVE_ROLES | frozenset({
    "menu",          # top-level "File"/"Edit"/"View" openers
    "menu-button",   # hamburger / dropdown buttons
    "popup-menu",    # already-opened menus (anchor)
})

# Roles we surface for grounding even if they don't match the
# interactive / container set, when they have a useful name.
TEXTUAL_ROLES = frozenset({
    "heading", "paragraph", "text", "label", "static",
})


# ── Parsing helpers ───────────────────────────────────────────────────

_PAREN_NUMS_RE = re.compile(r"-?\d+")


def _parse_tuple(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse strings like ``"(60, 80)"`` into ``(60, 80)``."""
    if not s:
        return None
    nums = _PAREN_NUMS_RE.findall(s)
    if len(nums) < 2:
        return None
    try:
        return tuple(int(n) for n in nums)
    except ValueError:
        return None


def _bbox_from(el: ET.Element) -> Optional[Tuple[int, int, int, int]]:
    """Read ``cp:screencoord`` + ``cp:size`` and combine to ``(x, y, w, h)``."""
    coord = _parse_tuple(_ns_attr(el, _NS_COMPONENT, "screencoord"))
    size = _parse_tuple(_ns_attr(el, _NS_COMPONENT, "size"))
    if not coord or not size:
        return None
    if len(coord) < 2 or len(size) < 2:
        return None
    return (coord[0], coord[1], size[0], size[1])


_FLAG_NAMES = (
    "visible", "showing", "enabled", "focused", "focusable",
    "editable", "expandable", "expanded", "checked", "checkable",
    "selected", "selectable", "pressed", "armed", "active",
    "modal", "horizontal", "vertical", "indeterminate",
    "iconified", "resizable", "transient", "sensitive",
)


def _states_from(el: ET.Element) -> List[str]:
    """Collect ``st:flag="true"`` boolean state names (lowercase)."""
    out: List[str] = []
    for flag in _FLAG_NAMES:
        v = _ns_attr(el, _NS_STATE, flag)
        if v is not None and v.strip().lower() == "true":
            out.append(flag)
    return out


def _value_from(el: ET.Element) -> Optional[str]:
    """Read the ``val:value`` numeric / scalar attribute, if any."""
    v = _ns_attr(el, _NS_VALUE, "value")
    if v is None:
        return None
    return v.strip()[:60]


# ── Element accumulator ───────────────────────────────────────────────

@dataclass
class _Entity:
    eid: str
    etype: str
    label: str
    role: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    states: List[str] = field(default_factory=list)
    value: Optional[str] = None
    text: Optional[str] = None


def _classify_etype(role: str, states: List[str]) -> str:
    """Map a role + state set onto our schema's ``type=`` field."""
    if role in CONTAINER_ROLES:
        return "container" if role not in {"frame", "window", "dialog"} else "window"
    if role in TEXTUAL_ROLES:
        return "text"
    if role in INTERACTIVE_ROLES:
        return "control"
    return "element"


def _truncate(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    s = " ".join(s.split())
    return s[:n]


def _collect_entities(
    root: ET.Element,
    *,
    max_entities: int,
    only_visible: bool,
) -> List[_Entity]:
    """Walk the tree and produce a flat list of entities (in tree order)."""
    out: List[_Entity] = []
    counter = 1
    for el in root.iter():
        role = _strip_ns(el.tag).lower()
        if not role or role == "root":
            continue
        if role in IGNORED_ROLES:
            continue
        if role not in INTERACTIVE_ROLES and role not in CONTAINER_ROLES \
                and role not in TEXTUAL_ROLES:
            continue

        states = _states_from(el)
        if only_visible and "showing" not in states and "visible" not in states:
            continue

        name = (el.get("name") or "").strip()
        if not name and role not in CONTAINER_ROLES:
            continue

        bbox = _bbox_from(el)
        if only_visible and bbox is not None and (bbox[2] <= 0 or bbox[3] <= 0):
            continue

        ent = _Entity(
            eid=f"e{counter}",
            etype=_classify_etype(role, states),
            label=_truncate(name, 60) or role,
            role=role,
            bbox=bbox,
            states=states,
            value=_value_from(el),
            text=_truncate(el.text, 80) if el.text else None,
        )
        out.append(ent)
        counter += 1
        if counter > max_entities:
            break
    return out


# ── Heuristic targets / actions ───────────────────────────────────────

def _pick_target(
    entities: List[_Entity],
    instruction: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Cheap lexical match: pick the first interactive entity whose label
    appears in the instruction (case-insensitive). Falls back to the
    first interactive entity. Returns ``(target_eid, blocker_eid)``."""
    if not entities:
        return None, None
    instr_lower = (instruction or "").lower()
    interactive = [e for e in entities if e.role in INTERACTIVE_ROLES]
    if instr_lower:
        for e in interactive:
            if e.label and e.label.lower() in instr_lower:
                return e.eid, None
    return (interactive[0].eid if interactive else entities[0].eid), None


def _suggest_actions(
    entities: List[_Entity],
    target_eid: Optional[str],
) -> List[str]:
    """Emit pyautogui calls hitting the target entity's centroid."""
    if not target_eid:
        return ["pyautogui.press('escape')", "WAIT"]
    for e in entities:
        if e.eid == target_eid and e.bbox is not None:
            x, y, w, h = e.bbox
            cx, cy = x + max(1, w // 2), y + max(1, h // 2)
            actions = [
                f"pyautogui.click({cx}, {cy})  # {e.role}: {e.label}",
            ]
            if e.role in {"text", "entry", "password-text", "edit-bar"}:
                actions.append("pyautogui.typewrite('…')")
            return actions
    return ["WAIT"]


# ── Public entry points ───────────────────────────────────────────────

def xml_to_schema(
    accessibility_tree_xml: str,
    *,
    instruction: str = "",
    task_id: str = "",
    step: int = 0,
    last_action: str = "",
    last_action_error: str = "",
    terminal_output: str = "",
    max_entities: int = 25,
    only_visible: bool = True,
) -> str:
    """Convert a raw OSWorld accessibility-tree XML string into the canonical
    ``<state>...</state>`` schema (``domain=desktop``).

    The XML is the string returned by ``controller.get_accessibility_tree()``
    (== the ``"AT"`` field of the VM's ``GET /accessibility`` response).
    Emits an empty schema body — but a well-formed envelope — when the
    XML is missing or unparseable so callers can use the result as a
    no-op placeholder.

    Parameters
    ----------
    accessibility_tree_xml : str
        Raw namespaced AT-SPI / UI-Automation XML.
    instruction : str
        Task instruction (used for cheap lexical target selection and
        to populate the ``goal=`` header).
    task_id : str
        Identifier such as ``"osworld.install-spotify"``.
    step : int
        Current step number.
    last_action / last_action_error : str
        Previous pyautogui action and any error raised running it.
    terminal_output : str
        Recent terminal output (last line is summarised in
        ``<state_flags>``).
    max_entities : int
        Hard cap on emitted entities (default 25).
    only_visible : bool
        Skip nodes that aren't both ``st:visible="true"`` and
        ``st:showing="true"`` (default True).
    """
    goal = _truncate(instruction, 240) or "unknown"
    err = _truncate(last_action_error, 120) or "null"

    lines: List[str] = ["<state>"]
    lines.append("domain=desktop")
    lines.append(f"task={task_id}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")

    entities: List[_Entity] = []
    parse_warning: Optional[str] = None
    if accessibility_tree_xml and accessibility_tree_xml.strip():
        try:
            root = ET.fromstring(accessibility_tree_xml)
        except ET.ParseError as exc:
            parse_warning = f"xml_parse_failed: {exc}"
            root = None
        if root is not None:
            entities = _collect_entities(
                root, max_entities=max_entities, only_visible=only_visible,
            )

    target_eid, blocker_eid = _pick_target(entities, instruction)

    lines.append("<entities>")
    for e in entities:
        parts = [f"type={e.etype}", f"label={e.label}"]
        if e.bbox:
            x, y, w, h = e.bbox
            parts.append(f"pos={x},{y},{w},{h}")
        parts.append(f"role={e.role}")
        lines.append(f"{e.eid}[{', '.join(parts)}]")
    lines.append("")

    lines.append("<attributes>")
    for e in entities:
        if e.value is not None:
            lines.append(f"{e.eid}.value={e.value}")
        if e.text:
            lines.append(f"{e.eid}.text={e.text}")
        if e.states:
            lines.append(f"{e.eid}.state={','.join(e.states)}")
    lines.append("")

    lines.append("<state_flags>")
    lines.append("progress=null")
    lines.append("phase=null")
    lines.append(f"error={err}")
    dialog_open = any(e.role in {"dialog", "alert"} for e in entities)
    lines.append(f"dialog_open={'true' if dialog_open else 'false'}")
    input_pending = any(
        "focused" in e.states and e.role in {"text", "entry", "password-text", "edit-bar"}
        for e in entities
    )
    lines.append(f"input_pending={'true' if input_pending else 'false'}")
    lines.append(f"n_entities={len(entities)}")
    if terminal_output:
        last_terminal = terminal_output.strip().splitlines()
        if last_terminal:
            lines.append(f"terminal_tail={_truncate(last_terminal[-1], 120)}")
    if last_action:
        lines.append(f"last_action={_truncate(last_action, 80)}")
    if parse_warning:
        lines.append(f"warning={parse_warning}")
    lines.append("")

    lines.append("<targets>")
    lines.append(f"target={target_eid or 'null'}")
    lines.append(f"blocker={blocker_eid or 'null'}")
    interactive_eids = [
        e.eid for e in entities if e.role in INTERACTIVE_ROLES
    ][:8]
    lines.append(f"candidate_set=[{','.join(interactive_eids)}]")
    lines.append("")

    actions = _suggest_actions(entities, target_eid)
    if actions:
        lines.append("<actions>")
        for i, act in enumerate(actions, 1):
            lines.append(f"a{i}={act}")
        lines.append("")

    lines.append("</state>")
    return "\n".join(lines)


def obs_to_schema(
    obs: dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    max_entities: int = 25,
    only_visible: bool = True,
) -> str:
    """Convenience wrapper that unpacks an :class:`OSWorldGymWrapper` obs dict.

    Reads ``accessibility_tree`` (XML string), ``instruction``,
    ``terminal``, plus optional ``last_action`` / ``last_action_error``
    keys, and forwards to :func:`xml_to_schema`.
    """
    return xml_to_schema(
        obs.get("accessibility_tree", "") or "",
        instruction=obs.get("instruction", "") or "",
        task_id=task_id,
        step=step,
        last_action=obs.get("last_action", "") or "",
        last_action_error=obs.get("last_action_error", "") or "",
        terminal_output=obs.get("terminal", "") or "",
        max_entities=max_entities,
        only_visible=only_visible,
    )


__all__ = ["obs_to_schema", "xml_to_schema"]
