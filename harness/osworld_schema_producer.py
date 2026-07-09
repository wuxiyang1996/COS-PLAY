"""Deterministic ``<state>...</state>`` producer for OSWorld desktop
tasks.

Mirror of :mod:`harness.gym_schema_producer` for the OSWorld transfer
target (rollout memo §6.1, §11.5.5). Two heads:

  1.  Pass-through: when ``info["schema_canonical"]`` is already a
      fully-formed ``<state>...</state>`` block (the cold-start AT-SPI
      heuristic head emitted by
      ``Cold-start-out-osworld/<ts>/<domain>/<task>/episode_*.json``),
      return it verbatim — that dump already has fully-resolved entity
      bid / pos / role columns the first-cut producer cannot
      synthesise.
  2.  Synthetic: otherwise build a minimal block from a structured
      ``info`` dict (``a11y_tree.nodes`` → entities, ``progress`` /
      ``error_text`` / ``dialog_open`` → state_flags, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

from harness.gym_schema_producer import SchemaProducer, render_state_block

logger = logging.getLogger("harness.osworld_schema_producer")


__all__ = [
    "OSWORLD_DOMAINS",
    "make_osworld_producer",
    "osworld_canonical_producer",
]


# Canonical OSWorld domain set we accept in the registry. Anything else
# returns ``None`` from :func:`make_osworld_producer` so callers can
# fall back to the executor's plain dispatch path.
OSWORLD_DOMAINS: frozenset[str] = frozenset({
    "vlc",
    "vs_code",
    "gimp",
    "chrome",
    "libreoffice_calc",
    "libreoffice_impress",
    "libreoffice_writer",
    "os",
    "thunderbird",
    "multi_apps",
})


_OSWORLD_GOAL_DEFAULT = (
    "Drive a desktop application to satisfy the OSWorld task. Issue "
    "primitive UI actions (click, type, hotkey, scroll) against the "
    "AT-SPI accessibility tree and observe the resulting state."
)


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:                                              # noqa: BLE001
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(default)


def _entities_from_a11y_tree(
    nodes: Any,
) -> tuple[List[Mapping[str, Any]], dict]:
    """Project AT-SPI tree nodes into the entity / attributes pair the
    cold-start ``<state>`` block uses. Returns ``([], {})`` when
    ``nodes`` isn't a non-empty list.
    """

    if not isinstance(nodes, list) or not nodes:
        return [], {}
    entities: List[Mapping[str, Any]] = []
    attributes: dict = {}
    for i, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        eid = f"e{i + 1}"
        role = _coerce_str(node.get("role"), "object")
        label = _coerce_str(node.get("name") or node.get("label"), "")
        bid = node.get("som_id") or node.get("bid")
        bid_s = _coerce_str(bid, "null") if bid is not None else "null"
        pos = node.get("bbox") or node.get("pos")
        pos_s = _coerce_str(pos, "null") if pos is not None else "null"
        entities.append({
            "id": eid,
            "type": role,
            "label": label,
            "bid": bid_s,
            "pos": pos_s,
            "ontology": "selectable_entity",
        })
        attrs: dict = {}
        if node.get("state") is not None:
            attrs["state"] = node["state"]
        if node.get("value") is not None:
            attrs["value"] = node["value"]
        if attrs:
            attributes[eid] = attrs
    return entities, attributes


def osworld_canonical_producer(
    info: Mapping[str, Any],
    obs: Any = None,
    *,
    step: int = 0,
    task: str = "",
    goal: str = "",
    domain: str = "osworld",
) -> str:
    """Emit a ``<state>...</state>`` block for an OSWorld step.

    Pass-through path: ``info["schema_canonical"]`` returned verbatim
    when present (cold-start AT-SPI head). Synthetic path: build a
    minimal block from optional ``info`` keys —
    ``a11y_tree.nodes`` (entities), ``progress`` / ``scene_type`` /
    ``error_text`` / ``dialog_open`` (state_flags),
    ``candidate_actions`` (actions list).
    """

    # Pass-through head: the cold-start AT-SPI dump.
    sc = info.get("schema_canonical") if isinstance(info, Mapping) else None
    if isinstance(sc, str) and "<state>" in sc and "</state>" in sc:
        return sc

    # Synthetic head: build a minimal block.
    a11y = (info.get("a11y_tree") if isinstance(info, Mapping) else None) or {}
    nodes = a11y.get("nodes") if isinstance(a11y, Mapping) else None
    entities, attributes = _entities_from_a11y_tree(nodes)

    if not entities:
        # Always emit at least one container entity so the parser has
        # something concrete and the eligibility filter doesn't trip
        # on an empty ``elements`` list.
        entities.append({
            "id": "e1",
            "type": "region",
            "label": "desktop",
            "bid": "null",
            "pos": "null",
            "ontology": "container_entity",
        })
        attributes["e1"] = {"state": "visible"}

    progress = _coerce_float(info.get("progress") if isinstance(info, Mapping) else 0.0)
    scene_type = _coerce_str(
        info.get("scene_type") if isinstance(info, Mapping) else None,
        "desktop",
    )
    error_text = info.get("error_text") if isinstance(info, Mapping) else None
    dialog_open = _coerce_bool(
        info.get("dialog_open") if isinstance(info, Mapping) else False
    )

    state_flags = {
        "phase": "play",
        "progress": progress,
        "scene_type": scene_type,
        "error": _coerce_str(error_text, "null") if error_text else "null",
        "dialog_open": "true" if dialog_open else "false",
        "input_pending": "true",
    }

    actions_raw: Any = (
        info.get("candidate_actions") if isinstance(info, Mapping) else None
    )
    if not isinstance(actions_raw, list):
        actions_raw = []

    return render_state_block(
        domain=domain,
        task=task or _coerce_str(info.get("task"), ""),
        goal=goal or _OSWORLD_GOAL_DEFAULT,
        step=step,
        entities=entities,
        attributes=attributes,
        state_flags=state_flags,
        affordances=None,
        relations=None,
        actions=[_coerce_str(a, "") for a in actions_raw if a],
    )


def make_osworld_producer(domain_name: str) -> Optional[SchemaProducer]:
    """Return :func:`osworld_canonical_producer` for any domain in
    :data:`OSWORLD_DOMAINS`; ``None`` otherwise so callers can fall
    back to the plain executor path. The producer is the same callable
    for every supported domain — OSWorld doesn't yet need per-app
    heuristics because the AT-SPI head already surfaces app-specific
    attributes through the schema_canonical pass-through.
    """

    if not domain_name or domain_name not in OSWORLD_DOMAINS:
        return None
    return osworld_canonical_producer
