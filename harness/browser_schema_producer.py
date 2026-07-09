"""BrowserGym ``<state>...</state>`` producer.

Stage 4 cross-domain transfer (rollout memo §6.1, §11.5.5). The
BrowserGym cold-start corpus already records a ``schema_canonical``
block per step (the AXTree-heuristic head), so production simply
returns it as-is when present. The minimal-AXTree fallback exists
for runtime cases where only an ``info`` dict is available — e.g.
when a real Playwright executor lands and feeds us the post-step
browser state without going through the cold-start labeler.

Mirrors `harness/gym_schema_producer.py` shape: pure producer
(``producer(info, obs, *, step, task, goal) -> str``) plus
``make_browsergym_producer(task_prefix)`` that returns the producer
for any known prefix; ``None`` otherwise so callers can fall back.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

from harness.gym_schema_producer import SchemaProducer, render_state_block

logger = logging.getLogger("harness.browser_schema_producer")


__all__ = [
    "SchemaProducer",
    "browsergym_canonical_producer",
    "make_browsergym_producer",
    "BROWSER_TASK_PREFIXES",
]


# Known BrowserGym sub-corpora. Cold-start dump lays each task dir
# out as ``<prefix>.<rest>`` (e.g. ``assistantbench.test.92``);
# `make_browsergym_producer` keys off the prefix so the dispatcher
# can validate ``--target`` early.
BROWSER_TASK_PREFIXES: tuple[str, ...] = (
    "assistantbench",
    "miniwob",
    "webarena",
    "workarena",
    "visualwebarena",
)


_BROWSER_GOAL_FALLBACK = (
    "Drive a browser via BrowserGym tools (click / fill / scroll / "
    "select_option / go_back / goto) to complete the task."
)


def _coerce_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:                                                # noqa: BLE001
        return default


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _axtree_nodes(info: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Extract a list of AXTree-shaped nodes from ``info``.

    Accepts ``info["axtree"]["nodes"]`` (cold-start convention) or a
    flat list at ``info["axtree_nodes"]``. Returns ``[]`` when neither
    is present.
    """
    axtree = info.get("axtree")
    if isinstance(axtree, Mapping):
        nodes = axtree.get("nodes")
        if isinstance(nodes, list):
            return [n for n in nodes if isinstance(n, Mapping)]
    flat = info.get("axtree_nodes")
    if isinstance(flat, list):
        return [n for n in flat if isinstance(n, Mapping)]
    return []


def browsergym_canonical_producer(
    info: Mapping[str, Any],
    obs: Any,
    *,
    step: int = 0,
    task: str = "",
    goal: str = "",
    domain: str = "browser",
) -> str:
    """Render a ``<state>`` block for a BrowserGym step.

    Stage 4's first cut prefers ``info["schema_canonical"]`` when
    present (cold-start corpus carries the AXTree-heuristic head).
    Otherwise builds a minimal block from the AXTree node list +
    URL / focused-bid scalars.

    Args:
      info: Per-step info dict. Recognised keys:
        ``schema_canonical`` (returned as-is when present),
        ``axtree`` (``{"nodes": [{"role", "name", "bid", "bbox",
        "attributes"?}]}``) or ``axtree_nodes``, ``url``,
        ``focused_element_bid`` / ``focused_bid``, ``error_text``,
        ``dialog_open``, ``candidate_actions``.
      obs: Unused in the stub path; cold-start corpus encodes obs
        into ``info``.
      step / task / goal / domain: Header fields.
    """
    _ = obs

    canonical_pre = info.get("schema_canonical")
    if isinstance(canonical_pre, str) and "<state>" in canonical_pre:
        return canonical_pre

    nodes = _axtree_nodes(info)
    entities: List[Mapping[str, Any]] = []
    attributes: dict = {}
    affordances: dict = {}

    for i, node in enumerate(nodes, start=1):
        eid = f"e{i}"
        role = _coerce_str(node.get("role") or node.get("type"), "element")
        label = _coerce_str(node.get("name") or node.get("label"), "")
        bid = _coerce_str(node.get("bid"), "null") or "null"
        bbox = node.get("bbox") or node.get("pos")
        if isinstance(bbox, (list, tuple)) and bbox:
            pos = ",".join(str(int(c)) for c in bbox if c is not None)
        elif isinstance(bbox, str):
            pos = bbox
        else:
            pos = "null"
        entities.append({
            "id": eid, "type": role, "label": label,
            "bid": bid, "pos": pos or "null",
            "ontology": "selectable_entity",
        })
        attrs = dict(node.get("attributes") or {})
        attrs.setdefault("state", "visible")
        attributes[eid] = attrs
        affordances[eid] = ["inspect"]

    state_flags = {
        "phase": _coerce_str(info.get("phase"), "play") or "play",
        "progress": float(info.get("progress") or 0.0),
        "scene_type": "page",
        "error": _coerce_str(info.get("error_text"), "null") or "null",
        "dialog_open": "true" if _coerce_bool(info.get("dialog_open")) else "false",
        "url": _coerce_str(info.get("url"), ""),
        "focused_bid": _coerce_str(
            info.get("focused_element_bid")
            or info.get("focused_bid"),
            "null",
        ) or "null",
    }

    actions: Sequence[str] = info.get("candidate_actions") or []
    return render_state_block(
        domain=domain, task=task,
        goal=goal or _BROWSER_GOAL_FALLBACK, step=step,
        entities=entities, attributes=attributes,
        state_flags=state_flags, affordances=affordances,
        relations=None, actions=[str(a) for a in actions],
    )


def make_browsergym_producer(task_prefix: str) -> Optional[SchemaProducer]:
    """Look up a deterministic producer for one BrowserGym task prefix.

    Returns `browsergym_canonical_producer` for any prefix in
    `BROWSER_TASK_PREFIXES`; ``None`` for unknown prefixes so callers
    can fall back to the executor's plain-text path.
    """
    if not task_prefix:
        return None
    if task_prefix in BROWSER_TASK_PREFIXES:
        return browsergym_canonical_producer
    return None
