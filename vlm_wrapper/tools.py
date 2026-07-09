"""Unified tool definitions for VLM visual grounding via function calling.

Defines tools in OpenAI function-calling format so the VLM can request
precise information from the environment instead of hallucinating it.
Each domain (game, browser, video) registers its own tool implementations
against a shared ToolRegistry.  The tool_loop module drives the
multi-turn call/response cycle.

Architecture:
  VLM sees screenshot  →  identifies entities visually  →  calls tools
  to get exact positions, properties, relations  →  fills schema fields
  with ground-truth data instead of pixel estimation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Tool definition (OpenAI function-calling schema) ──────────────────

@dataclass
class ToolDef:
    """One tool the VLM can invoke."""
    name: str
    description: str
    parameters: dict[str, Any]
    domain: str = "universal"

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    name: str
    result: Any
    error: str | None = None

    def to_message(self, tool_call_id: str) -> dict[str, Any]:
        if self.error:
            content = json.dumps({"error": self.error})
        else:
            content = json.dumps(self.result, default=str)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }


# ── Registry ──────────────────────────────────────────────────────────

class ToolRegistry:
    """Holds tool definitions + their callable implementations.

    Subclasses (or callers) register tools via ``register()``, then the
    tool_loop uses ``definitions()`` to build the ``tools=`` param and
    ``dispatch()`` to execute a tool call.
    """

    def __init__(self, domain: str = "universal"):
        self.domain = domain
        self._tools: dict[str, ToolDef] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, tool_def: ToolDef, handler: Callable[..., Any]) -> None:
        self._tools[tool_def.name] = tool_def
        self._handlers[tool_def.name] = handler

    def definitions(self) -> list[dict[str, Any]]:
        return [t.to_openai() for t in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(name=name, result=None, error=f"Unknown tool: {name}")
        try:
            result = handler(**arguments)
            return ToolResult(name=name, result=result)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            return ToolResult(name=name, result=None, error=str(exc))

    def merge(self, other: "ToolRegistry") -> "ToolRegistry":
        merged = ToolRegistry(domain=f"{self.domain}+{other.domain}")
        for name, tdef in self._tools.items():
            merged.register(tdef, self._handlers[name])
        for name, tdef in other._tools.items():
            merged.register(tdef, other._handlers[name])
        # Preserve auxiliary attributes (e.g. ``derivation_log`` from
        # the reasoning sub-registry) so callers can fish them out of
        # the merged registry.  Last-one-wins so a registry deeper in
        # the merge chain can replace an earlier value.
        for src in (self, other):
            for k, v in vars(src).items():
                if k.startswith("_") or k == "domain":
                    continue
                setattr(merged, k, v)
        return merged


# ── Shared / universal tool definitions ──────────────────────────────

TOOL_QUERY_ENTITY_POS = ToolDef(
    name="query_entity_pos",
    description=(
        "Look up the exact position of a named entity. Returns ground-truth "
        "coordinates from the environment state (grid coords for games, "
        "pixel bbox for browser, timestamp for video). Use this instead of "
        "estimating positions from the screenshot."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity_label": {
                "type": "string",
                "description": "Label or identifier of the entity (e.g. 'player', 'box', bid '42', frame index).",
            },
        },
        "required": ["entity_label"],
    },
    domain="universal",
)

TOOL_LIST_ENTITIES = ToolDef(
    name="list_entities",
    description=(
        "List all entities currently visible in the environment with their "
        "types, labels, and positions. Cheaper and more accurate than "
        "trying to enumerate everything from the screenshot alone."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filter_type": {
                "type": "string",
                "enum": ["all", "interactive", "container", "text", "object"],
                "description": "Filter by entity type. Default 'all'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum entities to return. Default 25.",
            },
        },
        "required": [],
    },
    domain="universal",
)

TOOL_CHECK_RELATION = ToolDef(
    name="check_relation",
    description=(
        "Check whether a spatial or semantic relation holds between two "
        "entities. Returns true/false plus the exact measurements. "
        "Use for relations that require game/page semantics beyond vision "
        "(e.g. 'blocks', 'merge_candidate', 'contains')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "entity_a": {
                "type": "string",
                "description": "First entity label, eid, or bid.",
            },
            "entity_b": {
                "type": "string",
                "description": "Second entity label, eid, or bid.",
            },
            "relation": {
                "type": "string",
                "enum": [
                    "adjacent", "contains", "blocks", "overlaps",
                    "same_row", "same_column", "merge_candidate",
                    "parent_of", "sibling",
                ],
                "description": "Relation type to check.",
            },
        },
        "required": ["entity_a", "entity_b", "relation"],
    },
    domain="universal",
)

TOOL_GET_STATE_FLAGS = ToolDef(
    name="get_state_flags",
    description=(
        "Get the current high-level state flags: progress, phase, errors, "
        "whether a dialog is open, input pending, etc. Faster and more "
        "reliable than interpreting these from pixels."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="universal",
)

TOOL_LIST_VALID_ACTIONS = ToolDef(
    name="list_valid_actions",
    description=(
        "Return the set of valid actions the agent can take right now. "
        "For games: the legal moves. For browser: actionable elements + "
        "navigation. For video: temporal navigation controls."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    domain="universal",
)
