"""Compatibility shim — implementation moved to :mod:`browsergym_wrapper.heuristic`."""

from browsergym_wrapper.heuristic import (
    CONTAINER_ROLES,
    IGNORED_ROLES,
    INTERACTIVE_ROLES,
    _Entity,
    _build_relations,
    _extract_entities,
    _extract_goal,
    _pick_targets,
    _suggest_actions,
    obs_to_schema,
)

__all__ = [
    "CONTAINER_ROLES",
    "IGNORED_ROLES",
    "INTERACTIVE_ROLES",
    "obs_to_schema",
]
