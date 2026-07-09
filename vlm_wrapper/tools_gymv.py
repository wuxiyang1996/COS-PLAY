"""Compatibility shim — implementation moved to :mod:`gymv_wrapper.tools`."""

from gymv_wrapper.tools import (
    TOOL_CHECK_DEADLOCK,
    TOOL_COUNT_MERGE_CANDIDATES,
    TOOL_GET_GRID,
    TOOL_SPATIAL_ANALYSIS,
    build_gymv_registry,
)

__all__ = [
    "TOOL_GET_GRID",
    "TOOL_CHECK_DEADLOCK",
    "TOOL_SPATIAL_ANALYSIS",
    "TOOL_COUNT_MERGE_CANDIDATES",
    "build_gymv_registry",
]
