"""Compatibility shim — browser tools moved to :mod:`browsergym_wrapper.tools`,
OSWorld tools moved to :mod:`osworld_wrapper.tools`."""

from browsergym_wrapper.tools import (
    TOOL_GET_ELEMENT_TREE,
    TOOL_GET_PAGE_INFO,
    TOOL_GET_SOM_ELEMENTS,
    TOOL_QUERY_ELEMENT_BBOX,
    TOOL_SEARCH_ELEMENTS,
    build_browser_registry,
)
from osworld_wrapper.tools import (
    TOOL_QUERY_OS_ELEMENT,
    build_osworld_registry,
)

__all__ = [
    "TOOL_QUERY_ELEMENT_BBOX",
    "TOOL_SEARCH_ELEMENTS",
    "TOOL_GET_PAGE_INFO",
    "TOOL_GET_ELEMENT_TREE",
    "TOOL_GET_SOM_ELEMENTS",
    "TOOL_QUERY_OS_ELEMENT",
    "build_browser_registry",
    "build_osworld_registry",
]
