"""Compatibility shim — BrowserGym grounding moved to
:mod:`browsergym_wrapper.grounding`, OSWorld grounding moved to
:mod:`osworld_wrapper.grounding`."""

from browsergym_wrapper.grounding import (
    grounding_image_to_schema,
    grounding_obs_to_schema,
)
from osworld_wrapper.grounding import grounding_osworld_obs_to_schema

__all__ = [
    "grounding_image_to_schema",
    "grounding_obs_to_schema",
    "grounding_osworld_obs_to_schema",
]
