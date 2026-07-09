"""Compatibility shim — implementation moved to :mod:`osworld_wrapper.adapter`."""

from osworld_wrapper.adapter import (
    generate_label,
    osworld_obs_to_schema,
)

__all__ = ["generate_label", "osworld_obs_to_schema"]
