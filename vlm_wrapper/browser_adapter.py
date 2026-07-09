"""Compatibility shim — implementation moved to :mod:`browsergym_wrapper.adapter`."""

from browsergym_wrapper.adapter import (
    browser_obs_to_schema,
    generate_label,
)

__all__ = ["generate_label", "browser_obs_to_schema"]
