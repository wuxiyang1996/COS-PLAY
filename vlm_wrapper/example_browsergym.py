#!/usr/bin/env python3
"""Compatibility shim — implementation moved to :mod:`browsergym_wrapper.example`.

The legacy fixture helpers (``make_fake_axtree``, ``make_fake_extra_props``,
``make_fake_screenshot``, ``build_browsergym_obs``) and the ``main()`` CLI
entry point are still importable from here for backward compatibility.
Prefer ``python -m browsergym_wrapper.example`` for new code.
"""

from browsergym_wrapper.example import (
    build_browsergym_obs,
    main,
    make_fake_axtree,
    make_fake_extra_props,
    make_fake_screenshot,
)

__all__ = [
    "make_fake_axtree",
    "make_fake_extra_props",
    "make_fake_screenshot",
    "build_browsergym_obs",
    "main",
]


if __name__ == "__main__":
    main()
