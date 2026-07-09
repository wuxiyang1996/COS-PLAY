"""Deprecated shim — moved to :mod:`browsergym_wrapper.example_grounding`.

This module re-exports the public surface of the BrowserGym OmniParser-v2
demo from its new home so that ``python -m vlm_wrapper.example_grounding``
and ``from vlm_wrapper.example_grounding import …`` keep working.
"""

from __future__ import annotations

from browsergym_wrapper.example_grounding import (
    main,
    run_head1,
    run_head3,
)

__all__ = ["main", "run_head1", "run_head3"]


if __name__ == "__main__":
    main()
