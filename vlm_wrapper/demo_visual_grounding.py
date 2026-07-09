"""Deprecated shim — moved to :mod:`browsergym_wrapper.demo_visual_grounding`.

Re-exports the public surface of the BrowserGym visual-grounding demo
(Mode A direct grounding, Mode B VLM + visual tools, Mode C full
comparison) so that ``python -m vlm_wrapper.demo_visual_grounding`` and
``from vlm_wrapper.demo_visual_grounding import …`` keep working.
"""

from __future__ import annotations

from browsergym_wrapper.demo_visual_grounding import (
    main,
    run_comparison,
    run_direct_grounding,
    run_vlm_tool_loop,
    save_annotated,
)

__all__ = [
    "main",
    "run_comparison",
    "run_direct_grounding",
    "run_vlm_tool_loop",
    "save_annotated",
]


if __name__ == "__main__":
    main()
