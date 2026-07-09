"""Compatibility shim -- implementation lives in
:mod:`visual_reasoning_wrapper.skill_executor`.

Mirrors the convention used by :mod:`vlm_wrapper.browser_adapter` and
:mod:`vlm_wrapper.osworld_adapter` so callers that wire up ``vlm_wrapper.*``
adapters uniformly can pick up the visual_reasoning (image QA) executor
from the same namespace::

    from vlm_wrapper.visual_reasoning_adapter import (
        VisualReasoningExecutor,
        bind_executor,
        make_visual_reasoning_executor,
    )
"""

from visual_reasoning_wrapper.skill_executor import (
    VisualReasoningExecutor,
    bind_executor,
    make_visual_reasoning_executor,
)

__all__ = [
    "VisualReasoningExecutor",
    "bind_executor",
    "make_visual_reasoning_executor",
]
