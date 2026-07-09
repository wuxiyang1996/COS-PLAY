"""Compatibility shim -- implementation lives in
:mod:`visual_reasoning_wrapper.video_skill_executor`.

Mirrors the convention used by :mod:`vlm_wrapper.browser_adapter`,
:mod:`vlm_wrapper.osworld_adapter`, and
:mod:`vlm_wrapper.visual_reasoning_adapter` so callers that wire up
``vlm_wrapper.*`` adapters uniformly can pick up the video QA executor
from the same namespace::

    from vlm_wrapper.video_adapter import (
        VideoReasoningExecutor,
        bind_executor,
        make_video_reasoning_executor,
    )

The executor binds the merged
:func:`visual_reasoning_wrapper.tools_video_visual.build_video_visual_registry`
(video navigation + single-frame visual + cross-frame analysis +
reasoning tools) to the harness ``VideoAdapter``. See the source module
docstring for the full action -> tool dispatch table.
"""

from visual_reasoning_wrapper.video_skill_executor import (
    VideoReasoningExecutor,
    bind_executor,
    make_video_reasoning_executor,
)

__all__ = [
    "VideoReasoningExecutor",
    "bind_executor",
    "make_video_reasoning_executor",
]
