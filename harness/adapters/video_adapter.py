"""`video` adapter — short-video evidence-grounded reasoning execution path.

Short-video is the **first transfer arena** under
PLAN-EVAL-FIRST-TARGET — game-learned protocols (e.g.
`collect_evidence_chain`, `disambiguate_target`) earn their first
non-game `verified_domains` entry here through the few-shot adapter
(PLAN-UNIFIED-SKILL-GATE Stage 3a). Real env binding (clip indexing,
frame loaders, multi-hop video evidence) is plugged in via
`set_executor()`; until then the deterministic stub keeps the
adapter-binding handshake testable.
"""

from __future__ import annotations

from common.enums import SkillType
from harness.adapters._stub_base import HopExecutor, StubTransferTargetAdapter


class VideoAdapter(StubTransferTargetAdapter):
    name = "video"
    supported_types = (
        SkillType.REASONING,
        SkillType.GROUNDING,
        SkillType.MIXED,
    )


def bind_video_executor(
    adapter: "VideoAdapter",
    *,
    frames=None,
    video_meta=None,
    **kwargs,
):
    """Wire a real video executor into ``adapter``.

    Thin re-export of :func:`harness.video_executor.make_video_executor`
    so callers don't need to know the concrete executor module exists
    if they only have the harness in scope. Mirrors
    :func:`harness.adapters.visual_reasoning_adapter.bind_visual_reasoning_executor`.

    Args:
      adapter: The :class:`VideoAdapter` to wire the executor into.
      frames: Optional pre-decoded frame iterable (reserved for the
        future real executor; the deterministic Stage-2 executor
        ignores it).
      video_meta: The cold-start sample's ``video_meta`` dict (carries
        ``video_path`` / ``indices`` / ``num_frames`` / ``duration_s``).
        Surfaced inside each evidence ``payload`` for traceability.
      **kwargs: Forwarded to :func:`make_video_executor` (e.g.
        ``on_unresolved="abort"``).

    Returns:
      The constructed executor (the same callable the adapter now
      holds via ``adapter.set_executor``).
    """
    from harness.video_executor import make_video_executor

    executor, _holder = make_video_executor(
        video_meta=video_meta,
        **kwargs,
    )
    adapter.set_executor(executor)
    return executor


__all__ = ["HopExecutor", "VideoAdapter", "bind_video_executor"]
