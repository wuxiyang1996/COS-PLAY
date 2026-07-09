"""`visual_reasoning` adapter — image-QA / visual reasoning execution path.

The fifth transfer-target domain
(PLAN-COMPONENTS-IMPLEMENTATION Phase A.5 / Phase D).  Two executor
options are available:

* The deterministic stub inherited from
  :class:`StubTransferTargetAdapter` — used by the gate's dry-run
  replay path to satisfy the evidence-driven invariant without
  touching pixels.
* A real executor backed by the visual + reasoning tool registries in
  :mod:`visual_reasoning_wrapper.tools_visual` and
  :mod:`visual_reasoning_wrapper.tools_reasoning`.  Callers wire it in
  with :func:`bind_visual_reasoning_executor` (an alias of
  :func:`visual_reasoning_wrapper.skill_executor.bind_executor`).

The harness keeps both paths interchangeable so the few-shot
adaptation gate (G3a) can binding-test a transferred ``SkillRecord``
on a real benchmark sample without changing the harness contract.
"""

from __future__ import annotations

from common.enums import SkillType
from harness.adapters._stub_base import HopExecutor, StubTransferTargetAdapter


class VisualReasoningAdapter(StubTransferTargetAdapter):
    name = "visual_reasoning"
    supported_types = (
        SkillType.REASONING,
        SkillType.GROUNDING,
        SkillType.MIXED,
    )


def bind_visual_reasoning_executor(adapter: VisualReasoningAdapter, *, image, **kwargs):
    """Wire a real visual + reasoning tool executor into ``adapter``.

    Thin re-export of
    :func:`visual_reasoning_wrapper.skill_executor.bind_executor` so
    callers do not need to know the wrapper module exists if they only
    have the harness in scope.  Returns the constructed executor.
    """
    from visual_reasoning_wrapper.skill_executor import bind_executor

    return bind_executor(adapter, image=image, **kwargs)


__all__ = [
    "HopExecutor",
    "VisualReasoningAdapter",
    "bind_visual_reasoning_executor",
]
