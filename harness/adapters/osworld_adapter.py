"""`osworld` adapter — os-agent execution path (stub for transfer target).

PLAN-COMPONENTS-IMPLEMENTATION Phase A.5 / Phase D — this adapter is a
*real* hop-loop executor but its concrete env binding (OSWorld /
desktop UI tooling) is plugged in later via `set_executor()`. Until
then the deterministic stub keeps the gate's transfer stage exercisable
and lets PLAN-UNIFIED-SKILL-GATE Stage 3a (few-shot adaptation) record
real adapter-binding attempts against the `osworld` target domain.
"""

from __future__ import annotations

from common.enums import SkillType
from harness.adapters._stub_base import HopExecutor, StubTransferTargetAdapter


class OsworldAdapter(StubTransferTargetAdapter):
    name = "osworld"
    supported_types = (
        SkillType.ACTION,
        SkillType.MIXED,
        SkillType.GROUNDING,
        SkillType.REASONING,
    )


__all__ = ["HopExecutor", "OsworldAdapter"]
