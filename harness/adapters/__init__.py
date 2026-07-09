"""Concrete `SkillAdapter` implementations.

Foundry domain (PLAN-COMPONENTS-IMPLEMENTATION §4 Phase A):
  - `gymv` — game-env (gym-v) adapter; the **source domain** under
    PLAN-SKILL-BANK §0.4 where every skill is first mined and hardened.

Transfer-target domains (PLAN-COMPONENTS-IMPLEMENTATION §4 Phase A.5 /
Phase D — the few-shot transfer arenas that earn `verified_domains`
entries through PLAN-UNIFIED-SKILL-GATE Stage 3a):
  - `browser` — webagent
  - `osworld` — os-agent
  - `video` — short-video evidence-grounded reasoning (first transfer arena)
  - `visual_reasoning` — image-QA / visual reasoning

Each adapter is deliberately thin: it translates
`SkillRecord.protocol` hops into adapter-native tool calls and returns
an `AdapterRunResult`. Real env binding lives in `vlm_wrapper/`; we
depend on it via late imports so the harness package can be imported
(and tested) without those heavy deps installed.
"""

from harness.adapters.browser_adapter import BrowserAdapter
from harness.adapters.gymv_adapter import GymvAdapter
from harness.adapters.osworld_adapter import OsworldAdapter
from harness.adapters.video_adapter import VideoAdapter
from harness.adapters.visual_reasoning_adapter import VisualReasoningAdapter

__all__ = [
    "BrowserAdapter",
    "GymvAdapter",
    "OsworldAdapter",
    "VideoAdapter",
    "VisualReasoningAdapter",
]
