"""Shared scaffolding for the SFT / GRPO actor flavours.

The two specialised actors under :mod:`decision_agents.SFT` and
:mod:`decision_agents.grpo` differ only in (a) which LLM backend they
hit, (b) what artefact they record per step, and (c) whether the LLM
call is sync (GPT-4o) or async (vLLM).

Everything they share — schema parsing, skill interface, skill
tracker, reward computation, action-prompt builder, and the per-task
:class:`Harness` that defines what an action *means* — already lives
at :mod:`decision_agents.<module>` (or here in :mod:`decision_agents.core`)
and is re-imported below for convenience.

The :class:`Harness` family unifies five task families behind one
:class:`~decision_agents.actor_agent.ActorAgent` MDP loop:

* :class:`GymHarness`     — game / Gymnasium-shaped envs (mutable world)
* :class:`BrowserHarness` — web / Playwright (mutable world; ``step`` stub)
* :class:`OSWorldHarness` — desktop / OSWorld (mutable world; ``step`` stub)
* :class:`VRHarness`      — visual reasoning (read-only image, scratchpad ops)
* :class:`VideoHarness`   — video understanding (read-only clip + cursor)
"""

from __future__ import annotations

from decision_agents.core.harness import (
    Harness,
    HarnessState,
    parse_op_call,
)
from decision_agents.core.harness_browser import BrowserHarness
from decision_agents.core.harness_gym import GymHarness
from decision_agents.core.harness_osworld import OSWorldHarness
from decision_agents.core.harness_video import VIDEO_OPS, VideoHarness
from decision_agents.core.harness_vr import VR_OPS, VRHarness
from decision_agents.core.multimodal import (
    VisualInput,
    build_openai_vision_messages,
    build_qwen_vl_messages,
    load_image_as_data_url,
)
from decision_agents.core.perception import (
    Detection,
    EvidenceCache,
    MockOCR,
    MockRegionDetector,
    MockSegmenter,
    OCREngine,
    OCRResult,
    RegionDetector,
    Segmentation,
    Segmenter,
)

__all__ = [
    # Multimodal scaffolding
    "VisualInput",
    "build_openai_vision_messages",
    "build_qwen_vl_messages",
    "load_image_as_data_url",
    # Harness family
    "Harness",
    "HarnessState",
    "GymHarness",
    "BrowserHarness",
    "OSWorldHarness",
    "VRHarness",
    "VideoHarness",
    "VR_OPS",
    "VIDEO_OPS",
    "parse_op_call",
    # Perception (Phase 8.0)
    "RegionDetector",
    "Segmenter",
    "OCREngine",
    "MockRegionDetector",
    "MockSegmenter",
    "MockOCR",
    "Detection",
    "Segmentation",
    "OCRResult",
    "EvidenceCache",
]
