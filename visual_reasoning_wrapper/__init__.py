"""Visual reasoning wrapper — design notes, benchmark set, and XSkill alignment.

This module does **not** replace ``tool_loop.visual_generate_label_with_tools`` or
``video_visual_generate_label_with_tools``; it documents how those entry points
map to a *visual-reasoning* agent and which public benchmarks we standardise on.

Motivation (XSkill, Jiang et al., 2026)
---------------------------------------
`XSkill: Continual Learning from Experience and Skills in Multimodal Agents
<https://arxiv.org/abs/2603.12056>`_ studies multimodal agents that combine
reasoning with **tool orchestration**, and argues for **visually grounded**
knowledge (experiences at the action level, skills at the task level) distilled
from rollouts.  Our stack instantiates a *single-episode* slice of that idea:

1. **Observation** — pixels are turned into a VLM-readable bundle (single image
   for image-QA; a short frame list or key frames + timestamps for video-QA).
   A strong vision-language model (e.g. **Qwen2.5-VL-32B / Qwen3-VL-30B-A3B**,
   colloquially “~35B class”) consumes that bundle plus the question text.
2. **Actions** — not low-level motor commands, but **reasoning hops**: natural
   language deliberation, **evidence** recorded in the ``<evidence>`` chain, and
   **tool calls** (sibling modules :mod:`.tools_visual`, :mod:`.tools_video`,
   :mod:`.tools_video_visual`) that return detectors, crops, captions, and
   temporal cues.  The final hop emits the structured ``<state>`` schema used
   downstream by the skill harness.

Any benchmark that provides (image or video) + question + verifiable answer fits
this pattern; the four below are **already wired** in
:mod:`visual_reasoning_wrapper.benchmarks` and
:mod:`vlm_wrapper.eval.run_eval`.

Primary benchmark quartet (2 image + 2 video)
---------------------------------------------
**Image**

1. **VisualToolBench** — tool-enabled perception, transformation, and reasoning
   on natural images (`think with images`). HuggingFace ``ScaleAI/VisualToolBench``.
   Official scoring uses rubrics; our harness records diagnostic string match on
   the first golden answer when present.
2. **TIR-Bench** — agentic thinking-with-images across 13 task families
   (arXiv:2511.01833). HuggingFace ``Agents-X/TIR-Bench`` test split.

**Video**

3. **Video-Holmes** — multi-hop multiple-choice video QA on short clips.
4. **SIV-Bench** — social-interaction video QA; complementary semantics vs. Holmes.

Operational note
----------------
Point eval or SFT jobs at these names: ``visual_toolbench``, ``tir_bench``,
``video_holmes``, ``siv_bench`` (see ``vlm_wrapper.eval.run_eval``).  Image rows
need HuggingFace ``datasets`` and a populated cache (or network on first run).
Set ``VLM_LABEL_MODEL`` (and API base URL / key) to your Qwen-VL deployment; keep
the same tool registries — the wrapper’s “actions” remain tool calls + structured
output, not raw pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

# Re-exports of the executor + registry surface so callers can do
# ``from visual_reasoning_wrapper import bind_executor`` / ``...build_visual_registry``
# instead of reaching into the sub-modules directly. These match the
# pattern already used by ``osworld_wrapper`` and ``browsergym_wrapper``.
# Lazy via ``__getattr__`` so importing this top-level module remains
# cheap (no PIL / torch / decord pull-in just to read the benchmark
# constants below).
_EXECUTOR_EXPORTS: dict[str, tuple[str, str]] = {
    # name in this namespace          (sub-module,             attr)
    "VisualReasoningExecutor":        (".skill_executor",      "VisualReasoningExecutor"),
    "bind_executor":                  (".skill_executor",      "bind_executor"),
    "make_visual_reasoning_executor": (".skill_executor",      "make_visual_reasoning_executor"),
    "VideoReasoningExecutor":         (".video_skill_executor", "VideoReasoningExecutor"),
    "bind_video_executor":            (".video_skill_executor", "bind_executor"),
    "make_video_reasoning_executor":  (".video_skill_executor", "make_video_reasoning_executor"),
    "build_visual_registry":          (".tools_visual",         "build_visual_registry"),
    "build_video_registry":           (".tools_video",          "build_video_registry"),
    "build_video_visual_registry":    (".tools_video_visual",   "build_video_visual_registry"),
    "build_reasoning_registry":       (".tools_reasoning",      "build_reasoning_registry"),
}

if TYPE_CHECKING:  # pragma: no cover - import-time hints for IDEs only
    from .skill_executor import (  # noqa: F401
        VisualReasoningExecutor,
        bind_executor,
        make_visual_reasoning_executor,
    )
    from .video_skill_executor import (  # noqa: F401
        VideoReasoningExecutor,
        bind_executor as bind_video_executor,
        make_video_reasoning_executor,
    )
    from .tools_visual import build_visual_registry  # noqa: F401
    from .tools_video import build_video_registry  # noqa: F401
    from .tools_video_visual import build_video_visual_registry  # noqa: F401
    from .tools_reasoning import build_reasoning_registry  # noqa: F401


def __getattr__(name: str) -> Any:
    if name in _EXECUTOR_EXPORTS:
        from importlib import import_module

        mod_name, attr = _EXECUTOR_EXPORTS[name]
        module = import_module(mod_name, package=__name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_EXECUTOR_EXPORTS.keys()})


Modality = Literal["image", "video"]


@dataclass(frozen=True)
class VisualReasoningBenchmark:
    """One row in the standard visual-reasoning eval matrix."""

    key: str
    modality: Modality
    short_name: str
    one_line: str


# Canonical four: matches ``visual_reasoning_wrapper.benchmarks`` + ``eval.run_eval`` dispatch.
PRIMARY_VISUAL_REASONING_BENCHMARKS: Final[tuple[VisualReasoningBenchmark, ...]] = (
    VisualReasoningBenchmark(
        key="visual_toolbench",
        modality="image",
        short_name="VisualToolBench",
        one_line="HF VTB — tool-enabled image perception / transform / reasoning.",
    ),
    VisualReasoningBenchmark(
        key="tir_bench",
        modality="image",
        short_name="TIR-Bench",
        one_line="HF TIR-Bench — 13 thinking-with-images task families (test split).",
    ),
    VisualReasoningBenchmark(
        key="video_holmes",
        modality="video",
        short_name="Video-Holmes",
        one_line="Multi-hop MC video QA over short clips.",
    ),
    VisualReasoningBenchmark(
        key="siv_bench",
        modality="video",
        short_name="SIV-Bench",
        one_line="Social-interaction video MC QA (complements Holmes).",
    ),
)

__all__ = [
    "PRIMARY_VISUAL_REASONING_BENCHMARKS",
    "VisualReasoningBenchmark",
    "Modality",
    # Executors + binding helpers (lazy via ``__getattr__``).
    "VisualReasoningExecutor",
    "bind_executor",
    "make_visual_reasoning_executor",
    "VideoReasoningExecutor",
    "bind_video_executor",
    "make_video_reasoning_executor",
    # Tool registries for direct use outside the harness.
    "build_visual_registry",
    "build_video_registry",
    "build_video_visual_registry",
    "build_reasoning_registry",
]
