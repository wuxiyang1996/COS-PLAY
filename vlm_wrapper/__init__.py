"""VLM structured-state wrappers + skill executors for the five supported domains.

Five domain-specific sibling packages produce the same ``<state>...</state>``
schema (see ``plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`` §3):

* :mod:`gymv_wrapper`            — Atari / classic-control, single PIL frame
* :mod:`browsergym_wrapper`      — BrowserGym (WebArena, MiniWoB), DOM/AXTree + screenshot
* :mod:`osworld_wrapper`         — OSWorld desktop, A11y tree + screenshot
* :mod:`visual_reasoning_wrapper` (image head)
                                — VisualToolBench / TIR-Bench, single image + question
* :mod:`visual_reasoning_wrapper` (video head)
                                — Video-Holmes / SIV-Bench, frame list + question

This package (``vlm_wrapper``) is the central re-export hub: it exposes a
flat namespace of schema generators, tool registries, the ``cascaded_ground``
pipeline, and now the **per-domain skill executors** that wire the harness
adapters to the same tool registries.

Three heads produce the schema, all five domains support at least head 1+3,
and ``visual_reasoning`` adds the multi-frame video flavour of head 3:

**Head 1 - Heuristic (text-in -> schema-out)**
  Fast, free, deterministic.  Parses native text state (obs.text,
  AXTree/DOM) into the schema with regex/tree-walking.
  Good for: real-time RL rollouts, cheap baselines, validation.

**Head 2 - Vision LLM (image-in -> schema-out)**
  Sends the screenshot to GPT-4o (or any vision LLM) and receives
  the schema.  The image is the primary input; native text is
  optional grounding context.
  Good for: training-label generation, Qwen3-VL-8B distillation.

**Head 3 - OmniParser-v2 grounding (image-in -> schema-out)**
  Runs OmniParser-v2 on the screenshot to extract structured screen
  elements (boxes + roles + captions) and converts those into the
  schema deterministically.  Good for: ablation against head 2,
  hybrid pipelines (Florence-2 captions + LLM glue), and any setting
  where you want to gate on detector confidence.

Skill executors (``visual_reasoning_adapter`` / ``video_adapter`` /
``browser_adapter`` / ``osworld_adapter`` shims in this package) bind the
domain's tool registry (``build_*_registry``) to the matching harness
adapter and dispatch ``InnerAction`` hops onto concrete tool calls.  The
unified entry point :func:`vlm_wrapper.ground.cascaded_ground` exercises
the same registries from the *labeling* side; together they share one set
of tools across cold-start labelling and harness replay.

Gym-V examples::

    from vlm_wrapper import gymv_heuristic_schema  # head 1
    schema = gymv_heuristic_schema(obs_text="...", description="...", task_id="Game2048-v0")

    from vlm_wrapper import gymv_generate_label  # head 2
    result = gymv_generate_label(frame, goal="Reach 2048", task_id="Game2048-v0")

BrowserGym examples::

    from vlm_wrapper import browser_heuristic_schema  # head 1
    schema = browser_heuristic_schema(obs, step=3, task_id="webarena.shopping.143")

    from vlm_wrapper import browser_obs_to_schema  # head 2
    result = browser_obs_to_schema(obs, step=3, task_id="webarena.shopping.143")

Visual reasoning examples (image + video heads of head 3)::

    # Image head: bind the harness VisualReasoningAdapter to the merged
    # OmniParser-v2 / detection / reasoning registry for one PIL.Image.
    from vlm_wrapper.visual_reasoning_adapter import bind_executor
    executor = bind_executor(adapter, image=pil_frame)

    # Video head: ditto for the harness VideoAdapter, against a frame list
    # or a video file path -- builds the merged video + visual + reasoning
    # registry under the hood.
    from vlm_wrapper.video_adapter import bind_executor as bind_video_executor
    executor = bind_video_executor(adapter, video_path="clip.mp4", num_frames=8)
"""

# ── Unified grounding pipeline ────────────────────────────────────────
from vlm_wrapper.ground import (
    GroundingRequest,
    GroundingResult,
    HopTrace,
    cascaded_ground,
    ground,
)

# ── Shared utilities ──────────────────────────────────────────────────
from vlm_wrapper.schema import (
    SCHEMA_VERSION,
    ValidationResult,
    build_adaptive_system_prompt,
    build_system_prompt,
    encode_image_b64,
    parse_answer_block,
    parse_answer_from_schema,
    parse_evidence_from_schema,
    parse_schema_output,
    semantic_validate,
    validate_schema,
)

# ── Head 1: Heuristic (text-in → schema-out) ─────────────────────────
# NOTE: the ``gymv_*`` / ``browser_*`` / ``osworld_*`` symbols are exposed
# lazily via ``__getattr__`` (PEP 562) below.  This is what lets
# ``browsergym_wrapper.adapter`` / ``osworld_wrapper.adapter`` do
# ``from vlm_wrapper.schema import …`` without tripping a circular import
# through this package's re-exports.

__all__ = [
    # unified pipeline
    "ground",
    "cascaded_ground",
    "GroundingRequest",
    "GroundingResult",
    "HopTrace",
    # shared
    "SCHEMA_VERSION",
    "ValidationResult",
    "build_adaptive_system_prompt",
    "build_system_prompt",
    "encode_image_b64",
    "parse_answer_block",
    "parse_answer_from_schema",
    "parse_evidence_from_schema",
    "parse_schema_output",
    "semantic_validate",
    "validate_schema",
    # head 1: heuristic
    "gymv_heuristic_schema",
    "browser_heuristic_schema",
    # head 2: vision
    "gymv_generate_label",
    "browser_generate_label",
    "browser_obs_to_schema",
    "osworld_generate_label",
    "osworld_obs_to_schema",
]

__all__.append("GymVSchemaWrapper")


# PEP 562 lazy re-exports for the gymv_wrapper / browsergym_wrapper /
# osworld_wrapper symbols. Defining these here (instead of eager top-level
# imports) breaks the circular dependency that would otherwise occur when
# any of those sibling packages is imported first and their
# ``from vlm_wrapper.schema import ...`` triggers this module before the
# sibling has finished initialising.
def __getattr__(name):  # noqa: D401
    if name == "gymv_heuristic_schema":
        from gymv_wrapper.heuristic import text_to_schema as _f
        return _f
    if name == "gymv_generate_label":
        from gymv_wrapper.adapter import generate_label as _f
        return _f
    if name == "GymVSchemaWrapper":
        from gymv_wrapper.adapter import GymVSchemaWrapper as _cls
        return _cls
    if name == "build_gymv_registry":
        from gymv_wrapper.tools import build_gymv_registry as _f
        return _f
    # ── BrowserGym ────────────────────────────────────────────────
    if name == "browser_heuristic_schema":
        from browsergym_wrapper.heuristic import obs_to_schema as _f
        return _f
    if name == "browser_generate_label":
        from browsergym_wrapper.adapter import generate_label as _f
        return _f
    if name == "browser_obs_to_schema":
        from browsergym_wrapper.adapter import browser_obs_to_schema as _f
        return _f
    if name == "build_browser_registry":
        from browsergym_wrapper.tools import build_browser_registry as _f
        return _f
    # ── OSWorld ───────────────────────────────────────────────────
    if name == "osworld_generate_label":
        from osworld_wrapper.adapter import generate_label as _f
        return _f
    if name == "osworld_obs_to_schema":
        from osworld_wrapper.adapter import osworld_obs_to_schema as _f
        return _f
    if name == "build_osworld_registry":
        from osworld_wrapper.tools import build_osworld_registry as _f
        return _f
    # ── visual_reasoning_wrapper ──────────────────────────────────
    if name == "build_video_registry":
        from visual_reasoning_wrapper.tools_video import build_video_registry as _f
        return _f
    if name == "build_visual_registry":
        from visual_reasoning_wrapper.tools_visual import build_visual_registry as _f
        return _f
    if name == "build_video_visual_registry":
        from visual_reasoning_wrapper.tools_video_visual import (
            build_video_visual_registry as _f,
        )
        return _f
    # ── visual_reasoning_wrapper (skill executors -- harness binding) ──
    if name == "VisualReasoningExecutor":
        from visual_reasoning_wrapper.skill_executor import VisualReasoningExecutor as _cls
        return _cls
    if name == "bind_visual_executor":
        from visual_reasoning_wrapper.skill_executor import bind_executor as _f
        return _f
    if name == "make_visual_reasoning_executor":
        from visual_reasoning_wrapper.skill_executor import (
            make_visual_reasoning_executor as _f,
        )
        return _f
    if name == "VideoReasoningExecutor":
        from visual_reasoning_wrapper.video_skill_executor import VideoReasoningExecutor as _cls
        return _cls
    if name == "bind_video_executor":
        from visual_reasoning_wrapper.video_skill_executor import bind_executor as _f
        return _f
    if name == "make_video_reasoning_executor":
        from visual_reasoning_wrapper.video_skill_executor import (
            make_video_reasoning_executor as _f,
        )
        return _f
    # ── OmniParser-v2 grounding (lazy because of torch / transformers) ──
    if name in _GROUNDING_EXPORTS:
        return _load_grounding_export(name)
    if name in _VISUAL_REASONING_EXPORTS:
        return _load_visual_reasoning_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── Head 3: Grounding (image-in → OmniParser-v2 → schema-out) ────────
# Lazy-loaded to avoid pulling torch / transformers on every ``import vlm_wrapper``.
_GROUNDING_EXPORTS = {
    "parse_screen",
    "parse_screen_annotated",
    "ScreenElement",
    "BBox",
    "grounding_image_to_schema",
    "grounding_obs_to_schema",
    "grounding_osworld_obs_to_schema",
}


def _load_grounding_export(name):
    if name in {"parse_screen", "parse_screen_annotated", "ScreenElement", "BBox"}:
        from vlm_wrapper import grounding as _g
        return getattr(_g, name)
    if name in {"grounding_image_to_schema", "grounding_obs_to_schema"}:
        from browsergym_wrapper import grounding as _g
        return getattr(_g, name)
    if name == "grounding_osworld_obs_to_schema":
        from osworld_wrapper.grounding import grounding_osworld_obs_to_schema as _f
        return _f
    raise AttributeError(name)


__all__ += sorted(_GROUNDING_EXPORTS)

# ── Tool-calling infrastructure ──────────────────────────────────────
from vlm_wrapper.tools import ToolRegistry, ToolDef, ToolResult

# Domain-specific registries — gymv / browser / osworld are all lazily
# re-exported via __getattr__ above so that importing
# ``browsergym_wrapper.adapter`` first does not trigger an eager
# ``vlm_wrapper.tools_browser`` load (which would in turn pull every
# heavy sibling).

# Tool-calling loop + convenience wrappers
from vlm_wrapper.tool_loop import (
    run_tool_loop,
    gymv_generate_label_with_tools,
    browser_generate_label_with_tools,
    video_generate_label_with_tools,
    visual_generate_label_with_tools,
    video_visual_generate_label_with_tools,
)

__all__ += [
    # tool infrastructure
    "ToolRegistry",
    "ToolDef",
    "ToolResult",
    # domain registries
    "build_gymv_registry",
    "build_browser_registry",
    "build_osworld_registry",
    "build_video_registry",
    "build_visual_registry",
    "build_video_visual_registry",
    # tool-calling loop
    "run_tool_loop",
    "gymv_generate_label_with_tools",
    "browser_generate_label_with_tools",
    "video_generate_label_with_tools",
    "visual_generate_label_with_tools",
    "video_visual_generate_label_with_tools",
    # skill executors -- harness binding for the visual_reasoning + video heads
    "VisualReasoningExecutor",
    "bind_visual_executor",
    "make_visual_reasoning_executor",
    "VideoReasoningExecutor",
    "bind_video_executor",
    "make_video_reasoning_executor",
]

# ── Benchmark loaders + parsers (lazily re-exported from visual_reasoning_wrapper)
_VISUAL_REASONING_EXPORTS = {
    "SIVBenchSample",
    "TIRBenchSample",
    "VideoHolmesSample",
    "VisualToolBenchSample",
    "default_siv_bench_root",
    "default_tir_bench_root",
    "default_video_holmes_root",
    "default_visual_toolbench_root",
    "iter_siv_bench_samples",
    "iter_tir_bench_samples",
    "iter_video_holmes_samples",
    "iter_visual_toolbench_samples",
    "load_siv_bench_questions",
    "load_tir_bench_image",
    "load_video_holmes_questions",
    "load_visual_toolbench_image",
    "parse_siv_bench_sample",
    "parse_tir_bench_sample",
    "parse_video_holmes_sample",
    "parse_visual_toolbench_sample",
    "sample_video_frames",
}


def _load_visual_reasoning_export(name):
    from visual_reasoning_wrapper import benchmarks as _benchmarks
    return getattr(_benchmarks, name)


__all__ += sorted(_VISUAL_REASONING_EXPORTS)

# ── Visual reasoning (design notes + standard 2×2 benchmark matrix) ─
from visual_reasoning_wrapper import (
    PRIMARY_VISUAL_REASONING_BENCHMARKS,
    VisualReasoningBenchmark,
)

__all__ += [
    "PRIMARY_VISUAL_REASONING_BENCHMARKS",
    "VisualReasoningBenchmark",
]
