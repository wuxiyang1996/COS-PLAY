"""Unified visual grounding pipeline.

Single entry point for visual grounding + structured schema generation
across ALL task domains: video games, web agents, OS agents, image QA,
and video understanding.

**All tasks are interactive.**  The multi-hop tool-calling loop IS the
interaction — whether the VLM is reasoning about an image-QA benchmark, playing
2048, or answering a video question.  Every task follows the same
interactive pattern:

  1. Input: image(s) + goal/question + optional domain context
  2. Process: VLM sees image → calls tools (multi-hop) → gathers
     evidence → produces structured schema
  3. Output: structured <state> with entities, relations, evidence,
     and a terminal section (actions for env tasks, answer for QA)

The only thing that varies per domain is:
  - Which tools are available (auto-composed from domain)
  - What context is injected (game rules, AXTree, question, etc.)
  - The terminal section: ``<actions>`` vs ``<answer>`` (or both)

Usage::

    from vlm_wrapper.ground import ground, GroundingRequest

    # Image QA (VisualToolBench, TIR-Bench, …) — interactive multi-hop reasoning
    result = ground(GroundingRequest(
        images=pil_image,
        goal="How many red spheres are left of the blue cube?",
        domain="image_qa",
    ))

    # Video game — interactive multi-hop grounding
    result = ground(GroundingRequest(
        images=frame,
        goal="Reach 2048",
        domain="gymv",
        context={"obs_text": obs.text, "description": env.description},
    ))

    # Browser — interactive multi-hop grounding
    result = ground(GroundingRequest(
        images=screenshot,
        goal="Find cheapest laptop",
        domain="browser",
        context={"obs": browsergym_obs},
    ))

    # Video understanding — interactive multi-hop temporal reasoning
    result = ground(GroundingRequest(
        images=frames,          # list of PIL Images
        goal="When does the character first appear?",
        domain="video_qa",
        context={"fps": 24.0},
    ))

    # All results share the same structure:
    result.schema        # <state>...</state>
    result.answer        # "3" (QA) or None (env tasks)
    result.evidence      # list of hop traces (always populated)
    result.tool_trace    # for SFT data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image

from .schema import (
    ValidationResult,
    build_adaptive_system_prompt,
    parse_answer_from_schema,
    parse_evidence_from_schema,
    parse_schema_output,
    semantic_validate,
    validate_schema,
)
from .tool_loop import run_tool_loop
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

DomainType = Literal[
    "gymv", "browser", "desktop", "image_qa", "video_qa", "video", "auto",
]

# All tasks are interactive.  The "output_mode" controls which terminal
# section(s) the schema includes: actions (env), answer (QA), or both.
OutputMode = Literal["actions", "answer", "both"]

_DOMAIN_TO_OUTPUT_MODE: dict[str, OutputMode] = {
    "gymv": "actions",
    "browser": "actions",
    "desktop": "actions",
    "image_qa": "answer",
    "video_qa": "answer",
    "video": "answer",
}

# Shared core sections — every task gets these.  The multi-hop evidence
# chain is always present because every task is interactive.
_CORE_SECTIONS: list[str] = [
    "entities", "attributes", "relations",
    "state_flags", "targets", "uncertainty", "evidence",
]

# Legacy alias kept for backward compat with schema.py defaults
TASK_TYPE_SECTIONS: dict[str, list[str]] = {
    "interactive": _CORE_SECTIONS + ["actions"],
    "qa": _CORE_SECTIONS + ["answer"],
    "temporal": _CORE_SECTIONS + ["answer"],
}


def _sections_for_output_mode(mode: OutputMode) -> list[str]:
    """Return the full section list for a given output mode."""
    if mode == "actions":
        return _CORE_SECTIONS + ["actions"]
    elif mode == "answer":
        return _CORE_SECTIONS + ["answer"]
    else:
        return _CORE_SECTIONS + ["actions", "answer"]


@dataclass
class GroundingRequest:
    """Universal input for the grounding pipeline.

    All tasks are interactive multi-hop reasoning.  The ``output_mode``
    controls the terminal section of the schema (actions vs answer).

    Parameters
    ----------
    images : Image or list[Image]
        Single frame (games, browser, desktop, image QA) or frame
        sequence (video QA, video understanding).
    goal : str
        Task goal or question to answer.
    domain : str
        ``"gymv"`` | ``"browser"`` | ``"desktop"`` | ``"image_qa"``
        | ``"video_qa"`` | ``"video"`` | ``"auto"``.
    output_mode : str
        ``"actions"`` (env tasks) | ``"answer"`` (QA tasks) |
        ``"both"`` | ``"auto"`` (inferred from domain).
    task_id : str
        Environment or benchmark identifier.
    step : int
        Current step number.
    context : dict
        Domain-specific context. Known keys:

        - ``obs_text`` (gymv): raw ``obs.text``
        - ``description`` (gymv): ``env.description`` (game rules)
        - ``obs`` (browser): full BrowserGym observation dict
        - ``axtree_text`` (browser): truncated AXTree string
        - ``a11y_tree_xml`` (desktop): OS accessibility tree XML
        - ``instruction`` (desktop): task instruction
        - ``fps`` (video): frames per second
        - ``current_index`` (video): current frame index
        - ``extra_tools`` (any): additional ``ToolRegistry`` to merge
    max_entities : int
        Entity cap for schema output.
    max_rounds : int
        Maximum tool-call rounds (each round = 1 VLM inference).
    model : str or None
        VLM model override. Defaults to env ``VLM_LABEL_MODEL``.
    api_key : str or None
        OpenAI API key override.
    base_url : str or None
        OpenAI base URL override.
    temperature : float or None
        Generation temperature override.
    """
    images: Image.Image | np.ndarray | list[Image.Image | np.ndarray] = field(
        default=None  # type: ignore[assignment]
    )
    goal: str = ""
    domain: DomainType = "auto"
    output_mode: OutputMode | Literal["auto"] = "auto"
    task_id: str = ""
    step: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    max_entities: int = 25
    max_rounds: int = 5
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    # PLAN-VISUAL-GROUNDING §4 — Option A vs B.  Domains that should
    # re-render / zoom into regions between hops (image QA, video QA)
    # flip this to True.  Games / web default to False (Option A).
    allow_reobservation: bool | None = None


@dataclass
class HopTrace:
    """One reasoning hop in the evidence chain."""
    hop_id: int
    tool: str | None = None
    result_ref: str | None = None
    frame: int | None = None
    timestamp: float | None = None
    confidence: str | None = None
    raw: str = ""


@dataclass
class GroundingResult:
    """Universal output from the grounding pipeline.

    Every domain produces the same structure.  ``evidence`` is always
    populated (multi-hop reasoning traces).  ``answer`` is populated for
    QA domains; ``None`` for env-action domains.

    ``validation`` is populated when the result went through the semantic
    validator (either via ``cascaded_ground`` or an explicit call).
    ``head_used`` and ``escalation_trace`` are populated by
    ``cascaded_ground``.
    """
    schema: str | None = None
    answer: str | None = None
    evidence: list[HopTrace] = field(default_factory=list)
    raw: str = ""
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    domain: str = ""
    output_mode: str = ""
    validation: ValidationResult | None = None
    head_used: str = ""
    escalation_trace: list[dict[str, Any]] = field(default_factory=list)


def _infer_domain(req: GroundingRequest) -> str:
    """Best-effort domain detection from the request contents."""
    ctx = req.context
    if ctx.get("obs") and isinstance(ctx["obs"], dict):
        if ctx["obs"].get("axtree_object") or ctx["obs"].get("screenshot") is not None:
            return "browser"
    if ctx.get("a11y_tree_xml"):
        return "desktop"
    if ctx.get("obs_text") or ctx.get("description"):
        return "gymv"
    if isinstance(req.images, list) and len(req.images) > 1:
        return "video_qa"
    return "image_qa"


def _resolve_output_mode(req: GroundingRequest, domain: str) -> OutputMode:
    """Determine output mode from explicit setting or domain default."""
    if req.output_mode != "auto":
        return req.output_mode  # type: ignore[return-value]
    return _DOMAIN_TO_OUTPUT_MODE.get(domain, "answer")


def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img


_NATURAL_IMAGE_DOMAINS = {"image_qa", "video_qa", "video"}


def _build_registry(
    req: GroundingRequest,
    domain: str,
    primary_image: Image.Image,
) -> ToolRegistry:
    """Auto-compose the right tool registry from domain + context."""
    from visual_reasoning_wrapper.tools_visual import build_visual_registry

    use_gdino = domain in _NATURAL_IMAGE_DOMAINS
    visual_reg = build_visual_registry(primary_image, prefer_gdino=use_gdino)
    registries: list[ToolRegistry] = [visual_reg]

    if domain == "gymv":
        from gymv_wrapper.tools import build_gymv_registry
        gymv_reg = build_gymv_registry(
            obs_text=req.context.get("obs_text", ""),
            description=req.context.get("description", ""),
            step=req.step,
        )
        registries.append(gymv_reg)

    elif domain == "browser":
        obs = req.context.get("obs")
        if obs and isinstance(obs, dict):
            from browsergym_wrapper.tools import build_browser_registry
            registries.append(build_browser_registry(obs))

    elif domain == "desktop":
        from osworld_wrapper.tools import build_osworld_registry
        registries.append(build_osworld_registry(
            a11y_tree_xml=req.context.get("a11y_tree_xml", ""),
            instruction=req.context.get("instruction", req.goal),
            terminal_output=req.context.get("terminal_output", ""),
        ))

    elif domain in ("video_qa", "video") and isinstance(req.images, list):
        from visual_reasoning_wrapper.tools_video import build_video_registry
        from visual_reasoning_wrapper.tools_video_visual import (
            build_video_visual_registry, make_openai_describer,
        )
        # Natural video → GroundingDINO for detection, and a VLM-backed
        # `describe_frame` tool so the model can get a coherent caption
        # instead of relying on the UI icon captioner (which hallucinates
        # subtitle-style text on cinematic frames).
        describer = None
        if req.model:
            try:
                import openai
                client_kwargs: dict[str, Any] = {}
                if req.api_key:
                    client_kwargs["api_key"] = req.api_key
                if req.base_url:
                    client_kwargs["base_url"] = req.base_url
                client = openai.OpenAI(**client_kwargs)
                describer = make_openai_describer(
                    client=client, model=req.model,
                )
            except Exception as exc:
                logger.warning(
                    "Could not build vlm_describer for describe_frame: %s",
                    exc,
                )
        vv_reg = build_video_visual_registry(
            frames=req.images,
            fps=req.context.get("fps", 1.0),
            current_index=req.context.get("current_index", 0),
            prefer_gdino=(domain in ("video_qa", "video")),
            vlm_describer=describer,
            sample_timestamps=req.context.get("sample_timestamps") or [],
        )
        registries = [vv_reg]

    extra = req.context.get("extra_tools")
    if isinstance(extra, ToolRegistry):
        registries.append(extra)

    combined = registries[0]
    for reg in registries[1:]:
        combined = combined.merge(reg)
    return combined


def _build_extra_context(req: GroundingRequest, domain: str) -> str:
    """Assemble domain-specific text context for the user message."""
    parts: list[str] = []

    if domain == "gymv":
        desc = req.context.get("description", "")
        obs_text = req.context.get("obs_text", "")
        valid_actions = req.context.get("valid_actions") or []
        # Callers can hide obs_text from the VLM prompt while still wiring
        # it into the tool registry (so gymv tools return ground truth).
        # Use case: force GPT-4o to actually exercise list_entities /
        # query_entity_pos / check_relation / count_merge_candidates
        # instead of paraphrasing the text grid we already handed it.
        show_obs_text = req.context.get("show_obs_text", True)
        if desc:
            parts.append(f"Game rules:\n{desc}")
        if obs_text and show_obs_text:
            parts.append(f"Environment text state (for reference):\n{obs_text}")
        elif obs_text and not show_obs_text:
            parts.append(
                "Ground-truth state is available via the gymv tool "
                "registry — call list_entities / get_grid_state / "
                "query_entity_pos / check_relation before emitting the "
                "schema.  Do NOT guess positions from pixels when a tool "
                "can return them exactly."
            )
        if valid_actions:
            # GPT-4o tends to invent prose action names like "slide_left"
            # when left to its own devices.  Force it to copy from the
            # env's real action vocabulary so the schema is executable.
            joined = ", ".join(str(a) for a in valid_actions)
            parts.append(
                "Valid actions (copy these EXACTLY into <actions>, "
                f"one per line as aN=<action>): {joined}"
            )

    elif domain == "browser":
        obs = req.context.get("obs", {})
        url = obs.get("url", "") if isinstance(obs, dict) else ""
        axtree = req.context.get("axtree_text", "")
        if url:
            parts.append(f"URL: {url}")
        if axtree:
            parts.append(f"AXTree (truncated):\n{axtree[:3000]}")

    elif domain == "desktop":
        instruction = req.context.get("instruction", "")
        if instruction:
            parts.append(f"Task instruction: {instruction}")

    elif domain in ("video_qa", "video") and isinstance(req.images, list):
        n = len(req.images)
        fps = req.context.get("fps", 1.0)
        cur = req.context.get("current_index", 0)
        dur = req.context.get("duration_s") or 0.0
        native_fps = req.context.get("native_fps")
        sample_ts = req.context.get("sample_timestamps") or []
        ts_hint = ""
        if sample_ts and len(sample_ts) == n:
            joined = ", ".join(f"f{i}={t:.1f}s" for i, t in enumerate(sample_ts))
            ts_hint = f"\nSample-frame wallclock timestamps: {joined}"
        question_type = req.context.get("question_type")
        qt_hint = ""
        if question_type in {"SR", "HR", "MM", "CS"}:
            qt_hint = (
                f"\nQuestion type: {question_type} — this is a narrative "
                "question (relationship / motive / causal / social).  "
                "Inspect MULTIPLE frames spanning different moments of "
                "the clip before answering; one frame is not enough."
            )
        parts.append(
            f"Video: {n} uniformly sampled frames out of a "
            f"{dur:.1f}-second clip "
            f"(native {native_fps or '?'} FPS, "
            f"effective sample FPS {fps:.4g}), "
            f"currently at frame {cur}.\n"
            f"Tools available: temporal navigation (sample_frames, "
            f"detect_scene_changes, get_frame), per-frame detection "
            f"(detect_objects_at_frame — natural-image backend), "
            f"cross-frame tracking (track_object, summarize_clip), and "
            f"`describe_frame(frame_index)` which returns a VLM-generated "
            f"caption.  For narrative questions, prefer describe_frame "
            f"on ≥ 2 well-spread frames over detect_objects_at_frame."
            f"{ts_hint}{qt_hint}"
        )

    elif domain == "image_qa":
        parts.append(
            "Vision-model tools available for precise element detection "
            "and spatial reasoning. Use tools to gather evidence before "
            "answering."
        )

    return "\n\n".join(parts)


def ground(req: GroundingRequest) -> GroundingResult:
    """Unified visual grounding entry point.

    All tasks are interactive multi-hop reasoning.  The pipeline:
    auto-detects domain if needed, composes the right tool registry,
    builds a system prompt with the universal schema (core + terminal),
    runs the VLM tool-calling loop, and parses the result.
    """
    if req.images is None:
        return GroundingResult(
            warnings=["no images provided"],
            domain=req.domain,
        )

    # 1. Resolve domain and output mode
    domain = req.domain if req.domain != "auto" else _infer_domain(req)
    output_mode = _resolve_output_mode(req, domain)
    sections = _sections_for_output_mode(output_mode)

    # 2. Get primary image for the VLM
    if isinstance(req.images, list):
        if not req.images:
            return GroundingResult(
                warnings=["empty frame list"],
                domain=domain,
                output_mode=output_mode,
            )
        idx = req.context.get("current_index", 0)
        idx = max(0, min(idx, len(req.images) - 1))
        primary_image = _to_pil(req.images[idx])
    else:
        primary_image = _to_pil(req.images)

    # 3. Auto-compose registry
    registry = _build_registry(req, domain, primary_image)
    derivation_log = getattr(registry, "derivation_log", None)

    # 4. Build extra context
    extra_context = _build_extra_context(req, domain)

    # 5. Run the tool loop — same loop for all tasks.  Option-B
    # re-observation defaults to ON for image/video QA (fine-grained
    # visual detail) and OFF for games/web where Option A (schema-only
    # updates between hops) is cheaper.
    if req.allow_reobservation is None:
        allow_reobservation = domain in _NATURAL_IMAGE_DOMAINS
    else:
        allow_reobservation = req.allow_reobservation

    loop_result = run_tool_loop(
        primary_image,
        domain=domain,
        registry=registry,
        goal=req.goal,
        task_id=req.task_id,
        step=req.step,
        extra_context=extra_context,
        max_entities=req.max_entities,
        max_rounds=req.max_rounds,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        temperature=req.temperature,
        sections=sections,
        task_type="interactive",
        allow_reobservation=allow_reobservation,
        question_type=req.context.get("question_type"),
    )

    # 6. Parse into universal result
    schema = loop_result.get("schema")
    raw = loop_result.get("raw", "")
    tool_trace = loop_result.get("tool_trace", [])

    # Stitch the typed-derivation log (from reasoning tool calls) into
    # the schema if the model forgot to render the <derivations>
    # block.  The log is the source of truth — even when the model
    # narrates the derivation in <evidence>, we want the typed rows in
    # <derivations> for downstream skill mining.
    if schema and derivation_log is not None and len(derivation_log) > 0:
        schema = _ensure_derivations_block(schema, derivation_log)

    answer = None
    evidence: list[HopTrace] = []
    if schema:
        answer = parse_answer_from_schema(schema)
        evidence = _parse_evidence_hops(schema)

    warnings = list(loop_result.get("warnings", []))
    validation: ValidationResult | None = None

    # Even non-cascaded calls should get the semantic + skill-context
    # checks and the tool-trace reconciliation.  Without this the
    # benchmark scripts (image QA, Video-Holmes) never see the warnings the
    # plan demands (PLAN-VISUAL-GROUNDING-MILESTONES §6 / §12).
    if schema:
        try:
            from .schema import reconcile_evidence_with_tool_trace
            primary_size = None
            if isinstance(req.images, list) and req.images:
                primary_size = _to_pil(req.images[0]).size
            elif req.images is not None:
                primary_size = _to_pil(req.images).size
            validation = semantic_validate(
                schema, domain=domain, image_size=primary_size,
            )
            warnings = warnings + validation.warnings + validation.errors
            warnings = warnings + reconcile_evidence_with_tool_trace(
                schema, tool_trace,
            )
        except Exception as exc:
            logger.warning("post-loop validation failed: %s", exc)

    return GroundingResult(
        schema=schema,
        answer=answer,
        evidence=evidence,
        raw=raw,
        warnings=warnings,
        model=loop_result.get("model", ""),
        tool_trace=tool_trace,
        rounds=loop_result.get("rounds", 0),
        messages=loop_result.get("messages", []),
        domain=domain,
        output_mode=output_mode,
        validation=validation,
    )


def _ensure_derivations_block(schema: str, derivation_log: Any) -> str:
    """Stitch the typed-derivation log into the schema block.

    If the model already rendered ``<derivations>…</derivations>`` we
    leave it alone.  Otherwise we insert a freshly rendered block right
    before ``<answer>`` (or before ``</state>`` if there is no
    ``<answer>``).  The schema validator can then reward the typed
    rows the model would have had to invent inline anyway.
    """
    if "<derivations>" in schema:
        return schema
    rendered = derivation_log.render_section()
    if not rendered.strip():
        return schema
    block = f"<derivations>\n{rendered}\n"
    if "<answer>" in schema:
        return schema.replace("<answer>", block + "<answer>", 1)
    return schema.replace("</state>", block + "</state>", 1)


def _parse_evidence_hops(schema: str) -> list[HopTrace]:
    """Extract HopTrace objects from the <evidence> section."""
    raw_evidence = parse_evidence_from_schema(schema)
    if not raw_evidence:
        return []

    hops: list[HopTrace] = []
    current_hop_id = 0
    current: dict[str, Any] = {}

    for line in raw_evidence.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("hop") and "." in line:
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            key_part, value = parts[0].strip(), parts[1].strip()
            dot_parts = key_part.split(".")
            if len(dot_parts) != 2:
                continue

            hop_tag, field_name = dot_parts
            try:
                hop_num = int(hop_tag.replace("hop", ""))
            except ValueError:
                continue

            if hop_num != current_hop_id and current:
                hops.append(HopTrace(
                    hop_id=current_hop_id,
                    raw=str(current),
                    **{k: v for k, v in current.items() if k != "hop_id"},
                ))
                current = {}

            current_hop_id = hop_num

            if field_name == "tool":
                current["tool"] = value
            elif field_name == "result_ref":
                current["result_ref"] = value
            elif field_name == "frame":
                current["frame"] = None if value == "null" else int(value)
            elif field_name == "timestamp":
                current["timestamp"] = None if value == "null" else float(value)
            elif field_name == "confidence":
                current["confidence"] = value

    if current:
        hops.append(HopTrace(
            hop_id=current_hop_id,
            raw=str(current),
            **{k: v for k, v in current.items() if k != "hop_id"},
        ))

    return hops


# ── Cascaded head escalation (PLAN-VISUAL-GROUNDING §12 Layer 2) ──────
#
# The milestone plan (§7) lays out one starting head per domain and a
# chain of fallbacks.  ``cascaded_ground`` walks that chain, running the
# semantic validator after each attempt, and returning the first
# ``GroundingResult`` whose schema passes validation.  If no head
# produces a clean schema, the best attempt is returned with
# ``escalation_recommended=True`` so the caller can route it to the
# offline teacher (Path C).
#
# Escalation chain per domain.  The heuristic adapters (obs-text grid
# parsing for gymv, AXTree/DOM walk for browsergym) are **NOT** on the
# default path — they short-circuit perception and would hide any VLM
# grounding bug.  They remain implemented (see ``_attempt_heuristic``
# and ``gymv_wrapper.heuristic`` / ``browsergym_wrapper.heuristic``) and
# can be requested explicitly via ``chain=["heuristic", "vlm", "tool_loop"]`` or the
# ``--gymv-head heuristic`` / ``--browser-head heuristic`` CLI flags
# for text-only smoke tests, offline regressions, and CI sanity runs.
#
# | domain    | Head 1         | Head 2       | Head 3      | Tool loop   |
# |-----------|:--------------:|:------------:|:-----------:|:-----------:|
# | gymv      | vlm            | —            | —           | on demand   |
# | browser   | vlm            | omniparser   | —           | on demand   |
# | desktop   | omniparser     | vlm          | —           | on demand   |
# | image_qa  | vlm            | —            | —           | always      |
# | video_qa  | tool_loop      | —            | —           | always      |
#
# Opt-in (heuristic-first) chains are exposed as ``_HEURISTIC_CHAINS``
# below for callers that explicitly want them.

_ESCALATION_CHAINS: dict[str, list[str]] = {
    "gymv":     ["vlm", "tool_loop"],
    "browser":  ["vlm", "omniparser", "tool_loop"],
    "desktop":  ["omniparser", "vlm", "tool_loop"],
    "image_qa": ["vlm", "tool_loop"],
    "video_qa": ["tool_loop"],
    "video":    ["tool_loop"],
}

# Legacy / opt-in chains that start with the heuristic adapter.  Not
# used by cascaded_ground() by default — callers who want the old
# "text-first" behaviour must pass one of these explicitly as ``chain``.
_HEURISTIC_CHAINS: dict[str, list[str]] = {
    "gymv":    ["heuristic", "vlm", "tool_loop"],
    "browser": ["heuristic", "vlm", "omniparser", "tool_loop"],
}


def _attempt_heuristic(req: GroundingRequest, domain: str) -> GroundingResult:
    """Head 1 — heuristic adapter for gymv / browser."""
    schema: str | None = None
    warnings: list[str] = []

    try:
        if domain == "gymv":
            from gymv_wrapper.heuristic import text_to_schema
            schema = text_to_schema(
                obs_text=req.context.get("obs_text", ""),
                description=req.context.get("description", ""),
                task_id=req.task_id,
                step=req.step,
                max_entities=req.max_entities,
            )
        elif domain == "browser":
            from browsergym_wrapper.heuristic import obs_to_schema
            obs = req.context.get("obs") or {}
            if not isinstance(obs, dict):
                warnings.append("browser heuristic requires context['obs'] dict")
            else:
                schema = obs_to_schema(
                    obs,
                    step=req.step,
                    task_id=req.task_id,
                    max_entities=req.max_entities,
                )
        else:
            warnings.append(f"no heuristic head for domain={domain}")
    except Exception as exc:  # heuristic is best-effort
        warnings.append(f"heuristic head failed: {exc}")

    return GroundingResult(
        schema=schema,
        warnings=warnings,
        domain=domain,
        head_used="heuristic",
    )


def _attempt_omniparser(req: GroundingRequest, domain: str,
                        primary_image: Image.Image) -> GroundingResult:
    """Head 3 — local OmniParser grounding (browser / desktop).

    ``grounding_*`` helpers return a dict with a ``schema`` string; we
    lift that out so the cascade sees the same interface as other heads.
    """
    warnings: list[str] = []
    schema: str | None = None
    out: dict[str, Any] = {}

    try:
        if domain == "browser":
            from browsergym_wrapper.grounding import grounding_obs_to_schema
            obs = req.context.get("obs") or {}
            if isinstance(obs, dict) and obs.get("screenshot") is not None:
                out = grounding_obs_to_schema(
                    obs,
                    step=req.step,
                    task_id=req.task_id,
                    max_entities=req.max_entities,
                )
            else:
                # Fall back to the raw image path when the browsergym obs
                # is missing the screenshot field.  We emit an explicit
                # warning so the cascade telemetry records the degraded
                # mode — without ``obs`` the OmniParser head loses bid /
                # role grounding and downstream skill selection can't
                # bind ``click(bid)`` actions cleanly.
                why = (
                    "no context['obs'] dict with 'screenshot' for browser;"
                    " running OmniParser on the raw image (no bid grounding)"
                )
                logger.warning("omniparser fallback: %s", why)
                warnings.append(why)
                from browsergym_wrapper.grounding import grounding_image_to_schema
                out = grounding_image_to_schema(
                    primary_image,
                    goal=req.goal,
                    task_id=req.task_id,
                    step=req.step,
                    domain="browser",
                    max_entities=req.max_entities,
                )
        elif domain == "desktop":
            from browsergym_wrapper.grounding import grounding_image_to_schema
            out = grounding_image_to_schema(
                primary_image,
                goal=req.context.get("instruction", req.goal),
                task_id=req.task_id,
                step=req.step,
                domain="desktop",
                max_entities=req.max_entities,
            )
        else:
            warnings.append(f"no omniparser head for domain={domain}")
    except ImportError as exc:
        warnings.append(f"omniparser unavailable: {exc}")
    except Exception as exc:
        warnings.append(f"omniparser head failed: {exc}")

    if out:
        schema = out.get("schema")
        warnings.extend(out.get("warnings", []) or [])

    return GroundingResult(
        schema=schema,
        warnings=warnings,
        domain=domain,
        head_used="omniparser",
    )


def _attempt_vlm(req: GroundingRequest, domain: str,
                 primary_image: Image.Image) -> GroundingResult:
    """Head 2 — direct single-shot VLM schema generation (no tools).

    For ``gymv`` / ``browser`` / ``desktop`` we delegate to the domain's
    ``generate_label`` adapter so Head 2 in the cascade and Head 2 as
    called directly by data-collection scripts share exactly one code
    path (same prompt, same extra-context assembly, same retry and
    validation semantics).  For ``image_qa`` / ``video_qa`` / ``video``
    there are no dedicated single-shot adapters, so we fall back to
    ``ground()`` with ``max_rounds=1`` — functionally single-shot
    because the VLM is given tools but, at temperature ~0.2 and with an
    aggressive system prompt, almost always emits the schema directly.
    """
    if domain in ("gymv", "browser", "desktop"):
        return _attempt_vlm_via_adapter(req, domain, primary_image)

    shortcut = GroundingRequest(
        images=req.images if isinstance(req.images, list) else primary_image,
        goal=req.goal,
        domain=domain,
        output_mode=req.output_mode,
        task_id=req.task_id,
        step=req.step,
        context=req.context,
        max_entities=req.max_entities,
        max_rounds=1,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        temperature=req.temperature,
    )
    result = ground(shortcut)
    result.head_used = "vlm"
    return result


def _attempt_vlm_via_adapter(
    req: GroundingRequest, domain: str, primary_image: Image.Image,
) -> GroundingResult:
    """Single-shot VLM via the per-domain ``generate_label`` adapter."""
    ctx = req.context
    output_mode = (
        req.output_mode
        if req.output_mode != "auto"
        else _DOMAIN_TO_OUTPUT_MODE.get(domain, "answer")
    )

    try:
        if domain == "gymv":
            from gymv_wrapper.adapter import generate_label as gymv_generate_label
            # ``show_obs_text=False`` (used by --gymv-head tool_loop) must
            # also zero out obs_text in the single-shot VLM path — otherwise
            # the VLM would still see the ground-truth grid and we'd be
            # validating paraphrasing rather than visual grounding.
            show_obs_text = ctx.get("show_obs_text", True)
            out = gymv_generate_label(
                primary_image,
                goal=req.goal,
                task_id=req.task_id,
                step=req.step,
                game_rules=ctx.get("description", ""),
                obs_text=ctx.get("obs_text", "") if show_obs_text else "",
                valid_actions=ctx.get("valid_actions"),
                max_entities=req.max_entities,
                model=req.model,
                api_key=req.api_key,
                base_url=req.base_url,
                temperature=req.temperature,
            )
        elif domain == "browser":
            from browsergym_wrapper.adapter import generate_label as browser_generate_label
            obs = ctx.get("obs") or {}
            url = obs.get("url", "") if isinstance(obs, dict) else ""
            axtree_text = ctx.get("axtree_text", "")
            if not axtree_text and isinstance(obs, dict):
                # browsergym helpfully exposes a pre-flattened tree under
                # ``axtree_txt``; fall through to it when present.
                axtree_text = obs.get("axtree_txt", "") or ""
            out = browser_generate_label(
                primary_image,
                goal=req.goal,
                task_id=req.task_id,
                step=req.step,
                url=url,
                axtree_text=axtree_text,
                last_action=ctx.get("last_action", ""),
                last_action_error=ctx.get("last_action_error", ""),
                max_entities=req.max_entities,
                model=req.model,
                api_key=req.api_key,
                base_url=req.base_url,
                temperature=req.temperature,
            )
        elif domain == "desktop":
            from osworld_wrapper.adapter import generate_label as osworld_generate_label
            out = osworld_generate_label(
                primary_image,
                instruction=ctx.get("instruction", req.goal),
                goal=req.goal,
                task_id=req.task_id,
                step=req.step,
                a11y_tree_xml=ctx.get("a11y_tree_xml", ""),
                terminal_output=ctx.get("terminal_output", ""),
                last_action=ctx.get("last_action", ""),
                last_action_error=ctx.get("last_action_error", ""),
                max_entities=req.max_entities,
                model=req.model,
                api_key=req.api_key,
                base_url=req.base_url,
                temperature=req.temperature,
            )
        else:  # pragma: no cover — defensive
            return GroundingResult(
                warnings=[f"no vlm adapter for domain={domain}"],
                domain=domain, head_used="vlm", output_mode=output_mode,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("vlm adapter (%s) failed: %s", domain, exc)
        return GroundingResult(
            warnings=[f"vlm adapter failed ({domain}): {exc}"],
            domain=domain, head_used="vlm", output_mode=output_mode,
        )

    schema = out.get("schema")
    answer = parse_answer_from_schema(schema) if schema else None
    evidence = _parse_evidence_hops(schema) if schema else []

    return GroundingResult(
        schema=schema,
        answer=answer,
        evidence=evidence,
        raw=out.get("raw", ""),
        warnings=list(out.get("warnings") or []),
        model=out.get("model", ""),
        tool_trace=[],
        rounds=1,
        domain=domain,
        output_mode=output_mode,
        head_used="vlm",
    )


def _attempt_tool_loop(req: GroundingRequest, domain: str) -> GroundingResult:
    """Final head — full multi-hop tool-calling loop (existing ``ground``)."""
    result = ground(req)
    result.head_used = "tool_loop"
    return result


def cascaded_ground(
    req: GroundingRequest,
    *,
    image_size: tuple[int, int] | None = None,
    chain: list[str] | None = None,
    stop_on_first_valid: bool = True,
) -> GroundingResult:
    """Domain-aware escalation wrapper around ``ground``.

    Implements PLAN-VISUAL-GROUNDING §12 Layer 2: run the domain-default
    head, validate with ``semantic_validate``, and escalate to the next
    head if validation recommends it.  The returned ``GroundingResult``
    carries ``validation``, ``head_used`` (head that produced the final
    schema), and ``escalation_trace`` (a record of every head attempted).

    Parameters
    ----------
    req : GroundingRequest
        Same payload used by ``ground``.
    image_size : (w, h), optional
        Passed through to the validator's coordinate-bounds check.
        When omitted, falls back to the primary image's actual size.
    chain : list[str], optional
        Override the default chain for the request's domain.  Valid
        head names: ``heuristic`` (opt-in only — not on the default
        path), ``vlm``, ``omniparser``, ``tool_loop``.  The default
        chains (``_ESCALATION_CHAINS``) are VLM-first; pass
        ``_HEURISTIC_CHAINS[domain]`` (or a custom list) to get the
        legacy text/AXTree heuristic as Head 1.
    stop_on_first_valid : bool
        If True (default), return as soon as any head produces a
        validator-clean schema.  If False, run the full chain and return
        the best attempt — useful for collecting escalation telemetry.

    Returns
    -------
    GroundingResult
        Best available result.  ``validation.valid`` indicates whether
        any head produced a clean schema.  If not,
        ``validation.escalation_recommended`` stays True so the caller
        can flag this observation for Path C (offline teacher).
    """
    if req.images is None:
        result = GroundingResult(
            warnings=["no images provided"],
            domain=req.domain,
        )
        result.validation = semantic_validate(None, domain=req.domain or "image_qa")
        return result

    domain = req.domain if req.domain != "auto" else _infer_domain(req)
    chain = chain or _ESCALATION_CHAINS.get(domain, ["vlm", "tool_loop"])

    if isinstance(req.images, list):
        primary = _to_pil(req.images[req.context.get("current_index", 0)])
    else:
        primary = _to_pil(req.images)
    if image_size is None:
        image_size = primary.size

    trace: list[dict[str, Any]] = []
    best: GroundingResult | None = None
    best_score: tuple[int, int] = (-1, -1)  # (not-escalated, entity_count)

    for head in chain:
        if head == "heuristic":
            attempt = _attempt_heuristic(req, domain)
        elif head == "omniparser":
            attempt = _attempt_omniparser(req, domain, primary)
        elif head == "vlm":
            attempt = _attempt_vlm(req, domain, primary)
        elif head == "tool_loop":
            attempt = _attempt_tool_loop(req, domain)
        else:
            logger.warning("Unknown head in escalation chain: %s", head)
            continue

        validation = semantic_validate(
            attempt.schema,
            domain=domain,
            image_size=image_size,
        )
        # Cross-check evidence hops against actual tool calls (catches
        # fabricated grounding when GroundingDINO/Florence-2 fail
        # silently).  Adds to warnings, not errors.
        try:
            from .schema import reconcile_evidence_with_tool_trace
            recon_warnings = reconcile_evidence_with_tool_trace(
                attempt.schema, attempt.tool_trace,
            )
        except Exception:
            recon_warnings = []

        # Fabricated grounding is a hard failure mode (PLAN-VISUAL-GROUNDING
        # §12 Layer 1): the schema claims evidence that never came from a
        # real tool call.  Promote those recon hits to errors and flag
        # escalation so the next head has a chance to actually ground.
        has_fabrication = any(
            ("fabricated" in w or "ignored" in w or "evidence gap" in w)
            for w in recon_warnings
        )
        if has_fabrication:
            validation.errors.extend(
                w for w in recon_warnings
                if "fabricated" in w or "evidence gap" in w
            )
            validation.valid = False
            validation.escalation_recommended = True

        attempt.validation = validation
        attempt.warnings = (
            list(attempt.warnings)
            + list(validation.warnings) + list(validation.errors)
            + recon_warnings
        )
        attempt.escalation_trace = list(trace) + [{
            "head": head,
            "valid": validation.valid,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings) + recon_warnings,
            "entity_count": validation.entity_count,
        }]
        trace = attempt.escalation_trace

        logger.info(
            "cascaded_ground: head=%s domain=%s valid=%s errors=%s",
            head, domain, validation.valid, validation.errors,
        )

        score = (1 if validation.valid else 0, validation.entity_count)
        if best is None or score > best_score:
            best = attempt
            best_score = score

        if validation.valid and stop_on_first_valid:
            return attempt

    assert best is not None  # chain had at least one entry
    return best
