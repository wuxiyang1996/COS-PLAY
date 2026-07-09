"""Multi-turn tool-calling loop for VLM visual grounding.

Replaces the single-shot generate_label() flow with a multi-turn
conversation where the VLM can call tools to gather ground-truth data
before producing the final <state> schema.

Flow:
  1. Send screenshot + system prompt + tool definitions to the VLM.
  2. If the VLM responds with tool_calls, execute them via the registry,
     append tool results as messages, and loop.
  3. When the VLM responds with text (no tool_calls), parse the
     <state>...</state> schema from the text content.
  4. Return the schema + full conversation trace for training data.

The trace is valuable: it becomes SFT data for teaching Qwen3-VL-8B
to emit the right tool calls.

Usage::

    from vlm_wrapper.tool_loop import run_tool_loop
    from gymv_wrapper.tools import build_gymv_registry

    registry = build_gymv_registry(obs_text=obs.text, description=env.description)
    result = run_tool_loop(
        image=obs.image,
        domain="gymv",
        registry=registry,
        goal="Reach 2048",
        task_id="Game2048-v0",
    )
    print(result["schema"])       # <state>...</state>
    print(result["tool_trace"])   # list of (call, result) for SFT
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import openai
from PIL import Image

from .few_shot_library import get_few_shot_examples
from .schema import (
    build_adaptive_system_prompt,
    build_system_prompt,
    build_user_message,
    count_entities,
    parse_schema_output,
    validate_schema,
)
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")
_DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_LABEL_MAX_TOKENS", "1200"))
_DEFAULT_TEMPERATURE = float(os.environ.get("VLM_LABEL_TEMPERATURE", "0.2"))

# Tool results carrying this key are treated as re-observation payloads
# (PLAN-VISUAL-GROUNDING §4 Option B).  The tool_loop strips the key
# from the JSON it sends to the model (keeping the textual part) and
# attaches the base64 PNG as an image in a follow-up user message.
_REOBSERVE_KEY = "_reobserve_image_b64"

TOOL_USE_INSTRUCTION = """\

You have access to tools that return exact data from the environment.
Instead of guessing positions, element properties, or spatial relations
from the screenshot alone, call the appropriate tool to get ground-truth
values.

Strategy:
1. Look at the screenshot to identify entities visually.
2. Call list_entities or search_elements to confirm what's present.
3. Call query_entity_pos / query_element_bbox for exact coordinates.
4. Call check_relation for spatial/semantic relations you need.
5. Once you have all the data, produce the final <state>...</state> schema.

You may call multiple tools before producing the schema.  When you are
ready to output the final schema, respond with ONLY the <state>...</state>
block and no tool calls.

Evidence rules (VERY IMPORTANT — violations break skill contract learning):
- In <evidence>, `hop_k.tool=` MUST be the concrete tool name you called
  (e.g. `detect_objects`, `read_text_region`, `query_entity_pos`).  NEVER
  write `tool=multi_tool_use.parallel`, `tool=functions.parallel`, or any
  other scheduler wrapper — if you issued parallel calls, record each
  inner call as a separate hop.
- Every positive tool result MUST be referenced by at least one <evidence>
  hop.  If a detection tool returned objects and you chose not to cite
  them, re-check whether those objects are relevant to the goal before
  dropping them.
- `type=` in <entities> MUST be exactly one of: element | object | region
  | text.  Domain-specific labels (icon, button, link, shape, sprite,
  frame, scene, caption, …) belong in the `ontology=` slot, not `type=`.

Entity-count targets by domain (emit at least this many <entities> lines
before finalising the schema — keep calling tools until you clear the
bar; the final schema will otherwise be rejected and re-run):
- gymv, game: 3
- browser, desktop: 5
- image_qa, video_qa: 1 (as many as needed to answer)

Video-QA special rules (domain=video_qa):
- You have BOTH a region-level detector (`detect_objects_at_frame`) and
  a caption-level VLM (`describe_frame(frame_index, focus?)`).  For
  narrative questions (social relationship, motive, causal, plot),
  prefer `describe_frame` — pixel bboxes of faces are less useful than
  a one-paragraph scene caption.
- For a narrative/QA question (question_type ∈ {SR, HR, MM, CS}),
  inspect at least 2 DIFFERENT frames before emitting the schema, and
  ideally 3+ frames spanning the start / middle / end of the clip.
  Use the wallclock `sample_timestamps` in the extra-context block (if
  provided) to pick well-spread frame indices.
- NEVER trust subtitle-style captions from `detect_objects_at_frame`
  unless the frame genuinely has overlaid text that you can also see
  in the screenshot.  On cinematic footage the icon captioner
  occasionally hallucinates chat-bubble text like "Yes bro, the flight
  was tite" — those detections must NOT become `type=text` entities in
  <entities>.
- Ground your <entities> to *things you can see in the frame*: people
  (describe them — "man_with_beard"), objects (phones, vehicles,
  weapons, food), places (kitchen, street, office).
"""


def _domain_entity_min(domain: str) -> int:
    """Mirror of `schema._ENTITY_MIN_BY_DOMAIN` — kept separate to avoid
    pulling a circular import in the prompt-assembly path."""
    return {
        "gymv": 3, "game": 3,
        "browser": 5, "desktop": 5,
        "image_qa": 1, "video_qa": 1, "video": 1,
    }.get(domain, 1)


_CONTINUE_INSTRUCTION_TEMPLATE = (
    "Your draft schema has only {found} entities but this domain "
    "({domain}) requires at least {needed}.  Call more grounding tools "
    "(detect_objects / visual_search / read_text_region / "
    "query_entity_pos / list_entities, whichever fits) to surface the "
    "missing entities, then re-emit the full <state>...</state> schema "
    "with the expanded <entities> section.  Do NOT output the schema "
    "until you have ≥{needed} <entities> lines."
)


_VIDEO_NARRATIVE_QTYPES: frozenset[str] = frozenset({"SR", "HR", "MM", "CS"})
_VIDEO_FRAME_TOOLS: frozenset[str] = frozenset({
    "describe_frame", "detect_objects_at_frame", "read_text_in_frame",
    "get_frame",
})
_MIN_DISTINCT_FRAMES_NARRATIVE: int = 2


_VIDEO_CONTINUE_TEMPLATE = (
    "Your draft schema was built from only {frames_seen} distinct "
    "frame(s), but this is a narrative video question "
    "(question_type={qtype}) — you need to inspect at least "
    "{frames_needed} frames spanning different moments of the clip "
    "before you can reliably answer.  Pick frame indices from "
    "different regions of the clip (e.g. near the start, middle, and "
    "end) and call `describe_frame(frame_index=...)` on them, then "
    "re-emit the full <state>...</state> schema grounded in what you "
    "observed across those frames.  Do NOT repeat the same frame "
    "index."
)


def _count_distinct_frames(tool_trace: list[dict[str, Any]]) -> set[int]:
    """Which frame indexes did the model actually inspect via tools?"""
    seen: set[int] = set()
    for tc in tool_trace:
        call = tc.get("call") or {}
        name = call.get("name")
        if name not in _VIDEO_FRAME_TOOLS:
            continue
        args = call.get("arguments") or {}
        idx = args.get("frame_index")
        if idx is None and name == "get_frame":
            idx = args.get("index")
        if isinstance(idx, int):
            seen.add(idx)
        # Also credit frames that the tool itself reported — e.g.
        # detect_objects_at_frame returns the `frame_index` it ran on.
        result = tc.get("result") or {}
        rf = result.get("frame_index") if isinstance(result, dict) else None
        if isinstance(rf, int):
            seen.add(rf)
    return seen


def run_tool_loop(
    image: Image.Image,
    *,
    domain: str,
    registry: ToolRegistry,
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    extra_context: str = "",
    max_entities: int = 25,
    max_rounds: int = 5,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    sections: list[str] | None = None,
    task_type: str = "interactive",
    allow_reobservation: bool = True,
    question_type: str | None = None,
    few_shot_n: int | None = None,
) -> dict[str, Any]:
    """Run the multi-turn tool-calling loop.

    Parameters
    ----------
    image : PIL.Image
        Screenshot (game frame, browser viewport, video frame).
    domain : str
        ``"gymv"``, ``"browser"``, ``"desktop"``, ``"image_qa"``,
        ``"video_qa"``, or ``"video"``.
    registry : ToolRegistry
        Domain-specific tool registry with handlers bound to current obs.
    goal : str
        Task goal description.
    task_id : str
        Environment/task identifier.
    step : int
        Current step number.
    extra_context : str
        Additional text context (game rules, AXTree excerpt, etc.).
    max_entities : int
        Entity cap for the schema.
    max_rounds : int
        Maximum tool-call rounds before forcing text output.
    model : str
        Vision-LLM model name.
    api_key / base_url : str
        OpenAI client overrides.
    temperature / max_tokens : float / int
        Generation parameters.
    sections : list[str] or None
        Which schema sections to include.  If None, uses legacy default.
        Valid: entities, attributes, relations, state_flags, targets,
        uncertainty, actions, evidence, answer.
    task_type : str
        ``"interactive"`` | ``"qa"`` | ``"temporal"``.  Controls default
        sections and prompt style.
    allow_reobservation : bool
        PLAN-VISUAL-GROUNDING §4 Option B.  When True (default), tool
        results carrying ``_reobserve_image_b64`` are intercepted: the
        base64 PNG is attached as an additional user-side image in the
        next turn so the VLM re-perceives a zoomed crop.  Set to False
        for latency-sensitive games/web where Option A (schema-only
        updates) is preferred.

    Returns
    -------
    dict with keys:
        ``"schema"``     – parsed ``<state>...</state>`` string (or None)
        ``"raw"``        – final raw text output
        ``"warnings"``   – schema validation warnings
        ``"model"``      – model used
        ``"tool_trace"`` – list of ``{"call": {...}, "result": {...}}`` dicts
        ``"rounds"``     – number of conversation rounds
        ``"messages"``   – full message history (for SFT data collection)
    """
    model = model or _DEFAULT_MODEL
    temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
    max_tokens = max_tokens or _DEFAULT_MAX_TOKENS

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    # 1-shot ICL wiring (T2.13', 2026-05-03): every adaptive-prompt caller
    # should be able to inject curated examples without wiring all the
    # plumbing themselves.  Default N depends on env (``VLM_FEW_SHOT_N``,
    # default 1) so deployments can flip the production path to zero-shot
    # by exporting ``VLM_FEW_SHOT_N=0``.  Looks up
    # ``{domain}.{task_slug}.txt`` first, falls back to ``{domain}.txt``,
    # then ``gymv`` for env_wrappers/gymv games.  See
    # ``vlm_wrapper/few_shot_library.py``.
    if few_shot_n is None:
        try:
            few_shot_n = int(os.environ.get("VLM_FEW_SHOT_N", "1"))
        except ValueError:
            few_shot_n = 1
    fallback_domain = "gymv" if domain in ("env_wrappers", "gymv") else None
    examples = get_few_shot_examples(
        domain, n=few_shot_n, task_id=task_id, fallback_domain=fallback_domain,
    ) if few_shot_n > 0 else []

    if sections is not None:
        system_prompt = build_adaptive_system_prompt(
            domain,
            sections=sections,
            task_type=task_type,
            max_entities=max_entities,
            few_shot_examples=examples or None,
        )
    else:
        system_prompt = build_system_prompt(domain, max_entities=max_entities)
    system_prompt += TOOL_USE_INSTRUCTION

    user_content = build_user_message(
        image,
        domain=domain,
        task_id=task_id,
        goal=goal,
        step=step,
        extra_context=extra_context,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tool_defs = registry.definitions()
    tool_trace: list[dict[str, Any]] = []
    raw = ""
    schema = None
    warnings: list[str] = []

    for round_num in range(1, max_rounds + 1):
        try:
            call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tool_defs and round_num < max_rounds:
                call_kwargs["tools"] = tool_defs
                call_kwargs["tool_choice"] = "auto"

            resp = client.chat.completions.create(**call_kwargs)
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append(msg.model_dump())

                pending_reobs: list[dict[str, Any]] = []

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    tool_result = registry.dispatch(fn_name, fn_args)

                    # Option-B re-observation: strip the image payload
                    # from the tool-result JSON (the model cannot read
                    # raw base64 inside a tool message) and stash it for
                    # a follow-up user message that carries the crop as
                    # a first-class image input.
                    reobs_b64 = None
                    if (
                        allow_reobservation
                        and not tool_result.error
                        and isinstance(tool_result.result, dict)
                        and _REOBSERVE_KEY in tool_result.result
                    ):
                        reobs_b64 = tool_result.result.pop(_REOBSERVE_KEY)

                    tool_msg = tool_result.to_message(tc.id)
                    messages.append(tool_msg)

                    if reobs_b64:
                        region = (
                            tool_result.result.get("region")
                            if isinstance(tool_result.result, dict) else None
                        )
                        pending_reobs.append({
                            "tool_call_id": tc.id,
                            "tool": fn_name,
                            "b64": reobs_b64,
                            "region": region,
                        })

                    tool_trace.append({
                        "call": {"name": fn_name, "arguments": fn_args},
                        "result": (
                            tool_result.result
                            if not tool_result.error
                            else {"error": tool_result.error}
                        ),
                        "reobserved": bool(reobs_b64),
                    })

                    logger.info(
                        "Round %d: tool %s(%s) -> %s%s",
                        round_num, fn_name,
                        json.dumps(fn_args, default=str)[:100],
                        json.dumps(tool_result.result, default=str)[:200]
                        if not tool_result.error else tool_result.error,
                        " [reobs]" if reobs_b64 else "",
                    )

                if pending_reobs:
                    reobs_content: list[dict[str, Any]] = []
                    for entry in pending_reobs:
                        region = entry.get("region") or {}
                        descriptor = (
                            f"Re-observation from tool {entry['tool']} "
                            f"(region={region}).  Use this zoomed view to "
                            f"refine <entities>/<attributes>/<uncertainty>."
                        )
                        reobs_content.append({"type": "text", "text": descriptor})
                        reobs_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{entry['b64']}",
                                "detail": "high",
                            },
                        })
                    messages.append({"role": "user", "content": reobs_content})

                continue

            raw = msg.content or ""
            messages.append({"role": "assistant", "content": raw})
            schema = parse_schema_output(raw)
            if schema:
                required = sections if sections is not None else None
                warnings = validate_schema(schema, required_sections=required)

                # If the domain has a minimum-entity floor and the model
                # emitted a schema below it, push a corrective user
                # message and keep looping — provided we still have
                # round budget.  This catches the common failure mode
                # where GPT-4o stops after one tool call on a page
                # (browser / desktop) with 50+ visible elements.
                ent_min = _domain_entity_min(domain)
                ent_found = count_entities(schema)
                rounds_left = max_rounds - round_num
                if (
                    ent_min > 1
                    and ent_found < ent_min
                    and rounds_left >= 1
                ):
                    logger.info(
                        "Round %d: schema has %d entities < min %d for "
                        "domain=%s; nudging model to continue",
                        round_num, ent_found, ent_min, domain,
                    )
                    messages.append({
                        "role": "user",
                        "content": _CONTINUE_INSTRUCTION_TEMPLATE.format(
                            found=ent_found,
                            domain=domain,
                            needed=ent_min,
                        ),
                    })
                    schema = None
                    warnings = []
                    continue

                # Video-QA narrative questions: make sure the model
                # actually looked at more than one frame.  Many video
                # benchmarks have a "relationship/motive" flavour where
                # one frame is not enough to ground the answer, but the
                # entity-count check above would accept a one-frame
                # schema because video_qa has a min of 1.
                if (
                    domain in ("video_qa", "video")
                    and question_type in _VIDEO_NARRATIVE_QTYPES
                    and rounds_left >= 1
                ):
                    distinct = _count_distinct_frames(tool_trace)
                    if len(distinct) < _MIN_DISTINCT_FRAMES_NARRATIVE:
                        logger.info(
                            "Round %d: narrative video-QA (type=%s) "
                            "grounded on %d distinct frame(s) < %d; "
                            "nudging model to sample more frames",
                            round_num, question_type, len(distinct),
                            _MIN_DISTINCT_FRAMES_NARRATIVE,
                        )
                        messages.append({
                            "role": "user",
                            "content": _VIDEO_CONTINUE_TEMPLATE.format(
                                frames_seen=len(distinct),
                                qtype=question_type,
                                frames_needed=_MIN_DISTINCT_FRAMES_NARRATIVE,
                            ),
                        })
                        schema = None
                        warnings = []
                        continue
            break

        except Exception as exc:
            logger.warning("Round %d failed: %s", round_num, exc)
            raw = f"Error: {exc}"
            break

    return {
        "schema": schema,
        "raw": raw,
        "warnings": warnings,
        "model": model,
        "tool_trace": tool_trace,
        "rounds": round_num,  # type: ignore[possibly-undefined]
        "messages": messages,
    }


# ── Convenience wrappers matching existing Head 2 signatures ─────────

def gymv_generate_label_with_tools(
    image: Image.Image,
    *,
    obs_text: str = "",
    description: str = "",
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    max_entities: int = 20,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Drop-in replacement for gymv_adapter.generate_label with tool calling.

    Builds the gymv registry from obs_text/description, then runs the
    multi-turn tool loop.
    """
    from gymv_wrapper.tools import build_gymv_registry
    registry = build_gymv_registry(obs_text=obs_text, description=description, step=step)

    extra_parts = []
    if description:
        extra_parts.append(f"Game rules:\n{description}")
    if obs_text:
        extra_parts.append(f"Environment text state (for reference):\n{obs_text}")

    return run_tool_loop(
        image,
        domain="gymv",
        registry=registry,
        goal=goal or _extract_first_line(description),
        task_id=task_id,
        step=step,
        extra_context="\n\n".join(extra_parts),
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def browser_generate_label_with_tools(
    obs: dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    axtree_text: str = "",
    max_entities: int = 25,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Drop-in replacement for browser_adapter.browser_obs_to_schema with tool calling."""
    import numpy as np
    from .tools_browser import build_browser_registry

    screenshot = obs.get("screenshot")
    if screenshot is None:
        return {"schema": None, "raw": "", "warnings": ["no screenshot"], "model": model or _DEFAULT_MODEL,
                "tool_trace": [], "rounds": 0, "messages": []}

    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return {"schema": None, "raw": "", "warnings": ["unknown screenshot type"], "model": model or _DEFAULT_MODEL,
                "tool_trace": [], "rounds": 0, "messages": []}

    registry = build_browser_registry(obs)

    goal = obs.get("goal", "")
    if not goal:
        goal_obj = obs.get("goal_object", ())
        texts = [m.get("text", "") for m in goal_obj if m.get("type") == "text"]
        goal = " ".join(texts)

    extra_parts = []
    url = obs.get("url", "")
    if url:
        extra_parts.append(f"URL: {url}")
    if axtree_text:
        extra_parts.append(f"AXTree (truncated):\n{axtree_text[:3000]}")

    return run_tool_loop(
        image,
        domain="browser",
        registry=registry,
        goal=goal,
        task_id=task_id,
        step=step,
        extra_context="\n".join(extra_parts),
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def video_generate_label_with_tools(
    frames: list,
    *,
    current_index: int = 0,
    fps: float = 1.0,
    goal: str = "",
    task_id: str = "",
    max_entities: int = 20,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a grounded schema from a video frame using tool calling.

    The VLM sees the current frame but can navigate temporally and
    query other frames to build multi-hop understanding.
    """
    import numpy as np
    from visual_reasoning_wrapper.tools_video import build_video_registry

    registry = build_video_registry(frames=frames, fps=fps, current_index=current_index)

    current_frame = frames[current_index]
    if isinstance(current_frame, np.ndarray):
        image = Image.fromarray(current_frame)
    elif isinstance(current_frame, Image.Image):
        image = current_frame
    else:
        return {"schema": None, "raw": "", "warnings": ["invalid frame type"], "model": model or _DEFAULT_MODEL,
                "tool_trace": [], "rounds": 0, "messages": []}

    extra = f"Video: {len(frames)} frames at {fps} FPS, currently at frame {current_index}"

    return run_tool_loop(
        image,
        domain="gymv",
        registry=registry,
        goal=goal,
        task_id=task_id,
        step=current_index,
        extra_context=extra,
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def visual_generate_label_with_tools(
    image: Image.Image,
    *,
    domain: str = "browser",
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    max_entities: int = 25,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a schema using vision-model tools on a single frame.

    The VLM can call detect_objects, describe_region, visual_search,
    spatial_query, etc. to gather ground-truth data from specialised
    vision models before producing the final schema.
    """
    import numpy as np
    from visual_reasoning_wrapper.tools_visual import build_visual_registry

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    registry = build_visual_registry(image)

    return run_tool_loop(
        image,
        domain=domain,
        registry=registry,
        goal=goal,
        task_id=task_id,
        step=step,
        extra_context="Vision-model tools available for precise element detection and spatial reasoning.",
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def video_visual_generate_label_with_tools(
    frames: list,
    *,
    current_index: int = 0,
    fps: float = 1.0,
    goal: str = "",
    task_id: str = "",
    max_entities: int = 25,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a schema with the full video+visual tool suite.

    Combines temporal navigation (get_frame, sample_frames, etc.) with
    vision-model analysis (detect_objects, describe_region, etc.) and
    cross-frame understanding (track_object, summarize_clip, etc.).

    This is the most capable wrapper — the VLM can navigate time,
    detect objects at any frame, track entities, find moments, and
    reason spatially, all via tool calling.
    """
    import numpy as np
    from visual_reasoning_wrapper.tools_video_visual import build_video_visual_registry

    registry = build_video_visual_registry(
        frames=frames, fps=fps, current_index=current_index,
    )

    current_frame = frames[current_index]
    if isinstance(current_frame, np.ndarray):
        image = Image.fromarray(current_frame)
    elif isinstance(current_frame, Image.Image):
        image = current_frame
    else:
        return {"schema": None, "raw": "", "warnings": ["invalid frame type"], "model": model or _DEFAULT_MODEL,
                "tool_trace": [], "rounds": 0, "messages": []}

    extra = (
        f"Video: {len(frames)} frames at {fps} FPS, currently at frame {current_index}.\n"
        f"Full tool suite available: temporal navigation, vision-model detection, "
        f"cross-frame tracking and comparison."
    )

    return run_tool_loop(
        image,
        domain="gymv",
        registry=registry,
        goal=goal,
        task_id=task_id,
        step=current_index,
        extra_context=extra,
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def _extract_first_line(text: str) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0][:120]
