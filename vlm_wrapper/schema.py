"""Shared schema definition, system prompts, and image-encoding helpers.

This is the single source of truth for the structured-state schema
(see plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md §3).  Both the Gym-V and BrowserGym adapters import from here.
The adaptive prompt builder supports all domains (game, browser, desktop,
image QA, video QA) through configurable schema sections.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

# ── Schema version (bumped when fields change) ───────────────────────
SCHEMA_VERSION = "0.2"

# ── Per-section schema snippets (composed by the adaptive builder) ────

_SECTION_HEADER = """\
<state>
domain={domain}
task={task_id}
goal={goal}
step={step}
"""

_SECTION_ENTITIES = """\
<entities>
e1[type={element|object|region|text}, label={short_label}, bid={str or null}, pos={x,y,w,h or null}, ontology={ontology_type}]
e2[...]
(list every visually distinct entity you can identify — ≤{max_entities})
(type MUST be exactly one of: element | object | region | text.
 Do NOT put ontology values into the type slot.)
(ontology must be one of: selectable_entity | interactive_entity | container_entity |
 textual_anchor | navigable_region | tracked_entity | goal_indicator | blocking_entity.
 This is the cross-domain bridge that lets skills transfer between games / browser /
 desktop / image_qa / video — pick the type that best matches the entity's role.)
(bid is the per-environment grounding id: browsergym id for web entities,
 a11y node id for desktop entities, or `null` when no id exists.  The
 downstream action agent needs it to execute `click(bid)` etc.)
"""

_SECTION_ATTRIBUTES = """\
<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={value or null}
(one line per entity that has a notable attribute)
"""

_SECTION_AFFORDANCES = """\
<affordances>
e1.affords=[focus, select, inspect]
e2.affords=[open, read]
(per-entity list of *abstract* operators it supports — drawn from:
 focus | approach | inspect | select | open | close | read | track |
 compare | wait_until | toggle | enter_text | navigate_to.
 These are skill-level verbs, NOT the concrete environment action.
 Skip entities that are purely decorative.)
"""

_SECTION_RELATIONS = """\
<relations>
contains(eA,eB)
adjacent(eA,eB)
blocks(eA,eB)
grouped(eA,eB,eC)
(spatial and semantic relations you observe)
"""

_SECTION_STATE_FLAGS = """\
<state_flags>
progress={0.0–1.0 or null}
phase={early|mid|late|null}
scene_type={main_menu|landing_page|form_entry|modal_dialog|loading|results_view|game_play|game_over|video_segment|image_qa|null}
error={description or null}
dialog_open={true|false}
input_pending={true|false}
"""

_SECTION_TARGETS = """\
<targets>
target={eid of the most task-relevant entity}
blocker={eid or null}
constraint={short constraint or null}
candidate_set=[eid,eid,...]
history_anchor={eid of an entity carried over from the previous step, or null}
"""

_SECTION_UNCERTAINTY = """\
<uncertainty>
{eid}.{field}={high|medium|low}
(only for entities where you are unsure)
"""

_SECTION_ACTIONS = """\
<actions>
a1={action_string}
a2={action_string}
(top 3–5 plausible next actions)
"""

_SECTION_EVIDENCE = """\
<evidence>
hop1.abstract_op={GROUND|CHECK|RETRIEVE|CONCLUDE}
hop1.tool={tool_name}
hop1.result_ref={e1,e3}
hop1.frame={frame_index or null}
hop1.timestamp={seconds or null}
hop1.confidence={high|medium|low}
hop2.abstract_op={GROUND|CHECK|RETRIEVE|CONCLUDE}
hop2.tool={tool_name}
hop2.result_ref={e2}
hop2.confidence={high|medium|low}
(one group per reasoning hop — record every tool you ACTUALLY called and
which entities its result grounded.  Do NOT invent hops for tools you did
not call.  abstract_op is the inner-MDP verb: GROUND (locate/bind), CHECK
(verify a relation/attribute), RETRIEVE (lookup memory or skill), CONCLUDE
(commit an intermediate result).  For video/temporal tasks include
frame/timestamp.)
"""

_SECTION_ANSWER = """\
<answer>
answer={predicted answer}
grounding=[e1,e3]
evidence_chain=[hop1,hop2,d1,d2]
confidence={high|medium|low}
"""

# Reasoning derivations — typed symbolic operations that the model
# performed via the reasoning toolset (count_value, compute_ratio,
# compare_values, verify_claim).  Each row is one ``_DerivationRow``
# rendered by the per-registry log; the orchestrator appends this
# block before <answer> when the log has rows but the model omitted
# the section.  Skill-mining downstream relies on the typed `kind=` to
# reuse the derivation across tasks.
_SECTION_DERIVATIONS = """\
<derivations>
d1.kind={COUNT|RATIO|COMPARE|VERIFY}
d1.label={short label}
d1.inputs={tool args dict}
d1.output={value or claim}
d1.refs=[e1,e2,hop1,…]
(one row per reasoning tool call.  Required when the question
implies counting / proportion / comparison / verification — without
these rows, the answer carries no auditable derivation.)
"""

# PLAN-VISUAL-GROUNDING §3a — `<evidence_refs>` is the canonical
# `evidence_out` channel that satisfies Skill-Bank Gate G0.  Each line
# is one ``GroundingRecord`` produced by a grounding head or tool call
# and addressable by `evidence_id`.  A grounding call that updates only
# `<entities>` / `<attributes>` without emitting a corresponding
# `<evidence_refs>` row does NOT satisfy Gate G0, because downstream
# REASON / COMMIT skills cannot cite it as warrant.
_SECTION_EVIDENCE_REFS = """\
<evidence_refs>
ev1.evidence_id={short_id}
ev1.source={heuristic|vision|omniparser|tool:<tool_id>}
ev1.kind={entity|region|frame|temporal_window|text_span|dom_node|desktop_object}
ev1.anchor={eid|bid|frame=<idx>|bbox=x,y,w,h|text_span=...}
ev1.confidence={high|medium|low}
ev1.verified_by={ev_id of a VERIFY-role record, or null}
(one block per addressable evidence record — emit at least one
 evidence_refs row for every grounding tool call so downstream skills
 can cite it; reuse `ev_id` instead of recomputing the same evidence.)
"""

_SECTION_FOOTER = "</state>"

_SECTION_MAP: dict[str, str] = {
    "entities": _SECTION_ENTITIES,
    "attributes": _SECTION_ATTRIBUTES,
    "affordances": _SECTION_AFFORDANCES,
    "relations": _SECTION_RELATIONS,
    "state_flags": _SECTION_STATE_FLAGS,
    "targets": _SECTION_TARGETS,
    "uncertainty": _SECTION_UNCERTAINTY,
    "actions": _SECTION_ACTIONS,
    "evidence": _SECTION_EVIDENCE,
    "evidence_refs": _SECTION_EVIDENCE_REFS,
    "derivations": _SECTION_DERIVATIONS,
    "answer": _SECTION_ANSWER,
}

# Cross-domain entity ontology (PLAN-VISUAL-SKILLS §5).
ONTOLOGY_TYPES: tuple[str, ...] = (
    "selectable_entity",
    "interactive_entity",
    "container_entity",
    "textual_anchor",
    "navigable_region",
    "tracked_entity",
    "goal_indicator",
    "blocking_entity",
)

# Abstract operators recognised in <affordances> (PLAN-VISUAL-SKILLS §3d).
ABSTRACT_OPERATORS: tuple[str, ...] = (
    "focus", "approach", "inspect", "select", "open", "close",
    "read", "track", "compare", "wait_until", "toggle",
    "enter_text", "navigate_to",
)

# Inner-MDP actions used in <evidence> hop.abstract_op
# (PLAN-SKILL-BANK §1.5 / PLAN-ACTION-AGENT §5).
INNER_MDP_OPS: tuple[str, ...] = (
    "GROUND", "CHECK", "RETRIEVE", "CONCLUDE", "VERIFY",
)

# Typed reasoning operations recorded inside the ``<derivations>`` block
# (PLAN-VISUAL-GROUNDING §3 reasoning-tool extension).  Mirrors
# ``visual_reasoning_wrapper.tools_reasoning.DERIVATION_KINDS`` — the
# import is kept lazy to avoid a top-level dependency cycle.
DERIVATION_KINDS: tuple[str, ...] = (
    "COUNT", "RATIO", "COMPARE", "VERIFY",
)

# Canonical scene-type enum (PLAN-VISUAL-SKILLS §5 scene descriptor).
SCENE_TYPES: tuple[str, ...] = (
    "main_menu", "landing_page", "form_entry", "modal_dialog", "loading",
    "results_view", "game_play", "game_over", "video_segment",
    "image_qa",
)

# Canonical entity type enum (PLAN-VISUAL-GROUNDING §3 entities schema).
ENTITY_TYPES: tuple[str, ...] = (
    "element", "object", "region", "text",
)

# Soft normalisation for colloquial / domain-specific type labels that the
# VLM emits despite the prompt spec.  Keys are lowercased non-canonical
# values; values are the canonical `ENTITY_TYPES` member they map to.
# These are *accepted* during validation (so a schema with
# `type=icon` is not rejected as unreasonable) but downstream tooling
# should still treat the canonical form as ground truth — we surface a
# single warning per schema when any normalisation fires so prompt
# regressions don't silently accumulate.
_TYPE_ALIASES: dict[str, str] = {
    # GUI / web widgets — collapse onto `element`.
    "icon": "element",
    "button": "element",
    "link": "element",
    "input": "element",
    "checkbox": "element",
    "menu": "element",
    "dialog": "element",
    "window": "element",
    "widget": "element",
    # Image / frame atoms — collapse onto `object`.
    "shape": "object",
    "sprite": "object",
    "tile": "object",
    "piece": "object",
    "image": "object",
    "frame": "region",
    "scene": "region",
    "panel": "region",
    "container": "region",
    # Text-ish.
    "label": "text",
    "caption": "text",
    "ocr": "text",
    "heading": "text",
}

# Scheduler / wrapper tool names emitted by the OpenAI function-calling
# runtime (and sometimes echoed by the VLM into <evidence>) that are not
# real tools.  They must not be flagged as fabricated — they indicate a
# prompt compliance issue instead.
_TOOL_WRAPPER_NAMES: frozenset[str] = frozenset({
    "multi_tool_use.parallel",
    "multi_tool_use",
    "parallel",
    "tools.parallel",
    "functions.parallel",
})

_SCHEMA_RULES = """\
Rules:
- pos= MUST be four comma-separated integers (x,y,w,h) or the literal word
  `null`.  Do NOT wrap pos in parentheses, braces, or brackets.
  REQUIRED examples: `pos=120,40,80,30`  `pos=null`
  FORBIDDEN examples: `pos={120,40,80,30}`  `pos=(120,40,80,30)`  `pos=[120,40,80,30]`
- For browser/desktop pos is pixel coordinates (x,y,width,height).
  For gymv pos is grid coordinates (r,c,1,1).
- `type=` MUST be exactly one of: element | object | region | text.
  Put the skill-role name (selectable_entity, interactive_entity,
  container_entity, textual_anchor, navigable_region, tracked_entity,
  goal_indicator, blocking_entity) in `ontology=`, NEVER in `type=`.
- `<affordances>` MUST be populated for every interactive or selectable
  entity — this section is what the skill bank scores against.  Use only
  these canonical verbs: focus / approach / inspect / select / open /
  close / read / track / compare / wait_until / toggle / enter_text /
  navigate_to.
- `candidate_set=[…]` contains entity IDs only (e1, e3, …), never raw
  labels, MCQ letters, or coordinates.
- Keep labels short (≤5 words).
- Entity IDs are sequential: e1, e2, e3 …
- Reuse entity IDs across sections — never repeat the full label.
- Output ≤{max_entities} entities.  Prefer interactive/important ones.
- If you cannot determine a field, write null.
"""

# Hardened rule block — appended ONLY when ``few_shot_examples`` is
# provided.  These rules sharpen the model on the format the worked
# example just demonstrated; they intentionally are NOT shown in the
# zero-shot variant because they shift the prompt distribution away
# from what the schema_gen LoRA was trained against (empirically a
# ~50 pp prefix-match regression for ``lora+0shot`` on in-distribution
# tasks like candy_crush).
_FEW_SHOT_HARDENING_RULES = """\

Verbatim-preservation rules (CRITICAL — these affect downstream lookup keys):
- `task=` MUST be reproduced VERBATIM from the user's `Task:` line,
  INCLUDING any prefix path segments (e.g. `browsergym/`,
  `make_gaming_env/`, `Temporal/`).  Do NOT shorten, rename, or strip
  the prefix.  If no `Task:` line is supplied, copy the example's
  `task=` format and substitute the right slug for the screenshot.
- `goal=` MUST be reproduced VERBATIM from the user's `Goal:` line.  Do
  NOT paraphrase, summarize, abbreviate, or rephrase — copy it exactly.
- `step=` MUST be the integer the user provided; do NOT default to `0`
  when a non-zero step is given.

Completeness rules (CRITICAL for grid-based / gymv games):
- For grid-based scenes (e.g. 8×8 candy_crush, 4×4 2048, multi-row
  tetris) you MUST emit EVERY visually distinct cell as a separate
  entity, in row-major order (row 0 first: (0,0),(0,1),(0,2)…; then
  row 1).  Do NOT stop early.  Do NOT emit only a "sample" of cells.
- An empty cell is still a cell — emit it (e.g. `label=empty` or the
  appropriate domain ontology) unless the schema for the task
  explicitly treats empties as background.
- The `<entities>` cap of {max_entities} does NOT apply to in-game grid
  cells: a 65-cell candy_crush board is required output.

Anti-leakage rules (CRITICAL — worked examples are below):
- The worked example is for STYLE only.  Do NOT copy its entity labels
  (e.g. `tile_R_1`, `candy_P`), coordinates, ontology vocabulary, or
  exact `goal=` string into your output unless they match the current
  screenshot.  Use the example's *naming convention* and *section
  ordering*, but ground all entities in the actual frame.
"""

# ── Legacy full spec (for backward compat with existing adapters) ─────
SCHEMA_SPEC = """\
You are a visual-state parser.  Given a screenshot, output a structured
summary using EXACTLY the tagged format below.  Do NOT output anything
outside the <state>…</state> block.

<state>
domain={domain}
task={task_id}
goal={goal}
step={step}

<entities>
e1[type={element|object|region|text}, label={short_label}, bid={str or null}, pos={x,y,w,h or null}, ontology={ontology_type}]
e2[...]
(list every visually distinct entity you can identify — ≤{max_entities})
(type MUST be exactly one of: element | object | region | text.  Do NOT put
 ontology values into the type slot.)
(ontology must be one of: selectable_entity | interactive_entity | container_entity |
 textual_anchor | navigable_region | tracked_entity | goal_indicator | blocking_entity.)
(bid is the action-agent handle: browsergym id for web, a11y node id for
 desktop, else `null`.)

<attributes>
e1.state={visible|hidden|disabled|focused|checked|...}
e1.value={value or null}
(one line per entity that has a notable attribute)

<affordances>
e1.affords=[focus, select, inspect]
(per-entity list of abstract operators it supports — focus/approach/inspect/
 select/open/close/read/track/compare/wait_until/toggle/enter_text/navigate_to.)

<relations>
contains(eA,eB)
adjacent(eA,eB)
blocks(eA,eB)
grouped(eA,eB,eC)
(spatial and semantic relations you observe)

<state_flags>
progress={0.0–1.0 or null}
phase={early|mid|late|null}
scene_type={main_menu|landing_page|form_entry|modal_dialog|loading|results_view|game_play|game_over|video_segment|image_qa|null}
error={description or null}
dialog_open={true|false}
input_pending={true|false}

<targets>
target={eid of the most task-relevant entity}
blocker={eid or null}
constraint={short constraint or null}
candidate_set=[eid,eid,...]
history_anchor={eid carried over from previous step, or null}

<uncertainty>
{eid}.{field}={high|medium|low}
(only for entities where you are unsure)

<actions>
a1={action_string}
a2={action_string}
(top 3–5 plausible next actions — copy verbatim from the env's valid action list)
</state>

Rules:
- pos= MUST be four comma-separated integers (x,y,w,h) or the literal word
  `null`.  Do NOT wrap pos in parentheses, braces, or brackets.  Examples of
  the REQUIRED form: `pos=120,40,80,30` or `pos=null`.  Examples to AVOID:
  `pos={120,40,80,30}`, `pos=(120,40,80,30)`, `pos=[120,40,80,30]`.
- For browser/desktop pos is pixel coordinates; for gymv pos is grid coords
  (r,c,1,1).
- `<affordances>` MUST be populated for every interactive or selectable
  entity — this section is what the skill bank scores against.  Use only
  canonical verbs: focus/approach/inspect/select/open/close/read/track/
  compare/wait_until/toggle/enter_text/navigate_to.
- Keep labels short (≤5 words).
- Entity IDs are sequential: e1, e2, e3 …
- Reuse entity IDs across sections — never repeat the full label.
- Output ≤{max_entities} entities.  Prefer interactive/important ones.
- If you cannot determine a field, write null.
"""

# ── Domain-specific context injected after the schema spec ────────────

GYMV_CONTEXT = """\
Domain: gymv (video game).
The screenshot is a rendered game frame.  Entities are game objects
(tiles, pieces, player, walls, targets, etc.).  Positions are grid
coordinates (row, col, 1, 1).  Actions are the valid game moves.

Guidance for populating the schema from the image alone:
- `target=` in <targets> MUST be a concrete eid.  Pick the entity the
  next action should manipulate (e.g. the highest-value tile to merge
  in 2048, the player in Sokoban, the piece being moved).  Never leave
  `target=null` — if unsure, pick the most salient entity.
- `candidate_set=[…]` lists the entity IDs relevant to the next move,
  ordered by priority.  Always include `target` as the first element.
- `scene_type=game_play` for mid-game frames; use `game_over` only when
  the board shows a terminal state.
- `<affordances>` should include at least `select`/`track` for tiles,
  `approach` for goal tiles, so the skill bank can match this frame.
- `<actions>` MUST copy the env's valid actions verbatim — never
  paraphrase them (e.g. use `[Left]`, not `slide_left`).
"""

BROWSER_CONTEXT = """\
Domain: browser (web page).
The screenshot is a browser viewport.  Entities are UI elements
(buttons, links, inputs, text blocks, images, etc.).  Positions are
pixel coordinates (x, y, width, height).  Actions are browser
commands: click(bid), fill(bid, "text"), scroll(direction), etc.
Every interactive entity MUST include a `bid=` slot inside its entity
line — read the id from the AXTree context when provided, or the
browsergym overlay on the screenshot.  Use `bid=null` only when there
is genuinely no AXTree element behind the visual element (e.g. an
image-only banner).  The action agent cannot execute `click()` without
a concrete `bid`.
"""

DESKTOP_CONTEXT = """\
Domain: desktop (OS-level application).
The screenshot is a desktop or application window.  Entities are UI
elements (buttons, menus, text fields, icons, windows, dialogs).
Positions are pixel coordinates (x, y, width, height) — no parens, no
brackets, just four comma-separated integers.  Actions are mouse /
keyboard operations: click(x, y), type("text"), etc.  Each entity's
`type=` MUST be one of element | object | region | text — put the
skill-role name (selectable_entity, interactive_entity, …) in the
`ontology=` slot, NEVER in `type=`.  `scene_type=` is a retrieval tag;
for a bare OS desktop use `main_menu`, for a dialog use `modal_dialog`.
"""

IMAGE_QA_CONTEXT = """\
Domain: image_qa (visual reasoning / question answering).
The screenshot is a static image.  This is an interactive multi-hop
reasoning task: call tools to detect objects, verify spatial relations,
count elements, describe regions, and gather grounded evidence.
Record each reasoning hop in <evidence> with the tool you called and
which entities it referenced.  Build your understanding step by step
before producing the final answer in <answer>.
"""

VIDEO_QA_CONTEXT = """\
Domain: video_qa (video understanding / question answering).
You have access to a video via temporal navigation tools.  This is an
interactive multi-hop reasoning task: call tools to navigate frames,
detect objects, track elements across time, find moments, and gather
grounded evidence.  Record each reasoning hop in <evidence> — include
frame index and timestamp for temporal grounding.  Build your
understanding step by step before producing the final answer in
<answer>.
"""

_DOMAIN_CONTEXT: dict[str, str] = {
    "gymv": GYMV_CONTEXT,
    # env_wrappers (tetris / candy_crush / super_mario / twenty_forty_eight)
    # are interactive game environments — they share the gymv schema
    # template (entities + actions + grid-style state).
    "env_wrappers": GYMV_CONTEXT,
    "browser": BROWSER_CONTEXT,
    "desktop": DESKTOP_CONTEXT,
    "image_qa": IMAGE_QA_CONTEXT,
    "video_qa": VIDEO_QA_CONTEXT,
    "video": VIDEO_QA_CONTEXT,
}


def build_system_prompt(
    domain: str,
    max_entities: int = 20,
) -> str:
    """Build the full system prompt for GPT-4o vision calls.

    Legacy interface — returns the full interactive-task schema with
    all sections.  Use ``build_adaptive_system_prompt`` for the new
    unified pipeline.
    """
    spec = SCHEMA_SPEC.replace("{max_entities}", str(max_entities))
    ctx = GYMV_CONTEXT if domain == "gymv" else BROWSER_CONTEXT
    return f"schema_version={SCHEMA_VERSION}\n\n{spec}\n{ctx}"


def build_adaptive_system_prompt(
    domain: str,
    *,
    sections: list[str] | None = None,
    task_type: str = "interactive",
    max_entities: int = 20,
    few_shot_examples: list[str] | None = None,
) -> str:
    """Build a system prompt that includes only the requested schema sections.

    All tasks are interactive multi-hop reasoning.  The *sections*
    parameter controls which parts of the schema to include.

    Parameters
    ----------
    domain : str
        One of ``"gymv"``, ``"browser"``, ``"desktop"``, ``"image_qa"``,
        ``"video_qa"``, ``"video"``.
    sections : list[str] or None
        Which schema sections to include.  If None, defaults to the
        full interactive set (core + actions).  Valid names: entities,
        attributes, relations, state_flags, targets, uncertainty,
        actions, evidence, answer.
    task_type : str
        Ignored (kept for backward compat).  All tasks are interactive.
    max_entities : int
        Entity cap.
    few_shot_examples : list[str] or None
        Optional list of canonical ``<state>...</state>`` example schemas
        for this domain (typically loaded via
        :func:`vlm_wrapper.few_shot_library.get_few_shot_examples`).  When
        provided, they are inserted between the rules block and the domain
        context block — anchoring the model on the gold's exact naming
        conventions, ``task=`` format, ``goal=`` phrasing, and ontology
        vocabulary.  Adds ~2 000 prompt tokens per example and roughly
        closes the structural-fidelity gap between base-Qwen3.5-35B and
        the schema_gen-LoRA-tuned variant *without* requiring per-domain
        SFT.
    """
    if sections is None:
        sections = [
            "entities", "attributes", "affordances", "relations",
            "state_flags", "targets", "uncertainty",
            "evidence", "actions",
        ]

    preamble = (
        "You are a visual-state parser performing interactive multi-hop "
        "reasoning.  Given a screenshot and tools, call tools to gather "
        "grounded evidence, then output a structured summary using "
        "EXACTLY the tagged format below.  Record every reasoning hop "
        "in the <evidence> section.  Do NOT output anything outside the "
        "<state>…</state> block.\n\n"
    )

    header = _SECTION_HEADER.replace("{max_entities}", str(max_entities))
    body_parts = [header]
    for sec_name in sections:
        snippet = _SECTION_MAP.get(sec_name)
        if snippet:
            body_parts.append(snippet.replace("{max_entities}", str(max_entities)))
    body_parts.append(_SECTION_FOOTER)

    schema_template = "\n".join(body_parts)

    rules = _SCHEMA_RULES.replace("{max_entities}", str(max_entities))
    ctx = _DOMAIN_CONTEXT.get(domain, IMAGE_QA_CONTEXT)

    examples_block = ""
    hardening_rules = ""
    if few_shot_examples:
        from vlm_wrapper.few_shot_library import render_examples_block
        examples_block = "\n\n" + render_examples_block(
            few_shot_examples, domain=domain,
        )
        # The hardening rules (verbatim, completeness, anti-leakage) are
        # designed to work in tandem with a worked example.  Including
        # them in zero-shot mode regresses LoRA performance because the
        # adapter was trained on the un-hardened prompt; in n-shot mode
        # they re-anchor the model on the example's exact format.
        hardening_rules = _FEW_SHOT_HARDENING_RULES.replace(
            "{max_entities}", str(max_entities),
        )

    return (
        f"schema_version={SCHEMA_VERSION}\n\n"
        f"{preamble}{schema_template}\n\n{rules}{hardening_rules}\n"
        f"{ctx}{examples_block}"
    )


def build_user_message(
    image: Image.Image | np.ndarray,
    *,
    domain: str,
    task_id: str = "",
    goal: str = "",
    step: int = 0,
    extra_context: str = "",
) -> list[dict[str, Any]]:
    """Build the multimodal user message (image + text context).

    Returns an OpenAI-compatible ``content`` list for the user role.
    """
    b64 = encode_image_b64(image)

    text_parts = [
        f"domain={domain}",
        f"task={task_id}",
        f"goal={goal}",
        f"step={step}",
    ]
    if extra_context:
        text_parts.append(f"\nAdditional context:\n{extra_context}")

    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        {"type": "text", "text": "\n".join(text_parts)},
    ]


# ── Image helpers ─────────────────────────────────────────────────────

def encode_image_b64(
    image: Image.Image | np.ndarray,
    max_side: int = 1024,
    quality: int = 90,
) -> str:
    """Encode a PIL Image or numpy array to a base64 PNG string.

    Down-scales to *max_side* on the longest edge to keep API costs
    reasonable while retaining enough detail for GPT-4o.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Output parsing / validation ───────────────────────────────────────

_STATE_BLOCK_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL)
_ENTITY_RE = re.compile(r"^(e\d+)\[", re.MULTILINE)


def parse_schema_output(raw: str) -> str | None:
    """Extract the <state>…</state> block from raw GPT output.

    Returns the block (including tags) or None if not found.
    """
    m = _STATE_BLOCK_RE.search(raw)
    if m:
        return f"<state>{m.group(1)}</state>"
    return None


def count_entities(schema_text: str) -> int:
    """Count entity lines in a schema string."""
    return len(_ENTITY_RE.findall(schema_text))


def validate_schema(
    schema_text: str,
    *,
    required_sections: list[str] | None = None,
) -> list[str]:
    """Quick structural checks.  Returns a list of warnings (empty = OK).

    Parameters
    ----------
    schema_text : str
        The ``<state>…</state>`` block.
    required_sections : list[str] or None
        Which sections to check for.  If None, uses the legacy default
        (entities, attributes, relations, state_flags, targets).
    """
    warnings: list[str] = []
    if not schema_text:
        warnings.append("empty schema")
        return warnings
    if "<state>" not in schema_text:
        warnings.append("missing <state> tag")
    if "</state>" not in schema_text:
        warnings.append("missing </state> tag")

    if required_sections is None:
        required_sections = ["entities", "attributes", "relations", "state_flags", "targets"]
    for section in required_sections:
        if f"<{section}>" not in schema_text:
            warnings.append(f"missing <{section}> section")

    n = count_entities(schema_text)
    if n == 0:
        warnings.append("no entities found")
    return warnings


# ── Evidence / answer extraction ──────────────────────────────────────

_EVIDENCE_RE = re.compile(r"<evidence>(.*?)(?=<\w|</state>)", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)(?=<\w|</state>)", re.DOTALL)
_ANSWER_FIELD_RE = re.compile(r"^answer=(.+)$", re.MULTILINE)
_GROUNDING_RE = re.compile(r"^grounding=\[([^\]]*)\]", re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^confidence=(.+)$", re.MULTILINE)


def parse_evidence_from_schema(schema_text: str) -> str | None:
    """Extract the raw <evidence> section content from a schema string."""
    m = _EVIDENCE_RE.search(schema_text)
    return m.group(1).strip() if m else None


def parse_answer_from_schema(schema_text: str) -> str | None:
    """Extract the answer value from the <answer> section."""
    block = _ANSWER_BLOCK_RE.search(schema_text)
    if not block:
        return None
    m = _ANSWER_FIELD_RE.search(block.group(1))
    return m.group(1).strip() if m else None


def parse_answer_block(schema_text: str) -> dict[str, str | None]:
    """Extract the full answer block: answer, grounding, confidence."""
    block = _ANSWER_BLOCK_RE.search(schema_text)
    if not block:
        return {"answer": None, "grounding": None, "confidence": None}
    content = block.group(1)

    answer_m = _ANSWER_FIELD_RE.search(content)
    grounding_m = _GROUNDING_RE.search(content)
    confidence_m = _CONFIDENCE_RE.search(content)

    return {
        "answer": answer_m.group(1).strip() if answer_m else None,
        "grounding": grounding_m.group(1).strip() if grounding_m else None,
        "confidence": confidence_m.group(1).strip() if confidence_m else None,
    }


# ── Semantic validator (PLAN-VISUAL-GROUNDING §12 Layer 1) ────────────
#
# The legacy ``validate_schema`` only checks that required *tags* are
# present.  The semantic validator goes further: it inspects the *content*
# of each section and decides whether the schema is strong enough for the
# reasoning layer to consume, or whether grounding should escalate to a
# stronger head / tool repair (Path B) / offline teacher (Path C).

_SECTION_RE = re.compile(
    r"<(?P<name>entities|attributes|affordances|relations|state_flags|targets|"
    r"uncertainty|actions|evidence|evidence_refs|derivations|answer)>(?P<body>.*?)"
    r"(?=<\w+>|</state>)",
    re.DOTALL,
)

_ENTITY_LINE_RE = re.compile(
    r"^(e\d+)\s*\[(.*?)\]\s*$",
    re.MULTILINE,
)

_TARGET_FIELD_RE = re.compile(r"^target\s*=\s*(.+?)\s*$", re.MULTILINE)
_BLOCKER_FIELD_RE = re.compile(r"^blocker\s*=\s*(.+?)\s*$", re.MULTILINE)
_CONSTRAINT_FIELD_RE = re.compile(r"^constraint\s*=\s*(.+?)\s*$", re.MULTILINE)
_CANDIDATE_SET_RE = re.compile(r"^candidate_set\s*=\s*\[([^\]]*)\]", re.MULTILINE)
_HISTORY_ANCHOR_RE = re.compile(r"^history_anchor\s*=\s*(.+?)\s*$", re.MULTILINE)

# Per-entity inline fields inside ``e1[...]`` (ontology key=value pairs).
_ENTITY_FIELD_RE = re.compile(r"(\w+)\s*=\s*([^,\]]+)")

_ONTOLOGY_FIELD_RE = re.compile(r"ontology\s*=\s*([a-zA-Z_]+)")
_AFFORDS_LINE_RE = re.compile(
    r"^(e\d+)\.affords\s*=\s*\[([^\]]*)\]\s*$",
    re.MULTILINE,
)
_HOP_ABSTRACT_OP_RE = re.compile(
    r"^hop(\d+)\.abstract_op\s*=\s*([A-Z_]+)\s*$",
    re.MULTILINE,
)
_HOP_TOOL_RE = re.compile(
    r"^hop(\d+)\.tool\s*=\s*([\w./-]+)\s*$",
    re.MULTILINE,
)
_HOP_RESULT_REF_RE = re.compile(
    r"^hop(\d+)\.result_ref\s*=\s*\{?([^}\n]*)\}?\s*$",
    re.MULTILINE,
)
_SCENE_TYPE_RE = re.compile(r"^scene_type\s*=\s*([\w_]+)\s*$", re.MULTILINE)

# Reasoning derivations — typed symbolic operations from
# count_value / compute_ratio / compare_values / verify_claim.
# The orchestrator renders rows in either of two layouts:
#   one-field-per-line (canonical schema spec)
#   multi-field-per-line separated by 2+ spaces (compact rendering
#   from ``_DerivationLog.render_section``).
# Match both with a non-anchored, non-greedy capture.
_DERIVATION_KIND_RE = re.compile(
    r"\b(d\d+)\.kind\s*=\s*([A-Z_]+)\b",
)
_DERIVATION_ID_RE = re.compile(r"\b(d\d+)\.")
# Schema body for the answer block uses `evidence_chain=[hop1,d1,…]`.
_EVIDENCE_CHAIN_RE = re.compile(
    r"^evidence_chain\s*=\s*\[([^\]]*)\]",
    re.MULTILINE,
)
_DID_RE = re.compile(r"d\d+")

_UNCERTAINTY_LINE_RE = re.compile(
    r"^(e\d+)\.(\w+)\s*=\s*(high|medium|low)\s*$",
    re.MULTILINE,
)

_RELATION_LINE_RE = re.compile(
    r"^(\w+)\(\s*(e\d+(?:\s*,\s*e\d+)*)\s*\)\s*$",
    re.MULTILINE,
)

_POS_FIELD_RE = re.compile(
    r"pos\s*=\s*(?P<val>null|\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)",
)

# Strict pos format: `pos=null` or `pos=x,y,w,h` with optional single
# space after each comma.  No parentheses, no brackets, no extra tokens.
_POS_STRICT_RE = re.compile(
    r"^pos\s*=\s*(null|\d+,\s?\d+,\s?\d+,\s?\d+)\s*$",
)

# Captures the whole `pos=…` field verbatim so we can detect malformed
# variants like `pos=(1, 2, 3, 4)` or `pos=[1,2,3,4]`.
_POS_VERBATIM_RE = re.compile(r"pos\s*=\s*([^,\]]+(?:,\s*\d+)?[^,\]]*)")

_TYPE_FIELD_RE = re.compile(r"(?:^|[,\s\[])type\s*=\s*([a-zA-Z_]+)")

_EID_REF_RE = re.compile(r"\be(\d+)\b")

_PROGRESS_RE = re.compile(r"^progress\s*=\s*(.+?)\s*$", re.MULTILINE)
_PHASE_RE = re.compile(r"^phase\s*=\s*(.+?)\s*$", re.MULTILINE)

# Per-domain minimum entity counts (PLAN-VISUAL-GROUNDING §12 Layer 1).
_ENTITY_MIN_BY_DOMAIN: dict[str, int] = {
    "gymv": 3,
    "game": 3,
    "browser": 5,
    "desktop": 5,
    "image_qa": 1,
    "video_qa": 1,
    "video": 1,
}

_ENV_DOMAINS: set[str] = {"gymv", "game", "browser", "desktop"}
_QA_DOMAINS: set[str] = {"image_qa", "video_qa", "video"}

# Sections that must exist AND have at least one content line (not just
# the tag).  This legacy default is kept for backward-compat callers
# that pass no ``required_sections`` and no ``domain``.
_REQUIRED_SECTIONS_CORE: list[str] = [
    "entities", "attributes", "relations", "state_flags", "targets",
]

# Domain-aware required-section defaults.  Env tasks must ship with a
# ``<targets>`` + ``<actions>`` tail (PLAN-ACTION-AGENT §2); QA tasks
# must ship with ``<evidence>`` + ``<answer>`` (PLAN-VISUAL-GROUNDING §3).
# Callers that explicitly pass ``required_sections`` still override this.
_REQUIRED_SECTIONS_BY_DOMAIN: dict[str, list[str]] = {
    "gymv":     ["entities", "attributes", "state_flags", "targets", "actions"],
    "game":     ["entities", "attributes", "state_flags", "targets", "actions"],
    "browser":  ["entities", "attributes", "state_flags", "targets", "actions"],
    "desktop":  ["entities", "attributes", "state_flags", "targets", "actions"],
    "image_qa": ["entities", "attributes", "state_flags", "targets",
                  "evidence", "answer"],
    "video_qa": ["entities",              "state_flags", "targets",
                  "evidence", "answer"],
    "video":    ["entities",              "state_flags", "targets",
                  "evidence", "answer"],
}


def required_sections_for_domain(domain: str) -> list[str]:
    """Return the default list of ``required_sections`` for a domain."""
    return list(_REQUIRED_SECTIONS_BY_DOMAIN.get(domain, _REQUIRED_SECTIONS_CORE))


@dataclass
class ValidationResult:
    """Outcome of ``semantic_validate`` — tells the caller whether the
    schema is consumable and, if not, whether to escalate the grounding
    head (see PLAN-VISUAL-GROUNDING §12 Layer 2 and Milestones §7).

    Attributes
    ----------
    valid : bool
        True iff there are no hard errors.  A ``valid=True`` result can
        still carry warnings.
    warnings : list[str]
        Soft issues (e.g. no relations when ≥3 entities) — the schema is
        usable but the reasoning layer should be cautious.
    errors : list[str]
        Hard failures (missing tag, unresolved entity reference, missing
        ``target`` for env task).  Triggers escalation.
    missing_slots : list[str]
        Names of required slots that are unpopulated, e.g.
        ``["target", "candidate_set"]``.
    escalation_recommended : bool
        True iff the caller should try the next head in the escalation
        chain (PLAN-VISUAL-GROUNDING-MILESTONES §7).
    entity_count : int
        Number of distinct entity IDs parsed from ``<entities>``.
    high_uncertainty_frac : float
        Fraction of entities flagged with ``uncertainty=high`` (0–1).
    """

    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    escalation_recommended: bool = False
    entity_count: int = 0
    high_uncertainty_frac: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "missing_slots": list(self.missing_slots),
            "escalation_recommended": self.escalation_recommended,
            "entity_count": self.entity_count,
            "high_uncertainty_frac": round(self.high_uncertainty_frac, 3),
        }


def _split_sections(schema_text: str) -> dict[str, str]:
    """Parse ``<section>…</section>`` bodies into a name→body dict.

    Bodies are stripped of leading/trailing whitespace.  Missing sections
    are simply absent from the returned dict.
    """
    out: dict[str, str] = {}
    for m in _SECTION_RE.finditer(schema_text):
        out[m.group("name")] = m.group("body").strip()
    return out


def _extract_entity_ids(entities_body: str) -> list[str]:
    """Return the ordered list of entity IDs declared in ``<entities>``."""
    ids = []
    seen: set[str] = set()
    for m in _ENTITY_LINE_RE.finditer(entities_body):
        eid = m.group(1)
        if eid not in seen:
            ids.append(eid)
            seen.add(eid)
    return ids


def _content_lines(body: str) -> list[str]:
    """Lines that look like real content (not empty, not a comment).

    Comments in the schema templates are written as ``(one line per …)``
    — we strip those so placeholder prompt text doesn't count as content.
    """
    lines: list[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("(") and s.endswith(")"):
            continue
        lines.append(s)
    return lines


def _referenced_eids(body: str) -> set[str]:
    """Return all ``eN`` tokens referenced in a section body."""
    return {f"e{n}" for n in _EID_REF_RE.findall(body)}


def _split_top_level_commas(text: str) -> list[str]:
    """Split ``text`` on commas that are NOT inside parens/brackets/braces.

    Used to tokenize the inline field list inside an ``e1[…]`` entity
    line where some fields (``pos=x,y,w,h``) legitimately contain
    commas.
    """
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _extract_inline_field_value(inline: str, key: str) -> str | None:
    """Return the value of ``key=…`` inside an entity inline list.

    ``pos`` contains internal commas so we cannot use the simple
    ``key\\s*=\\s*[^,]+`` pattern; we instead split on TOP-LEVEL commas
    (ignoring parens/brackets) so ``pos=x,y,w,h`` is recovered intact.
    For ``pos`` specifically we also greedily absorb up to three extra
    top-level tokens that look like bare ``\\d+``, because a correctly
    formatted ``pos=x,y,w,h`` is itself four top-level tokens.
    """
    tokens = _split_top_level_commas(inline)
    for i, tok in enumerate(tokens):
        m = re.match(rf"{re.escape(key)}\s*=\s*(.+)$", tok)
        if not m:
            continue
        value = m.group(1).strip()
        if key == "pos" and re.fullmatch(r"\d+", value):
            follow = [
                tokens[j] for j in range(i + 1, min(i + 4, len(tokens)))
                if re.fullmatch(r"\d+", tokens[j])
            ]
            if len(follow) == 3:
                return ",".join([value] + follow)
        return value
    return None


def semantic_validate(
    schema_text: str | None,
    domain: str = "image_qa",
    *,
    image_size: tuple[int, int] | None = None,
    required_sections: list[str] | None = None,
) -> ValidationResult:
    """Semantic schema validator — implements PLAN-VISUAL-GROUNDING §12 Layer 1.

    Runs seven checks and returns a ``ValidationResult`` summarising
    whether the schema is consumable by the reasoning layer and, if not,
    whether the caller should escalate to the next grounding head
    (PLAN-VISUAL-GROUNDING-MILESTONES §7).

    Parameters
    ----------
    schema_text : str or None
        A full ``<state>…</state>`` block (what ``parse_schema_output``
        returns).  ``None`` or empty strings produce a hard error.
    domain : str
        Task domain — controls the entity-minimum threshold and whether
        ``target`` is required.  One of ``gymv``, ``browser``, ``desktop``,
        ``image_qa``, ``video_qa``, ``video``.
    image_size : (width, height), optional
        If given, ``pos=`` values are checked for being inside the image.
        Otherwise that check is skipped.
    required_sections : list[str], optional
        Override the default list of required sections.  Defaults to
        ``["entities", "attributes", "relations", "state_flags", "targets"]``.

    Checks (see PLAN-VISUAL-GROUNDING-MILESTONES §6)
    ------
    1. Slot population — ``<targets> target=`` is set (not ``null``) for
       env tasks.  Missing target on env task → error, escalate.
    2. Entity minimum — ≥3 (gymv), ≥5 (browser/desktop), ≥1 (image/video).
       Below threshold → error, escalate.
    3. Uncertainty budget — ≤50 % of entities flagged ``uncertainty=high``.
       Over budget → error, escalate.
    4. Section content — every required section has ≥1 content line.
       Empty section → error, escalate.
    5. Relation coverage — ≥1 relation if ≥3 entities present.  Violation
       is a warning, not an error (soft).
    6. Coordinate consistency — ``pos=`` values inside image bounds.
       Out-of-bounds is a warning.
    7. Entity reference integrity — every ``eN`` in relations/targets/
       attributes/uncertainty exists in ``<entities>``.  Unknown eid →
       error, escalate.
    """
    if required_sections is None:
        required_sections = required_sections_for_domain(domain)

    result = ValidationResult()

    if not schema_text:
        result.errors.append("empty schema")
        result.valid = False
        result.escalation_recommended = True
        return result

    if "<state>" not in schema_text or "</state>" not in schema_text:
        result.errors.append("missing <state>…</state> delimiters")

    sections = _split_sections(schema_text)
    missing_sections = [s for s in required_sections if s not in sections]
    for name in missing_sections:
        result.errors.append(f"missing <{name}> section")

    entities_body = sections.get("entities", "")
    entity_ids = _extract_entity_ids(entities_body)
    result.entity_count = len(entity_ids)
    known_eids = set(entity_ids)

    # Check 2 — entity minimum (domain-aware)
    entity_min = _ENTITY_MIN_BY_DOMAIN.get(domain, 1)
    if result.entity_count < entity_min:
        result.errors.append(
            f"entity_count={result.entity_count} below minimum "
            f"{entity_min} for domain={domain}"
        )

    # Check 4 — every required section has ≥1 content line
    for name in required_sections:
        body = sections.get(name)
        if body is None:
            continue  # already reported above
        if not _content_lines(body):
            result.errors.append(f"<{name}> section is empty")

    # Check 1 — slot population (target for env tasks)
    targets_body = sections.get("targets", "")
    target_match = _TARGET_FIELD_RE.search(targets_body)
    target_val = target_match.group(1).strip() if target_match else None
    is_missing_target = target_val is None or target_val.lower() in {
        "null", "none", "", "{eid}",
    }
    if domain in _ENV_DOMAINS and is_missing_target:
        result.errors.append("target= unset in <targets> (required for env task)")
        result.missing_slots.append("target")

    # Also surface missing soft slots so callers can react (warnings only
    # unless the skill that runs next actually requires them).
    blocker_match = _BLOCKER_FIELD_RE.search(targets_body)
    if blocker_match is None:
        result.missing_slots.append("blocker")
    constraint_match = _CONSTRAINT_FIELD_RE.search(targets_body)
    if constraint_match is None:
        result.missing_slots.append("constraint")
    candidate_match = _CANDIDATE_SET_RE.search(targets_body)
    if candidate_match is None or not candidate_match.group(1).strip():
        result.missing_slots.append("candidate_set")

    # Check 3 — uncertainty budget
    uncertainty_body = sections.get("uncertainty", "")
    high_count = 0
    high_fields: list[str] = []
    for m in _UNCERTAINTY_LINE_RE.finditer(uncertainty_body):
        if m.group(3).lower() == "high":
            high_count += 1
            high_fields.append(m.group(2))          # attribute name: pos | state | …
    if result.entity_count > 0:
        frac = high_count / result.entity_count
        result.high_uncertainty_frac = frac
        if frac > 0.5:
            # On QA domains (image_qa / video_qa / video) entities are
            # routinely caption-grounded rather than bbox-grounded, so
            # `pos=high` across the board is expected and should NOT
            # fail validation.  Only escalate to a hard error when
            # non-positional fields are mass-flagged (indicating the
            # model is generally unsure about the scene) or the domain
            # is pixel-grounded (games, browser, desktop).
            only_positional = all(f == "pos" for f in high_fields)
            if domain in _QA_DOMAINS and only_positional:
                result.warnings.append(
                    f"{high_count}/{result.entity_count} entities have "
                    f"pos=high — expected for caption-grounded QA; add "
                    f"pixel bboxes via detect_objects_at_frame if a "
                    f"downstream skill needs spatial references."
                )
            else:
                result.errors.append(
                    f"{high_count}/{result.entity_count} entities flagged "
                    f"uncertainty=high (>50%)"
                )

    # Check 7 — entity reference integrity (hard error)
    cross_ref_sections = ["attributes", "relations", "targets", "uncertainty",
                           "evidence", "answer"]
    for name in cross_ref_sections:
        body = sections.get(name, "")
        if not body:
            continue
        refs = _referenced_eids(body)
        unknown = refs - known_eids
        if unknown:
            result.errors.append(
                f"<{name}> references unknown entity ids: "
                f"{','.join(sorted(unknown))}"
            )

    # Check 5 — relation coverage (soft / warning only)
    relations_body = sections.get("relations", "")
    rel_count = len(_RELATION_LINE_RE.findall(relations_body))
    if result.entity_count >= 3 and rel_count == 0:
        result.warnings.append(
            "no relations declared despite ≥3 entities "
            "(schema may be under-described)"
        )

    # Check 6 — coordinate consistency (soft / warning only)
    if image_size is not None and entities_body:
        img_w, img_h = image_size
        for m in _POS_FIELD_RE.finditer(entities_body):
            val = m.group("val")
            if val == "null":
                continue
            try:
                x, y, w, h = [int(p.strip()) for p in val.split(",")]
            except ValueError:
                continue
            if x < 0 or y < 0 or w <= 0 or h <= 0 \
                    or x + w > img_w or y + h > img_h:
                result.warnings.append(
                    f"pos={val} is outside image bounds "
                    f"({img_w}x{img_h})"
                )

    # Check 6b — strict pos= formatting (PLAN-VISUAL-GROUNDING §3 rule:
    # "pos= uses pixel coordinates (x,y,w,h)").  Reject parens/brackets
    # and anything that isn't ``null`` or four comma-separated ints.
    # We pull the value out of each entity line by parsing the inline
    # key=value list at the top level (splitting on commas that are NOT
    # inside parens/brackets).
    if entities_body:
        bad_pos: list[str] = []
        for line in _content_lines(entities_body):
            line_m = _ENTITY_LINE_RE.match(line)
            if not line_m:
                continue
            inline = line_m.group(2)
            pos_val = _extract_inline_field_value(inline, "pos")
            if pos_val is None:
                continue
            candidate = f"pos={pos_val.strip()}"
            if _POS_STRICT_RE.match(candidate):
                continue
            bad_pos.append(candidate)
        if bad_pos:
            result.warnings.append(
                "pos= fields with non-canonical format (expect `x,y,w,h` "
                "or `null`, no parens/brackets): "
                + "; ".join(bad_pos[:3])
            )

    # Check 8 — entity type= enum (element|object|region|text).
    if entities_body:
        bad_types: list[str] = []
        normalised_types: list[str] = []
        for line in _content_lines(entities_body):
            line_m = _ENTITY_LINE_RE.match(line)
            if not line_m:
                continue
            inline = line_m.group(2)
            t = _TYPE_FIELD_RE.search(inline)
            if not t:
                continue
            val = t.group(1).strip()
            if val in ENTITY_TYPES:
                continue
            # Soft-accept common colloquial / domain labels by normalising
            # them onto a canonical bucket.  Surface ONE warning with the
            # full alias list so prompt drift is still visible but the
            # schema isn't discarded over a label taxonomy disagreement.
            if val.lower() in _TYPE_ALIASES:
                normalised_types.append(
                    f"{line_m.group(1)}={val}→{_TYPE_ALIASES[val.lower()]}"
                )
                continue
            bad_types.append(f"{line_m.group(1)}={val}")
        if bad_types:
            result.errors.append(
                "entities with non-canonical type= (must be one of "
                f"{ENTITY_TYPES}): " + ", ".join(bad_types[:5])
            )
        if normalised_types:
            result.warnings.append(
                "entities with colloquial type= auto-normalised "
                "(prompt compliance regression — consider updating the "
                f"system prompt): {', '.join(normalised_types[:5])}"
            )

    # Check 9 — candidate_set must resolve to declared entity ids
    # (PLAN-VISUAL-GROUNDING §3 targets block).  The Video-Holmes runner
    # used to drop A/B/C/D MCQ letters here, which breaks any skill that
    # consumes the candidate list.
    cs_m = _CANDIDATE_SET_RE.search(targets_body)
    if cs_m and cs_m.group(1).strip():
        raw_items = [s.strip() for s in cs_m.group(1).split(",") if s.strip()]
        bad_cs = [
            s for s in raw_items
            if not re.fullmatch(r"e\d+", s)
        ]
        unknown_cs = [
            s for s in raw_items
            if re.fullmatch(r"e\d+", s) and s not in known_eids
        ]
        if bad_cs:
            result.errors.append(
                "candidate_set contains non-entity tokens (expect eN): "
                + ",".join(bad_cs[:5])
            )
        if unknown_cs:
            result.errors.append(
                "candidate_set references unknown entity ids: "
                + ",".join(unknown_cs[:5])
            )

    # Check 10 — scene_type enum.  This is an enrichment (PLAN-VISUAL-
    # SKILLS §5), not a plan §3a mandate, so out-of-enum values are a
    # warning — they still flow to the skill bank as a coarse string
    # but don't cause hard escalation.
    state_flags_body = sections.get("state_flags", "")
    scene_m = _SCENE_TYPE_RE.search(state_flags_body)
    if scene_m:
        val = scene_m.group(1).strip()
        if val.lower() != "null" and val not in SCENE_TYPES:
            result.warnings.append(
                f"scene_type={val} is not in the canonical enum "
                f"({SCENE_TYPES}); skill-retrieval index will be coarser"
            )

    # Check 11 — env-only fields populated in a QA schema.  progress/
    # phase are meaningful for env tasks (game progress, planning phase)
    # but for image/video QA they should stay null.  Surface as a
    # warning so the caller can clean up the schema before skill mining.
    if domain in _QA_DOMAINS and state_flags_body:
        for rx, fname in ((_PROGRESS_RE, "progress"), (_PHASE_RE, "phase")):
            m = rx.search(state_flags_body)
            if m and m.group(1).strip().lower() not in {"null", "none", ""}:
                result.warnings.append(
                    f"{fname}={m.group(1).strip()} is an env-only field; "
                    f"should be null in a QA schema"
                )

    # Check 12 — browser entities should carry a bid= (browsergym id)
    # when one is available.  We only emit a warning because the VLM may
    # be ungrounded on a raw screenshot without AXTree.
    if domain == "browser" and entities_body:
        have_bid = 0
        total = 0
        for line in _content_lines(entities_body):
            if not _ENTITY_LINE_RE.match(line):
                continue
            total += 1
            if re.search(r"(?:^|[,\s\[])bid\s*=\s*([\w.-]+)", line):
                have_bid += 1
        if total > 0 and have_bid == 0:
            result.warnings.append(
                "no browser entity carries bid= "
                "(browser actions need the AXTree id to be executable)"
            )

    # ── Skill-context completeness checks (PLAN-SKILL-BANK §3 / ─────
    # PLAN-VISUAL-SKILLS §3-§5).  These produce *warnings*, not errors —
    # the schema is still consumable by the existing reasoning layer, but
    # downstream skill discovery / contract learning will be weaker.

    # Ontology coverage — every entity should declare an ontology type.
    if entities_body:
        ontology_count = 0
        bad_ontology: list[str] = []
        for line in _content_lines(entities_body):
            m = _ENTITY_LINE_RE.match(line)
            if not m:
                continue
            inline = m.group(2)
            o = _ONTOLOGY_FIELD_RE.search(inline)
            if o:
                ontology_count += 1
                val = o.group(1).strip()
                if val not in ONTOLOGY_TYPES and val.lower() != "null":
                    bad_ontology.append(f"{m.group(1)}={val}")
        if ontology_count < result.entity_count:
            result.warnings.append(
                f"only {ontology_count}/{result.entity_count} entities "
                f"declare an ontology= type (skill transfer signal missing)"
            )
        if bad_ontology:
            result.warnings.append(
                "entities with non-canonical ontology types: "
                + ", ".join(bad_ontology[:5])
            )

    # Affordance coverage — interactive/selectable entities should declare
    # at least one abstract operator.
    affordances_body = sections.get("affordances", "")
    afforded_eids: set[str] = set()
    bad_affords: list[str] = []
    for m in _AFFORDS_LINE_RE.finditer(affordances_body):
        afforded_eids.add(m.group(1))
        ops = [o.strip() for o in m.group(2).split(",") if o.strip()]
        for op in ops:
            if op not in ABSTRACT_OPERATORS:
                bad_affords.append(f"{m.group(1)}:{op}")
    if domain in _ENV_DOMAINS and result.entity_count > 0 and not afforded_eids:
        result.warnings.append(
            "<affordances> empty — no abstract operators recorded "
            "(skill applicability scoring will be weak)"
        )
    if bad_affords:
        result.warnings.append(
            "non-canonical abstract operators in <affordances>: "
            + ", ".join(bad_affords[:5])
        )
    unknown_afford_eids = afforded_eids - known_eids
    if unknown_afford_eids:
        result.errors.append(
            "<affordances> references unknown entity ids: "
            + ",".join(sorted(unknown_afford_eids))
        )

    # scene_type — soft signal but very useful for skill retrieval.
    state_flags_body = sections.get("state_flags", "")
    if state_flags_body and not _SCENE_TYPE_RE.search(state_flags_body):
        result.warnings.append(
            "scene_type missing from <state_flags> "
            "(skill retrieval index will be coarser)"
        )

    # history_anchor — only flag for env tasks where multi-step tracking
    # actually matters (gymv / browser / desktop).
    if domain in _ENV_DOMAINS and targets_body \
            and not _HISTORY_ANCHOR_RE.search(targets_body):
        result.warnings.append(
            "history_anchor missing from <targets> "
            "(tracking-family skills cannot bind across steps)"
        )

    # Evidence: every hop with a result_ref should also declare an
    # abstract_op.  Otherwise we cannot mine reasoning-skill protocols.
    evidence_body = sections.get("evidence", "")
    if evidence_body:
        op_hops = {h for h, _ in _HOP_ABSTRACT_OP_RE.findall(evidence_body)}
        result_hops = {h for h, _ in _HOP_RESULT_REF_RE.findall(evidence_body)}
        if result_hops and not op_hops:
            result.warnings.append(
                "<evidence> hops have no abstract_op= "
                "(reasoning-skill protocol cannot be extracted)"
            )
        bad_ops = [
            f"hop{h}={op}" for h, op in _HOP_ABSTRACT_OP_RE.findall(evidence_body)
            if op not in INNER_MDP_OPS
        ]
        if bad_ops:
            result.warnings.append(
                "non-canonical abstract_op in <evidence>: "
                + ", ".join(bad_ops[:5])
            )

    # Derivations: typed reasoning steps from count_value / compute_ratio
    # / compare_values / verify_claim.  Each row should declare a
    # ``kind=`` from ``DERIVATION_KINDS`` and a non-empty ``output=``.
    # When the answer's ``evidence_chain=`` cites ``dN`` ids but no
    # ``<derivations>`` block exists, that's a fabrication-grade error.
    derivations_body = sections.get("derivations", "")
    derivation_ids: set[str] = set()
    if derivations_body:
        bad_kinds: list[str] = []
        for m in _DERIVATION_KIND_RE.finditer(derivations_body):
            derivation_ids.add(m.group(1))
            kind = m.group(2).strip()
            if kind and kind not in DERIVATION_KINDS:
                bad_kinds.append(f"{m.group(1)}={kind}")
        if bad_kinds:
            result.warnings.append(
                "non-canonical kind= in <derivations>: "
                + ", ".join(bad_kinds[:5])
            )
        # rows must carry an output= so downstream skills can read them.
        for line in _content_lines(derivations_body):
            m_id = _DERIVATION_ID_RE.match(line)
            if not m_id:
                continue
            if ".output=" in line:
                continue
        # Verify: every derivation id referenced from <answer>
        # ``evidence_chain=`` exists in <derivations>.  We treat this as
        # a soft warning (orchestrator stitches the block automatically
        # if missing) but a hard error when neither tool calls nor
        # rows back the citation.
    answer_body = sections.get("answer", "")
    if answer_body:
        chain_m = _EVIDENCE_CHAIN_RE.search(answer_body)
        if chain_m:
            chain_ids = [
                t.strip() for t in chain_m.group(1).split(",") if t.strip()
            ]
            cited_dids = {t for t in chain_ids if _DID_RE.fullmatch(t)}
            unknown_dids = cited_dids - derivation_ids
            if unknown_dids and not derivations_body:
                result.errors.append(
                    "<answer> evidence_chain cites derivation ids "
                    + ",".join(sorted(unknown_dids))
                    + " but no <derivations> section is present"
                )
            elif unknown_dids:
                result.warnings.append(
                    "<answer> evidence_chain cites unknown derivation ids: "
                    + ",".join(sorted(unknown_dids))
                )

    # Final verdict + escalation recommendation.
    result.valid = not result.errors
    result.escalation_recommended = bool(result.errors)
    return result


# ── Tool-trace reconciliation (catches fabricated evidence) ──────────


def reconcile_evidence_with_tool_trace(
    schema_text: str | None,
    tool_trace: list[dict[str, Any]] | None,
) -> list[str]:
    """Cross-check ``<evidence>`` hops against the actual ``tool_trace``.

    The VLM sometimes records a hop with a confident ``result_ref`` even
    when the tool itself returned no detections (e.g. GroundingDINO
    failed, Florence-2 missing ``timm``).  The schema looks fine on its
    own but is not actually grounded — which poisons downstream skill
    contract learning.

    Returns a list of warning strings.  Empty list ⇒ all hops are
    grounded by real tool outputs.

    Parameters
    ----------
    schema_text : str
        Full ``<state>…</state>`` block.
    tool_trace : list[dict]
        Each entry should have at minimum ``"tool"`` (str) and
        ``"result"`` (the raw tool return value, dict/list/str).
        ``None`` or empty trace skips reconciliation entirely.
    """
    warnings: list[str] = []
    if not schema_text or not tool_trace:
        return warnings

    sections = _split_sections(schema_text)
    evidence_body = sections.get("evidence", "")
    if not evidence_body:
        return warnings

    # Normalise tool names so the `functions.` prefix some OpenAI
    # function-calling logs emit doesn't mask a real match, and casing
    # differences don't create phantom fabrications.
    def _norm_tool(name: str) -> str:
        n = str(name or "").strip()
        for prefix in ("functions.", "tool.", "tools."):
            if n.startswith(prefix):
                n = n[len(prefix):]
        return n.lower()

    # Build lookup: tool_name -> list of trace records.  ``tool_loop.py``
    # writes records as ``{"call": {"name": ..., "arguments": ...},
    # "result": ..., "reobserved": bool}`` while ad-hoc test traces use
    # ``{"tool": ..., "result": ...}``.  Accept both shapes.
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for rec in tool_trace:
        tool_name = ""
        call = rec.get("call")
        if isinstance(call, dict):
            tool_name = str(call.get("name", "")).strip()
        if not tool_name:
            tool_name = str(rec.get("tool", "")).strip()
        if not tool_name:
            tool_name = str(rec.get("name", "")).strip()
        if not tool_name:
            continue
        by_tool.setdefault(_norm_tool(tool_name), []).append(rec)

    # Helper — does a tool record contain ANY positive results?
    def _has_positive_result(rec: dict[str, Any]) -> bool:
        res = rec.get("result")
        if res is None:
            return False
        if isinstance(res, (list, tuple)):
            return len(res) > 0
        if isinstance(res, dict):
            for k in ("elements", "detections", "objects", "matches",
                      "frames", "moments", "boxes", "regions",
                      "changes", "sampled", "added", "removed"):
                v = res.get(k)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return True
            count = res.get("count")
            if isinstance(count, int) and count > 0:
                return True
            # Caption-style tools (describe_frame, summarize_clip,
            # describe_region, classify_scene, detect_activity) return
            # a text field instead of a detections list.  Referencing
            # them with result_ref=eN in <evidence> is legitimate when
            # the model used that caption to introduce entity eN.
            for k in ("description", "summary", "caption", "answer",
                      "text", "label", "activity"):
                v = res.get(k)
                if isinstance(v, str) and v.strip():
                    return True
            return any(_has_positive_result({"result": v})
                       for v in res.values() if isinstance(v, (list, dict)))
        if isinstance(res, str):
            return bool(res.strip())
        return True

    # Scan each hop.
    hop_tools = dict(_HOP_TOOL_RE.findall(evidence_body))   # {hop_idx: tool}
    hop_refs: dict[str, str] = {
        h: refs for h, refs in _HOP_RESULT_REF_RE.findall(evidence_body)
    }

    for hop_idx, tool_name in hop_tools.items():
        norm_name = _norm_tool(tool_name)
        # OpenAI's function-calling runtime exposes a pseudo-tool called
        # `multi_tool_use.parallel` that fans out concurrent calls.  It
        # is not a real tool; flagging it as "fabricated" is misleading
        # and the real fix is to get the VLM to cite the concrete inner
        # tools in <evidence>.  Surface that as a distinct, softer
        # warning so callers (and prompt regression tests) can tell the
        # two failure modes apart.
        if norm_name in _TOOL_WRAPPER_NAMES:
            warnings.append(
                f"hop{hop_idx} cites scheduler wrapper "
                f"tool={tool_name} instead of the concrete inner "
                f"tool — update the prompt to ask for the underlying "
                f"tool name"
            )
            continue
        records = by_tool.get(norm_name, [])
        if not records:
            warnings.append(
                f"hop{hop_idx} claims tool={tool_name} but no such call "
                f"appears in the tool_trace (fabricated)"
            )
            continue
        # If hop declares grounded entities but every recorded call to
        # this tool returned nothing, flag it.
        refs = hop_refs.get(hop_idx, "").strip()
        has_refs = bool(re.search(r"\be\d+\b", refs))
        if has_refs and not any(_has_positive_result(r) for r in records):
            warnings.append(
                f"hop{hop_idx} ({tool_name}) declares result_ref={refs} "
                f"but every tool call returned no detections — likely "
                f"fabricated grounding"
            )

    # Catch the inverse too: tools that were called but ignored.
    declared_tools = {_norm_tool(t) for t in hop_tools.values()}
    for tool_name, records in by_tool.items():
        if tool_name in declared_tools:
            continue
        if any(_has_positive_result(r) for r in records):
            warnings.append(
                f"tool {tool_name} was called {len(records)}x with positive "
                f"results but is not referenced in any hop (evidence gap)"
            )

    return warnings


# ── GroundingRecord — canonical evidence_out (PLAN-VISUAL-GROUNDING §3a) ─

# Allowed enum values for ``GroundingRecord.source`` and ``.kind``.  The
# `tool:<tool_id>` form is also accepted for `source`, with the suffix
# being any registered tool name.
GROUNDING_SOURCES_ATOMIC: tuple[str, ...] = (
    "heuristic", "vision", "omniparser",
)
GROUNDING_KINDS: tuple[str, ...] = (
    "entity", "region", "frame", "temporal_window",
    "text_span", "dom_node", "desktop_object",
)
GROUNDING_CONFIDENCES: tuple[str, ...] = ("high", "medium", "low")


@dataclass
class GroundingRecord:
    """Canonical ``evidence_out`` emitted by any grounding head / tool.

    A ``GroundingRecord`` is what the Skill-Bank Harness counts at
    Gate G0 (PLAN-SKILL-BANK §0.3 evidence-driven invariant): a
    GATHER-role skill must put one row per produced piece of evidence
    into ``<state>.<evidence_refs>`` so a downstream REASON / COMMIT
    skill can cite it as warrant.

    Attributes
    ----------
    evidence_id : str
        Short stable id (``"ev1"``, ``"ev2"`` …).  Sequential within a
        single ``<state>`` block; reused across hops if the same record
        is referenced again.
    source : str
        ``"heuristic"`` | ``"vision"`` | ``"omniparser"`` | ``"tool:<id>"``.
    kind : str
        What the record is anchored to.  One of ``GROUNDING_KINDS``.
    anchor : str
        Free-form anchor descriptor — typically ``eN``, ``bid=...``,
        ``frame=N``, ``bbox=x,y,w,h``, or ``text_span=...``.  The
        downstream skill consults `kind` to know how to interpret it.
    confidence : str
        ``"high"`` | ``"medium"`` | ``"low"`` — mirrors the per-entity
        ``<uncertainty>`` value for this record.
    verified_by : str or None
        Optional back-reference to a VERIFY-role evidence id that
        checked this record (PLAN-SKILL-BANK §1.2 verification chain).
    """

    evidence_id: str
    source: str
    kind: str
    anchor: str
    confidence: str = "medium"
    verified_by: str | None = None

    def to_lines(self) -> list[str]:
        """Render this record as the ``evN.<key>=<value>`` lines used
        inside the ``<evidence_refs>`` block."""
        ev = self.evidence_id
        return [
            f"{ev}.evidence_id={self.evidence_id}",
            f"{ev}.source={self.source}",
            f"{ev}.kind={self.kind}",
            f"{ev}.anchor={self.anchor}",
            f"{ev}.confidence={self.confidence}",
            f"{ev}.verified_by={self.verified_by or 'null'}",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "kind": self.kind,
            "anchor": self.anchor,
            "confidence": self.confidence,
            "verified_by": self.verified_by,
        }

    def is_valid(self) -> tuple[bool, list[str]]:
        """Return ``(ok, errors)`` after enforcing the §3a contract."""
        errors: list[str] = []
        if not re.fullmatch(r"ev\d+", self.evidence_id):
            errors.append(
                f"evidence_id={self.evidence_id!r} must match `ev\\d+`"
            )
        valid_source = (
            self.source in GROUNDING_SOURCES_ATOMIC
            or self.source.startswith("tool:")
        )
        if not valid_source:
            errors.append(
                f"source={self.source!r} must be one of "
                f"{GROUNDING_SOURCES_ATOMIC} or `tool:<id>`"
            )
        if self.kind not in GROUNDING_KINDS:
            errors.append(
                f"kind={self.kind!r} must be one of {GROUNDING_KINDS}"
            )
        if self.confidence not in GROUNDING_CONFIDENCES:
            errors.append(
                f"confidence={self.confidence!r} must be one of "
                f"{GROUNDING_CONFIDENCES}"
            )
        if not self.anchor.strip():
            errors.append("anchor must be non-empty")
        if self.verified_by is not None and not re.fullmatch(
            r"ev\d+", self.verified_by,
        ):
            errors.append(
                f"verified_by={self.verified_by!r} must match `ev\\d+` "
                f"or be null"
            )
        return (not errors, errors)


# Per-record line patterns inside ``<evidence_refs>``.
_EV_REF_FIELD_RE = re.compile(
    r"^(ev\d+)\.(evidence_id|source|kind|anchor|confidence|verified_by)\s*=\s*(.+?)\s*$",
    re.MULTILINE,
)


def parse_evidence_refs(
    schema_text: str,
) -> list[GroundingRecord]:
    """Parse ``<evidence_refs>`` rows out of a schema string.

    Returns an ordered list of ``GroundingRecord`` (one per ``evN``).
    Missing fields fall back to safe defaults so a partially-emitted
    block still yields usable records — call ``record.is_valid()`` to
    reject malformed ones at Gate G0.
    """
    sections = _split_sections(schema_text)
    body = sections.get("evidence_refs")
    if not body:
        return []

    by_id: dict[str, dict[str, str]] = {}
    for m in _EV_REF_FIELD_RE.finditer(body):
        ev_id, key, val = m.group(1), m.group(2), m.group(3).strip()
        by_id.setdefault(ev_id, {})[key] = val

    records: list[GroundingRecord] = []
    for ev_id in sorted(by_id, key=lambda x: int(x[2:] or "0")):
        fields = by_id[ev_id]
        verified = fields.get("verified_by", "null")
        records.append(
            GroundingRecord(
                evidence_id=fields.get("evidence_id", ev_id),
                source=fields.get("source", "heuristic"),
                kind=fields.get("kind", "entity"),
                anchor=fields.get("anchor", ""),
                confidence=fields.get("confidence", "medium"),
                verified_by=None if verified.lower() == "null" else verified,
            )
        )
    return records


def render_evidence_refs(
    records: list[GroundingRecord],
) -> str:
    """Render a list of ``GroundingRecord`` as an ``<evidence_refs>`` block.

    Intended for tools / heuristic adapters that build the schema
    programmatically rather than letting the VLM write it — the result
    is a drop-in replacement for the placeholder section in the system
    prompt template.
    """
    if not records:
        return ""
    lines = ["<evidence_refs>"]
    for rec in records:
        lines.extend(rec.to_lines())
    return "\n".join(lines) + "\n"


def validate_evidence_refs(
    schema_text: str,
    *,
    require_for_tool_calls: list[str] | None = None,
) -> list[str]:
    """Check that ``<evidence_refs>`` satisfies Gate G0.

    Parameters
    ----------
    schema_text : str
        Full ``<state>…</state>`` block.
    require_for_tool_calls : list[str], optional
        If given, every tool name in this list must appear as the suffix
        of a ``source=tool:<id>`` row.  Use this from the Harness when
        you know which grounding tools were actually invoked and want to
        flag schemas that updated entities silently without recording
        the warrant.

    Returns
    -------
    list[str]
        Warnings (empty = OK).  These are warnings, not errors, because
        the upstream ``<evidence>`` (per-hop trace) is still useful even
        when ``<evidence_refs>`` is missing — but Gate G0 callers should
        treat any non-empty result as a contract violation.
    """
    warnings: list[str] = []
    records = parse_evidence_refs(schema_text)

    if not records:
        if "<evidence_refs>" in schema_text:
            warnings.append(
                "<evidence_refs> section present but contains no parseable "
                "ev_id rows"
            )
        if require_for_tool_calls:
            warnings.append(
                "<evidence_refs> is empty but required_for_tool_calls="
                f"{require_for_tool_calls!r} — Gate G0 violation: "
                "downstream skills cannot cite this grounding"
            )
        return warnings

    seen_ids: set[str] = set()
    for rec in records:
        ok, errors = rec.is_valid()
        if not ok:
            warnings.append(
                f"{rec.evidence_id} is malformed: {'; '.join(errors)}"
            )
        if rec.evidence_id in seen_ids:
            warnings.append(
                f"duplicate evidence_id={rec.evidence_id} in <evidence_refs>"
            )
        seen_ids.add(rec.evidence_id)

    if require_for_tool_calls:
        recorded_tools = {
            rec.source.split("tool:", 1)[1].strip()
            for rec in records
            if rec.source.startswith("tool:")
        }
        missing = [
            t for t in require_for_tool_calls
            if t not in recorded_tools
        ]
        if missing:
            warnings.append(
                "tools called but not recorded in <evidence_refs> "
                f"(Gate G0): {missing}"
            )

    # verified_by must point at a known ev_id within the same block
    known = {rec.evidence_id for rec in records}
    for rec in records:
        if rec.verified_by and rec.verified_by not in known:
            warnings.append(
                f"{rec.evidence_id}.verified_by={rec.verified_by} points "
                f"at unknown record (not declared in <evidence_refs>)"
            )

    return warnings
