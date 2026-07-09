"""Structured ``<state>…</state>`` schema parser for the Actor Agent.

The Visual Grounding pipeline (``vlm_wrapper``) emits a tagged text schema
(see ``plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`` §3).  The Actor Agent consumes
that schema as its *state representation* — not as raw text.  This module
is the decision-agent-facing bridge: it parses the tag soup into a
typed :class:`StateSchema` that the per-step loop reads with dot-access
instead of regex gymnastics.

Why keep this separate from ``vlm_wrapper.schema``?

* ``vlm_wrapper`` owns *production* of the schema — prompts, validation,
  escalation, grounding heads.  It intentionally returns plain text so it
  can be serialised, logged, diffed, and fed back to the teacher pipeline.
* ``decision_agents`` owns *consumption* — the actor needs a structured
  view with entity lookup tables, target resolution, and slot-coverage
  checks.  Parsing is a one-line call at the top of every step and the
  result is immutable within that step.

The parser leans on regexes from ``vlm_wrapper.schema`` where possible
(entity-line format, candidate_set shape, pos format) so the two stay in
lock-step as the schema evolves.  See ``PLAN-ACTION-AGENT.md`` §7 for the
integration story and §10 for the slot-coverage contract that depends on
this structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────
# Regexes.  We import the canonical ones from vlm_wrapper.schema when
# available so there is one source of truth, and fall back to locally
# defined copies when running decision_agents in isolation (unit tests,
# offline labelling).
# ──────────────────────────────────────────────────────────────────────

try:  # pragma: no cover — the import path is exercised indirectly
    from vlm_wrapper.schema import (
        _SECTION_RE as _VLM_SECTION_RE,
        _ENTITY_LINE_RE as _VLM_ENTITY_LINE_RE,
        _CANDIDATE_SET_RE as _VLM_CANDIDATE_SET_RE,
        _HISTORY_ANCHOR_RE as _VLM_HISTORY_ANCHOR_RE,
        _TARGET_FIELD_RE as _VLM_TARGET_FIELD_RE,
        _BLOCKER_FIELD_RE as _VLM_BLOCKER_FIELD_RE,
        _CONSTRAINT_FIELD_RE as _VLM_CONSTRAINT_FIELD_RE,
        _SCENE_TYPE_RE as _VLM_SCENE_TYPE_RE,
        _AFFORDS_LINE_RE as _VLM_AFFORDS_LINE_RE,
        _UNCERTAINTY_LINE_RE as _VLM_UNCERTAINTY_LINE_RE,
        _RELATION_LINE_RE as _VLM_RELATION_LINE_RE,
        _split_top_level_commas as _vlm_split_top_level_commas,
        _extract_inline_field_value as _vlm_extract_inline_field_value,
        parse_answer_block as _vlm_parse_answer_block,
    )
    _SECTION_RE = _VLM_SECTION_RE
    _ENTITY_LINE_RE = _VLM_ENTITY_LINE_RE
    _CANDIDATE_SET_RE = _VLM_CANDIDATE_SET_RE
    _HISTORY_ANCHOR_RE = _VLM_HISTORY_ANCHOR_RE
    _TARGET_FIELD_RE = _VLM_TARGET_FIELD_RE
    _BLOCKER_FIELD_RE = _VLM_BLOCKER_FIELD_RE
    _CONSTRAINT_FIELD_RE = _VLM_CONSTRAINT_FIELD_RE
    _SCENE_TYPE_RE = _VLM_SCENE_TYPE_RE
    _AFFORDS_LINE_RE = _VLM_AFFORDS_LINE_RE
    _UNCERTAINTY_LINE_RE = _VLM_UNCERTAINTY_LINE_RE
    _RELATION_LINE_RE = _VLM_RELATION_LINE_RE
    _split_top_level_commas = _vlm_split_top_level_commas
    _extract_inline_field_value = _vlm_extract_inline_field_value
    _parse_answer_block = _vlm_parse_answer_block
except ImportError:  # pragma: no cover — fall back to locally defined copies
    _SECTION_RE = re.compile(
        r"<(?P<name>entities|attributes|affordances|relations|state_flags|targets|"
        r"uncertainty|actions|evidence|answer)>(?P<body>.*?)"
        r"(?=<\w+>|</state>)",
        re.DOTALL,
    )
    _ENTITY_LINE_RE = re.compile(r"^(e\d+)\s*\[(.*?)\]\s*$", re.MULTILINE)
    _CANDIDATE_SET_RE = re.compile(
        r"^candidate_set\s*=\s*\[([^\]]*)\]", re.MULTILINE
    )
    _HISTORY_ANCHOR_RE = re.compile(
        r"^history_anchor\s*=\s*(.+?)\s*$", re.MULTILINE
    )
    _TARGET_FIELD_RE = re.compile(r"^target\s*=\s*(.+?)\s*$", re.MULTILINE)
    _BLOCKER_FIELD_RE = re.compile(r"^blocker\s*=\s*(.+?)\s*$", re.MULTILINE)
    _CONSTRAINT_FIELD_RE = re.compile(
        r"^constraint\s*=\s*(.+?)\s*$", re.MULTILINE
    )
    _SCENE_TYPE_RE = re.compile(
        r"^scene_type\s*=\s*([\w_]+)\s*$", re.MULTILINE
    )
    _AFFORDS_LINE_RE = re.compile(
        r"^(e\d+)\.affords\s*=\s*\[([^\]]*)\]\s*$", re.MULTILINE
    )
    _UNCERTAINTY_LINE_RE = re.compile(
        r"^(e\d+)\.(\w+)\s*=\s*(high|medium|low)\s*$", re.MULTILINE
    )
    _RELATION_LINE_RE = re.compile(
        r"^(\w+)\(\s*(e\d+(?:\s*,\s*e\d+)*)\s*\)\s*$", re.MULTILINE
    )

    def _split_top_level_commas(text: str) -> List[str]:
        parts: List[str] = []
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

    def _extract_inline_field_value(inline: str, key: str) -> Optional[str]:
        for tok in _split_top_level_commas(inline):
            m = re.match(rf"{re.escape(key)}\s*=\s*(.+)$", tok)
            if m:
                return m.group(1).strip()
        return None

    _ANSWER_BLOCK_LOCAL_RE = re.compile(
        r"<answer>(.*?)(?=<\w|</state>)", re.DOTALL
    )
    _ANSWER_FIELD_LOCAL_RE = re.compile(r"^answer=(.+)$", re.MULTILINE)
    _GROUNDING_LOCAL_RE = re.compile(r"^grounding=\[([^\]]*)\]", re.MULTILINE)
    _CONFIDENCE_LOCAL_RE = re.compile(r"^confidence=(.+)$", re.MULTILINE)

    def _parse_answer_block(schema_text: str) -> Dict[str, Optional[str]]:
        block = _ANSWER_BLOCK_LOCAL_RE.search(schema_text)
        if not block:
            return {"answer": None, "grounding": None, "confidence": None}
        content = block.group(1)
        a = _ANSWER_FIELD_LOCAL_RE.search(content)
        g = _GROUNDING_LOCAL_RE.search(content)
        c = _CONFIDENCE_LOCAL_RE.search(content)
        return {
            "answer": a.group(1).strip() if a else None,
            "grounding": g.group(1).strip() if g else None,
            "confidence": c.group(1).strip() if c else None,
        }


# Regexes local to the header and actions/evidence sections — these are
# not yet exposed by ``vlm_wrapper.schema`` but are simple enough to
# duplicate without risk of drift.
_HEADER_FIELD_RE = re.compile(
    r"^(domain|task|goal|step)\s*=\s*(.+?)\s*$", re.MULTILINE
)
_ATTR_LINE_RE = re.compile(
    r"^(e\d+)\.(\w+)\s*=\s*(.+?)\s*$", re.MULTILINE
)
_ACTION_LINE_RE = re.compile(
    r"^a(\d+)\s*=\s*(.+?)\s*$", re.MULTILINE
)
_HOP_FIELD_RE = re.compile(
    r"^hop(\d+)\.(\w+)\s*=\s*(.+?)\s*$", re.MULTILINE
)
_PROGRESS_RE = re.compile(r"^progress\s*=\s*(.+?)\s*$", re.MULTILINE)
_PHASE_RE = re.compile(r"^phase\s*=\s*(.+?)\s*$", re.MULTILINE)
_ERROR_RE = re.compile(r"^error\s*=\s*(.+?)\s*$", re.MULTILINE)
_DIALOG_OPEN_RE = re.compile(r"^dialog_open\s*=\s*(.+?)\s*$", re.MULTILINE)
_INPUT_PENDING_RE = re.compile(r"^input_pending\s*=\s*(.+?)\s*$", re.MULTILINE)

_EID_TOKEN_RE = re.compile(r"e\d+")


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Entity:
    """One parsed entity line from ``<entities>``.

    Fields match the schema slot list in PLAN-VISUAL-GROUNDING §3.  All
    string fields default to the empty string (not ``None``) so the actor
    can always call ``.label``, ``.ontology`` etc. without an extra None
    check — absent info is conveyed via :attr:`present` flags below.
    """

    eid: str
    type: str = ""
    label: str = ""
    bid: Optional[str] = None
    pos: Optional[Tuple[int, int, int, int]] = None
    ontology: str = ""
    # Extra inline ``key=value`` pairs we didn't recognise (kept for
    # forward-compat so the parser doesn't silently drop new fields).
    extra: Dict[str, str] = field(default_factory=dict)

    # Decorated from other sections during parse():
    state: Optional[str] = None          # <attributes> e1.state=...
    value: Optional[str] = None          # <attributes> e1.value=...
    attributes: Dict[str, str] = field(default_factory=dict)  # other e1.foo=...
    affords: List[str] = field(default_factory=list)          # <affordances>
    uncertainty: Dict[str, str] = field(default_factory=dict) # <uncertainty>


@dataclass
class Targets:
    """Parsed ``<targets>`` block."""

    target: Optional[str] = None
    blocker: Optional[str] = None
    constraint: Optional[str] = None
    candidate_set: List[str] = field(default_factory=list)
    history_anchor: Optional[str] = None


@dataclass
class StateFlags:
    """Parsed ``<state_flags>`` block."""

    progress: Optional[float] = None
    phase: Optional[str] = None
    scene_type: Optional[str] = None
    error: Optional[str] = None
    dialog_open: Optional[bool] = None
    input_pending: Optional[bool] = None


@dataclass
class Relation:
    """Parsed ``<relations>`` line."""

    name: str
    args: List[str] = field(default_factory=list)


@dataclass
class Hop:
    """Parsed ``<evidence>`` hop group.

    Keys follow PLAN-VISUAL-GROUNDING §3 evidence block:
    ``abstract_op``, ``tool``, ``result_ref``, ``frame``, ``timestamp``,
    ``confidence``.
    """

    idx: int
    abstract_op: str = ""
    tool: str = ""
    result_ref: List[str] = field(default_factory=list)
    frame: Optional[int] = None
    timestamp: Optional[float] = None
    confidence: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class Answer:
    """Parsed ``<answer>`` block (QA domains)."""

    answer: Optional[str] = None
    grounding: List[str] = field(default_factory=list)
    confidence: Optional[str] = None


@dataclass
class StateSchema:
    """Structured view of one ``<state>…</state>`` schema block.

    This is what the Actor Agent consumes each step.  Section fields
    default to empty so the decision loop can reason about partial
    schemas without guards.

    See PLAN-ACTION-AGENT §7 (schema as inner-MDP state) and §10
    (uncertainty-driven GROUND triggering) for how the actor uses these
    fields.
    """

    domain: str = ""
    task: str = ""
    goal: str = ""
    step: Optional[int] = None

    entities: Dict[str, Entity] = field(default_factory=dict)
    entity_order: List[str] = field(default_factory=list)

    relations: List[Relation] = field(default_factory=list)
    state_flags: StateFlags = field(default_factory=StateFlags)
    targets: Targets = field(default_factory=Targets)
    actions: List[str] = field(default_factory=list)
    evidence: List[Hop] = field(default_factory=list)
    answer: Optional[Answer] = None

    # Raw input so downstream code can round-trip / log it verbatim.
    raw: str = ""

    # ── Convenience lookup helpers ───────────────────────────────────

    def get_entity(self, eid: Optional[str]) -> Optional[Entity]:
        """Return the Entity for *eid*, or None if not present."""
        if not eid:
            return None
        return self.entities.get(eid)

    def label_of(self, eid: Optional[str]) -> str:
        """Return the label for *eid*, or the eid itself if missing."""
        ent = self.get_entity(eid)
        if ent and ent.label:
            return ent.label
        return eid or ""

    def entities_by_ontology(self, ontology: str) -> List[Entity]:
        """All entities whose ``ontology=`` matches (e.g. ``blocking_entity``)."""
        return [e for e in self.entities.values() if e.ontology == ontology]

    def interactive_entities(self) -> List[Entity]:
        """Entities the actor can plausibly act on.

        Heuristic: entities carrying an ontology of ``interactive_entity``
        or ``selectable_entity``, OR entities with a non-empty
        ``affords`` list that contains an actionable verb.  Used by the
        default action resolver when the LLM emits an entity reference
        without saying what to do with it.
        """
        actionable_verbs = {"select", "open", "toggle", "enter_text",
                             "navigate_to", "focus"}
        out: List[Entity] = []
        for e in self.entities.values():
            if e.ontology in {"interactive_entity", "selectable_entity"}:
                out.append(e)
                continue
            if any(v in actionable_verbs for v in e.affords):
                out.append(e)
        return out

    # ── Inner-MDP / slot-coverage helpers (PLAN §10) ─────────────────

    def high_uncertainty_eids(self, field_name: Optional[str] = None) -> List[str]:
        """Entity IDs with ``uncertainty=high`` on (optionally) *field_name*.

        Used by the inner-MDP hop policy: if ``target``'s uncertainty is
        high, insert a GROUND hop; if position uncertainty is high but
        the action is coarse, EXECUTE is still fine.
        """
        if field_name is None:
            return [
                e.eid for e in self.entities.values()
                if "high" in e.uncertainty.values()
            ]
        return [
            e.eid for e in self.entities.values()
            if e.uncertainty.get(field_name) == "high"
        ]

    def slot_coverage(self, required_slots: List[str]) -> Dict[str, bool]:
        """Return ``{slot_name: True if populated}`` for a list of slot names.

        Recognised slots:

        * ``target`` — targets.target is a declared entity
        * ``blocker`` — targets.blocker is a declared entity
        * ``candidate_set`` — targets.candidate_set non-empty
        * ``constraint`` — targets.constraint is a non-null string
        * ``history_anchor`` — targets.history_anchor is a declared entity
        * any entity eid (``e3``) — entity is declared
        * ``e*.field`` — entity exists AND entity.attributes contains field

        Unknown slot names return False.  Used by
        :class:`decision_agents.skill_tracker.SkillTracker` to decide
        whether a skill can activate or whether hop 0 should be a GROUND.
        """
        ent_ids = set(self.entities)
        cov: Dict[str, bool] = {}
        for slot in required_slots:
            cov[slot] = _slot_is_populated(self, slot, ent_ids)
        return cov

    def missing_slots(self, required_slots: List[str]) -> List[str]:
        """Slot names from *required_slots* that are not populated."""
        return [s for s, ok in self.slot_coverage(required_slots).items() if not ok]

    # ── Compact summary for prompt budgets / memory / RAG ────────────

    def compact_summary(self, max_chars: int = 400) -> str:
        """Return a dense ``key=value | key=value`` summary of the schema.

        Mirrors ``decision_agents.agent_helper.compact_structured_state``
        but uses the parsed schema directly so we never have to re-parse
        the same text twice.
        """
        parts: List[Tuple[str, str]] = []

        if self.domain:
            parts.append(("domain", self.domain))
        if self.goal:
            parts.append(("goal", self.goal[:60]))
        if self.step is not None:
            parts.append(("step", str(self.step)))

        sf = self.state_flags
        if sf.scene_type:
            parts.append(("scene", sf.scene_type))
        if sf.phase:
            parts.append(("phase", sf.phase))
        if sf.progress is not None:
            parts.append(("progress", f"{sf.progress:.2f}"))
        if sf.error:
            parts.append(("error", sf.error[:40]))
        if sf.dialog_open:
            parts.append(("dialog", "open"))

        if self.entities:
            parts.append(("n_entities", str(len(self.entities))))

        t = self.targets
        if t.target:
            parts.append(("target", t.target))
        if t.blocker:
            parts.append(("blocker", t.blocker))
        if t.candidate_set:
            parts.append(("cands", ",".join(t.candidate_set[:5])))
        if t.constraint:
            parts.append(("constraint", t.constraint[:40]))

        if self.actions:
            parts.append(("actions", ",".join(self.actions[:4])))

        # assemble
        segments: List[str] = []
        length = 0
        for k, v in parts:
            seg = f"{k}={v}"
            added = len(seg) + (3 if segments else 0)
            if length + added > max_chars:
                break
            segments.append(seg)
            length += added
        return " | ".join(segments)


# ──────────────────────────────────────────────────────────────────────
# Slot-coverage internal helper
# ──────────────────────────────────────────────────────────────────────


def _slot_is_populated(
    schema: "StateSchema",
    slot: str,
    ent_ids: set,
) -> bool:
    """Resolve a single slot name against *schema*.  Public via ``StateSchema``."""
    t = schema.targets
    if slot == "target":
        return t.target is not None and t.target in ent_ids
    if slot == "blocker":
        return t.blocker is not None and t.blocker in ent_ids
    if slot == "constraint":
        return bool(t.constraint and t.constraint.lower() not in {"null", "none"})
    if slot == "candidate_set":
        return bool(t.candidate_set)
    if slot == "history_anchor":
        return (
            t.history_anchor is not None and t.history_anchor in ent_ids
        )
    # Raw entity id (``e3``)
    if re.fullmatch(r"e\d+", slot):
        return slot in ent_ids
    # Entity attribute reference (``e3.value``)
    m = re.fullmatch(r"(e\d+)\.(\w+)", slot)
    if m:
        eid, field_name = m.group(1), m.group(2)
        ent = schema.entities.get(eid)
        if ent is None:
            return False
        if field_name == "state":
            return bool(ent.state)
        if field_name == "value":
            return bool(ent.value)
        if field_name == "pos":
            return ent.pos is not None
        if field_name == "bid":
            return bool(ent.bid)
        if field_name == "affords":
            return bool(ent.affords)
        return field_name in ent.attributes
    return False


# ──────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────


def _split_sections(schema_text: str) -> Dict[str, str]:
    """Return ``{section_name: body}`` for every recognised section.

    We append a ``</state>`` sentinel before running the regex because
    the canonical ``_SECTION_RE`` looks ahead for either another
    ``<section>`` tag or ``</state>``.  When the caller already stripped
    the closing tag, the last section would otherwise never match.
    """
    haystack = schema_text if "</state>" in schema_text else schema_text + "</state>"
    out: Dict[str, str] = {}
    for m in _SECTION_RE.finditer(haystack):
        out[m.group("name")] = m.group("body").strip()
    return out


def _parse_pos(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    v = value.strip().strip("{}[]()")
    if v.lower() in {"null", "none", ""}:
        return None
    parts = [p.strip() for p in v.split(",")]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        return None
    return (x, y, w, h)


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"true", "yes", "1"}:
        return True
    if v in {"false", "no", "0"}:
        return False
    if v in {"null", "none", ""}:
        return None
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if v.lower() in {"null", "none", ""}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    v = value.strip()
    if v.lower() in {"null", "none", ""}:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _parse_entities(body: str) -> Tuple[Dict[str, Entity], List[str]]:
    entities: Dict[str, Entity] = {}
    order: List[str] = []
    for m in _ENTITY_LINE_RE.finditer(body):
        eid = m.group(1)
        inline = m.group(2)
        if eid in entities:
            continue
        ent = Entity(eid=eid)

        ent.type = (_extract_inline_field_value(inline, "type") or "").strip()
        ent.label = (_extract_inline_field_value(inline, "label") or "").strip()
        bid_val = _extract_inline_field_value(inline, "bid")
        if bid_val is not None:
            bid_val = bid_val.strip()
            ent.bid = None if bid_val.lower() in {"null", "none", ""} else bid_val
        ent.pos = _parse_pos(_extract_inline_field_value(inline, "pos"))
        ent.ontology = (
            _extract_inline_field_value(inline, "ontology") or ""
        ).strip()

        known = {"type", "label", "bid", "pos", "ontology"}
        for tok in _split_top_level_commas(inline):
            mm = re.match(r"(\w+)\s*=\s*(.+)$", tok)
            if mm and mm.group(1) not in known:
                ent.extra[mm.group(1)] = mm.group(2).strip()

        entities[eid] = ent
        order.append(eid)
    return entities, order


def _decorate_with_attributes(entities: Dict[str, Entity], body: str) -> None:
    for m in _ATTR_LINE_RE.finditer(body):
        eid, field_name, value = m.group(1), m.group(2), m.group(3).strip()
        ent = entities.get(eid)
        if ent is None:
            continue
        if field_name == "state":
            ent.state = None if value.lower() in {"null", "none"} else value
        elif field_name == "value":
            ent.value = None if value.lower() in {"null", "none"} else value
        else:
            ent.attributes[field_name] = value


def _decorate_with_affordances(entities: Dict[str, Entity], body: str) -> None:
    for m in _AFFORDS_LINE_RE.finditer(body):
        eid, inner = m.group(1), m.group(2)
        ent = entities.get(eid)
        if ent is None:
            continue
        ent.affords = [
            tok.strip() for tok in inner.split(",") if tok.strip()
        ]


def _decorate_with_uncertainty(entities: Dict[str, Entity], body: str) -> None:
    for m in _UNCERTAINTY_LINE_RE.finditer(body):
        eid, field_name, level = m.group(1), m.group(2), m.group(3)
        ent = entities.get(eid)
        if ent is None:
            continue
        ent.uncertainty[field_name] = level


def _parse_relations(body: str) -> List[Relation]:
    rels: List[Relation] = []
    for m in _RELATION_LINE_RE.finditer(body):
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        rels.append(Relation(name=name, args=args))
    return rels


def _parse_state_flags(body: str) -> StateFlags:
    sf = StateFlags()
    if not body:
        return sf
    pm = _PROGRESS_RE.search(body)
    sf.progress = _parse_float(pm.group(1) if pm else None)
    ph = _PHASE_RE.search(body)
    if ph:
        v = ph.group(1).strip()
        sf.phase = None if v.lower() in {"null", "none", ""} else v
    sc = _SCENE_TYPE_RE.search(body)
    if sc:
        v = sc.group(1).strip()
        sf.scene_type = None if v.lower() in {"null", "none", ""} else v
    er = _ERROR_RE.search(body)
    if er:
        v = er.group(1).strip()
        sf.error = None if v.lower() in {"null", "none", ""} else v
    do = _DIALOG_OPEN_RE.search(body)
    sf.dialog_open = _parse_bool(do.group(1) if do else None)
    ip = _INPUT_PENDING_RE.search(body)
    sf.input_pending = _parse_bool(ip.group(1) if ip else None)
    return sf


def _parse_targets(body: str, known_eids: set) -> Targets:
    t = Targets()
    if not body:
        return t

    tm = _TARGET_FIELD_RE.search(body)
    if tm:
        v = tm.group(1).strip()
        if v.lower() not in {"null", "none", ""}:
            # strip surrounding braces if the VLM wrote `{e3}`
            v_clean = v.strip("{}")
            t.target = v_clean if v_clean in known_eids else (
                # tolerate a bare eid-ish token even if not in known set
                v_clean if re.fullmatch(r"e\d+", v_clean) else None
            )

    bm = _BLOCKER_FIELD_RE.search(body)
    if bm:
        v = bm.group(1).strip().strip("{}")
        if v.lower() not in {"null", "none", ""}:
            t.blocker = v if re.fullmatch(r"e\d+", v) else None

    cm = _CONSTRAINT_FIELD_RE.search(body)
    if cm:
        v = cm.group(1).strip()
        t.constraint = None if v.lower() in {"null", "none", ""} else v

    cs = _CANDIDATE_SET_RE.search(body)
    if cs:
        t.candidate_set = [
            tok.strip() for tok in cs.group(1).split(",")
            if tok.strip() and re.fullmatch(r"e\d+", tok.strip())
        ]

    ha = _HISTORY_ANCHOR_RE.search(body)
    if ha:
        v = ha.group(1).strip().strip("{}")
        if v.lower() not in {"null", "none", ""}:
            t.history_anchor = v if re.fullmatch(r"e\d+", v) else None
    return t


def _parse_actions(body: str) -> List[str]:
    actions: List[str] = []
    for m in _ACTION_LINE_RE.finditer(body):
        actions.append(m.group(2).strip())
    return actions


def _parse_evidence(body: str) -> List[Hop]:
    hops: Dict[int, Hop] = {}
    for m in _HOP_FIELD_RE.finditer(body):
        idx = int(m.group(1))
        field_name = m.group(2)
        value = m.group(3).strip()
        hop = hops.setdefault(idx, Hop(idx=idx))
        if field_name == "abstract_op":
            hop.abstract_op = value
        elif field_name == "tool":
            hop.tool = value
        elif field_name == "result_ref":
            cleaned = value.strip().strip("{}[]")
            hop.result_ref = _EID_TOKEN_RE.findall(cleaned)
        elif field_name == "frame":
            hop.frame = _parse_int(value)
        elif field_name == "timestamp":
            hop.timestamp = _parse_float(value)
        elif field_name == "confidence":
            hop.confidence = None if value.lower() in {"null", "none"} else value
        else:
            hop.extra[field_name] = value
    return [hops[k] for k in sorted(hops)]


def _parse_answer(schema_text: str) -> Optional[Answer]:
    fields = _parse_answer_block(schema_text)
    if not fields.get("answer") and not fields.get("grounding"):
        return None
    grounding_raw = fields.get("grounding") or ""
    grounding = _EID_TOKEN_RE.findall(grounding_raw)
    return Answer(
        answer=fields.get("answer"),
        grounding=grounding,
        confidence=fields.get("confidence"),
    )


def _parse_header(schema_text: str) -> Dict[str, str]:
    """Parse the header fields that sit before the first section tag."""
    head_end = schema_text.find("<entities>")
    head = schema_text if head_end < 0 else schema_text[:head_end]
    out: Dict[str, str] = {}
    for m in _HEADER_FIELD_RE.finditer(head):
        out[m.group(1)] = m.group(2).strip()
    return out


def parse_state_schema(schema_text: Optional[str]) -> Optional[StateSchema]:
    """Parse a ``<state>…</state>`` block into a :class:`StateSchema`.

    Returns ``None`` for empty input or when the ``<state>`` delimiters
    are absent.  Does NOT validate the schema — for that use
    :func:`vlm_wrapper.schema.semantic_validate`.  The parser is
    permissive by design: garbage in any single section yields an empty
    dataclass for that section but does not fail the whole parse.
    """
    if not schema_text:
        return None
    if "<state>" not in schema_text or "</state>" not in schema_text:
        return None

    # Trim to inside of the <state> block so header parsing works even
    # when the model wrapped the schema in extra prose.
    start = schema_text.find("<state>") + len("<state>")
    end = schema_text.rfind("</state>")
    body = schema_text[start:end]

    header = _parse_header(body)
    sections = _split_sections(body)

    entities, order = _parse_entities(sections.get("entities", ""))
    _decorate_with_attributes(entities, sections.get("attributes", ""))
    _decorate_with_affordances(entities, sections.get("affordances", ""))
    _decorate_with_uncertainty(entities, sections.get("uncertainty", ""))

    schema = StateSchema(
        domain=header.get("domain", ""),
        task=header.get("task", ""),
        goal=header.get("goal", ""),
        step=_parse_int(header.get("step")),
        entities=entities,
        entity_order=order,
        relations=_parse_relations(sections.get("relations", "")),
        state_flags=_parse_state_flags(sections.get("state_flags", "")),
        targets=_parse_targets(sections.get("targets", ""), set(entities)),
        actions=_parse_actions(sections.get("actions", "")),
        evidence=_parse_evidence(sections.get("evidence", "")),
        answer=_parse_answer(schema_text),
        raw=schema_text,
    )
    return schema


# ──────────────────────────────────────────────────────────────────────
# Entity-referenced action resolution (PLAN §7 Phase 3)
# ──────────────────────────────────────────────────────────────────────


# Matches the common forms the actor / LLM might emit:
#   click(e5)       → ("click", "e5")
#   click e5        → ("click", "e5")
#   type[e3]=foo    → ("type", "e3")
#   e5              → (None, "e5")
_ENTITY_REF_PATTERNS = [
    re.compile(r"^\s*(\w+)\s*\(\s*(e\d+)\s*\)\s*$"),
    re.compile(r"^\s*(\w+)\s*\[\s*(e\d+)\s*\]"),
    re.compile(r"^\s*(\w+)\s+(e\d+)\b"),
    re.compile(r"^\s*(e\d+)\s*$"),
]


@dataclass
class ResolvedAction:
    """An action string with any entity reference resolved to real payload.

    Used by the Actor Agent action resolver: the LLM is encouraged to emit
    ``click(e5)`` rather than raw coordinates, and this class captures the
    mapping back to the concrete environment action (``click(bid=…)`` or
    ``click(x,y)`` or a game move).
    """

    raw: str
    verb: Optional[str] = None
    eid: Optional[str] = None
    entity: Optional[Entity] = None
    # The action string the env can actually execute.  Callers decide how
    # to fold in ``entity.bid`` / ``entity.pos`` — we just expose both.
    resolved: str = ""


def resolve_entity_action(
    action_text: str,
    schema: Optional[StateSchema],
) -> ResolvedAction:
    """Parse *action_text* and resolve any ``eN`` reference against *schema*.

    * If the text has no entity reference, returns the text unchanged.
    * If it has an entity reference but no schema / unknown eid, the
      reference is preserved verbatim so the env-side runner can decide
      what to do (log a warning, fall back to a coarse click, etc.).
    * If the entity has a ``bid``, the resolved action uses the bid; else
      it uses the label; else it falls back to the raw string.

    We intentionally do not rewrite coordinate tuples here — the final
    action format is env-specific (BrowserGym uses ``click(bid)``,
    OSWorld uses ``click(x,y)``, Gym-V uses move strings) and belongs in
    the env-adapter layer, not this shared resolver.
    """
    raw = (action_text or "").strip()
    res = ResolvedAction(raw=raw, resolved=raw)

    for pat in _ENTITY_REF_PATTERNS:
        m = pat.match(raw)
        if not m:
            continue
        if len(m.groups()) == 2:
            res.verb = m.group(1)
            res.eid = m.group(2)
        else:
            res.eid = m.group(1)
        break

    if res.eid and schema is not None:
        ent = schema.get_entity(res.eid)
        if ent is not None:
            res.entity = ent
            if res.verb and ent.bid:
                res.resolved = f"{res.verb}({ent.bid})"
            elif res.verb and ent.label:
                res.resolved = f"{res.verb}({ent.label})"
    return res


__all__ = [
    "Answer",
    "Entity",
    "Hop",
    "Relation",
    "ResolvedAction",
    "StateFlags",
    "StateSchema",
    "Targets",
    "parse_state_schema",
    "resolve_entity_action",
]
