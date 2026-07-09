"""
Adapter: ``<state>...</state>`` grounding schema → skill-agent predicate map.

This is the glue between the unified visual grounding head
(``vlm_wrapper.schema``, used by ``visual_grounding_tests/generate_*_schema.py``)
and the skill-mining pipeline (``skill_agents.stage3_mvp``). Stage 3 contract
learning expects a ``Dict[str, float]`` predicate-probability map per
timestep; this module produces one from a parsed ``<state>`` block.

Plug-in pattern (mirrors ``skill_agents.default_predicates``)::

    from skill_agents.stage3_mvp.extract_predicates import (
        CompositePredicateExtractor,
    )
    from skill_agents.stage3_mvp.predicate_vocab import PredicateVocab
    from skill_agents.schema_predicates import schema_to_predicates

    vocab = PredicateVocab()
    extractor = CompositePredicateExtractor(vocab)
    extractor.add_source(schema_to_predicates)

    preds = extractor(experience.summary_state)  # str or Experience-like

Predicate key conventions
-------------------------

The emitted keys form a small, stable vocabulary the contract learner
(``stage3_mvp.contract_learn``) and bank maintenance (``bank_maintenance``)
both consume. Keys are flat strings; the prefix encodes the source
section so cross-domain skills can match against typed slots
(see ``skill_agents/skill_template.py``):

* ``entity:<eid>:exists``                  -- entity is present this frame
* ``entity:<eid>:type:<canonical_type>``   -- one of ``element|object|region|text``
* ``entity:<eid>:ontology:<role>``         -- ``selectable_entity``, ``goal_indicator``, ...
* ``attr:<eid>:<key>=<value>``             -- ``<attributes>`` lines (``e1.state=visible``)
* ``afford:<eid>:<verb>``                  -- one entry per ``<affordances>`` verb
* ``flag:<name>``                          -- boolean ``<state_flags>`` (e.g. ``flag:dialog_open``)
* ``flag:<name>=<bucket>``                 -- discretised ``<state_flags>`` (``progress=high``)
* ``flag:scene_type=<value>``              -- direct passthrough
* ``rel:<verb>:<eA>:<eB>``                 -- ``<relations>`` triples (binary form)
* ``rel:<verb>:<eA>:<eB>:<eC>``            -- 3-arity relations
* ``target:eid=<eid>``                     -- the chosen ``target=`` entity
* ``target:blocker=<eid>``                 -- the ``blocker=`` entity (omitted if null)
* ``target:candidate:<eid>``               -- one entry per candidate-set member

Probabilities
-------------

By default every predicate is emitted with probability ``1.0``. The
``<uncertainty>`` block attenuates per-(eid, field) confidences to
``0.4`` when the model marked them ``high``-uncertainty, ``0.6`` for
``medium``, and leaves them at ``1.0`` for ``low``.  Stage 3
booleanises predicates by thresholding at ~0.5, so ``high`` uncertainty
flips an entity-attribute predicate into the "absent" bucket — exactly
the desired behaviour.

The function is a pure parser: no I/O, no LLM, no global state. It
accepts any of: a raw ``<state>...</state>`` string, the dict-form
record produced by ``generate_envwrappers_visual_schema.py`` /
``generate_gymv_image_schema.py`` (with a ``schema_image_llm`` /
``schema_text_llm`` / ``schema_canonical`` / ``schema`` key, in that
preference order), an Experience-like object exposing ``summary_state``
or ``state`` attributes, or ``None`` (returns ``{}``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────

# Probabilities used to attenuate per-(eid, field) predicates that the
# schema marks uncertain.  Stage 3 booleanises at ~0.5, so anything
# below that flips into the "absent" bucket on contract learning.
_UNCERTAINTY_PROB: Dict[str, float] = {
    "high": 0.4,
    "medium": 0.6,
    "low": 1.0,
}

# Buckets for continuous-valued state flags (``progress=0.0..1.0``).
# Stage 3 prefers categorical predicates so the contract is interpretable;
# we discretise rather than emitting continuous values.
_PROGRESS_BUCKETS: Tuple[Tuple[float, str], ...] = (
    (0.33, "low"),
    (0.66, "mid"),
    (1.01, "high"),
)


# ─────────────────────────────────────────────────────────────────────
# Section / line regexes
#
# These mirror the patterns in ``vlm_wrapper.schema`` (which keeps them
# private) so this module remains self-contained — refactoring schema
# internals will not silently break predicate extraction. Where the two
# ever diverge, the schema spec at ``vlm_wrapper/schema.py`` is the
# source of truth and this module should be updated.
# ─────────────────────────────────────────────────────────────────────

_STATE_BLOCK_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL)

_SECTION_RE = re.compile(
    r"<(?P<name>entities|attributes|affordances|relations|state_flags|"
    r"targets|uncertainty|actions|evidence|evidence_refs|derivations|answer)>"
    r"(?P<body>.*?)(?=<\w+>|</state>)",
    re.DOTALL,
)

_ENTITY_LINE_RE = re.compile(r"^(e\d+)\s*\[(.*?)\]\s*$", re.MULTILINE)
_ENTITY_FIELD_RE = re.compile(r"(\w+)\s*=\s*([^,\]]+)")

_ATTR_LINE_RE = re.compile(
    r"^(e\d+)\.([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$",
    re.MULTILINE,
)

_AFFORDS_LINE_RE = re.compile(
    r"^(e\d+)\.affords\s*=\s*\[([^\]]*)\]\s*$",
    re.MULTILINE,
)

_RELATION_LINE_RE = re.compile(
    r"^(\w+)\(\s*(e\d+(?:\s*,\s*e\d+)*)\s*\)\s*$",
    re.MULTILINE,
)

_FLAG_LINE_RE = re.compile(
    r"^([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$",
    re.MULTILINE,
)

_TARGET_FIELD_RE = re.compile(r"^target\s*=\s*(.+?)\s*$", re.MULTILINE)
_BLOCKER_FIELD_RE = re.compile(r"^blocker\s*=\s*(.+?)\s*$", re.MULTILINE)
_CANDIDATE_SET_RE = re.compile(r"^candidate_set\s*=\s*\[([^\]]*)\]", re.MULTILINE)
_HISTORY_ANCHOR_RE = re.compile(r"^history_anchor\s*=\s*(.+?)\s*$", re.MULTILINE)

_UNCERTAINTY_LINE_RE = re.compile(
    r"^(e\d+)\.([A-Za-z_][\w]*)\s*=\s*(high|medium|low)\s*$",
    re.MULTILINE,
)

_EID_RE = re.compile(r"\be\d+\b")
_NULL_TOKENS: frozenset[str] = frozenset({"null", "none", "nan", "n/a", "-", ""})


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def schema_to_predicates(obs: Any) -> Dict[str, float]:
    """Extract predicate probabilities from a ``<state>`` schema.

    Compatible with ``CompositePredicateExtractor.add_source(...)``.

    Parameters
    ----------
    obs
        Any of: the raw ``<state>...</state>`` text, a per-step dict
        produced by the unified visual grounding head, or an
        Experience-like object exposing ``summary_state`` /
        ``state`` / ``schema``. ``None`` and unparseable inputs return
        ``{}``.

    Returns
    -------
    Dict[str, float]
        Predicate-probability map. Boolean predicates are ``1.0``;
        per-(eid, field) predicates marked ``high`` uncertainty are
        attenuated to ``0.4`` (below Stage 3's booleanisation
        threshold).
    """
    state_text = _extract_state_block(obs)
    if not state_text:
        return {}

    sections = _split_sections(state_text)
    uncertainty = _parse_uncertainty(sections.get("uncertainty", ""))

    preds: Dict[str, float] = {}

    entities = _parse_entities(sections.get("entities", ""))
    for eid, fields in entities.items():
        u_prob = uncertainty.get((eid, "*"), 1.0)
        preds[f"entity:{eid}:exists"] = u_prob
        if (etype := fields.get("type")) and not _is_null(etype):
            preds[f"entity:{eid}:type:{etype}"] = u_prob
        if (ontology := fields.get("ontology")) and not _is_null(ontology):
            preds[f"entity:{eid}:ontology:{ontology}"] = u_prob

    for eid, key, value in _parse_attributes(sections.get("attributes", "")):
        if _is_null(value):
            continue
        u_prob = uncertainty.get((eid, key), uncertainty.get((eid, "*"), 1.0))
        preds[f"attr:{eid}:{key}={value}"] = u_prob

    for eid, verbs in _parse_affordances(sections.get("affordances", "")):
        u_prob = uncertainty.get((eid, "affords"), uncertainty.get((eid, "*"), 1.0))
        for verb in verbs:
            preds[f"afford:{eid}:{verb}"] = u_prob

    for verb, args in _parse_relations(sections.get("relations", "")):
        key = f"rel:{verb}:" + ":".join(args)
        preds[key] = 1.0

    for flag_key, flag_value in _parse_state_flags(sections.get("state_flags", "")):
        for k, v in _flag_to_predicates(flag_key, flag_value):
            preds[k] = v

    for k, v in _parse_targets(sections.get("targets", "")).items():
        preds[k] = v

    return preds


# ─────────────────────────────────────────────────────────────────────
# CompositePredicateExtractor convenience
# ─────────────────────────────────────────────────────────────────────


def register_with(extractor: Any) -> None:
    """Register :func:`schema_to_predicates` as a source on a composite extractor.

    Equivalent to ``extractor.add_source(schema_to_predicates)`` but
    documents the intent at call sites and stays a one-liner if the
    extractor API gains keyword arguments later.
    """
    extractor.add_source(schema_to_predicates)


# ─────────────────────────────────────────────────────────────────────
# Input coercion: pull a <state>...</state> string out of arbitrary obs
# ─────────────────────────────────────────────────────────────────────


_RECORD_KEY_PREFERENCE: Tuple[str, ...] = (
    "schema_image_llm",
    "schema_text_llm",
    "schema_canonical",
    "schema_text_heuristic",
    "schema",
    "summary_state",
    "state",
)


def _extract_state_block(obs: Any) -> Optional[str]:
    """Coerce *obs* into a ``<state>...</state>`` string or ``None``."""
    if obs is None:
        return None

    if isinstance(obs, str):
        return _slice_state_block(obs)

    if isinstance(obs, Mapping):
        for key in _RECORD_KEY_PREFERENCE:
            value = obs.get(key)
            if value is None:
                continue
            if isinstance(value, Mapping):
                inner = value.get("schema")
                if isinstance(inner, str):
                    block = _slice_state_block(inner)
                    if block:
                        return block
            elif isinstance(value, str):
                block = _slice_state_block(value)
                if block:
                    return block
        return None

    for attr in _RECORD_KEY_PREFERENCE:
        value = getattr(obs, attr, None)
        if isinstance(value, str):
            block = _slice_state_block(value)
            if block:
                return block

    return None


def _slice_state_block(text: str) -> Optional[str]:
    """Return ``<state>...</state>`` if present anywhere in *text*, else ``None``."""
    if not text:
        return None
    if "<state>" not in text or "</state>" not in text:
        return None
    m = _STATE_BLOCK_RE.search(text)
    return m.group(0) if m else None


# ─────────────────────────────────────────────────────────────────────
# Section split + per-section parsers
# ─────────────────────────────────────────────────────────────────────


def _split_sections(state_text: str) -> Dict[str, str]:
    """Split a ``<state>...</state>`` block into ``{section_name: body}``."""
    sections: Dict[str, str] = {}
    for match in _SECTION_RE.finditer(state_text):
        sections[match.group("name")] = match.group("body")
    return sections


def _parse_entities(body: str) -> Dict[str, Dict[str, str]]:
    """Parse the ``<entities>`` body into ``{eid: {field: value}}``."""
    out: Dict[str, Dict[str, str]] = {}
    for match in _ENTITY_LINE_RE.finditer(body):
        eid = match.group(1)
        fields: Dict[str, str] = {}
        for k, v in _ENTITY_FIELD_RE.findall(match.group(2)):
            fields[k.strip()] = v.strip()
        out[eid] = fields
    return out


def _parse_attributes(body: str) -> Iterable[Tuple[str, str, str]]:
    """Yield ``(eid, key, value)`` for each ``e1.state=visible`` line.

    Skips ``e1.affords=[...]`` lines (handled by :func:`_parse_affordances`).
    """
    for match in _ATTR_LINE_RE.finditer(body):
        eid, key, value = match.group(1), match.group(2), match.group(3)
        if key == "affords":
            continue
        yield eid, key, value.strip()


def _parse_affordances(body: str) -> Iterable[Tuple[str, List[str]]]:
    """Yield ``(eid, [verb, ...])`` for each ``<affordances>`` line."""
    for match in _AFFORDS_LINE_RE.finditer(body):
        eid = match.group(1)
        verbs = [v.strip() for v in match.group(2).split(",") if v.strip()]
        if verbs:
            yield eid, verbs


def _parse_relations(body: str) -> Iterable[Tuple[str, List[str]]]:
    """Yield ``(verb, [eid, eid, ...])`` for each ``contains(e1,e2)`` line."""
    for match in _RELATION_LINE_RE.finditer(body):
        verb = match.group(1).strip()
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        if args:
            yield verb, args


def _parse_state_flags(body: str) -> Iterable[Tuple[str, str]]:
    """Yield ``(flag_name, raw_value)`` for each ``key=value`` line.

    Skips lines that look like attributes / relations / parenthesised
    schema-spec comments. The ``<state_flags>`` body is otherwise just
    a flat ``key=value`` map.
    """
    for match in _FLAG_LINE_RE.finditer(body):
        key, value = match.group(1).strip(), match.group(2).strip()
        if key.startswith("e") and key[1:].isdigit():
            continue
        yield key, value


def _parse_targets(body: str) -> Dict[str, float]:
    """Emit ``target:*`` predicates from the ``<targets>`` body."""
    out: Dict[str, float] = {}

    if (m := _TARGET_FIELD_RE.search(body)) and not _is_null(m.group(1)):
        eid = _first_eid(m.group(1))
        if eid:
            out[f"target:eid={eid}"] = 1.0

    if (m := _BLOCKER_FIELD_RE.search(body)) and not _is_null(m.group(1)):
        eid = _first_eid(m.group(1))
        if eid:
            out[f"target:blocker={eid}"] = 1.0

    if (m := _CANDIDATE_SET_RE.search(body)):
        for eid in _EID_RE.findall(m.group(1)):
            out[f"target:candidate:{eid}"] = 1.0

    if (m := _HISTORY_ANCHOR_RE.search(body)) and not _is_null(m.group(1)):
        eid = _first_eid(m.group(1))
        if eid:
            out[f"target:history_anchor={eid}"] = 1.0

    return out


def _parse_uncertainty(body: str) -> Dict[Tuple[str, str], float]:
    """Parse ``e1.field=high`` lines into ``{(eid, field): prob}``.

    A wildcard entry ``(eid, "*")`` is also written when *every*
    declared field for an entity has the same level — Stage 3 then
    attenuates all that entity's predicates uniformly.
    """
    per_field: Dict[Tuple[str, str], float] = {}
    levels_per_entity: Dict[str, set[str]] = {}
    for match in _UNCERTAINTY_LINE_RE.finditer(body):
        eid, field, level = match.group(1), match.group(2), match.group(3)
        prob = _UNCERTAINTY_PROB.get(level, 1.0)
        per_field[(eid, field)] = prob
        levels_per_entity.setdefault(eid, set()).add(level)

    for eid, levels in levels_per_entity.items():
        if len(levels) == 1:
            per_field[(eid, "*")] = _UNCERTAINTY_PROB.get(next(iter(levels)), 1.0)

    return per_field


# ─────────────────────────────────────────────────────────────────────
# Flag value → predicate(s)
# ─────────────────────────────────────────────────────────────────────


def _flag_to_predicates(key: str, value: str) -> Iterable[Tuple[str, float]]:
    """Translate one ``<state_flags>`` entry into 0+ predicate emissions."""
    if _is_null(value):
        return
    lowered = value.lower()

    if lowered in {"true", "yes"}:
        yield (f"flag:{key}", 1.0)
        return
    if lowered in {"false", "no"}:
        return

    if key == "progress":
        try:
            v = float(value)
        except ValueError:
            return
        for upper, label in _PROGRESS_BUCKETS:
            if v < upper:
                yield (f"flag:progress={label}", 1.0)
                return
        return

    yield (f"flag:{key}={value}", 1.0)


# ─────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────


def _is_null(value: Optional[str]) -> bool:
    return value is None or value.strip().lower() in _NULL_TOKENS


def _first_eid(value: str) -> Optional[str]:
    """Return the first ``e\\d+`` token in *value*, or ``None``."""
    if not value:
        return None
    m = _EID_RE.search(value)
    return m.group(0) if m else None


__all__ = [
    "schema_to_predicates",
    "register_with",
]
