"""Single-shot QA lift driver — VTB / TIR-Bench / Video-Holmes / SIV-Bench.

Each input file is one rollout: ``{schema, answer_reasoning, answer,
gold_answer, correct, judge, valid_actions, options_block,
raw_sample, ...}``. There is no ``experiences[]`` sequence to
segment; the whole trace is the schema + reasoning chain + final
answer.

Lift contract (per sample):

    schema           -> GameSchemaIndex (auto-mined entity labels +
                        explicit ``e\\d+`` IDs)
    answer_reasoning -> prose hops (sentence-tokenized, with implicit
                        GROUND for cited entities prepended)
    answer           -> COMMIT hop notes
    gold_answer      -> verification target (recorded in
                        report.expected_answer)
    correct          -> verified_status / report.overall_pass_rate

The lifted protocol is then routed through
:func:`labeling._protocol_lift.lift_protocol_to_typed_hops` exactly
the way the labeling/ pipeline routes lifted env_wrappers/gym_v
skills. Output records match the
``labeling/skill_bank_out/run_<ts>/<corpus>/<game>/skill_bank.jsonl``
``{report, skill}`` shape verified empirically against
``run_20260430_030637/env_wrappers/twenty_forty_eight/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from labeling._protocol_lift import (
    GameSchemaIndex,
    LiftStats,
    _parse_schema_block_entities,
    lift_protocol_to_typed_hops,
)

from ._corpus_specs import CorpusSpec


# ── Entity-reference parsing ───────────────────────────────────────────

#: ``e\d+`` identifier — ``\b`` word-boundary on both sides so
#: ``e-mail`` doesn't collide.
_ENTITY_REF = re.compile(r"\be(\d+)\b")


def _entity_ids_in_text(text: str) -> List[str]:
    """Return ordered, deduplicated list of ``e\\d+`` IDs cited in *text*."""
    seen: Dict[str, None] = {}
    for m in _ENTITY_REF.finditer(text or ""):
        eid = f"e{m.group(1)}"
        seen.setdefault(eid, None)
    return list(seen.keys())


def _entity_id_to_label(schema: str) -> Dict[str, str]:
    """Map ``e\\d+`` ids to their ``label=`` attribute from a schema's
    ``<entities>`` block.

    VR/video schemas declare entities as
    ``e1[type=object, label=bearded man, bid=null, pos=..., ontology=tracked_entity]``.
    """
    out: Dict[str, str] = {}
    rx = re.compile(
        r"^(e\d+)\[type=([\w\-]+)(?:,\s*([^\]]+))?\]\s*$",
        re.M,
    )
    for m in rx.finditer(schema or ""):
        eid = m.group(1)
        attrs = (m.group(3) or "")
        d: Dict[str, str] = {}
        for kv in attrs.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k.strip()] = v.strip().strip("'\"")
        label = d.get("label")
        if label:
            out[eid] = label
    return out


# ── Schema-index construction ─────────────────────────────────────────

def build_schema_index_from_sample(
    schema: str,
    *,
    benchmark: str,
) -> GameSchemaIndex:
    """Build a `GameSchemaIndex` directly from one sample's schema string.

    Mirrors :func:`labeling._protocol_lift.build_schema_index_for_game`
    but reads from an in-memory schema string rather than walking
    ``actions_root/<corpus>/<game>/episode_*.json``. Single-shot QA
    has no actions_root — every sample carries its own schema.

    The resulting index's ``entity_labels`` includes BOTH the
    human-readable ``label=`` strings (e.g. "bearded man") AND the
    canonical ``e\\d+`` IDs, since reasoning chains can reference
    either. Both map to ``ontology="tracked_entity"`` as a sane default
    for the v0 slot-binder; the auto-mined ontology wins on collision.
    """
    label_to_ontology: Dict[str, str] = {}

    # 1) Auto-mine from <entities>: human label -> ontology.
    for label, ontology in _parse_schema_block_entities(schema or ""):
        label_to_ontology.setdefault(label, ontology)

    # 2) Add e_N ids themselves so reasoning chains that reference
    #    "e3" by id (not by human label) also slot-bind. ID inherits
    #    the same ontology as the entity it names.
    eid_to_label = _entity_id_to_label(schema or "")
    for eid, label in eid_to_label.items():
        ontology = label_to_ontology.get(label) or "tracked_entity"
        label_to_ontology.setdefault(eid, ontology)

    return GameSchemaIndex(
        game=benchmark,
        entity_labels=frozenset(label_to_ontology.keys()),
        label_to_ontology=dict(label_to_ontology),
        affordances_by_ontology={},
    )


# ── Reasoning-chain -> prose-hop list ─────────────────────────────────

#: Sentence boundary — coarse but adequate for ``answer_reasoning`` text.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    return sents


#: Heuristic patterns mapping narrative-reasoning sentence starts to
#: classifier-recognised verbs in
#: :data:`labeling._protocol_lift.VERB_TABLE`. Order matters -- first
#: hit wins. Patterns are case-insensitive.
#:
#: Tightening rationale (post audit-2026-05-01): the previous v1 rule
#: ``does\s+not | cannot | no\s+evidence | absent -> penalize`` fired
#: 128 times in 100 browsergym episodes, almost all false positives
#: ("scrolling had no effect", "this step only allows..."). The v2
#: rules below require an explicit avoidance / discard signal at the
#: clause head, not just inside the sentence. Likewise the v1 rule
#: ``\bis\s+\w -> evaluate`` matched almost any declarative sentence
#: ("X is Y") and is replaced with a clause-head pattern.
_SENTENCE_REWRITES: List[Tuple[re.Pattern, str]] = [
    # Comparative / proportionality: "Among the options ...", "The best match is ..."
    (re.compile(r"^(among|of)\s+(the\s+)?(options|candidates|choices)", re.I), "compare"),
    (re.compile(r"\b(best\s+match|closest\s+to|most\s+(visible|likely|relevant|plausible))", re.I), "compare"),
    # Logical inference: "Therefore ...", "Since ...", "So ..."
    (re.compile(r"^(therefore|hence|thus|so|since|because)\b", re.I), "evaluate"),
    # Math / derivation: "Subtracting gives ...", "Substituting yields ...", "X equals Y"
    (re.compile(r"^(subtract\w*|add\w*|multipl\w*|divid\w*|substitut\w*|comput\w*|deriv\w*)\b", re.I), "compute"),
    (re.compile(r"\b(yields?|gives?|equals?|implies?|means)\b", re.I), "evaluate"),
    # Evidence reporting: "The schema highlights ...", "The chat text anchors ..."
    (re.compile(r"^(the\s+)?(schema|chat|notes?|text|evidence|frame|video|image|figure)\b", re.I), "inspect"),
    (re.compile(r"\b(highlights?|anchors?|shows?|indicates?|suggests?|displays?|depicts?)\b", re.I), "inspect"),
    # Selection / decision: "Pick option B", "We choose ..."
    (re.compile(r"^(pick|choose|select|opt\b)", re.I), "select"),
    # Verification: "Confirm ...", "This matches ..."
    (re.compile(r"^(confirm|check|verify)\b", re.I), "verify"),
    (re.compile(r"^this\s+matches\b", re.I), "verify"),
    # Counting: "There are 4 rebars"
    (re.compile(r"^there\s+(are|is)\s+\d", re.I), "report"),
    # v2 narrowed PENALIZE: only when the sentence head is an explicit
    # avoidance / discard signal. Drops the over-broad
    # ``does\s+not | cannot | no\s+evidence | absent`` matches that
    # mistagged 128 browsergym sentences.
    (re.compile(r"^(avoid|reject|skip|discard|exclude|ignore|rule\s+out|discount)\b", re.I), "penalize"),
    (re.compile(r"^there\s+(is|are)\s+no\b", re.I), "penalize"),
    # Action-leading: "Clicking ...", "Filling ...", "Entering ..." -> EXECUTE.
    # These dominated browsergym fallbacks (~10% of all browser hops).
    (re.compile(r"^(click\w*|fill\w*|enter\w*|press\w*|scroll\w*|navigat\w*|tap\w*|drag\w*|drop\w*|hover\w*|going\b|submit\w*)", re.I), "perform"),
    # State observation: "We are on the X page", "We're back on..."
    (re.compile(r"^(we\s+are|we['\u2019]re|we\s+have|we\s+need)\b", re.I), "inspect"),
    # Goal/intent: "To restore X ...", "To complete the task ..."
    (re.compile(r"^to\s+\w+\s+", re.I), "evaluate"),
    # Conditional reasoning: "If X were Y ..."
    (re.compile(r"^if\s+\w+", re.I), "evaluate"),
    # Catch-all declaratives: "The X is Y", "Other Y...", "Recent Z..."
    # Final-tier rule -- everything before this should already have
    # matched a more specific pattern. Without this rule ~83% of
    # browsergym fallbacks (and ~70% of osworld fallbacks) opened
    # with "The"/"A" and ended up as opaque EXEC blobs.
    (re.compile(r"^(other|prior|recent|previous|next|past)\s+\w", re.I), "inspect"),
    (re.compile(r"^(only|just)\s+\w", re.I), "inspect"),
    (re.compile(r"^(the|a|an|this|that|these|those)\s+\w+\s+(is|are|was|were|has|have|had|does|do|did)\b", re.I), "inspect"),
    # Last-resort: any "The X ..." sentence we haven't classified yet
    # is most likely an evidence observation. Order matters -- this
    # MUST come last so more specific rules above win.
    (re.compile(r"^(the|a|an|this|that|these|those)\s+\w", re.I), "inspect"),
]


def _rewrite_sentence_for_lift(sentence: str) -> str:
    """Prepend a classifier-recognised verb when the sentence opens with
    narrative-prose tokens (articles, demonstratives, factual subjects).

    Returns the original sentence unchanged when the first content
    token is already a known verb lemma. The rewritten string is
    intended for the lift's first-token classifier ONLY -- callers
    that need to display the prose to humans should keep the
    pre-rewrite sentence around (see
    :func:`reasoning_to_prose_steps_with_originals`).
    """
    s = sentence.strip()
    if not s:
        return s
    # If the sentence already starts with a known verb, don't touch it.
    head = re.match(r"^\s*([A-Za-z]+)", s)
    if head:
        from labeling._protocol_lift import _LEMMA_INDEX
        if head.group(1).lower() in _LEMMA_INDEX:
            return s
    # First matching pattern wins.
    for pat, prefix_verb in _SENTENCE_REWRITES:
        if pat.search(s):
            return f"{prefix_verb} that {s[0].lower()}{s[1:]}"
    return s


def reasoning_to_prose_steps_with_originals(
    reasoning: str,
    *,
    schema: str,
    answer: str,
) -> List[Tuple[str, str]]:
    """Split a reasoning chain into ``(rewritten, original)`` prose pairs.

    The first element of each pair is fed to the lift's first-token
    classifier (potentially with a narrative-prose -> verb prefix);
    the second element is the original prose, suitable for the
    ``hop.notes`` field after :func:`restore_original_notes` swaps
    them back in.

    The reasoning chain rarely opens with a verb the classifier
    recognises (e.g. "The chat text anchors e3/e4..."), so we
    (a) synthesise an explicit GROUND hop citing every ``e\\d+``
    referenced anywhere in the chain, (b) prefix narrative sentences
    with the closest-matching gaming-taxonomy verb, and (c) synthesise
    an explicit VERIFY + COMMIT pair around the final answer.
    """
    eids = _entity_ids_in_text(reasoning)
    eid_to_label = _entity_id_to_label(schema)

    pairs: List[Tuple[str, str]] = []

    if eids:
        labelled = [
            f"{eid} ({eid_to_label[eid]})" if eid in eid_to_label else eid
            for eid in eids[:6]
        ]
        ground = (
            f"Inspect the cited evidence anchors {', '.join(labelled)} in the schema."
        )
    else:
        ground = "Inspect the schema for the entities most relevant to the question."
    pairs.append((ground, ground))

    sentences = _split_sentences(reasoning)
    if not sentences:
        s = "Reason over the visible evidence and derive the answer."
        pairs.append((s, s))
    else:
        for s in sentences:
            pairs.append((_rewrite_sentence_for_lift(s), s))

    verify = f"Verify the candidate answer ({answer}) against the cited evidence."
    commit = f"Execute the commit by emitting the final answer: {answer}."
    pairs.append((verify, verify))
    pairs.append((commit, f"Commit the final answer: {answer}."))

    return pairs


def reasoning_to_prose_steps(
    reasoning: str,
    *,
    schema: str,
    answer: str,
) -> List[str]:
    """Backwards-compat wrapper returning only the rewritten strings."""
    return [
        rewritten
        for rewritten, _orig in reasoning_to_prose_steps_with_originals(
            reasoning, schema=schema, answer=answer
        )
    ]


def mine_single_shot_effects(
    *,
    sample: Dict[str, Any],
    schema_index: GameSchemaIndex,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build ``(effects_add, effects_del)`` for a single-shot QA sample.

    The canonical
    :func:`labeling._protocol_lift.mine_effects` walks
    ``success_criteria`` / ``abort_criteria`` strings against a
    gaming-centric ``_PREDICATE_TRIGGERS`` table
    (``entity_value_increased``, ``phase_transitioned``, ...) which
    never fires on QA criteria like ``"answer matches gold (D)"``.
    This v0 single-shot miner produces three effects-flavoured
    predicates the harness's transfer machinery can match on:

    - ``answer_emitted{value=<answer>}`` -- always added on COMMIT.
    - ``answer_matches_gold{gold=<gold>}`` -- added when ``correct``.
    - ``entity_grounded{e=<e_id>}`` -- one per ``e\\d+`` cited in
      ``answer_reasoning`` AND present in the schema_index.

    Failure cases (``correct=False``) get an
    ``answer_diverged_from_gold`` predicate in ``effects_del`` to
    signal a non-confirming attempt without polluting the additive
    contract. Phase-2 will replace this with a learned predicate
    miner over the reasoning chain; for now the contract is
    populated, transferable, and obviously v0.
    """
    answer = (sample.get("answer") or "").strip()
    gold = (sample.get("gold_answer") or "").strip()
    reasoning = sample.get("answer_reasoning") or ""
    correct = bool(sample.get("correct"))

    effects_add: List[Dict[str, Any]] = []
    effects_del: List[Dict[str, Any]] = []

    if answer:
        effects_add.append({
            "type": "answer_emitted",
            "args": {"value": answer[:80]},
            "from_phrase": f"agent emitted answer: {answer[:80]}",
        })
    if correct and gold:
        effects_add.append({
            "type": "answer_matches_gold",
            "args": {"gold": gold[:80]},
            "from_phrase": f"answer matches gold: {gold[:80]}",
        })
    if not correct and answer and gold:
        effects_del.append({
            "type": "answer_diverged_from_gold",
            "args": {"answer": answer[:80], "gold": gold[:80]},
            "from_phrase": f"answer {answer[:40]!r} != gold {gold[:40]!r}",
        })

    eids = _entity_ids_in_text(reasoning)
    grounded_seen: set = set()
    for eid in eids:
        if eid in grounded_seen:
            continue
        if eid not in schema_index.entity_labels:
            continue
        grounded_seen.add(eid)
        effects_add.append({
            "type": "entity_grounded",
            "args": {"e": eid, "ontology": schema_index.label_to_ontology.get(eid, "unknown")},
            "from_phrase": f"reasoning chain cites schema entity {eid}",
        })

    return effects_add, effects_del


def _bind_entity_refs_in_payloads(
    typed_hops: List[Dict[str, Any]],
    schema_index: Optional[GameSchemaIndex] = None,
) -> None:
    """Populate ``${slot}`` placeholders with entity references mined
    from the hop's notes.

    The canonical lift's :func:`labeling._protocol_lift.extract_payload_slots`
    deliberately leaves ``"any"``-typed slots (EVALUATE.subject /
    .criterion, COMPARE.lhs / .rhs) as ``${slot}`` placeholders --
    "any" means the lift can't statically choose an entity. For
    transfer experiments we still want bindings, so this post-pass:

    1. Scans for ``e1, e2, ...`` references and fills them into
       placeholders in declaration order (preferred -- canonical IDs).
    2. **Falls back** to scanning for human-readable entity labels
       from the schema_index (e.g. ``"the search box"``, ``"Chrome
       icon"``) when no ``e\\d+`` references exist in the hop. Match
       requires the full label to appear (lowercase) in the notes;
       longer labels win on ties so producer-canonical labels are
       preferred. Browser/OSWorld reasoning prose rarely cites
       ``e\\d+`` IDs (they reference labels), so without this fallback
       100% of browser/osworld ``"any"``-slots remained unbound
       (audit-2026-05-01).

    Mutates ``typed_hops`` in place.
    """
    label_pool: List[Tuple[str, str]] = []  # (label_lower, original_label)
    if schema_index is not None:
        # Prefer human labels over bare e_N ids for label-fallback.
        # Sort by length descending so multi-word labels win matches
        # over their constituent single tokens.
        label_pool = sorted(
            [(l.lower(), l) for l in schema_index.entity_labels
             if not (l.startswith("e") and len(l) <= 4)],
            key=lambda lp: -len(lp[0]),
        )

    for hop in typed_hops:
        notes = hop.get("notes") or ""
        notes_lower = notes.lower()
        eids = _entity_ids_in_text(notes)
        # Build the binding pool: e_N first (canonical), then
        # label-matches found in the notes (descending by length).
        candidates: List[str] = list(eids)
        if not candidates and label_pool:
            for lbl_lower, lbl_orig in label_pool:
                if len(lbl_lower) >= 4 and lbl_lower in notes_lower:
                    candidates.append(lbl_orig)
                if len(candidates) >= 2:
                    break
        if not candidates:
            continue
        payload = hop.get("payload") or {}
        slot_types = hop.get("slot_types") or {}
        slot_idx = 0
        for slot_name in list(payload.keys()):
            val = payload[slot_name]
            if not (isinstance(val, str) and val.startswith("${") and val.endswith("}")):
                continue
            # Respect slot types: enum slots want direction tokens
            # ("up"/"down"/"left"/"right"/"cw"/"ccw"), not entity
            # labels. Skip enums and effect_predicates -- the post-
            # binder only fills entity-shaped slots.
            slot_type = slot_types.get(slot_name)
            if slot_type in {"enum", "effect_predicate"}:
                continue
            if slot_idx < len(candidates):
                payload[slot_name] = candidates[slot_idx]
                slot_idx += 1


def restore_original_notes(
    typed_hops: List[Dict[str, Any]],
    pairs: List[Tuple[str, str]],
) -> None:
    """Swap each hop's ``notes`` from the rewritten-for-classifier
    prose back to the original prose.

    The lift uses each prose step verbatim as the hop's ``notes`` so
    its output is human-readable. We swapped in synthetic
    "evaluate that ..." / "perform that ..." prefixes upstream so the
    classifier could find a known verb; this pass undoes the prefix
    in the *visible* notes while leaving the lift's verb / role /
    payload assignments intact.

    Mutates ``typed_hops`` in place.
    """
    if len(typed_hops) != len(pairs):
        # Length mismatch -> the lift dropped or merged hops; bail
        # out rather than mis-align originals.
        return
    for hop, (_rewritten, original) in zip(typed_hops, pairs):
        if hop.get("notes"):
            hop["notes"] = original


# ── Full single-sample lift ───────────────────────────────────────────

def lift_one_sample(
    sample: Dict[str, Any],
    *,
    spec: CorpusSpec,
    include_incorrect: bool = False,
) -> Optional[Dict[str, Any]]:
    """Lift one VR/video sample into a `{report, skill}` record.

    Returns ``None`` when the sample is filtered out (missing schema /
    reasoning, or ``correct=False`` with ``include_incorrect=False``).
    """
    correct = bool(sample.get("correct"))
    if not include_incorrect and not correct:
        return None

    schema = sample.get("schema") or ""
    reasoning = sample.get("answer_reasoning") or ""
    answer = sample.get("answer") or ""
    gold = sample.get("gold_answer") or ""
    task_id = sample.get("task_id") or sample.get("sample_id") or "unknown"
    if not schema or not reasoning:
        return None

    schema_index = build_schema_index_from_sample(
        schema, benchmark=spec.extra.get("benchmark", spec.name)
    )

    pairs = reasoning_to_prose_steps_with_originals(
        reasoning, schema=schema, answer=answer,
    )
    prose_steps = [rewritten for rewritten, _orig in pairs]

    success_criteria = [
        f"answer matches gold ({gold[:40]})" if not sample.get("is_mcq")
        else f"answer letter equals gold ({gold})"
    ]

    pseudo_skill: Dict[str, Any] = {
        "evidence_role": "COMMIT",
        "protocol": {
            "steps": prose_steps,
            "success_criteria": success_criteria,
            "abort_criteria": [],
        },
    }

    stats = LiftStats()
    typed, contract_add, contract_del = lift_protocol_to_typed_hops(
        pseudo_skill, schema_index=schema_index, stats=stats,
    )

    if typed is None:
        return None

    # Restore original prose to hop.notes (was overwritten with the
    # classifier-friendly rewritten version during the lift).
    restore_original_notes(typed, pairs)
    # Post-bind e\d+ references (and labels as fallback) into
    # otherwise-unbound ${slot} payloads.
    _bind_entity_refs_in_payloads(typed, schema_index=schema_index)
    # v0 single-shot effects miner -- canonical mine_effects produces
    # zero contract entries for QA-style criteria.
    ss_eff_add, ss_eff_del = mine_single_shot_effects(
        sample=sample, schema_index=schema_index,
    )
    contract_add = (contract_add or []) + ss_eff_add
    contract_del = (contract_del or []) + ss_eff_del

    eids = _entity_ids_in_text(reasoning)
    cluster_key = _resolve_cluster_key(sample, spec)

    report = {
        "skill_id": task_id,
        "n_instances": 1,
        "overall_pass_rate": 1.0 if correct else 0.0,
        "eff_add_success_rate": None,
        "eff_del_success_rate": None,
        "eff_event_rate": None,
        "failure_signatures": [],
        "worst_segments": [],
        "lift_stats": {
            "n_hops": stats.n_hops,
            "n_first": stats.n_first,
            "n_rescued": stats.n_rescued,
            "n_fallback_exec": stats.n_fallback_exec,
            "verbs": dict(stats.verbs),
            "fallback_rate": (
                stats.n_fallback_exec / stats.n_hops if stats.n_hops else 0.0
            ),
        },
        "expected_answer": gold,
        "model_answer": answer,
        "judge_correct": correct,
        "n_explicit_entity_refs": len(eids),
    }

    skill = {
        "skill_id": task_id,
        "name": _derive_skill_name(sample, eids, schema_index),
        "strategic_description": _derive_description(sample, prose_steps),
        "applicable_domains": [spec.domain],
        "feasible_tasks": [task_id],
        "feasible_domains": [spec.domain],
        "verified_domains": [spec.domain] if correct else [],
        "verified_tasks": [task_id] if correct else [],
        "evidence_role": "COMMIT",
        "execution_hint": None,
        "expected_tag_pattern": None,
        "protocol": typed,
        "protocol_history": [],
        "protocol_raw": {
            "steps": prose_steps,
            "success_criteria": success_criteria,
            "abort_criteria": [],
        },
        "contract": {
            "effects_add": contract_add,
            "effects_del": contract_del,
        },
        "provenance": {
            "corpus": spec.name,
            "benchmark": spec.extra.get("benchmark", spec.name),
            "modality": spec.modality,
            "bank_kind": "per_sample",
            "source_sample": sample.get("sample_id"),
            "model": sample.get("model"),
            "model_routed": sample.get("model_routed"),
            "schema_source": sample.get("schema_source"),
            "elapsed_seconds": sample.get("elapsed_seconds"),
            "cluster_key": cluster_key,
        },
        "tags": [spec.name, spec.modality, "single_shot", cluster_key or "uncategorized"],
        "n_instances": 1,
        "sub_episodes": [],
        "source_type": "single_shot_qa",
        "status": "draft",
        "retired": False,
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    return {"report": report, "skill": skill}


def _derive_skill_name(
    sample: Dict[str, Any],
    eids: List[str],
    schema_index: GameSchemaIndex,
) -> str:
    """Build a short human-readable skill name from the question + cited entities.

    No LLM call: we just pull a noun-phrase from the question and tag
    it with the modality. The name is **not** unique by itself --
    several VR/video samples share question stems verbatim
    ("answer/kind_of_relationship_plays_a_dominant_role_in_this_video"
    appears 8x in siv_bench, "answer/of_the_following_values_..."
    appears 7x in tir_bench). A short hash of the task_id is appended
    when this short name turns out to be ambiguous; uniqueness in the
    bank is still enforced via ``skill_id`` (which IS the task_id).

    Audit-2026-05-01 found 16/41 (39%) tir_bench records and 7/60
    (12%) siv_bench records collided on bare name. Hash suffix
    eliminates the collision while keeping names readable.
    """
    q = (sample.get("question") or "").strip()
    if not q:
        return f"answer/q_{sample.get('sample_id','unknown')}"
    head = q.split("?")[0].split(".")[0].strip()
    head = re.sub(r"^(what|who|where|when|why|how|is|are|do|does|did|can|will|which)\s+",
                  "", head, flags=re.I)
    head = head[:60].rstrip()
    if not head:
        return f"answer/q_{sample.get('sample_id','unknown')}"
    base = f"answer/{head.lower().replace(' ', '_')}"
    # Disambiguating suffix from a stable 5-char hash of the task_id;
    # consumers that want the bare name can split on '#'.
    task_id = str(sample.get("task_id") or sample.get("sample_id") or "")
    if task_id:
        import hashlib
        suffix = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:5]
        return f"{base}#{suffix}"
    return base


def _derive_description(sample: Dict[str, Any], steps: List[str]) -> str:
    return (sample.get("question") or "").strip()[:300]


def _resolve_cluster_key(sample: Dict[str, Any], spec: CorpusSpec) -> Optional[str]:
    """Pull the archetype-cluster field per the corpus spec.

    Supports dotted paths like ``raw_sample.question_type``.
    """
    field = spec.archetype_cluster_field
    if not field:
        return None
    cur: Any = sample
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    if isinstance(cur, str):
        return cur
    return str(cur)


# ── Corpus-level driver ───────────────────────────────────────────────

def lift_corpus(
    spec: CorpusSpec,
    *,
    output_dir: Path,
    max_samples: Optional[int] = None,
    include_incorrect: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Walk one VR/video corpus and emit per-sample lifted records.

    Writes ``per_sample/skill_bank.jsonl`` under *output_dir* and a
    parallel ``extraction_summary.json``.

    Returns the summary dict so :mod:`runner` can roll up stats across
    corpora.
    """
    input_root = spec.resolve_input_root()
    if not input_root.exists():
        raise FileNotFoundError(f"input_root for {spec.name!r} not found: {input_root}")

    sample_paths = sorted(input_root.glob(spec.sample_glob))
    if max_samples is not None:
        sample_paths = sample_paths[:max_samples]

    out_dir = output_dir / spec.name / "per_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "skill_bank.jsonl"

    n_samples = 0
    n_lifted = 0
    n_filtered_incorrect = 0
    n_filtered_no_schema = 0
    n_disambiguated = 0
    fallback_total = 0
    hops_total = 0
    seen_skill_ids: Dict[str, int] = {}

    with bank_path.open("w") as f:
        for sf in sample_paths:
            n_samples += 1
            try:
                sample = json.loads(sf.read_text())
            except Exception as exc:
                if verbose:
                    print(f"  [skip] {sf.name}: parse error: {exc}")
                continue
            if not sample.get("schema") or not sample.get("answer_reasoning"):
                n_filtered_no_schema += 1
                continue
            if not include_incorrect and not bool(sample.get("correct")):
                n_filtered_incorrect += 1
                continue
            record = lift_one_sample(sample, spec=spec, include_incorrect=include_incorrect)
            if record is None:
                continue
            # Disambiguate skill_id when cold-start data has rerun
            # records with identical task_id (audit-2026-05-01: 1 dup
            # each in tir_bench / visual_toolbench). Suffix with the
            # collision count so the bank stays uniquely keyable.
            base_id = record["skill"]["skill_id"]
            seen = seen_skill_ids.get(base_id, 0)
            if seen:
                disambig = f"{base_id}#run{seen}"
                record["skill"]["skill_id"] = disambig
                record["report"]["skill_id"] = disambig
                # Keep the original task_id discoverable for joins.
                record["skill"]["provenance"]["base_task_id"] = base_id
                n_disambiguated += 1
            seen_skill_ids[base_id] = seen + 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_lifted += 1
            stats = record["report"]["lift_stats"]
            fallback_total += stats["n_fallback_exec"]
            hops_total += stats["n_hops"]

    fallback_rate = fallback_total / hops_total if hops_total else 0.0

    summary = {
        "corpus": spec.name,
        "lift_kind": "single_shot",
        "modality": spec.modality,
        "input_root": str(input_root),
        "output_root": str(out_dir),
        "n_samples_seen": n_samples,
        "n_samples_lifted": n_lifted,
        "n_filtered_incorrect": n_filtered_incorrect,
        "n_filtered_no_schema": n_filtered_no_schema,
        "n_disambiguated": n_disambiguated,
        "n_hops_total": hops_total,
        "n_fallback_exec": fallback_total,
        "fallback_rate": fallback_rate,
        "bank_path": str(bank_path),
        "timestamp": datetime.utcnow().isoformat(),
    }

    summary_path = output_dir / spec.name / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


__all__ = [
    "build_schema_index_from_sample",
    "reasoning_to_prose_steps",
    "lift_one_sample",
    "lift_corpus",
]
