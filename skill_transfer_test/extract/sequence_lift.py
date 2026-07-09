"""Sequence-segment lift driver — browsergym + osworld.

These corpora have multi-step rollouts with an ``experiences[]``
spine, byte-identical in shape to env_wrappers / gym_v (same 20
keys per step). The canonical pipeline routes them through
:class:`skill_agents.pipeline.SkillBankAgent` (segment + effects-
contract + cluster + materialize + protocol-lift), which makes
many GPT-5.4 calls.

This module exposes **two modes**:

- :func:`lift_corpus_with_agent` — the canonical path, drives
  ``SkillBankAgent`` end-to-end and lifts each materialised
  cluster's prose protocol via
  :func:`labeling._protocol_lift.lift_protocol_to_typed_hops`.
  Requires ``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``.

- :func:`lift_corpus_per_episode` — the LLM-free smoke path. Treats
  each episode as a single skill: concatenates ``experiences[i].
  intentions`` (per-step prose already present in cold-start
  output) into a prose protocol, mines effect predicates from
  ``experiences[i].subgoal`` / ``tasks``, and routes through the
  same :func:`lift_protocol_to_typed_hops`. Useful for validating
  the data pipeline before API budget is committed, and for
  measuring lift quality on the verb taxonomy without the LLM-
  segmenter as a confound.

Both modes write per-episode (or per-sub-episode) ``{report, skill}``
records to ``<output_dir>/<corpus>/<game>/skill_bank.jsonl``,
matching the
``labeling/skill_bank_out/run_<ts>/<corpus>/<game>/skill_bank.jsonl``
layout the env_wrappers gold-standard pipeline produces.
"""

from __future__ import annotations

import json
import re
from collections import Counter
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
from .single_shot_lift import (
    _bind_entity_refs_in_payloads,
    _rewrite_sentence_for_lift,
    _split_sentences,
    restore_original_notes,
)


# ── Cross-corpus entity-declaration parser ─────────────────────────────

#: Maps the OSWorld AT-SPI ``role=`` attribute to one of the canonical
#: ontologies the lift's slot-binder understands. Mirrors the role
#: hierarchy the OSWorld heuristic schema wrapper emits.
_OSWORLD_ROLE_TO_ONTOLOGY: Dict[str, str] = {
    # Interactive controls -- the SELECT / SWAP / DROP slot-binder
    # expects ``selectable_entity``.
    "push-button": "selectable_entity",
    "toggle-button": "selectable_entity",
    "menu-item": "selectable_entity",
    "menu": "container_entity",
    "menu-bar": "container_entity",
    "check-box": "selectable_entity",
    "radio-button": "selectable_entity",
    "combo-box": "selectable_entity",
    "list-item": "selectable_entity",
    "tree-item": "selectable_entity",
    "text-entry": "selectable_entity",
    "text": "textual_anchor",
    "label": "textual_anchor",
    "tab": "selectable_entity",
    "tab-list": "container_entity",
    # Containers / regions
    "frame": "navigable_region",
    "window": "navigable_region",
    "panel": "navigable_region",
    "tool-bar": "container_entity",
    "scroll-bar": "selectable_entity",
    "scroll-pane": "navigable_region",
    "application": "container_entity",
    "split-pane": "container_entity",
    "page": "navigable_region",
    "page-tab": "selectable_entity",
    "page-tab-list": "container_entity",
}

#: Maps the BrowserGym ``type=element`` declaration's role-prefix in
#: the label string (``link 'Foo'``, ``button 'Bar'``, ``combobox
#: 'Hae'``, ...) to canonical ontology.
_BROWSER_LABEL_PREFIX_TO_ONTOLOGY: Dict[str, str] = {
    "link": "selectable_entity",
    "button": "selectable_entity",
    "checkbox": "selectable_entity",
    "radio": "selectable_entity",
    "menuitem": "selectable_entity",
    "combobox": "selectable_entity",
    "textbox": "selectable_entity",
    "searchbox": "selectable_entity",
    "spinbutton": "selectable_entity",
    "tab": "selectable_entity",
    "image": "tracked_entity",
    "img": "tracked_entity",
    "heading": "textual_anchor",
    "static_text": "textual_anchor",
    "text": "textual_anchor",
    "paragraph": "textual_anchor",
    "list": "container_entity",
    "listitem": "selectable_entity",
    "form": "container_entity",
    "navigation": "container_entity",
    "main": "navigable_region",
    "region": "navigable_region",
    "dialog": "navigable_region",
    "menu": "container_entity",
    "menubar": "container_entity",
    "toolbar": "container_entity",
}

_RX_ENTITY_GENERIC = re.compile(
    r"^(e\d+)\[type=([\w\-]+)(?:,\s*([^\]]+))?\]\s*$",
    re.M,
)


def _parse_entity_decl_attrs(attrs: str) -> Dict[str, str]:
    """Tolerant ``key=value, key=value`` attribute parser.

    Tolerates values containing spaces/quotes (e.g. ``label=link 'Foo'``)
    and stops at the next ``key=`` token, not the next comma. Strips
    leading + trailing whitespace AND a *single* surrounding pair of
    matching quotes per side -- previously the parser only stripped a
    single layer at each end, so ``label=link 'Foo'`` produced
    ``link 'Foo`` (trailing apostrophe lost; trailing whitespace
    inside the value retained), which leaked into 84.6% of browsergym
    payload bindings as ``'navigation '`` etc. (audit-2026-05-01).
    """
    out: Dict[str, str] = {}
    parts = re.split(r",\s*(?=[a-z_]+=)", attrs)
    for kv in parts:
        kv = kv.strip()
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        v = v.strip()
        # Strip up to one matched pair of quotes (single or double),
        # then re-strip whitespace so embedded leading/trailing
        # whitespace inside the quotes is normalised.
        if len(v) >= 2 and v[0] in ("'", '"') and v[-1] == v[0]:
            v = v[1:-1].strip()
        else:
            # Asymmetric trailing apostrophe (BrowserGym AXTree
            # serialiser emits ``label=link 'Foo'`` -- single quote on
            # both sides but our split swallowed the closing quote
            # along with following key=value pairs).
            v = v.strip("'\"").strip()
        out[k.strip()] = v
    return out


def _ontology_for_browser_label(label: str) -> str:
    """Pick an ontology for a BrowserGym entity label.

    BrowserGym labels are formatted as ``<role> '<text>'`` (e.g.
    ``link 'Tietoja'``, ``button 'Jaa'``); the leading token is the
    AXTree role. Falls back to ``selectable_entity`` when no prefix
    matches (most browser entities are interactable).
    """
    head = (label or "").split()[0].lower() if label else ""
    head = re.sub(r"[^a-z_]", "_", head)
    return _BROWSER_LABEL_PREFIX_TO_ONTOLOGY.get(head, "selectable_entity")


def parse_schema_entities_cross_corpus(
    schema_text: str,
    *,
    corpus_name: str,
) -> Iterable[Tuple[str, str, str]]:
    """Yield ``(eid, label, ontology)`` triples from a canonical schema
    block, handling the four entity-declaration formats the cold-start
    pipelines emit:

    - env_wrappers / gym_v / VR / video: ``e1[type=..., label=..., ontology=...]``
      (canonical -- ``ontology=`` present)
    - osworld: ``e1[type=..., label=..., role=...]`` (no ``ontology=``,
      derive from ``role=``)
    - browsergym: ``e1[type=element, label=link 'Foo', bid=...]``
      (no ``ontology=``, derive from label prefix)
    """
    if not schema_text:
        return
    for m in _RX_ENTITY_GENERIC.finditer(schema_text):
        eid = m.group(1)
        attrs = _parse_entity_decl_attrs(m.group(3) or "")
        label = attrs.get("label", "")
        if not label:
            continue
        ontology = attrs.get("ontology")
        if not ontology:
            role = attrs.get("role")
            if role and corpus_name == "osworld":
                ontology = _OSWORLD_ROLE_TO_ONTOLOGY.get(role, "selectable_entity")
            elif corpus_name == "browsergym":
                ontology = _ontology_for_browser_label(label)
            else:
                ontology = "tracked_entity"
        yield eid, label, ontology


# ── Per-step prose extraction ─────────────────────────────────────────

def _extract_prose_per_step(
    experiences: List[Dict[str, Any]],
    *,
    max_steps: int = 8,
) -> List[Tuple[str, str]]:
    """Pull ``(intent_prose, action_prose)`` pairs from each kept step.

    Each kept step becomes TWO hops in the lifted protocol: one
    GATHER/REASON hop for the agent's intent and one COMMIT hop for
    the action it actually emitted. Without the second hop, OSWorld
    protocols lose 99% of the action information (subgoal text is
    abstract -- "the desktop MP4 icon is the target..." -- with no
    mention of the ``pyautogui.click(x,y)`` actually emitted; only
    3/234 osworld hops referenced the actor verb in audit-2026-05-01).
    Browser episodes preserve action info ~91% of the time in the
    intent text alone but still benefit from explicit COMMIT hops for
    transferability.

    ``intent_prose`` prefers ``subgoal`` then ``intentions``, falling
    back to a templated line. ``action_prose`` is templated from
    ``experiences[i].action`` so the lift's classifier sees an
    ``execute`` head; when the step has no action we emit empty
    string and the COMMIT hop is dropped.

    Caps at ``max_steps`` (default 8) -- without an LLM-driven
    segmenter the per-episode lift would otherwise produce 30-60 hop
    mega-skills; 8 keeps records at 16-18 hops total (8 intent + 8
    action + ground/verify/commit scaffolding).
    """
    out: List[Tuple[str, str]] = []
    last_kept_intent: Optional[str] = None
    for step in experiences:
        if len(out) >= max_steps:
            break
        intent = (
            (step.get("subgoal") or "").strip()
            or (step.get("intentions") or "").strip()
        )
        action = (step.get("action") or "").strip()
        if not intent:
            if not action:
                continue
            intent = f"Execute the action {action}."
        if intent == last_kept_intent:
            continue
        action_prose = (
            f"Execute the action {action}." if action else ""
        )
        out.append((intent, action_prose))
        last_kept_intent = intent
    return out


def _build_schema_index_from_episode(
    episode: Dict[str, Any],
    *,
    benchmark: str,
    corpus_name: str = "",
) -> GameSchemaIndex:
    """Build a schema index by walking the episode's ``experiences[i].
    metadata.schema_canonical`` blocks.

    Mirrors :func:`labeling._protocol_lift.build_schema_index_for_game`
    in spirit, but uses :func:`parse_schema_entities_cross_corpus` so
    osworld (``role=``) and browsergym (``type=element`` only) entity
    declarations -- which the canonical
    :func:`_parse_schema_block_entities` drops because they lack an
    ``ontology=`` attribute -- contribute vocabulary just as cleanly
    as VR/video declarations.

    The resulting ``entity_labels`` includes BOTH the human-readable
    ``label=`` strings AND the canonical ``e\\d+`` IDs (since prose
    can reference either) -- both map to the same ontology.
    """
    label_to_ontology: Dict[str, str] = {}
    # Walk a few steps; later steps can reveal entities that didn't
    # exist at step 0 (e.g. dialogs that opened mid-episode).
    for step in (episode.get("experiences") or [])[:8]:
        md = step.get("metadata") or {}
        sc = md.get("schema_canonical") or md.get("schema") or ""
        for eid, label, ontology in parse_schema_entities_cross_corpus(
            sc, corpus_name=corpus_name or benchmark,
        ):
            label_to_ontology.setdefault(label, ontology)
            # Also register the canonical e_N id with the same ontology
            # so prose references like "click e6" bind cleanly.
            label_to_ontology.setdefault(eid, ontology)
    return GameSchemaIndex(
        game=benchmark,
        entity_labels=frozenset(label_to_ontology.keys()),
        label_to_ontology=dict(label_to_ontology),
        affordances_by_ontology={},
    )


def _action_verb_head(action: str) -> str:
    """Extract a useful verb head from an emitted action string.

    Cold-start sequence data uses a mix of action conventions:

    - OSWorld desktop:   ``pyautogui.click(464, 47)``
                         ``pyautogui.hotkey('ctrl','t')``
                         ``DONE`` / ``FAIL`` / ``WAIT``
    - BrowserGym:        ``click("493")`` / ``go_back()`` / ``scroll(-200)``

    The previous head extractor used ``action.split("(")[0].split(".")[0]``,
    which collapsed every ``pyautogui.X(...)`` call to ``pyautogui`` --
    losing the actually useful verb (``click``, ``hotkey``, ``typewrite``,
    ``scroll``, ``moveTo``, ``dragTo``, ``press``). For a 30-record
    OSWorld smoke this turned ``actor_used_action`` from a per-skill
    fingerprint into a constant. Audit-2026-05-01 confirmed verbs were
    ``{pyautogui: 30, DONE: 11, WAIT: 8, FAIL: 4}``.

    The fix prefers the *last* dotted segment before the call paren --
    so ``pyautogui.click(...)`` heads as ``click`` and bare actions
    like ``DONE`` head as ``DONE`` unchanged. Sentinel verbs (DONE /
    FAIL / WAIT) are normalised to upper-case so transfer matching
    treats them as a fixed vocabulary; named pyautogui sub-calls keep
    their original case so callers can split on ``str.islower()``.
    """
    a = (action or "").strip()
    if not a:
        return ""
    pre_paren = a.split("(", 1)[0].strip()
    last_seg = pre_paren.split(".")[-1].strip()
    if not last_seg:
        return ""
    if last_seg.isupper() and last_seg.isalpha():
        return last_seg
    if last_seg.upper() in {"DONE", "FAIL", "WAIT"}:
        return last_seg.upper()
    return last_seg


def mine_sequence_episode_effects(
    *,
    episode: Dict[str, Any],
    success: Optional[bool],
    schema_index: GameSchemaIndex,
    spec_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build ``(effects_add, effects_del)`` for a sequence-lifted episode.

    Without LLM-driven success-criteria mining, the canonical
    :func:`labeling._protocol_lift.mine_effects` produces empty
    contracts. This v0 sequence miner emits a small set of episode-
    level predicates the harness's transfer machinery can match on:

    - ``task_status{value=success|failure|incomplete}`` (always)
    - ``last_action{verb=<head>}`` mined from the agent's final emitted
      action (e.g. ``DONE``, ``FAIL``, ``pyautogui``).
    - ``actor_used_action{verb=<head>}`` -- one per distinct action
      head observed across the episode (``click``, ``scroll``,
      ``go_back``, ...).
    - ``visited_entity{label=<L>}`` -- entity labels referenced by
      multiple ``experiences[i].subgoal`` strings (heuristic: appears
      in 2+ steps after lower-casing). Capped at 5 to keep the
      contract compact.
    """
    effects_add: List[Dict[str, Any]] = []
    effects_del: List[Dict[str, Any]] = []

    # 1) task_status -- the headline outcome.
    status = "success" if success is True else "failure" if success is False else "incomplete"
    bucket = effects_add if success is True else effects_del if success is False else effects_add
    bucket.append({
        "type": "task_status",
        "args": {"value": status, "corpus": spec_name},
        "from_phrase": f"episode resolved with status={status}",
    })

    experiences = episode.get("experiences") or []

    # 2) last_action -- final emitted action head.
    if experiences:
        last_action = (experiences[-1].get("action") or "").strip()
        if last_action:
            head = _action_verb_head(last_action) or "UNKNOWN"
            (effects_add if success else effects_del).append({
                "type": "last_action",
                "args": {"verb": head},
                "from_phrase": f"episode terminated with action head: {head}",
            })

    # 3) actor_used_action -- distinct action verbs observed.
    used: Counter = Counter()
    for step in experiences:
        a = (step.get("action") or "").strip()
        if not a:
            continue
        head = _action_verb_head(a)
        if head:
            used[head] += 1
    for verb, _count in used.most_common(8):
        effects_add.append({
            "type": "actor_used_action",
            "args": {"verb": verb},
            "from_phrase": f"actor emitted action verb: {verb}",
        })

    # 4) visited_entity -- entity labels referenced by 2+ subgoal strings.
    label_hits: Counter = Counter()
    label_set = {l.lower() for l in schema_index.entity_labels if not l.startswith("e") or len(l) > 4}
    for step in experiences:
        prose = ((step.get("subgoal") or "") + " " + (step.get("intentions") or "")).lower()
        if not prose:
            continue
        for lbl in label_set:
            if len(lbl) >= 4 and lbl in prose:
                label_hits[lbl] += 1
    for lbl, hits in label_hits.most_common(5):
        if hits >= 2:
            effects_add.append({
                "type": "visited_entity",
                "args": {"label": lbl, "n_subgoal_refs": hits},
                "from_phrase": f"entity {lbl!r} referenced in {hits} subgoals",
            })

    return effects_add, effects_del


def _episode_outcome_success(
    episode: Dict[str, Any],
    *,
    corpus_name: str,
) -> Tuple[Optional[bool], Optional[float], str]:
    """Best-effort success extraction from an episode's outcome / summary.

    Returns ``(succeeded, reward_total, signal_source)`` where
    ``succeeded`` may be ``None`` (incomplete / unknown), ``True``
    (passed), or ``False`` (failed). ``signal_source`` documents
    which heuristic produced the verdict so we can trace mistakes.

    Per-corpus policy (audit empirically established):

    - ``osworld``: ``outcome=True`` is **always** present in cold-start
      data and is therefore meaningless; the trustworthy signal is
      ``experiences[-1].action`` -- "DONE" -> success, "FAIL" -> fail,
      anything else -> incomplete (likely hit max_steps).
    - ``browsergym``: ``outcome`` is a real bool (False/True ~= 60/40
      across the corpus). Use it directly. Reward in
      ``experiences[-1].reward`` is a secondary signal in [0,1].
    - other (env_wrappers / gym_v shape): legacy ``outcome.success``
      / ``outcome.passed`` lookup.
    """
    rewards = [
        e.get("reward") for e in (episode.get("experiences") or [])
        if isinstance(e.get("reward"), (int, float))
    ]
    total = float(sum(rewards)) if rewards else None

    if corpus_name == "osworld":
        last = ((episode.get("experiences") or [{}])[-1].get("action") or "").strip()
        head = last.split("(")[0].split(".")[0].strip().upper()
        if head == "DONE":
            return True, total, "osworld:last_action=DONE"
        if head == "FAIL":
            return False, total, "osworld:last_action=FAIL"
        return None, total, "osworld:last_action=incomplete"

    outcome = episode.get("outcome")
    if isinstance(outcome, bool):
        if corpus_name == "browsergym":
            return outcome, total, "browsergym:outcome_bool"
        return outcome, total, "outcome_bool"
    if isinstance(outcome, (int, float)):
        return bool(outcome), total, "outcome_numeric"
    if isinstance(outcome, dict):
        for k in ("success", "passed", "task_success", "is_success"):
            if k in outcome:
                return bool(outcome[k]), total, f"outcome_dict.{k}"
    return None, total, "no_signal"


# ── Per-episode (LLM-free) lift ───────────────────────────────────────

def lift_one_episode(
    episode: Dict[str, Any],
    *,
    spec: CorpusSpec,
    episode_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Lift one whole episode into a single ``{report, skill}`` record.

    Treats every step's ``subgoal`` / ``intentions`` as a prose
    protocol step, concatenates them, and routes through
    :func:`lift_protocol_to_typed_hops`. The lifted skill represents
    "the way this episode achieved its goal" rather than the
    fine-grained sub-skills SkillBankAgent would tease out — but the
    output shape is identical, so the harness's transfer machinery
    sees no difference.
    """
    experiences = episode.get("experiences") or []
    if not experiences:
        return None

    prose_steps_raw = _extract_prose_per_step(experiences)
    if not prose_steps_raw:
        return None

    # Each kept step becomes (intent_hop, action_hop) -- the COMMIT
    # action_hop captures what the agent emitted, not just what it
    # thought. Without this, OSWorld protocols carry zero action
    # information (audit-2026-05-01).
    pairs: List[Tuple[str, str]] = []
    for intent, action_prose in prose_steps_raw:
        pairs.append((_rewrite_sentence_for_lift(intent), intent))
        if action_prose:
            pairs.append((_rewrite_sentence_for_lift(action_prose), action_prose))
    sentences = [rewritten for rewritten, _orig in pairs]

    success, reward_total, success_source = _episode_outcome_success(episode, corpus_name=spec.name)

    # Goal / task identification
    task_id = episode.get("episode_id") or episode.get("game_name") or "unknown"
    goal_field = experiences[0].get("goal") or ""
    if not goal_field:
        goal_field = (episode.get("task") or {}).get("instruction") or ""
    if not goal_field:
        goal_field = task_id

    schema_index = _build_schema_index_from_episode(
        episode,
        benchmark=spec.extra.get("benchmark", spec.name),
        corpus_name=spec.name,
    )

    if success is True:
        success_criteria = [f"task succeeded ({task_id})"]
    elif success is False:
        success_criteria = [f"task failed ({task_id})"]
    else:
        success_criteria = [f"task incomplete ({task_id})"]
    abort_criteria: List[str] = []

    pseudo_skill = {
        "evidence_role": "COMMIT",
        "protocol": {
            "steps": sentences,
            "success_criteria": success_criteria,
            "abort_criteria": abort_criteria,
        },
    }

    stats = LiftStats()
    typed, contract_add, contract_del = lift_protocol_to_typed_hops(
        pseudo_skill, schema_index=schema_index, stats=stats,
    )
    if typed is None:
        return None

    # Restore originals + post-bind e\d+ refs (and labels as
    # fallback) for "any"-typed slots.
    restore_original_notes(typed, pairs)
    _bind_entity_refs_in_payloads(typed, schema_index=schema_index)

    # v0 sequence-episode effect miner -- canonical mine_effects
    # produces empty contracts for QA / browser / desktop criteria.
    seq_eff_add, seq_eff_del = mine_sequence_episode_effects(
        episode=episode,
        success=success,
        schema_index=schema_index,
        spec_name=spec.name,
    )
    contract_add = (contract_add or []) + seq_eff_add
    contract_del = (contract_del or []) + seq_eff_del

    # Verb diversity — count unique action verbs the actor actually emitted.
    actor_verbs: Dict[str, int] = {}
    for step in experiences:
        a = step.get("action") or ""
        head_match = re.match(r"^\s*\(?([\w]+)", a)
        if head_match:
            actor_verbs[head_match.group(1)] = actor_verbs.get(head_match.group(1), 0) + 1

    if success is True:
        pass_rate = 1.0
    elif success is False:
        pass_rate = 0.0
    else:
        pass_rate = None

    report = {
        "skill_id": task_id,
        "n_instances": 1,
        "n_steps": len(experiences),
        "overall_pass_rate": pass_rate,
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
        "actor_verb_distribution": dict(actor_verbs),
        "reward_total": reward_total,
        "judge_correct": success,           # may be None for incomplete osworld episodes
        "success_source": success_source,   # which heuristic decided judge_correct
    }

    skill = {
        "skill_id": task_id,
        "name": _derive_episode_skill_name(goal_field, task_id),
        "strategic_description": goal_field[:300],
        "applicable_domains": [spec.domain],
        "feasible_tasks": [task_id],
        "feasible_domains": [spec.domain],
        "verified_domains": [spec.domain] if success is True else [],
        "verified_tasks": [task_id] if success is True else [],
        "evidence_role": "COMMIT",
        "execution_hint": None,
        "expected_tag_pattern": None,
        "protocol": typed,
        "protocol_history": [],
        "protocol_raw": {
            "steps": sentences,
            "success_criteria": success_criteria,
            "abort_criteria": abort_criteria,
        },
        "contract": {
            "effects_add": contract_add,
            "effects_del": contract_del,
        },
        "provenance": {
            "corpus": spec.name,
            "benchmark": spec.extra.get("benchmark", spec.name),
            "modality": spec.modality,
            "bank_kind": "episode",
            "source_episode": str(episode_path) if episode_path else None,
            "n_steps": len(experiences),
            "lift_mode": "per_episode_no_llm",
        },
        "tags": [spec.name, spec.modality, "sequence", "per_episode"],
        "n_instances": 1,
        "sub_episodes": [],
        "source_type": "sequence_per_episode",
        "status": "draft",
        "retired": False,
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    return {"report": report, "skill": skill}


def _derive_episode_skill_name(goal: str, task_id: str) -> str:
    g = (goal or "").strip()
    if not g:
        return f"episode/{task_id}"
    # Strip prefixes like "OSWorld task (vlc):" or "Solve the BrowserGym task ...:"
    g = re.sub(r"^(OSWorld\s+task\s+\([^)]+\):|Solve\s+the\s+BrowserGym\s+task\s+\S+\.)\s*", "", g, flags=re.I).strip()
    g = g.split("?")[0].split(".")[0].strip()
    g = g[:60].rstrip()
    return f"episode/{g.lower().replace(' ', '_')}" if g else f"episode/{task_id}"


# ── Corpus-level driver (LLM-free path) ───────────────────────────────

def lift_corpus_per_episode(
    spec: CorpusSpec,
    *,
    output_dir: Path,
    max_episodes: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Walk one sequence corpus and emit per-episode records (LLM-free)."""
    input_root = spec.resolve_input_root()
    if not input_root.exists():
        raise FileNotFoundError(f"input_root for {spec.name!r} not found: {input_root}")

    ep_paths = sorted(input_root.glob(spec.sample_glob))
    # Skip episode_buffer.json files
    ep_paths = [p for p in ep_paths if "episode_buffer" not in p.name]
    if max_episodes is not None:
        ep_paths = ep_paths[:max_episodes]

    out_dir = output_dir / spec.name / "per_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "skill_bank.jsonl"

    n_seen = 0
    n_lifted = 0
    n_skipped_empty = 0
    n_disambiguated = 0
    fallback_total = 0
    hops_total = 0
    n_succeeded = 0
    n_failed = 0
    n_incomplete = 0
    seen_skill_ids: Dict[str, int] = {}

    with bank_path.open("w") as f:
        for ep_path in ep_paths:
            n_seen += 1
            try:
                episode = json.loads(ep_path.read_text())
            except Exception as exc:
                if verbose:
                    print(f"  [skip] {ep_path.name}: {exc}")
                continue
            record = lift_one_episode(episode, spec=spec, episode_path=ep_path)
            if record is None:
                n_skipped_empty += 1
                continue
            # Disambiguate skill_id when cold-start data has rerun
            # episodes with identical task/episode id (mirrors the
            # single-shot disambiguator).
            base_id = record["skill"]["skill_id"]
            seen = seen_skill_ids.get(base_id, 0)
            if seen:
                disambig = f"{base_id}#run{seen}"
                record["skill"]["skill_id"] = disambig
                record["report"]["skill_id"] = disambig
                record["skill"]["provenance"]["base_task_id"] = base_id
                n_disambiguated += 1
            seen_skill_ids[base_id] = seen + 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_lifted += 1
            stats = record["report"]["lift_stats"]
            fallback_total += stats["n_fallback_exec"]
            hops_total += stats["n_hops"]
            jc = record["report"]["judge_correct"]
            if jc is True:
                n_succeeded += 1
            elif jc is False:
                n_failed += 1
            else:
                n_incomplete += 1

    fallback_rate = fallback_total / hops_total if hops_total else 0.0
    summary = {
        "corpus": spec.name,
        "lift_kind": "sequence_per_episode_no_llm",
        "modality": spec.modality,
        "input_root": str(input_root),
        "output_root": str(out_dir),
        "n_episodes_seen": n_seen,
        "n_episodes_lifted": n_lifted,
        "n_skipped_empty": n_skipped_empty,
        "n_disambiguated": n_disambiguated,
        "n_succeeded": n_succeeded,
        "n_failed": n_failed,
        "n_incomplete": n_incomplete,
        "n_hops_total": hops_total,
        "n_fallback_exec": fallback_total,
        "fallback_rate": fallback_rate,
        "bank_path": str(bank_path),
        "timestamp": datetime.utcnow().isoformat(),
    }

    summary_path = output_dir / spec.name / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ── Corpus-level driver (canonical, requires API key) ─────────────────

def lift_corpus_with_agent(
    spec: CorpusSpec,
    *,
    output_dir: Path,
    max_episodes: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Canonical lift via `skill_agents.pipeline.SkillBankAgent`.

    Stub: full implementation drives the SkillBankAgent end-to-end and
    is functionally equivalent to
    :func:`labeling.extract_skillbank_gpt54.run_one_game` minus the
    env_wrappers hard-coding. Requires
    ``OPENROUTER_API_KEY`` / ``OPENAI_API_KEY``.

    Raises ``NotImplementedError`` until the full agent integration
    lands; callers should fall back to
    :func:`lift_corpus_per_episode` for LLM-free smoke runs.
    """
    raise NotImplementedError(
        "lift_corpus_with_agent is not yet wired to SkillBankAgent — "
        "use lift_corpus_per_episode for LLM-free runs"
    )


__all__ = [
    "lift_one_episode",
    "lift_corpus_per_episode",
    "lift_corpus_with_agent",
]
