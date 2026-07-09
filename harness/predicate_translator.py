"""Per-domain runtime predicate translator.

Phase-5 §11.5.0 / Stage 6 §12.3 of the cross-domain transfer measurement
plan: when a *game* skill is transferred to a target domain whose
success_fn evaluates a different vocabulary (image-QA's
``answer_emitted`` / ``answer_matches_gold`` vs game's
``cumulative_reward_increased`` / ``entity_value_increased``), the
contract's ``effects_add`` / ``effects_del`` predicate names need to be
remapped to predicates the target's success_fn can actually evaluate
against its schema.

Without this layer, even a real (Tier 1 + Tier 2) executor will reject
every game->VR contract because the contract advertises predicates the
target's
:func:`~harness.qa_success.make_qa_success_fn` /
:func:`~harness.video_qa_success.make_video_qa_success_fn` /
:func:`~harness.osworld_success.make_osworld_success_fn` /
:func:`~harness.browser_success.make_browser_success_fn`
do not know how to ground -- so cells admit at 0% via static-vocab
miss before any executor runs (see
``cross_domain_results/_phase0/<run_id>/upper_bounds.csv``: rows where
``upper_bound_admit_rate=0.0`` are exactly the ones this layer
unblocks).

Design
------
The translator is a pure-data + thin-glue module structured as:

* :data:`PREDICATE_TRANSLATIONS` -- a ``dict[(source, target), dict[str, list[str]]]``
  table. ``source`` and ``target`` are
  :data:`common.enums.DOMAINS` strings. The inner mapping is
  ``source_predicate -> [target_predicates]``: an empty list means
  "drop this predicate; not meaningful in the target vocabulary"; a
  one-element list is a 1:1 rename; a multi-element list fans out one
  source predicate into multiple target ones (e.g.
  ``cumulative_reward_increased`` -> ``[answer_emitted, answer_matches_gold]``).

* :func:`translate_predicates` -- pure ``list[str] -> list[str]``
  rewrite of one effects list given a ``(source, target)`` pair.
  Predicates not in the table pass through unchanged (so the harness
  retains its default behaviour for cells where no translation is
  registered).

* :func:`translate_skill_contract` -- returns a *copy* of a
  :class:`~data_structure.extensions.skill_record.SkillRecord` with
  its ``contract.effects_add`` / ``contract.effects_del`` translated.
  Never mutates the input. Tags the new record's ``notes`` so the
  trace surfaces that translation occurred.

* :func:`with_predicate_translation` -- success-fn factory wrapper.
  Wraps a ``make_*_success_fn`` factory so the success_fn it returns
  translates ``skill`` before evaluating it. Source domain is read at
  evaluation time from ``skill.source_domains[0]``, so the same
  wrapped factory handles cross-modal AND diagonal (gymv->gymv) calls
  -- diagonal cells get the identity translation by default and
  remain mechanism-equivalent to the un-wrapped path.

Vocabulary provenance
---------------------
The target predicate vocabularies in the table below are aligned with
:data:`skill_transfer_test.extract.audits._target_vocabularies.TARGET_PREDICATE_VOCAB`.
When a target predicate appears in the right-hand list it is
guaranteed to be in that target's success_fn's recognised set --
otherwise the translation would just shift the static-vocab miss from
the source predicate to the target one without unblocking the cell.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger("harness.predicate_translator")

__all__ = [
    "PREDICATE_TRANSLATIONS",
    "translate_predicates",
    "translate_skill_contract",
    "with_predicate_translation",
]


# ---------------------------------------------------------------------------
# Translation table
# ---------------------------------------------------------------------------

# Non-trivial cells. The diagonal cells (e.g. (gymv, gymv)) are absent
# from the table by design -- :func:`translate_predicates` falls back to
# identity for any (source, target) pair not registered here, which
# means the harness behaves exactly like the un-wrapped path on
# diagonal evaluations.
#
# Empty mappings mean "no remapping registered for this cell; identity
# applies". Empty *lists* in a registered mapping mean "drop this
# predicate entirely; the source predicate is not meaningful in the
# target vocabulary."
#
# Read each row as: when transferring a skill *from* `source` *to*
# `target`, rewrite each occurrence of `source_predicate` in
# ``contract.effects_{add,del}`` to the listed target predicates.

PREDICATE_TRANSLATIONS: Dict[Tuple[str, str], Dict[str, List[str]]] = {

    # ---- gymv -> visual_reasoning (image QA) ----
    # Image-VR has no notion of "play time" or "scalar value" persisting
    # across frames -- only "the answer was produced and matched gold."
    # So all reward / scoring / counting predicates collapse onto the
    # answer-grounded predicates the QA success_fn actually evaluates.
    ("gymv", "visual_reasoning"): {
        "cumulative_reward_increased": ["answer_emitted", "answer_matches_gold"],
        # Identity-passthrough mappings are listed explicitly so the
        # downstream auditor can prove the table was deliberate (and not
        # the result of a forgotten cell).
        "phase_transitioned":          ["phase_transitioned"],
        "entity_appeared":             ["entity_appeared", "entity_grounded"],
        "entity_value_increased":      ["entity_value_increased"],
        "entity_value_decreased":      ["entity_value_decreased"],
        # Predicates with no QA analogue: drop them so the contract gate
        # checks the QA-relevant subset only. The skill still admits IFF
        # the surviving predicates fire on the target schema.
        "entity_disappeared":          [],
        "entity_count_changed":        [],
        "attribute_changed":           [],
    },

    # ---- gymv -> video (video QA) ----
    # Video-VR has the entire visual_reasoning vocabulary plus the two
    # video-specific predicates (`temporal_ordering_correct`,
    # `frame_referent_grounded`). We can map slightly more aggressively
    # because video has a real time axis.
    ("gymv", "video"): {
        "cumulative_reward_increased": ["answer_emitted", "answer_matches_gold"],
        "phase_transitioned":          ["phase_transitioned"],
        "entity_appeared":             ["entity_appeared", "entity_grounded",
                                        "frame_referent_grounded"],
        "entity_value_increased":      ["entity_value_increased"],
        "entity_value_decreased":      ["entity_value_decreased"],
        # Video has time -- a sequence of "appeared then disappeared"
        # is exactly `temporal_ordering_correct`.
        "entity_disappeared":          ["temporal_ordering_correct"],
        # No analogue: drop.
        "entity_count_changed":        [],
        "attribute_changed":           [],
    },

    # ---- gymv -> osworld (desktop A11y) ----
    # OSWorld has most game predicates as same-name analogues (its
    # schema producer surfaces entity counts, attribute changes, etc.).
    # The translation is mostly identity; the interesting bits are
    # collapsing the game-specific reward signal into desktop's
    # `task_status` / `last_action`.
    ("gymv", "osworld"): {
        "cumulative_reward_increased": ["task_status"],
        "phase_transitioned":          ["phase_transitioned"],
        "entity_appeared":             ["entity_appeared", "visited_entity"],
        "entity_disappeared":          ["entity_disappeared"],
        "entity_count_changed":        ["entity_count_changed"],
        "attribute_changed":           ["attribute_changed"],
        # Desktop has no scalar "value" attribute (no health bar / score),
        # so value-changed predicates do not have an osworld analogue.
        "entity_value_increased":      [],
        "entity_value_decreased":      [],
    },

    # ---- gymv -> browser (webagent) ----
    # Same shape as osworld. Browser's TARGET_PREDICATE_VOCAB lacks
    # `entity_disappeared` (DOM nodes typically detach via attribute
    # changes rather than disappearance events), so we map to
    # `attribute_changed` instead.
    ("gymv", "browser"): {
        "cumulative_reward_increased": ["task_status"],
        "phase_transitioned":          ["phase_transitioned"],
        "entity_appeared":             ["entity_appeared", "visited_entity"],
        "entity_disappeared":          ["attribute_changed"],
        "entity_count_changed":        ["entity_count_changed"],
        "attribute_changed":           ["attribute_changed"],
        "entity_value_increased":      [],
        "entity_value_decreased":      [],
    },
}


# ---------------------------------------------------------------------------
# Pure-data translation
# ---------------------------------------------------------------------------

def translate_predicates(
    predicates: List[str],
    *,
    source: str,
    target: str,
) -> List[str]:
    """Translate one ``effects_*`` list from source to target vocabulary.

    Predicates not in the registered ``(source, target)`` table pass
    through unchanged. An empty target list means the predicate is
    dropped. Order is preserved per source predicate; duplicates from
    the fan-out (e.g. ``entity_appeared`` mapping to both
    ``entity_appeared`` and ``entity_grounded``) are deduplicated
    while preserving first-occurrence order.

    Identity is returned when ``source == target`` (diagonal cells).
    """
    if not predicates:
        return []
    if source == target:
        return list(predicates)
    table = PREDICATE_TRANSLATIONS.get((source, target))
    if table is None:
        # No translation registered -- identity. Mirrors the old
        # behaviour for cells nobody has audited yet, so adding a row
        # to this table is the *only* way to change cross-domain
        # admit rates downstream.
        return list(predicates)
    out: List[str] = []
    seen: set[str] = set()
    for p in predicates:
        translated = table.get(p)
        if translated is None:
            # Not in the per-cell rewrite table -- pass through. This
            # lets a partial translation table still ship without
            # breaking source predicates we have not yet audited.
            if p not in seen:
                seen.add(p)
                out.append(p)
            continue
        for tp in translated:
            if tp and tp not in seen:
                seen.add(tp)
                out.append(tp)
    return out


def translate_skill_contract(
    skill: Any,
    *,
    source: str,
    target: str,
) -> Any:
    """Return a deep-copy of ``skill`` with its contract translated.

    Never mutates ``skill``. When ``source == target`` or no
    ``(source, target)`` table is registered, the returned record is
    structurally identical to the input (deep-copied so the caller can
    still mutate it freely without aliasing the bank entry).

    Tags ``out.notes`` with a one-line audit trail so transfer traces
    surface that translation happened. Idempotent: a repeated call
    with the same ``(source, target)`` does not stack the note.
    """
    if skill is None:
        return None
    out = copy.deepcopy(skill)
    contract = getattr(out, "contract", None)
    if contract is None:
        return out
    eff_add = list(getattr(contract, "effects_add", []) or [])
    eff_del = list(getattr(contract, "effects_del", []) or [])
    new_add = translate_predicates(eff_add, source=source, target=target)
    new_del = translate_predicates(eff_del, source=source, target=target)
    contract.effects_add = new_add
    contract.effects_del = new_del

    if source != target and (eff_add != new_add or eff_del != new_del):
        marker = f"[predicate_translator: {source}->{target}]"
        existing = getattr(out, "notes", "") or ""
        if marker not in existing:
            out.notes = (existing + " " + marker).strip()
    return out


# ---------------------------------------------------------------------------
# success_fn factory wrapper
# ---------------------------------------------------------------------------

def _resolve_source_domain(skill: Any, default: str) -> str:
    """Pull the canonical source domain off a skill record.

    Reads ``skill.source_domains[0]`` (the canonical foundry domain
    per PLAN-SKILL-BANK §0.4). Falls back to ``default`` when the
    field is empty / missing -- tolerant to legacy records that
    pre-date the source/target asymmetry.
    """
    src = getattr(skill, "source_domains", None) or []
    if isinstance(src, (list, tuple)) and src:
        first = src[0]
        if isinstance(first, str) and first:
            return first
    return default


def with_predicate_translation(
    success_fn_factory: Callable[..., Callable[..., Any]],
    *,
    target_domain: str,
    default_source: str = "gymv",
) -> Callable[..., Callable[..., Any]]:
    """Wrap a ``make_*_success_fn`` factory with per-skill translation.

    The wrapped factory returns a success_fn that:

    1. Reads the source domain off ``skill.source_domains[0]`` (or
       ``default_source`` if empty).
    2. Calls :func:`translate_skill_contract` to produce a translated
       copy of the skill.
    3. Delegates to the original success_fn with the translated copy.

    Forwards ``*args, **kwargs`` to the original factory verbatim, so
    the wrapper is a drop-in replacement -- the dispatcher calls it
    exactly the way it called the original::

        # Before:
        success_fn_factory=make_qa_success_fn,

        # After:
        success_fn_factory=with_predicate_translation(
            make_qa_success_fn, target_domain="visual_reasoning",
        ),
    """
    def wrapped_factory(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        inner_success_fn = success_fn_factory(*args, **kwargs)

        def translated_success_fn(skill: Any, *fn_args: Any, **fn_kwargs: Any) -> Any:
            source = _resolve_source_domain(skill, default_source)
            translated = translate_skill_contract(
                skill, source=source, target=target_domain,
            )
            return inner_success_fn(translated, *fn_args, **fn_kwargs)

        # Preserve the original __name__ / __doc__ so logs that print
        # success_fn.__name__ stay informative.
        translated_success_fn.__name__ = (
            f"translated_{getattr(inner_success_fn, '__name__', 'success_fn')}"
        )
        translated_success_fn.__doc__ = (
            f"{getattr(inner_success_fn, '__doc__', '') or ''}\n\n"
            f"Wrapped by with_predicate_translation(target_domain="
            f"{target_domain!r}, default_source={default_source!r})."
        )
        return translated_success_fn

    wrapped_factory.__name__ = (
        f"translated_{getattr(success_fn_factory, '__name__', 'factory')}"
    )
    wrapped_factory.__wrapped__ = success_fn_factory  # type: ignore[attr-defined]
    return wrapped_factory
