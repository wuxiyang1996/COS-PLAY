"""Harness → RAG feedback signal: deboost candidates that the
eligibility filter has been rejecting.

Closes the ``coevolution-cross-domain-integration.md`` follow-up
"harness teaches RAG" loop. Without this layer, a skill that survives
the RAG retrieval but always fails the harness's eligibility check
keeps being surfaced at top-K forever — wasting the LLM's
``skill_selection`` budget on candidates the harness will veto a few
microseconds later. The aggregated rejection signal already lives on
:attr:`data_structure.extensions.skill_record.SkillRecord.false_binding_patterns`
(populated by :class:`harness.rejected_skill_sink.RejectedSkillSink` →
:meth:`skill_bank.lifecycle.SkillLifecycleManager.record_false_binding_pattern`);
this module turns that history into a multiplicative deboost the
RAG ranker can apply to its `confidence` / `relevance` scores.

Design notes
------------
* **Pure-data + thin glue**, by deliberate construction:
  :func:`compute_deboost` is a stateless function over a
  ``false_binding_patterns`` list. The only IO surface is the optional
  :func:`apply_deboost_to_candidates` convenience helper.
* **Multiplicative, not subtractive.** The deboost factor is in
  ``[deboost_floor, 1.0]``. Multiplying preserves the relative
  ordering RAG learned from embeddings — heavily-vetoed skills sink,
  but never below ``deboost_floor`` so they can still be sampled if
  no fresh candidate exists.
* **Domain/task scoped.** Patterns observed for the current
  ``(domain, task)`` weigh more than patterns observed elsewhere — a
  skill that's only ever been vetoed in a *different* task isn't
  evidence it'll fail here.
* **Recency-aware.** When ``time.time()``-style ``last_observed_at``
  fields are present, very old vetoes weigh less (geometric decay).
  Records without timestamps fall back to count-only weighting.
* **Cap-bounded.** ``SkillRecord.false_binding_patterns`` is itself
  capped at 64 entries by the lifecycle manager, so the per-skill
  cost of this function is tiny (≤64 dict reads).

Cross-refs
----------
* Design memo (this loop): the answer to the user's
  "RAG-vs-harness for picking" design question — harness *informs*
  RAG without *replacing* the LLM picker. See
  ``harness/README.md`` §22.5.
* :data:`harness.rejected_skill_sink.RejectedSkillSink` — the
  upstream sink that flushes into ``false_binding_patterns``.
* :func:`skill_bank.lifecycle.SkillLifecycleManager.record_false_binding_pattern`
  — the writer; see for the canonical pattern dict schema.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger("harness.rejection_deboost")

__all__ = [
    "DEFAULT_DEBOOST_FLOOR",
    "DEFAULT_DEBOOST_HALF_LIFE_COUNT",
    "DEFAULT_DEBOOST_RECENCY_HALF_LIFE_S",
    "DEFAULT_OFF_AXIS_DISCOUNT",
    "compute_deboost",
    "apply_deboost_to_candidates",
]


# ---------------------------------------------------------------------------
# Tunables (kept low-magnitude on purpose)
# ---------------------------------------------------------------------------

# Minimum deboost factor. A skill with infinite vetoes still has a
# floor of this value — never zero. This keeps the trainer's
# exploration-vs-exploitation balance: if the bank is sparse, even
# heavily-vetoed skills can still surface.
DEFAULT_DEBOOST_FLOOR: float = 0.10

# Geometric half-life on the per-(domain, task) veto count: at this
# many cumulative observations, the deboost factor halves (relative
# to the current floor). Default 3 means three vetoes ⇒ 0.5
# multiplier; six vetoes ⇒ 0.33; etc.
DEFAULT_DEBOOST_HALF_LIFE_COUNT: float = 3.0

# Recency half-life in seconds. A pattern last observed ``half_life``
# seconds ago contributes half its full weight. ``None`` disables
# recency decay (count-only). Default 1 day — generous enough that
# in-session vetoes dominate, but not so generous that ancient
# bank entries keep punishing rehabilitated skills.
DEFAULT_DEBOOST_RECENCY_HALF_LIFE_S: Optional[float] = 86400.0

# Off-axis discount: weight applied to patterns observed on a
# *different* (domain, task) than the current request. ``1.0`` would
# punish a skill equally regardless of where the vetoes happened.
# ``0.0`` would ignore off-axis history entirely. Default 0.25 is
# intentionally low — cross-domain vetoes are weak evidence on the
# current axis, but not zero.
DEFAULT_OFF_AXIS_DISCOUNT: float = 0.25


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


def compute_deboost(
    false_binding_patterns: Sequence[Mapping[str, Any]],
    *,
    domain: str = "",
    task: str = "",
    deboost_floor: float = DEFAULT_DEBOOST_FLOOR,
    half_life_count: float = DEFAULT_DEBOOST_HALF_LIFE_COUNT,
    recency_half_life_s: Optional[float] = DEFAULT_DEBOOST_RECENCY_HALF_LIFE_S,
    off_axis_discount: float = DEFAULT_OFF_AXIS_DISCOUNT,
    now_s: Optional[float] = None,
) -> float:
    """Return a multiplicative deboost factor in
    ``[deboost_floor, 1.0]`` for a skill with the given veto history.

    Parameters
    ----------
    false_binding_patterns
        Iterable of pattern dicts as written by
        :meth:`SkillLifecycleManager.record_false_binding_pattern`.
        Each entry is read for its ``domain``, ``task``, ``count``,
        and (optionally) ``last_observed_at`` fields. Missing keys
        fall back to safe defaults (count=1, no recency adjustment).
    domain, task
        The current request's axis. Patterns observed on a matching
        ``(domain, task)`` are weighted at full strength; off-axis
        patterns are scaled by ``off_axis_discount``. Pass ``""`` for
        either axis to disable that filter.
    deboost_floor
        Minimum return value. Defaults to
        :data:`DEFAULT_DEBOOST_FLOOR`.
    half_life_count
        Geometric half-life on the effective veto count. The
        deboost factor decays as
        ``floor + (1 - floor) * 2 ** (-effective_count / half_life_count)``.
    recency_half_life_s
        Optional seconds-half-life for the recency weight on each
        pattern's contribution. ``None`` disables the recency term.
    off_axis_discount
        Weight applied to patterns observed on a non-matching
        ``(domain, task)``. Range ``[0.0, 1.0]``.
    now_s
        Override clock for testing. Defaults to ``time.time()`` at
        call time.

    Returns
    -------
    float
        Deboost factor. ``1.0`` when the patterns list is empty or
        every pattern carries effectively-zero weight; approaches
        ``deboost_floor`` as the on-axis veto count grows large.

    Notes
    -----
    The function is intentionally non-destructive: it never mutates
    the input list. It also never raises — malformed entries are
    silently treated as ``count=0`` so a partially-corrupt
    persisted bank can't break retrieval.
    """
    if not false_binding_patterns:
        return 1.0

    floor = max(0.0, min(1.0, float(deboost_floor)))
    half_life = max(1e-6, float(half_life_count))
    if now_s is None:
        now_s = time.time()

    effective_count = 0.0
    for entry in false_binding_patterns:
        if not isinstance(entry, Mapping):
            continue
        raw_count = entry.get("count", 1)
        try:
            count = int(raw_count) if raw_count is not None else 1
        except (TypeError, ValueError):
            count = 1
        if count <= 0:
            continue

        # Domain/task axis weight
        entry_domain = str(entry.get("domain") or "")
        entry_task = str(entry.get("task") or "")
        on_axis = (
            (not domain or entry_domain == domain)
            and (not task or entry_task == task)
        )
        axis_weight = 1.0 if on_axis else max(0.0, min(1.0, float(off_axis_discount)))
        if axis_weight <= 0.0:
            continue

        # Recency weight (geometric decay).
        recency_weight = 1.0
        if recency_half_life_s and recency_half_life_s > 0:
            try:
                last = float(entry.get("last_observed_at") or 0.0)
            except (TypeError, ValueError):
                last = 0.0
            if last > 0:
                age_s = max(0.0, float(now_s) - last)
                recency_weight = math.pow(2.0, -age_s / float(recency_half_life_s))

        effective_count += count * axis_weight * recency_weight

    if effective_count <= 0.0:
        return 1.0

    factor = floor + (1.0 - floor) * math.pow(2.0, -effective_count / half_life)
    # Clamp for paranoia (math is well-behaved, but caller-supplied
    # floors at 1.0 etc. should still produce sensible output).
    return max(floor, min(1.0, factor))


# ---------------------------------------------------------------------------
# Convenience: thread the deboost through a candidate-dict list
# ---------------------------------------------------------------------------


def apply_deboost_to_candidates(
    candidates: List[Dict[str, Any]],
    *,
    fetch_record: Callable[[str], Any],
    domain: str = "",
    task: str = "",
    score_keys: Sequence[str] = ("confidence", "relevance"),
    annotation_key: str = "_harness_deboost",
    sort_by: Optional[str] = "confidence",
    **deboost_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Apply the deboost to each candidate's score and (optionally)
    re-sort the list.

    Parameters
    ----------
    candidates
        List of candidate dicts as produced by
        :func:`scripts.qwen3_decision_agent.get_top_k_skill_candidates`.
        Each must carry a ``skill_id`` field; candidates without one
        pass through unchanged.
    fetch_record
        Callable mapping ``skill_id -> SkillRecord | None``. Typically
        a thin wrapper over the live ``skill_bank``'s ``get_skill``
        method. ``None`` returns are tolerated (skill not in cache ⇒
        no deboost).
    domain, task
        Forwarded to :func:`compute_deboost`. Match the harness's
        current request axis so on-axis vetoes weigh fully.
    score_keys
        Which numeric fields on each candidate dict to multiply by
        the deboost factor. Defaults to ``("confidence", "relevance")``.
        Missing keys are ignored.
    annotation_key
        The candidate dict gets an extra field with this name carrying
        the deboost factor (≤1.0). Useful for the prompt layer
        (Refinement B) and for downstream audit. Pass ``""`` to skip
        the annotation.
    sort_by
        Re-sort candidates descending by this score-key after
        deboost application. ``None`` preserves the input order.
        Defaults to ``"confidence"``.
    **deboost_kwargs
        Passed through to :func:`compute_deboost` (e.g.
        ``deboost_floor=0.05`` for a less aggressive floor in tests).

    Returns
    -------
    list[dict]
        New list with deboosted scores. Input dicts are mutated
        in-place (we copy upstream); to avoid that, the caller
        should pre-copy.

    Notes
    -----
    Best-effort: any exception raised by ``fetch_record`` for a
    specific skill is logged at DEBUG and that candidate is left
    unchanged. The reranker must never break the rollout.
    """
    if not candidates:
        return list(candidates or [])

    out: List[Dict[str, Any]] = []
    for cand in candidates:
        sid = (cand or {}).get("skill_id")
        if not sid:
            out.append(cand)
            continue
        try:
            rec = fetch_record(sid)
        except Exception as exc:                         # noqa: BLE001
            logger.debug(
                "rejection_deboost: fetch_record(%s) raised %s — skipping",
                sid, exc,
            )
            out.append(cand)
            continue
        if rec is None:
            out.append(cand)
            continue
        patterns = list(getattr(rec, "false_binding_patterns", []) or [])
        factor = compute_deboost(
            patterns,
            domain=domain,
            task=task,
            **deboost_kwargs,
        )
        # Mutate score keys (safe — we treat candidate dicts as
        # transient per-step structures, mirrors `_enrich_candidate`'s
        # contract in `qwen3_decision_agent`).
        for key in score_keys:
            v = cand.get(key)
            if isinstance(v, (int, float)):
                cand[key] = float(v) * factor
        if annotation_key:
            cand[annotation_key] = factor
        out.append(cand)

    if sort_by:
        out.sort(
            key=lambda c: float((c or {}).get(sort_by, 0.0) or 0.0),
            reverse=True,
        )
    return out
