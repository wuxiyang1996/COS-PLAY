"""Per-rollout `SkillHarnessHook` for the trainer's co-evolution loop.

Splices into ``trainer.coevolution.episode_runner.run_episode_async`` so
the harness's two LLM-free, deterministic surfaces (PLAN-HARNESS §5.2 +
PLAN-UNIFIED-SKILL-GATE §3.4) get applied around the actor's
``skill_selection`` LLM call:

  1. **Pre-LLM filter** — :meth:`SkillHarnessHook.filter_candidates`
     wraps the cold-start RAG output from
     :func:`scripts.qwen3_decision_agent.get_top_k_skill_candidates`
     through :meth:`harness.skill_harness.SkillHarness.select_eligible_skills`.
     Skills the harness vetoes (status / domain / task / adapter / can_handle)
     are dropped *before* the LLM sees them, and their veto reason is
     observed by an in-process :class:`harness.RejectedSkillSink`.

  2. **Post-LLM validation** — :meth:`SkillHarnessHook.validate_choice`
     wraps the LLM's chosen skill through
     :meth:`harness.skill_harness.SkillHarness.validate_invocation`.
     If vetoed, the actor falls back to the next surviving eligible
     candidate (or runs unguided when none remains).

The aggregated rejections survive across episodes within a single
trainer step. After Phase A, :meth:`SkillHarnessHook.flush_to_lifecycle`
drains the sink into a freshly-seeded :class:`SkillLifecycleManager`
inside the existing Crafter hook so each
``SkillRecord.false_binding_patterns`` lights up the Repairer's
patch-or-retire path.

Strict trainer-mode contract
----------------------------
* No live LLM call; no env binding. The hook is pure CPU (microseconds
  per step). The deterministic ``GymvAdapter`` stub is registered so
  ``F3 adapter`` and ``F4 can_handle`` do not falsely veto every
  candidate. Real ``set_executor(env_step_fn)`` wiring stays out of
  scope — that lives in ``vlm_wrapper/gymv_wrapper`` (see
  ``harness/README.md`` §16.1).
* All public methods are wrapped in ``try / except`` so a harness bug
  can never break a training rollout. On internal failure the hook
  reports a degraded result and the actor proceeds unchanged.
* Skills the live bank.jsonl exposes are *de-facto* provisional — they
  passed the legacy 4-stage pipeline. We mint them with
  ``status = SkillStatus.PROVISIONAL`` so :class:`EligibilityFilter`'s
  F1 status check admits them. The lifecycle manager is the only
  authority that may transition status; this hook never persists any
  status mutation — it just produces an in-memory snapshot for the
  filter to consume.

Cross-refs
----------
* ``harness/README.md`` §22 (trainer integration block).
* ``trainer/coevolution/_crafter_hook.py`` (consumes the rejection
  sink via ``flush_to_lifecycle``).
* ``crafter-harness-orchestrator-roles.md`` §2.2 (Harness role boundary).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from common.enums import SkillStatus, SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.adapters.gymv_adapter import GymvAdapter
from harness.eligibility import EligibleSkill, RejectedSkill, task_id_from_state
from harness.predicate_translator import translate_skill_contract
from harness.rejected_skill_sink import FlushReport, RejectedSkillSink
from harness.skill_harness import HarnessConfig, SkillHarness, ValidateInvocationResult
from trainer.coevolution._run_loggers import (
    log_harness_rejection,
    log_harness_validate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step diagnostic records the orchestrator can roll up into wandb / TB
# ---------------------------------------------------------------------------


@dataclass
class HarnessStepStats:
    """Roll-up of one episode's harness activity (logged in `experiences`)."""

    n_candidates_in: int = 0
    n_candidates_admitted: int = 0
    n_candidates_rejected: int = 0
    n_validate_ok: int = 0
    n_validate_veto: int = 0
    n_validate_skipped: int = 0   # candidate not in lifecycle cache
    n_predicate_translations_applied: int = 0   # Layer C: how many candidate
                                                # contracts had at least one
                                                # effects_{add,del} predicate
                                                # rewritten before eligibility.
    n_predicate_translations_failed: int = 0    # translator threw -> identity
                                                # fallback; counted but never
                                                # propagated to the caller.
    veto_class_distribution: Dict[str, int] = field(default_factory=dict)
    last_eligible_ids: List[str] = field(default_factory=list)
    last_rejected: List[Dict[str, Any]] = field(default_factory=list)
    last_validate: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "n_candidates_in": self.n_candidates_in,
            "n_candidates_admitted": self.n_candidates_admitted,
            "n_candidates_rejected": self.n_candidates_rejected,
            "n_validate_ok": self.n_validate_ok,
            "n_validate_veto": self.n_validate_veto,
            "n_validate_skipped": self.n_validate_skipped,
            "n_predicate_translations_applied": self.n_predicate_translations_applied,
            "n_predicate_translations_failed": self.n_predicate_translations_failed,
            "veto_class_distribution": dict(self.veto_class_distribution),
            "last_eligible_ids": list(self.last_eligible_ids),
            "last_rejected": list(self.last_rejected),
            "last_validate": dict(self.last_validate) if self.last_validate else None,
        }


# ---------------------------------------------------------------------------
# `SkillRecord` cache hydration from the trainer's per-game bank.jsonl
# ---------------------------------------------------------------------------


def _hydrate_records_from_bank(
    bank_path: Path,
    *,
    default_domain: str,
    runtime_status: SkillStatus = SkillStatus.PROVISIONAL,
) -> Dict[str, SkillRecord]:
    """Lift one per-game ``skill_bank.jsonl`` into ``Dict[skill_id, SkillRecord]``.

    Reuses :func:`trainer.coevolution._crafter_hook._record_from_bank_entry`
    so the hydration logic is shared with the Crafter hook (single
    source of truth for the legacy → record schema).

    Skills are mounted with ``runtime_status=PROVISIONAL`` rather than
    the default ``DRAFT`` so the eligibility filter's F1 status check
    admits them. This is a *runtime view*, not a persisted transition —
    the lifecycle manager remains the only authority that may write
    status to disk (PLAN-SKILL-BANK §0.5).

    Malformed lines are silently skipped (matching the Crafter hook's
    tolerance — the live bank is a running artefact and we'd rather
    seed N-1 records than refuse to fire).
    """
    if not bank_path.is_file():
        return {}
    # Lazy import — `_crafter_hook` pulls in heavyweight crafter deps via
    # transitive imports; we only want them when the harness hook actually
    # builds. Importing under a leading-underscore name is intentional:
    # both hook modules live in the same package and share the loader.
    from trainer.coevolution._crafter_hook import _record_from_bank_entry

    out: Dict[str, SkillRecord] = {}
    try:
        import json as _json
        with bank_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = _json.loads(raw)
                except Exception:
                    continue
                rec = _record_from_bank_entry(entry, default_domain)
                if rec is None:
                    continue
                # Override the default DRAFT status with the runtime view.
                # `SkillRecord` is a plain dataclass; assignment works,
                # but we go through `object.__setattr__` for consistency
                # with `_record_from_bank_entry`'s skill_id override and
                # to be forward-compatible if a `__setattr__` lock lands.
                object.__setattr__(rec, "status", runtime_status)
                out[rec.skill_id] = rec
    except OSError as exc:  # noqa: BLE001
        logger.debug(
            "harness_hook: failed to read bank %s: %s", bank_path, exc,
        )
    return out


# ---------------------------------------------------------------------------
# Refinement B — adaptation score helpers
# ---------------------------------------------------------------------------
#
# Closes the "harness *informs* the LLM picker, not replace it" loop the
# user asked for: each admitted candidate carries an
# ``_harness_adaptation_score`` in ``[0.0, 1.0]`` summarising how well
# the harness expects the skill to adapt to the current ``(domain,
# task)`` axis. The LLM's skill_selection prompt surfaces this score so
# the LLM picks with a structured prior rather than text-only RAG
# guidance. See ``harness/README.md`` §22.5 for the full design memo.
#
# Components (all in [0, 1], simple arithmetic mean):
#
# * **task-axis match** — was this skill verified / mined for this
#   exact task? Mirrors the ``task_match`` field on
#   :class:`harness.eligibility.EligibleSkill`. ``verified`` is the
#   strongest signal (the skill has *succeeded* on this task at
#   foundry time); ``same_task`` is feasibility-only; ``agnostic``
#   is the back-compat catch-all.
# * **adapter native to target** — does the chosen adapter natively
#   speak the state's domain? When the only available adapter is one
#   transitively bridging a different source domain, the predicate
#   translator had to rewrite the contract — that's a softer signal
#   than a same-domain adapter would be.
# * **predicate translation provenance** — diagonal cells get full
#   credit; successful cross-domain rewrites get partial credit; an
#   identity-fallback (translator raised) is the weakest signal —
#   the harness admitted but the contract may not be perfectly
#   grounded in the target's predicate vocabulary.

_TASK_MATCH_WEIGHTS: Dict[str, float] = {
    "verified": 1.0,
    "same_task": 0.85,
    "agnostic": 0.60,
}
_ADAPTER_NATIVE_SCORE: float = 1.0
_ADAPTER_BRIDGED_SCORE: float = 0.70
_TRANSLATION_DIAGONAL_SCORE: float = 1.0
_TRANSLATION_REWRITTEN_SCORE: float = 0.85
_TRANSLATION_FAILED_SCORE: float = 0.55


def _adapter_native_score(adapter_name: str, target_domain: str) -> float:
    """Heuristic: 1.0 when the chosen adapter natively speaks the target
    domain, else :data:`_ADAPTER_BRIDGED_SCORE`.

    The :class:`AdapterRegistry` doesn't return adapter→domain
    associations on the verdict, so we fall back to a name match —
    e.g. adapter_name ``"gymv"`` × target ``"gymv"`` ⇒ native;
    adapter_name ``"video"`` × target ``"gymv"`` ⇒ bridged. This is
    intentionally conservative: when in doubt about provenance, the
    bridged score still admits the skill but with a softer prior.
    """
    if not adapter_name or not target_domain:
        return _ADAPTER_BRIDGED_SCORE
    return _ADAPTER_NATIVE_SCORE if adapter_name == target_domain else _ADAPTER_BRIDGED_SCORE


def _compute_adaptation_score(
    eligible: EligibleSkill,
    *,
    target_domain: str,
    translation_status: str,
) -> Tuple[float, Dict[str, float]]:
    """Combine task-match, adapter, and translation signals into a
    single :math:`[0,1]` score. Returns ``(score, breakdown)`` so the
    diagnostic dict can render the components for debugging.

    ``translation_status`` is one of ``{"diagonal", "rewritten",
    "failed"}`` (corresponding to the ``rewrote`` / ``failed`` flags
    returned by :func:`_translate_record_for_target`). Unknown values
    default to the diagonal score (no translation evidence).
    """
    s_task = _TASK_MATCH_WEIGHTS.get(eligible.task_match, _TASK_MATCH_WEIGHTS["agnostic"])
    s_adapter = _adapter_native_score(eligible.adapter_name, target_domain)
    if translation_status == "rewritten":
        s_trans = _TRANSLATION_REWRITTEN_SCORE
    elif translation_status == "failed":
        s_trans = _TRANSLATION_FAILED_SCORE
    else:
        s_trans = _TRANSLATION_DIAGONAL_SCORE
    score = (s_task + s_adapter + s_trans) / 3.0
    return score, {
        "task_match": s_task,
        "adapter": s_adapter,
        "translation": s_trans,
    }


# ---------------------------------------------------------------------------
# Layer C — predicate translator integration helpers
# ---------------------------------------------------------------------------


def _resolve_record_source_domain(
    rec: SkillRecord,
    *,
    fallback: str,
) -> str:
    """Pull the canonical *source* domain off a cached :class:`SkillRecord`.

    Mirrors :func:`harness.predicate_translator._resolve_source_domain` but
    inlined here so the hook stays decoupled from a private helper.

    Order of preference (most specific first):

    1. ``rec.source_domains[0]`` — the canonical foundry domain
       (PLAN-SKILL-BANK §0.4). This is set when the cold-start mining
       path stamps the skill's origin corpus on the record.
    2. ``rec.feasible_domains[0]`` — the runtime adapter domain. For
       trainer-side records this collapses to the hook's ``domain``,
       which makes the translator's identity branch fire (no rewrite).
    3. ``fallback`` — the hook's own ``self._domain``. Last-resort so the
       translator always has a non-empty source string to reason about.
    """
    src = getattr(rec, "source_domains", None) or []
    if isinstance(src, (list, tuple)) and src:
        first = src[0]
        if isinstance(first, str) and first:
            return first
    feas = getattr(rec, "feasible_domains", None) or []
    if isinstance(feas, (list, tuple)) and feas:
        first = feas[0]
        if isinstance(first, str) and first:
            return first
    return fallback


def _translate_record_for_target(
    rec: SkillRecord,
    *,
    target_domain: str,
    fallback_source_domain: str,
) -> Tuple[SkillRecord, bool, bool]:
    """Wrap :func:`harness.predicate_translator.translate_skill_contract`
    for the hook's per-candidate loop.

    Returns ``(out_record, was_rewritten, translator_failed)``.

    * ``out_record`` — the (possibly translated) record. On a translator
      crash this is the *original* ``rec``: the eligibility filter must
      still run.
    * ``was_rewritten`` — ``True`` only when the translator returned a
      record whose ``contract.effects_add`` / ``effects_del`` differ
      from the input. Used for the per-step counter only — diagonal /
      identity translations don't bump it.
    * ``translator_failed`` — ``True`` iff the translator raised. Tracked
      separately so a buggy translation table can't masquerade as
      "translation applied" in the dashboard.
    """
    src = _resolve_record_source_domain(rec, fallback=fallback_source_domain)
    if not src or src == target_domain:
        # Same-domain (diagonal) — translator would deep-copy and
        # return the input unchanged. Skip the copy for the common
        # case to keep the per-candidate cost ~free on the hot path.
        return rec, False, False
    try:
        out = translate_skill_contract(rec, source=src, target=target_domain)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "harness_hook: predicate translation %s->%s failed for %s: %s; "
            "falling back to identity",
            src, target_domain, getattr(rec, "skill_id", "?"), exc,
        )
        return rec, False, True

    if out is None:
        return rec, False, True

    rewrote = False
    in_contract = getattr(rec, "contract", None)
    out_contract = getattr(out, "contract", None)
    if in_contract is not None and out_contract is not None:
        in_add = list(getattr(in_contract, "effects_add", []) or [])
        in_del = list(getattr(in_contract, "effects_del", []) or [])
        out_add = list(getattr(out_contract, "effects_add", []) or [])
        out_del = list(getattr(out_contract, "effects_del", []) or [])
        rewrote = (in_add != out_add) or (in_del != out_del)
    return out, rewrote, False


# ---------------------------------------------------------------------------
# `StateSchema` synthesis from the trainer's per-step text bundle
# ---------------------------------------------------------------------------


def _state_for_step(
    *,
    game: str,
    summary_state: str,
    intention: str,
    domain: str = "gymv",
    inner_step: int = 0,
    outer_step: int = 0,
) -> StateSchema:
    """Build a minimal :class:`StateSchema` from the trainer's per-step
    text bundle so the eligibility filter can run F2/F2′/F4.

    The trainer's actor doesn't yet emit a full :class:`StateSchema`
    (that lives in the cold-start mining path). We synthesise one with
    just enough information to make the eligibility filter and
    ``validate_invocation`` produce a meaningful verdict:

    * ``domain``    — fixed to the adapter domain for the trainer's
                      games. All trainer-side games map to ``"gymv"``.
    * ``task``      — set to ``game`` so F2′ task-axis is enforced
                      consistently with the offline mirror.
    * ``facts``     — empty (no precondition predicates evaluated
                      against text). The deterministic adapter only
                      needs the domain to admit, so this is fine.
    * ``evidence``  — empty list; the live actor's evidence chain is
                      not lifted here. The harness's
                      ``validate_invocation`` only checks
                      ``evidence_in`` for ACTION skills (which require
                      it); GROUNDING / REASONING skills are exempt.

    Failures degrade to a permissive ``StateSchema(domain=domain,
    task=game)`` so the live runner is never starved by a malformed
    summary.
    """
    try:
        return StateSchema(
            domain=domain,
            task=game,
            evidence=[],
            inner_step=int(inner_step),
            outer_step=int(outer_step),
            extra={
                "summary_state": (summary_state or "")[:1024],
                "intention": (intention or "")[:256],
            },
        )
    except Exception:  # noqa: BLE001
        return StateSchema(domain=domain, task=game)


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------


class SkillHarnessHook:
    """Trainer-side façade around the harness's eligibility + validate
    surfaces.

    One instance per (game, step). The hook is cheap to build (registers
    a :class:`GymvAdapter` and hydrates the per-game bank.jsonl into a
    flat dict) and stateful only across the episodes within a single
    step — the orchestrator drains the rejection sink into the Crafter
    hook's lifecycle and discards the hook at end of step.

    Lifecycle
    ---------
    1. Orchestrator builds one ``SkillHarnessHook`` per game per step
       (``cls.for_game(...)``), passing the path to the per-game
       ``skill_bank.jsonl``.
    2. ``filter_candidates(candidates, state)`` is called inside the
       episode loop, before the ``skill_selection`` LLM. Returns the
       narrowed candidate list and an opaque diagnostic dict for
       ``experiences[].harness``.
    3. ``validate_choice(skill_id, state)`` is called after the LLM has
       picked one. Returns ``(ok, validation_dict)``.
    4. After Phase A, the orchestrator passes the hook's
       :class:`RejectedSkillSink` to the Crafter hook so the
       per-skill_id veto patterns end up on
       ``SkillRecord.false_binding_patterns`` via
       ``SkillLifecycleManager.record_false_binding_pattern``.

    All public methods catch internal failures and downgrade to
    pass-through behaviour — the trainer's rollout must never break
    because the harness disagreed with itself.
    """

    def __init__(
        self,
        *,
        domain: str = "gymv",
        records: Optional[Mapping[str, SkillRecord]] = None,
        allow_shadow: bool = True,
        adapters: Optional[List[Any]] = None,
        sink: Optional[RejectedSkillSink] = None,
        # ── Path 4 — LLM Harness validator wire-up ──────────────────
        # When ``llm_validator`` is non-None, every successful
        # ``validate_choice`` admit is offered to the validator for a
        # second-pass 35B veto.  Construction of the validator (and
        # the trainer-step / game_profile threading) lives in
        # :meth:`for_game`; passing ``None`` (default) preserves the
        # historical pure-deterministic behaviour.
        llm_validator: Any = None,
    ) -> None:
        self._domain = domain
        self._records: Dict[str, SkillRecord] = dict(records or {})
        self._sink = sink if sink is not None else RejectedSkillSink()

        registry = AdapterRegistry()
        adapter_list = list(adapters) if adapters is not None else [GymvAdapter()]
        for ad in adapter_list:
            try:
                registry.register(ad)
            except ValueError:
                # Already registered (shared registry) — fine.
                pass
        self._registry = registry

        self._harness = SkillHarness(
            registry=registry,
            config=HarnessConfig(allow_shadow=allow_shadow),
        )

        self._stats = HarnessStepStats()
        self._llm_validator = llm_validator
        # Outer trainer step + game name — used by the per-event
        # reviewer-facing JSONL log (block A1/A2).  Set by ``for_game``;
        # default to -1 / "" when constructed directly (tests).
        self._trainer_step: int = -1
        self._game: str = ""
        # Block B1 — harness mode for the §5.5 ablation:
        #   * ``"full"`` (default) — eligibility filter +
        #     validate_invocation as historical.
        #   * ``"plain-text-skills"`` — skills surfaced as-is to the
        #     actor; eligibility filter and validate_invocation
        #     bypassed.
        self._mode: str = "full"

    # ── Construction helpers ─────────────────────────────────────────

    @classmethod
    def for_game(
        cls,
        *,
        game: str,
        bank_path: Optional[Path],
        domain: str = "gymv",
        allow_shadow: bool = True,
        # ── Path 4 — optional LLM validator wiring ──────────────────
        # All four ``llm_validator_*`` knobs are passive: when
        # ``llm_validator_enabled=False`` (default) we fast-path to
        # the historical pure-deterministic behaviour and the rest
        # are ignored.
        llm_validator_enabled: bool = False,
        llm_validator_model: str = "",
        trainer_step: int = 0,
        bootstrap_steps: int = 20,
        llm_validator_max_tokens: int = 256,
        llm_validator_temperature: float = 0.2,
        llm_validator_timeout_s: float = 30.0,
        game_profile: Any = None,
        # Block B1 — harness ablation mode (see ``self._mode`` docs).
        mode: str = "full",
    ) -> "SkillHarnessHook":
        """Build a hook for one game, hydrating its bank.jsonl into the
        SkillRecord cache.

        Returns an empty-cache hook (still functional, just with no
        records to filter) when the bank file is missing — matching
        the cold-start convention where step 0's bank is empty.
        """
        records: Dict[str, SkillRecord] = {}
        if bank_path is not None:
            records = _hydrate_records_from_bank(
                Path(bank_path), default_domain=domain,
            )

        # Path 4 — instantiate the LLM validator when enabled.  Lazy
        # import keeps the trainer's hot path light when the flag is
        # off (the validator imports ``API_func`` transitively).
        validator: Any = None
        if llm_validator_enabled:
            try:
                from trainer.coevolution._llm_harness_validator import (
                    LLMHarnessValidator,
                )
                # Empty model slug → defer to the env-exported
                # ``VLM_AGENT_BACKBONE_JUDGE_MODEL`` (set by the launch
                # script in tandem with VLLM_BASE_URL_MAP).  Final
                # fallback: the canonical 35B-A3B slug.
                import os as _os
                _resolved_validator_model = (
                    (llm_validator_model or "").strip()
                    or _os.environ.get(
                        "VLM_AGENT_BACKBONE_JUDGE_MODEL", "",
                    ).strip()
                    or "Qwen/Qwen3.5-35B-A3B"
                )
                validator = LLMHarnessValidator(
                    model=_resolved_validator_model,
                    trainer_step=trainer_step,
                    bootstrap_steps=bootstrap_steps,
                    max_tokens=llm_validator_max_tokens,
                    temperature=llm_validator_temperature,
                    timeout_s=llm_validator_timeout_s,
                    game_profile=game_profile,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "harness_hook.for_game(%s): LLM validator init failed "
                    "(%s) — proceeding with deterministic-only validation",
                    game, exc,
                )
                validator = None

        hook = cls(
            domain=domain,
            records=records,
            allow_shadow=allow_shadow,
            llm_validator=validator,
        )
        hook._trainer_step = int(trainer_step)
        hook._game = str(game)
        # Block B1 — clamp to the three known modes; unknown values
        # silently fall back to "full" so a config typo can never
        # accidentally disable the harness mid-run.
        hook._mode = (
            mode if mode in ("full", "plain-text-skills", "off") else "full"
        )
        return hook

    # ── Properties ───────────────────────────────────────────────────

    @property
    def sink(self) -> RejectedSkillSink:
        return self._sink

    @property
    def stats(self) -> HarnessStepStats:
        return self._stats

    def n_records(self) -> int:
        return len(self._records)

    # ── Per-step API ─────────────────────────────────────────────────

    def state_for_step(
        self,
        *,
        game: str,
        summary_state: str = "",
        intention: str = "",
        inner_step: int = 0,
        outer_step: int = 0,
    ) -> StateSchema:
        return _state_for_step(
            game=game,
            summary_state=summary_state,
            intention=intention,
            domain=self._domain,
            inner_step=inner_step,
            outer_step=outer_step,
        )

    def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        state: StateSchema,
        *,
        episode_id: str = "",
        outer_step: Optional[int] = None,
        game: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Pre-LLM eligibility filter (PLAN-HARNESS §5.2).

        Maps cold-start RAG candidate dicts → ``SkillRecord`` (via the
        per-game cache) → :meth:`SkillHarness.select_eligible_skills`.
        Skills the cache doesn't know are passed through unchanged
        (degrade gracefully — no veto, no observe).

        Returns ``(filtered_candidates, diagnostic_dict)``. The
        diagnostic dict matches the
        ``HarnessStepStats.to_json()`` shape and is suitable for
        embedding in ``experiences[].harness``.

        ``episode_id`` / ``outer_step`` / ``game`` are best-effort
        context for the per-event rejection log (``run_dir/harness_log/
        rejections.jsonl``).  Empty / -1 means "unknown" — the log row
        still emits, just with empty context fields.

        When :attr:`_mode` == ``"plain-text-skills"`` the eligibility
        filter is bypassed entirely (block B1 ablation): all input
        candidates pass through unchanged so the actor sees the raw
        skill-bank content with no grounding / precondition /
        admissibility check.  The diagnostic is still populated so
        downstream logging stays consistent.
        """
        n_in = len(candidates or [])
        diag: Dict[str, Any] = {
            "n_in": n_in,
            "n_admitted": 0,
            "n_rejected": 0,
            "n_unknown": 0,
            "n_predicate_translations_applied": 0,
            "n_predicate_translations_failed": 0,
            "eligible_ids": [],
            "rejected": [],
            "task_match_distribution": {},
        }

        if not candidates:
            return [], diag

        # Block B1: plain-text-skills bypass.  Skills are still surfaced
        # to the actor but no eligibility check fires.  See the
        # docstring above for ablation rationale.
        if self._mode == "plain-text-skills":
            diag["n_admitted"] = n_in
            diag["mode"] = "plain-text-skills"
            return list(candidates), diag

        # Map each candidate dict to its corresponding cached SkillRecord.
        # Unknowns are passed through (the harness has no opinion on
        # skills it doesn't know — that may happen in unit tests or
        # when the bank.jsonl was rotated mid-step).
        #
        # Layer C — predicate translator splice
        # -------------------------------------
        # Skills in the cache may have been mined in a *source* domain
        # (e.g. ``visual_reasoning`` / ``video``) different from the
        # trainer's current ``state.domain`` (typically ``gymv``). When
        # the source and target diverge, the contract's effects_add /
        # effects_del predicates need to be rewritten through
        # :func:`harness.predicate_translator.translate_skill_contract`
        # so the harness's eligibility filter — and the downstream
        # success_fn — see predicates the target adapter can actually
        # ground (PLAN-PHASE5 §11.5.0 / coevolution-cross-domain-
        # integration.md Layer C). Diagonal / identity cells short-
        # circuit and bypass the deep-copy cost. Translator crashes
        # degrade to the original record so a buggy translation table
        # can never starve the trainer.
        records_for_filter: List[SkillRecord] = []
        unknown_idx: List[int] = []
        sid_to_cand: Dict[str, Dict[str, Any]] = {}
        # Refinement B: per-skill translation status, consumed below
        # when we compute the adaptation_score for each admitted skill.
        sid_to_translation_status: Dict[str, str] = {}
        n_translated = 0
        n_translation_errs = 0
        target_domain = getattr(state, "domain", None) or self._domain
        for i, cand in enumerate(candidates):
            sid = (cand or {}).get("skill_id")
            rec = self._records.get(sid) if sid else None
            if rec is None:
                unknown_idx.append(i)
                continue
            translated_rec, rewrote, failed = _translate_record_for_target(
                rec,
                target_domain=target_domain,
                fallback_source_domain=self._domain,
            )
            if rewrote:
                n_translated += 1
            if failed:
                n_translation_errs += 1
            if failed:
                sid_to_translation_status[sid] = "failed"
            elif rewrote:
                sid_to_translation_status[sid] = "rewritten"
            else:
                sid_to_translation_status[sid] = "diagonal"
            records_for_filter.append(translated_rec)
            sid_to_cand[sid] = cand

        try:
            admitted, rejected = self._harness._eligibility.filter_with_rejections(
                records_for_filter, state,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "harness_hook.filter_candidates: filter raised %s — "
                "degrading to pass-through", exc,
            )
            return list(candidates), {
                **diag,
                "n_admitted": n_in,
                "n_predicate_translations_applied": n_translated,
                "n_predicate_translations_failed": n_translation_errs,
                "harness_error": repr(exc),
            }

        # Observe rejections into the sink (keyed by the source domain
        # / task — the receiving lifecycle uses these for `false_binding_patterns`).
        if rejected:
            try:
                self._sink.observe(
                    rejected,
                    domain=state.domain,
                    task=task_id_from_state(state),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "harness_hook.filter_candidates: sink.observe raised %s",
                    exc,
                )
            # Block A1: also stream every rejection event to the
            # reviewer-facing per-event JSONL log so the §4.3 failure-
            # mode pie chart can be reconstructed post-hoc.  In-process
            # sink only retains aggregated patterns; the per-event
            # veto code + skill_id pair is otherwise discarded.
            try:
                _task = task_id_from_state(state)
            except Exception:
                _task = ""
            _step = self._trainer_step if outer_step is None else int(outer_step)
            _game = self._game if game is None else str(game)
            for _r in rejected:
                try:
                    log_harness_rejection(
                        step=_step,
                        episode_id=episode_id,
                        game=_game,
                        domain=state.domain,
                        task=_task,
                        skill_id=getattr(_r.skill, "skill_id", ""),
                        veto=_r.veto,
                        veto_reason=_r.veto_reason,
                    )
                except Exception as _logexc:  # noqa: BLE001
                    logger.debug(
                        "harness_hook: log_harness_rejection raised %s",
                        _logexc,
                    )

        # Reconstruct the filtered candidate list, preserving the order
        # the LLM expects (admitted first, then unknowns — the LLM
        # already had unknowns in its prior view, so dropping them
        # would be a behavioural change beyond the eligibility surface).
        admitted_sids = {e.skill.skill_id for e in admitted}
        filtered: List[Dict[str, Any]] = []
        eligible_ids: List[str] = []
        task_match_dist: Dict[str, int] = {}
        adaptation_scores: List[float] = []
        for e in admitted:
            cand = sid_to_cand.get(e.skill.skill_id)
            if cand is None:
                continue
            # Decorate with the eligibility verdict so the actor's
            # downstream logging / GRPO records can reason about it.
            cand2 = dict(cand)
            cand2["_harness_eligible"] = e.to_json()
            # Refinement B: emit the adaptation score so the
            # skill_selection prompt can render it for the LLM. The
            # breakdown is included for offline inspection but isn't
            # surfaced to the LLM by default (it's redundant with
            # the headline score).
            translation_status = sid_to_translation_status.get(
                e.skill.skill_id, "diagonal",
            )
            score, breakdown = _compute_adaptation_score(
                e,
                target_domain=target_domain,
                translation_status=translation_status,
            )
            cand2["_harness_adaptation_score"] = float(score)
            cand2["_harness_adaptation_breakdown"] = {
                **breakdown,
                "translation_status": translation_status,
            }
            adaptation_scores.append(float(score))
            filtered.append(cand2)
            eligible_ids.append(e.skill.skill_id)
            task_match_dist[e.task_match] = task_match_dist.get(e.task_match, 0) + 1
        for i in unknown_idx:
            filtered.append(candidates[i])

        # Diagnostic block.
        diag["n_admitted"] = len(admitted)
        diag["n_rejected"] = len(rejected)
        diag["n_unknown"] = len(unknown_idx)
        diag["n_predicate_translations_applied"] = n_translated
        diag["n_predicate_translations_failed"] = n_translation_errs
        diag["eligible_ids"] = eligible_ids
        diag["rejected"] = [r.to_json() for r in rejected]
        diag["task_match_distribution"] = task_match_dist
        # Refinement B summary stats — let the orchestrator log
        # adaptation-score moments without re-iterating over the
        # filtered candidate dicts.
        if adaptation_scores:
            diag["adaptation_score_min"] = min(adaptation_scores)
            diag["adaptation_score_max"] = max(adaptation_scores)
            diag["adaptation_score_mean"] = (
                sum(adaptation_scores) / len(adaptation_scores)
            )
        else:
            diag["adaptation_score_min"] = None
            diag["adaptation_score_max"] = None
            diag["adaptation_score_mean"] = None

        # Step stats (per-episode rollup).
        self._stats.n_candidates_in += n_in
        self._stats.n_candidates_admitted += len(admitted)
        self._stats.n_candidates_rejected += len(rejected)
        self._stats.n_predicate_translations_applied += n_translated
        self._stats.n_predicate_translations_failed += n_translation_errs
        for r in rejected:
            self._stats.veto_class_distribution[r.veto] = (
                self._stats.veto_class_distribution.get(r.veto, 0) + 1
            )
        self._stats.last_eligible_ids = list(eligible_ids)
        self._stats.last_rejected = list(diag["rejected"])

        return filtered, diag

    def validate_choice(
        self,
        skill_id: Optional[str],
        state: StateSchema,
        *,
        bindings: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None,
        inner_step: int = 0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Post-LLM second-pass invocation veto (PLAN-UNIFIED §3.4).

        Returns ``(ok, diagnostic_dict)``. ``ok=False`` means the actor
        should fall back to the next surviving eligible candidate.
        ``ok=True`` is the default for skills the cache doesn't know
        (we can't prove a veto, so we admit — degrade gracefully).

        When the hook was built with a Path 4 LLM validator,
        a successful deterministic admit is additionally offered to
        the validator for a second-pass 35B veto.  The LLM verdict
        can ONLY downgrade admit→veto; it never upgrades a
        deterministic veto to admit.  ``episode_id`` is forwarded to
        the validator for per-episode caching.
        """
        if not skill_id:
            self._stats.n_validate_skipped += 1
            return True, {"status": "no_skill_id_supplied"}

        # Block B1: plain-text-skills mode bypasses all post-LLM
        # validation — the actor's pick stands regardless.
        if self._mode == "plain-text-skills":
            self._stats.n_validate_skipped += 1
            return True, {"status": "plain_text_skills_bypass", "skill_id": skill_id}

        rec = self._records.get(skill_id)
        if rec is None:
            self._stats.n_validate_skipped += 1
            d = {"status": "skill_not_in_cache", "skill_id": skill_id}
            self._stats.last_validate = d
            return True, d

        try:
            res: ValidateInvocationResult = self._harness.validate_invocation(
                rec, state, bindings=bindings or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "harness_hook.validate_choice: validate_invocation raised %s — "
                "degrading to admit", exc,
            )
            self._stats.n_validate_skipped += 1
            return True, {"status": "harness_error", "error": repr(exc)}

        d = res.to_json()
        d["status"] = "ok" if res.ok else "veto"
        d["skill_id"] = skill_id
        if res.ok:
            self._stats.n_validate_ok += 1
        else:
            self._stats.n_validate_veto += 1

        # ── Path 4 — supplemental LLM second-pass on a deterministic
        # admit only.  One-way: the LLM can downgrade admit→veto but
        # cannot upgrade a deterministic veto.
        if res.ok and self._llm_validator is not None:
            try:
                outcome = self._llm_validator.validate(
                    episode_id=episode_id or "",
                    skill=rec, state=state,
                    deterministic_diag=d,
                )
                d.update(outcome.to_diag())
                if not outcome.ok:
                    # LLM downgrade — flip the verdict.
                    self._stats.n_validate_ok -= 1
                    self._stats.n_validate_veto += 1
                    d["status"] = "veto"
                    d["veto_class"] = "llm_downgrade"
                    self._stats.last_validate = dict(d)
                    return False, d
            except Exception as exc:  # noqa: BLE001
                # Validator already absorbs every internal exception
                # — this branch is the belt-and-braces guard against
                # future regressions.  We log at debug level (loud
                # logging would make the failure mode hard to ignore
                # but easy to drown in) and proceed with the
                # deterministic admit.
                logger.debug(
                    "harness_hook.validate_choice: llm validator raised "
                    "(skill=%s err=%s) — kept deterministic admit",
                    skill_id, exc,
                )

        self._stats.last_validate = dict(d)

        # Block A2: stream the per-event validate_invocation diagnostic
        # to the reviewer-facing JSONL log.  Without this row the
        # binding/precondition/evidence/adapter booleans + missing-
        # binding lists vanish at episode end (consumed transiently by
        # Phase B' / GRPO record builder).
        try:
            log_harness_validate(
                step=self._trainer_step,
                episode_id=str(episode_id or ""),
                game=self._game,
                inner_step=int(inner_step),
                skill_id=skill_id,
                ok=bool(res.ok),
                binding_ok=bool(getattr(res, "binding_ok", False)),
                precondition_ok=bool(getattr(res, "precondition_ok", False)),
                evidence_ok=bool(getattr(res, "evidence_ok", False)),
                adapter_ok=bool(getattr(res, "adapter_ok", False)),
                shadow_only=bool(getattr(res, "shadow_only", False)),
                veto_reasons=list(getattr(res, "veto_reasons", []) or []),
                missing_bindings=list(getattr(res, "missing_bindings", []) or []),
                missing_evidence_in=list(getattr(res, "missing_evidence_in", []) or []),
                failed_preconditions=list(getattr(res, "failed_preconditions", []) or []),
            )
        except Exception as _logexc:  # noqa: BLE001
            logger.debug(
                "harness_hook: log_harness_validate raised %s", _logexc,
            )

        return res.ok, d

    # ── Drainage ─────────────────────────────────────────────────────

    def flush_to_lifecycle(
        self,
        lifecycle: Any,
        *,
        min_count: int = 1,
        reset: bool = True,
    ) -> FlushReport:
        """Drain the in-memory rejection sink into ``lifecycle``.

        Wraps :meth:`harness.RejectedSkillSink.flush_to`. Skills that
        the lifecycle doesn't know are skipped (and reported in the
        ``FlushReport``) — this is expected because the per-step hook
        sees the *runtime view* (PROVISIONAL records minted from
        bank.jsonl) while the Crafter hook's lifecycle holds an
        ephemeral ``SkillRepository`` seeded with CANDIDATE records;
        the two share skill_ids but the lifecycle owns the
        ``record_false_binding_pattern`` write surface.
        """
        try:
            return self._sink.flush_to(
                lifecycle, min_count=min_count, reset=reset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "harness_hook.flush_to_lifecycle: flush raised %s", exc,
            )
            return FlushReport(
                n_skills_touched=0, n_patterns_written=0, n_errors=1,
                skipped_unknown_skill_ids=[],
                errors=[{"error": repr(exc)}],
            )


__all__ = [
    "HarnessStepStats",
    "SkillHarnessHook",
]
