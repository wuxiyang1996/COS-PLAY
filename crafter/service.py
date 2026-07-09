"""`SkillCrafterService` — the crafter's outward-facing API.

Spec: PLAN-SKILL-CRAFTER §4 + PLAN-COMPONENTS-IMPLEMENTATION §4 +
``implementation_notes/legacy/crafter-harness-orchestrator-roles.md`` §"Two-tier
trigger model".

This is the **only** module the orchestrator calls into for crafter
work. It bundles `Composer`, `Generalizer`, `Hypothesizer`, the failure
diagnoser, and the failure memory, and ensures *every* output:

  1. is persisted to the artifact store as a typed proposal,
  2. is materialized as a DRAFT `SkillRecord` via the lifecycle manager,
  3. carries provenance (`parent_skill_ids`, `proposal_id`,
     `seed_failure_ids`).

The service does NOT decide what to do with the proposal next — that's
the gate's job. It does NOT mutate any active store.

Two-tier trigger model
----------------------
The Crafter is invoked at two cadences (see implementation note for
the rationale):

* **Per-episode reactive pass** — :meth:`reflect_on_episode`. Fires
  immediately after the Skill Bank Agent has produced this episode's
  candidates / bank-mgmt updates. Runs:

    - Failure-Reflector (every failure in this episode, threshold=1);
    - per-episode Hypothesizer (fall-through when no base skill matches);
    - Subsumption-retire (a freshly-minted candidate strictly covers an
      existing active skill).

  This pass *does not* run Composer / Generalizer — those need
  multi-episode statistics that are not yet stable after a single
  rollout.

* **Per-batch reflective pass** — :meth:`cycle`. Fires every K episodes
  (orchestrator-scheduled). Runs the original failure-driven dispatch
  with the configured ``hot_pattern_threshold`` (default 3) so only
  patterns observed across multiple episodes are acted on. Composer /
  Generalizer should be wired in here in a follow-up — they are the
  natural fit for the per-batch surface.

Both passes share the same proposer components (``Composer``,
``Generalizer``, ``Hypothesizer``, ``Repairer``, ``FailureDiagnoser``).
The only difference is the *entry point* and the threshold applied to
``FailureMemory.hot_patterns``.

Wider read-scope
----------------
Per the implementation note §"Read scope", per-episode reflection
needs to see candidate skills the Bank Agent just minted (for
subsumption detection) plus optional bank-mgmt action history (for
dedup). The service builds a frozen :class:`crafter._bank_view.BankView`
covering the active / candidate / draft stores and passes it to
proposers as a single read-only argument.

The bank view never leaves the service's process and is not persisted
— it is rebuilt at the start of every reflect / cycle call. Mutations
that ``ingest_draft`` performs during the call do not retroactively
appear in the view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from common.enums import (
    LANE_A_RECOVERY_STRATEGIES,
    RecoveryStrategy,
    SkillStatus,
    SkillType,
)
from common.ids import new_skill_id
from common.models import (
    BACKBONE_TEACHER_MODEL,
    is_frozen_qwen_teacher,
    phase_f_teacher_from_env,
    qwen3_vl_teacher,
)
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace
from data_structure.extensions.skill_record import SkillRecord
from crafter._bank_view import BankView, take_bank_view
from crafter.composer import Composer
from crafter.failure_diagnoser import FailureDiagnoser
from crafter.failure_memory import FailureMemory, FailurePattern
from crafter.generalizer import Generalizer
from crafter.hypothesizer import Hypothesizer
from crafter.repairer import Repairer
from orchestrator.artifact_store import ArtifactStore
from skill_bank.lifecycle import SkillLifecycleManager


@dataclass
class CrafterCycleResult:
    """Result of one Crafter pass — works for both `cycle` and `reflect_on_episode`.

    ``trigger`` distinguishes the two surfaces so downstream audit logs
    can break proposal volume down by per-episode vs per-batch source
    (the implementation note §6.1 calls this out as a required label
    for the gate dashboard).

    ``n_subsumption_retires`` is non-zero only on the per-episode path —
    the per-batch ``cycle()`` does not currently run subsumption checks
    (it has no ``new_candidate_skill_ids`` input) and reports zero.

    ``n_patches_coalesced`` and ``n_patches_skipped_cooldown`` surface
    the two early-training noise filters introduced alongside the
    two-tier trigger:

    * **coalesce** — when a fresh failure for the same
      ``(base_skill_id, recovery_strategy)`` arrives while an earlier
      ``PatchProposal`` is still in DRAFT, the new evidence is
      appended to that proposal's ``seed_failure_ids`` instead of
      minting a duplicate proposal.
    * **cooldown** — when no coalescable open patch exists *and* the
      same base skill was patched within the last
      ``SkillCrafterService.cooldown_passes`` Crafter passes, the
      mint is skipped (the failure still lands in ``FailureMemory``
      so the per-batch ``cycle()`` can pick it up).

    ``proposals`` always counts only newly-minted proposals — coalesces
    do not double-count.
    """

    n_failures_ingested: int
    n_patterns_examined: int
    proposals: List[BankMutationProposal]
    trigger: str = "cycle"                       # "cycle" | "reflect_on_episode"
    episode_id: Optional[str] = None
    n_subsumption_retires: int = 0
    n_patches_coalesced: int = 0
    n_patches_skipped_cooldown: int = 0
    bank_view_summary: dict = field(default_factory=dict)


class SkillCrafterService:
    def __init__(
        self,
        *,
        lifecycle: SkillLifecycleManager,
        artifact_store: ArtifactStore,
        composer: Optional[Composer] = None,
        generalizer: Optional[Generalizer] = None,
        hypothesizer: Optional[Hypothesizer] = None,
        repairer: Optional[Repairer] = None,
        diagnoser: Optional[FailureDiagnoser] = None,
        failure_memory: Optional[FailureMemory] = None,
        teacher_model: Optional[str] = None,
        hot_pattern_threshold: int = 3,
        cooldown_passes: int = 5,
        enable_protocol_patching: bool = False,
        hypothesize_min_recurrences: int = 3,
        hypothesize_related_skill_jaccard: float = 0.30,
    ) -> None:
        self._lifecycle = lifecycle
        self._artifacts = artifact_store
        self._composer = composer or Composer()
        self._generalizer = generalizer or Generalizer()
        self._hypothesizer = hypothesizer or Hypothesizer()
        self._repairer = repairer or Repairer()
        self._diagnoser = diagnoser or FailureDiagnoser()
        self._failures = failure_memory or FailureMemory()
        # Lane-(a) feature flag — see ``implementation_notes/legacy/skill-lane-decision.md``.
        # Default ``False`` parks the Repairer / protocol-edit path: skills
        # are retrieval payloads, not runnable programs, so PatchProposal
        # mints would be edits to a contract no live runtime executes.
        # The dispatcher's existing ``_STATUS_NO_OP`` → Hypothesizer fall-
        # through (``_run_failure_dispatch``) carries the signal through.
        # Set ``True`` only on the offline diagnostic stack
        # (labeling_supplement) or in lane-(b) experiments.
        self._enable_protocol_patching = bool(enable_protocol_patching)
        # Default teacher = project-wide backbone (currently GPT-4o); see
        # `common/models.py`. The 8B / 32B / 72B Qwen tracks plus the
        # Phase-F frozen Qwen3-VL teachers are deferred and may be
        # re-enabled by passing `teacher_model="Qwen/Qwen3-VL-32B"` (or
        # the 235B-A22B variant) — see `common.models.qwen3_vl_teacher`.
        self._teacher = teacher_model or BACKBONE_TEACHER_MODEL
        self._threshold = hot_pattern_threshold
        # ── early-training noise filters (impl-note §"Two-tier trigger") ──
        # Coalesce index: keyed by (base_skill_id, recovery_strategy),
        # caches the in-flight DRAFT PatchProposal so subsequent
        # failures with the same shape append evidence instead of
        # minting a parallel proposal. The cached proposal is the live
        # PatchProposal object (mutable dataclass); we re-`put_proposal`
        # to overwrite the artifact-store JSON each time we coalesce.
        self._open_patches: Dict[Tuple[str, str], Tuple[PatchProposal, str]] = {}
        # Per-base cooldown: `last pass index at which we minted a
        # patch for this base`. Cooldown is checked only when
        # `_open_patches` does NOT contain a coalescable entry — a
        # coalesce is always preferred over either minting or skipping.
        self._patch_last_pass: Dict[str, int] = {}
        # Monotonic counter advanced once per `cycle()` /
        # `reflect_on_episode()` call. Used as the unit of cooldown.
        self._pass_counter: int = 0
        self._cooldown_passes = max(0, cooldown_passes)
        # ── hypothesizer fallthrough gate (added 2026-05-04 after v11 audit) ──
        # The dispatch chain (patch → retire → hypothesize) treats
        # hypothesize as last-resort by design, but the trigger conditions
        # were too loose: a single orphan failure (``pattern.skill_id is
        # None``) bypassed the patch path entirely and minted an empty
        # placeholder skill on every episode, polluting the bank with
        # 73-85% boilerplate "hypothesis__prop-..." records and
        # collapsing contract GRPO's effect-literal learning rate from
        # 70-85% (5/3 baseline) to 6%. The two gates below restore the
        # architectural intent that hypothesize fires only when:
        #   (a) the same failure pattern has recurred ≥ N times within
        #       the FailureMemory window (``hypothesize_min_recurrences``),
        #       AND
        #   (b) no existing bank skill is plausibly related to the
        #       failure context (Jaccard token overlap with skill names
        #       and strategic_descriptions below
        #       ``hypothesize_related_skill_jaccard``).
        # Rationale belongs in service.py (the dispatcher) so both
        # ``cycle()`` and ``reflect_on_episode`` inherit the same gate
        # without each surface having to remember to apply it.
        self._hypothesize_min_recurrences = max(1, int(hypothesize_min_recurrences))
        self._hypothesize_related_jaccard = float(hypothesize_related_skill_jaccard)

    # -- phase-F frozen teacher swap -------------------------------------

    @property
    def teacher_model(self) -> str:
        """The frozen-teacher backbone the crafter stamps on every proposal."""
        return self._teacher

    @property
    def enable_protocol_patching(self) -> bool:
        """Whether the Repairer / PatchProposal mint path is live.

        Lane (a) — skills are retrieval payloads — defaults to ``False``;
        lane (b) / offline diagnostic drivers may pass ``True``. See
        ``implementation_notes/legacy/skill-lane-decision.md`` for the verdict.
        """
        return self._enable_protocol_patching

    @property
    def is_phase_f_active(self) -> bool:
        """True iff the active teacher is one of the Phase-F frozen Qwen3-VL teachers."""
        return is_frozen_qwen_teacher(self._teacher)

    def set_teacher_model(self, model: str) -> None:
        """Swap the frozen-teacher backbone in place.

        Phase-F entry point — call with
        ``qwen3_vl_teacher("32b")`` or ``qwen3_vl_teacher("235b-a22b")``
        to flip the crafter's teacher without rebuilding the service.
        Existing component LLM hooks (set on `FailureDiagnoser`,
        `Hypothesizer`, `Repairer`) are preserved; this only changes
        which model name gets stamped on emitted proposals.
        """
        if not model:
            raise ValueError("teacher_model must be a non-empty string")
        self._teacher = model

    @classmethod
    def with_qwen3_vl_teacher(
        cls,
        *,
        lifecycle: SkillLifecycleManager,
        artifact_store: ArtifactStore,
        size: str = "32b",
        **kwargs,
    ) -> "SkillCrafterService":
        """Phase-F constructor — instantiate with a frozen Qwen3-VL teacher.

        Equivalent to passing
        ``teacher_model=qwen3_vl_teacher(size)`` explicitly; provided
        as a one-liner so deployment scripts don't need to import the
        ``common.models`` helpers themselves.
        """
        kwargs.setdefault("teacher_model", qwen3_vl_teacher(size))
        return cls(lifecycle=lifecycle, artifact_store=artifact_store, **kwargs)

    @classmethod
    def from_env(
        cls,
        *,
        lifecycle: SkillLifecycleManager,
        artifact_store: ArtifactStore,
        **kwargs,
    ) -> "SkillCrafterService":
        """Construct the service, honoring the Phase-F env switch.

        Reads ``VLM_AGENT_PHASE_F_TEACHER`` via
        :func:`common.models.phase_f_teacher_from_env`.  When set, the
        env value (e.g. ``qwen3-vl-32b``) overrides the default
        ``BACKBONE_TEACHER_MODEL``; otherwise behaviour matches the
        plain constructor.
        """
        phase_f = phase_f_teacher_from_env()
        if phase_f is not None:
            kwargs.setdefault("teacher_model", phase_f)
        return cls(lifecycle=lifecycle, artifact_store=artifact_store, **kwargs)

    # -- explicit invocations --------------------------------------------

    def propose_composition(
        self,
        components: Iterable[SkillRecord],
        *,
        name: str,
        rationale: str,
        target_domains: Optional[List[str]] = None,
    ) -> BankMutationProposal:
        proposal = self._composer.compose(
            components=components,
            name=name,
            rationale=rationale,
            target_domains=target_domains,
            teacher_model=self._teacher,
        )
        self._persist(proposal, skill_type=SkillType.MIXED, name=name)
        return proposal

    def propose_generalization(
        self,
        base: SkillRecord,
        *,
        new_domains: Iterable[str],
        rationale: str,
    ) -> BankMutationProposal:
        proposal = self._generalizer.generalize(
            base=base,
            new_domains=new_domains,
            rationale=rationale,
            teacher_model=self._teacher,
        )
        self._persist(proposal, skill_type=base.skill_type, name=proposal.name)
        return proposal

    def propose_retirement(self, skill_id: str, *, reason: str) -> BankMutationProposal:
        proposal = RetireProposal(
            target_skill_id=skill_id,
            rationale=reason,
            reason=reason,
            proposed_at=time.time(),
        )
        self._artifacts.put_proposal(proposal)
        self._artifacts.append_audit(
            {"kind": "proposal", "type": "RetireProposal", "proposal_id": proposal.proposal_id, "target_skill_id": skill_id}
        )
        return proposal

    def propose_repair(
        self,
        *,
        base_skill_id: Optional[str] = None,
        base: Optional[SkillRecord] = None,
        pattern_id: Optional[str] = None,
        pattern: Optional[FailurePattern] = None,
        diagnosis: Optional[FailureDiagnosis] = None,
        rationale: Optional[str] = None,
    ) -> Optional[BankMutationProposal]:
        """Phase-D entry point — emit a `PatchProposal` for a known skill.

        Resolves the base skill (via the lifecycle manager's read-through
        `get`), the failure pattern (via `FailureMemory`), and a
        diagnosis (via `FailureDiagnoser`) when not supplied, then asks
        the `Repairer` to build a `PatchProposal`. The proposal lands as
        a DRAFT `SkillRecord` whose `parent_skill_ids = [base.skill_id]`
        and whose `content_hash` differs from the base, so the gate
        revalidates from scratch (PLAN-UNIFIED-SKILL-GATE §3.2).

        Coalesce: when a still-DRAFT ``PatchProposal`` already exists
        for ``(base.skill_id, recovery_strategy)`` (registered in
        ``self._open_patches``), the new failure ids are appended to
        that proposal's ``seed_failure_ids`` and the artifact-store
        JSON is overwritten in place. In that case this method
        returns ``None`` (no NEW proposal was minted; the caller can
        consult ``self._open_patches`` if it needs the running
        proposal handle).

        Returns ``None`` when the diagnosis recommends retirement
        instead — in that case the caller (or the cycle loop) routes to
        `propose_retirement`.
        """
        proposal, _status = self._propose_repair_internal(
            base_skill_id=base_skill_id,
            base=base,
            pattern_id=pattern_id,
            pattern=pattern,
            diagnosis=diagnosis,
            rationale=rationale,
        )
        return proposal

    # -- internal patch routing (mint / coalesce / no-op) ----------------

    # Status values returned alongside the proposal so the dispatch loop
    # in `_run_failure_dispatch` can count `n_patches_coalesced` /
    # `n_patches_skipped_cooldown` without having to peek at internal
    # state. `propose_repair` (public) discards the status and only
    # surfaces the freshly-minted proposal.
    _STATUS_MINTED = "minted"
    _STATUS_COALESCED = "coalesced"
    _STATUS_NO_OP = "no_op"           # Repairer / diagnoser short-circuit

    def _propose_repair_internal(
        self,
        *,
        base_skill_id: Optional[str] = None,
        base: Optional[SkillRecord] = None,
        pattern_id: Optional[str] = None,
        pattern: Optional[FailurePattern] = None,
        diagnosis: Optional[FailureDiagnosis] = None,
        rationale: Optional[str] = None,
    ) -> Tuple[Optional[PatchProposal], str]:
        # Lane-(a) gate (T1.3a). When protocol patching is disabled, the
        # Repairer never mints a PatchProposal and the dispatcher's
        # `_STATUS_NO_OP` fall-through routes the signal to the
        # Hypothesizer instead — the bank-gap response under the
        # "skill = retrieval payload" lane.
        if not self._enable_protocol_patching:
            return None, self._STATUS_NO_OP

        base = base or self._resolve_base(base_skill_id)
        if base is None:
            raise ValueError(
                "propose_repair requires a base SkillRecord or a base_skill_id "
                "that resolves through the lifecycle manager."
            )

        pattern = pattern or self._resolve_pattern(pattern_id)
        if pattern is None:
            raise ValueError(
                "propose_repair requires a FailurePattern or a pattern_id "
                "that resolves through FailureMemory."
            )

        if diagnosis is None:
            diagnosis = self._diagnose_pattern(pattern)
            if diagnosis is None:
                # No representative trace available — cannot repair safely.
                return None, self._STATUS_NO_OP

        # Retirement is dispatched by the caller; propose_repair never
        # mints a Patch when the diagnosis recommends retirement.
        if diagnosis.recommended_strategy == RecoveryStrategy.SKILL_RETIREMENT:
            return None, self._STATUS_NO_OP

        # ── coalesce check ──────────────────────────────────────────
        coalesce_key = (base.skill_id, diagnosis.recommended_strategy.value)
        existing = self._lookup_open_patch(coalesce_key)
        if existing is not None:
            self._coalesce_open_patch(existing, pattern, base=base)
            return None, self._STATUS_COALESCED

        # ── mint fresh ──────────────────────────────────────────────
        proposal = self._repairer.repair(
            base=base,
            pattern=pattern,
            diagnosis=diagnosis,
            teacher_model=self._teacher,
            rationale=rationale,
        )
        if proposal is None:
            return None, self._STATUS_NO_OP
        draft = self._persist(proposal, skill_type=base.skill_type, name=f"{base.name}__patched")
        if draft is not None:
            self._open_patches[coalesce_key] = (proposal, draft.skill_id)
        # NB: `_patch_last_pass` is updated by the dispatch caller, not
        # here, so direct callers of `propose_repair` (tests / scripts)
        # don't pollute the cooldown index.
        return proposal, self._STATUS_MINTED

    def _lookup_open_patch(
        self,
        key: Tuple[str, str],
    ) -> Optional[Tuple[PatchProposal, str]]:
        """Return the cached open patch for ``key`` if its DRAFT skill
        is still alive; lazily evict otherwise.

        A draft skill leaves DRAFT only via the orchestrator (gate
        decision: promote → CANDIDATE / archive → ARCHIVE). The Crafter
        gets no notification, so we re-check the lifecycle on every
        coalesce attempt and drop the cache entry the moment the draft
        moves on.
        """
        existing = self._open_patches.get(key)
        if existing is None:
            return None
        _proposal, draft_skill_id = existing
        record = self._lifecycle.get(draft_skill_id)
        if record is None or record.status != SkillStatus.DRAFT:
            del self._open_patches[key]
            return None
        return existing

    def _coalesce_open_patch(
        self,
        existing: Tuple[PatchProposal, str],
        pattern: FailurePattern,
        *,
        base: SkillRecord,
    ) -> None:
        """Append ``pattern.failure_ids`` to the cached proposal's
        ``seed_failure_ids`` (deduped, order-preserving), restamp
        ``proposed_at``, overwrite the artifact-store JSON, and append
        an audit event so the dashboard can attribute proposal volume
        to coalesces vs. fresh mints."""
        proposal, draft_skill_id = existing
        before_n = len(proposal.seed_failure_ids)
        seen = set(proposal.seed_failure_ids)
        for fid in pattern.failure_ids:
            if fid not in seen:
                proposal.seed_failure_ids.append(fid)
                seen.add(fid)
        proposal.proposed_at = time.time()
        # Overwrite the artifact-store JSON in place (put_proposal is
        # keyed by proposal_id and atomically rewrites the same file).
        self._artifacts.put_proposal(proposal)
        self._artifacts.append_audit({
            "kind": "patch_coalesced",
            "proposal_id": proposal.proposal_id,
            "draft_skill_id": draft_skill_id,
            "base_skill_id": base.skill_id,
            "recovery_strategy": proposal.recovery_strategy,
            "pattern_id": pattern.pattern_id,
            "n_seeds_before": before_n,
            "n_seeds_after": len(proposal.seed_failure_ids),
        })

    # -- failure-driven cycle --------------------------------------------

    def ingest_failures(self, traces: Iterable[FailureTrace]) -> int:
        n = 0
        for t in traces:
            self._failures.add(t)
            self._artifacts.put_failure(t)
            n += 1
        return n

    def cycle(
        self,
        *,
        new_failures: Optional[Iterable[FailureTrace]] = None,
    ) -> CrafterCycleResult:
        """Per-batch reflective pass (PLAN-SKILL-CRAFTER §6.5 dispatch).

        Ingests ``new_failures`` (typically the union of failure traces
        across the last K episodes), then dispatches every pattern whose
        count crosses ``self._threshold`` (default 3). This is the
        cadence that keeps the gate stack from being flooded by single-
        episode noise; for per-episode reactive reflection use
        :meth:`reflect_on_episode` instead.
        """
        self._pass_counter += 1
        n_in = self.ingest_failures(new_failures or [])
        hot = self._failures.hot_patterns(min_count=self._threshold)
        proposals, n_coalesced, n_cooldown = self._run_failure_dispatch(hot)
        return CrafterCycleResult(
            n_failures_ingested=n_in,
            n_patterns_examined=len(hot),
            proposals=proposals,
            trigger="cycle",
            n_patches_coalesced=n_coalesced,
            n_patches_skipped_cooldown=n_cooldown,
            bank_view_summary={},
        )

    def reflect_on_episode(
        self,
        reflection: EpisodeReflection,
    ) -> CrafterCycleResult:
        """Per-episode reactive pass (implementation note §"Two-tier trigger").

        Called by the orchestrator immediately after the Skill Bank
        Agent has finished processing one episode. Runs the
        Failure-Reflector + per-episode Hypothesizer subset of the
        Crafter on the failures observed *in this episode only*, with
        ``min_count=1`` (one observation is enough to act on a single
        episode — no aggregation across episodes).

        Also runs subsumption-retire detection: for every freshly-minted
        candidate skill in ``reflection.new_candidate_skill_ids``, check
        whether it strictly subsumes an active skill (per
        :func:`crafter._bank_view._subsumes`); if so, emit a
        ``RetireProposal`` for the active.

        ``Composer`` and ``Generalizer`` are intentionally **not** run
        here — both need batch-level statistics
        (cross-episode co-occurrence, multi-episode pass-rate) that are
        not stable after a single rollout. Use :meth:`cycle` for those.

        Returns a :class:`CrafterCycleResult` with ``trigger`` set to
        ``"reflect_on_episode"``, the bank-view size summary attached,
        and ``n_subsumption_retires`` populated.
        """
        if not isinstance(reflection, EpisodeReflection):
            raise TypeError(
                "reflect_on_episode expects an EpisodeReflection; "
                f"got {type(reflection).__name__}"
            )

        # Short-circuit: a healthy episode (no failures, no fresh
        # candidates) produces zero proposals — the gate stack should
        # never see a Crafter pass with nothing to evaluate. Note: we
        # do NOT advance `_pass_counter` here so cooldowns are measured
        # in productive Crafter passes only (silent passes shouldn't
        # bleed off cooldown that protects against actual churn).
        if not reflection.has_signal:
            view = self._take_bank_view()
            return CrafterCycleResult(
                n_failures_ingested=0,
                n_patterns_examined=0,
                proposals=[],
                trigger="reflect_on_episode",
                episode_id=reflection.episode_id,
                n_subsumption_retires=0,
                n_patches_coalesced=0,
                n_patches_skipped_cooldown=0,
                bank_view_summary=view.size_summary(),
            )

        self._pass_counter += 1
        n_in = self.ingest_failures(reflection.failure_traces)

        # Per-episode dispatch — threshold=1, so any pattern that has at
        # least one failure_id from this batch's ingest gets examined.
        hot = self._failures.hot_patterns(min_count=1)
        # Restrict to patterns whose latest failure came from THIS
        # episode (otherwise re-running the per-episode pass against the
        # accumulated FailureMemory would re-fire on stale patterns).
        episode_failure_ids = {t.failure_id for t in reflection.failure_traces}
        hot = [
            p for p in hot
            if any(fid in episode_failure_ids for fid in p.failure_ids)
        ]

        proposals, n_coalesced, n_cooldown = self._run_failure_dispatch(hot)

        # Subsumption-retire path — uses the wider read-scope (active +
        # candidate). Costs one bank-view build and one per-candidate
        # comparison; no LLM call.
        view = self._take_bank_view()
        n_subsumes = 0
        if reflection.new_candidate_skill_ids:
            for cand_id, active_id, reason in view.subsumed_pairs(
                candidate_ids=reflection.new_candidate_skill_ids
            ):
                proposals.append(
                    self.propose_retirement(
                        active_id,
                        reason=(
                            f"subsumed_by={cand_id} (episode={reflection.episode_id}); "
                            f"{reason}"
                        ),
                    )
                )
                n_subsumes += 1

        return CrafterCycleResult(
            n_failures_ingested=n_in,
            n_patterns_examined=len(hot),
            proposals=proposals,
            trigger="reflect_on_episode",
            episode_id=reflection.episode_id,
            n_subsumption_retires=n_subsumes,
            n_patches_coalesced=n_coalesced,
            n_patches_skipped_cooldown=n_cooldown,
            bank_view_summary=view.size_summary(),
        )

    # -- shared dispatch -------------------------------------------------

    def _run_failure_dispatch(
        self,
        patterns: Iterable[FailurePattern],
    ) -> Tuple[List[BankMutationProposal], int, int]:
        """Run the PLAN-SKILL-CRAFTER §6.5 per-pattern dispatch chain.

        Shared between :meth:`cycle` (per-batch) and
        :meth:`reflect_on_episode` (per-episode); the *only* difference
        between the two surfaces is which patterns reach this method
        and how the result gets labelled. The dispatch order itself
        (repair > retire > hypothesize) is identical because the
        per-pattern decision should not depend on cadence.

        Two early-training noise filters apply uniformly to both
        surfaces:

        * **Coalesce** (always-on): handled inside
          :meth:`_propose_repair_internal` — when an open DRAFT patch
          already covers ``(base.skill_id, recovery_strategy)`` the
          new evidence is appended in place and no new proposal is
          minted. Counted as ``n_coalesced``.
        * **Cooldown** (``self._cooldown_passes`` Crafter passes,
          default 5): if no coalescable open patch exists *and* the
          same base was patched within the cooldown window, the mint
          is skipped. The failure still landed in
          ``FailureMemory.add`` (via the caller's ``ingest_failures``)
          so the per-batch ``cycle()`` can still pick it up later.
          Counted as ``n_cooldown_skipped``.

        Returns ``(proposals, n_coalesced, n_cooldown_skipped)``;
        ``proposals`` only includes newly-minted records (mints +
        retirements + hypotheses), so ``n_coalesced`` does not
        double-count.
        """
        proposals: List[BankMutationProposal] = []
        n_coalesced = 0
        n_cooldown_skipped = 0

        for pattern in patterns:
            diagnosis = self._diagnose_pattern(pattern)
            if diagnosis is None:
                continue

            # Dispatch order (PLAN-SKILL-CRAFTER §6.5):
            #   1. If the failing `pattern.skill_id` resolves to an
            #      existing bank skill → propose a *patch* (Phase D).
            #   2. If the diagnosis recommends retirement → emit a
            #      `RetireProposal` (still gate-bound).
            #   3. Else fall back to the hypothesizer's novel-skill
            #      proposal (the original Phase C path).
            base = self._resolve_base(pattern.skill_id) if pattern.skill_id else None
            if base is not None:
                if diagnosis.recommended_strategy == RecoveryStrategy.SKILL_RETIREMENT:
                    proposals.append(
                        self.propose_retirement(
                            base.skill_id,
                            reason=diagnosis.root_cause or "persistent failure pattern",
                        )
                    )
                    continue

                # T1.3c — lane-(a) retrieval-centric strategies are NOT
                # protocol edits; the Repairer is skipped entirely and
                # the dispatch falls through to the Hypothesizer below
                # (the bank-gap response under the "skill = retrieval
                # payload" lane). This is the live-trainer behaviour
                # regardless of whether ``enable_protocol_patching`` is
                # True or False — protocol patching only governs the
                # *protocol-edit* taxonomy, not the new lane-(a)
                # signals.
                if diagnosis.recommended_strategy in LANE_A_RECOVERY_STRATEGIES:
                    self._artifacts.append_audit({
                        "kind": "lane_a_dispatch",
                        "base_skill_id": base.skill_id,
                        "recovery_strategy": diagnosis.recommended_strategy.value,
                        "pattern_id": pattern.pattern_id,
                        "primary_route": "hypothesizer",
                    })
                    # Fall through to the hypothesizer block below.
                else:
                    # Coalesce always wins over cooldown — appending
                    # evidence to an in-flight DRAFT is free and is what
                    # the gate wants to see (richer seed_failure_ids =
                    # better diagnosis quality).
                    coalesce_key = (base.skill_id, diagnosis.recommended_strategy.value)
                    has_open = self._lookup_open_patch(coalesce_key) is not None

                    if not has_open and self._is_under_cooldown(base.skill_id):
                        n_cooldown_skipped += 1
                        self._artifacts.append_audit({
                            "kind": "patch_skipped_cooldown",
                            "base_skill_id": base.skill_id,
                            "recovery_strategy": coalesce_key[1],
                            "pattern_id": pattern.pattern_id,
                            "passes_since_last_patch": (
                                self._pass_counter
                                - self._patch_last_pass.get(base.skill_id, 0)
                            ),
                            "cooldown_passes": self._cooldown_passes,
                        })
                        continue

                    patch, status = self._propose_repair_internal(
                        base=base, pattern=pattern, diagnosis=diagnosis
                    )
                    if status == self._STATUS_MINTED and patch is not None:
                        proposals.append(patch)
                        self._patch_last_pass[base.skill_id] = self._pass_counter
                        continue
                    if status == self._STATUS_COALESCED:
                        n_coalesced += 1
                        continue
                    # _STATUS_NO_OP — Repairer/diagnoser short-circuited.
                    # Fall through to the hypothesizer below as the original
                    # PLAN-SKILL-CRAFTER §6.5 dispatch chain prescribes.

            # ── Hypothesizer fallthrough gate (post-v11 fix) ──
            # Hypothesize is intended as a last-resort, not a per-episode
            # default. Two conditions must hold before falling through:
            #
            #   1. ``pattern.count >= self._hypothesize_min_recurrences``
            #      The same failure pattern has recurred ≥ N times. A
            #      single orphan failure (e.g. one episode death) is
            #      noise; minting a brand-new skill from it is what
            #      caused 60-90 hypothesis stubs per phase in v11.
            #
            #   2. No existing bank skill is plausibly related to the
            #      failure context (token Jaccard < threshold against
            #      every active/candidate skill's name + description).
            #      If a related skill exists, the right answer is to
            #      patch it (or retire and re-mint via segmentation),
            #      not invent a parallel placeholder.
            #
            # Both gates default to "open" semantics — a fresh
            # ``FailureMemory`` always counts >= 1, and an empty bank
            # has no related skills — so cold-start stays expressive.
            if pattern.count < self._hypothesize_min_recurrences:
                self._artifacts.append_audit({
                    "kind": "hypothesize_skipped_recurrence_gate",
                    "pattern_id": pattern.pattern_id,
                    "pattern_count": pattern.count,
                    "min_recurrences": self._hypothesize_min_recurrences,
                })
                continue

            if self._has_related_bank_skill(pattern, diagnosis):
                self._artifacts.append_audit({
                    "kind": "hypothesize_skipped_related_skill_gate",
                    "pattern_id": pattern.pattern_id,
                    "pattern_count": pattern.count,
                    "jaccard_threshold": self._hypothesize_related_jaccard,
                })
                continue

            hypothesis = self._hypothesizer.propose(
                pattern=pattern,
                diagnosis=diagnosis,
                teacher_model=self._teacher,
            )
            if hypothesis is None:
                continue
            self._persist(hypothesis, skill_type=SkillType.MIXED, name=hypothesis.name)
            proposals.append(hypothesis)
        return proposals, n_coalesced, n_cooldown_skipped

    # -- hypothesizer fallthrough gate helpers (post-v11 fix) -----------------

    @staticmethod
    def _tokenize_for_relatedness(text: Any) -> set:
        """Lowercase token set used for related-skill Jaccard.

        Mirrors :func:`skill_agents.query._tokenize` (length-2 alnum)
        so the gate's relatedness signal is consistent with the
        actor's downstream selection signal.
        """
        if not text:
            return set()
        import re as _re
        return {
            w for w in _re.split(r"[^a-zA-Z0-9]+", str(text).lower())
            if len(w) >= 2
        }

    def _has_related_bank_skill(
        self,
        pattern: "FailurePattern",
        diagnosis: Optional["FailureDiagnosis"],
    ) -> bool:
        """Return True iff some active/candidate skill plausibly covers
        the failure context (Jaccard token overlap >= threshold).

        We score against the union of the pattern's signature, the
        diagnosis's root cause, and any free-text from the latest
        failure trace — that's the same evidence the hypothesizer
        itself sees, so the gate's notion of "related" matches the
        proposer's.
        """
        if self._hypothesize_related_jaccard <= 0.0:
            return False  # gate disabled by configuration

        # Build a token signature for the failure context.
        ctx_tokens: set = set()
        ctx_tokens |= self._tokenize_for_relatedness(getattr(pattern, "signature", ""))
        ctx_tokens |= self._tokenize_for_relatedness(getattr(pattern, "skill_id", ""))
        if diagnosis is not None:
            ctx_tokens |= self._tokenize_for_relatedness(
                getattr(diagnosis, "root_cause", "")
            )
        if pattern.failure_ids:
            trace = self._failures.trace(pattern.failure_ids[-1])
            if trace is not None:
                ctx_tokens |= self._tokenize_for_relatedness(
                    getattr(trace, "abort_reason", "")
                )
                ctx_tokens |= self._tokenize_for_relatedness(
                    getattr(trace, "contract_violation", "")
                )

        if len(ctx_tokens) < 2:
            # Too little signal to make a defensible call — let the
            # hypothesizer through (we still have the recurrence gate).
            return False

        # Walk every active + candidate skill; bail out as soon as we
        # find a related match. ``BankView.all_iter`` covers active +
        # candidate + draft (the three stores any actor / segmenter
        # could have inserted into).
        view = self._take_bank_view()
        for rec in view.all_iter():
            sk_tokens = (
                self._tokenize_for_relatedness(getattr(rec, "skill_id", ""))
                | self._tokenize_for_relatedness(getattr(rec, "name", ""))
                | self._tokenize_for_relatedness(
                    getattr(rec, "strategic_description", "")
                )
            )
            if not sk_tokens:
                continue
            inter = len(ctx_tokens & sk_tokens)
            if inter == 0:
                continue
            jaccard = inter / len(ctx_tokens | sk_tokens)
            if jaccard >= self._hypothesize_related_jaccard:
                return True
        return False

    def _is_under_cooldown(self, base_skill_id: str) -> bool:
        if self._cooldown_passes <= 0:
            return False
        last = self._patch_last_pass.get(base_skill_id)
        if last is None:
            return False
        return (self._pass_counter - last) < self._cooldown_passes

    def _take_bank_view(self) -> BankView:
        """Build a frozen read-only multi-store snapshot.

        The service is the only crafter component that may construct a
        ``BankView`` (architectural invariant 2 — the lifecycle manager
        is held only here). Proposers receive the view as a parameter
        when they need cross-store visibility.
        """
        return take_bank_view(self._lifecycle.repository)

    # -- internals --------------------------------------------------------

    def _resolve_base(self, skill_id: Optional[str]) -> Optional[SkillRecord]:
        if not skill_id:
            return None
        return self._lifecycle.get(skill_id)

    def _resolve_pattern(self, pattern_id: Optional[str]) -> Optional[FailurePattern]:
        if not pattern_id:
            return None
        return self._failures.pattern(pattern_id)

    def _diagnose_pattern(self, pattern: FailurePattern) -> Optional[FailureDiagnosis]:
        if not pattern.failure_ids:
            return None
        trace = self._failures.trace(pattern.failure_ids[-1])
        if trace is None:
            return None
        return self._diagnoser.diagnose(trace)


    def _persist(
        self,
        proposal: BankMutationProposal,
        *,
        skill_type: SkillType,
        name: str,
    ) -> Optional[SkillRecord]:
        """Persist a freshly-minted proposal and materialize its DRAFT.

        Returns the materialized DRAFT ``SkillRecord`` (so callers like
        :meth:`_propose_repair_internal` can register it in the
        coalesce index). Returns ``None`` when the proposal type does
        not produce a draft (e.g. ``RetireProposal``).
        """
        self._artifacts.put_proposal(proposal)
        skill = self._proposal_to_draft(proposal, skill_type=skill_type, name=name)
        if skill is not None:
            self._lifecycle.ingest_draft(skill)
        self._artifacts.append_audit(
            {
                "kind": "proposal",
                "type": type(proposal).__name__,
                "proposal_id": proposal.proposal_id,
                "draft_skill_id": skill.skill_id if skill else None,
            }
        )
        return skill

    def _proposal_to_draft(
        self,
        proposal: BankMutationProposal,
        *,
        skill_type: SkillType,
        name: str,
    ) -> Optional[SkillRecord]:
        if isinstance(proposal, ComposeProposal):
            protocol = proposal.composed_protocol
            contract = proposal.contract
            domains = proposal.target_domains
        elif isinstance(proposal, GeneralizeProposal):
            protocol = proposal.abstracted_protocol
            contract = proposal.contract
            domains = proposal.target_domains
        elif isinstance(proposal, HypothesisProposal):
            protocol = proposal.novel_protocol
            contract = proposal.contract
            domains = proposal.target_domains
        elif isinstance(proposal, PatchProposal):
            protocol = proposal.patched_protocol
            contract = proposal.patched_contract  # may be None; gate Stage 0 will catch
            domains = proposal.target_domains
        elif isinstance(proposal, RetireProposal):
            return None
        else:
            return None
        return SkillRecord.new(
            name=name or "unnamed",
            skill_type=skill_type,
            source_type=proposal.source_type,
            feasible_domains=list(domains),
            protocol=protocol,
            contract=contract or None,  # type: ignore[arg-type]
            proposal_id=proposal.proposal_id,
            parent_skill_ids=list(proposal.parent_skill_ids),
        )


__all__ = ["CrafterCycleResult", "SkillCrafterService"]
