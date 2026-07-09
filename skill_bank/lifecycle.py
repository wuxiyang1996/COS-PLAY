"""`SkillLifecycleManager` — sole authority for skill status transitions.

Spec: PLAN-UNIFIED-SKILL-GATE §6.

This is the only module in the codebase that may write to a `SkillStore`.
It enforces the canonical state-machine and physically migrates records
between stores on each transition (DRAFT → CANDIDATE moves the JSON file
from `draft_store/` to `candidate_store/`).

External callers go through this method exclusively; direct
`store.put(...)` calls raise `StoreLockedError`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence

from common.enums import (
    DOMAINS,
    SOURCE_DOMAINS,
    TRANSFER_TARGET_DOMAINS,
    SkillStatus,
)
from data_structure.extensions.skill_record import SkillRecord
from skill_bank.repository import SkillRepository
from skill_bank.stores import SkillStore, StoreName, store_for_status


class LifecycleError(RuntimeError):
    """Raised on disallowed transitions or invariant violations."""


# Allowed transitions (PLAN-UNIFIED-SKILL-GATE §2.3).
_ALLOWED: Dict[SkillStatus, FrozenSet[SkillStatus]] = {
    SkillStatus.DRAFT: frozenset({SkillStatus.CANDIDATE, SkillStatus.REJECTED}),
    SkillStatus.CANDIDATE: frozenset(
        {SkillStatus.SHADOW, SkillStatus.PROVISIONAL, SkillStatus.ACTIVE, SkillStatus.REJECTED, SkillStatus.DEPRECATED}
    ),
    SkillStatus.SHADOW: frozenset(
        {SkillStatus.PROVISIONAL, SkillStatus.ACTIVE, SkillStatus.REJECTED, SkillStatus.DEPRECATED}
    ),
    SkillStatus.PROVISIONAL: frozenset(
        {SkillStatus.ACTIVE, SkillStatus.SHADOW, SkillStatus.DEPRECATED, SkillStatus.ROLLED_BACK}
    ),
    SkillStatus.ACTIVE: frozenset({SkillStatus.DEPRECATED, SkillStatus.ROLLED_BACK, SkillStatus.PROVISIONAL}),
    SkillStatus.DEPRECATED: frozenset({SkillStatus.ROLLED_BACK}),
    SkillStatus.REJECTED: frozenset(),
    SkillStatus.ROLLED_BACK: frozenset({SkillStatus.DEPRECATED}),
}


class SkillLifecycleManager:
    """The only module allowed to mutate skill stores.

    Construct with one repo; the manager owns the cross-store migration
    transactions. All transitions are serialized by a single mutex —
    bank state is always internally consistent at quiesce.
    """

    def __init__(
        self,
        repository: SkillRepository,
        *,
        min_retrievals_per_skill: int = 0,
    ) -> None:
        """Construct a lifecycle manager.

        Args:
            repository: the skill repository this manager owns.
            min_retrievals_per_skill: lane-(a) ACTIVE invariant
                (T1.3d). When > 0, ACTIVE transitions require
                ``record.metrics["retrievals"] >=
                min_retrievals_per_skill``. Defaults to ``0`` (no
                enforcement) so unit tests and standalone drivers
                are unaffected; the orchestrator wires the configured
                threshold from
                ``OrchestratorConfig.gate_thresholds.min_retrievals_per_skill``.
        """

        self._repo = repository
        self._mutex = threading.RLock()
        self._token: object = object()
        self._min_retrievals_per_skill = max(0, int(min_retrievals_per_skill))

    # -- read-through helpers -------------------------------------------

    @property
    def repository(self) -> SkillRepository:
        return self._repo

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        return self._repo.get(skill_id)

    # -- ingest ---------------------------------------------------------

    def ingest_draft(self, record: SkillRecord) -> SkillRecord:
        """Insert a new skill at status=DRAFT.

        Used by the crafter and by mining flows. Always lands in the
        draft store regardless of any status the caller set on the record.
        """
        with self._mutex:
            if self._repo.get(record.skill_id) is not None:
                raise LifecycleError(f"Skill {record.skill_id!r} already exists.")
            record.status = SkillStatus.DRAFT
            self._write(self._repo.draft, record)
            return record

    # -- transition ------------------------------------------------------

    def transition(
        self,
        skill_id: str,
        *,
        to_status: SkillStatus,
        rationale: str,
    ) -> SkillRecord:
        with self._mutex:
            record = self._repo.get(skill_id)
            if record is None:
                raise LifecycleError(f"Unknown skill {skill_id!r}")
            if to_status not in _ALLOWED.get(record.status, frozenset()):
                raise LifecycleError(
                    f"Disallowed transition {record.status.value} -> {to_status.value} "
                    f"for skill {skill_id!r}."
                )
            self._validate_invariants(record, to_status, rationale)

            from_store = self._store_for(record.status)
            to_store = self._store_for(to_status)

            from_status = record.status

            with from_store._unlocked(self._token):
                from_store.remove(record.skill_id, token=self._token)
            record.status = to_status
            with to_store._unlocked(self._token):
                to_store.put(record, token=self._token)

            # Block A3: stream lifecycle transition to reviewer-facing
            # JSONL log so §5.3 skill-lifetime distribution / promotion
            # / deprecation curves can be reconstructed post-hoc.
            # Best-effort, non-fatal on import / I/O failure.
            try:
                from trainer.coevolution._run_loggers import (  # noqa: WPS433
                    log_lifecycle_transition,
                )
                log_lifecycle_transition(
                    skill_id=skill_id,
                    from_status=from_status.value,
                    to_status=to_status.value,
                    reason=str(rationale or ""),
                )
            except Exception:  # noqa: BLE001
                pass

            return record

    # -- batch promotion (used by PromotionOrchestrator) -----------------

    def transition_many(
        self,
        transitions: Iterable[tuple[str, SkillStatus, str]],
    ) -> Dict[str, SkillRecord]:
        """Apply multiple transitions atomically. Either all succeed or all
        are rolled back."""
        with self._mutex:
            staged: list[tuple[SkillStatus, SkillRecord]] = []
            try:
                for skill_id, to_status, rationale in transitions:
                    record = self._repo.get(skill_id)
                    if record is None:
                        raise LifecycleError(f"Unknown skill {skill_id!r}")
                    staged.append((record.status, record))
                # Re-resolve and apply (transition() re-validates each).
                result: Dict[str, SkillRecord] = {}
                for skill_id, to_status, rationale in transitions:
                    result[skill_id] = self.transition(
                        skill_id, to_status=to_status, rationale=rationale
                    )
                return result
            except Exception:
                # Best-effort rollback: revert any record whose status changed.
                for original_status, record in staged:
                    if record.status == original_status:
                        continue
                    cur_store = self._store_for(record.status)
                    orig_store = self._store_for(original_status)
                    with cur_store._unlocked(self._token):
                        cur_store.remove(record.skill_id, token=self._token)
                    record.status = original_status
                    with orig_store._unlocked(self._token):
                        orig_store.put(record, token=self._token)
                raise

    # -- few-shot transfer bookkeeping (PLAN-UNIFIED-SKILL-GATE Stage 3a) --

    def record_false_binding_pattern(
        self,
        skill_id: str,
        *,
        veto: str,
        veto_reason: str,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        observed_at: Optional[float] = None,
        max_patterns: int = 64,
    ) -> SkillRecord:
        """Day-9c (PLAN-SKILL-BANK §4.3b): append a `RejectedSkill` to
        the skill's `false_binding_patterns` list.

        The eligibility filter's `RejectedSkill` channel records *why*
        a candidate was excluded (`veto`, `veto_reason`, plus the
        per-check booleans). This function aggregates those into the
        durable `SkillRecord.false_binding_patterns` list the Crafter
        reads to surface "this skill keeps getting vetoed for reason
        X — patch it / retire it" hot patterns. Same pattern is
        idempotently deduped on ``(veto, domain, task)``; the count
        on the existing entry is incremented instead of duplicating.

        ``max_patterns`` caps the list so a misconfigured filter loop
        can't unbounded-grow the record. The oldest entry is dropped
        when the cap is hit (FIFO).
        """
        with self._mutex:
            record = self._repo.get(skill_id)
            if record is None:
                raise LifecycleError(f"Unknown skill {skill_id!r}")
            now = observed_at if observed_at is not None else time.time()
            patterns = list(record.false_binding_patterns or [])
            key = (veto, domain or "", task or "")
            for entry in patterns:
                if (
                    entry.get("veto") == key[0]
                    and (entry.get("domain") or "") == key[1]
                    and (entry.get("task") or "") == key[2]
                ):
                    entry["count"] = int(entry.get("count", 0)) + 1
                    entry["last_observed_at"] = now
                    break
            else:
                patterns.append({
                    "veto": veto,
                    "veto_reason": veto_reason,
                    "domain": domain,
                    "task": task,
                    "count": 1,
                    "first_observed_at": now,
                    "last_observed_at": now,
                })
            if len(patterns) > max_patterns:
                patterns = patterns[-max_patterns:]
            record.false_binding_patterns = patterns
            store = self._store_for(record.status)
            with store._unlocked(self._token):
                store.put(record, token=self._token)
            return record

    def record_task_verification(
        self,
        skill_id: str,
        *,
        verified_tasks: Sequence[str],
        evaluation_id: Optional[str] = None,
        per_task_metrics: Optional[Mapping[str, Mapping[str, float]]] = None,
        rationale: str = "",
    ) -> SkillRecord:
        """Append `verified_tasks` to `SkillRecord.verified_tasks`.

        Day-7 (PLAN-HARNESS §22 task axis) analog of
        `record_transfer_verification` for the **intra-domain task
        axis**. Called by the Stage 3a transfer-cycle driver
        (`labeling_supplement/_phase4_transfer_cycle.py --persist`)
        after a target-task probe passes the FewShotAdapter pass-rate
        threshold. This is the *only* sanctioned writer of
        `verified_tasks` and the matching `adapter_history` entries.

        Per PLAN-UNIFIED-SKILL-GATE §7, each task verification appends
        one `adapter_history` entry tagged ``kind="task_verification"``
        so the downstream lineage tooling can distinguish task-axis
        verifications from cross-domain ones. ``verified_tasks`` are
        free-form strings (every new env / website is a task) so we
        don't enforce a closed enum the way ``verified_domains`` does
        — but we do require the rationale.
        """
        if not rationale:
            raise LifecycleError(
                "record_task_verification requires a non-empty rationale."
            )
        if not verified_tasks:
            raise LifecycleError(
                "record_task_verification requires non-empty verified_tasks."
            )
        with self._mutex:
            record = self._repo.get(skill_id)
            if record is None:
                raise LifecycleError(f"Unknown skill {skill_id!r}")
            unique_tasks: list[str] = []
            seen: set[str] = set()
            for t in verified_tasks:
                key = (t or "").strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                unique_tasks.append(key)
            current = list(record.verified_tasks)
            now = time.time()
            metrics = per_task_metrics or {}
            for t in unique_tasks:
                if t not in current:
                    current.append(t)
                entry: Dict[str, Any] = {
                    "kind": "task_verification",
                    "target_task": t,
                    "evaluation_id": evaluation_id,
                    "verified_at": now,
                    "rationale": rationale,
                }
                if t in metrics:
                    entry["metrics"] = dict(metrics[t])
                record.adapter_history.append(entry)
            record.verified_tasks = current
            store = self._store_for(record.status)
            with store._unlocked(self._token):
                store.put(record, token=self._token)
            return record

    def record_transfer_verification(
        self,
        skill_id: str,
        *,
        verified_targets: Sequence[str],
        evaluation_id: Optional[str] = None,
        per_target_metrics: Optional[Mapping[str, Mapping[str, float]]] = None,
        rationale: str = "",
    ) -> SkillRecord:
        """Append `verified_targets` to `SkillRecord.verified_domains`.

        This is the *only* sanctioned writer of `verified_domains` and
        `adapter_history`. Called by `PromotionOrchestrator.promote` once
        per transition whose gate verdict carried Stage 3a verifications.

        Per PLAN-UNIFIED-SKILL-GATE §7, each call appends one
        `adapter_history` entry per target — these accumulate across
        re-evaluations so the Crafter / orchestrator can reconstruct the
        full transfer lineage of a skill.
        """
        if not rationale:
            raise LifecycleError(
                "record_transfer_verification requires a non-empty rationale."
            )
        with self._mutex:
            record = self._repo.get(skill_id)
            if record is None:
                raise LifecycleError(f"Unknown skill {skill_id!r}")
            unique_targets: list[str] = []
            seen: set[str] = set()
            for t in verified_targets:
                if t in seen:
                    continue
                seen.add(t)
                if t not in TRANSFER_TARGET_DOMAINS:
                    raise LifecycleError(
                        f"record_transfer_verification: {t!r} is not in "
                        f"TRANSFER_TARGET_DOMAINS={TRANSFER_TARGET_DOMAINS}."
                    )
                if t not in DOMAINS:
                    raise LifecycleError(
                        f"record_transfer_verification: {t!r} is not in DOMAINS."
                    )
                unique_targets.append(t)

            current = list(record.verified_domains)
            now = time.time()
            metrics = per_target_metrics or {}
            for t in unique_targets:
                if t not in current:
                    current.append(t)
                entry: Dict[str, Any] = {
                    "target_domain": t,
                    "evaluation_id": evaluation_id,
                    "verified_at": now,
                    "rationale": rationale,
                }
                if t in metrics:
                    entry["metrics"] = dict(metrics[t])
                record.adapter_history.append(entry)
            record.verified_domains = current
            store = self._store_for(record.status)
            with store._unlocked(self._token):
                store.put(record, token=self._token)
            return record

    # -- internals -------------------------------------------------------

    def _store_for(self, status: SkillStatus) -> SkillStore:
        target = store_for_status(status)
        if target == StoreName.DRAFT:
            return self._repo.draft
        if target == StoreName.CANDIDATE:
            return self._repo.candidate
        if target == StoreName.ACTIVE:
            return self._repo.active
        if target == StoreName.ARCHIVE:
            return self._repo.archive
        raise LifecycleError(f"No store for status {status.value!r}")

    def _write(self, store: SkillStore, record: SkillRecord) -> None:
        with store._unlocked(self._token):
            store.put(record, token=self._token)

    def _validate_invariants(
        self,
        record: SkillRecord,
        to_status: SkillStatus,
        rationale: str,
    ) -> None:
        if not rationale:
            raise LifecycleError("Transition requires a non-empty rationale.")
        # T1.3d: lane-(a) replacement for the legacy "feasible_domains
        # ≥ 2" ACTIVE invariant (PLAN-SKILL-BANK §0.1, superseded).
        # ACTIVE requires ``min_retrievals_per_skill`` retrievals when
        # the manager was constructed with a non-zero threshold; the
        # orchestrator wires
        # ``OrchestratorConfig.gate_thresholds.min_retrievals_per_skill``
        # at construction time. Standalone tests / drivers default to
        # ``0`` (no enforcement) so existing call-sites remain green.
        if to_status == SkillStatus.ACTIVE and self._min_retrievals_per_skill > 0:
            n_retrievals = int(record.metrics.get("retrievals", 0.0))
            if n_retrievals < self._min_retrievals_per_skill:
                raise LifecycleError(
                    f"Cannot promote {record.skill_id!r} to ACTIVE: "
                    f"min_retrievals_per_skill requires "
                    f"≥{self._min_retrievals_per_skill} retrievals "
                    f"(got {n_retrievals})."
                )
        # PLAN-UNIFIED-SKILL-GATE §3: ACTIVE requires non-empty
        # expected_evidence_roles.
        if to_status == SkillStatus.ACTIVE and not record.contract.expected_evidence_roles:
            raise LifecycleError(
                f"Cannot promote {record.skill_id!r} to ACTIVE: "
                f"contract.expected_evidence_roles is empty (G0 violation)."
            )
        # PLAN-SKILL-BANK §0.4 / PLAN-UNIFIED-SKILL-GATE Stage 3a — source/target
        # asymmetry. We enforce this *only* when the record actually declares
        # source-/transfer-target metadata; legacy records (with empty
        # source_domains) fall back to the older 2-domain check above. New
        # records produced by the crafter / mining flow MUST populate these
        # fields, so the strict path applies to them.
        if to_status == SkillStatus.ACTIVE and record.source_domains:
            if not any(d in SOURCE_DOMAINS for d in record.source_domains):
                raise LifecycleError(
                    f"Cannot promote {record.skill_id!r} to ACTIVE: "
                    f"source-domain (game-foundry) lineage required, got "
                    f"source_domains={sorted(set(record.source_domains))} "
                    f"(SOURCE_DOMAINS={SOURCE_DOMAINS})."
                )
            if not any(d in TRANSFER_TARGET_DOMAINS for d in record.verified_domains):
                raise LifecycleError(
                    f"Cannot promote {record.skill_id!r} to ACTIVE: "
                    f"few-shot transfer gate (G3a) requires ≥1 verified target "
                    f"domain, got verified_domains={sorted(set(record.verified_domains))} "
                    f"(TRANSFER_TARGET_DOMAINS={TRANSFER_TARGET_DOMAINS})."
                )


__all__ = ["LifecycleError", "SkillLifecycleManager"]
