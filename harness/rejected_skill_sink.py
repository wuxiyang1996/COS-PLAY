"""`RejectedSkillSink` — Day-9c (PLAN-SKILL-BANK §4.3b).

The eligibility filter's `RejectedSkill` channel (Day-8a) reports
*why* a candidate was vetoed. By itself that's a per-call diagnostic;
the actor uses it to render a veto log. The Crafter, however, needs
the *aggregated* signal: "skill X keeps getting vetoed for reason Y on
domain Z" — that's `false_binding_patterns` evidence, the input to
the Repairer's patch-or-retire decision.

This sink is the in-process aggregator that lives between the harness
and the Crafter:

  1. The harness (or the dump driver / orchestrator) calls
     `sink.observe(rejected, domain=…, task=…)` after every
     `EligibilityFilter.filter_with_rejections(...)` call.
  2. The sink dedupes on ``(skill_id, veto, domain, task)`` and
     maintains an in-memory hot count.
  3. When `sink.flush_to(lifecycle)` is invoked, every observed
     pattern lands on the corresponding `SkillRecord.false_binding_patterns`
     via `SkillLifecycleManager.record_false_binding_pattern`. The
     sink resets after a flush.

The flush is decoupled from the observation so the harness's hot path
doesn't pay the I/O cost — typical wiring runs the flush on the
`PromotionOrchestrator`'s scheduled tick or the dump driver's per-source
teardown.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from harness.eligibility import RejectedSkill


@dataclass
class _RejectionPattern:
    skill_id: str
    veto: str
    veto_reason: str
    domain: Optional[str] = None
    task: Optional[str] = None
    count: int = 0
    first_observed_at: Optional[float] = None
    last_observed_at: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "domain": self.domain,
            "task": self.task,
            "count": self.count,
            "first_observed_at": self.first_observed_at,
            "last_observed_at": self.last_observed_at,
        }


@dataclass
class FlushReport:
    """Diagnostic record returned by `flush_to(lifecycle)`."""

    n_skills_touched: int
    n_patterns_written: int
    n_errors: int
    skipped_unknown_skill_ids: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "n_skills_touched": self.n_skills_touched,
            "n_patterns_written": self.n_patterns_written,
            "n_errors": self.n_errors,
            "skipped_unknown_skill_ids": list(self.skipped_unknown_skill_ids),
            "errors": list(self.errors),
        }


class RejectedSkillSink:
    """In-process aggregator for `RejectedSkill` records.

    Thread-safe; the sink uses an internal RLock so a multi-threaded
    harness (or a multi-source dump driver) can share one sink.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._patterns: Dict[Tuple[str, str, str, str], _RejectionPattern] = {}

    # -- ingest ---------------------------------------------------------

    def observe(
        self,
        rejected: Iterable[RejectedSkill],
        *,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        observed_at: Optional[float] = None,
    ) -> int:
        """Aggregate `rejected` records into the in-memory buffer.

        Returns the number of patterns *touched* (created or
        incremented). De-dupes on ``(skill_id, veto, domain, task)``.
        Empty lists are valid no-ops.
        """
        now = observed_at if observed_at is not None else time.time()
        n_touched = 0
        with self._lock:
            for r in rejected:
                key = (r.skill.skill_id, r.veto, domain or "", task or "")
                p = self._patterns.get(key)
                if p is None:
                    p = _RejectionPattern(
                        skill_id=r.skill.skill_id,
                        veto=r.veto,
                        veto_reason=r.veto_reason,
                        domain=domain,
                        task=task,
                        count=0,
                        first_observed_at=now,
                    )
                    self._patterns[key] = p
                p.count += 1
                p.last_observed_at = now
                # Newest reason wins (it carries the freshest data).
                p.veto_reason = r.veto_reason
                n_touched += 1
        return n_touched

    # -- queries --------------------------------------------------------

    def patterns(self) -> List[_RejectionPattern]:
        with self._lock:
            return list(self._patterns.values())

    def hot_patterns(self, *, min_count: int = 3) -> List[_RejectionPattern]:
        with self._lock:
            return [p for p in self._patterns.values() if p.count >= min_count]

    def class_distribution(self) -> Dict[str, int]:
        """Per-veto-class histogram across all observed patterns."""
        with self._lock:
            return dict(Counter(p.veto for p in self._patterns.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._patterns)

    # -- flush ----------------------------------------------------------

    def flush_to(
        self,
        lifecycle: Any,
        *,
        min_count: int = 1,
        reset: bool = True,
    ) -> FlushReport:
        """Write every aggregated pattern to the bank via
        `SkillLifecycleManager.record_false_binding_pattern`.

        ``min_count`` filters patterns whose hot count is below the
        threshold (useful when the sink saw a one-off harmless veto
        that shouldn't pollute the record's `false_binding_patterns`).

        Patterns for skill_ids that aren't in the bank are skipped
        and reported in `FlushReport.skipped_unknown_skill_ids` rather
        than raised — the eligibility filter may see candidates
        synthesised from a transient repository the lifecycle doesn't
        own.

        Resets the sink on success unless ``reset=False``.
        """
        report = FlushReport(n_skills_touched=0, n_patterns_written=0, n_errors=0)
        touched: set = set()
        with self._lock:
            patterns = [p for p in self._patterns.values() if p.count >= min_count]
            for p in patterns:
                if lifecycle.get(p.skill_id) is None:
                    if p.skill_id not in report.skipped_unknown_skill_ids:
                        report.skipped_unknown_skill_ids.append(p.skill_id)
                    continue
                try:
                    lifecycle.record_false_binding_pattern(
                        p.skill_id,
                        veto=p.veto,
                        veto_reason=p.veto_reason,
                        domain=p.domain,
                        task=p.task,
                        observed_at=p.last_observed_at,
                    )
                    report.n_patterns_written += 1
                    touched.add(p.skill_id)
                except Exception as exc:  # noqa: BLE001
                    report.n_errors += 1
                    report.errors.append({
                        "skill_id": p.skill_id,
                        "veto": p.veto,
                        "error": repr(exc),
                    })
            report.n_skills_touched = len(touched)
            if reset:
                self._patterns.clear()
        return report


__all__ = ["FlushReport", "RejectedSkillSink"]
