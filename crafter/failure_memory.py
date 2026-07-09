"""`FailureMemory` and `FailurePattern` — aggregated failure traces.

Spec: PLAN-SKILL-CRAFTER §6.6.

The memory deduplicates `FailureTrace`s by (skill_id, failure_class,
failed_step_index) into `FailurePattern`s the crafter can reason over.
Patterns over a configurable threshold trigger a repair / hypothesis
pass.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common.ids import new_proposal_id
from data_structure.extensions.failure_trace import FailureTrace


@dataclass
class FailurePattern:
    pattern_id: str
    skill_id: str
    failure_class: str
    failed_step_index: Optional[int]
    domains: List[str] = field(default_factory=list)
    failure_ids: List[str] = field(default_factory=list)
    sample_abort_reasons: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.failure_ids)

    def to_json(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "skill_id": self.skill_id,
            "failure_class": self.failure_class,
            "failed_step_index": self.failed_step_index,
            "domains": list(self.domains),
            "failure_ids": list(self.failure_ids),
            "sample_abort_reasons": list(self.sample_abort_reasons),
            "count": self.count,
        }


class FailureMemory:
    """Append-only failure aggregator with pattern-level reads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._traces: Dict[str, FailureTrace] = {}
        self._patterns: Dict[str, FailurePattern] = {}
        self._key_to_pattern: Dict[tuple, str] = {}

    # -- ingest -----------------------------------------------------------

    def add(self, trace: FailureTrace) -> FailurePattern:
        with self._lock:
            if not trace.failure_id:
                trace.failure_id = f"fail-{new_proposal_id().split('-', 1)[1]}"
            self._traces[trace.failure_id] = trace
            key = (trace.skill_id, trace.failure_class, trace.failed_step_index)
            pid = self._key_to_pattern.get(key)
            if pid is None:
                pattern = FailurePattern(
                    pattern_id=f"pat-{new_proposal_id().split('-', 1)[1]}",
                    skill_id=trace.skill_id,
                    failure_class=trace.failure_class,
                    failed_step_index=trace.failed_step_index,
                )
                self._patterns[pattern.pattern_id] = pattern
                self._key_to_pattern[key] = pattern.pattern_id
                pid = pattern.pattern_id
            pattern = self._patterns[pid]
            pattern.failure_ids.append(trace.failure_id)
            if trace.domain and trace.domain not in pattern.domains:
                pattern.domains.append(trace.domain)
            if trace.abort_reason and trace.abort_reason not in pattern.sample_abort_reasons:
                if len(pattern.sample_abort_reasons) < 5:
                    pattern.sample_abort_reasons.append(trace.abort_reason)
            return pattern

    def add_many(self, traces: List[FailureTrace]) -> List[FailurePattern]:
        return [self.add(t) for t in traces]

    # -- queries ----------------------------------------------------------

    def patterns(self) -> List[FailurePattern]:
        with self._lock:
            return list(self._patterns.values())

    def pattern(self, pattern_id: str) -> Optional[FailurePattern]:
        with self._lock:
            return self._patterns.get(pattern_id)

    def trace(self, failure_id: str) -> Optional[FailureTrace]:
        with self._lock:
            return self._traces.get(failure_id)

    def hot_patterns(self, *, min_count: int = 3) -> List[FailurePattern]:
        with self._lock:
            return [p for p in self._patterns.values() if p.count >= min_count]

    def class_distribution(self) -> Dict[str, int]:
        with self._lock:
            return dict(Counter(t.failure_class for t in self._traces.values()))


__all__ = ["FailureMemory", "FailurePattern"]
