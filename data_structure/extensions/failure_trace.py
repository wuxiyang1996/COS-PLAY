"""`FailureTrace` and `FailureDiagnosis` (PLAN-SKILL-CRAFTER §6).

Captured by the harness on every contract violation / abort and consumed
by the crafter's failure-reflection layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.enums import RecoveryStrategy
from common.ids import new_proposal_id


@dataclass
class FailureTrace:
    """One observed failure of a single skill invocation."""

    failure_id: str = field(default_factory=lambda: f"fail-{new_proposal_id().split('-', 1)[1]}")
    skill_id: str = ""
    skill_episode_id: str = ""
    domain: str = ""
    failed_step_index: Optional[int] = None
    failure_class: str = ""              # e.g. "PRECONDITION_VIOLATION"
    abort_reason: Optional[str] = None
    pre_state: Optional[Dict[str, Any]] = None
    failed_step: Optional[Dict[str, Any]] = None
    contract_violation: Optional[str] = None
    observed_evidence_roles: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "skill_id": self.skill_id,
            "skill_episode_id": self.skill_episode_id,
            "domain": self.domain,
            "failed_step_index": self.failed_step_index,
            "failure_class": self.failure_class,
            "abort_reason": self.abort_reason,
            "pre_state": self.pre_state,
            "failed_step": self.failed_step,
            "contract_violation": self.contract_violation,
            "observed_evidence_roles": list(self.observed_evidence_roles),
            "extra": dict(self.extra),
        }


@dataclass
class FailureDiagnosis:
    """Crafter-side analysis of a single FailureTrace."""

    failure_id: str
    locus: str                 # "precondition" | "protocol_step" | "effect_check" | ...
    root_cause: str
    recommended_strategy: RecoveryStrategy
    counterfactual: Optional[str] = None
    confidence: float = 0.0
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "locus": self.locus,
            "root_cause": self.root_cause,
            "recommended_strategy": self.recommended_strategy.value,
            "counterfactual": self.counterfactual,
            "confidence": self.confidence,
            "notes": self.notes,
        }


__all__ = ["FailureDiagnosis", "FailureTrace"]
