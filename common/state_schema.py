"""Canonical structured `<state>` schema used by every component.

Defined in PLAN-SKILL-BANK §3 ("Skill as a structured-state program") and
referenced by the Action-Agent inner-MDP, the Skill-Harness adapters, and
the Skill-Crafter when proposing new skills.

A `<state>` is *typed*, *finite*, and *evidence-backed*: every claim that
shows up here must be derivable from at least one `EvidenceRef`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.enums import EVIDENCE_ROLES


@dataclass(frozen=True)
class EvidenceRef:
    """A pointer to a concrete observation in the trace.

    The role enum distinguishes evidence used to *gather* a fact, *verify*
    a claim, *reason* about it, or *commit* it to the answer
    (PLAN-SKILL-BANK §0.3 Clause B).
    """

    source: str            # e.g. "grounding", "tool:wikipedia.search", "ocr"
    locator: str           # frame index, span id, bbox id, url, etc.
    role: str              # one of EVIDENCE_ROLES
    confidence: float = 1.0
    payload: Optional[Dict[str, Any]] = None  # small inlined snippet, optional

    def __post_init__(self) -> None:
        if self.role not in EVIDENCE_ROLES:
            raise ValueError(
                f"EvidenceRef.role={self.role!r} not in {EVIDENCE_ROLES}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"EvidenceRef.confidence={self.confidence} must be in [0,1]"
            )

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "source": self.source,
            "locator": self.locator,
            "role": self.role,
            "confidence": self.confidence,
        }
        if self.payload is not None:
            out["payload"] = self.payload
        return out


@dataclass
class StateTargets:
    """The decomposed sub-goals the Actor maintains under the current task.

    PLAN-ACTION-AGENT §4.2: targets are typed, addressable, and
    individually closeable so the inner-MDP can prove progress.
    """

    pending: List[str] = field(default_factory=list)
    achieved: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)


@dataclass
class StateSchema:
    """Structured `<state>` snapshot at a single inner-MDP tick.

    Fields mirror PLAN-SKILL-BANK §3 ("Skill as a structured-state
    program") and PLAN-ACTION-AGENT §4 (canonical state snapshot).
    """

    task: str                                                  # natural-language task spec
    domain: str                                                # one of common.enums.DOMAINS
    targets: StateTargets = field(default_factory=StateTargets)
    elements: List[Dict[str, Any]] = field(default_factory=list)   # grounded screen / scene elements
    facts: Dict[str, Any] = field(default_factory=dict)            # currently believed facts
    open_questions: List[str] = field(default_factory=list)        # what still needs verifying
    evidence: List[EvidenceRef] = field(default_factory=list)      # cumulative evidence in scope
    inner_step: int = 0                                            # inner-MDP tick counter
    outer_step: int = 0                                            # outer-env tick counter
    extra: Dict[str, Any] = field(default_factory=dict)            # adapter-specific scratch

    # ---- evidence-driven invariant helpers ---------------------------------

    def has_evidence(self) -> bool:
        return bool(self.evidence)

    def covers_role(self, role: str) -> bool:
        if role not in EVIDENCE_ROLES:
            raise ValueError(f"Unknown evidence role {role!r}")
        return any(e.role == role for e in self.evidence)

    def role_counts(self) -> Tuple[Tuple[str, int], ...]:
        counts = {role: 0 for role in EVIDENCE_ROLES}
        for ref in self.evidence:
            counts[ref.role] = counts.get(ref.role, 0) + 1
        return tuple(counts.items())

    def to_json(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "domain": self.domain,
            "targets": {
                "pending": list(self.targets.pending),
                "achieved": list(self.targets.achieved),
                "blocked": list(self.targets.blocked),
            },
            "elements": list(self.elements),
            "facts": dict(self.facts),
            "open_questions": list(self.open_questions),
            "evidence": [e.to_json() for e in self.evidence],
            "inner_step": self.inner_step,
            "outer_step": self.outer_step,
            "extra": dict(self.extra),
        }


__all__ = ["EvidenceRef", "StateSchema", "StateTargets"]
