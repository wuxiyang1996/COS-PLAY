"""Per-stage and aggregate gate verdicts.

Spec: PLAN-UNIFIED-SKILL-GATE §3.3 (`GateVerdict`, `GateVerdictPayload`).

Note: the `GateVerdict` *enum* lives in `common.enums` so consumers can
import the symbol without dragging in the dataclasses. This module owns
the per-stage payload structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.enums import GateStage, GateVerdict


@dataclass
class StageVerdict:
    """Verdict for one of the seven canonical gate stages."""

    stage: GateStage
    verdict: GateVerdict
    metrics: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "verdict": self.verdict.value,
            "metrics": dict(self.metrics),
            "failures": list(self.failures),
            "notes": self.notes,
        }


@dataclass
class GateVerdictPayload:
    """Aggregate verdict — what the orchestrator's PromotionOrchestrator reads.

    A skill may only be promoted to ACTIVE if `final_verdict == PASS`.
    `LIMITED_PASS` permits promotion to PROVISIONAL (PLAN-UNIFIED-SKILL-GATE
    §7 Stage 5).
    """

    proposal_id: str
    skill_id: str
    skill_content_hash: str
    stages: List[StageVerdict] = field(default_factory=list)
    final_verdict: GateVerdict = GateVerdict.FAIL
    rationale: str = ""
    eligible_domains: List[str] = field(default_factory=list)  # may be a strict subset
    notes: Optional[str] = None

    def stage_for(self, stage: GateStage) -> Optional[StageVerdict]:
        for s in self.stages:
            if s.stage == stage:
                return s
        return None

    def to_json(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "skill_id": self.skill_id,
            "skill_content_hash": self.skill_content_hash,
            "stages": [s.to_json() for s in self.stages],
            "final_verdict": self.final_verdict.value,
            "rationale": self.rationale,
            "eligible_domains": list(self.eligible_domains),
            "notes": self.notes,
        }


__all__ = ["GateVerdictPayload", "StageVerdict"]
