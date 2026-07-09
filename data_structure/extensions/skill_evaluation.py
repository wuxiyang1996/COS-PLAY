"""`SkillEvaluationRecord` — the harness/gate's signed report on a skill.

Spec: PLAN-UNIFIED-SKILL-GATE §3.2 — owned by the harness/gate,
NOT by the bank. The bank persists *references* to evaluation records
(via `SkillRecord.last_evaluation_id`) but never mutates them.

Day-8 (PLAN-UNIFIED-SKILL-GATE §3.2 + harness/README §11): the record
gained a reproducibility-anchor block so two evaluations against
different bank snapshots / eval suites / adapter versions /
ontology revisions are distinguishable on disk. None of the new
fields are mandatory; callers that don't set them get the legacy
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.enums import SkillStatus
from data_structure.extensions.gate_verdict import GateVerdictPayload


@dataclass
class SkillEvaluationRecord:
    evaluation_id: str
    proposal_id: str
    skill_id: str
    skill_content_hash: str
    episode_ids: List[str] = field(default_factory=list)   # SkillEpisode IDs evaluated
    verdict: Optional[GateVerdictPayload] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    failure_class_distribution: Dict[str, int] = field(default_factory=dict)
    transfer_labels: Dict[str, int] = field(default_factory=dict)  # PLAN-HARNESS §6.4 labels
    judge_model: Optional[str] = None
    seed: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Day-8: reproducibility anchors. Each is None when the evaluator
    # didn't pin it — the GateRunner does so by default.
    bank_snapshot_id: Optional[str] = None
    eval_suite_id: Optional[str] = None
    adapter_versions: Dict[str, str] = field(default_factory=dict)
    ontology_version: Optional[str] = None
    version: Optional[str] = None              # skill version at evaluation time
    status_before: Optional[SkillStatus] = None
    status_after: Optional[SkillStatus] = None
    rejected_domains: List[str] = field(default_factory=list)
    rollback_target: Optional[str] = None
    diagnostic_labels: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "proposal_id": self.proposal_id,
            "skill_id": self.skill_id,
            "skill_content_hash": self.skill_content_hash,
            "episode_ids": list(self.episode_ids),
            "verdict": self.verdict.to_json() if self.verdict else None,
            "metrics": dict(self.metrics),
            "failure_class_distribution": dict(self.failure_class_distribution),
            "transfer_labels": dict(self.transfer_labels),
            "judge_model": self.judge_model,
            "seed": self.seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # Day-8 anchors — emitted unconditionally so consumers can
            # depend on them being present (None when unset).
            "bank_snapshot_id": self.bank_snapshot_id,
            "eval_suite_id": self.eval_suite_id,
            "adapter_versions": dict(self.adapter_versions),
            "ontology_version": self.ontology_version,
            "version": self.version,
            "status_before": self.status_before.value if self.status_before else None,
            "status_after": self.status_after.value if self.status_after else None,
            "rejected_domains": list(self.rejected_domains),
            "rollback_target": self.rollback_target,
            "diagnostic_labels": list(self.diagnostic_labels),
        }


__all__ = ["SkillEvaluationRecord"]
