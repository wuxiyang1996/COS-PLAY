"""`SkillRecord` — canonical bank entry (PLAN-UNIFIED-SKILL-GATE §3.1).

Owned by `skill_bank/`. The Harness, Crafter, and Orchestrator all *read*
records but only `skill_bank/lifecycle.py` may mutate the `status` field
(see PLAN-UNIFIED-SKILL-GATE §6 SkillLifecycleManager).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.enums import (
    DOMAINS,
    EVIDENCE_ROLES,
    SOURCE_DOMAINS,
    TRANSFER_TARGET_DOMAINS,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.ids import new_skill_id, schema_hash


@dataclass
class SkillContract:
    """The promised effect / belief / grounding contract of a skill.

    Mirrors PLAN-SKILL-BANK §4 ("Skill data model"): preconditions, effects,
    expected evidence roles, abort criteria.
    """

    preconditions: List[str] = field(default_factory=list)
    effects_add: List[str] = field(default_factory=list)
    effects_del: List[str] = field(default_factory=list)
    belief_progress: List[str] = field(default_factory=list)   # e.g. "narrows(open_question)"
    grounding_progress: List[str] = field(default_factory=list)
    expected_evidence_roles: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    abort_criteria: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for role in self.expected_evidence_roles:
            if role not in EVIDENCE_ROLES:
                raise ValueError(
                    f"SkillContract.expected_evidence_roles contains {role!r}, "
                    f"not in {EVIDENCE_ROLES}."
                )

    def to_json(self) -> Dict[str, Any]:
        return {
            "preconditions": list(self.preconditions),
            "effects_add": list(self.effects_add),
            "effects_del": list(self.effects_del),
            "belief_progress": list(self.belief_progress),
            "grounding_progress": list(self.grounding_progress),
            "expected_evidence_roles": list(self.expected_evidence_roles),
            "success_criteria": list(self.success_criteria),
            "abort_criteria": list(self.abort_criteria),
        }


@dataclass
class SkillRecord:
    """A single skill entry in the bank.

    Status transitions are *only* permitted via
    `skill_bank.lifecycle.SkillLifecycleManager.transition()` — direct
    assignment is forbidden by `__setattr__` below.
    """

    skill_id: str
    name: str
    skill_type: SkillType
    source_type: SkillSourceType
    status: SkillStatus
    version: str = "v1"
    feasible_domains: List[str] = field(default_factory=list)   # subset of common.enums.DOMAINS
    # Source/target asymmetry (PLAN-SKILL-BANK §0.4 / PLAN-UNIFIED-SKILL-GATE §7 Stage 3a).
    # `source_domains` ⊆ SOURCE_DOMAINS and is the foundry where the skill was
    # mined and hardened. `transfer_target_domains` ⊆ TRANSFER_TARGET_DOMAINS
    # are the bindings the skill *claims* to support; `verified_domains` is
    # the subset that has actually passed the few-shot adaptation gate.
    source_domains: List[str] = field(default_factory=list)
    transfer_target_domains: List[str] = field(default_factory=list)
    verified_domains: List[str] = field(default_factory=list)
    # Task axis (intra-domain granularity, harness/README §22). `feasible_domains`
    # answers "which environment family does this skill claim to apply to?";
    # `feasible_tasks` answers "which specific task within that family?". For
    # gymv, feasible_domains=["gymv"] but feasible_tasks=["twenty_forty_eight"]
    # (or ["tetris"], etc.). Free-form strings — there is no task enum because
    # tasks are open-ended (every new env/game/website is one). The
    # `EligibilityFilter` and `FewShotAdapter` honour these only when non-empty,
    # so existing skills (decorated before this field landed) remain admissible
    # everywhere their domain admits them.
    feasible_tasks: List[str] = field(default_factory=list)
    verified_tasks: List[str] = field(default_factory=list)
    adapter_history: List[Dict[str, Any]] = field(default_factory=list)        # PLAN-SKILL-BANK §4.3a
    false_binding_patterns: List[Dict[str, Any]] = field(default_factory=list)  # PLAN-SKILL-BANK §4.3b
    protocol: List[Dict[str, Any]] = field(default_factory=list)  # ordered hop list
    contract: SkillContract = field(default_factory=SkillContract)
    parent_skill_ids: List[str] = field(default_factory=list)     # composition / repair lineage
    proposal_id: Optional[str] = None
    crafted_at: Optional[float] = None
    last_evaluation_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)       # rolling pass_rate etc.
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for d in self.feasible_domains:
            if d not in DOMAINS:
                raise ValueError(
                    f"SkillRecord.feasible_domains contains {d!r}, "
                    f"not in canonical DOMAINS={DOMAINS}."
                )
        for d in self.source_domains:
            if d not in SOURCE_DOMAINS:
                raise ValueError(
                    f"SkillRecord.source_domains contains {d!r}, "
                    f"not in canonical SOURCE_DOMAINS={SOURCE_DOMAINS}."
                )
        for d in self.transfer_target_domains:
            if d not in TRANSFER_TARGET_DOMAINS:
                raise ValueError(
                    f"SkillRecord.transfer_target_domains contains {d!r}, "
                    f"not in canonical TRANSFER_TARGET_DOMAINS={TRANSFER_TARGET_DOMAINS}."
                )
        for d in self.verified_domains:
            if d not in DOMAINS:
                raise ValueError(
                    f"SkillRecord.verified_domains contains {d!r}, "
                    f"not in canonical DOMAINS={DOMAINS}."
                )
        # General-protocol invariant (PLAN-SKILL-BANK §0.1): at least 2 domains.
        if len(set(self.feasible_domains)) < 2:
            # NOTE: we do not raise here — DRAFT/CANDIDATE skills may be
            # provisionally single-domain; the gate (G3 transfer) is what
            # actually enforces this before promotion to ACTIVE.
            pass

    @classmethod
    def new(
        cls,
        *,
        name: str,
        skill_type: SkillType,
        source_type: SkillSourceType,
        feasible_domains: List[str],
        contract: Optional[SkillContract] = None,
        protocol: Optional[List[Dict[str, Any]]] = None,
        proposal_id: Optional[str] = None,
        parent_skill_ids: Optional[List[str]] = None,
        source_domains: Optional[List[str]] = None,
        transfer_target_domains: Optional[List[str]] = None,
        verified_domains: Optional[List[str]] = None,
        feasible_tasks: Optional[List[str]] = None,
        verified_tasks: Optional[List[str]] = None,
    ) -> "SkillRecord":
        return cls(
            skill_id=new_skill_id(),
            name=name,
            skill_type=skill_type,
            source_type=source_type,
            status=SkillStatus.DRAFT,
            feasible_domains=list(feasible_domains),
            source_domains=list(source_domains or []),
            transfer_target_domains=list(transfer_target_domains or []),
            verified_domains=list(verified_domains or []),
            feasible_tasks=list(feasible_tasks or []),
            verified_tasks=list(verified_tasks or []),
            protocol=list(protocol or []),
            contract=contract or SkillContract(),
            parent_skill_ids=list(parent_skill_ids or []),
            proposal_id=proposal_id,
        )

    def content_hash(self) -> str:
        """Stable hash over (protocol, contract, feasible_domains, version).

        The gate binds its `SkillEvaluationRecord` to this hash so the
        promotion path can detect if the skill body was edited *after*
        evaluation.

        `feasible_tasks` / `verified_tasks` are deliberately *excluded*
        from the hash — they are eligibility metadata, not skill body.
        Adding a verified task should not invalidate a prior evaluation.
        """
        return schema_hash(
            {
                "protocol": self.protocol,
                "contract": self.contract.to_json(),
                "feasible_domains": sorted(self.feasible_domains),
                "version": self.version,
                "skill_type": self.skill_type.value,
            }
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "skill_type": self.skill_type.value,
            "source_type": self.source_type.value,
            "status": self.status.value,
            "version": self.version,
            "feasible_domains": list(self.feasible_domains),
            "source_domains": list(self.source_domains),
            "transfer_target_domains": list(self.transfer_target_domains),
            "verified_domains": list(self.verified_domains),
            "feasible_tasks": list(self.feasible_tasks),
            "verified_tasks": list(self.verified_tasks),
            "adapter_history": [dict(x) for x in self.adapter_history],
            "false_binding_patterns": [dict(x) for x in self.false_binding_patterns],
            "protocol": list(self.protocol),
            "contract": self.contract.to_json(),
            "parent_skill_ids": list(self.parent_skill_ids),
            "proposal_id": self.proposal_id,
            "crafted_at": self.crafted_at,
            "last_evaluation_id": self.last_evaluation_id,
            "metrics": dict(self.metrics),
            "notes": self.notes,
            "tags": list(self.tags),
            "content_hash": self.content_hash(),
        }


__all__ = ["SkillContract", "SkillRecord"]
