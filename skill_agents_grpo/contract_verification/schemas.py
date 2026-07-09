"""
Data schemas for Stage 3 ContractVerification.

Three core types:
  - ``SegmentRecord``: enriched segment with predicate summaries and derived effects.
  - ``SkillContract``: pre/eff/inv contract for a skill, versioned.
  - ``VerificationReport``: per-skill verification metrics and counterexample links.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ── Segment-level data ───────────────────────────────────────────────

@dataclass
class SegmentRecord:
    """One labelled segment enriched with predicate summaries and derived effects.

    Produced by combining Stage 2 output with predicate extraction (Step 1-2).
    """

    seg_id: str
    traj_id: str
    t_start: int
    t_end: int
    skill_label: str  # existing skill id or "__NEW__"

    P_start: Dict[str, float] = field(default_factory=dict)
    P_end: Dict[str, float] = field(default_factory=dict)
    P_all: List[Dict[str, float]] = field(default_factory=list)

    effects_add: Set[str] = field(default_factory=set)
    effects_del: Set[str] = field(default_factory=set)

    embedding: Optional[np.ndarray] = None

    def effect_signature(self) -> str:
        """Canonical string signature for clustering: ``A:p1,p2|D:q1,q2``."""
        adds = ",".join(sorted(self.effects_add)) if self.effects_add else ""
        dels = ",".join(sorted(self.effects_del)) if self.effects_del else ""
        return f"A:{adds}|D:{dels}"

    def effect_vector(self, vocab: List[str]) -> np.ndarray:
        """Binary vector over a shared predicate vocabulary."""
        vec = np.zeros(len(vocab), dtype=np.float32)
        for i, pred in enumerate(vocab):
            if pred in self.effects_add or pred in self.effects_del:
                vec[i] = 1.0
        return vec

    def to_dict(self) -> dict:
        d: dict = {
            "seg_id": self.seg_id,
            "traj_id": self.traj_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "skill_label": self.skill_label,
            "P_start": self.P_start,
            "P_end": self.P_end,
            "effects_add": sorted(self.effects_add),
            "effects_del": sorted(self.effects_del),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SegmentRecord:
        return cls(
            seg_id=d["seg_id"],
            traj_id=d["traj_id"],
            t_start=d["t_start"],
            t_end=d["t_end"],
            skill_label=d["skill_label"],
            P_start=d.get("P_start", {}),
            P_end=d.get("P_end", {}),
            effects_add=set(d.get("effects_add", [])),
            effects_del=set(d.get("effects_del", [])),
        )


# ── Skill contract ───────────────────────────────────────────────────

@dataclass
class SkillContract:
    """Pre/Eff/Inv contract for one skill, with version tracking."""

    skill_id: str
    version: int = 0

    pre: Set[str] = field(default_factory=set)
    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    inv: Set[str] = field(default_factory=set)
    termination_cues: Set[str] = field(default_factory=set)
    soft_pre: Set[str] = field(default_factory=set)

    deprecated: bool = False
    children: List[str] = field(default_factory=list)  # if split
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "pre": sorted(self.pre),
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "inv": sorted(self.inv),
            "termination_cues": sorted(self.termination_cues),
            "soft_pre": sorted(self.soft_pre),
            "deprecated": self.deprecated,
            "children": self.children,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillContract:
        return cls(
            skill_id=d["skill_id"],
            version=d.get("version", 0),
            pre=set(d.get("pre", [])),
            eff_add=set(d.get("eff_add", [])),
            eff_del=set(d.get("eff_del", [])),
            inv=set(d.get("inv", [])),
            termination_cues=set(d.get("termination_cues", [])),
            soft_pre=set(d.get("soft_pre", [])),
            deprecated=d.get("deprecated", False),
            children=d.get("children", []),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )

    def to_action_language(self, fmt: str = "pddl", **kwargs) -> str:
        """Render this contract in an action language format.

        Parameters
        ----------
        fmt : str
            One of ``"pddl"``, ``"strips"``, ``"sas"``, ``"compact"``.
        **kwargs
            Passed to the formatter (e.g. ``include_inv=False``).

        Returns
        -------
        str
            Formatted action language representation.
        """
        from skill_agents_grpo.contract_verification.action_language import format_contract
        return format_contract(self, fmt=fmt, **kwargs)

    def compat_score(
        self,
        P_start: Dict[str, float],
        P_end: Dict[str, float],
        p_thresh: float = 0.7,
    ) -> float:
        """Contract compatibility score for use by Stage 2 ``SegmentScorer``.

        Returns a value in [-1, 1]: fraction of satisfied contract clauses
        minus fraction violated.
        """
        checks: List[bool] = []
        for p in self.pre:
            checks.append(P_start.get(p, 0.0) >= p_thresh)
        for p in self.eff_add:
            checks.append(P_end.get(p, 0.0) >= p_thresh)
        for p in self.eff_del:
            checks.append(P_end.get(p, 0.0) < p_thresh)
        if not checks:
            return 0.0
        return (sum(checks) / len(checks)) * 2.0 - 1.0


# ── Verification report ─────────────────────────────────────────────

@dataclass
class VerificationReport:
    """Verification metrics for one skill's contract against its instances."""

    skill_id: str
    n_instances: int = 0

    pre_violation_rate: Dict[str, float] = field(default_factory=dict)
    eff_add_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_del_success_rate: Dict[str, float] = field(default_factory=dict)
    inv_hold_rate: Dict[str, float] = field(default_factory=dict)

    overall_pass_rate: float = 0.0
    counterexample_ids: List[str] = field(default_factory=list)
    failure_signatures: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "n_instances": self.n_instances,
            "pre_violation_rate": self.pre_violation_rate,
            "eff_add_success_rate": self.eff_add_success_rate,
            "eff_del_success_rate": self.eff_del_success_rate,
            "inv_hold_rate": self.inv_hold_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "counterexample_ids": self.counterexample_ids,
            "failure_signatures": self.failure_signatures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerificationReport:
        return cls(**d)


# ── Update action enum ───────────────────────────────────────────────

@dataclass
class UpdateAction:
    """Deterministic action decision from verification."""

    skill_id: str
    action: str  # "KEEP" | "REFINE" | "SPLIT" | "MATERIALIZE_NEW"
    details: Dict[str, Any] = field(default_factory=dict)
    new_skill_ids: List[str] = field(default_factory=list)
    dropped_literals: Dict[str, List[str]] = field(default_factory=dict)
    demoted_to_soft: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "action": self.action,
            "details": self.details,
            "new_skill_ids": self.new_skill_ids,
            "dropped_literals": self.dropped_literals,
            "demoted_to_soft": self.demoted_to_soft,
        }
