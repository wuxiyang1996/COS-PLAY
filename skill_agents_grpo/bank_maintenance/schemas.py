"""
Data schemas for Bank Maintenance: Split / Merge / Refine.

Core types:
  - ``SkillProfile``: cached per-skill summary for fast index/trigger checks.
  - ``BankDiffEntry`` / ``BankDiffReport``: audit log of all bank mutations.
  - ``RedecodeRequest``: specification for local Stage-2 re-decode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ── SkillProfile ─────────────────────────────────────────────────────

@dataclass
class SkillProfile:
    """Pre-computed per-skill summary stored once, read by every trigger/index."""

    skill_id: str

    # Contract snapshot (effects only for MVP)
    eff_add: FrozenSet[str] = field(default_factory=frozenset)
    eff_del: FrozenSet[str] = field(default_factory=frozenset)
    eff_event: FrozenSet[str] = field(default_factory=frozenset)

    effect_signature_hash: int = 0

    # Sparse effect vector: predicate -> normalised support weight
    effect_sparse_vec: Dict[str, float] = field(default_factory=dict)

    # Embedding statistics (populated when embeddings are available)
    embedding_centroid: Optional[List[float]] = None
    embedding_var_diag: Optional[List[float]] = None

    # Transition top-K neighbours
    transition_topk_prev: List[Tuple[str, float]] = field(default_factory=list)
    transition_topk_next: List[Tuple[str, float]] = field(default_factory=list)

    # Duration statistics
    duration_mean: float = 0.0
    duration_var: float = 0.0

    # Verification summary
    overall_pass_rate: float = 0.0
    top_violating_literals: List[str] = field(default_factory=list)
    failure_signature_counts: Dict[str, int] = field(default_factory=dict)
    n_instances: int = 0

    @property
    def all_effects(self) -> FrozenSet[str]:
        return self.eff_add | self.eff_del | self.eff_event

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "eff_event": sorted(self.eff_event),
            "effect_signature_hash": self.effect_signature_hash,
            "effect_sparse_vec": self.effect_sparse_vec,
            "embedding_centroid": self.embedding_centroid,
            "embedding_var_diag": self.embedding_var_diag,
            "transition_topk_prev": self.transition_topk_prev,
            "transition_topk_next": self.transition_topk_next,
            "duration_mean": self.duration_mean,
            "duration_var": self.duration_var,
            "overall_pass_rate": self.overall_pass_rate,
            "top_violating_literals": self.top_violating_literals,
            "failure_signature_counts": self.failure_signature_counts,
            "n_instances": self.n_instances,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillProfile:
        return cls(
            skill_id=d["skill_id"],
            eff_add=frozenset(d.get("eff_add", [])),
            eff_del=frozenset(d.get("eff_del", [])),
            eff_event=frozenset(d.get("eff_event", [])),
            effect_signature_hash=d.get("effect_signature_hash", 0),
            effect_sparse_vec=d.get("effect_sparse_vec", {}),
            embedding_centroid=d.get("embedding_centroid"),
            embedding_var_diag=d.get("embedding_var_diag"),
            transition_topk_prev=[
                tuple(x) for x in d.get("transition_topk_prev", [])
            ],
            transition_topk_next=[
                tuple(x) for x in d.get("transition_topk_next", [])
            ],
            duration_mean=d.get("duration_mean", 0.0),
            duration_var=d.get("duration_var", 0.0),
            overall_pass_rate=d.get("overall_pass_rate", 0.0),
            top_violating_literals=d.get("top_violating_literals", []),
            failure_signature_counts=d.get("failure_signature_counts", {}),
            n_instances=d.get("n_instances", 0),
        )


# ── Bank diff / audit log ───────────────────────────────────────────

class DiffOp(str, Enum):
    SPLIT = "split"
    MERGE = "merge"
    REFINE = "refine"
    MATERIALIZE = "materialize"
    PROMOTE = "promote"
    DURATION_UPDATE = "duration_update"
    ADD = "add"
    REMOVE = "remove"


@dataclass
class BankDiffEntry:
    """One atomic change applied to the skill bank."""

    op: DiffOp
    skill_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "skill_id": self.skill_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class BankDiffReport:
    """Full audit log for one bank maintenance run."""

    entries: List[BankDiffEntry] = field(default_factory=list)
    n_splits: int = 0
    n_merges: int = 0
    n_refines: int = 0
    n_materializations: int = 0
    n_promotions: int = 0
    n_duration_updates: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def add(self, entry: BankDiffEntry) -> None:
        self.entries.append(entry)
        if entry.op == DiffOp.SPLIT:
            self.n_splits += 1
        elif entry.op == DiffOp.MERGE:
            self.n_merges += 1
        elif entry.op == DiffOp.REFINE:
            self.n_refines += 1
        elif entry.op == DiffOp.MATERIALIZE:
            self.n_materializations += 1
        elif entry.op == DiffOp.PROMOTE:
            self.n_promotions += 1
        elif entry.op == DiffOp.DURATION_UPDATE:
            self.n_duration_updates += 1

    def finalize(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "n_merges": self.n_merges,
            "n_refines": self.n_refines,
            "n_materializations": self.n_materializations,
            "n_promotions": self.n_promotions,
            "n_duration_updates": self.n_duration_updates,
            "elapsed_s": (
                self.finished_at - self.started_at if self.finished_at else 0
            ),
            "entries": [e.to_dict() for e in self.entries],
        }

    def summary(self) -> str:
        lines = [
            f"Bank Maintenance Diff: {self.n_splits} splits, "
            f"{self.n_merges} merges, {self.n_refines} refines, "
            f"{self.n_materializations} materializations, "
            f"{self.n_promotions} promotions, "
            f"{self.n_duration_updates} duration updates",
        ]
        for e in self.entries:
            lines.append(f"  [{e.op.value}] {e.skill_id}: {e.details}")
        return "\n".join(lines)


# ── Re-decode request ────────────────────────────────────────────────

@dataclass
class RedecodeRequest:
    """Request for local Stage-2 re-decode over a trajectory window."""

    traj_id: str
    window_start: int
    window_end: int
    reason: str = ""
    affected_skills: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "reason": self.reason,
            "affected_skills": self.affected_skills,
        }
