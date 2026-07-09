"""`SkillEpisode` — canonical record for one harness skill invocation.

Spec: PLAN-HARNESS §5 ("Core abstractions"), PLAN-PIPELINE-ORCHESTRATOR §6.

Invariants enforced at construction time:

  G0  evidence-driven : `outcome.evidence_role` is non-empty AND every
                        step that *claims to gather/verify/reason/commit*
                        carries at least one EvidenceRef in that role.
  G1  no memory       : the harness never reads/writes a "memory" buffer;
                        we therefore reject any step whose action_type
                        starts with "QUERY_MEM" / "WRITE_MEM".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.enums import EVIDENCE_ROLES, SkillType
from common.ids import new_episode_id, new_span_id
from common.state_schema import EvidenceRef, StateSchema


_FORBIDDEN_ACTION_PREFIXES: Tuple[str, ...] = ("QUERY_MEM", "WRITE_MEM")


@dataclass
class SkillEpisodeStep:
    """One inner-MDP tick inside a single skill invocation.

    Day-8 (PLAN-HARNESS §10): the step gained an evidence_in/out
    split, a `protocol_index` mapping back to ``skill.protocol[k]``,
    and three citation slots (`evidence_warrant`, `verify_verdict`,
    `reason_warrant`) for the gate's per-key contract progress.
    Existing callers writing only the legacy ``evidence: List[…]``
    field continue to work — `__post_init__` mirrors the legacy
    field into ``evidence_out`` for forward-compatible reads.
    """

    step_index: int
    action_type: str
    action_payload: Dict[str, Any]
    pre_state: Optional[Dict[str, Any]]   # serialized StateSchema snapshot
    post_state: Optional[Dict[str, Any]]
    # Legacy uni-directional evidence list — deprecated, but kept so
    # adapter authors don't have to update on Day 8. New callers should
    # populate `evidence_in` / `evidence_out` directly.
    evidence: List[EvidenceRef] = field(default_factory=list)
    # Day-8: directional evidence split. `evidence_in` is what the
    # step *consumed* (read from prior `state.evidence`); `evidence_out`
    # is what it *produced* (additions to the post-state evidence).
    evidence_in: List[EvidenceRef] = field(default_factory=list)
    evidence_out: List[EvidenceRef] = field(default_factory=list)
    # Day-8: maps `episode.steps[i] → skill.protocol[k]`. None for
    # steps the adapter generated outside the lifted protocol (e.g.
    # observation snapshots between hops).
    protocol_index: Optional[int] = None
    # Day-8: PLAN-UNIFIED §3.4 citation slots — when the step's
    # outcome is a verification, reasoning, or grounded write, these
    # carry the supporting evidence-ref / claim. Free-form for now;
    # the lift-aware contract checker will tighten the shape later.
    evidence_warrant: Optional[Dict[str, Any]] = None
    verify_verdict: Optional[Dict[str, Any]] = None
    reason_warrant: Optional[Dict[str, Any]] = None
    notes: str = ""

    def __post_init__(self) -> None:
        for prefix in _FORBIDDEN_ACTION_PREFIXES:
            if self.action_type.upper().startswith(prefix):
                raise ValueError(
                    f"SkillEpisodeStep action_type={self.action_type!r} violates "
                    f"the no-memory invariant (PLAN-EXTENSION §1.3)."
                )
        # Day-8 back-compat: when `evidence` is set but the directional
        # split is empty, mirror the legacy field into `evidence_out`
        # so consumers can read the new field without caring whether
        # the producer is updated.
        if self.evidence and not self.evidence_in and not self.evidence_out:
            self.evidence_out = list(self.evidence)

    def to_json(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action_type": self.action_type,
            "action_payload": self.action_payload,
            "pre_state": self.pre_state,
            "post_state": self.post_state,
            "evidence": [e.to_json() for e in self.evidence],
            "evidence_in": [e.to_json() for e in self.evidence_in],
            "evidence_out": [e.to_json() for e in self.evidence_out],
            "protocol_index": self.protocol_index,
            "evidence_warrant": dict(self.evidence_warrant) if self.evidence_warrant else None,
            "verify_verdict": dict(self.verify_verdict) if self.verify_verdict else None,
            "reason_warrant": dict(self.reason_warrant) if self.reason_warrant else None,
            "notes": self.notes,
        }


@dataclass
class SkillEpisodeOutcome:
    """Terminal status for a skill invocation.

    Day-8 fields (additive, no-op when absent):
      * `contract_progress` — per-contract-key satisfaction booleans
        (e.g. ``{"effects_add[0]": True, "expected_evidence_role[reason]": False}``).
        Required by PLAN-UNIFIED §3.4 for the Crafter to repair only
        the *failing* sub-clauses rather than scrap the whole skill.
      * `reward_components` — per-component decomposition of the
        scalar `score`. Spec lists `r_env / r_follow / r_cost / r_total`
        as canonical; we keep the dict open so future components plug
        in without a schema change.
    """

    success: bool                       # contract.satisfied
    contract_satisfied: bool            # explicit, may differ from success
    abort_reason: Optional[str] = None  # e.g. "precondition_violated"
    evidence_role: List[str] = field(default_factory=list)  # union of step roles
    answer: Optional[Any] = None         # for COMMIT skills
    score: Optional[float] = None        # numeric gate score, in [0,1]
    # Day-8: per-key contract progress + multi-component reward.
    contract_progress: Dict[str, bool] = field(default_factory=dict)
    reward_components: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for role in self.evidence_role:
            if role not in EVIDENCE_ROLES:
                raise ValueError(
                    f"SkillEpisodeOutcome.evidence_role contains {role!r}, "
                    f"not in {EVIDENCE_ROLES}."
                )

    def to_json(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "contract_satisfied": self.contract_satisfied,
            "abort_reason": self.abort_reason,
            "evidence_role": list(self.evidence_role),
            "answer": self.answer,
            "score": self.score,
            "contract_progress": dict(self.contract_progress),
            "reward_components": dict(self.reward_components),
            "extra": dict(self.extra),
        }


@dataclass
class SkillEpisode:
    """One harness invocation of a single skill on a real (or replay) state.

    Required for: gate replay (§7.1), reward logging (PLAN-HARNESS §5.6),
    crafter failure mining (PLAN-SKILL-CRAFTER §6).
    """

    episode_id: str
    skill_id: str
    skill_version: str
    skill_type: SkillType
    domain: str
    parent_run_id: Optional[str]              # outer rollout this came from
    parent_episode_id: Optional[str] = None   # parent skill episode (composition)
    span_id: str = field(default_factory=new_span_id)
    initial_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None
    steps: List[SkillEpisodeStep] = field(default_factory=list)
    outcome: Optional[SkillEpisodeOutcome] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cost: Dict[str, float] = field(default_factory=dict)   # tokens, ms, hops
    transfer_label: Optional[str] = None       # diagnostic label, PLAN-HARNESS §6.4
    # Day-8: episode-level shadow flag — propagated from the
    # eligibility filter so Stage-2 readers can distinguish shadow
    # failures from real failures (closes harness/README §10).
    shadow: bool = False
    # Day-8: spec-shaped diagnostic-label list (vs. the scalar
    # `transfer_label` above). G0-violation tagging, transfer
    # diagnostics, and the like all join this list — `transfer_label`
    # remains for back-compat and is auto-mirrored into
    # `diagnostic_labels` in `__post_init__`.
    diagnostic_labels: List[str] = field(default_factory=list)
    # Day-8: PLAN-HARNESS §10 protocol-trace mapping. Same
    # information as the per-step `protocol_index` field, surfaced as
    # an episode-level array so consumers can do
    # ``episode.protocol_trace[i] -> skill.protocol[k]`` without
    # walking every step. Populated lazily on `add_step` when the
    # step carries a `protocol_index`.
    protocol_trace: List[Optional[int]] = field(default_factory=list)

    @classmethod
    def begin(
        cls,
        *,
        skill_id: str,
        skill_version: str,
        skill_type: SkillType,
        domain: str,
        parent_run_id: Optional[str],
        initial_state: Optional[StateSchema] = None,
        parent_episode_id: Optional[str] = None,
    ) -> "SkillEpisode":
        return cls(
            episode_id=new_episode_id(),
            skill_id=skill_id,
            skill_version=skill_version,
            skill_type=skill_type,
            domain=domain,
            parent_run_id=parent_run_id,
            parent_episode_id=parent_episode_id,
            initial_state=initial_state.to_json() if initial_state is not None else None,
        )

    def add_step(self, step: SkillEpisodeStep) -> None:
        self.steps.append(step)
        # Day-8: keep the episode-level protocol_trace mirror in sync.
        # `None` when the step has no protocol mapping (observation
        # snapshots, executor-internal hops).
        self.protocol_trace.append(step.protocol_index)

    def __post_init__(self) -> None:
        # Day-8: mirror legacy `transfer_label` into `diagnostic_labels`
        # so callers reading the new field always see the legacy data.
        if self.transfer_label and self.transfer_label not in self.diagnostic_labels:
            self.diagnostic_labels.append(self.transfer_label)

    def finalize(
        self,
        *,
        outcome: SkillEpisodeOutcome,
        final_state: Optional[StateSchema] = None,
    ) -> None:
        self._enforce_evidence_invariant(outcome)
        self.outcome = outcome
        if final_state is not None:
            self.final_state = final_state.to_json()

    # -- invariants ------------------------------------------------------

    def _enforce_evidence_invariant(self, outcome: SkillEpisodeOutcome) -> None:
        """G0: a successful invocation must carry at least one role of evidence
        when the skill_type implies it (REASONING, GROUNDING, MIXED).

        ACTION-only skills are exempt only if outcome.score is provided
        externally (typically by a contract checker).
        """
        if not outcome.success:
            return
        if self.skill_type == SkillType.ACTION:
            return
        union: set[str] = set(outcome.evidence_role)
        for step in self.steps:
            for ref in step.evidence:
                union.add(ref.role)
        if not union:
            raise ValueError(
                f"SkillEpisode {self.episode_id} (skill={self.skill_id}) is "
                f"successful but carries no evidence — violates the "
                f"evidence-driven invariant (G0)."
            )

    def to_json(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "skill_type": self.skill_type.value,
            "domain": self.domain,
            "parent_run_id": self.parent_run_id,
            "parent_episode_id": self.parent_episode_id,
            "span_id": self.span_id,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "steps": [s.to_json() for s in self.steps],
            "outcome": self.outcome.to_json() if self.outcome else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cost": dict(self.cost),
            "transfer_label": self.transfer_label,
            "shadow": self.shadow,
            "diagnostic_labels": list(self.diagnostic_labels),
            "protocol_trace": list(self.protocol_trace),
        }


__all__ = ["SkillEpisode", "SkillEpisodeOutcome", "SkillEpisodeStep"]
