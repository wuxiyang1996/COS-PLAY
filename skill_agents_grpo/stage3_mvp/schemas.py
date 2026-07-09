"""
Data schemas for Stage 3 MVP: effects-only contracts.

Core types:
  - ``SegmentRecord``: enriched segment with booleanized predicates and effects.
  - ``SkillEffectsContract``: eff_add / eff_del / eff_event contract per skill.
  - ``VerificationReport``: per-skill pass rates and failure diagnostics.
  - ``SubEpisodeRef``: lightweight pointer to a stored rollout segment + cached summary.
  - ``Protocol``: actionable step-by-step guidance for the decision agent.
  - ``ScoredBoundary``: boundary candidate with penalty-based confidence score.
  - ``Skill``: two-part stored concept:
      Part 1 (Protocol Store) — queried by the decision agent.
      Part 2 (Evidence Store) — sub-episode pointers managed by the skill agent.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set


@dataclass
class SegmentRecord:
    """One labelled segment with predicate summaries and per-instance effects."""

    seg_id: str
    traj_id: str
    t_start: int
    t_end: int
    skill_label: str  # existing skill id or "NEW"

    # Smoothed predicate probabilities at segment boundaries
    P_start: Dict[str, float] = field(default_factory=dict)
    P_end: Dict[str, float] = field(default_factory=dict)

    # UI events observed within the segment (optional)
    events: List[str] = field(default_factory=list)

    # Booleanized predicate sets
    B_start: Set[str] = field(default_factory=set)
    B_end: Set[str] = field(default_factory=set)

    # Per-instance effects (computed in Step 2)
    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    eff_event: Set[str] = field(default_factory=set)

    # Cumulative game reward earned during this segment (optional, 0.0 if unknown)
    cumulative_reward: float = 0.0

    def effect_signature(self) -> str:
        """Canonical string for debugging: ``A:p1,p2|D:q1,q2|E:e1``."""
        a = ",".join(sorted(self.eff_add)) if self.eff_add else ""
        d = ",".join(sorted(self.eff_del)) if self.eff_del else ""
        e = ",".join(sorted(self.eff_event)) if self.eff_event else ""
        return f"A:{a}|D:{d}|E:{e}"

    def to_dict(self) -> dict:
        return {
            "seg_id": self.seg_id,
            "traj_id": self.traj_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "skill_label": self.skill_label,
            "P_start": self.P_start,
            "P_end": self.P_end,
            "events": self.events,
            "B_start": sorted(self.B_start),
            "B_end": sorted(self.B_end),
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "eff_event": sorted(self.eff_event),
        }

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
            events=d.get("events", []),
            B_start=set(d.get("B_start", [])),
            B_end=set(d.get("B_end", [])),
            eff_add=set(d.get("eff_add", [])),
            eff_del=set(d.get("eff_del", [])),
            eff_event=set(d.get("eff_event", [])),
        )


@dataclass
class SkillEffectsContract:
    """Effects-only contract for one skill (MVP: no pre/inv)."""

    skill_id: str
    version: int = 1

    # Human-readable name and description (e.g. from LLM when materializing new skills)
    name: Optional[str] = None
    description: Optional[str] = None

    eff_add: Set[str] = field(default_factory=set)
    eff_del: Set[str] = field(default_factory=set)
    eff_event: Set[str] = field(default_factory=set)

    # Per-literal instance support count
    support: Dict[str, int] = field(default_factory=dict)
    n_instances: int = 0

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def bump_version(self) -> None:
        self.version += 1
        self.updated_at = time.time()

    @property
    def total_literals(self) -> int:
        return len(self.eff_add) + len(self.eff_del) + len(self.eff_event)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "eff_add": sorted(self.eff_add),
            "eff_del": sorted(self.eff_del),
            "eff_event": sorted(self.eff_event),
            "support": self.support,
            "n_instances": self.n_instances,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillEffectsContract:
        return cls(
            skill_id=d["skill_id"],
            version=d.get("version", 1),
            name=d.get("name"),
            description=d.get("description"),
            eff_add=set(d.get("eff_add", [])),
            eff_del=set(d.get("eff_del", [])),
            eff_event=set(d.get("eff_event", [])),
            support=d.get("support", {}),
            n_instances=d.get("n_instances", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


@dataclass
class VerificationReport:
    """Per-skill verification metrics for an effects-only contract."""

    skill_id: str
    n_instances: int = 0

    eff_add_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_del_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_event_rate: Dict[str, float] = field(default_factory=dict)

    overall_pass_rate: float = 0.0
    worst_segments: List[str] = field(default_factory=list)
    failure_signatures: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "n_instances": self.n_instances,
            "eff_add_success_rate": self.eff_add_success_rate,
            "eff_del_success_rate": self.eff_del_success_rate,
            "eff_event_rate": self.eff_event_rate,
            "overall_pass_rate": self.overall_pass_rate,
            "worst_segments": self.worst_segments,
            "failure_signatures": self.failure_signatures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VerificationReport:
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────
# New schema types for the Skill Architecture Redesign
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SubEpisodeRef:
    """Lightweight pointer to a stored rollout segment that evidences a skill.

    This is **not** a data container — it carries no actual ``Experience``
    objects.  The real rollout data lives in the episode/rollout storage
    (``Episode_Buffer``, rollout JSONL files, etc.).  This ref is just an
    index into that storage plus cached metadata so the skill agent can
    reason about quality without loading the full trajectory.

    Stored inside ``Skill.sub_episodes`` (evidence store) — never exposed
    to the decision agent.

    Fields
    ------
    Pointer:
        episode_id, seg_start, seg_end — locate the segment in rollout storage.
        rollout_source — path/key to the rollout file or buffer that holds the
            actual Experience data (e.g. "rollouts/ep_001.json").

    Cached summary (produced once during data processing):
        summary — one-sentence description of what happened in this segment.
        intention_tags — tag sequence observed (e.g. ["MERGE", "MERGE", "POSITION"]).

    Quality metadata (updated by the skill agent quality pipeline):
        outcome, cumulative_reward, quality_score.
    """

    # ── Pointer ──────────────────────────────────────────────────────
    episode_id: str = ""
    seg_start: int = 0
    seg_end: int = 0
    rollout_source: str = ""

    # ── Cached summary ───────────────────────────────────────────────
    summary: str = ""
    intention_tags: List[str] = field(default_factory=list)

    # ── Quality metadata ─────────────────────────────────────────────
    outcome: str = "partial"  # "success" | "partial" | "failure"
    cumulative_reward: float = 0.0
    quality_score: float = 0.0
    added_at: float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        return self.seg_end - self.seg_start + 1

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "seg_start": self.seg_start,
            "seg_end": self.seg_end,
            "rollout_source": self.rollout_source,
            "summary": self.summary,
            "intention_tags": self.intention_tags,
            "outcome": self.outcome,
            "cumulative_reward": self.cumulative_reward,
            "quality_score": self.quality_score,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubEpisodeRef:
        return cls(
            episode_id=d.get("episode_id", ""),
            seg_start=d.get("seg_start", 0),
            seg_end=d.get("seg_end", 0),
            rollout_source=d.get("rollout_source", ""),
            summary=d.get("summary", ""),
            intention_tags=d.get("intention_tags", []),
            outcome=d.get("outcome", "partial"),
            cumulative_reward=d.get("cumulative_reward", 0.0),
            quality_score=d.get("quality_score", 0.0),
            added_at=d.get("added_at", 0.0),
        )


@dataclass
class Protocol:
    """Actionable decision guidance that the decision agent follows.

    Contains preconditions (when to invoke), ordered steps, success/abort
    criteria, and expected duration.  Updated when new high-quality
    sub-episodes provide better strategies.

    Extended fields (all optional, backward-compatible):

    ``step_checks``
        Per-step completion conditions (same length as ``steps``).  Each
        entry is a ``key=value`` predicate or short observable condition
        that must be true before the agent advances to the next step.
        Empty string means "advance after one timestep".

    ``predicate_success`` / ``predicate_abort``
        Machine-checkable ``key=value`` or ``key<N`` / ``key>N``
        conditions evaluated against the parsed ``summary_state`` dict.
        Used *alongside* the free-text ``success_criteria`` /
        ``abort_criteria`` (which remain for prompt display).

    ``action_vocab``
        The set of primitive action names available in this game,
        recorded so protocol steps can reference real actions.
    """

    preconditions: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    abort_criteria: List[str] = field(default_factory=list)
    expected_duration: int = 10

    step_checks: List[str] = field(default_factory=list)
    predicate_success: List[str] = field(default_factory=list)
    predicate_abort: List[str] = field(default_factory=list)
    action_vocab: List[str] = field(default_factory=list)
    source: str = "template"

    def to_dict(self) -> dict:
        d = {
            "preconditions": self.preconditions,
            "steps": self.steps,
            "success_criteria": self.success_criteria,
            "abort_criteria": self.abort_criteria,
            "expected_duration": self.expected_duration,
        }
        if self.step_checks:
            d["step_checks"] = self.step_checks
        if self.predicate_success:
            d["predicate_success"] = self.predicate_success
        if self.predicate_abort:
            d["predicate_abort"] = self.predicate_abort
        if self.action_vocab:
            d["action_vocab"] = self.action_vocab
        if self.source and self.source != "template":
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Protocol:
        if not d:
            return cls()
        return cls(
            preconditions=d.get("preconditions", []),
            steps=d.get("steps", []),
            success_criteria=d.get("success_criteria", []),
            abort_criteria=d.get("abort_criteria", []),
            expected_duration=d.get("expected_duration", 10),
            step_checks=d.get("step_checks", []),
            predicate_success=d.get("predicate_success", []),
            predicate_abort=d.get("predicate_abort", []),
            action_vocab=d.get("action_vocab", []),
            source=d.get("source", "template"),
        )


@dataclass
class ScoredBoundary:
    """Boundary candidate with penalty-based confidence score.

    Returned by ``IntentionSignalExtractor.score_boundary_candidates()``
    instead of flat timestep lists.  Stage 2 uses ``score`` as a prior
    in the segmentation decode.
    """

    time: int = 0
    score: float = 1.0
    tag_before: str = ""
    tag_after: str = ""
    is_ping_pong: bool = False
    time_since_last: int = 0

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "score": self.score,
            "tag_before": self.tag_before,
            "tag_after": self.tag_after,
            "is_ping_pong": self.is_ping_pong,
            "time_since_last": self.time_since_last,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScoredBoundary:
        return cls(
            time=d.get("time", 0),
            score=d.get("score", 1.0),
            tag_before=d.get("tag_before", ""),
            tag_after=d.get("tag_after", ""),
            is_ping_pong=d.get("is_ping_pong", False),
            time_since_last=d.get("time_since_last", 0),
        )


@dataclass
class Skill:
    """Strategic concept stored as two disjoint parts.

    **Part 1 — Protocol Store** (queried by the decision agent):
        name, strategic_description, tags, protocol, confidence.
        Accessed via ``to_decision_agent_view()``.
        Contains everything the decision agent needs to decide whether and
        how to execute this skill.  No raw trajectory data.

    **Part 2 — Evidence Store** (managed by the skill agent):
        sub_episodes — list of ``SubEpisodeRef`` pointers into rollout
        storage + cached summaries.  The actual ``Experience`` data lives
        in episode buffers / rollout files; sub_episodes are lightweight
        pointers with summary text so the skill agent can reason about
        quality, aggregate evidence, and update the protocol without
        loading full trajectories.

    The ``contract`` (eff_add/eff_del/eff_event) bridges both parts:
    it's derived from evidence and consumed by the protocol update loop.

    Backward compatible: old ``skill_bank.jsonl`` files that only have
    ``contract`` are loadable as ``Skill`` objects with empty ``protocol``
    and ``sub_episodes``.
    """

    skill_id: str = ""
    version: int = 1

    # ═══════════════════════════════════════════════════════════════
    # Part 1 — Protocol Store (decision agent queries this)
    # ═══════════════════════════════════════════════════════════════
    name: str = ""
    strategic_description: str = ""
    tags: List[str] = field(default_factory=list)
    protocol: Protocol = field(default_factory=Protocol)

    # ═══════════════════════════════════════════════════════════════
    # Part 2 — Evidence Store (skill agent manages this)
    #   sub_episodes: pointers to stored rollouts + summaries
    #   contract: effects derived from evidence
    # ═══════════════════════════════════════════════════════════════
    contract: Optional[SkillEffectsContract] = None
    sub_episodes: List[SubEpisodeRef] = field(default_factory=list)
    expected_tag_pattern: List[str] = field(default_factory=list)
    execution_hint: Optional[ExecutionHint] = None

    # Protocol version history for rollback
    protocol_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    n_instances: int = 0
    retired: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Version management ───────────────────────────────────────

    def bump_version(self) -> None:
        if self.protocol.steps:
            self.protocol_history.append({
                "version": self.version,
                "protocol": self.protocol.to_dict(),
                "timestamp": time.time(),
            })
            if len(self.protocol_history) > 5:
                self.protocol_history = self.protocol_history[-5:]
        self.version += 1
        self.updated_at = time.time()

    # ── Evidence statistics ──────────────────────────────────────

    @property
    def success_rate(self) -> float:
        if not self.sub_episodes:
            return 0.0
        successes = sum(1 for se in self.sub_episodes if se.outcome == "success")
        return successes / len(self.sub_episodes)

    @property
    def confidence(self) -> float:
        """Confidence based on instance count and success rate."""
        if self.n_instances == 0:
            return 0.0
        evidence_factor = min(1.0, self.n_instances / 10.0)
        return evidence_factor * self.success_rate

    def compute_skill_score(
        self,
        n_selections: int = 0,
        n_total_selections: int = 0,
        n_bank_skills: int = 1,
        contract_pass_rate: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Composite skill quality score in [0, 1].

        Five components (configurable via *weights* dict):
          mean_segment_quality (0.30) -- mean quality_score across sub-episodes
          reuse_success        (0.25) -- successful reuse across instances
          contract_pass        (0.20) -- contract verification pass rate
          cross_ep_consistency (0.15) -- reward consistency across episodes
          exploration_value    (0.10) -- bonus for new skills with evidence

        Backward compatible: old callers without new args still work;
        the usage_frequency parameter is accepted but no longer dominant.
        """
        w = {
            "mean_segment_quality": 0.30,
            "reuse_success": 0.25,
            "contract_pass": 0.20,
            "cross_ep_consistency": 0.15,
            "exploration_value": 0.10,
        }
        if weights:
            w.update(weights)

        if not self.sub_episodes:
            return 0.5

        # -- mean_segment_quality: mean quality_score across sub-episodes
        r_segment = sum(se.quality_score for se in self.sub_episodes) / len(self.sub_episodes)

        # -- reuse_success: fraction of sub-episodes with successful outcomes
        n_success = sum(1 for se in self.sub_episodes if se.outcome == "success")
        n_used = len(self.sub_episodes)
        r_reuse = n_success / max(n_used, 1)
        reuse_evidence = min(1.0, n_used / 5.0)
        r_reuse *= reuse_evidence

        # -- contract_pass: contract verification pass rate
        if contract_pass_rate is not None:
            r_contract = min(1.0, contract_pass_rate)
        elif self.contract and self.contract.total_literals > 0:
            r_contract = 0.5
        else:
            r_contract = 0.0

        # -- cross_ep_consistency: reward consistency across different episodes
        ep_rewards: Dict[str, List[float]] = {}
        for se in self.sub_episodes:
            eid = se.episode_id or "unknown"
            ep_rewards.setdefault(eid, []).append(se.cumulative_reward)

        if len(ep_rewards) >= 2:
            ep_means = [sum(rs) / len(rs) for rs in ep_rewards.values()]
            overall_mean = sum(ep_means) / len(ep_means)
            if overall_mean != 0:
                cv = (sum((m - overall_mean) ** 2 for m in ep_means) / len(ep_means)) ** 0.5 / (abs(overall_mean) + 1e-8)
                r_consistency = max(0.0, 1.0 - min(cv, 2.0) / 2.0)
            else:
                r_consistency = 0.5
        else:
            r_consistency = 0.5

        # -- exploration_value: new skills with growing evidence get a bonus
        is_young = n_used <= 5
        has_contract = self.contract is not None and self.contract.total_literals > 0
        if is_young and has_contract and r_segment > 0.3:
            r_exploration = min(1.0, r_segment * 1.5)
        else:
            r_exploration = 0.0

        return max(0.0, min(1.0,
            w["mean_segment_quality"] * r_segment
            + w["reuse_success"] * r_reuse
            + w["contract_pass"] * r_contract
            + w["cross_ep_consistency"] * r_consistency
            + w["exploration_value"] * r_exploration,
        ))

    @property
    def evidence_summaries(self) -> List[str]:
        """Collect summaries from all sub-episode refs (for skill agent reasoning)."""
        return [se.summary for se in self.sub_episodes if se.summary]

    # ── Views ────────────────────────────────────────────────────

    def to_decision_agent_view(self) -> Dict[str, Any]:
        """Part 1 only: protocol + metadata the decision agent needs.

        Never includes sub-episodes, raw contract internals, or rollout
        pointers.
        """
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "strategic_description": self.strategic_description,
            "protocol": self.protocol.to_dict(),
            "confidence": round(self.confidence, 3),
            "expected_duration": self.protocol.expected_duration,
            "tags": self.tags,
        }

    def to_evidence_view(self) -> Dict[str, Any]:
        """Part 2 only: evidence pointers + summaries for the skill agent.

        Includes sub-episode refs (with summaries and quality scores) but
        not the actual Experience data (that lives in rollout storage).
        """
        return {
            "skill_id": self.skill_id,
            "n_instances": self.n_instances,
            "success_rate": round(self.success_rate, 3),
            "sub_episodes": [se.to_dict() for se in self.sub_episodes],
            "evidence_summaries": self.evidence_summaries,
            "expected_tag_pattern": self.expected_tag_pattern,
            "retired": self.retired,
        }

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "name": self.name,
            "strategic_description": self.strategic_description,
            "tags": self.tags,
            "protocol": self.protocol.to_dict(),
            "contract": self.contract.to_dict() if self.contract else None,
            "sub_episodes": [se.to_dict() for se in self.sub_episodes],
            "expected_tag_pattern": self.expected_tag_pattern,
            "execution_hint": self.execution_hint.to_dict() if self.execution_hint else None,
            "protocol_history": self.protocol_history,
            "n_instances": self.n_instances,
            "retired": self.retired,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        contract_data = d.get("contract")
        contract = SkillEffectsContract.from_dict(contract_data) if contract_data else None

        sub_eps = [SubEpisodeRef.from_dict(se) for se in d.get("sub_episodes", [])]
        protocol = Protocol.from_dict(d.get("protocol", {}))
        exec_hint_data = d.get("execution_hint")
        exec_hint = ExecutionHint.from_dict(exec_hint_data) if exec_hint_data else None

        return cls(
            skill_id=d.get("skill_id", ""),
            version=d.get("version", 1),
            name=d.get("name", ""),
            strategic_description=d.get("strategic_description", ""),
            tags=d.get("tags", []),
            protocol=protocol,
            contract=contract,
            sub_episodes=sub_eps,
            expected_tag_pattern=d.get("expected_tag_pattern", []),
            execution_hint=exec_hint,
            protocol_history=d.get("protocol_history", []),
            n_instances=d.get("n_instances", 0),
            retired=d.get("retired", False),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )

    @classmethod
    def from_contract(cls, contract: SkillEffectsContract) -> Skill:
        """Wrap an existing SkillEffectsContract as a Skill (migration helper)."""
        return cls(
            skill_id=contract.skill_id,
            version=contract.version,
            name=contract.name or "",
            strategic_description=contract.description or "",
            contract=contract,
            n_instances=contract.n_instances,
            created_at=contract.created_at,
            updated_at=contract.updated_at,
        )


# ─────────────────────────────────────────────────────────────────────
# Proto-Skill: intermediate representation between NEW and real skills
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ProtoSkill:
    """Lightweight intermediate skill representation.

    Sits between raw ``__NEW__`` segments and fully materialized ``Skill``
    objects.  A proto-skill collects evidence gradually and can be used
    by Stage 2 as a candidate label *before* full promotion.

    Lifecycle::

        __NEW__ segments → cluster → ProtoSkill → light verification → Skill

    Fields provide enough structure for Stage 2 to reason about the
    proto-skill without requiring a full effects contract or protocol.
    """

    proto_id: str = ""
    member_seg_ids: List[str] = field(default_factory=list)
    candidate_effects_add: Set[str] = field(default_factory=set)
    candidate_effects_del: Set[str] = field(default_factory=set)
    candidate_effects_event: Set[str] = field(default_factory=set)
    support: int = 0
    consistency: float = 0.0
    separability: float = 0.0
    tag_distribution: Dict[str, int] = field(default_factory=dict)
    typical_length_mean: float = 10.0
    typical_length_std: float = 5.0
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

    # Verification status
    verified: bool = False
    verification_pass_rate: float = 0.0
    n_verifications: int = 0

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    _relaxed_thresholds: "ClassVar[bool]" = False

    @classmethod
    def set_relaxed(cls, relaxed: bool) -> None:
        """Enable relaxed thresholds for early training steps (0-15).

        When relaxed, support drops from 5 to 3, consistency from 0.5
        to 0.35, and pass_rate from 0.6 to 0.4, allowing more
        proto-skills to promote when data is scarce.
        """
        cls._relaxed_thresholds = relaxed

    @property
    def is_promotable(self) -> bool:
        """Whether the proto-skill has enough evidence for full promotion."""
        if self._relaxed_thresholds:
            return (
                self.support >= 3
                and self.consistency >= 0.35
                and self.verification_pass_rate >= 0.4
                and self.n_verifications >= 1
            )
        return (
            self.support >= 5
            and self.consistency >= 0.5
            and self.verification_pass_rate >= 0.6
            and self.n_verifications >= 1
        )

    @property
    def candidate_label(self) -> str:
        """Label that Stage 2 can use as a candidate during decoding."""
        return f"__PROTO__{self.proto_id}"

    def to_skill(self) -> Skill:
        """Promote this proto-skill to a full Skill."""
        contract = SkillEffectsContract(
            skill_id=self.proto_id,
            eff_add=set(self.candidate_effects_add),
            eff_del=set(self.candidate_effects_del),
            eff_event=set(self.candidate_effects_event),
            n_instances=self.support,
        )

        top_tags = sorted(
            self.tag_distribution.items(), key=lambda x: -x[1]
        )[:5]

        return Skill(
            skill_id=self.proto_id,
            contract=contract,
            tags=[t for t, _ in top_tags],
            n_instances=self.support,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict:
        return {
            "proto_id": self.proto_id,
            "member_seg_ids": self.member_seg_ids,
            "candidate_effects_add": sorted(self.candidate_effects_add),
            "candidate_effects_del": sorted(self.candidate_effects_del),
            "candidate_effects_event": sorted(self.candidate_effects_event),
            "support": self.support,
            "consistency": self.consistency,
            "separability": self.separability,
            "tag_distribution": self.tag_distribution,
            "typical_length_mean": self.typical_length_mean,
            "typical_length_std": self.typical_length_std,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "verified": self.verified,
            "verification_pass_rate": self.verification_pass_rate,
            "n_verifications": self.n_verifications,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProtoSkill:
        return cls(
            proto_id=d.get("proto_id", ""),
            member_seg_ids=d.get("member_seg_ids", []),
            candidate_effects_add=set(d.get("candidate_effects_add", [])),
            candidate_effects_del=set(d.get("candidate_effects_del", [])),
            candidate_effects_event=set(d.get("candidate_effects_event", [])),
            support=d.get("support", 0),
            consistency=d.get("consistency", 0.0),
            separability=d.get("separability", 0.0),
            tag_distribution=d.get("tag_distribution", {}),
            typical_length_mean=d.get("typical_length_mean", 10.0),
            typical_length_std=d.get("typical_length_std", 5.0),
            context_before=d.get("context_before", []),
            context_after=d.get("context_after", []),
            verified=d.get("verified", False),
            verification_pass_rate=d.get("verification_pass_rate", 0.0),
            n_verifications=d.get("n_verifications", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


@dataclass
class ExecutionHint:
    """Distilled execution guidance derived from successful segments.

    Stored per-skill, bridges the gap from "what this skill achieves"
    to "how to carry it out".  See Phase 5 of the agentic skill-memory plan.
    """

    common_preconditions: List[str] = field(default_factory=list)
    common_target_objects: List[str] = field(default_factory=list)
    state_transition_pattern: str = ""
    termination_cues: List[str] = field(default_factory=list)
    common_failure_modes: List[str] = field(default_factory=list)
    execution_description: str = ""
    n_source_segments: int = 0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "common_preconditions": self.common_preconditions,
            "common_target_objects": self.common_target_objects,
            "state_transition_pattern": self.state_transition_pattern,
            "termination_cues": self.termination_cues,
            "common_failure_modes": self.common_failure_modes,
            "execution_description": self.execution_description,
            "n_source_segments": self.n_source_segments,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionHint:
        if not d:
            return cls()
        return cls(
            common_preconditions=d.get("common_preconditions", []),
            common_target_objects=d.get("common_target_objects", []),
            state_transition_pattern=d.get("state_transition_pattern", ""),
            termination_cues=d.get("termination_cues", []),
            common_failure_modes=d.get("common_failure_modes", []),
            execution_description=d.get("execution_description", ""),
            n_source_segments=d.get("n_source_segments", 0),
            updated_at=d.get("updated_at", 0.0),
        )
