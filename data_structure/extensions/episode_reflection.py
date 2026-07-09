"""`EpisodeReflection` — per-episode bundle handed to the Skill Crafter.

Spec: PLAN-SKILL-CRAFTER §6.4 (Failure-Reflector mode), PLAN-PIPELINE-
ORCHESTRATOR §3a (online control plane), and the implementation note
``implementation_notes/legacy/crafter-harness-orchestrator-roles.md`` §"Two-tier
trigger" — the *per-episode reactive pass* fires immediately after the
Skill Bank Agent has produced this episode's candidates / bank-mgmt
updates, while the slower *per-batch reflective pass* (`cycle()`) runs
every K episodes for Composer / Generalizer.

This dataclass is the canonical input contract for the per-episode pass.
It is the *only* shape ``SkillCrafterService.reflect_on_episode``
accepts so the live Pipeline Orchestrator and the offline
``labeling_supplement`` mirror can both target one input schema.

Why a separate record (not just a list of FailureTraces)?
--------------------------------------------------------
The per-episode pass has access to richer context than a batch pass —
specifically the candidate skills the Skill Bank Agent just minted from
*this* episode, and the splits / merges / refines it issued during the
same step. Without these the Crafter would re-propose patches the Bank
Agent already issued (and miss subsumption decisions when a freshly-
minted candidate strictly supersedes an existing active skill).

The bundle therefore carries:

  * ``failure_traces``         — failures observed in this episode (any
                                 count; per-episode threshold is 1, not
                                 the batch ``hot_pattern_threshold``).
  * ``skill_episodes``         — `SkillEpisode` records the harness
                                 emitted (success + abort traces alike;
                                 used for context when a single failure
                                 references a successful peer skill).
  * ``new_candidate_skill_ids``— skill_ids the Bank Agent freshly
                                 minted into the candidate store from
                                 this episode; the Crafter looks them up
                                 against the active store to decide
                                 subsumption / dedup.
  * ``bank_agent_actions``     — splits / merges / refines counts and
                                 details emitted by the per-episode bank
                                 management stage (mirrors the on-disk
                                 ``bank_management_io.json::stage_4_bank_maintenance``
                                 shape). Used only for dedup against
                                 patches the Bank Agent already
                                 proposed.
  * ``outcome_summary``        — episode-level reward / win flag / etc.
                                 (free-form; used by Hypothesizer as
                                 context).

All collections default to empty so callers can construct minimal
reflections (e.g. ``EpisodeReflection(episode_id=..., failure_traces=[t])``)
without having to populate every optional field.

The Crafter never *writes* to this record — it is consumed read-only.
The four-way ownership boundary documented in
``crafter-harness-orchestrator-roles.md`` is preserved: the Bank Agent
fills ``new_candidate_skill_ids`` / ``bank_agent_actions``, the Harness
fills ``skill_episodes`` / ``failure_traces``, and the Orchestrator
glues them into one ``EpisodeReflection`` before invoking
``SkillCrafterService.reflect_on_episode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_episode import SkillEpisode


@dataclass
class EpisodeReflection:
    """Per-episode bundle for the Crafter's reactive pass."""

    episode_id: str
    domain: str = ""
    parent_run_id: Optional[str] = None
    failure_traces: List[FailureTrace] = field(default_factory=list)
    skill_episodes: List[SkillEpisode] = field(default_factory=list)
    new_candidate_skill_ids: List[str] = field(default_factory=list)
    bank_agent_actions: Dict[str, Any] = field(default_factory=dict)
    outcome_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("EpisodeReflection.episode_id must be non-empty.")
        # Defensive copies of the mutable containers so callers can't
        # mutate the reflection through their own references after handing
        # it to the service.
        self.failure_traces = list(self.failure_traces)
        self.skill_episodes = list(self.skill_episodes)
        self.new_candidate_skill_ids = list(self.new_candidate_skill_ids)
        self.bank_agent_actions = dict(self.bank_agent_actions)
        self.outcome_summary = dict(self.outcome_summary)

    @property
    def n_failures(self) -> int:
        return len(self.failure_traces)

    @property
    def has_signal(self) -> bool:
        """True iff the reflection carries something the Crafter can act on.

        A reflection with no failures *and* no fresh candidates is a no-op
        for the Crafter (a healthy episode the Bank Agent didn't touch).
        ``SkillCrafterService.reflect_on_episode`` short-circuits in that
        case so a successful run produces zero proposals — exactly the
        behaviour the Phase-1 cold-start corpus needs.
        """
        return bool(self.failure_traces) or bool(self.new_candidate_skill_ids)

    def to_json(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "parent_run_id": self.parent_run_id,
            "failure_traces": [t.to_json() for t in self.failure_traces],
            "skill_episodes": [e.to_json() for e in self.skill_episodes],
            "new_candidate_skill_ids": list(self.new_candidate_skill_ids),
            "bank_agent_actions": dict(self.bank_agent_actions),
            "outcome_summary": dict(self.outcome_summary),
        }


__all__ = ["EpisodeReflection"]
