"""
Shared rollout record schema and metric aggregation.

RolloutRecord is the single source of truth consumed by both the Decision
Agent GRPO trainer and the SkillBank Hard-EM trainer.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Core data record — one per timestep
# ---------------------------------------------------------------------------

@dataclass
class RolloutStep:
    """Single timestep within a rollout."""

    step: int = 0
    obs_id: str = ""
    action: str = ""
    action_type: str = "primitive"  # primitive | QUERY_MEM | QUERY_SKILL | CALL_SKILL

    ui_events: List[str] = field(default_factory=list)
    predicates: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    r_env: float = 0.0
    r_follow: float = 0.0
    r_cost: float = 0.0
    r_total: float = 0.0

    done: bool = False
    episode_id: str = ""
    traj_id: str = ""
    seed: int = 0

    active_skill_id: Optional[str] = None
    query_key: Optional[str] = None

    # Strategy C: intention tag from the decision agent (e.g. "[MERGE] combine tiles")
    intentions: Optional[str] = None

    logprob: Optional[float] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.embedding is not None:
            d.pop("embedding", None)
        return d


@dataclass
class RolloutRecord:
    """Complete episode rollout — list of steps with episode metadata."""

    episode_id: str = ""
    traj_id: str = ""
    seed: int = 0
    env_name: str = ""
    game_name: str = ""

    steps: List[RolloutStep] = field(default_factory=list)

    total_reward: float = 0.0
    total_r_env: float = 0.0
    total_r_follow: float = 0.0
    total_r_cost: float = 0.0
    episode_length: int = 0
    won: bool = False
    score: float = 0.0

    def finalize(self) -> None:
        """Compute episode-level aggregates from steps."""
        self.episode_length = len(self.steps)
        self.total_r_env = sum(s.r_env for s in self.steps)
        self.total_r_follow = sum(s.r_follow for s in self.steps)
        self.total_r_cost = sum(s.r_cost for s in self.steps)
        self.total_reward = sum(s.r_total for s in self.steps)

    def action_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for s in self.steps:
            counts[s.action_type] += 1
        return dict(counts)

    def skill_switch_count(self) -> int:
        switches = 0
        prev_skill: Optional[str] = None
        for s in self.steps:
            if s.active_skill_id != prev_skill and prev_skill is not None:
                switches += 1
            prev_skill = s.active_skill_id
        return switches

    def query_keys(self) -> List[str]:
        return [s.query_key for s in self.steps if s.query_key]


# ---------------------------------------------------------------------------
# Metric aggregation across episodes
# ---------------------------------------------------------------------------

@dataclass
class DecisionMetrics:
    """Aggregated metrics for the Decision Agent."""

    win_rate: float = 0.0
    mean_score: float = 0.0
    mean_reward: float = 0.0
    mean_r_env: float = 0.0
    mean_r_follow: float = 0.0
    mean_r_cost: float = 0.0
    mean_episode_length: float = 0.0

    query_skill_rate: float = 0.0
    query_mem_rate: float = 0.0
    call_skill_rate: float = 0.0
    skill_switch_rate: float = 0.0
    mean_query_key_len: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def aggregate_decision_metrics(rollouts: Sequence[RolloutRecord]) -> DecisionMetrics:
    """Compute aggregate metrics over a batch of rollout records."""
    if not rollouts:
        return DecisionMetrics()

    n = len(rollouts)
    m = DecisionMetrics()

    m.win_rate = sum(1 for r in rollouts if r.won) / n
    m.mean_score = statistics.mean(r.score for r in rollouts)
    m.mean_reward = statistics.mean(r.total_reward for r in rollouts)
    m.mean_r_env = statistics.mean(r.total_r_env for r in rollouts)
    m.mean_r_follow = statistics.mean(r.total_r_follow for r in rollouts)
    m.mean_r_cost = statistics.mean(r.total_r_cost for r in rollouts)
    m.mean_episode_length = statistics.mean(r.episode_length for r in rollouts)

    total_steps = sum(r.episode_length for r in rollouts) or 1
    all_counts: Dict[str, int] = defaultdict(int)
    for r in rollouts:
        for k, v in r.action_type_counts().items():
            all_counts[k] += v

    m.query_skill_rate = all_counts.get("QUERY_SKILL", 0) / total_steps
    m.query_mem_rate = all_counts.get("QUERY_MEM", 0) / total_steps
    m.call_skill_rate = all_counts.get("CALL_SKILL", 0) / total_steps
    m.skill_switch_rate = sum(r.skill_switch_count() for r in rollouts) / total_steps

    all_keys = []
    for r in rollouts:
        all_keys.extend(r.query_keys())
    m.mean_query_key_len = statistics.mean(len(k) for k in all_keys) if all_keys else 0.0

    return m


@dataclass
class SkillBankMetrics:
    """Aggregated metrics for the SkillBank Agent."""

    n_skills: int = 0
    new_pool_size: int = 0
    mean_pass_rate: float = 0.0
    mean_margin: float = 0.0
    n_refine: int = 0
    n_materialize: int = 0
    n_merge: int = 0
    n_split: int = 0
    bank_size_growth: float = 0.0
    churn_rate: float = 0.0
    confusion_pairs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
