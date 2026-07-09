"""
Ingest recent Decision Agent rollouts for SkillBank Hard-EM training.

Converts RolloutRecords (from the decision agent buffer) into trajectory
objects suitable for the EM pipeline stages (boundary proposal, decode, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from trainer.common.metrics import RolloutRecord, RolloutStep

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryFrame:
    """Single frame within a trajectory, enriched for EM processing."""

    t: int = 0
    obs_id: str = ""
    observation_text: str = ""
    action: str = ""
    action_type: str = "primitive"
    ui_events: List[str] = field(default_factory=list)
    predicates: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    r_env: float = 0.0
    active_skill_id: Optional[str] = None
    intentions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        if self.embedding is not None:
            d.pop("embedding", None)
        return d


@dataclass
class TrajectoryForEM:
    """Complete trajectory prepared for Hard-EM processing."""

    traj_id: str = ""
    episode_id: str = ""
    env_name: str = ""
    seed: int = 0
    frames: List[TrajectoryFrame] = field(default_factory=list)
    total_reward: float = 0.0
    won: bool = False
    length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "episode_id": self.episode_id,
            "env_name": self.env_name,
            "seed": self.seed,
            "length": self.length,
            "total_reward": self.total_reward,
            "won": self.won,
        }


def ingest_rollouts(
    rollouts: Sequence[RolloutRecord],
    observation_store: Optional[Dict[str, str]] = None,
    max_trajectories: Optional[int] = None,
) -> List[TrajectoryForEM]:
    """Convert Decision Agent rollout records to trajectory objects for EM.

    Args:
        rollouts: sequence of RolloutRecords from the replay buffer
        observation_store: optional mapping obs_id -> observation text
            (if step.obs_id was a pointer rather than inline text)
        max_trajectories: limit the number of trajectories to process

    Returns:
        List of TrajectoryForEM objects ready for Stage 1/2 processing.
    """
    trajectories: List[TrajectoryForEM] = []
    obs_store = observation_store or {}

    source = rollouts
    if max_trajectories is not None:
        source = rollouts[-max_trajectories:]

    for record in source:
        frames: List[TrajectoryFrame] = []
        for step in record.steps:
            obs_text = obs_store.get(step.obs_id, step.obs_id)

            frame = TrajectoryFrame(
                t=step.step,
                obs_id=step.obs_id,
                observation_text=obs_text,
                action=step.action,
                action_type=step.action_type,
                ui_events=step.ui_events,
                predicates=step.predicates,
                embedding=step.embedding,
                r_env=step.r_env,
                active_skill_id=step.active_skill_id,
                intentions=step.intentions,
            )
            frames.append(frame)

        traj = TrajectoryForEM(
            traj_id=record.traj_id or record.episode_id,
            episode_id=record.episode_id,
            env_name=record.env_name,
            seed=record.seed,
            frames=frames,
            total_reward=record.total_reward,
            won=record.won,
            length=len(frames),
        )
        trajectories.append(traj)

    logger.info(
        "Ingested %d trajectories (%d total frames)",
        len(trajectories),
        sum(t.length for t in trajectories),
    )
    return trajectories


def split_holdout(
    trajectories: List[TrajectoryForEM],
    holdout_fraction: float = 0.1,
) -> tuple:
    """Split trajectories into train and holdout sets for EM gating.

    Returns (train_trajs, holdout_trajs).
    """
    n = len(trajectories)
    n_holdout = max(1, int(n * holdout_fraction))
    return trajectories[:-n_holdout], trajectories[-n_holdout:]
