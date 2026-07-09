"""
Episode replay buffer for GRPO training.

Stores complete RolloutRecords with optional prioritized sampling.
Supports min-episode warmup and capacity management.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Sequence

import numpy as np

from trainer.common.metrics import RolloutRecord


class ReplayBuffer:
    """Fixed-capacity episode replay buffer with optional prioritized sampling.

    Episodes are stored as complete RolloutRecords. Priority is based on
    episode return (higher return → higher priority for GRPO group sampling).
    """

    def __init__(
        self,
        capacity: int = 10000,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        min_episodes: int = 64,
    ):
        self.capacity = capacity
        self.alpha = priority_alpha
        self.beta = priority_beta
        self.min_episodes = min_episodes
        self._buffer: deque[RolloutRecord] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)

    def add(self, record: RolloutRecord) -> None:
        """Add an episode to the buffer."""
        if not record.steps:
            return
        if record.episode_length == 0:
            record.finalize()
        self._buffer.append(record)
        priority = abs(record.total_reward) + 1e-6
        self._priorities.append(priority)

    def add_batch(self, records: Sequence[RolloutRecord]) -> None:
        """Add a batch of episodes."""
        for r in records:
            self.add(r)

    def sample(self, batch_size: int, prioritized: bool = True) -> List[RolloutRecord]:
        """Sample a batch of episodes.

        Args:
            batch_size: number of episodes to sample
            prioritized: use priority-weighted sampling

        Returns:
            list of RolloutRecords
        """
        n = len(self._buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)

        if not prioritized or self.alpha == 0:
            indices = random.sample(range(n), batch_size)
        else:
            priorities = np.array(list(self._priorities), dtype=np.float64)
            priorities = priorities ** self.alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(n, size=batch_size, replace=False, p=probs).tolist()

        return [self._buffer[i] for i in indices]

    def sample_recent(self, batch_size: int) -> List[RolloutRecord]:
        """Sample from the most recent episodes."""
        n = len(self._buffer)
        if n == 0:
            return []
        k = min(batch_size, n)
        return list(self._buffer)[-k:]

    def is_ready(self) -> bool:
        """Whether the buffer has enough episodes to start training."""
        return len(self._buffer) >= self.min_episodes

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def all_records(self) -> List[RolloutRecord]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._priorities.clear()

    def stats(self) -> dict:
        if not self._buffer:
            return {"size": 0}
        rewards = [r.total_reward for r in self._buffer]
        return {
            "size": len(self._buffer),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean([r.episode_length for r in self._buffer])),
        }
