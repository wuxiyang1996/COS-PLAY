"""
Deterministic seed management for reproducible training and evaluation.

Ensures both the Decision Agent rollouts and SkillBank evaluation use
identical seeds for fair comparison across bank versions.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import List, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and optionally PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_eval_seeds(n: int = 8, base_seed: int = 42) -> List[int]:
    """Generate a deterministic list of evaluation seeds."""
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def get_train_seed(episode: int, base_seed: int = 0) -> int:
    """Deterministic per-episode training seed."""
    h = hashlib.sha256(f"{base_seed}_{episode}".encode()).hexdigest()
    return int(h[:8], 16)


class SeedManager:
    """Manages seed sequences for training and evaluation.

    Provides separate reproducible seed streams for rollout collection,
    evaluation runs, and SkillBank EM iterations.
    """

    def __init__(self, base_seed: int = 42, eval_seeds: Optional[List[int]] = None):
        self.base_seed = base_seed
        self.eval_seeds = eval_seeds or get_eval_seeds(8, base_seed)
        self._train_counter = 0

    def next_train_seed(self) -> int:
        seed = get_train_seed(self._train_counter, self.base_seed)
        self._train_counter += 1
        return seed

    def get_eval_seed(self, idx: int) -> int:
        return self.eval_seeds[idx % len(self.eval_seeds)]

    def reset_train_counter(self) -> None:
        self._train_counter = 0
