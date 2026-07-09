"""
Optional: Top-2 tie-breaker — small supervised learner that resolves
ambiguous skill assignments when the top-2 decode candidates have
similar scores.

Trained on EM-decoded segmentation where margins are low, using features
from the segment to predict the correct label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trainer.skillbank.ingest_rollouts import TrajectoryForEM
from trainer.skillbank.stages.stage2_decode import DecodeResult, DecodedSegment

logger = logging.getLogger(__name__)


@dataclass
class TiebreakerConfig:
    """Configuration for the tie-breaker model."""

    margin_threshold: float = 1.0
    hidden_dim: int = 32
    lr: float = 1e-3
    epochs: int = 15
    batch_size: int = 64
    min_training_samples: int = 50


@dataclass
class TiebreakerTrainResult:
    """Training result for the tie-breaker."""

    n_ties: int = 0
    n_resolved: int = 0
    accuracy: float = 0.0
    mean_margin_before: float = 0.0


def extract_tie_cases(
    decode_results: List[DecodeResult],
    margin_threshold: float = 1.0,
) -> List[Tuple[DecodedSegment, str, str]]:
    """Find segments where top-2 skills are within margin_threshold.

    Returns list of (segment, best_label, runner_up_label).
    """
    ties: List[Tuple[DecodedSegment, str, str]] = []
    for dr in decode_results:
        for seg in dr.segments:
            if seg.margin < margin_threshold and seg.runner_up:
                ties.append((seg, seg.skill_label, seg.runner_up))
    return ties


def extract_tiebreaker_features(
    segment: DecodedSegment,
    trajectory: Optional[TrajectoryForEM] = None,
) -> np.ndarray:
    """Extract features for tie-breaking between two candidate skills.

    Features:
      - Segment length
      - Number of eff_add / eff_del predicates
      - Best score and runner-up score
      - Margin
    """
    features = [
        float(segment.t_end - segment.t_start),
        float(len(segment.eff_add)),
        float(len(segment.eff_del)),
        float(len(segment.B_start)),
        float(len(segment.B_end)),
        float(segment.score),
        float(segment.runner_up_score),
        float(segment.margin),
    ]
    return np.array(features, dtype=np.float32)


class TiebreakerClassifier:
    """Simple binary classifier: predict whether the current best label
    is correct (1) or the runner-up should be preferred (0).

    Uses a shallow MLP. For production, replace with sklearn or PyTorch.
    """

    def __init__(self, config: Optional[TiebreakerConfig] = None):
        self.cfg = config or TiebreakerConfig()
        self._weights: Optional[Dict[str, np.ndarray]] = None
        self._trained = False

    def train(
        self,
        decode_results: List[DecodeResult],
        ground_truth: Optional[Dict[str, str]] = None,
    ) -> TiebreakerTrainResult:
        """Train the tie-breaker on ambiguous cases.

        If ground_truth is provided (seg_id -> correct_label), uses that.
        Otherwise, treats the current EM assignment as ground truth
        (self-training on confident iterations).
        """
        ties = extract_tie_cases(decode_results, self.cfg.margin_threshold)

        if len(ties) < self.cfg.min_training_samples:
            logger.info("Too few tie cases (%d) for training", len(ties))
            return TiebreakerTrainResult(n_ties=len(ties))

        X_list, y_list = [], []
        margins = []

        for seg, best, runner_up in ties:
            feat = extract_tiebreaker_features(seg)
            X_list.append(feat)

            if ground_truth and hasattr(seg, "seg_id"):
                seg_id = f"{seg.t_start}_{seg.t_end}"
                gt = ground_truth.get(seg_id, best)
                y_list.append(1.0 if gt == best else 0.0)
            else:
                y_list.append(1.0)

            margins.append(seg.margin)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        input_dim = X.shape[1]
        h = self.cfg.hidden_dim
        rng = np.random.RandomState(42)
        W1 = rng.randn(input_dim, h).astype(np.float32) * 0.1
        b1 = np.zeros(h, dtype=np.float32)
        W2 = rng.randn(h, 1).astype(np.float32) * 0.1
        b2 = np.zeros(1, dtype=np.float32)

        lr = self.cfg.lr
        for epoch in range(self.cfg.epochs):
            indices = rng.permutation(len(X))
            for start in range(0, len(X), self.cfg.batch_size):
                batch_idx = indices[start:start + self.cfg.batch_size]
                xb, yb = X[batch_idx], y[batch_idx]

                z1 = xb @ W1 + b1
                a1 = np.maximum(z1, 0)
                z2 = (a1 @ W2 + b2).flatten()
                pred = 1.0 / (1.0 + np.exp(-np.clip(z2, -20, 20)))

                grad = (pred - yb) / len(yb)
                grad_W2 = a1.T @ grad.reshape(-1, 1)
                grad_b2 = grad.sum(keepdims=True)
                grad_a1 = grad.reshape(-1, 1) @ W2.T
                grad_z1 = grad_a1 * (z1 > 0).astype(np.float32)
                grad_W1 = xb.T @ grad_z1
                grad_b1 = grad_z1.sum(axis=0)

                W1 -= lr * grad_W1
                b1 -= lr * grad_b1
                W2 -= lr * grad_W2
                b2 -= lr * grad_b2

        self._weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        self._trained = True

        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)
        z2 = (a1 @ W2 + b2).flatten()
        preds = (1.0 / (1.0 + np.exp(-np.clip(z2, -20, 20)))) > 0.5
        accuracy = (preds == y).mean()

        return TiebreakerTrainResult(
            n_ties=len(ties),
            n_resolved=int((preds != y).sum()),
            accuracy=float(accuracy),
            mean_margin_before=float(np.mean(margins)) if margins else 0.0,
        )

    def predict(self, segment: DecodedSegment) -> float:
        """Predict probability that the current best label is correct."""
        if not self._trained or self._weights is None:
            return 0.5

        feat = extract_tiebreaker_features(segment).reshape(1, -1)
        W1, b1 = self._weights["W1"], self._weights["b1"]
        W2, b2 = self._weights["W2"], self._weights["b2"]

        z1 = feat @ W1 + b1
        a1 = np.maximum(z1, 0)
        z2 = (a1 @ W2 + b2).flatten()
        return float(1.0 / (1.0 + np.exp(-np.clip(z2[0], -20, 20))))

    def should_swap(self, segment: DecodedSegment, threshold: float = 0.4) -> bool:
        """Returns True if the runner-up should replace the best label."""
        return self.predict(segment) < threshold

    @property
    def is_trained(self) -> bool:
        return self._trained
