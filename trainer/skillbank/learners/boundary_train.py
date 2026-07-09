"""
Optional: Boundary classifier — small supervised learner that predicts
whether a given timestep is a skill boundary.

Trained on EM-decoded segmentation results as supervision. Can replace
or supplement the heuristic boundary proposal in Stage 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trainer.skillbank.ingest_rollouts import TrajectoryForEM
from trainer.skillbank.stages.stage2_decode import DecodeResult

logger = logging.getLogger(__name__)


@dataclass
class BoundaryClassifierConfig:
    """Configuration for the boundary classifier."""

    feature_window: int = 5
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 128
    pos_weight: float = 3.0
    min_training_samples: int = 200


@dataclass
class BoundaryTrainResult:
    """Training result for the boundary classifier."""

    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0


def extract_boundary_labels(
    trajectories: List[TrajectoryForEM],
    decode_results: List[DecodeResult],
) -> List[Tuple[str, int, bool]]:
    """Extract (traj_id, timestep, is_boundary) labels from decode results.

    Boundary = first timestep of each decoded segment (except t=0).
    """
    labels: List[Tuple[str, int, bool]] = []
    boundaries_by_traj: Dict[str, set] = {}

    for dr in decode_results:
        boundary_times = set()
        for seg in dr.segments:
            if seg.t_start > 0:
                boundary_times.add(seg.t_start)
        boundaries_by_traj[dr.traj_id] = boundary_times

    for traj in trajectories:
        bounds = boundaries_by_traj.get(traj.traj_id, set())
        for frame in traj.frames:
            is_boundary = frame.t in bounds
            labels.append((traj.traj_id, frame.t, is_boundary))

    return labels


def extract_boundary_features(
    trajectory: TrajectoryForEM,
    t: int,
    window: int = 5,
) -> np.ndarray:
    """Extract features for boundary classification at timestep t.

    Features include:
      - Predicate change count in a window around t
      - Action type transitions
      - Simple statistics of predicate probabilities
    """
    frames = trajectory.frames
    n = len(frames)
    half_w = window // 2
    start = max(0, t - half_w)
    end = min(n, t + half_w + 1)

    features = []

    if t > 0 and t < n:
        prev_preds = set(frames[t - 1].predicates.keys())
        curr_preds = set(frames[t].predicates.keys())
        features.append(float(len(curr_preds - prev_preds)))
        features.append(float(len(prev_preds - curr_preds)))
    else:
        features.extend([0.0, 0.0])

    action_types = [frames[j].action_type for j in range(start, end)]
    n_transitions = sum(
        1 for j in range(1, len(action_types))
        if action_types[j] != action_types[j - 1]
    )
    features.append(float(n_transitions))

    n_retrieval = sum(1 for a in action_types if a != "primitive")
    features.append(float(n_retrieval))

    pred_values = []
    for j in range(start, end):
        pred_values.extend(frames[j].predicates.values())
    if pred_values:
        features.append(float(np.mean(pred_values)))
        features.append(float(np.std(pred_values)))
    else:
        features.extend([0.0, 0.0])

    features.append(float(t) / max(n, 1))

    return np.array(features, dtype=np.float32)


class BoundaryClassifier:
    """Simple MLP boundary classifier (NumPy-only implementation).

    For production use, replace with a PyTorch or sklearn model.
    """

    def __init__(self, config: Optional[BoundaryClassifierConfig] = None):
        self.cfg = config or BoundaryClassifierConfig()
        self._weights: Optional[Dict[str, np.ndarray]] = None
        self._trained = False

    def train(
        self,
        trajectories: List[TrajectoryForEM],
        decode_results: List[DecodeResult],
    ) -> BoundaryTrainResult:
        """Train the boundary classifier on decoded segmentation labels."""
        labels = extract_boundary_labels(trajectories, decode_results)

        if len(labels) < self.cfg.min_training_samples:
            logger.warning("Too few samples (%d) for boundary training", len(labels))
            return BoundaryTrainResult(n_samples=len(labels))

        traj_map = {t.traj_id: t for t in trajectories}
        X_list, y_list = [], []

        for traj_id, t, is_boundary in labels:
            traj = traj_map.get(traj_id)
            if traj is None:
                continue
            feat = extract_boundary_features(traj, t, window=self.cfg.feature_window)
            X_list.append(feat)
            y_list.append(1.0 if is_boundary else 0.0)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

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
                xb = X[batch_idx]
                yb = y[batch_idx]

                z1 = xb @ W1 + b1
                a1 = np.maximum(z1, 0)
                z2 = (a1 @ W2 + b2).flatten()
                pred = 1.0 / (1.0 + np.exp(-np.clip(z2, -20, 20)))

                weight = np.where(yb == 1, self.cfg.pos_weight, 1.0)
                loss_vec = -(yb * np.log(pred + 1e-7) + (1 - yb) * np.log(1 - pred + 1e-7))
                loss = (loss_vec * weight).mean()

                grad_z2 = (pred - yb) * weight / len(yb)
                grad_W2 = a1.T @ grad_z2.reshape(-1, 1)
                grad_b2 = grad_z2.sum(keepdims=True)
                grad_a1 = grad_z2.reshape(-1, 1) @ W2.T
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

        return BoundaryTrainResult(
            n_samples=len(y),
            n_positive=n_pos,
            n_negative=n_neg,
            train_loss=float(loss),
            train_accuracy=float(accuracy),
        )

    def predict(self, trajectory: TrajectoryForEM, t: int) -> float:
        """Predict boundary probability at timestep t."""
        if not self._trained or self._weights is None:
            return 0.5

        feat = extract_boundary_features(trajectory, t, window=self.cfg.feature_window)
        x = feat.reshape(1, -1)
        W1, b1 = self._weights["W1"], self._weights["b1"]
        W2, b2 = self._weights["W2"], self._weights["b2"]

        z1 = x @ W1 + b1
        a1 = np.maximum(z1, 0)
        z2 = (a1 @ W2 + b2).flatten()
        return float(1.0 / (1.0 + np.exp(-np.clip(z2[0], -20, 20))))

    @property
    def is_trained(self) -> bool:
        return self._trained
