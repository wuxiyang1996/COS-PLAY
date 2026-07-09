"""
Change-point detection from embedding sequences.

Given a sequence of embedding vectors x_0, ..., x_{T-1}, produce a
change-point score cp_t for each timestep.  Peaks in cp_t indicate
times where the latent context shifts (e.g. exploration->combat).

Methods:
  - CUSUM-style cosine distance detector (lightweight, no training).
  - Sliding-window KL/cosine divergence between past and future windows.

These scores are consumed by the boundary proposal pipeline as the
``changepoint_scores`` signal.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def _cosine_distance_seq(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive embeddings: 1 - cos(x_t, x_{t-1})."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normed = embeddings / norms
    # dot product between consecutive vectors
    dots = np.sum(normed[:-1] * normed[1:], axis=1)
    dists = 1.0 - dots
    return np.concatenate([[0.0], dists])  # length T, cp_0 = 0


def cusum_changepoint_scores(
    embeddings: np.ndarray,
    drift: float = 0.05,
) -> np.ndarray:
    """
    CUSUM-style change-point scores from embeddings.

    Computes cosine distance between consecutive embeddings, then runs
    a one-sided CUSUM accumulator.  High values indicate sustained
    deviation from the recent context — i.e. a change-point.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (T, dim).
    drift : float
        CUSUM drift parameter (allowance for normal variation).
        Smaller → more sensitive.

    Returns
    -------
    np.ndarray
        Shape (T,) change-point scores.  Peaks are candidate boundaries.
    """
    dists = _cosine_distance_seq(embeddings)
    T = len(dists)
    scores = np.zeros(T, dtype=np.float64)
    cusum = 0.0
    for t in range(T):
        cusum = max(0.0, cusum + dists[t] - drift)
        scores[t] = cusum
    return scores


def sliding_window_divergence(
    embeddings: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """
    Sliding-window cosine divergence between past and future context.

    For each timestep t, compute the mean cosine distance between the
    mean embedding of [t-w, t) and [t, t+w).  This is high when the
    context before and after t are dissimilar.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (T, dim).
    window_size : int
        Number of timesteps in each half-window.

    Returns
    -------
    np.ndarray
        Shape (T,) divergence scores.
    """
    T, dim = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normed = embeddings / norms

    scores = np.zeros(T, dtype=np.float64)
    w = window_size
    for t in range(w, T - w):
        past_mean = normed[t - w : t].mean(axis=0)
        future_mean = normed[t : t + w].mean(axis=0)
        past_norm = np.linalg.norm(past_mean)
        future_norm = np.linalg.norm(future_mean)
        if past_norm < 1e-9 or future_norm < 1e-9:
            continue
        cos_sim = np.dot(past_mean, future_mean) / (past_norm * future_norm)
        scores[t] = 1.0 - cos_sim
    return scores


def compute_changepoint_scores(
    embeddings: np.ndarray,
    method: str = "cusum",
    drift: float = 0.05,
    window_size: int = 10,
) -> np.ndarray:
    """
    Compute change-point scores from an embedding sequence.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (T, dim).
    method : str
        "cusum" or "sliding_window".
    drift : float
        CUSUM drift (only used if method="cusum").
    window_size : int
        Half-window size (only used if method="sliding_window").

    Returns
    -------
    np.ndarray
        Shape (T,) change-point scores for boundary proposal.
    """
    if method == "cusum":
        return cusum_changepoint_scores(embeddings, drift=drift)
    elif method == "sliding_window":
        return sliding_window_divergence(embeddings, window_size=window_size)
    else:
        raise ValueError(f"Unknown method '{method}'; use 'cusum' or 'sliding_window'.")
