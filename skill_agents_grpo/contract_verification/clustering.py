"""
Step 6 — Clustering for SPLIT and NEW materialization.

Two clustering strategies:

(A) **Effect-signature clustering** — represent each instance by its binary
    add/del effect vector, cluster by Jaccard or Hamming distance.

(B) **Hybrid** — concatenate effect vector (sparse) with PCA-reduced embedding
    (dense) and cluster the combined representation.

Clustering methods: Agglomerative (default, robust) or HDBSCAN (variable k).

Used by:
  - SPLIT: partition instances of a failing skill into coherent sub-skills.
  - MATERIALIZE_NEW: group ``__NEW__`` instances into candidate new skills.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from skill_agents_grpo.contract_verification.config import ClusteringConfig
from skill_agents_grpo.contract_verification.schemas import SegmentRecord


# ── Vocabulary + feature matrix construction ─────────────────────────

def _build_effect_vocab(records: List[SegmentRecord]) -> List[str]:
    """Collect a sorted vocabulary of all predicates seen in add/del effects."""
    vocab: Set[str] = set()
    for rec in records:
        vocab.update(rec.effects_add)
        vocab.update(rec.effects_del)
    return sorted(vocab)


def _build_effect_matrix(
    records: List[SegmentRecord],
    vocab: List[str],
) -> np.ndarray:
    """Build an (N x V) binary matrix from effect signatures."""
    N = len(records)
    V = len(vocab)
    mat = np.zeros((N, V), dtype=np.float32)
    pred_to_idx = {p: i for i, p in enumerate(vocab)}
    for row, rec in enumerate(records):
        for p in rec.effects_add:
            if p in pred_to_idx:
                mat[row, pred_to_idx[p]] = 1.0
        for p in rec.effects_del:
            if p in pred_to_idx:
                mat[row, pred_to_idx[p]] = 1.0
    return mat


def _build_hybrid_matrix(
    records: List[SegmentRecord],
    vocab: List[str],
    pca_dims: int,
) -> np.ndarray:
    """Concatenate effect vectors with PCA-reduced embeddings."""
    effect_mat = _build_effect_matrix(records, vocab)
    embeddings = []
    for rec in records:
        if rec.embedding is not None:
            embeddings.append(rec.embedding)
        else:
            embeddings.append(np.zeros(pca_dims, dtype=np.float32))

    if embeddings and embeddings[0].shape[0] > 0:
        emb_mat = np.stack(embeddings, axis=0).astype(np.float32)
        if emb_mat.shape[1] > pca_dims:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(pca_dims, emb_mat.shape[0], emb_mat.shape[1]))
            emb_mat = pca.fit_transform(emb_mat)
        return np.hstack([effect_mat, emb_mat])
    return effect_mat


# ── Jaccard distance ────────────────────────────────────────────────

def _jaccard_distance_matrix(mat: np.ndarray) -> np.ndarray:
    """Compute pairwise Jaccard distance for binary matrix (N x V)."""
    N = mat.shape[0]
    dist = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            union = np.sum((mat[i] + mat[j]) > 0)
            inter = np.sum((mat[i] * mat[j]) > 0)
            d = 1.0 - (inter / union) if union > 0 else 0.0
            dist[i, j] = d
            dist[j, i] = d
    return dist


# ── Clustering entry point ──────────────────────────────────────────

def cluster_records(
    records: List[SegmentRecord],
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """Cluster segment records by effect signatures (+ optional embeddings).

    Parameters
    ----------
    records : list[SegmentRecord]
        Segments to cluster (must have ``effects_add``/``effects_del`` populated).
    config : ClusteringConfig
        Clustering method and parameters.

    Returns
    -------
    labels : np.ndarray
        Cluster label per record (shape ``(N,)``).
    n_clusters : int
        Number of clusters found.
    quality : dict
        ``"silhouette"`` (if sklearn available), ``"sizes"`` (per-cluster counts).
    """
    N = len(records)
    if N <= 1:
        return np.zeros(N, dtype=int), 1, {"sizes": {0: N}}

    vocab = _build_effect_vocab(records)
    if not vocab:
        return np.zeros(N, dtype=int), 1, {"sizes": {0: N}}

    if config.use_embeddings:
        feat_mat = _build_hybrid_matrix(records, vocab, config.embedding_pca_dims)
    else:
        feat_mat = _build_effect_matrix(records, vocab)

    if config.method == "hdbscan":
        labels, n_clusters = _cluster_hdbscan(feat_mat, config)
    else:
        labels, n_clusters = _cluster_agglomerative(feat_mat, vocab, config)

    quality = _compute_quality(feat_mat, labels, n_clusters)
    return labels, n_clusters, quality


def _cluster_agglomerative(
    feat_mat: np.ndarray,
    vocab: List[str],
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int]:
    """Agglomerative clustering with precomputed Jaccard distance."""
    from sklearn.cluster import AgglomerativeClustering

    N = feat_mat.shape[0]
    if N < config.split_min_clusters:
        return np.zeros(N, dtype=int), 1

    if config.distance_metric == "jaccard":
        dist_mat = _jaccard_distance_matrix(feat_mat)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=config.agglomerative_distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist_mat)
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=config.agglomerative_distance_threshold,
            linkage="ward",
        )
        labels = model.fit_predict(feat_mat)

    n_clusters = len(set(labels))
    return labels, n_clusters


def _cluster_hdbscan(
    feat_mat: np.ndarray,
    config: ClusteringConfig,
) -> Tuple[np.ndarray, int]:
    """HDBSCAN clustering for variable cluster count."""
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3

    model = HDBSCAN(
        min_cluster_size=max(2, config.split_min_clusters),
        metric="euclidean",
    )
    labels = model.fit_predict(feat_mat)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 1:
        n_clusters = 1
        labels = np.zeros(len(labels), dtype=int)
    return labels, max(n_clusters, 1)


def _compute_quality(
    feat_mat: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    quality: Dict[str, float] = {}
    counts = Counter(int(l) for l in labels)
    quality["sizes"] = {str(k): float(v) for k, v in counts.items()}

    if n_clusters >= 2 and len(labels) >= 3:
        try:
            from sklearn.metrics import silhouette_score
            quality["silhouette"] = float(silhouette_score(feat_mat, labels))
        except Exception:
            pass

    return quality


# ── Utility: compute effect-set Jaccard gap between clusters ────────

def cluster_effect_jaccard_gap(
    records: List[SegmentRecord],
    labels: np.ndarray,
) -> float:
    """Compute the minimum Jaccard distance between any two cluster effect centroids.

    Used to decide whether clusters are sufficiently different for a SPLIT.
    """
    clusters: Dict[int, List[SegmentRecord]] = {}
    for rec, label in zip(records, labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(rec)

    if len(clusters) < 2:
        return 0.0

    def _centroid_effects(recs: List[SegmentRecord]) -> Set[str]:
        n = len(recs)
        counter: Counter = Counter()
        for r in recs:
            for p in r.effects_add:
                counter[p] += 1
            for p in r.effects_del:
                counter[p] += 1
        return {p for p, c in counter.items() if c / n >= 0.5}

    centroids = {k: _centroid_effects(v) for k, v in clusters.items()}
    keys = sorted(centroids.keys())

    min_gap = float("inf")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = centroids[keys[i]], centroids[keys[j]]
            union = len(a | b)
            inter = len(a & b)
            jacc_dist = 1.0 - (inter / union) if union > 0 else 0.0
            min_gap = min(min_gap, jacc_dist)

    return min_gap if min_gap != float("inf") else 0.0
