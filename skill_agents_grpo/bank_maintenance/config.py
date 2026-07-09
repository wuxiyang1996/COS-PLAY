"""
Configuration for Bank Maintenance: Split / Merge / Refine + fast re-decode.

All tunables in one flat dataclass with conservative defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class BankMaintenanceConfig:
    """All tunables for the bank-maintenance pipeline."""

    # ── Split thresholds ────────────────────────────────────────────
    split_pass_rate_thresh: float = 0.70
    split_failure_concentration: float = 0.60
    split_sse_ratio: float = 0.25
    split_embedding_var_thresh: float = 2.0
    min_child_size: int = 10
    child_pass_rate_thresh: float = 0.80
    max_split_ways: int = 2

    # ── Merge thresholds ────────────────────────────────────────────
    merge_eff_jaccard_thresh: float = 0.85
    merge_emb_cosine_thresh: float = 0.90
    merge_transition_overlap_k: int = 5
    merge_transition_overlap_min: float = 0.50

    # ── Refine thresholds ───────────────────────────────────────────
    refine_drop_success_rate: float = 0.60
    refine_add_freq_self: float = 0.90
    refine_add_max_confuser_freq: float = 0.30
    refine_top_n_add: int = 5
    refine_top_confusers: int = 3

    # ── Duration model ──────────────────────────────────────────────
    duration_n_bins: int = 20
    duration_min_len: int = 1
    duration_max_len: int = 2000
    duration_smoothing: float = 1.0

    # ── MinHash / LSH for merge candidates ──────────────────────────
    minhash_num_perm: int = 128
    lsh_threshold: float = 0.50

    # ── Local re-decode ─────────────────────────────────────────────
    redecode_window_pad: int = 300
    redecode_min_window: int = 50

    # ── Contract re-learn inside split/merge (reuse Stage3MVPConfig) ─
    eff_freq: float = 0.80
    min_instances_per_skill: int = 5
    max_effects_per_skill: int = 50
    instance_pass_literal_frac: float = 0.70

    # ── Misc ────────────────────────────────────────────────────────
    embedding_dim_pca: int = 16
    profile_topk_transitions: int = 5
