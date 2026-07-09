"""
Configuration for Stage 3 ContractVerification.

Dataclasses for predicate thresholds, contract aggregation, verification
pass criteria, clustering parameters, and NEW-skill materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredicateConfig:
    """Thresholds for predicate booleanization and temporal smoothing."""

    p_thresh: float = 0.7
    start_end_window: int = 3
    downsample_interval: int = 1  # 1 = keep all timesteps


@dataclass
class ContractAggregationConfig:
    """Frequency thresholds for aggregating per-instance data into contracts."""

    pre_freq: float = 0.8
    eff_freq: float = 0.8
    inv_freq: float = 0.9


@dataclass
class VerificationConfig:
    """Pass-rate criteria that decide whether to KEEP, REFINE, or SPLIT."""

    min_pass_rate_keep: float = 0.8
    min_pass_rate_refine: float = 0.7
    violation_drop_thresh: float = 0.3  # drop literal if violation > this
    max_counterexamples: int = 10


@dataclass
class ClusteringConfig:
    """Parameters for effect-signature / embedding clustering."""

    method: str = "agglomerative"  # "agglomerative" | "hdbscan"
    split_min_clusters: int = 2
    split_effect_jaccard_gap: float = 0.3
    embedding_pca_dims: int = 16
    use_embeddings: bool = False  # hybrid mode: effect + embedding
    distance_metric: str = "jaccard"  # "jaccard" | "hamming"
    agglomerative_distance_threshold: float = 0.5


@dataclass
class NewSkillConfig:
    """Thresholds for materializing NEW segments into bank skills."""

    min_cluster_size: int = 5
    min_pass_rate_create: float = 0.8
    min_effect_distinctiveness: float = 0.3  # Jaccard distance from nearest existing skill


@dataclass
class ContractVerificationConfig:
    """Top-level configuration for Stage 3 ContractVerification."""

    predicates: PredicateConfig = field(default_factory=PredicateConfig)
    aggregation: ContractAggregationConfig = field(default_factory=ContractAggregationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    new_skill: NewSkillConfig = field(default_factory=NewSkillConfig)

    bank_path: Optional[str] = None  # path to persist the skill bank

    # Action language output format: "pddl" | "strips" | "sas" | "compact" | None
    # When set, skill generation outputs include action language representation.
    action_language_format: Optional[str] = None
