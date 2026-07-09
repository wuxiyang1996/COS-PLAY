"""
Stage 1: High-recall boundary proposal for trajectory segmentation.

Produces candidate cut points C so that later DP/HSMM/beam search only
considers boundaries at those times (or in small windows), reducing
search from O(T^2) to O(|C|^2).

Framework integration
---------------------
- ``segment_episode(episode, env_name=...)``
    Full pipeline: extract signals from an Episode, propose boundaries,
    return SubTask_Experience segments.

- ``propose_from_episode(episode, env_name=...)``
    Propose boundary candidates from an Episode without segmenting.

- ``annotate_episode_boundaries(episode, env_name=...)``
    Mark boundaries on Experience objects so the existing
    Episode.separate_into_sub_episodes() respects them.

Signal extraction strategies
----------------------------
- ``env_name="overcooked"`` etc. — rule-based (fast, brittle)
- ``env_name="llm"`` — LLM-based predicates (general, adaptive)
- ``env_name="llm+overcooked"`` — hybrid: LLM predicates + per-env hard events (recommended)

Low-level API
-------------
- ``propose_boundary_candidates(T, predicates=..., surprisal=..., ...)``
    Propose from raw signal arrays.

- ``compute_changepoint_scores(embeddings, method=...)``
    CUSUM / sliding-window change-point detection from embeddings.

- ``get_signal_extractor(env_name)``
    Signal extractor factory (rule-based, LLM, or hybrid).
"""

from skill_agents_grpo.boundary_proposal.proposal import (
    propose_boundary_candidates,
    BoundaryCandidate,
    ProposalConfig,
    candidate_centers_only,
    candidate_windows,
)
from skill_agents_grpo.boundary_proposal.changepoint import compute_changepoint_scores
from skill_agents_grpo.boundary_proposal.signal_extractors import (
    get_signal_extractor,
    SignalExtractorBase,
    HybridSignalExtractor,
    IntentionSignalExtractor,
    parse_intention_tag,
)
from skill_agents_grpo.boundary_proposal.episode_adapter import (
    segment_episode,
    propose_from_episode,
    annotate_episode_boundaries,
    extract_signals,
)
from skill_agents_grpo.boundary_proposal.boundary_preference import (
    BoundaryPreferenceScorer,
    BoundaryPreferenceConfig,
)

__all__ = [
    # High-level (framework-integrated)
    "segment_episode",
    "propose_from_episode",
    "annotate_episode_boundaries",
    "extract_signals",
    # Low-level
    "propose_boundary_candidates",
    "BoundaryCandidate",
    "ProposalConfig",
    "candidate_centers_only",
    "candidate_windows",
    # Change-point
    "compute_changepoint_scores",
    # Signal extractors
    "get_signal_extractor",
    "SignalExtractorBase",
    "HybridSignalExtractor",
    "IntentionSignalExtractor",
    "parse_intention_tag",
    # Boundary preference
    "BoundaryPreferenceScorer",
    "BoundaryPreferenceConfig",
]
