"""
Stage 2: InferSegmentation — preference-learned skill-sequence decoding.

The LLM acts as a **preference teacher**: it ranks skills for each segment
(not numeric scores).  A ``PreferenceScorer`` is trained from those rankings
via Bradley-Terry, then used by DP/beam decoders for optimal segmentation.

Pipeline:
  1. Stage 1 proposes candidate boundaries C.
  2. LLM teacher ranks skills for candidate segments → pairwise preferences.
  3. Train PreferenceScorer from preferences.
  4. Decode with trained scorer (DP or beam).
  5. On uncertain segments, query LLM for more preferences → retrain → re-decode.

High-level API
--------------
- ``infer_and_segment(episode, skill_names, ...)``
    Inference with LLM (e.g. GPT-5): Stage 1 → LLM preferences → train → decode.
    Use when you want to interface with an LLM at inference time.

- ``infer_and_segment_offline(episode, skill_names, ...)``
    Offline (no LLM): decode using provided behavior_fit_fn and transition_fn only.
    Use for training from pre-collected preferences, or testing without API.

- ``infer_segmentation(candidates, T, skill_names, ...)``
    Low-level: run decoder with a given scorer.

LLM Teacher
-----------
- ``collect_segment_preferences(segments, ...)``
    Cold-start: LLM ranks skills for every segment.

- ``collect_uncertain_preferences(result, ...)``
    Active learning: LLM resolves uncertain segments only.

- ``collect_transition_preferences(skill_names, ...)``
    LLM ranks transition likelihoods between skills.
"""

from skill_agents_grpo.infer_segmentation.config import (
    SegmentationConfig,
    ScorerWeights,
    DurationPriorConfig,
    NewSkillConfig,
    LLMTeacherConfig,
    PreferenceLearningConfig,
    DecoderConfig,
)
from skill_agents_grpo.infer_segmentation.scorer import (
    SegmentScorer,
    NEW_SKILL,
)
from skill_agents_grpo.infer_segmentation.diagnostics import (
    SegmentationResult,
    SegmentDiagnostic,
    SkillCandidate,
    BoundaryDiagnostic,
)
from skill_agents_grpo.infer_segmentation.dp_decoder import viterbi_decode
from skill_agents_grpo.infer_segmentation.beam_decoder import beam_decode
from skill_agents_grpo.infer_segmentation.preference import (
    PreferenceExample,
    PreferenceQuery,
    PreferenceStore,
    PreferenceScorer,
    generate_preference_queries,
)
from skill_agents_grpo.infer_segmentation.llm_teacher import (
    collect_segment_preferences,
    collect_transition_preferences,
    collect_uncertain_preferences,
    ranking_to_pairwise,
    flush_teacher_io_records,
    get_teacher_io_records,
    reset_teacher_io_records,
)
from skill_agents_grpo.infer_segmentation.episode_adapter import (
    infer_segmentation,
    infer_and_segment,
    infer_and_segment_offline,
    grpo_scorer_factory,
    grpo_decode_fn,
)
from skill_agents_grpo.infer_segmentation.phase_detector import (
    detect_phases,
    make_compound_label,
)

__all__ = [
    # Config
    "SegmentationConfig",
    "ScorerWeights",
    "DurationPriorConfig",
    "NewSkillConfig",
    "LLMTeacherConfig",
    "PreferenceLearningConfig",
    "DecoderConfig",
    # Scorer
    "SegmentScorer",
    "NEW_SKILL",
    # Diagnostics
    "SegmentationResult",
    "SegmentDiagnostic",
    "SkillCandidate",
    "BoundaryDiagnostic",
    # Decoders
    "viterbi_decode",
    "beam_decode",
    # Preference learning
    "PreferenceExample",
    "PreferenceQuery",
    "PreferenceStore",
    "PreferenceScorer",
    "generate_preference_queries",
    # LLM teacher
    "collect_segment_preferences",
    "collect_transition_preferences",
    "collect_uncertain_preferences",
    "ranking_to_pairwise",
    # High-level pipeline
    "infer_segmentation",
    "infer_and_segment",
    "infer_and_segment_offline",
    # GRPO helpers
    "grpo_scorer_factory",
    "grpo_decode_fn",
    # Teacher I/O recording (cold-start data)
    "flush_teacher_io_records",
    "get_teacher_io_records",
    "reset_teacher_io_records",
    # Phase detection
    "detect_phases",
    "make_compound_label",
]
