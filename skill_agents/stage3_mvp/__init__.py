"""
Stage 3 MVP: effects-only contract learning, verification, and refinement.

Given segmented trajectories from Stage 2 (segments with skill labels),
learn a stable effect signature per skill:

  - **eff_add**: predicates that reliably become true by segment end.
  - **eff_del**: predicates that reliably become false by segment end.
  - **eff_event**: event-like predicates that occur during the segment.

Then verify these effects across all instances and refine by removing
unreliable literals.  No preconditions, no invariants, no split/new.

Pipeline:
  1. Extract per-timestep predicates and build smoothed summaries.
  2. Compute per-instance add/delete/event effects.
  3. Aggregate into initial effects contracts per skill.
  4. Verify contracts against instances.
  5. Refine by dropping unreliable literals.
  6. Persist into a Skill Bank with versioning.

High-level API
--------------
- ``run_stage3_mvp(segments, observations_by_traj, ...)``
    Full pipeline: summarize → effects → learn → verify → refine → persist.

- ``specs_from_segmentation_result(result, traj_id, ...)``
    Convert Stage 2 output into ``SegmentSpec`` list.
"""

from skill_agents.stage3_mvp.config import Stage3MVPConfig
from skill_agents.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents.stage3_mvp.predicate_vocab import (
    PredicateVocab,
    predicate_namespace,
    normalize_event,
)
from skill_agents.stage3_mvp.extract_predicates import (
    CompositePredicateExtractor,
    default_extract_predicates,
    extract_ui_events_from_log,
)
from skill_agents.stage3_mvp.segment_summarize import summarize_segment
from skill_agents.stage3_mvp.effects_compute import compute_effects
from skill_agents.stage3_mvp.contract_learn import learn_effects_contract
from skill_agents.stage3_mvp.contract_verify import verify_effects_contract
from skill_agents.stage3_mvp.contract_refine import refine_effects_contract
from skill_agents.stage3_mvp.run_stage3_mvp import (
    run_stage3_mvp,
    specs_from_segmentation_result,
    SegmentSpec,
    SkillResult,
    Stage3MVPSummary,
)

__all__ = [
    # Config
    "Stage3MVPConfig",
    # Schemas
    "SegmentRecord",
    "SkillEffectsContract",
    "VerificationReport",
    # Predicate vocab
    "PredicateVocab",
    "predicate_namespace",
    "normalize_event",
    # Extraction
    "CompositePredicateExtractor",
    "default_extract_predicates",
    "extract_ui_events_from_log",
    # Pipeline steps
    "summarize_segment",
    "compute_effects",
    "learn_effects_contract",
    "verify_effects_contract",
    "refine_effects_contract",
    # Orchestrator
    "run_stage3_mvp",
    "specs_from_segmentation_result",
    "SegmentSpec",
    "SkillResult",
    "Stage3MVPSummary",
]
