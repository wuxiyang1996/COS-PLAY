"""
Stage 3: ContractVerification — skill bank construction and verification.

Given segmented trajectories from Stage 2 (segments with skill labels,
including ``__NEW__``), build and maintain a Skill Bank where each skill
has a verifiable contract:

  - **Pre** (preconditions)
  - **Eff+** / **Eff-** (add/delete effects)
  - **Inv** (invariants)

and a verification report that triggers:

  - **KEEP** — contract holds, no changes needed.
  - **REFINE** — drop/soften unstable literals.
  - **SPLIT** — instances cluster into distinct sub-skills.
  - **MATERIALIZE_NEW** — repeated ``__NEW__`` segments become a new skill.

Pipeline:
  1. Extract per-timestep predicates and build smoothed summaries.
  2. Compute per-instance add/delete effects.
  3. Aggregate into initial contracts per skill.
  4. Verify contracts and produce counterexamples.
  5. Decide KEEP / REFINE / SPLIT actions.
  6. Materialize ``__NEW__`` segments into new skills.
  7. Persist updates with versioning.

High-level API
--------------
- ``run_stage3(result, traj_id, observations, ...)``
    Full pipeline: extract predicates → contracts → verify → update bank.

- ``run_stage3_batch(results_and_obs, ...)``
    Process multiple trajectories into one bank.

Skill Bank
----------
- ``SkillBank`` — persistent skill bank with versioning and ``compat_fn``
  for Stage 2 integration.

Stage 2 Integration
-------------------
After running Stage 3, feed the bank back to Stage 2:

    >>> scorer = SegmentScorer(
    ...     skill_names=bank.get_skill_names(),
    ...     config=config,
    ...     compat_fn=bank.compat_fn,
    ... )
"""

from skill_agents_grpo.contract_verification.config import (
    ContractVerificationConfig,
    PredicateConfig,
    ContractAggregationConfig,
    VerificationConfig,
    ClusteringConfig,
    NewSkillConfig,
)
from skill_agents_grpo.contract_verification.schemas import (
    SegmentRecord,
    SkillContract,
    VerificationReport,
    UpdateAction,
)
from skill_agents_grpo.contract_verification.predicates import (
    build_segment_predicates,
    build_records_from_result,
    default_extract_predicates,
)
from skill_agents_grpo.contract_verification.contract_init import (
    compute_instance_effects,
    compute_instance_invariants,
    compute_all_effects,
    build_initial_contracts,
)
from skill_agents_grpo.contract_verification.contract_verify import (
    verify_contract,
    verify_all_contracts,
)
from skill_agents_grpo.contract_verification.clustering import (
    cluster_records,
    cluster_effect_jaccard_gap,
)
from skill_agents_grpo.contract_verification.updates import (
    decide_action,
    apply_refine,
    apply_split,
    materialize_new_skills,
)
from skill_agents_grpo.contract_verification.skill_bank import SkillBank
from skill_agents_grpo.contract_verification.action_language import (
    format_contract,
    contract_to_pddl,
    contract_to_strips,
    contract_to_sas,
    contract_to_compact,
    bank_to_pddl_domain,
    bank_to_action_language,
    SUPPORTED_FORMATS,
)
from skill_agents_grpo.contract_verification.run_stage3 import (
    run_stage3,
    run_stage3_batch,
    Stage3Summary,
)

__all__ = [
    # Config
    "ContractVerificationConfig",
    "PredicateConfig",
    "ContractAggregationConfig",
    "VerificationConfig",
    "ClusteringConfig",
    "NewSkillConfig",
    # Schemas
    "SegmentRecord",
    "SkillContract",
    "VerificationReport",
    "UpdateAction",
    # Predicates
    "build_segment_predicates",
    "build_records_from_result",
    "default_extract_predicates",
    # Contract init
    "compute_instance_effects",
    "compute_instance_invariants",
    "compute_all_effects",
    "build_initial_contracts",
    # Verification
    "verify_contract",
    "verify_all_contracts",
    # Clustering
    "cluster_records",
    "cluster_effect_jaccard_gap",
    # Updates
    "decide_action",
    "apply_refine",
    "apply_split",
    "materialize_new_skills",
    # Skill bank
    "SkillBank",
    # Action language
    "format_contract",
    "contract_to_pddl",
    "contract_to_strips",
    "contract_to_sas",
    "contract_to_compact",
    "bank_to_pddl_domain",
    "bank_to_action_language",
    "SUPPORTED_FORMATS",
    # Pipeline
    "run_stage3",
    "run_stage3_batch",
    "Stage3Summary",
]
