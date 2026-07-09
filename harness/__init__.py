"""Skill Harness — per-invocation runtime for skill execution and verification.

Spec: PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION §4 (Phase A).

The harness's role:
  1.  Filter the Bank's candidate set to a domain- and contract-eligible
      sub-set (`select_eligible_skills`).
  2.  Execute one chosen skill via the right adapter (`run_skill`).
  3.  Record everything as a SkillEpisode + reward log entry.
  4.  Provide replay validation for the gate (Stage 1, PLAN-UNIFIED-SKILL-GATE §7).

It does NOT:
  - Choose which skill to commit to (that is the Actor's job).
  - Mutate `SkillRecord.status` (only `SkillLifecycleManager` may).
  - Read/write a "memory" buffer.
  - Train policies / call the teacher.

Public surface (what `decision_agents/` and `orchestrator/` import):

    from harness import (
        SkillHarness,
        AdapterRegistry,
        SkillAdapter,
        EligibleSkill,
        ReplayValidator,
        RewardLogger,
    )
"""

from harness.adapter_registry import AdapterRegistry
from harness.eligibility import EligibleSkill, EligibilityFilter, task_id_from_state
from harness.few_shot_adapter import (
    AdaptResult,
    FewShotAdapter,
    FewShotAdapterError,
    FewShotDemo,
    default_success_fn,
)
from harness.gym_schema_producer import (
    SchemaProducer,
    candy_crush_producer,
    make_gaming_env_producer,
    render_state_block,
    super_mario_producer,
    tetris_producer,
    twenty_forty_eight_producer,
)
from harness.gymv_executor import (
    ACTION_ALIAS_MAP,
    GymvExecutorState,
    initial_state_from_env,
    make_gymv_executor,
)
from harness.gymv_success import (
    EFFECT_PREDICATE_TYPES,
    HopEffectResult,
    PredicateResult,
    SuccessFnFactory,
    evaluate_episode_effects,
    evaluate_hop_effects,
    evaluate_predicate,
    make_per_step_success_fn,
    register_success_fn,
    registered_success_fn_domains,
    success_fn_for_domain,
)
from harness.qa_success import make_qa_success_fn, qa_answer_matches  # noqa: F401
from harness.browsergym_executor import (  # noqa: F401
    BrowserExecutorState,
    make_browsergym_executor,
)
from harness.browser_schema_producer import (  # noqa: F401
    browsergym_canonical_producer,
    make_browsergym_producer,
)
from harness.browser_success import make_browser_per_step_success_fn  # noqa: F401
from harness.osworld_executor import make_osworld_executor  # noqa: F401
from harness.osworld_schema_producer import make_osworld_producer  # noqa: F401
from harness.osworld_success import make_osworld_per_step_success_fn  # noqa: F401
from harness.rejected_skill_sink import FlushReport, RejectedSkillSink
from harness.rejection_deboost import (  # noqa: F401
    apply_deboost_to_candidates,
    compute_deboost,
)
from harness.replay_validator import ReplayValidator, ReplayResult
from harness.reward_logger import RewardLogger
from harness.skill_adapter import SkillAdapter, AdapterRunContext, AdapterRunResult
from harness.skill_harness import HarnessConfig, SkillHarness
from harness.video_executor import make_video_executor  # noqa
from harness.video_qa_success import make_video_qa_success_fn  # noqa

# Day-7: GateRunner is the spec-named offline gate surface (PLAN-UNIFIED-SKILL-GATE §6).
# Import after `SkillHarness` to keep the import order stable; the
# `orchestrator.gate_service` dependency is loaded lazily on first
# attribute access via `__getattr__` to avoid an import cycle for
# consumers that don't need the gate (e.g. the cold-start labelers).
def __getattr__(name: str):  # noqa: D401
    if name in {"EvalSuite", "GateRunner", "GateRunnerConfig"}:
        from harness.gate_runner import EvalSuite, GateRunner, GateRunnerConfig
        globals().update(
            EvalSuite=EvalSuite,
            GateRunner=GateRunner,
            GateRunnerConfig=GateRunnerConfig,
        )
        return globals()[name]
    raise AttributeError(f"module 'harness' has no attribute {name!r}")

__all__ = [
    "ACTION_ALIAS_MAP",
    "AdaptResult",
    "AdapterRegistry",
    "AdapterRunContext",
    "AdapterRunResult",
    "EFFECT_PREDICATE_TYPES",
    "EligibilityFilter",
    "EligibleSkill",
    "FewShotAdapter",
    "FewShotAdapterError",
    "FewShotDemo",
    "GymvExecutorState",
    "HarnessConfig",
    "HopEffectResult",
    "PredicateResult",
    "ReplayResult",
    "ReplayValidator",
    "RewardLogger",
    "EvalSuite",
    "FlushReport",
    "GateRunner",
    "GateRunnerConfig",
    "RejectedSkillSink",
    "apply_deboost_to_candidates",
    "compute_deboost",
    "SchemaProducer",
    "SkillAdapter",
    "SkillHarness",
    "SuccessFnFactory",
    "BrowserExecutorState",
    "browsergym_canonical_producer",
    "candy_crush_producer",
    "default_success_fn",
    "make_browser_per_step_success_fn",
    "make_browsergym_executor",
    "make_browsergym_producer",
    "evaluate_episode_effects",
    "evaluate_hop_effects",
    "evaluate_predicate",
    "initial_state_from_env",
    "make_gaming_env_producer",
    "make_gymv_executor",
    "make_osworld_executor",
    "make_osworld_per_step_success_fn",
    "make_osworld_producer",
    "make_per_step_success_fn",
    "make_video_executor",
    "make_video_qa_success_fn",
    "make_qa_success_fn",
    "qa_answer_matches",
    "register_success_fn",
    "registered_success_fn_domains",
    "render_state_block",
    "success_fn_for_domain",
    "super_mario_producer",
    "task_id_from_state",
    "tetris_producer",
    "twenty_forty_eight_producer",
]
