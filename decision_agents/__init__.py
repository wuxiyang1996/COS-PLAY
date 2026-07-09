# Decision agents: dummy language agent and LLM decision-making agent.

from typing import Dict

from .dummy_agent import (
    language_agent_action,
    detect_game,
    run_episode_with_experience_collection,
    AgentBufferManager,
    GAME_AVALON,
    GAME_DIPLOMACY,
    GAME_GAMINGAGENT,
)

from .agent_helper import (
    get_state_summary,
    compact_structured_state,
    compact_text_observation,
    DEFAULT_SUMMARY_CHAR_BUDGET,
    HARD_SUMMARY_CHAR_LIMIT,
    infer_intention,
    EpisodicMemoryStore,
    skill_bank_to_text,
    select_skill_from_bank,
    query_skill_bank,
)

from .agent import (
    VLMDecisionAgent,
    LLMDecisionAgent,
    AgentState,
    run_tool,
    run_episode_vlm_agent,
    run_episode_llm_agent,
    TOOL_TAKE_ACTION,
    TOOL_GET_STATE_SUMMARY,
    TOOL_GET_INTENTION,
    TOOL_SELECT_SKILL,
    TOOL_REWARD,
)

from .reward_func import (
    RewardConfig,
    RewardResult,
    RewardComputer,
    compute_reward,
)

# Schema-native Actor Agent stack (PLAN-ACTION-AGENT.md).  These are the
# classes the future GRPO training pipeline will target.  The older
# ``VLMDecisionAgent`` path above is kept for backward compatibility with
# scripts/qwen3_decision_agent.py; new code should prefer ``ActorAgent``.
from .schema_parser import (
    Answer,
    Entity,
    Hop,
    Relation,
    ResolvedAction,
    StateFlags,
    StateSchema,
    Targets,
    parse_state_schema,
    resolve_entity_action,
)
from .skill_interface import (
    NullSkillProvider,
    SkillBankProvider,
    SkillGuidance,
    SkillProvider,
)
from .skill_tracker import (
    ActivationCheck,
    SkillTracker,
    TrackerState,
)
from .actor_agent import (
    ActorAgent,
    ActorDecision,
    ActorState,
    run_actor_episode,
)

# ── New flavours of the Actor Agent (PLAN-ACTION-AGENT.md §2.3) ──
#
# Two specialised subclasses live in their own sub-packages so the
# GPT-4o data-collection path stays cleanly separated from the
# Qwen3-VL inference + GRPO path.  They are imported lazily through
# attribute access below so that ``import decision_agents`` does not
# pull in ``openai`` / ``trainer.coevolution.vllm_client`` /
# ``trainer.common.metrics`` unless the caller actually asks for them.
from .core import (
    BrowserHarness,
    Detection,
    EvidenceCache,
    GymHarness,
    Harness,
    HarnessState,
    MockOCR,
    MockRegionDetector,
    MockSegmenter,
    OCREngine,
    OCRResult,
    OSWorldHarness,
    RegionDetector,
    Segmentation,
    Segmenter,
    VIDEO_OPS,
    VR_OPS,
    VRHarness,
    VideoHarness,
    VisualInput,
    build_openai_vision_messages,
    build_qwen_vl_messages,
    load_image_as_data_url,
    parse_op_call,
)

# PEP 562 lazy attribute access so importing ``decision_agents`` stays
# light: ``openai`` (SFT actor) and ``trainer.coevolution.vllm_client``
# (GRPO actor) are pulled in only when the matching symbol is touched.
_LAZY_ATTRS = {
    "GPT4oCollectorActor": ("decision_agents.SFT", "GPT4oCollectorActor"),
    "SFTRecorder":         ("decision_agents.SFT", "SFTRecorder"),
    "SFTRecord":           ("decision_agents.SFT", "SFTRecord"),
    "QwenVLActor":         ("decision_agents.grpo", "QwenVLActor"),
    "GRPORolloutLogger":   ("decision_agents.grpo", "GRPORolloutLogger"),
    "DEFAULT_QWEN_VL_MODEL": ("decision_agents.grpo", "DEFAULT_QWEN_VL_MODEL"),
}

# ── Deprecation shim for the removed inner-MDP scaffold ──────────────
#
# Phase 3 of the unified-harness migration deletes
# ``decision_agents/inner_mdp.py`` (HopAction / HopPolicy /
# HeuristicHopPolicy / HopStep / HopTrace / parse_hop_action).  Their
# semantics now live inside :class:`VRHarness` / :class:`VideoHarness`
# action vocabularies (see ``decision_agents/README.md`` "Migration of
# inner-MDP operators" table).  We keep the names reachable for one
# release so out-of-tree imports get a clear deprecation warning
# pointing at the new harness symbols instead of an ``ImportError``.

_DEPRECATED_INNER_MDP: Dict[str, str] = {
    "HopAction":           "Use harness action strings (e.g. VRHarness LOOK / RETRIEVE / NOTE / ANSWER) instead.",
    "HopPolicy":           "The inner-MDP loop has been removed; per-task action vocabularies live on Harness implementations.",
    "HopStep":             "Removed with the inner-MDP loop; harness ``step`` returns gym-shaped tuples instead.",
    "HopTrace":            "Removed with the inner-MDP loop; per-step traces are recorded by GRPORolloutLogger / SFTRecorder.",
    "HeuristicHopPolicy":  "Removed; VRHarness / VideoHarness encode the equivalent operators as first-class actions.",
    "parse_hop_action":    "Use ``decision_agents.core.harness.parse_op_call`` for ``OP(arg)`` action strings.",
}


def __getattr__(name: str):  # pragma: no cover — thin shim
    """Lazy import for SFT / GRPO actors + deprecation shim for inner-MDP names."""
    if name in _LAZY_ATTRS:
        import importlib
        mod_name, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    if name in _DEPRECATED_INNER_MDP:
        import warnings
        warnings.warn(
            f"decision_agents.{name} has been removed in the unified-harness "
            f"migration. {_DEPRECATED_INNER_MDP[name]} See decision_agents/"
            f"README.md \u201cMigration of inner-MDP operators\u201d.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise AttributeError(
            f"decision_agents.{name} has been removed; see decision_agents/README.md "
            f"\u201cMigration of inner-MDP operators\u201d for the harness equivalent."
        )
    raise AttributeError(f"module 'decision_agents' has no attribute {name!r}")


__all__ = [
    "language_agent_action",
    "detect_game",
    "run_episode_with_experience_collection",
    "AgentBufferManager",
    "GAME_AVALON",
    "GAME_DIPLOMACY",
    "GAME_GAMINGAGENT",
    "get_state_summary",
    "compact_structured_state",
    "compact_text_observation",
    "DEFAULT_SUMMARY_CHAR_BUDGET",
    "HARD_SUMMARY_CHAR_LIMIT",
    "infer_intention",
    "EpisodicMemoryStore",
    "skill_bank_to_text",
    "select_skill_from_bank",
    "query_skill_bank",
    "VLMDecisionAgent",
    "LLMDecisionAgent",
    "AgentState",
    "run_tool",
    "run_episode_vlm_agent",
    "run_episode_llm_agent",
    "TOOL_TAKE_ACTION",
    "TOOL_GET_STATE_SUMMARY",
    "TOOL_GET_INTENTION",
    "TOOL_SELECT_SKILL",
    "TOOL_REWARD",
    "RewardConfig",
    "RewardResult",
    "RewardComputer",
    "compute_reward",
    # Schema parser
    "Answer",
    "Entity",
    "Hop",
    "Relation",
    "ResolvedAction",
    "StateFlags",
    "StateSchema",
    "Targets",
    "parse_state_schema",
    "resolve_entity_action",
    # Skill interface
    "NullSkillProvider",
    "SkillBankProvider",
    "SkillGuidance",
    "SkillProvider",
    # Skill tracker
    "ActivationCheck",
    "SkillTracker",
    "TrackerState",
    # Actor agent
    "ActorAgent",
    "ActorDecision",
    "ActorState",
    "run_actor_episode",
    # Harness family (decision_agents/core/) — unified single-MDP contract
    "Harness",
    "HarnessState",
    "GymHarness",
    "BrowserHarness",
    "OSWorldHarness",
    "VRHarness",
    "VideoHarness",
    "VR_OPS",
    "VIDEO_OPS",
    "parse_op_call",
    # Multimodal scaffolding (decision_agents/core/)
    "VisualInput",
    "build_openai_vision_messages",
    "build_qwen_vl_messages",
    "load_image_as_data_url",
    # Perception (decision_agents/core/perception/) — Phase 8.0
    "RegionDetector",
    "Segmenter",
    "OCREngine",
    "MockRegionDetector",
    "MockSegmenter",
    "MockOCR",
    "Detection",
    "Segmentation",
    "OCRResult",
    "EvidenceCache",
    # SFT collection flavour (decision_agents/SFT/) — lazy-loaded
    "GPT4oCollectorActor",
    "SFTRecorder",
    "SFTRecord",
    # GRPO + LoRA flavour (decision_agents/grpo/) — lazy-loaded
    "QwenVLActor",
    "GRPORolloutLogger",
    "DEFAULT_QWEN_VL_MODEL",
]
