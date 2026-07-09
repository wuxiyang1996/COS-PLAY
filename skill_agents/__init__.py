"""
Skill agents: modules for trajectory segmentation and sub-task decomposition.

See PLAN.md in this directory for the SkillBank Agent operating plan (stages, data model,
constraints, and how modules plug together).

Top-level API:
  - SkillBankAgent: agentic pipeline that ingests episodes, builds/maintains a
    Skill Bank, and serves queries for decision_agents.
  - SkillQueryEngine: rich retrieval over the Skill Bank (keyword, effect-based).
  - PipelineConfig: configuration for the full pipeline.
  - TransferableSkillExtractor: extraction pipeline for cross-domain skill transfer.

Subpackages:
  - boundary_proposal: Stage 1 high-recall boundary proposal for trajectory segmentation.
  - infer_segmentation: Stage 2 optimal skill-sequence decoding with preference learning.
  - stage3_mvp: Stage 3 effects-only contract learning, verification, and refinement.
  - skill_bank: Persistent storage for learned skill contracts.
  - skill_evaluation: Holistic quality assessment of extracted skills (coherence,
    discriminability, composability, generalization, utility, granularity).
  - bank_maintenance: Split, merge, refine, and local re-decode.
  - skill_template: Cross-domain transferable skill templates and reasoning protocols.
  - extract_transferable: Pipeline for discovering reusable skills across tasks.
"""

from skill_agents.pipeline import SkillBankAgent, PipelineConfig, IterationSnapshot
from skill_agents.query import SkillQueryEngine, SkillSelectionResult
from skill_agents.skill_bank.bank import SkillBankMVP
from skill_agents.skill_bank.new_pool import NewPoolManager, NewPoolConfig
from skill_agents.tool_call_reward import (
    ToolCallRewardConfig,
    ToolCallRewardResult,
    compute_tool_call_reward,
    compute_episode_tool_call_returns,
)
from skill_agents.skill_template import (
    TransferableSkill,
    SlotBinding,
    ReasoningProtocol,
    HopStep,
    AbstractPredicate,
    FAMILY_PROTOCOLS,
)
from skill_agents.extract_transferable import (
    TransferableSkillExtractor,
    extract_transferable_skills,
)

__all__ = [
    "SkillBankAgent",
    "PipelineConfig",
    "IterationSnapshot",
    "SkillQueryEngine",
    "SkillSelectionResult",
    "SkillBankMVP",
    "NewPoolManager",
    "NewPoolConfig",
    "ToolCallRewardConfig",
    "ToolCallRewardResult",
    "compute_tool_call_reward",
    "compute_episode_tool_call_returns",
    "TransferableSkill",
    "TransferableSkillExtractor",
    "extract_transferable_skills",
    "SlotBinding",
    "ReasoningProtocol",
    "HopStep",
    "AbstractPredicate",
    "FAMILY_PROTOCOLS",
]
