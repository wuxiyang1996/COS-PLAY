"""
Skill agents: modules for trajectory segmentation and sub-task decomposition.

See PLAN.md in this directory for the SkillBank Agent operating plan (stages, data model,
constraints, and how modules plug together).

Top-level API:
  - SkillBankAgent: agentic pipeline that ingests episodes, builds/maintains a
    Skill Bank, and serves queries for decision_agents.
  - SkillQueryEngine: rich retrieval over the Skill Bank (keyword, effect-based).
  - PipelineConfig: configuration for the full pipeline.

Subpackages:
  - boundary_proposal: Stage 1 high-recall boundary proposal for trajectory segmentation.
  - infer_segmentation: Stage 2 optimal skill-sequence decoding with preference learning.
  - contract_verification: Stage 3 skill bank construction and contract verification.
  - stage3_mvp: Stage 3 MVP effects-only contract learning, verification, and refinement.
  - skill_bank: Persistent storage for learned skill contracts.
  - skill_evaluation: Holistic quality assessment of extracted skills (coherence,
    discriminability, composability, generalization, utility, granularity).
  - bank_maintenance: Split, merge, refine, and local re-decode.
"""

from skill_agents_grpo.pipeline import SkillBankAgent, PipelineConfig, IterationSnapshot
from skill_agents_grpo.query import SkillQueryEngine, SkillSelectionResult
from skill_agents_grpo.skill_bank.bank import SkillBankMVP
from skill_agents_grpo.skill_bank.new_pool import NewPoolManager, NewPoolConfig
from skill_agents_grpo.tool_call_reward import (
    ToolCallRewardConfig,
    ToolCallRewardResult,
    compute_tool_call_reward,
    compute_episode_tool_call_returns,
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
]
