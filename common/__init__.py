"""Common types shared across `harness/`, `crafter/`, and `orchestrator/`.

Single import point for canonical enums, ID helpers, the typed `<state>`
schema (PLAN-SKILL-BANK §3), and core `EvidenceRef`.

Re-export only — every concrete definition lives in a sub-module so
import graphs stay readable.
"""

from common.enums import (
    DOMAINS,
    EVIDENCE_ROLES,
    GateStage,
    GateVerdict,
    InnerAction,
    RecoveryStrategy,
    SkillSourceType,
    SkillStatus,
    SkillType,
)
from common.ids import (
    new_episode_id,
    new_proposal_id,
    new_run_id,
    new_skill_id,
    new_snapshot_id,
    new_span_id,
    schema_hash,
)
from common.models import (
    BACKBONE_JUDGE_MODEL,
    BACKBONE_MODEL,
    BACKBONE_SFT_TEACHER_MODEL,
    BACKBONE_TEACHER_MODEL,
    DEFERRED_MODELS,
    assert_default_backbone,
    assert_default_is_gpt4o,
    is_deferred,
)
from common.state_schema import (
    EvidenceRef,
    StateSchema,
    StateTargets,
)
from common.typing import JSONDict

__all__ = [
    "BACKBONE_JUDGE_MODEL",
    "BACKBONE_MODEL",
    "BACKBONE_SFT_TEACHER_MODEL",
    "BACKBONE_TEACHER_MODEL",
    "DEFERRED_MODELS",
    "DOMAINS",
    "EVIDENCE_ROLES",
    "EvidenceRef",
    "GateStage",
    "GateVerdict",
    "InnerAction",
    "JSONDict",
    "RecoveryStrategy",
    "SkillSourceType",
    "SkillStatus",
    "SkillType",
    "StateSchema",
    "StateTargets",
    "assert_default_backbone",
    "assert_default_is_gpt4o",
    "is_deferred",
    "new_episode_id",
    "new_proposal_id",
    "new_run_id",
    "new_skill_id",
    "new_snapshot_id",
    "new_span_id",
    "schema_hash",
]
