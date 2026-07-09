"""Canonical enums shared across the Harness / Orchestrator / Crafter / Bank.

All values are normative ? see the cited canonical specs.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple


# The five general target domains every skill must be feasible in
# (PLAN-SKILL-BANK ?0.1 general-protocol invariant).
DOMAINS: Tuple[str, ...] = (
    "gymv",                # game (gym-v adapter)
    "browser",             # webagent
    "osworld",             # os-agent
    "video",               # short-video evidence-grounded reasoning
    "visual_reasoning",    # image-QA / visual reasoning
)


# Source-domain / transfer-target asymmetry
# (PLAN-SKILL-BANK ?0.4, PLAN-UNIFIED-SKILL-GATE Stage 3a).
#
# Games are the foundry where skills are first mined, hardened, and
# stress-tested under dense verifiable reward. Every other domain is a
# **transfer target** that earns its `verified_domains` entry only after
# passing the few-shot adaptation gate (G3a) on a handful of episodes.
SOURCE_DOMAINS: Tuple[str, ...] = ("gymv",)
TRANSFER_TARGET_DOMAINS: Tuple[str, ...] = (
    "browser",
    "osworld",
    "video",
    "visual_reasoning",
)


# Evidence-role taxonomy (PLAN-SKILL-BANK ?0.3 Clause B).
EVIDENCE_ROLES: Tuple[str, ...] = ("GATHER", "VERIFY", "REASON", "COMMIT")


class SkillType(str, Enum):
    """Skill categorical type used by the Harness for adapter dispatch.

    See PLAN-HARNESS ?5.1 (`SkillEpisode.skill_type`) and ?8 (unified skill
    interface).
    """

    REASONING = "reasoning"
    ACTION = "action"
    GROUNDING = "grounding"
    MIXED = "mixed"


class SkillStatus(str, Enum):
    """Canonical lifecycle (PLAN-UNIFIED-SKILL-GATE ?2.1)."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class SkillSourceType(str, Enum):
    """Where the skill came from (PLAN-UNIFIED-SKILL-GATE ?2.2).

    All sources share the same gate; there is no fast path based on
    model size, lineage, or human authorship.
    """

    MINED = "mined_from_trace"
    CRAFTED = "crafted_by_composition"
    REPAIRED = "repaired_from_failure"
    TRANSFERRED = "transferred_from_other_domain"
    FEW_SHOT_ADAPTED = "few_shot_adapted_from_source"
    TEACHER = "teacher_proposed"
    SEEDED = "human_seeded"


class GateStage(str, Enum):
    """Unified gate stages (PLAN-UNIFIED-SKILL-GATE ?7)."""

    STATIC = "static"
    REPLAY = "replay"
    SHADOW = "shadow"
    TRANSFER = "transfer"
    NON_REGRESSION = "non_regression"


class GateVerdict(str, Enum):
    """Per-stage and final verdicts (PLAN-UNIFIED-SKILL-GATE ?3.3)."""

    PASS = "pass"
    FAIL = "fail"
    LIMITED_PASS = "limited_pass"


class InnerAction(str, Enum):
    """Inner-MDP action vocabulary (PLAN-ACTION-AGENT ?5)."""

    GROUND = "GROUND"
    CHECK = "CHECK"
    RETRIEVE = "RETRIEVE"
    COMMIT = "COMMIT"
    EXECUTE = "EXECUTE"


class RecoveryStrategy(str, Enum):
    """Recovery strategies (PLAN-SKILL-CRAFTER S6.5).

    The first six values (``PROTOCOL_PATCH`` ... ``REGROUNDING_TRIGGER``)
    are *protocol-edit* strategies -- they correspond to lane-(b)
    Repairer mints (``PatchProposal`` with the strategy attached) and
    are gated off by default in the live trainer
    (``CoEvolutionConfig.crafter_enable_protocol_patching=False``).

    ``SKILL_RETIREMENT`` is the lane-spanning retirement signal -- both
    lanes emit ``RetireProposal`` for it.

    The final three values (``BANK_GAP`` / ``RETRIEVAL_MISLEAD`` /
    ``STALE_DESCRIPTION``) are the **lane-(a) retrieval-centric**
    additions (T1.3c). They are *not* protocol edits -- they are
    routed by ``crafter/service.py::_run_failure_dispatch`` directly
    to the Hypothesizer (BANK_GAP), the Composer + Hypothesizer
    (RETRIEVAL_MISLEAD), or a Rewrite/Hypothesizer pair
    (STALE_DESCRIPTION). These never persist as
    ``PatchProposal.recovery_strategy`` because the dispatcher skips
    the Repairer for them. The string values mirror
    ``configs/failure_routing.yaml``'s ``lane_a_taxonomy`` block.
    """

    PROTOCOL_PATCH = "protocol_patch"
    PRECONDITION_STRENGTHENING = "precondition_strengthening"
    FALLBACK_INJECTION = "fallback_injection"
    HOP_INSERTION = "hop_insertion"
    SKILL_DECOMPOSITION = "skill_decomposition"
    REGROUNDING_TRIGGER = "regrounding_trigger"
    SKILL_RETIREMENT = "skill_retirement"
    # T1.3c -- lane-(a) retrieval taxonomy (skill = retrieval payload).
    BANK_GAP = "bank_gap"
    RETRIEVAL_MISLEAD = "retrieval_mislead"
    STALE_DESCRIPTION = "stale_description"


# T1.3c -- convenience set used by ``crafter/service.py`` to short-circuit
# Repairer-bound strategies and route the lane-(a) signal directly to
# the Hypothesizer / Composer / Rewriter. Membership tests against this
# set are cheaper than enum chains and they keep the Repairer's
# branch table unchanged.
LANE_A_RECOVERY_STRATEGIES = frozenset({
    RecoveryStrategy.BANK_GAP,
    RecoveryStrategy.RETRIEVAL_MISLEAD,
    RecoveryStrategy.STALE_DESCRIPTION,
})


__all__ = [
    "DOMAINS",
    "EVIDENCE_ROLES",
    "GateStage",
    "GateVerdict",
    "InnerAction",
    "LANE_A_RECOVERY_STRATEGIES",
    "RecoveryStrategy",
    "SOURCE_DOMAINS",
    "SkillSourceType",
    "SkillStatus",
    "SkillType",
    "TRANSFER_TARGET_DOMAINS",
]
