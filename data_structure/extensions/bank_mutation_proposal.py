"""Typed crafter proposals (PLAN-SKILL-CRAFTER §4 + PLAN-UNIFIED-SKILL-GATE §3).

Crafter outputs are *always* proposals — every one of them must pass
through the unified gate. The crafter never writes to `active_store`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from common.enums import SkillSourceType
from common.ids import new_proposal_id
from data_structure.extensions.skill_record import SkillContract


@dataclass
class _ProposalBase:
    proposal_id: str = field(default_factory=new_proposal_id)
    rationale: str = ""
    parent_skill_ids: List[str] = field(default_factory=list)
    seed_failure_ids: List[str] = field(default_factory=list)
    target_domains: List[str] = field(default_factory=list)
    teacher_model: Optional[str] = None       # frozen-only, e.g. "qwen2.5-72b@v1"
    proposed_at: Optional[float] = None

    @property
    def source_type(self) -> SkillSourceType:
        raise NotImplementedError


@dataclass
class ComposeProposal(_ProposalBase):
    """Combine N existing skills into a new skill protocol."""

    name: str = ""
    component_skill_ids: List[str] = field(default_factory=list)
    composed_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.CRAFTED


@dataclass
class GeneralizeProposal(_ProposalBase):
    """Generalize a source-domain (game) skill to additional target domains
    via few-shot adaptation.

    PLAN-SKILL-BANK §0.4 / PLAN-UNIFIED-SKILL-GATE Stage 3a — the
    crafter no longer hand-asserts that a skill is "transferable"; it
    instead emits an explicit few-shot recipe (source → target domain,
    K-shot budget, slot remap, demo selection criteria) that the gate
    runs through `harness.FewShotAdapter` before any new
    `verified_domains` entry is granted. The crafter is therefore
    responsible for *proposing the binding*, not *asserting it works*.
    """

    name: str = ""
    base_skill_id: str = ""
    abstracted_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)
    # Few-shot adaptation recipe (PLAN-UNIFIED-SKILL-GATE §7 Stage 3a).
    source_domain: str = ""                                  # ⊆ SOURCE_DOMAINS
    target_domain: str = ""                                  # ⊆ TRANSFER_TARGET_DOMAINS
    slot_remap: Dict[str, str] = field(default_factory=dict)  # base_slot → target_slot
    demo_selection: Dict[str, Any] = field(default_factory=dict)  # criteria for picking demos
    demo_episode_ids: List[str] = field(default_factory=list)     # explicit demo seed IDs
    k_shot_budget: int = 5

    @property
    def source_type(self) -> SkillSourceType:
        # If the proposal carries a concrete source/target binding it
        # is a few-shot adaptation; otherwise it is a generic
        # cross-domain transfer (legacy path).
        if self.source_domain and self.target_domain:
            return SkillSourceType.FEW_SHOT_ADAPTED
        return SkillSourceType.TRANSFERRED


@dataclass
class HypothesisProposal(_ProposalBase):
    """Net-new skill proposed from failure / rule reasoning."""

    name: str = ""
    novel_protocol: List[Dict[str, Any]] = field(default_factory=list)
    contract: SkillContract = field(default_factory=SkillContract)
    source_failure_pattern_ids: List[str] = field(default_factory=list)

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.TEACHER if self.teacher_model else SkillSourceType.CRAFTED


@dataclass
class PatchProposal(_ProposalBase):
    """A repair / patch to an existing skill (PLAN-SKILL-CRAFTER §6.5)."""

    base_skill_id: str = ""
    patched_protocol: List[Dict[str, Any]] = field(default_factory=list)
    patched_contract: Optional[SkillContract] = None
    recovery_strategy: str = ""

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.REPAIRED


@dataclass
class RetireProposal(_ProposalBase):
    """Mark a skill for deprecation."""

    target_skill_id: str = ""
    reason: str = ""

    @property
    def source_type(self) -> SkillSourceType:
        return SkillSourceType.CRAFTED


@dataclass
class RewriteProposal(_ProposalBase):
    """Rewrite an existing skill's *retrieval-payload* fields in place.

    T1.3b (lane-(a)). The live Crafter emits a RewriteProposal when the
    diagnoser flags ``STALE_DESCRIPTION`` or ``RETRIEVAL_MISLEAD`` — the
    skill's protocol and contract are *not* touched, only the textual
    fields the actor reads from when retrieving (name, description,
    slot guidance, tags, notes). This is distinct from PatchProposal:

    * **PatchProposal** is *Repairer-driven*, may alter
      ``patched_protocol`` and ``patched_contract``, and bumps the
      record's ``source_type`` to ``REPAIRED``. It belongs to the
      protocol-edit lane (lane (b)) which the live trainer turns OFF
      (``CoEvolutionConfig.crafter_enable_protocol_patching=False``);
      the offline driver may still emit it.
    * **RewriteProposal** is *Crafter-driven*, never touches the
      protocol or contract, and preserves the record's ``source_type``.
      It is safe to fire under lane-(a) live training because it
      cannot break replay / few-shot / shadow gates — only retrieval
      text changes.

    The wire format encodes this as ``"type": "RewriteProposal"`` so
    the offline mirror and CI gates can distinguish it from
    PatchProposal at audit time. Any field left ``None`` means "leave
    the existing value untouched"; this is how the Crafter signals a
    targeted edit (e.g. only ``rewritten_description``) without
    overwriting unrelated text.

    See ``implementation_notes/legacy/skill-lane-decision.md`` and
    ``implementation_notes/legacy/crafter-harness-orchestrator-roles.md`` for
    the broader lane partitioning.
    """

    base_skill_id: str = ""
    rewritten_name: Optional[str] = None
    rewritten_description: Optional[str] = None
    rewritten_retrieval_text: Optional[str] = None
    rewritten_slot_guidance: Optional[Dict[str, str]] = None
    rewritten_tags: Optional[List[str]] = None
    rewritten_notes: Optional[str] = None

    @property
    def source_type(self) -> SkillSourceType:
        # Rewrites preserve source_type because the original *protocol*
        # is untouched. The proposed source_type here is the type the
        # CRAFTER assigns to its output (CRAFTED); the lifecycle's
        # apply step keeps the underlying SkillRecord.source_type as
        # whatever it was (SEEDED / TRANSFERRED / FEW_SHOT_ADAPTED /
        # REPAIRED) — the rewrite does not change provenance. The
        # ``source_type`` returned here is consumed only by Stage-0's
        # ``proposal.source_type != skill.source_type`` check (see
        # ``orchestrator/gate_service.py::_run_static``); that check is
        # bypassed for RewriteProposal in callers that opt in by
        # short-circuiting before the static stage. New call-sites
        # SHOULD treat ``proposal.source_type`` as informational on a
        # rewrite.
        return SkillSourceType.CRAFTED


# T1.3b — ``MergeProposal`` is a *clearer alias* for ``ComposeProposal``
# (kept under both names so existing wire-format JSON, downstream
# isinstance() chains, and persisted artifacts continue to load
# unchanged). New design docs (``skill-lane-decision.md``,
# ``crafter-harness-orchestrator-roles.md``) prefer ``MergeProposal``;
# new code may import either symbol. Renaming the canonical class is
# deferred because ``proposal_to_json`` writes ``type(p).__name__`` to
# disk and 12 production call-sites and 6 plan documents reference
# ``ComposeProposal`` by name.
MergeProposal = ComposeProposal


BankMutationProposal = Union[
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
    RewriteProposal,
]


def proposal_to_json(p: BankMutationProposal) -> Dict[str, Any]:
    """Helper for `ArtifactStore`-friendly serialization."""
    out: Dict[str, Any] = {
        "type": type(p).__name__,
        "proposal_id": p.proposal_id,
        "rationale": p.rationale,
        "parent_skill_ids": list(p.parent_skill_ids),
        "seed_failure_ids": list(p.seed_failure_ids),
        "target_domains": list(p.target_domains),
        "teacher_model": p.teacher_model,
        "source_type": p.source_type.value,
        "proposed_at": p.proposed_at,
    }
    if isinstance(p, ComposeProposal):
        out.update(
            name=p.name,
            component_skill_ids=list(p.component_skill_ids),
            composed_protocol=list(p.composed_protocol),
            contract=p.contract.to_json(),
        )
    elif isinstance(p, GeneralizeProposal):
        out.update(
            name=p.name,
            base_skill_id=p.base_skill_id,
            abstracted_protocol=list(p.abstracted_protocol),
            contract=p.contract.to_json(),
            source_domain=p.source_domain,
            target_domain=p.target_domain,
            slot_remap=dict(p.slot_remap),
            demo_selection=dict(p.demo_selection),
            demo_episode_ids=list(p.demo_episode_ids),
            k_shot_budget=p.k_shot_budget,
        )
    elif isinstance(p, HypothesisProposal):
        out.update(
            name=p.name,
            novel_protocol=list(p.novel_protocol),
            contract=p.contract.to_json(),
            source_failure_pattern_ids=list(p.source_failure_pattern_ids),
        )
    elif isinstance(p, PatchProposal):
        out.update(
            base_skill_id=p.base_skill_id,
            patched_protocol=list(p.patched_protocol),
            patched_contract=p.patched_contract.to_json() if p.patched_contract else None,
            recovery_strategy=p.recovery_strategy,
        )
    elif isinstance(p, RetireProposal):
        out.update(target_skill_id=p.target_skill_id, reason=p.reason)
    elif isinstance(p, RewriteProposal):
        # T1.3b — only emit the fields that were actually rewritten
        # (None == "leave existing value"). The reader can therefore
        # distinguish "rewrite cleared the field" (explicit empty
        # string / empty list) from "rewrite did not touch the field".
        out.update(
            base_skill_id=p.base_skill_id,
            rewritten_name=p.rewritten_name,
            rewritten_description=p.rewritten_description,
            rewritten_retrieval_text=p.rewritten_retrieval_text,
            rewritten_slot_guidance=(
                dict(p.rewritten_slot_guidance)
                if p.rewritten_slot_guidance is not None
                else None
            ),
            rewritten_tags=(
                list(p.rewritten_tags)
                if p.rewritten_tags is not None
                else None
            ),
            rewritten_notes=p.rewritten_notes,
        )
    return out


__all__ = [
    "BankMutationProposal",
    "ComposeProposal",
    "GeneralizeProposal",
    "HypothesisProposal",
    "MergeProposal",
    "PatchProposal",
    "RetireProposal",
    "RewriteProposal",
    "proposal_to_json",
]
