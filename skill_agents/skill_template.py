"""
Transferable skill template — domain-agnostic skill representation.

Extends the existing ``Skill`` schema with cross-domain abstractions:

- **SlotBinding**: maps domain-specific predicates to shared schema slots
  (target, blocker, constraint, candidate_set, history_anchor).
- **ReasoningProtocol**: multi-step policy over inner MDP actions
  (GROUND, CHECK, RETRIEVE, CONCLUDE, EXECUTE).
- **AbstractPredicate**: parameterised predicate that instantiates
  per-domain (e.g. ``slot.target.value >= $threshold``).
- **TransferableSkill**: wraps a ``Skill`` with the above, enabling
  cross-domain retrieval, transfer, and composition.

Design constraints:
  - Compatible with existing ``Skill`` / ``SkillBankMVP`` serialisation.
  - Uses shared slot names from the canonical ``<state>`` schema
    (see ``plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`` §3).
  - Inner MDP action vocabulary from
    ``plans/02-action-agent/PLAN-ACTION-AGENT.md`` §5.

Usage::

    from skill_agents.skill_template import (
        TransferableSkill, SlotBinding, ReasoningProtocol,
        HopStep, AbstractPredicate,
    )

    # Build a transferable skill from an existing bank skill
    ts = TransferableSkill.from_skill(skill, domain="gymv")
    ts.slot_bindings.append(SlotBinding(
        slot="target", domain_predicate="highest_tile_positioned",
    ))
    ts.reasoning_protocol = ReasoningProtocol(hops=[
        HopStep(action="GROUND", query="locate target entity"),
        HopStep(action="CHECK",  query="verify constraint not violated"),
        HopStep(action="CONCLUDE", query="subgoal = best candidate"),
        HopStep(action="EXECUTE", query="apply action to target"),
    ])

    # Instantiate for a new domain
    browser_skill = ts.instantiate(
        domain="browser",
        slot_map={"target": "form_field", "constraint": "validation_error"},
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from skill_agents.stage3_mvp.schemas import (
    ExecutionHint,
    Protocol,
    Skill,
    SkillEffectsContract,
)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

SHARED_SLOTS = ("target", "blocker", "constraint", "candidate_set", "history_anchor")

class InnerAction(str, Enum):
    GROUND = "GROUND"
    CHECK = "CHECK"
    RETRIEVE = "RETRIEVE"
    CONCLUDE = "CONCLUDE"
    EXECUTE = "EXECUTE"

SKILL_FAMILIES = (
    "locate_filter_select",
    "blocker_prerequisite_replan",
    "history_hidden_state_act",
    "compare_under_constraint",
)

DOMAINS = ("gymv", "browser", "desktop", "image_qa", "video_qa")


# ═══════════════════════════════════════════════════════════════════════
# Slot binding — maps domain predicates to shared schema slots
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SlotBinding:
    """Maps a domain-specific predicate to a shared schema slot.

    Example: in 2048, ``highest_tile_positioned`` → slot ``target``.
    In browser, ``focused_element`` → slot ``target``.
    The abstract predicate ``slot.target.positioned`` transfers across both.
    """

    slot: str                       # one of SHARED_SLOTS
    domain_predicate: str           # original predicate key from the domain
    domain: str = ""                # gymv, browser, desktop, image_qa, video_qa
    direction: str = "eff_add"      # eff_add | eff_del | eff_event
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "domain_predicate": self.domain_predicate,
            "domain": self.domain,
            "direction": self.direction,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SlotBinding:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ═══════════════════════════════════════════════════════════════════════
# Abstract predicate — parameterised over domains
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AbstractPredicate:
    """Domain-agnostic predicate expressed in terms of schema slots.

    ``template`` uses ``$slot`` placeholders:
      - ``$target.value >= $threshold``
      - ``$blocker == null``
      - ``adjacent($target, $candidate_set)``

    ``instantiations`` maps domain → concrete predicate string.
    """

    template: str                                     # e.g. "$target.positioned"
    direction: str = "eff_add"                        # eff_add | eff_del | eff_event
    instantiations: Dict[str, str] = field(default_factory=dict)

    def instantiate(self, domain: str, slot_map: Dict[str, str]) -> str:
        """Substitute slots with domain-specific terms."""
        result = self.template
        for slot, concrete in slot_map.items():
            result = result.replace(f"${slot}", concrete)
        return result

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "direction": self.direction,
            "instantiations": self.instantiations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AbstractPredicate:
        return cls(
            template=d.get("template", ""),
            direction=d.get("direction", "eff_add"),
            instantiations=d.get("instantiations", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# Reasoning protocol — inner MDP hop chain
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HopStep:
    """One hop in a reasoning protocol (inner MDP action)."""

    action: str              # GROUND | CHECK | RETRIEVE | CONCLUDE | EXECUTE
    query: str = ""          # parameterised query (uses $slot placeholders)
    slot_refs: List[str] = field(default_factory=list)  # slots this hop reads/writes
    fallback: str = ""       # alternative action on failure

    def to_dict(self) -> dict:
        d: dict = {"action": self.action, "query": self.query}
        if self.slot_refs:
            d["slot_refs"] = self.slot_refs
        if self.fallback:
            d["fallback"] = self.fallback
        return d

    @classmethod
    def from_dict(cls, d: dict) -> HopStep:
        return cls(
            action=d.get("action", "EXECUTE"),
            query=d.get("query", ""),
            slot_refs=d.get("slot_refs", []),
            fallback=d.get("fallback", ""),
        )


@dataclass
class ReasoningProtocol:
    """Multi-step reasoning policy over inner MDP actions.

    Each hop is an explicit step in the inner MDP.  The protocol
    defines *how to think* about executing the skill, not just
    *what to do*.

    ``trigger`` fires the protocol; ``hops`` is the ordered chain;
    ``max_hops`` caps the inner loop.
    """

    trigger: str = ""                          # condition that activates this protocol
    hops: List[HopStep] = field(default_factory=list)
    max_hops: int = 8
    family: str = ""                           # one of SKILL_FAMILIES or custom

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    @property
    def action_sequence(self) -> List[str]:
        return [h.action for h in self.hops]

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "hops": [h.to_dict() for h in self.hops],
            "max_hops": self.max_hops,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReasoningProtocol:
        if not d:
            return cls()
        return cls(
            trigger=d.get("trigger", ""),
            hops=[HopStep.from_dict(h) for h in d.get("hops", [])],
            max_hops=d.get("max_hops", 8),
            family=d.get("family", ""),
        )


# ═══════════════════════════════════════════════════════════════════════
# Transferable skill — the cross-domain skill template
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TransferableSkill:
    """Domain-agnostic skill template built from observed domain skills.

    Wraps the concrete ``Skill`` with abstractions that enable
    cross-domain retrieval, transfer, and composition:

    - ``slot_bindings``: how domain predicates map to shared schema slots
    - ``abstract_effects``: parameterised eff_add/eff_del/eff_event
    - ``reasoning_protocol``: inner MDP hop chain
    - ``source_skills``: concrete domain skills this was derived from
    - ``family``: which transferable skill family this belongs to
    - ``transferability_score``: how well this skill transfers (0..1)
    """

    template_id: str = ""
    family: str = ""                           # one of SKILL_FAMILIES or custom
    name: str = ""
    description: str = ""                      # domain-agnostic strategic description
    tags: List[str] = field(default_factory=list)

    # Cross-domain abstractions
    slot_bindings: List[SlotBinding] = field(default_factory=list)
    abstract_effects: List[AbstractPredicate] = field(default_factory=list)
    reasoning_protocol: ReasoningProtocol = field(default_factory=ReasoningProtocol)

    # Source provenance
    source_skills: List[Dict[str, str]] = field(default_factory=list)
    source_domains: List[str] = field(default_factory=list)

    # Quality metrics
    transferability_score: float = 0.0
    n_domain_instances: int = 0
    n_domains: int = 0
    avg_pass_rate: float = 0.0

    created_at: float = field(default_factory=time.time)

    # ── Construction ─────────────────────────────────────────────

    @classmethod
    def from_skill(
        cls,
        skill: Skill,
        domain: str = "",
        family: str = "",
    ) -> TransferableSkill:
        """Wrap a concrete Skill as a transferable template."""
        ts = cls(
            template_id=f"xfer_{skill.skill_id}",
            family=family,
            name=skill.name,
            description=skill.strategic_description,
            tags=list(skill.tags),
            source_skills=[{"skill_id": skill.skill_id, "domain": domain}],
            source_domains=[domain] if domain else [],
            n_domain_instances=skill.n_instances,
            n_domains=1,
            avg_pass_rate=skill.success_rate,
        )

        if skill.contract:
            for pred in skill.contract.eff_add:
                ts.abstract_effects.append(AbstractPredicate(
                    template=pred,
                    direction="eff_add",
                    instantiations={domain: pred} if domain else {},
                ))
            for pred in skill.contract.eff_del:
                ts.abstract_effects.append(AbstractPredicate(
                    template=pred,
                    direction="eff_del",
                    instantiations={domain: pred} if domain else {},
                ))

        return ts

    # ── Instantiation ────────────────────────────────────────────

    def instantiate(
        self,
        domain: str,
        slot_map: Optional[Dict[str, str]] = None,
        game_name: str = "",
    ) -> Skill:
        """Produce a concrete ``Skill`` for a target domain.

        ``slot_map`` maps shared slots to domain-specific entity names.
        """
        _map = slot_map or {}
        skill_id = f"{self.template_id}__{domain}"
        if game_name:
            skill_id = f"{self.template_id}__{game_name}"

        # Instantiate abstract effects into concrete predicates
        eff_add: Set[str] = set()
        eff_del: Set[str] = set()
        for ap in self.abstract_effects:
            concrete = ap.instantiations.get(domain)
            if concrete is None and _map:
                concrete = ap.instantiate(domain, _map)
            if concrete:
                if ap.direction == "eff_add":
                    eff_add.add(concrete)
                elif ap.direction == "eff_del":
                    eff_del.add(concrete)

        contract = SkillEffectsContract(
            skill_id=skill_id,
            eff_add=eff_add,
            eff_del=eff_del,
        )

        # Convert reasoning protocol to action-level protocol
        steps = []
        for hop in self.reasoning_protocol.hops:
            query = hop.query
            for slot, concrete in _map.items():
                query = query.replace(f"${slot}", concrete)
            steps.append(f"[{hop.action}] {query}")

        protocol = Protocol(
            steps=steps,
            preconditions=[],
            success_criteria=[],
            abort_criteria=[],
            expected_duration=len(steps) * 2,
            source="transfer",
        )

        return Skill(
            skill_id=skill_id,
            name=self.name,
            strategic_description=self.description,
            tags=list(self.tags),
            protocol=protocol,
            contract=contract,
        )

    # ── Merging ──────────────────────────────────────────────────

    def merge_source(
        self,
        other_skill: Skill,
        domain: str,
        bindings: Optional[List[SlotBinding]] = None,
    ) -> None:
        """Incorporate another domain skill as additional evidence."""
        self.source_skills.append({
            "skill_id": other_skill.skill_id,
            "domain": domain,
        })
        if domain not in self.source_domains:
            self.source_domains.append(domain)
        self.n_domains = len(self.source_domains)
        self.n_domain_instances += other_skill.n_instances

        if bindings:
            self.slot_bindings.extend(bindings)

        if other_skill.contract:
            for pred in other_skill.contract.eff_add:
                existing = [
                    a for a in self.abstract_effects
                    if a.direction == "eff_add"
                    and domain not in a.instantiations
                ]
                if existing:
                    existing[0].instantiations[domain] = pred
                else:
                    self.abstract_effects.append(AbstractPredicate(
                        template=pred,
                        direction="eff_add",
                        instantiations={domain: pred},
                    ))

    # ── Scoring ──────────────────────────────────────────────────

    def compute_transferability(self) -> float:
        """Score how transferable this skill is (0..1).

        Factors:
        - domain_coverage: fraction of known domains with instances
        - slot_coverage: fraction of shared slots with bindings
        - protocol_quality: whether reasoning protocol is defined
        - evidence_weight: amount of supporting data
        """
        domain_coverage = min(1.0, self.n_domains / 3.0)
        bound_slots = len(set(b.slot for b in self.slot_bindings))
        slot_coverage = bound_slots / len(SHARED_SLOTS)
        protocol_quality = (
            0.8 if self.reasoning_protocol.hops else 0.2
        )
        evidence_weight = min(1.0, self.n_domain_instances / 10.0)

        self.transferability_score = (
            0.35 * domain_coverage
            + 0.25 * slot_coverage
            + 0.25 * protocol_quality
            + 0.15 * evidence_weight
        )
        return self.transferability_score

    # ── Serialisation ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "family": self.family,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "slot_bindings": [b.to_dict() for b in self.slot_bindings],
            "abstract_effects": [a.to_dict() for a in self.abstract_effects],
            "reasoning_protocol": self.reasoning_protocol.to_dict(),
            "source_skills": self.source_skills,
            "source_domains": self.source_domains,
            "transferability_score": self.transferability_score,
            "n_domain_instances": self.n_domain_instances,
            "n_domains": self.n_domains,
            "avg_pass_rate": self.avg_pass_rate,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransferableSkill:
        return cls(
            template_id=d.get("template_id", ""),
            family=d.get("family", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            slot_bindings=[
                SlotBinding.from_dict(b) for b in d.get("slot_bindings", [])
            ],
            abstract_effects=[
                AbstractPredicate.from_dict(a)
                for a in d.get("abstract_effects", [])
            ],
            reasoning_protocol=ReasoningProtocol.from_dict(
                d.get("reasoning_protocol", {})
            ),
            source_skills=d.get("source_skills", []),
            source_domains=d.get("source_domains", []),
            transferability_score=d.get("transferability_score", 0.0),
            n_domain_instances=d.get("n_domain_instances", 0),
            n_domains=d.get("n_domains", 0),
            avg_pass_rate=d.get("avg_pass_rate", 0.0),
            created_at=d.get("created_at", 0.0),
        )


# ═══════════════════════════════════════════════════════════════════════
# Predefined reasoning protocol templates for the 4 skill families
# ═══════════════════════════════════════════════════════════════════════

def make_locate_filter_select() -> ReasoningProtocol:
    """Family 1: Locate → filter → select.

    Game: candidate moves → best legal.
    Browser: UI candidates → relevant control.
    Image QA: objects → attributes → answer target.
    """
    return ReasoningProtocol(
        trigger="$candidate_set is non-empty AND $target is unresolved",
        family="locate_filter_select",
        hops=[
            HopStep(
                action="GROUND",
                query="locate $candidate_set entities",
                slot_refs=["candidate_set"],
            ),
            HopStep(
                action="CHECK",
                query="filter by $constraint",
                slot_refs=["candidate_set", "constraint"],
            ),
            HopStep(
                action="CONCLUDE",
                query="select best $target from filtered set",
                slot_refs=["target", "candidate_set"],
            ),
            HopStep(
                action="EXECUTE",
                query="apply action to $target",
                slot_refs=["target"],
            ),
        ],
    )


def make_blocker_prerequisite_replan() -> ReasoningProtocol:
    """Family 2: Blocker → prerequisite → replan.

    Game: deadlock → missing setup → fix first.
    Browser: disabled control → missing field → fill first.
    Image QA: weak evidence → gather anchor → re-examine.
    """
    return ReasoningProtocol(
        trigger="$blocker is not null OR state_flags.error is not null",
        family="blocker_prerequisite_replan",
        hops=[
            HopStep(
                action="GROUND",
                query="identify $blocker entity",
                slot_refs=["blocker"],
            ),
            HopStep(
                action="CHECK",
                query="what $constraint is violated by $blocker",
                slot_refs=["blocker", "constraint"],
            ),
            HopStep(
                action="RETRIEVE",
                query="find skill that resolves $blocker",
                slot_refs=["blocker", "history_anchor"],
            ),
            HopStep(
                action="CONCLUDE",
                query="subgoal = resolve $blocker before $target",
                slot_refs=["blocker", "target"],
            ),
            HopStep(
                action="EXECUTE",
                query="action addressing $blocker",
                slot_refs=["blocker"],
            ),
        ],
    )


def make_history_hidden_state_act() -> ReasoningProtocol:
    """Family 3: History → hidden state → act.

    Game: dialogue → alliance/threat inference → act.
    Browser: prior pages → session state → next step.
    Video QA: prior frames → disambiguate → answer.
    """
    return ReasoningProtocol(
        trigger="$history_anchor exists AND $target requires disambiguation",
        family="history_hidden_state_act",
        hops=[
            HopStep(
                action="RETRIEVE",
                query="recall $history_anchor context",
                slot_refs=["history_anchor"],
            ),
            HopStep(
                action="CHECK",
                query="infer hidden state from $history_anchor",
                slot_refs=["history_anchor", "target"],
            ),
            HopStep(
                action="GROUND",
                query="verify inference against current $target",
                slot_refs=["target"],
            ),
            HopStep(
                action="CONCLUDE",
                query="commit action given inferred state",
                slot_refs=["target"],
            ),
            HopStep(
                action="EXECUTE",
                query="act on $target with historical context",
                slot_refs=["target", "history_anchor"],
            ),
        ],
    )


def make_compare_under_constraint() -> ReasoningProtocol:
    """Family 4: Compare under future constraint.

    Game: move preserving board structure.
    Browser: path minimising risk/steps.
    Image QA: candidate consistent with visual constraints.
    """
    return ReasoningProtocol(
        trigger="$candidate_set has multiple options AND $constraint is active",
        family="compare_under_constraint",
        hops=[
            HopStep(
                action="GROUND",
                query="enumerate $candidate_set",
                slot_refs=["candidate_set"],
            ),
            HopStep(
                action="CHECK",
                query="evaluate each candidate against $constraint",
                slot_refs=["candidate_set", "constraint"],
            ),
            HopStep(
                action="CHECK",
                query="project future state for top candidates",
                slot_refs=["candidate_set", "target"],
                fallback="RETRIEVE",
            ),
            HopStep(
                action="CONCLUDE",
                query="select $target that best satisfies $constraint",
                slot_refs=["target", "constraint"],
            ),
            HopStep(
                action="EXECUTE",
                query="apply chosen action to $target",
                slot_refs=["target"],
            ),
        ],
    )


FAMILY_PROTOCOLS: Dict[str, ReasoningProtocol] = {
    "locate_filter_select": make_locate_filter_select(),
    "blocker_prerequisite_replan": make_blocker_prerequisite_replan(),
    "history_hidden_state_act": make_history_hidden_state_act(),
    "compare_under_constraint": make_compare_under_constraint(),
}
