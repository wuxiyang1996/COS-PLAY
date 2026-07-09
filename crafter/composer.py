"""`Composer` — combine N existing skills into a candidate composition.

Spec: PLAN-SKILL-CRAFTER §5.1.

Composition is *protocol concatenation* with shared bindings. Whether the
composed protocol is actually useful is decided by the gate, not us.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional

from common.enums import SkillType
from data_structure.extensions.bank_mutation_proposal import ComposeProposal
from data_structure.extensions.skill_record import SkillContract, SkillRecord


class Composer:
    def compose(
        self,
        *,
        components: Iterable[SkillRecord],
        name: str,
        rationale: str,
        target_domains: Optional[List[str]] = None,
        teacher_model: Optional[str] = None,
    ) -> ComposeProposal:
        comps = list(components)
        if len(comps) < 2:
            raise ValueError("Composer requires at least two component skills.")

        # Concatenate protocols, prefixing each hop with its source skill_id
        # so the gate can reason about composition lineage.
        combined_protocol: List[dict] = []
        for c in comps:
            for hop in c.protocol:
                combined_protocol.append({**hop, "_source_skill_id": c.skill_id})

        # Union the contracts (a *very* coarse approximation; the gate
        # validates the actual semantics).
        contract = SkillContract(
            preconditions=_unique([p for c in comps for p in c.contract.preconditions]),
            effects_add=_unique([e for c in comps for e in c.contract.effects_add]),
            effects_del=_unique([e for c in comps for e in c.contract.effects_del]),
            belief_progress=_unique([b for c in comps for b in c.contract.belief_progress]),
            grounding_progress=_unique([g for c in comps for g in c.contract.grounding_progress]),
            expected_evidence_roles=_unique(
                [r for c in comps for r in c.contract.expected_evidence_roles]
            ),
            success_criteria=_unique([s for c in comps for s in c.contract.success_criteria]),
            abort_criteria=_unique([a for c in comps for a in c.contract.abort_criteria]),
        )

        # Domain feasibility = intersection of components (≥ 2 required for
        # the gate to allow ACTIVE promotion later).
        domain_sets = [set(c.feasible_domains) for c in comps]
        feasible = sorted(set.intersection(*domain_sets)) if domain_sets else []
        if target_domains:
            feasible = [d for d in feasible if d in set(target_domains)]

        return ComposeProposal(
            name=name,
            rationale=rationale,
            parent_skill_ids=[c.skill_id for c in comps],
            target_domains=feasible,
            teacher_model=teacher_model,
            component_skill_ids=[c.skill_id for c in comps],
            composed_protocol=combined_protocol,
            contract=contract,
            proposed_at=time.time(),
        )


def _unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


__all__ = ["Composer"]
