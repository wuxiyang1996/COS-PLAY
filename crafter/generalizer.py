"""`Generalizer` — propose a few-shot adaptation of a source-domain skill.

Spec: PLAN-SKILL-CRAFTER §5.2, PLAN-SKILL-BANK §0.4 (source / target
asymmetry), PLAN-UNIFIED-SKILL-GATE §7 Stage 3a.

The Generalizer no longer asserts cross-domain feasibility on its own.
Instead it:
  1. Abstracts the base skill's hops by lifting domain-specific tokens
     into typed slots adapters fill in at run time.
  2. Emits a `GeneralizeProposal` with an explicit *few-shot
     adaptation recipe* — `(source_domain, target_domain, slot_remap,
     k_shot_budget, demo_episode_ids)` — that the gate's Stage 3a
     consumes through `harness.FewShotAdapter`.
  3. Falls back to the legacy "promote against a list of domains"
     path only when the caller does not supply source/target metadata
     (used by older test fixtures and the early Crafter MVP).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from common.enums import DOMAINS, SOURCE_DOMAINS, TRANSFER_TARGET_DOMAINS
from data_structure.extensions.bank_mutation_proposal import GeneralizeProposal
from data_structure.extensions.skill_record import SkillContract, SkillRecord


_DOMAIN_TOKENS = re.compile(
    r"(?:bbox|dom_id|css_selector|xpath|cell|grid_xy|tile_id|frame_index)"
)


class Generalizer:
    def generalize(
        self,
        *,
        base: SkillRecord,
        new_domains: Iterable[str] = (),
        rationale: str,
        teacher_model: Optional[str] = None,
        source_domain: Optional[str] = None,
        target_domain: Optional[str] = None,
        slot_remap: Optional[Dict[str, str]] = None,
        demo_episode_ids: Optional[Sequence[str]] = None,
        demo_selection: Optional[Dict[str, Any]] = None,
        k_shot_budget: int = 5,
    ) -> GeneralizeProposal:
        new_domains = sorted({d for d in new_domains if d in DOMAINS})

        # Few-shot transfer recipe path (preferred under the
        # source/target asymmetry).
        if source_domain or target_domain:
            self._validate_recipe(
                base=base,
                source_domain=source_domain or "",
                target_domain=target_domain or "",
            )
            target_domains = sorted(
                set(base.feasible_domains)
                | ({target_domain} if target_domain else set())
                | set(new_domains)
            )
        else:
            if not new_domains:
                raise ValueError(
                    "Generalizer requires either (source_domain, target_domain) "
                    "or at least one new target domain."
                )
            target_domains = sorted(set(base.feasible_domains) | set(new_domains))

        abstracted = [self._abstract_hop(h) for h in base.protocol]
        contract = SkillContract(
            preconditions=list(base.contract.preconditions),
            effects_add=list(base.contract.effects_add),
            effects_del=list(base.contract.effects_del),
            belief_progress=list(base.contract.belief_progress),
            grounding_progress=list(base.contract.grounding_progress),
            expected_evidence_roles=list(base.contract.expected_evidence_roles),
            success_criteria=list(base.contract.success_criteria),
            abort_criteria=list(base.contract.abort_criteria),
        )
        return GeneralizeProposal(
            name=f"{base.name}__generalized",
            rationale=rationale,
            parent_skill_ids=[base.skill_id],
            target_domains=target_domains,
            teacher_model=teacher_model,
            base_skill_id=base.skill_id,
            abstracted_protocol=abstracted,
            contract=contract,
            source_domain=source_domain or "",
            target_domain=target_domain or "",
            slot_remap=dict(slot_remap or {}),
            demo_selection=dict(demo_selection or {}),
            demo_episode_ids=list(demo_episode_ids or []),
            k_shot_budget=int(k_shot_budget),
            proposed_at=time.time(),
        )

    @staticmethod
    def _abstract_hop(hop: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(hop)
        payload = dict(out.get("payload", {}))
        for k, v in list(payload.items()):
            if isinstance(v, str) and _DOMAIN_TOKENS.search(v):
                payload[k] = "${target}"
        out["payload"] = payload
        return out

    @staticmethod
    def _validate_recipe(
        *,
        base: SkillRecord,
        source_domain: str,
        target_domain: str,
    ) -> None:
        if source_domain and source_domain not in SOURCE_DOMAINS:
            raise ValueError(
                f"source_domain={source_domain!r} not in SOURCE_DOMAINS={SOURCE_DOMAINS}; "
                f"few-shot transfer must originate in the game foundry."
            )
        if target_domain and target_domain not in TRANSFER_TARGET_DOMAINS:
            raise ValueError(
                f"target_domain={target_domain!r} not in "
                f"TRANSFER_TARGET_DOMAINS={TRANSFER_TARGET_DOMAINS}."
            )
        if (
            source_domain
            and base.source_domains
            and source_domain not in base.source_domains
        ):
            raise ValueError(
                f"Generalizer recipe source_domain={source_domain!r} does not match "
                f"base.source_domains={base.source_domains}."
            )


__all__ = ["Generalizer"]
