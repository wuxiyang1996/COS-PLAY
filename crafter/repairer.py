"""`Repairer` — turn a (skill, failure pattern, diagnosis) triple into a
typed `PatchProposal`.

Spec: PLAN-SKILL-CRAFTER §6.5 (recovery strategies) and the crafter
README's Phase D entry — *"PatchProposal repair plumbing exposed via
`SkillCrafterService.propose_repair`."*

The repairer is the only crafter component that mutates an *existing*
skill's body (protocol / contract). Like the other proposers it is
purely a *proposal*: the patch lands as a `PatchProposal` (and a DRAFT
`SkillRecord` carrying `parent_skill_ids=[base_skill_id]` and the
patched `content_hash`); the gate revalidates from scratch before
anything ACTIVE is touched.

Design notes
------------

* The mapping from `RecoveryStrategy` to a concrete protocol/contract
  edit is intentionally a small, deterministic rule table.  Anything
  smarter (LLM-rewritten hops, multi-step patches) is a future
  enhancement — the rule path keeps the pipeline runnable without an
  external model and is easy to test.
* A teacher-LLM hook (`set_llm_repairer`) is provided for symmetry
  with `FailureDiagnoser` / `Hypothesizer`.  It is consulted first;
  any exception or `None` falls through to the rule path.
* The repairer never imports `skill_bank.stores` or the lifecycle
  manager (invariant 1 + 2).  It receives the base `SkillRecord` from
  `SkillCrafterService`, which is the only crafter component
  permitted to read the bank (via the lifecycle manager's
  read-through `get`).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from common.enums import EVIDENCE_ROLES, RecoveryStrategy
from data_structure.extensions.bank_mutation_proposal import PatchProposal
from data_structure.extensions.failure_trace import FailureDiagnosis
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from crafter.failure_memory import FailurePattern


_LLMRepairer = Callable[
    [SkillRecord, FailurePattern, FailureDiagnosis], Optional[PatchProposal]
]


class Repairer:
    """Build a `PatchProposal` from a base skill + diagnosed pattern."""

    def __init__(self, llm: Optional[_LLMRepairer] = None) -> None:
        self._llm = llm

    def set_llm_repairer(self, llm: _LLMRepairer) -> None:
        self._llm = llm

    def repair(
        self,
        *,
        base: SkillRecord,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
        teacher_model: Optional[str] = None,
        rationale: Optional[str] = None,
    ) -> Optional[PatchProposal]:
        """Return a `PatchProposal` patching `base` against `pattern`.

        Returns ``None`` when the diagnosis recommends retirement
        (callers should route to `propose_retirement` instead).
        """
        if self._llm is not None:
            try:
                proposal = self._llm(base, pattern, diagnosis)
                if proposal is not None:
                    return proposal
            except Exception:                                  # noqa: BLE001
                pass
        return self._rule_repair(
            base=base,
            pattern=pattern,
            diagnosis=diagnosis,
            teacher_model=teacher_model,
            rationale=rationale,
        )

    # -- rule path --------------------------------------------------------

    def _rule_repair(
        self,
        *,
        base: SkillRecord,
        pattern: FailurePattern,
        diagnosis: FailureDiagnosis,
        teacher_model: Optional[str],
        rationale: Optional[str],
    ) -> Optional[PatchProposal]:
        strategy = diagnosis.recommended_strategy
        if strategy == RecoveryStrategy.SKILL_RETIREMENT:
            # Repair is not the right channel for this pattern; signal
            # to the caller (typically `SkillCrafterService.propose_repair`)
            # so it can dispatch to `propose_retirement` instead.
            return None

        protocol = [dict(h) for h in base.protocol]
        contract = _clone_contract(base.contract)
        idx = _resolve_step_index(pattern.failed_step_index, protocol)

        if strategy == RecoveryStrategy.HOP_INSERTION:
            self._insert_verify_hop(protocol, idx, contract)
        elif strategy == RecoveryStrategy.PRECONDITION_STRENGTHENING:
            self._strengthen_precondition(contract, pattern)
        elif strategy == RecoveryStrategy.FALLBACK_INJECTION:
            self._inject_fallback(protocol, idx, contract)
        elif strategy == RecoveryStrategy.REGROUNDING_TRIGGER:
            self._insert_reground_hop(protocol, idx, contract)
        elif strategy == RecoveryStrategy.PROTOCOL_PATCH:
            self._patch_protocol_step(protocol, idx, contract)
        elif strategy == RecoveryStrategy.SKILL_DECOMPOSITION:
            # No direct in-place edit captures decomposition; emit a
            # protocol-level annotation that the gate can pick up so a
            # follow-up Composer pass knows where to slice.
            self._annotate_decomposition(protocol, idx)
        else:
            return None

        if not rationale:
            rationale = (
                f"repair[{strategy.value}] for pattern={pattern.pattern_id}: "
                f"{diagnosis.root_cause}"
            )

        return PatchProposal(
            rationale=rationale,
            parent_skill_ids=[base.skill_id],
            seed_failure_ids=list(pattern.failure_ids),
            target_domains=list(base.feasible_domains),
            teacher_model=teacher_model,
            base_skill_id=base.skill_id,
            patched_protocol=protocol,
            patched_contract=contract,
            recovery_strategy=strategy.value,
            proposed_at=time.time(),
        )

    # -- per-strategy edits ----------------------------------------------

    @staticmethod
    def _insert_verify_hop(
        protocol: List[Dict[str, Any]],
        idx: int,
        contract: SkillContract,
    ) -> None:
        verify_hop = {
            "action": "VERIFY",
            "payload": {"target": "${commit_target}"},
            "notes": "auto-inserted by repairer (HOP_INSERTION)",
        }
        protocol.insert(idx, verify_hop)
        _ensure_evidence_role(contract, "VERIFY")
        _ensure_in(contract.success_criteria, "verify_passed")
        _ensure_in(contract.abort_criteria, "verify_failed")

    @staticmethod
    def _strengthen_precondition(
        contract: SkillContract,
        pattern: FailurePattern,
    ) -> None:
        seed = pattern.sample_abort_reasons[0] if pattern.sample_abort_reasons else ""
        new_pre = _precondition_from_reason(seed) or "preconditions_strengthened"
        _ensure_in(contract.preconditions, new_pre)

    @staticmethod
    def _inject_fallback(
        protocol: List[Dict[str, Any]],
        idx: int,
        contract: SkillContract,
    ) -> None:
        # Wrap the failed hop with a marker the action-layer / harness
        # adapter can read as "try original, on adapter exception attempt
        # this fallback". The adapter contract for `_fallback` is
        # tracked in PLAN-HARNESS §5.4 (fallback dispatch).
        if 0 <= idx < len(protocol):
            original = dict(protocol[idx])
            original.setdefault("_fallback", []).append(
                {"action": "RETRY", "payload": {}, "notes": "auto-inserted fallback"}
            )
            protocol[idx] = original
        else:
            protocol.append(
                {
                    "action": "RETRY",
                    "payload": {},
                    "notes": "auto-inserted fallback (tail)",
                }
            )
        _ensure_in(contract.abort_criteria, "fallback_exhausted")

    @staticmethod
    def _insert_reground_hop(
        protocol: List[Dict[str, Any]],
        idx: int,
        contract: SkillContract,
    ) -> None:
        reground_hop = {
            "action": "GROUND",
            "payload": {"target": "${commit_target}", "force_refresh": True},
            "notes": "auto-inserted by repairer (REGROUNDING_TRIGGER)",
        }
        protocol.insert(idx, reground_hop)
        _ensure_evidence_role(contract, "GATHER")
        _ensure_in(contract.grounding_progress, "confirms(target)")

    @staticmethod
    def _patch_protocol_step(
        protocol: List[Dict[str, Any]],
        idx: int,
        contract: SkillContract,
    ) -> None:
        replacement = [
            {
                "action": "GROUND",
                "payload": {"target": "${commit_target}"},
                "notes": "auto-patched by repairer (PROTOCOL_PATCH) — observe before act",
            },
            {
                "action": "EXECUTE",
                "payload": {"target": "${commit_target}"},
                "notes": "auto-patched by repairer (PROTOCOL_PATCH) — act on fresh observation",
            },
        ]
        if 0 <= idx < len(protocol):
            protocol[idx : idx + 1] = replacement
        else:
            protocol.extend(replacement)
        _ensure_evidence_role(contract, "GATHER")
        _ensure_in(contract.grounding_progress, "confirms(target)")
        _ensure_in(contract.success_criteria, "execute_succeeded")

    @staticmethod
    def _annotate_decomposition(
        protocol: List[Dict[str, Any]],
        idx: int,
    ) -> None:
        if 0 <= idx < len(protocol):
            hop = dict(protocol[idx])
            hop["_decompose_hint"] = True
            protocol[idx] = hop


# ---- helpers ----------------------------------------------------------------


def _clone_contract(c: SkillContract) -> SkillContract:
    return SkillContract(
        preconditions=list(c.preconditions),
        effects_add=list(c.effects_add),
        effects_del=list(c.effects_del),
        belief_progress=list(c.belief_progress),
        grounding_progress=list(c.grounding_progress),
        expected_evidence_roles=list(c.expected_evidence_roles),
        success_criteria=list(c.success_criteria),
        abort_criteria=list(c.abort_criteria),
    )


def _resolve_step_index(
    failed_step_index: Optional[int], protocol: List[Dict[str, Any]]
) -> int:
    """Clamp ``failed_step_index`` to ``[0, len(protocol)]`` (inclusive on the
    right because `list.insert` accepts the tail position).  Falls back to
    the *last* step when the failure has no positional information so that
    inserted hops appear next to the most likely failing region rather than
    silently shifting the entire chain."""
    if failed_step_index is None:
        return max(0, len(protocol) - 1)
    if failed_step_index < 0:
        return 0
    if failed_step_index > len(protocol):
        return len(protocol)
    return failed_step_index


def _ensure_in(items: List[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _ensure_evidence_role(contract: SkillContract, role: str) -> None:
    if role not in EVIDENCE_ROLES:
        raise ValueError(f"role={role!r} not in {EVIDENCE_ROLES}")
    if role not in contract.expected_evidence_roles:
        contract.expected_evidence_roles.append(role)


_REASON_TOKEN_TO_PRECONDITION = {
    "empty evidence": "evidence_in_nonempty",
    "missing target": "have_target",
    "stale": "state_fresh",
    "blocked": "path_clear",
    "not visible": "target_visible",
    "permission": "has_permission",
}


def _precondition_from_reason(reason: str) -> Optional[str]:
    if not reason:
        return None
    low = reason.lower()
    for needle, predicate in _REASON_TOKEN_TO_PRECONDITION.items():
        if needle in low:
            return predicate
    return None


__all__ = ["Repairer"]
