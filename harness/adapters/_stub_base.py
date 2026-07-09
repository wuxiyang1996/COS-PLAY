"""Shared boilerplate for transfer-target adapters.

PLAN-HARNESS §5.4 — adapters are the only place that talk to a concrete
env / tool surface. The four non-game adapters (`browser`, `osworld`,
`video`, `visual_reasoning`) all share the same hop-loop scaffolding:
walk `skill.protocol`, resolve slots, call a pluggable executor, honor
budget. We factor that loop here so each transfer-target adapter only
has to declare its `name`, `supported_types`, and (later) plug in its
real `HopExecutor` via `set_executor`.

The default executor is deterministic and evidence-emitting; it is
sufficient for the gate's dry-run path and for stub testing under
PLAN-COMPONENTS-IMPLEMENTATION Phase A.5 / Phase D, where target-domain
binding is exercised before the real env wiring is wired in.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from common.enums import SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapters._common import (
    BudgetGuard,
    HopBindings,
    iter_hops,
    normalize_hop_action,
)
from harness.skill_adapter import AdapterRunContext, AdapterRunResult, SkillAdapter

HopExecutor = Callable[[str, Dict[str, Any], AdapterRunContext], Dict[str, Any]]


_ACTION_VERB_TO_ROLE: Dict[str, str] = {
    # Day-7d: Map cold-start prose verbs / typed-hop actions to the
    # evidence role the stub executor should emit. Ensures the
    # action-level ReplayValidator's evidence-non-worse check has
    # *something* to compare against on transfer-target seeds, even
    # before the typed-hop migration (§21) lifts protocol shape.
    "GATHER": "GATHER", "OBSERVE": "GATHER", "INSPECT": "GATHER",
    "READ": "GATHER", "PERCEIVE": "GATHER", "SCAN": "GATHER",
    "VERIFY": "VERIFY", "VALIDATE": "VERIFY", "CHECK": "VERIFY",
    "CONFIRM": "VERIFY", "TEST": "VERIFY",
    "REASON": "REASON", "INFER": "REASON", "DEDUCE": "REASON",
    "PLAN": "REASON", "DECIDE": "REASON",
    "COMMIT": "COMMIT", "ANSWER": "COMMIT", "RETURN": "COMMIT",
    "EXECUTE": "COMMIT", "ACT": "COMMIT", "EMIT": "COMMIT",
}


def _role_for_action(action_type: str) -> str:
    """Return the evidence role the stub executor should emit for
    `action_type`. Falls back to ``GATHER`` so unknown verbs still
    satisfy the G0 evidence-driven invariant.
    """
    upper = action_type.upper()
    if upper in _ACTION_VERB_TO_ROLE:
        return _ACTION_VERB_TO_ROLE[upper]
    for verb, role in _ACTION_VERB_TO_ROLE.items():
        if upper.startswith(verb):
            return role
    return "GATHER"


def make_deterministic_executor(domain: str, *, confidence: float = 0.8) -> HopExecutor:
    """Return a deterministic stub executor tagged for `domain`.

    Day-7d: the stub now emits a *typed* evidence role keyed off the
    hop's `action_type` (`GATHER`/`VERIFY`/`REASON`/`COMMIT`) instead
    of unconditionally `GATHER`. That gives the action-level
    `ReplayValidator(mode="action_level")` walk a meaningful
    role-non-worsening signal even on transfer-target stubs, and
    prevents the deterministic stub from masking real role-regression
    bugs the moment a real executor is plugged in.

    Returns a `dict[str, Any]` per hop with a *split* evidence_in /
    evidence_out signal (`evidence_in` reads the prior `state.evidence`
    union; `evidence_out` is what the stub just emitted). Callers that
    only consume the legacy `evidence` key still see the unified union
    for back-compat.
    """

    def _exec(action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext) -> Dict[str, Any]:
        role = _role_for_action(action_type)
        ev_out = EvidenceRef(
            source=f"{domain}:{action_type.lower()}",
            locator=f"step={ctx.state.inner_step}",
            role=role,
            confidence=confidence,
        )
        # `evidence_in` = the union of roles already on the state at
        # the time of this hop. The stub doesn't re-emit them; it
        # surfaces them so the harness's directional-split fields can
        # populate.
        ev_in: List[EvidenceRef] = list(ctx.state.evidence or [])
        return {
            "ok": True,
            "observation": {
                "domain": domain,
                "echo_action": action_type,
                "echo_payload": payload,
                "role": role,
            },
            # Legacy uni-directional union (back-compat).
            "evidence": [ev_out],
            # Day-8b directional split.
            "evidence_in": ev_in,
            "evidence_out": [ev_out],
        }

    return _exec


class StubTransferTargetAdapter(SkillAdapter):
    """Hop-loop scaffolding shared by all transfer-target adapters.

    Concrete subclasses set `name` and may override `supported_types`
    or `_default_executor()`. Real env binding is plugged in by
    `vlm_wrapper/<domain>_adapter.py` via `set_executor()`.
    """

    name: str = ""
    supported_types: tuple[SkillType, ...] = (
        SkillType.ACTION,
        SkillType.MIXED,
        SkillType.GROUNDING,
        SkillType.REASONING,
    )

    def __init__(self, executor: Optional[HopExecutor] = None) -> None:
        self._executor: HopExecutor = executor or self._default_executor()

    def _default_executor(self) -> HopExecutor:
        return make_deterministic_executor(self.name)

    def set_executor(self, executor: HopExecutor) -> None:
        self._executor = executor

    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool:
        if state.domain != self.name:
            return False
        if not skill.protocol:
            return False
        return skill.skill_type in self.supported_types

    def run(self, skill: SkillRecord, ctx: AdapterRunContext) -> AdapterRunResult:
        bindings = HopBindings(
            bindings=dict(ctx.bindings),
            state_facts=dict(ctx.state.facts),
        )
        budget = BudgetGuard(
            max_hops=int(ctx.budget.get("hops", 8)),
            max_ms=float(ctx.budget.get("ms", 30_000.0)),
        )
        steps: List[Dict[str, Any]] = []
        evidence: List[EvidenceRef] = []
        executor = self._executor if not ctx.dry_run else self._default_executor()

        for i, hop in iter_hops(skill):
            abort = budget.check(i)
            if abort:
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=abort,
                    steps=steps,
                    new_evidence=evidence,
                    cost={
                        "hops": float(i),
                        "ms": (time.time() - budget.started_at) * 1000,
                    },
                )
            action_type = normalize_hop_action(hop)
            payload = bindings.resolve_dict(hop.get("payload", {}))
            try:
                hop_result = executor(action_type, payload, ctx)
            except Exception as exc:  # noqa: BLE001
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=f"adapter_exception: {exc!r}",
                    steps=steps,
                    new_evidence=evidence,
                )
            step_evidence: List[EvidenceRef] = list(hop_result.get("evidence", []))
            evidence.extend(step_evidence)
            steps.append(
                {
                    "action_type": action_type,
                    "payload": payload,
                    "pre_state": ctx.state.to_json() if i == 0 else None,
                    "post_state": None,
                    "evidence": step_evidence,
                    # Day-7d: surface the directional split when the
                    # executor emits it (deterministic stub does so
                    # since Day-7d). The harness's adapter-level
                    # translator copies these into `SkillEpisodeStep`.
                    "evidence_in": list(hop_result.get("evidence_in", [])),
                    "evidence_out": list(hop_result.get("evidence_out", [])),
                    "protocol_index": i,
                    "notes": hop.get("notes", ""),
                }
            )
            ctx.state.inner_step += 1
            if not hop_result.get("ok", True):
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=str(hop_result.get("reason", "hop_failed")),
                    steps=steps,
                    new_evidence=evidence,
                )

        return AdapterRunResult(
            success=True,
            contract_satisfied=True,
            final_state=ctx.state,
            steps=steps,
            new_evidence=evidence,
            score=1.0,
            cost={
                "hops": float(len(steps)),
                "ms": (time.time() - budget.started_at) * 1000,
            },
        )


__all__ = [
    "HopExecutor",
    "StubTransferTargetAdapter",
    "make_deterministic_executor",
]
