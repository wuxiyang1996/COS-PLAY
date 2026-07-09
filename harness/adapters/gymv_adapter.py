"""`gymv` adapter — game-env execution path.

This is a *minimal* but real adapter: it walks `skill.protocol` hop by
hop, treats each hop as `{action: <tool_name>, payload: {...}}`, and
emits a corresponding `SkillEpisodeStep`. In `dry_run=True` mode it
short-circuits to a deterministic outcome derived from the seed
state's stored facts (used by the gate's replay validator).

Real env wiring (gymnasium / gym-v) lives in `harness.gymv_executor`
(`make_gymv_executor`); the executor is plugged in via `set_executor`
so the harness package stays importable in a unit-test environment
without dragging in `env_wrappers` / `gym_v`.

Day-3 (PLAN-HARNESS §22): the adapter additionally captures per-hop
pre/post `StateSchema` snapshots (when the executor surfaces them via
`hop_result["post_state"]`) and rolls up the protocol's typed
`effects_add` predicates against those snapshots. The roll-up is
attached to `AdapterRunResult.extra["per_hop_effects"]` so the
`FewShotAdapter`'s `success_fn` can score on real effect satisfaction
rather than on "ran without raising".
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from common.enums import SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapters._common import BudgetGuard, HopBindings, iter_hops, normalize_hop_action
from harness.skill_adapter import AdapterRunContext, AdapterRunResult, SkillAdapter


# Pluggable executor: a callable `(action_type, payload, ctx) -> dict`.
# The default executor is a deterministic stub useful for tests and
# the gate's dry-run path. Real env binding (gymnasium step()) is plugged
# in by `gymv_wrapper.adapter` at runtime via `set_executor`.
HopExecutor = Callable[[str, Dict[str, Any], AdapterRunContext], Dict[str, Any]]


def _deterministic_executor(
    action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext
) -> Dict[str, Any]:
    facts = ctx.state.facts
    return {
        "ok": True,
        "observation": {
            "echo_action": action_type,
            "echo_payload": payload,
            "facts_seen": list(facts.keys()),
        },
        "evidence": [
            EvidenceRef(
                source=f"gymv:{action_type.lower()}",
                locator=f"step={ctx.state.inner_step}",
                role="GATHER",
                confidence=0.9,
            )
        ],
    }


class GymvAdapter(SkillAdapter):
    name = "gymv"
    supported_types = (SkillType.ACTION, SkillType.MIXED, SkillType.GROUNDING)

    def __init__(self, executor: Optional[HopExecutor] = None) -> None:
        self._executor: HopExecutor = executor or _deterministic_executor

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
        executor = self._executor if not ctx.dry_run else _deterministic_executor

        # Day-3 §22: every hop records its pre/post StateSchema so the
        # success_fn can evaluate `effects_add` predicates against
        # consecutive snapshots. The pre-state for hop 0 is `ctx.state`;
        # subsequent pre-states chain off the previous post.
        prev_post_serialized: Optional[Dict[str, Any]] = None

        for i, hop in iter_hops(skill):
            abort = budget.check(i)
            if abort:
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=abort,
                    steps=steps,
                    new_evidence=evidence,
                    cost={"hops": float(i), "ms": (time.time() - budget.started_at) * 1000},
                )
            action_type = normalize_hop_action(hop)
            payload = bindings.resolve_dict(hop.get("payload", {}))
            pre_state_serialized = (
                prev_post_serialized if prev_post_serialized is not None
                else ctx.state.to_json()
            )
            try:
                hop_result = executor(action_type, payload, ctx)
            except Exception as exc:                          # noqa: BLE001
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=f"adapter_exception: {exc!r}",
                    steps=steps,
                    new_evidence=evidence,
                )
            step_evidence: List[EvidenceRef] = list(hop_result.get("evidence", []))
            evidence.extend(step_evidence)

            # Executor may surface a post-state dict via
            # `hop_result["post_state"]` (the gymv executor in
            # `harness.gymv_executor` always does). Fallback to a
            # facts-only echo so the success_fn at least sees the
            # cumulative reward / terminal flag.
            post_state_serialized = hop_result.get("post_state")
            if post_state_serialized is None:
                obs = hop_result.get("observation") or {}
                post_state_serialized = dict(pre_state_serialized)
                post_state_serialized["facts"] = {
                    **(post_state_serialized.get("facts") or {}),
                    "last_observation": {
                        k: v for k, v in obs.items()
                        if isinstance(v, (str, int, float, bool, type(None)))
                    },
                }

            steps.append(
                {
                    "action_type": action_type,
                    "payload": payload,
                    "pre_state": pre_state_serialized,
                    "post_state": post_state_serialized,
                    "evidence": step_evidence,
                    "notes": hop.get("notes", ""),
                }
            )
            prev_post_serialized = post_state_serialized
            ctx.state.inner_step += 1
            if not hop_result.get("ok", True):
                return AdapterRunResult(
                    success=False,
                    contract_satisfied=False,
                    abort_reason=str(hop_result.get("reason", "hop_failed")),
                    steps=steps,
                    new_evidence=evidence,
                )

        # ── Day-3 §22 effect-predicate roll-up ─────────────────────────
        per_hop_effects = self._evaluate_effects(skill, steps)

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
            extra={"per_hop_effects": per_hop_effects} if per_hop_effects else {},
        )

    @staticmethod
    def _evaluate_effects(
        skill: SkillRecord, steps: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Roll up per-hop `effects_add` against the recorded pre/post
        snapshots. Returns None when the protocol carries no typed
        effects (pre-lift skills) so the AdapterRunResult.extra stays
        clean for callers that don't care about the gymv-shape contract.
        """

        protocol = list(getattr(skill, "protocol", None) or [])
        has_any_effects = any(
            isinstance(h, dict) and h.get("effects_add") for h in protocol
        )
        if not has_any_effects:
            return None
        # Local import — `harness.gymv_success` lives in this package and
        # only depends on `common.state_schema`, so the cost is just the
        # one-time module load.
        from harness.gymv_success import (
            evaluate_hop_effects,
            _hydrate_state,
        )

        per_hop: List[Dict[str, Any]] = []
        n_total = 0
        n_pass = 0
        for i, step in enumerate(steps):
            if i >= len(protocol):
                break
            hop = protocol[i] if isinstance(protocol[i], dict) else {}
            if not hop.get("effects_add"):
                continue
            pre = step.get("pre_state")
            post = step.get("post_state")
            if pre is None or post is None:
                continue
            res = evaluate_hop_effects(
                {**hop, "hop_index": i},
                _hydrate_state(pre),
                _hydrate_state(post),
            )
            per_hop.append(res.to_json())
            n_total += 1
            if res.passed:
                n_pass += 1

        if n_total == 0:
            return None
        return {
            "n_hops_evaluated": n_total,
            "n_hops_passed": n_pass,
            "pass_rate": n_pass / n_total,
            "per_hop": per_hop,
        }


__all__ = ["GymvAdapter", "HopExecutor"]
