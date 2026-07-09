"""`ReplayValidator` — gate Stage 1 (PLAN-UNIFIED-SKILL-GATE §7.1).

Re-runs a SkillRecord on a fixed set of stored `SkillEpisode`s ("replay
seeds") and checks two things:

  (a) the proposed skill produces non-worse outcomes on the seed set;
  (b) it does not introduce evidence-driven invariant violations (G0).

Validator is *deterministic* — it never calls the live env. It executes
adapters in `dry_run=True` mode, which adapters MUST honor by returning
their cached / stored response for the seed state.

**Day-7 (mode="action_level"):** in addition to the adapter-level
"dry_run rerun" the original Phase-A stub does, the validator can now
walk the seed's recorded `steps` and compare the proposed skill's
adapter output **step-by-step**. For each ``(seed.steps[i],
proposed.steps[i])`` pair it diffs:

  * the action_type / action_payload tuple — does the proposal pick
    the same op + slot fills the seed did?
  * the evidence role union — does the proposal still produce the
    expected gather/verify/reason/commit evidence?

The action-level pass criterion is monotonic-non-worse: every seed
step that gathered evidence must still gather *at least* the same
roles, and the action_type sequence must be identical (extra
proposed steps are tolerated; missing seed steps are not). This
matches PLAN-UNIFIED-SKILL-GATE §7.1's "skill produces non-worse
outcomes on the seed set" requirement at a finer granularity than
the adapter-level outcome bool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.enums import GateStage, GateVerdict
from common.state_schema import StateSchema
from data_structure.extensions.gate_verdict import StageVerdict
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.skill_adapter import AdapterRunContext

logger = logging.getLogger("harness.replay_validator")


REPLAY_MODE_ADAPTER = "adapter_level"
REPLAY_MODE_ACTION = "action_level"
_REPLAY_MODES = (REPLAY_MODE_ADAPTER, REPLAY_MODE_ACTION)


@dataclass
class StepDiff:
    """One ``(seed.step, proposed.step)`` comparison record."""

    step_index: int
    seed_action_type: Optional[str]
    proposed_action_type: Optional[str]
    action_match: bool
    seed_evidence_roles: List[str]
    proposed_evidence_roles: List[str]
    evidence_non_worse: bool      # proposed roles ⊇ seed roles
    payload_match: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "seed_action_type": self.seed_action_type,
            "proposed_action_type": self.proposed_action_type,
            "action_match": self.action_match,
            "seed_evidence_roles": list(self.seed_evidence_roles),
            "proposed_evidence_roles": list(self.proposed_evidence_roles),
            "evidence_non_worse": self.evidence_non_worse,
            "payload_match": self.payload_match,
        }


@dataclass
class ReplayResult:
    n_seeds: int
    n_pass: int
    n_fail: int
    n_invariant_violations: int = 0
    per_seed_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    # Day-7: action-level walk diagnostics. Empty when
    # `mode == "adapter_level"` (the default).
    mode: str = REPLAY_MODE_ADAPTER
    n_steps_compared: int = 0
    n_steps_action_match: int = 0
    n_steps_evidence_non_worse: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.n_pass / self.n_seeds) if self.n_seeds else 0.0

    @property
    def step_action_match_rate(self) -> float:
        return (self.n_steps_action_match / self.n_steps_compared) \
            if self.n_steps_compared else 0.0

    @property
    def step_evidence_non_worse_rate(self) -> float:
        return (self.n_steps_evidence_non_worse / self.n_steps_compared) \
            if self.n_steps_compared else 0.0

    def as_stage_verdict(self, *, threshold: float = 0.8) -> StageVerdict:
        verdict = GateVerdict.PASS if self.pass_rate >= threshold and self.n_invariant_violations == 0 else GateVerdict.FAIL
        metrics: Dict[str, float] = {
            "pass_rate": self.pass_rate,
            "n_seeds": float(self.n_seeds),
            "n_invariant_violations": float(self.n_invariant_violations),
        }
        if self.mode == REPLAY_MODE_ACTION:
            metrics["step_action_match_rate"] = self.step_action_match_rate
            metrics["step_evidence_non_worse_rate"] = self.step_evidence_non_worse_rate
            metrics["n_steps_compared"] = float(self.n_steps_compared)
        return StageVerdict(
            stage=GateStage.REPLAY,
            verdict=verdict,
            metrics=metrics,
            failures=[
                f"seed={o['episode_id']} reason={o.get('abort_reason')}"
                for o in self.per_seed_outcomes
                if not o.get("success")
            ],
        )


class ReplayValidator:
    def __init__(self, registry: AdapterRegistry) -> None:
        self._registry = registry

    def validate(
        self,
        *,
        skill: SkillRecord,
        seeds: List[SkillEpisode],
        budget: Optional[Dict[str, float]] = None,
        mode: str = REPLAY_MODE_ADAPTER,
    ) -> ReplayResult:
        if mode not in _REPLAY_MODES:
            raise ValueError(
                f"ReplayValidator.validate(mode={mode!r}) unknown — "
                f"valid modes: {_REPLAY_MODES}"
            )
        n_pass = 0
        n_violations = 0
        n_steps_compared = 0
        n_steps_action_match = 0
        n_steps_evidence_non_worse = 0
        outcomes: List[Dict[str, Any]] = []
        for seed in seeds:
            adapter = self._registry.get(seed.domain, skill.skill_type)
            if adapter is None:
                outcomes.append(
                    {
                        "episode_id": seed.episode_id,
                        "success": False,
                        "abort_reason": f"no_adapter_for({seed.domain},{skill.skill_type.value})",
                    }
                )
                continue
            state = _state_from_seed(seed, fallback_domain=seed.domain)
            ctx = AdapterRunContext(
                state=state,
                parent_run_id=seed.parent_run_id,
                budget=dict(budget or {}),
                dry_run=True,
            )
            try:
                result = adapter.run(skill, ctx)
            except Exception as exc:                        # noqa: BLE001
                outcomes.append(
                    {
                        "episode_id": seed.episode_id,
                        "success": False,
                        "abort_reason": f"adapter_error: {exc!r}",
                    }
                )
                continue
            success = result.success and result.contract_satisfied
            if success and not result.new_evidence and skill.skill_type.value != "action":
                # G0: no evidence on a non-ACTION success.
                n_violations += 1
                success = False
            # ----- Day-7 action-level walk -----
            step_diffs: List[StepDiff] = []
            seed_step_failures: List[str] = []
            if mode == REPLAY_MODE_ACTION:
                seed_steps = list(seed.steps or [])
                proposed_steps = list(result.steps or [])
                # PLAN-UNIFIED §7.1 "non-worse" semantics:
                # the proposal must produce ≥ as many steps as the
                # seed (extra steps tolerated; truncation is a
                # regression). We compare 0..len(seed_steps)-1.
                if len(proposed_steps) < len(seed_steps):
                    seed_step_failures.append(
                        f"step_count_regressed: seed={len(seed_steps)} "
                        f"proposed={len(proposed_steps)}"
                    )
                    success = False
                for i, seed_step in enumerate(seed_steps):
                    n_steps_compared += 1
                    proposed = (
                        proposed_steps[i] if i < len(proposed_steps) else None
                    )
                    diff = _diff_step(i, seed_step, proposed)
                    step_diffs.append(diff)
                    if diff.action_match:
                        n_steps_action_match += 1
                    if diff.evidence_non_worse:
                        n_steps_evidence_non_worse += 1
                    else:
                        seed_step_failures.append(
                            f"step={i}_evidence_regressed: "
                            f"seed_roles={diff.seed_evidence_roles!r} "
                            f"proposed_roles={diff.proposed_evidence_roles!r}"
                        )
                if seed_step_failures:
                    success = False
            # ------------------------------------
            if success:
                n_pass += 1
            outcomes.append(
                {
                    "episode_id": seed.episode_id,
                    "success": success,
                    "abort_reason": result.abort_reason,
                    "score": result.score,
                    **(
                        {
                            "step_diffs": [d.to_json() for d in step_diffs],
                            "step_failures": seed_step_failures,
                        } if mode == REPLAY_MODE_ACTION else {}
                    ),
                }
            )
        return ReplayResult(
            n_seeds=len(seeds),
            n_pass=n_pass,
            n_fail=len(seeds) - n_pass,
            n_invariant_violations=n_violations,
            per_seed_outcomes=outcomes,
            mode=mode,
            n_steps_compared=n_steps_compared,
            n_steps_action_match=n_steps_action_match,
            n_steps_evidence_non_worse=n_steps_evidence_non_worse,
        )


def _diff_step(
    step_index: int,
    seed_step: Any,
    proposed_step: Optional[Any],
) -> StepDiff:
    """Compare one seed step (a `SkillEpisodeStep`) against the
    proposal's adapter output (a raw dict from
    `AdapterRunResult.steps[i]`).

    The proposal's step shape is the same dict shape adapters return:
    ``{"action_type": str, "payload": dict, "evidence": List[EvidenceRef], …}``.
    The seed step is a structured `SkillEpisodeStep` with the same
    fields under different names. Tolerant of either shape on either
    side — the action-level walk has to work both pre- and post- the
    Day-8 `SkillEpisode` field expansion.
    """
    seed_action = _step_get(seed_step, "action_type")
    seed_payload = _step_get(seed_step, "action_payload") or {}
    seed_ev_refs = _step_get(seed_step, "evidence") or []

    if proposed_step is None:
        return StepDiff(
            step_index=step_index,
            seed_action_type=seed_action,
            proposed_action_type=None,
            action_match=False,
            seed_evidence_roles=sorted({_role(e) for e in seed_ev_refs if _role(e)}),
            proposed_evidence_roles=[],
            evidence_non_worse=False,
            payload_match=False,
        )

    proposed_action = _step_get(proposed_step, "action_type")
    proposed_payload = (
        _step_get(proposed_step, "payload")
        or _step_get(proposed_step, "action_payload")
        or {}
    )
    proposed_ev_refs = (
        _step_get(proposed_step, "evidence")
        or _step_get(proposed_step, "new_evidence")
        or []
    )

    seed_roles = sorted({_role(e) for e in seed_ev_refs if _role(e)})
    proposed_roles = sorted({_role(e) for e in proposed_ev_refs if _role(e)})
    return StepDiff(
        step_index=step_index,
        seed_action_type=seed_action,
        proposed_action_type=proposed_action,
        action_match=(seed_action == proposed_action and seed_action is not None),
        seed_evidence_roles=seed_roles,
        proposed_evidence_roles=proposed_roles,
        # Non-worse: every role the seed produced still appears in
        # the proposal. Extra roles are a feature, not a regression.
        evidence_non_worse=set(seed_roles).issubset(set(proposed_roles)),
        payload_match=(dict(seed_payload) == dict(proposed_payload)),
    )


def _step_get(step: Any, key: str) -> Any:
    """Read ``step[key]`` whether `step` is a `SkillEpisodeStep`
    dataclass or a raw dict."""
    if step is None:
        return None
    if isinstance(step, dict):
        return step.get(key)
    return getattr(step, key, None)


def _role(ev: Any) -> Optional[str]:
    """Read ``role`` from either an `EvidenceRef` dataclass or a
    `Mapping`."""
    if ev is None:
        return None
    if isinstance(ev, dict):
        return ev.get("role")
    return getattr(ev, "role", None)


def _state_from_seed(seed: SkillEpisode, *, fallback_domain: str) -> StateSchema:
    """Reconstruct a minimal StateSchema from a seed episode's stored
    initial state."""
    init = seed.initial_state or {}
    return StateSchema(
        task=str(init.get("task", "")),
        domain=str(init.get("domain", fallback_domain)),
        elements=list(init.get("elements", [])),
        facts=dict(init.get("facts", {})),
        open_questions=list(init.get("open_questions", [])),
        inner_step=int(init.get("inner_step", 0)),
        outer_step=int(init.get("outer_step", 0)),
        extra=dict(init.get("extra", {})),
    )


__all__ = [
    "REPLAY_MODE_ACTION",
    "REPLAY_MODE_ADAPTER",
    "ReplayResult",
    "ReplayValidator",
    "StepDiff",
]
