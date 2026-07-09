"""`SkillHarness` — the actual runtime.

Public API (PLAN-HARNESS §5.2 & §8 unified skill interface):

    h = SkillHarness(registry, reward_logger=..., config=...)

    # 1. Narrow the bank-supplied candidate list to runnable skills.
    eligible = h.select_eligible_skills(candidates, state, skill_type_hint=...)

    # 2. Day-8: second-pass invocation validation (PLAN-UNIFIED §3.4)
    invoc = h.validate_invocation(skill, state, bindings=…)
    if not invoc.ok:
        # Veto with structured reason — actor can pick another skill.
        ...

    # 3. Execute one chosen skill (chosen by the Actor, not by us).
    episode = h.run_skill(skill, state, parent_run_id=..., bindings=...)

    # 4. Replay-validate a proposed skill against stored seeds (gate Stage 1).
    result = h.replay_validate(skill, seeds=...)

The harness writes nothing to the bank. It writes SkillEpisodes to the
RewardLogger (which is also the orchestrator's artifact source) and
returns failure traces when execution aborts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from common.enums import SkillStatus, SkillType
from common.ids import new_episode_id
from common.state_schema import StateSchema
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_episode import (
    SkillEpisode,
    SkillEpisodeOutcome,
    SkillEpisodeStep,
)
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.eligibility import EligibilityFilter, EligibleSkill
from harness.replay_validator import ReplayResult, ReplayValidator
from harness.reward_logger import RewardLogger
from harness.skill_adapter import AdapterRunContext, AdapterRunResult


@dataclass
class HarnessConfig:
    allow_shadow: bool = True
    default_budget_tokens: float = 4096.0
    default_budget_hops: float = 8.0
    default_budget_ms: float = 30_000.0
    fail_on_missing_adapter: bool = True
    seed: Optional[int] = None


@dataclass
class ValidateInvocationResult:
    """Day-8: structured second-pass veto result.

    Closes harness/README §9. The eligibility filter narrows the
    candidate list (cheap predicate filters); `validate_invocation`
    runs *after* the actor has picked one, on the actually-bound
    invocation, and produces a structured pass/veto verdict the actor
    can route back into its own decision log.

    Per-check booleans:
      * ``adapter_ok``      — registered adapter exists
      * ``binding_ok``      — every ``${slot}`` in skill.protocol is filled
      * ``precondition_ok`` — skill.contract.preconditions all hold
      * ``evidence_ok``     — required ``evidence_in`` references are present
      * ``shadow_only``     — propagated from the eligibility filter

    The veto reason channel is the union of *which* checks failed,
    formatted so the actor can render it in its veto log without
    string-parsing the per-check fields.
    """

    ok: bool
    skill_id: str
    adapter_name: Optional[str]
    adapter_ok: bool
    binding_ok: bool
    precondition_ok: bool
    evidence_ok: bool
    shadow_only: bool = False
    veto_reasons: List[str] = field(default_factory=list)
    missing_bindings: List[str] = field(default_factory=list)
    missing_evidence_in: List[str] = field(default_factory=list)
    failed_preconditions: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "skill_id": self.skill_id,
            "adapter_name": self.adapter_name,
            "adapter_ok": self.adapter_ok,
            "binding_ok": self.binding_ok,
            "precondition_ok": self.precondition_ok,
            "evidence_ok": self.evidence_ok,
            "shadow_only": self.shadow_only,
            "veto_reasons": list(self.veto_reasons),
            "missing_bindings": list(self.missing_bindings),
            "missing_evidence_in": list(self.missing_evidence_in),
            "failed_preconditions": list(self.failed_preconditions),
        }


_RX_SLOT = re.compile(r"\$\{([a-zA-Z_][\w]*)\}")


class SkillHarness:
    """Per-invocation runtime for skill execution and verification."""

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        reward_logger: Optional[RewardLogger] = None,
        config: Optional[HarnessConfig] = None,
    ) -> None:
        self._registry = registry
        self._logger = reward_logger or RewardLogger()
        self._config = config or HarnessConfig()
        self._eligibility = EligibilityFilter(registry, allow_shadow=self._config.allow_shadow)
        self._replay = ReplayValidator(registry)
        self._failure_sink: List[FailureTrace] = []

    # ---------------------------------------------------------------- API

    def select_eligible_skills(
        self,
        candidates: Iterable[SkillRecord],
        state: StateSchema,
        *,
        skill_type_hint: Optional[SkillType] = None,
    ) -> List[EligibleSkill]:
        return self._eligibility.filter(candidates, state, skill_type_hint=skill_type_hint)

    def validate_invocation(
        self,
        skill: SkillRecord,
        state: StateSchema,
        *,
        bindings: Optional[Dict[str, Any]] = None,
        eligible: Optional[EligibleSkill] = None,
    ) -> ValidateInvocationResult:
        """Day-8: structured second-pass invocation veto (PLAN-UNIFIED §3.4).

        Runs *after* the actor has picked a skill from the eligibility
        filter and bound its slots, but *before* `run_skill` mutates
        state. Returns a `ValidateInvocationResult` whose per-check
        booleans + veto_reasons let the actor decide whether to fall
        through to a different skill or to surface the veto.

        Three checks are wired today; a fourth (numeric `fit_score` /
        `risk_score`) is roadmapped for Day-10+ once the LoRA scoring
        head lands.

        Pass `eligible=…` (the corresponding `EligibleSkill` from
        `select_eligible_skills`) so the result can propagate
        `shadow_only` faithfully into downstream `SkillEpisode`
        artifacts.
        """
        bindings = dict(bindings or {})
        veto_reasons: List[str] = []

        adapter = self._registry.get(state.domain, skill.skill_type)
        adapter_ok = adapter is not None
        if not adapter_ok:
            veto_reasons.append(
                f"missing_adapter({state.domain},{skill.skill_type.value})"
            )

        missing_bindings = _check_protocol_slots(skill.protocol, bindings)
        binding_ok = not missing_bindings
        if not binding_ok:
            veto_reasons.append(
                f"missing_bindings={missing_bindings}"
            )

        failed_preconditions = _check_preconditions(skill, state, bindings)
        precondition_ok = not failed_preconditions
        if not precondition_ok:
            veto_reasons.append(
                f"preconditions_violated={failed_preconditions}"
            )

        missing_evidence_in = _check_evidence_in(skill, state)
        evidence_ok = not missing_evidence_in
        if not evidence_ok:
            veto_reasons.append(
                f"missing_evidence_in={missing_evidence_in}"
            )

        shadow_only = bool(eligible.shadow_only) if eligible is not None else False
        ok = adapter_ok and binding_ok and precondition_ok and evidence_ok

        return ValidateInvocationResult(
            ok=ok,
            skill_id=skill.skill_id,
            adapter_name=adapter.name if adapter is not None else None,
            adapter_ok=adapter_ok,
            binding_ok=binding_ok,
            precondition_ok=precondition_ok,
            evidence_ok=evidence_ok,
            shadow_only=shadow_only,
            veto_reasons=veto_reasons,
            missing_bindings=missing_bindings,
            missing_evidence_in=missing_evidence_in,
            failed_preconditions=failed_preconditions,
        )

    def run_skill(
        self,
        skill: SkillRecord,
        state: StateSchema,
        *,
        parent_run_id: Optional[str],
        bindings: Optional[Dict[str, Any]] = None,
        budget: Optional[Dict[str, float]] = None,
        parent_episode_id: Optional[str] = None,
        eligible: Optional[EligibleSkill] = None,
    ) -> SkillEpisode:
        adapter = self._registry.get(state.domain, skill.skill_type)
        episode = SkillEpisode.begin(
            skill_id=skill.skill_id,
            skill_version=skill.version,
            skill_type=skill.skill_type,
            domain=state.domain,
            parent_run_id=parent_run_id,
            initial_state=state,
            parent_episode_id=parent_episode_id,
        )
        # Day-8: propagate shadow_only into the episode artifact so
        # Stage-2 readers can distinguish shadow vs. real failures.
        # Defaults to skill.status==SHADOW when no `eligible` is
        # supplied — keeps prior callers' SkillEpisodes correctly
        # tagged.
        episode.shadow = (
            bool(eligible.shadow_only) if eligible is not None
            else (skill.status == SkillStatus.SHADOW)
        )
        episode.started_at = time.time()

        if adapter is None:
            outcome = SkillEpisodeOutcome(
                success=False,
                contract_satisfied=False,
                abort_reason=f"no_adapter_for({state.domain},{skill.skill_type.value})",
            )
            episode.finalize(outcome=outcome, final_state=state)
            episode.finished_at = time.time()
            self._logger.log_episode(episode)
            self._record_failure(episode, "MISSING_ADAPTER")
            if self._config.fail_on_missing_adapter:
                return episode
            return episode

        ctx = AdapterRunContext(
            state=state,
            bindings=dict(bindings or {}),
            parent_run_id=parent_run_id,
            parent_episode_id=parent_episode_id,
            budget=self._effective_budget(budget),
            seed=self._config.seed,
            dry_run=False,
        )
        try:
            result = adapter.run(skill, ctx)
        except Exception as exc:                          # noqa: BLE001
            outcome = SkillEpisodeOutcome(
                success=False,
                contract_satisfied=False,
                abort_reason=f"adapter_exception: {exc!r}",
            )
            episode.finalize(outcome=outcome, final_state=state)
            episode.finished_at = time.time()
            self._logger.log_episode(episode)
            self._record_failure(episode, "ADAPTER_EXCEPTION")
            return episode

        # Translate adapter steps -> SkillEpisodeSteps (lossy but
        # invariant-preserving). Day-7d: surface the
        # `evidence_in / evidence_out / protocol_index` fields the
        # adapter shim now emits.
        for i, raw in enumerate(result.steps):
            step = SkillEpisodeStep(
                step_index=i,
                action_type=str(raw.get("action_type", "STEP")),
                action_payload=dict(raw.get("payload", {})),
                pre_state=raw.get("pre_state"),
                post_state=raw.get("post_state"),
                evidence=list(raw.get("evidence", [])),
                evidence_in=list(raw.get("evidence_in", [])),
                evidence_out=list(raw.get("evidence_out", [])),
                protocol_index=raw.get("protocol_index"),
                notes=str(raw.get("notes", "")),
            )
            episode.add_step(step)

        evidence_role = sorted({e.role for e in result.new_evidence}) or [
            r for r in skill.contract.expected_evidence_roles if r
        ]
        outcome = SkillEpisodeOutcome(
            success=bool(result.success),
            contract_satisfied=bool(result.contract_satisfied),
            abort_reason=result.abort_reason,
            evidence_role=evidence_role,
            answer=result.answer,
            score=result.score,
            extra=dict(result.extra),
        )
        episode.cost.update(result.cost)
        episode.transfer_label = result.diagnostic_label
        try:
            episode.finalize(outcome=outcome, final_state=result.final_state or state)
        except ValueError as exc:
            # G0 violation — record as failure & flip success false rather
            # than corrupt the episode.
            outcome.success = False
            outcome.contract_satisfied = False
            outcome.abort_reason = (outcome.abort_reason or "") + f"|invariant:{exc}"
            episode.finalize(outcome=outcome, final_state=result.final_state or state)
        episode.finished_at = time.time()
        self._logger.log_episode(episode)
        if not outcome.success:
            self._record_failure(episode, _classify_abort(outcome.abort_reason))
        return episode

    def replay_validate(
        self,
        skill: SkillRecord,
        seeds: List[SkillEpisode],
        *,
        budget: Optional[Dict[str, float]] = None,
    ) -> ReplayResult:
        return self._replay.validate(skill=skill, seeds=seeds, budget=budget)

    # -- accessors ---------------------------------------------------------

    @property
    def reward_logger(self) -> RewardLogger:
        return self._logger

    @property
    def adapter_registry(self) -> AdapterRegistry:
        return self._registry

    def drain_failures(self) -> List[FailureTrace]:
        out, self._failure_sink = self._failure_sink, []
        return out

    # -- helpers -----------------------------------------------------------

    def _effective_budget(self, override: Optional[Dict[str, float]]) -> Dict[str, float]:
        budget = {
            "tokens": self._config.default_budget_tokens,
            "hops": self._config.default_budget_hops,
            "ms": self._config.default_budget_ms,
        }
        if override:
            budget.update(override)
        return budget

    def _record_failure(self, episode: SkillEpisode, failure_class: str) -> None:
        last_step = episode.steps[-1] if episode.steps else None
        trace = FailureTrace(
            skill_id=episode.skill_id,
            skill_episode_id=episode.episode_id,
            domain=episode.domain,
            failed_step_index=last_step.step_index if last_step else None,
            failure_class=failure_class,
            abort_reason=episode.outcome.abort_reason if episode.outcome else None,
            pre_state=episode.initial_state,
            failed_step=last_step.to_json() if last_step else None,
            observed_evidence_roles=(
                list(episode.outcome.evidence_role) if episode.outcome else []
            ),
        )
        self._failure_sink.append(trace)


def _check_protocol_slots(
    protocol: List[Dict[str, Any]],
    bindings: Dict[str, Any],
) -> List[str]:
    """Return the sorted list of `${slot}` placeholders that appear in
    `protocol` but are not filled by `bindings`.

    The protocol's slot syntax is ``${name}`` (matching
    `harness/adapters/_common.py::resolve_slot`). We scan every hop's
    string-valued payload entries and collect every unique slot name.
    Bindings are *case-sensitive* and must exactly match the slot
    name; alternative-name aliasing is the actor's responsibility.
    """
    found: set = set()
    for hop in protocol or []:
        if not isinstance(hop, dict):
            continue
        payload = hop.get("payload") or hop.get("args") or {}
        for v in (payload or {}).values():
            if isinstance(v, str):
                for m in _RX_SLOT.finditer(v):
                    found.add(m.group(1))
    missing = sorted(s for s in found if s not in bindings)
    return missing


def _check_preconditions(
    skill: SkillRecord,
    state: StateSchema,
    bindings: Dict[str, Any],
) -> List[str]:
    """Return the sorted list of `skill.contract.preconditions` strings
    we *cannot prove* hold against `state` + `bindings`.

    Preconditions are free-form strings in the cold-start corpus
    (e.g. ``"a slidable direction exists"``); we don't have a logic
    checker for them yet. The Day-8 stub does the syntactic check
    every PLAN-UNIFIED §3.4 implementation must do as a baseline:

      1. Slot-references in preconditions (``${X}``) must be in
         bindings.
      2. The empty-string precondition is silently ignored.

    This is the conservative-pass policy — when we *can't* check, we
    pass. The formal-precondition checker is Day-9+ work and lands
    when the lift starts emitting typed predicate ASTs (the same lift
    that emits typed effects today).
    """
    failures: List[str] = []
    for raw in skill.contract.preconditions or []:
        cond = (raw or "").strip()
        if not cond:
            continue
        for m in _RX_SLOT.finditer(cond):
            if m.group(1) not in bindings:
                failures.append(f"unbound_slot_in_precondition:{m.group(1)}")
    return sorted(set(failures))


def _check_evidence_in(skill: SkillRecord, state: StateSchema) -> List[str]:
    """Return the sorted list of evidence roles the contract advertises
    as ``expected_evidence_roles`` but the current `state.evidence`
    doesn't surface.

    PLAN-HARNESS §10's spec calls this `evidence_in` (vs. `evidence_out`
    which the episode adds). For the Day-8 baseline we read the contract's
    `expected_evidence_roles` as the input requirement — the lift will
    eventually split these into in/out roles, at which point this check
    tightens to the input subset only.

    REASONING / GROUNDING / MIXED skills *always* require non-empty
    evidence by G0; ACTION skills are exempt — they consume world-state
    rather than evidence. We honour that here so the Day-8 veto doesn't
    block the gymv ACTION path.
    """
    if skill.skill_type == SkillType.ACTION:
        return []
    required = list(skill.contract.expected_evidence_roles or [])
    if not required:
        return []
    present_roles = {ev.role for ev in (state.evidence or []) if getattr(ev, "role", None)}
    missing = sorted(set(required) - present_roles)
    return missing


def _classify_abort(reason: Optional[str]) -> str:
    if not reason:
        return "UNKNOWN"
    r = reason.lower()
    if "precondition" in r:
        return "PRECONDITION_VIOLATION"
    if "invariant" in r:
        return "INVARIANT_VIOLATION"
    if "budget" in r:
        return "BUDGET_EXCEEDED"
    if "no_adapter" in r:
        return "MISSING_ADAPTER"
    if "exception" in r:
        return "ADAPTER_EXCEPTION"
    return "OTHER"


__all__ = ["HarnessConfig", "SkillHarness", "ValidateInvocationResult"]
