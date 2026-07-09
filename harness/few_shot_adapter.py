"""`FewShotAdapter` — K-shot adaptation gate (Stage 3a).

Spec: PLAN-UNIFIED-SKILL-GATE §7 Stage 3a, PLAN-HARNESS §5.4
(`FewShotAdapter`), PLAN-SKILL-BANK §0.4 source/target asymmetry.

This is the runtime that actually *checks transferability*: given a
skill that was learned in the game foundry, it tries to bind that
skill's abstract protocol to a target domain using a small budget of
demonstration episodes (the "shots") plus the target adapter
registered for the candidate domain.

The adapter does **not** mutate the bank; it returns an
`AdaptResult` for each `(skill, target_domain)` pair. The caller
(`GateService._run_transfer`) is responsible for translating the
verdicts into:

  * Stage 3a `StageVerdict` metrics, and
  * (on PASS / LIMITED_PASS) appending the target domain to the
    candidate `SkillRecord.verified_domains` so downstream lifecycle
    invariants can let the skill reach ACTIVE.

The default per-shot success criterion is "the harness ran the skill
to completion via the target-domain adapter without aborting on a
contract violation". Real binding probes (with concrete demos and
golden answers) plug their own scorer in via `success_fn`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from common.enums import SOURCE_DOMAINS, TRANSFER_TARGET_DOMAINS, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_record import SkillRecord
from harness.skill_harness import SkillHarness


# A `shot` is a tiny demonstration: a starting state and (optionally)
# a per-shot bindings dict that pre-fills typed slots from the demo's
# golden trace. The success_fn maps a finished SkillEpisode + the
# original shot to a 0/1 success score (or any float ≥ 0).
@dataclass
class FewShotDemo:
    state: StateSchema
    bindings: Dict[str, Any] = field(default_factory=dict)
    expected: Optional[Dict[str, Any]] = None  # golden answer / labels
    notes: str = ""


SuccessFn = Callable[[SkillEpisode, FewShotDemo], float]


@dataclass
class AdaptResult:
    """Per `(skill, target_domain[, target_task])` outcome of the few-shot
    adapter.

    `target_task` is the intra-domain task axis (harness/README §22). When
    set, the synthesized demo state and `_coerce_state_to_target` propagate
    it into `state.task` so the eligibility filter and adapter dispatch see
    the right task; the field is also recorded here for downstream lifecycle
    bookkeeping (`SkillRecord.verified_tasks` append on PASS / LIMITED_PASS).
    """

    skill_id: str
    target_domain: str
    k_used: int
    pass_rate: float
    n_success: int
    n_total: int
    aborted: int = 0
    cost_tokens: float = 0.0
    cost_ms: float = 0.0
    diagnostic_label: Optional[str] = None
    episode_ids: List[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    target_task: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.n_total > 0 and self.pass_rate >= 0.5

    def to_json(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "target_domain": self.target_domain,
            "target_task": self.target_task,
            "k_used": self.k_used,
            "pass_rate": self.pass_rate,
            "n_success": self.n_success,
            "n_total": self.n_total,
            "aborted": self.aborted,
            "cost_tokens": self.cost_tokens,
            "cost_ms": self.cost_ms,
            "diagnostic_label": self.diagnostic_label,
            "episode_ids": list(self.episode_ids),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def default_success_fn(episode: SkillEpisode, demo: FewShotDemo) -> float:
    """Default scorer: success ⇔ episode succeeded AND contract held.

    Real callers (the orchestrator's transfer-eval driver) should
    replace this with a domain-aware scorer that compares
    `episode.outcome.answer` to `demo.expected`.
    """

    out = episode.outcome
    if out is None:
        return 0.0
    return 1.0 if (out.success and out.contract_satisfied) else 0.0


class FewShotAdapterError(RuntimeError):
    """Raised on configuration errors that must abort the gate stage."""


class FewShotAdapter:
    """Few-shot adapter — Stage 3a of the unified skill gate.

    The adapter is *stateless* across calls: it never writes to the
    bank, never mutates `SkillRecord`, and never logs to the long-term
    artifact store directly. All effects are in the returned
    `AdaptResult`.
    """

    def __init__(
        self,
        *,
        harness: SkillHarness,
        success_fn: SuccessFn = default_success_fn,
        k_shot_default: int = 5,
        k_shot_max: int = 16,
        target_domain_pass_rate_min: float = 0.5,
        adaptation_cost_max_tokens: int = 8_000,
    ) -> None:
        self._harness = harness
        self._success_fn = success_fn
        self._k_shot_default = k_shot_default
        self._k_shot_max = k_shot_max
        self._pass_rate_min = target_domain_pass_rate_min
        self._max_tokens = adaptation_cost_max_tokens

    # --- public API ------------------------------------------------------

    def adapt(
        self,
        *,
        skill: SkillRecord,
        target_domain: str,
        demos: Sequence[FewShotDemo] = (),
        k: Optional[int] = None,
        target_task: Optional[str] = None,
    ) -> AdaptResult:
        """Try to bind `skill` to `(target_domain[, target_task])` using
        up to `k` shots.

        `target_task` (harness/README §22) is the intra-domain task axis.
        When set, the synthesized demo state and `_coerce_state_to_target`
        retag `state.task` so the harness's eligibility filter sees the
        right task; the value is also propagated into `AdaptResult` so
        the lifecycle path can append it to `verified_tasks` on success.
        Pass `None` to fall back to pre-task-axis behaviour.

        Day-5: when ``target_task`` is set AND ``target_domain`` is in
        ``SOURCE_DOMAINS`` (e.g. ``gymv``), this is an **intra-domain
        task transfer** — the milestone PLAN-HARNESS §22 was scoped
        for. The cross-domain validation (which would reject ``gymv``
        as a target) is relaxed in that case.

        Returns an `AdaptResult` in all cases — even when the
        configuration prevents real execution (e.g. no adapter
        registered for `target_domain`). The diagnostic_label encodes
        why a binding failed so the orchestrator can label
        `false_binding_patterns` (PLAN-SKILL-BANK §4.3b).
        """

        self._validate(
            skill=skill,
            target_domain=target_domain,
            target_task=target_task,
        )

        k_use = self._clip_k(k)
        started = time.time()

        registry = self._harness.adapter_registry
        if registry.get(target_domain, skill.skill_type) is None:
            return AdaptResult(
                skill_id=skill.skill_id,
                target_domain=target_domain,
                target_task=target_task,
                k_used=0,
                pass_rate=0.0,
                n_success=0,
                n_total=0,
                aborted=0,
                diagnostic_label="target_domain_demo_unavailable",
                started_at=started,
                finished_at=time.time(),
            )

        if not demos:
            # No demonstrations available → produce a synthetic empty
            # state so we still exercise the adapter. The diagnostic
            # makes it clear why the verdict is LIMITED. When
            # `target_task` is set, tag the synthetic state with it so
            # an F2′-aware filter sees the right task.
            synth_task = (
                f"few_shot_probe/{target_task}"
                if target_task
                else f"few_shot_probe:{skill.name}"
            )
            demos = [
                FewShotDemo(
                    state=StateSchema(
                        task=synth_task,
                        domain=target_domain,
                    )
                )
            ]
            synthetic = True
        else:
            synthetic = False

        used = list(demos)[:k_use]
        n_total = len(used)
        n_success = 0
        aborted = 0
        episode_ids: List[str] = []
        cost_tokens = 0.0
        cost_ms = 0.0
        diagnostic: Optional[str] = None
        if synthetic:
            diagnostic = "target_domain_demo_unavailable"

        # Day-6: when the orchestrator left ``self._success_fn`` at the
        # default (`default_success_fn`) AND a domain-specific scorer
        # is registered for `target_domain`, use the registered one.
        # Explicit overrides still win — passing a custom scorer at
        # FewShotAdapter construction time bypasses the registry.
        scorer = self._success_fn
        if scorer is default_success_fn:
            from harness.gymv_success import success_fn_for_domain
            scorer = success_fn_for_domain(target_domain, fallback=scorer)

        for shot in used:
            shot_state = self._coerce_state_to_target(
                shot.state, target_domain, target_task=target_task
            )
            episode = self._harness.run_skill(
                skill,
                shot_state,
                parent_run_id=None,
                bindings=dict(shot.bindings),
            )
            episode_ids.append(episode.episode_id)
            cost_tokens += float(episode.cost.get("tokens", 0.0))
            cost_ms += float(episode.cost.get("ms", 0.0))
            score = scorer(episode, shot)
            if score >= 1.0:
                n_success += 1
            elif episode.outcome is None or not episode.outcome.success:
                aborted += 1
            if cost_tokens > self._max_tokens:
                diagnostic = "few_shot_budget_exceeded"
                break

        pass_rate = (n_success / n_total) if n_total else 0.0
        if not synthetic and n_total > 0 and pass_rate < self._pass_rate_min:
            diagnostic = diagnostic or "adaptation_overfitting"

        return AdaptResult(
            skill_id=skill.skill_id,
            target_domain=target_domain,
            target_task=target_task,
            k_used=n_total,
            pass_rate=pass_rate,
            n_success=n_success,
            n_total=n_total,
            aborted=aborted,
            cost_tokens=cost_tokens,
            cost_ms=cost_ms,
            diagnostic_label=diagnostic,
            episode_ids=episode_ids,
            started_at=started,
            finished_at=time.time(),
        )

    def adapt_many(
        self,
        *,
        skill: SkillRecord,
        target_domains: Sequence[str],
        demos_by_domain: Optional[Dict[str, Sequence[FewShotDemo]]] = None,
        k: Optional[int] = None,
        target_task_by_domain: Optional[Dict[str, str]] = None,
    ) -> List[AdaptResult]:
        results: List[AdaptResult] = []
        for d in target_domains:
            shots = (demos_by_domain or {}).get(d, ())
            t_task = (target_task_by_domain or {}).get(d)
            results.append(
                self.adapt(
                    skill=skill, target_domain=d, demos=shots, k=k, target_task=t_task
                )
            )
        return results

    @property
    def pass_rate_min(self) -> float:
        return self._pass_rate_min

    # --- internals -------------------------------------------------------

    def _clip_k(self, k: Optional[int]) -> int:
        if k is None:
            return self._k_shot_default
        if k < 1:
            raise FewShotAdapterError("k must be ≥ 1")
        return min(k, self._k_shot_max)

    def _validate(
        self,
        *,
        skill: SkillRecord,
        target_domain: str,
        target_task: Optional[str] = None,
    ) -> None:
        # Day-5: intra-source-domain task transfer (e.g. gymv→gymv with
        # target_task="tetris") is allowed when target_task is set and
        # target_domain is a source domain. This is the PLAN-HARNESS
        # §22 task-axis milestone — the cross-domain transfer set is a
        # separate concern that doesn't apply here.
        intra_domain_task_transfer = (
            target_task is not None
            and target_domain in SOURCE_DOMAINS
        )
        if not intra_domain_task_transfer and target_domain not in TRANSFER_TARGET_DOMAINS:
            raise FewShotAdapterError(
                f"target_domain={target_domain!r} is not in "
                f"TRANSFER_TARGET_DOMAINS={TRANSFER_TARGET_DOMAINS} "
                f"(intra-domain task transfer requires target_task)"
            )
        if skill.source_domains and not any(
            d in SOURCE_DOMAINS for d in skill.source_domains
        ):
            # The asymmetric thesis: only skills with a source-domain
            # (game) lineage are eligible for few-shot transfer.
            raise FewShotAdapterError(
                f"skill {skill.skill_id!r} has no source-domain lineage; "
                f"source_domains={skill.source_domains}"
            )
        # SkillType must be supported by the target adapter — we'll
        # detect missing registration in `adapt()` itself.
        _ = skill.skill_type  # silence linter

    def _coerce_state_to_target(
        self,
        state: StateSchema,
        target_domain: str,
        *,
        target_task: Optional[str] = None,
    ) -> StateSchema:
        """Return a copy of `state` whose `.domain` matches `target_domain`
        and (optionally) whose `.task` matches `target_task`.

        Demonstrations may carry their own domain / task tag (e.g. when
        reused from prior bank evidence); we re-tag them so the harness's
        adapter dispatch goes through the *target* adapter and so the
        F2′ task-axis filter sees the intended task identifier.
        """

        if state.domain == target_domain and (
            target_task is None or state.task == target_task
        ):
            return state
        # Use the bare task token unmodified — `task_id_from_state`
        # extracts the trailing path-segment, so a bare `"tetris"` and
        # a fully-qualified `"make_gaming_env/tetris"` both project to
        # `"tetris"`. We prefer the bare form for synthesised states.
        new_task = state.task if target_task is None else target_task
        return StateSchema(
            task=new_task,
            domain=target_domain,
            targets=state.targets,
            elements=list(state.elements),
            facts=dict(state.facts),
            open_questions=list(state.open_questions),
            evidence=list(state.evidence),
            inner_step=state.inner_step,
            outer_step=state.outer_step,
            extra=dict(state.extra),
        )


__all__ = [
    "AdaptResult",
    "FewShotAdapter",
    "FewShotAdapterError",
    "FewShotDemo",
    "SuccessFn",
    "default_success_fn",
]
