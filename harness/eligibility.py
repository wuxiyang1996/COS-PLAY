"""Eligibility filtering — narrowing the bank's candidate list down to
skills that are actually runnable *now*.

PLAN-HARNESS §5.2 (`select_eligible_skills`).

The harness applies five kinds of filters, in order:

  F1  status       : only ACTIVE / SHADOW / PROVISIONAL skills are returned
                     to the Actor; CANDIDATE / DRAFT / DEPRECATED never are
                     (PLAN-UNIFIED-SKILL-GATE §6 GateRunner / "no shadow ⇒
                     no active" invariant).
  F2  domain       : `state.domain` must be in `skill.feasible_domains`.
  F2′ task         : if `skill.feasible_tasks` is non-empty, the task token
                     extracted from `state.task` must be in it. Skills with
                     `feasible_tasks=[]` are *task-agnostic* and pass this
                     filter unconditionally — back-compat for skills
                     decorated before the task axis landed.
                     See `harness/README.md` §22.
  F3  adapter      : there must be a registered adapter for (domain, type).
  F4  applicability: adapter `can_handle()` returns True.

We never *score* skills here — that's the Actor's job. We only *narrow*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from common.enums import SkillStatus, SkillType
from common.state_schema import StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry


_RUNNABLE_STATUSES = frozenset(
    {SkillStatus.ACTIVE, SkillStatus.SHADOW, SkillStatus.PROVISIONAL}
)


def task_id_from_state(state: StateSchema) -> Optional[str]:
    """Extract the bare task identifier from `state.task`.

    The cold-start corpus emits `state.task = "make_gaming_env/<game>"`
    (e.g. `"make_gaming_env/twenty_forty_eight"`). Other domains may
    use bare strings (`"<game>"`) or different prefixes. We take the
    last `/`-separated segment, which collapses both shapes onto the
    same canonical form. Returns `None` for unset or whitespace-only
    `state.task` so the F2′ filter degrades gracefully (admit, don't
    veto) rather than silently dropping every skill.
    """

    raw = (state.task or "").strip()
    if not raw:
        return None
    if "/" in raw:
        return raw.rsplit("/", 1)[-1].strip() or None
    return raw


@dataclass
class EligibleSkill:
    """A skill the harness deems runnable right now, with provenance."""

    skill: SkillRecord
    adapter_name: str
    reasons: List[str] = field(default_factory=list)
    shadow_only: bool = False    # SHADOW skills must not affect outer-env actions
    # F2′ task-axis classification (harness/README §22). Three values:
    #   "agnostic"  — skill.feasible_tasks is empty (back-compat)
    #   "same_task" — state task ∈ skill.feasible_tasks
    #   "verified"  — state task ∈ skill.verified_tasks (subset of same_task)
    # `task_match == "agnostic"` is admitted to preserve behaviour for
    # cold-start banks decorated before this field landed.
    task_match: str = "agnostic"
    # Day-8: per-check booleans (PLAN-UNIFIED §3.4 / harness/README §9).
    # The filter sets these *for skills it admits*; rejected skills are
    # surfaced via the sibling `RejectedSkill` channel returned from
    # `EligibilityFilter.filter_with_rejections`.
    binding_ok: bool = True
    precondition_ok: bool = True
    evidence_ok: bool = True
    adapter_ok: bool = True

    def to_json(self) -> dict:
        return {
            "skill_id": self.skill.skill_id,
            "skill_name": self.skill.name,
            "skill_status": self.skill.status.value,
            "adapter_name": self.adapter_name,
            "shadow_only": self.shadow_only,
            "task_match": self.task_match,
            "reasons": list(self.reasons),
            "binding_ok": self.binding_ok,
            "precondition_ok": self.precondition_ok,
            "evidence_ok": self.evidence_ok,
            "adapter_ok": self.adapter_ok,
        }


@dataclass
class RejectedSkill:
    """A skill the eligibility filter rejected, with the reason.

    Day-8: closes harness/README §9 — rejected candidates were
    silently dropped before. With this channel the actor can render a
    veto log the planner can reason about (``why was this skill not
    available?``) and the Crafter can surface as
    ``false_binding_patterns`` evidence.

    Vetoes carry the *same* per-check booleans `EligibleSkill` does so
    consumers don't need a parallel decoding path.
    """

    skill: SkillRecord
    veto: str                  # short tag, e.g. "no_adapter" / "task_mismatch"
    veto_reason: str = ""      # human-readable explanation
    binding_ok: bool = True
    precondition_ok: bool = True
    evidence_ok: bool = True
    adapter_ok: bool = True

    def to_json(self) -> dict:
        return {
            "skill_id": self.skill.skill_id,
            "skill_name": self.skill.name,
            "skill_status": self.skill.status.value,
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "binding_ok": self.binding_ok,
            "precondition_ok": self.precondition_ok,
            "evidence_ok": self.evidence_ok,
            "adapter_ok": self.adapter_ok,
        }


class EligibilityFilter:
    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        allow_shadow: bool = True,
    ) -> None:
        self._registry = registry
        self._allow_shadow = allow_shadow

    def filter(
        self,
        candidates: Iterable[SkillRecord],
        state: StateSchema,
        *,
        skill_type_hint: Optional[SkillType] = None,
    ) -> List[EligibleSkill]:
        admitted, _rejected = self.filter_with_rejections(
            candidates, state, skill_type_hint=skill_type_hint,
        )
        return admitted

    def filter_with_rejections(
        self,
        candidates: Iterable[SkillRecord],
        state: StateSchema,
        *,
        skill_type_hint: Optional[SkillType] = None,
    ) -> tuple[List[EligibleSkill], List["RejectedSkill"]]:
        """Day-8 (PLAN-UNIFIED §3.4 / harness/README §9): same as
        ``filter`` but also returns the per-skill rejection channel.

        Both lists are returned in candidate iteration order. The
        rejection list lets the actor render a veto log
        (``why was X not available?``) and the Crafter surface
        ``false_binding_patterns`` evidence — concerns the original
        ``filter()`` API silently dropped.
        """
        out: List[EligibleSkill] = []
        rejected: List["RejectedSkill"] = []
        state_task = task_id_from_state(state)
        for skill in candidates:
            reasons: List[str] = []
            if skill.status not in _RUNNABLE_STATUSES:
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="status_not_runnable",
                    veto_reason=f"status={skill.status.value!r} not in {sorted(s.value for s in _RUNNABLE_STATUSES)}",
                ))
                continue
            if skill.status == SkillStatus.SHADOW and not self._allow_shadow:
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="shadow_disallowed",
                    veto_reason="harness allow_shadow=False",
                ))
                continue
            if state.domain not in skill.feasible_domains:
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="domain_mismatch",
                    veto_reason=f"state.domain={state.domain!r} not in feasible_domains={list(skill.feasible_domains)!r}",
                ))
                continue
            # F2′ task-axis veto (harness/README §22). When the skill
            # advertises a non-empty `feasible_tasks`, only states whose
            # task matches are admitted. Two cases pass:
            #   (a) `feasible_tasks=[]` (task-agnostic, back-compat),
            #   (b) state's bare task token ∈ skill.feasible_tasks.
            # If state has no task tag, we admit (degraded) rather than
            # veto blindly so single-step adapters / synthesised states
            # without a `state.task` continue to work.
            if skill.feasible_tasks:
                if state_task is None:
                    task_match = "agnostic"  # state-side missing → can't enforce
                elif state_task in skill.feasible_tasks:
                    task_match = (
                        "verified"
                        if state_task in skill.verified_tasks
                        else "same_task"
                    )
                else:
                    rejected.append(RejectedSkill(
                        skill=skill,
                        veto="task_mismatch",
                        veto_reason=(
                            f"state task={state_task!r} not in "
                            f"feasible_tasks={list(skill.feasible_tasks)!r}"
                        ),
                    ))
                    continue
            else:
                task_match = "agnostic"
            if skill_type_hint is not None and skill.skill_type != skill_type_hint and skill.skill_type != SkillType.MIXED:
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="skill_type_mismatch",
                    veto_reason=(
                        f"skill_type={skill.skill_type.value!r} != "
                        f"hint={skill_type_hint.value!r}"
                    ),
                ))
                continue
            adapter = self._registry.get(state.domain, skill.skill_type)
            if adapter is None:
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="no_adapter",
                    veto_reason=(
                        f"no adapter registered for ({state.domain},"
                        f"{skill.skill_type.value})"
                    ),
                    adapter_ok=False,
                ))
                continue
            try:
                ok = adapter.can_handle(skill, state)
            except Exception as exc:                        # noqa: BLE001
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="adapter_raised",
                    veto_reason=f"adapter.can_handle raised: {exc!r}",
                    adapter_ok=False,
                ))
                continue
            if not ok:
                rejected.append(RejectedSkill(
                    skill=skill,
                    veto="adapter_cannot_handle",
                    veto_reason=f"adapter={adapter.name} returned can_handle=False",
                    adapter_ok=False,
                ))
                continue
            reasons.append(
                f"status={skill.status.value} domain={state.domain} "
                f"task={state_task or '<unset>'}/match={task_match} "
                f"adapter={adapter.name} type={skill.skill_type.value}"
            )
            out.append(
                EligibleSkill(
                    skill=skill,
                    adapter_name=adapter.name,
                    reasons=reasons,
                    shadow_only=skill.status == SkillStatus.SHADOW,
                    task_match=task_match,
                )
            )
        return out, rejected


__all__ = [
    "EligibilityFilter",
    "EligibleSkill",
    "RejectedSkill",
    "task_id_from_state",
]
