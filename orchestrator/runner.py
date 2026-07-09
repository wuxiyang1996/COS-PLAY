"""`EpisodeRunner` — drives one outer-environment rollout.

PLAN-PIPELINE-ORCHESTRATOR §4.1 (online control path).

The runner is intentionally policy-agnostic: it accepts an Actor object
that conforms to a minimal protocol (`choose_action(state, eligible)`).
The runner's responsibilities:

  1. Tick the outer environment.
  2. For each tick, ask the bank for candidate skills, narrow them via
     the harness `select_eligible_skills`, hand the eligible set to the
     actor, then `run_skill(actor.choice)`.
  3. Account every step in the budget.
  4. Persist `SkillEpisode`s and the outer `EpisodeMeta` to the
     artifact store.

If the actor declines to use a skill (returns `None`), the runner
records a "no-skill tick" but takes no env action — the env adapter is
responsible for any default behavior.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from common.ids import new_run_id
from common.state_schema import StateSchema
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_record import SkillRecord
from harness import EligibleSkill, SkillHarness
from orchestrator.artifact_store import ArtifactStore
from orchestrator.budget import BudgetController, BudgetExceeded
from skill_bank.repository import SkillRepository


class ActorLike(Protocol):
    """Minimal protocol an Actor must satisfy to be drivable by the runner."""

    def choose_action(
        self,
        state: StateSchema,
        eligible: List[EligibleSkill],
    ) -> Optional["ActorChoice"]:
        ...


class EnvLike(Protocol):
    """Minimal protocol an env adapter must satisfy."""

    def reset(self) -> StateSchema: ...

    def step(self, episode: Optional[SkillEpisode]) -> tuple[StateSchema, bool]:
        """Apply the latest skill_episode (if any) and return (next_state, done)."""
        ...


@dataclass
class ActorChoice:
    skill: SkillRecord
    bindings: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class EpisodeResult:
    run_id: str
    outer_steps: int
    skill_episodes: List[SkillEpisode]
    final_state: Optional[StateSchema]
    budget_snapshot: Dict[str, Any]
    aborted: bool
    abort_reason: Optional[str] = None


class EpisodeRunner:
    def __init__(
        self,
        *,
        env: EnvLike,
        actor: ActorLike,
        harness: SkillHarness,
        bank: SkillRepository,
        artifact_store: ArtifactStore,
    ) -> None:
        self._env = env
        self._actor = actor
        self._harness = harness
        self._bank = bank
        self._store = artifact_store

    def run(
        self,
        *,
        budget: BudgetController,
        max_outer_steps: Optional[int] = None,
    ) -> EpisodeResult:
        run_id = new_run_id()
        skill_eps: List[SkillEpisode] = []
        state = self._env.reset()
        aborted = False
        abort_reason: Optional[str] = None
        last_episode: Optional[SkillEpisode] = None

        try:
            while True:
                if max_outer_steps is not None and state.outer_step >= max_outer_steps:
                    break
                budget.add_outer_step()

                eligible = self._harness.select_eligible_skills(
                    self._bank.runnable(),
                    state,
                )
                choice = self._actor.choose_action(state, eligible)
                if choice is None:
                    next_state, done = self._env.step(None)
                else:
                    budget.add_skill_invocation()
                    last_episode = self._harness.run_skill(
                        choice.skill,
                        state,
                        parent_run_id=run_id,
                        bindings=choice.bindings,
                    )
                    skill_eps.append(last_episode)
                    self._store.put_skill_episode(last_episode)
                    next_state, done = self._env.step(last_episode)
                state = next_state
                state.outer_step += 1
                if done:
                    break
        except BudgetExceeded as exc:
            aborted = True
            abort_reason = f"budget_exceeded: {exc}"
        except Exception as exc:                              # noqa: BLE001
            aborted = True
            abort_reason = f"runner_exception: {exc!r}"

        meta = {
            "run_id": run_id,
            "outer_steps": state.outer_step,
            "n_skill_episodes": len(skill_eps),
            "skill_episode_ids": [e.episode_id for e in skill_eps],
            "final_state": state.to_json() if state else None,
            "aborted": aborted,
            "abort_reason": abort_reason,
            "budget": budget.snapshot(),
            "started_at": time.time(),
        }
        self._store.put_episode(run_id, meta)
        self._store.append_audit(
            {"kind": "episode_done", "run_id": run_id, "aborted": aborted}
        )

        return EpisodeResult(
            run_id=run_id,
            outer_steps=state.outer_step,
            skill_episodes=skill_eps,
            final_state=state,
            budget_snapshot=budget.snapshot(),
            aborted=aborted,
            abort_reason=abort_reason,
        )


__all__ = [
    "ActorChoice",
    "ActorLike",
    "EnvLike",
    "EpisodeResult",
    "EpisodeRunner",
]
