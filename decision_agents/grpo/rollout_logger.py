"""Per-step rollout logger that emits trainer/common metrics records.

The GRPO trainer in :mod:`trainer.coevolution.grpo_training` ingests
:class:`trainer.common.metrics.RolloutRecord` instances composed of
:class:`RolloutStep` rows.  This logger lets the
:class:`~decision_agents.grpo.actor_qwen_vl.QwenVLActor` produce those
records without coupling the actor to the GRPO trainer's internal
conventions (predicate dict shape, action_type taxonomy, etc.).

Lifecycle
---------
::

    logger = GRPORolloutLogger(env_name="tetris", game_name="tetris")
    logger.start_episode(seed=42)
    for step in range(N):
        decision = actor.step(...)
        env_obs, reward, done, info = env.step(decision.action)
        rr = actor.observe_result(decision, reward=reward, done=done)
        logger.log_step(decision=decision, reward_result=rr, done=done)
    record = logger.finalize_episode(score=info["score"], won=info["won"])
    grpo_trainer.add(record)

Every field on :class:`RolloutStep` is populated, including the
``r_env`` / ``r_follow`` / ``r_cost`` decomposition the GRPO trainer
needs to compute advantages.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from decision_agents.actor_agent import ActorDecision
from decision_agents.reward_func import RewardResult
from trainer.common.metrics import RolloutRecord, RolloutStep

_LOGGER = logging.getLogger(__name__)


class GRPORolloutLogger:
    """Build :class:`RolloutRecord` from a stream of actor decisions.

    Parameters
    ----------
    env_name / game_name
        Tags propagated onto the :class:`RolloutRecord` for downstream
        per-game metric aggregation in
        :func:`trainer.common.metrics.aggregate_decision_metrics`.
    """

    def __init__(self, *, env_name: str = "", game_name: str = "") -> None:
        self.env_name = env_name
        self.game_name = game_name or env_name
        self._record: Optional[RolloutRecord] = None
        self._step_idx: int = 0

    # ── episode lifecycle ────────────────────────────────────────────

    def start_episode(self, *, seed: int = 0, episode_id: Optional[str] = None) -> str:
        """Begin a fresh episode; returns the assigned episode_id."""
        eid = episode_id or uuid.uuid4().hex[:12]
        self._record = RolloutRecord(
            episode_id=eid,
            traj_id=eid,
            seed=seed,
            env_name=self.env_name,
            game_name=self.game_name,
        )
        self._step_idx = 0
        return eid

    def log_step(
        self,
        *,
        decision: ActorDecision,
        reward_result: RewardResult,
        done: bool = False,
        obs_id: str = "",
        ui_events: Optional[List[str]] = None,
        predicates: Optional[Dict[str, float]] = None,
        embedding: Optional[List[float]] = None,
        logprob: Optional[float] = None,
    ) -> RolloutStep:
        """Append one step to the active episode and return it."""
        if self._record is None:
            raise RuntimeError("call start_episode() before log_step()")
        eid = self._record.episode_id
        action_type = self._infer_action_type(decision)
        query_key = self._extract_query_key(decision, action_type)

        step = RolloutStep(
            step=self._step_idx,
            obs_id=obs_id,
            action=decision.action or "",
            action_type=action_type,
            ui_events=list(ui_events or []),
            predicates=dict(predicates or {}),
            embedding=embedding,
            r_env=float(reward_result.r_env),
            r_follow=float(reward_result.r_follow),
            r_cost=float(reward_result.r_cost),
            r_total=float(reward_result.r_total),
            done=bool(done),
            episode_id=eid,
            traj_id=eid,
            seed=self._record.seed,
            active_skill_id=decision.active_skill_id,
            query_key=query_key,
            intentions=decision.intention or None,
            logprob=logprob,
        )
        self._record.steps.append(step)
        self._step_idx += 1
        return step

    def finalize_episode(
        self,
        *,
        score: float = 0.0,
        won: bool = False,
    ) -> RolloutRecord:
        """Close the active episode and return the finalised record.

        After this call ``start_episode()`` must be invoked again before
        the next ``log_step()``.  Helps catch the common bug where a
        runner forgets to reset between episodes.
        """
        if self._record is None:
            raise RuntimeError("call start_episode() before finalize_episode()")
        rec = self._record
        rec.score = float(score)
        rec.won = bool(won)
        rec.finalize()
        self._record = None
        return rec

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _infer_action_type(decision: ActorDecision) -> str:
        """Map the :class:`ActorDecision` flags onto the GRPO action_type
        taxonomy (``primitive`` / ``QUERY_MEM`` / ``QUERY_SKILL`` /
        ``CALL_SKILL``).

        Priority — a single step can fire several seams; we pick the
        highest-cost label so reward shaping in
        :class:`~decision_agents.reward_func.RewardComputer` stays
        consistent with the action_type the trainer reads here.
        """
        if decision.queried_skill:
            return "QUERY_SKILL"
        if decision.active_skill_id is not None:
            return "CALL_SKILL"
        if decision.queried_mem:
            return "QUERY_MEM"
        return "primitive"

    @staticmethod
    def _extract_query_key(decision: ActorDecision, action_type: str) -> Optional[str]:
        """Pull the RAG query key the actor used, when one applies.

        The skill / memory queries aren't kept verbatim on
        :class:`ActorDecision`, so we approximate using the intention or
        summary — enough for the GRPO trainer's
        ``mean_query_key_len`` metric.
        """
        if action_type not in ("QUERY_SKILL", "QUERY_MEM"):
            return None
        return (decision.intention or decision.summary or "").strip() or None
