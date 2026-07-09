"""
Fixed-seed evaluation harness shared by both trainers.

Used for:
  - Decision Agent evaluation (win rate, reward breakdown on fixed seeds)
  - SkillBank quick evaluation / gating (re-decode holdout with Bank')

Runs a frozen agent on deterministic seeds and computes metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from trainer.common.metrics import (
    DecisionMetrics,
    RolloutRecord,
    SkillBankMetrics,
    aggregate_decision_metrics,
)
from trainer.common.seeds import SeedManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision Agent evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of one fixed-seed evaluation run."""

    metrics: DecisionMetrics = field(default_factory=DecisionMetrics)
    rollouts: List[RolloutRecord] = field(default_factory=list)
    seeds_used: List[int] = field(default_factory=list)
    bank_version: int = 0


def run_decision_eval(
    env_factory: Callable[[int], Any],
    agent: Any,
    seed_manager: SeedManager,
    num_episodes: int = 8,
    max_steps: int = 1000,
    rollout_fn: Optional[Callable] = None,
    bank_version: int = 0,
) -> EvalResult:
    """Evaluate the decision agent on fixed seeds.

    Args:
        env_factory: callable(seed) -> env instance
        agent: VLMDecisionAgent (frozen weights during eval)
        seed_manager: provides deterministic eval seeds
        num_episodes: how many eval episodes to run
        max_steps: per-episode step limit
        rollout_fn: optional custom rollout function; if None, uses
                     decision_agents.agent.run_episode_vlm_agent
        bank_version: current bank version for logging

    Returns:
        EvalResult with aggregated metrics and per-episode rollouts.
    """
    if rollout_fn is None:
        from decision_agents.agent import run_episode_vlm_agent as _run
        rollout_fn = _run

    rollouts: List[RolloutRecord] = []
    seeds_used: List[int] = []

    for i in range(num_episodes):
        seed = seed_manager.get_eval_seed(i)
        seeds_used.append(seed)
        env = env_factory(seed)

        try:
            result = rollout_fn(env=env, agent=agent, max_steps=max_steps)
            record = _result_to_rollout_record(result, seed, f"eval_{i}")
            rollouts.append(record)
        except Exception as exc:
            logger.warning("Eval episode %d (seed=%d) failed: %s", i, seed, exc)

    metrics = aggregate_decision_metrics(rollouts)
    return EvalResult(
        metrics=metrics,
        rollouts=rollouts,
        seeds_used=seeds_used,
        bank_version=bank_version,
    )


def _result_to_rollout_record(
    result, seed: int, episode_id: str
) -> RolloutRecord:
    """Convert run_episode_vlm_agent output to a RolloutRecord.

    Accepts either an Episode (new format) or a flat dict (legacy).
    """
    from trainer.common.metrics import RolloutStep

    try:
        from data_structure.experience import Episode as _Episode
        is_episode = isinstance(result, _Episode)
    except ImportError:
        is_episode = False

    if is_episode:
        meta = result.metadata or {}
        cumulative = meta.get("cumulative_reward", {})
        done_flag = meta.get("done", result.outcome if result.outcome is not None else False)
        ep_id = getattr(result, "episode_id", None) or episode_id
        env_name = getattr(result, "env_name", "") or ""
        game_name = getattr(result, "game_name", "") or ""

        steps = []
        for t, exp in enumerate(result.experiences):
            rd = exp.reward_details or {"r_env": exp.reward}
            step = RolloutStep(
                step=t,
                obs_id=f"obs_{t}",
                action=str(exp.action),
                action_type=exp.action_type or "primitive",
                r_env=rd.get("r_env", 0.0),
                r_follow=rd.get("r_follow", 0.0),
                r_cost=rd.get("r_cost", 0.0),
                r_total=rd.get("r_total", 0.0),
                done=exp.done,
                episode_id=ep_id,
                seed=seed,
                active_skill_id=exp.sub_tasks,
            )
            steps.append(step)

        record = RolloutRecord(
            episode_id=ep_id,
            seed=seed,
            env_name=env_name,
            game_name=game_name,
            steps=steps,
        )
        record.total_reward = cumulative.get("r_total", sum(s.r_total for s in steps))
        record.total_r_env = cumulative.get("r_env", sum(s.r_env for s in steps))
        record.episode_length = len(steps)
        record.won = done_flag
        record.score = record.total_r_env
        return record

    # Legacy flat dict path
    meta = result if isinstance(result, dict) else {}
    actions = meta.get("actions", [])
    reward_details = meta.get("reward_details", [])
    cumulative = meta.get("cumulative_reward", {})
    done_flag = meta.get("done", False)

    steps = []
    for t in range(len(actions)):
        rd = reward_details[t] if t < len(reward_details) else {}
        step = RolloutStep(
            step=t,
            obs_id=f"obs_{t}",
            action=str(actions[t]),
            r_env=rd.get("r_env", 0.0),
            r_follow=rd.get("r_follow", 0.0),
            r_cost=rd.get("r_cost", 0.0),
            r_total=rd.get("r_total", 0.0),
            episode_id=episode_id,
            seed=seed,
        )
        steps.append(step)

    record = RolloutRecord(
        episode_id=episode_id,
        seed=seed,
        steps=steps,
    )
    record.total_reward = cumulative.get("r_total", sum(s.r_total for s in steps))
    record.total_r_env = cumulative.get("r_env", sum(s.r_env for s in steps))
    record.episode_length = len(steps)
    record.won = done_flag
    record.score = record.total_r_env
    return record


# ---------------------------------------------------------------------------
# SkillBank quick evaluation (gating)
# ---------------------------------------------------------------------------

@dataclass
class SkillBankQuickEvalResult:
    """Result of SkillBank quick evaluation for gating."""

    accepted: bool = True
    new_rate: float = 0.0
    mean_margin: float = 0.0
    mean_pass_rate: float = 0.0
    confusion_pairs: int = 0
    reason: str = ""
    metrics: Optional[SkillBankMetrics] = None


def run_skillbank_quick_eval(
    bank_candidate: Any,
    holdout_trajectories: Sequence[Dict[str, Any]],
    decode_fn: Callable,
    gating_config: Optional[Dict[str, Any]] = None,
    bank_current: Optional[Any] = None,
) -> SkillBankQuickEvalResult:
    """Evaluate a candidate bank on holdout trajectories for gating.

    Re-decodes a small holdout set with the candidate bank and checks:
      - NEW rate (fraction of segments labelled NEW)
      - Average margin (decode confidence)
      - Contract pass rate distribution
      - Confusion matrix of top confusers

    If metrics regress beyond thresholds, returns accepted=False.

    Args:
        bank_candidate: proposed Bank_{k+1}
        holdout_trajectories: list of trajectory dicts (from ingest_rollouts)
        decode_fn: callable(trajectory, bank) -> decode_result
        gating_config: thresholds dict (from skillbank_em.yaml gating section)
        bank_current: current Bank_k for regression comparison

    Returns:
        SkillBankQuickEvalResult with accept/reject decision and metrics.
    """
    cfg = gating_config or {}
    max_new_rate = cfg.get("max_new_rate", 0.3)
    min_pass_rate = cfg.get("min_pass_rate", 0.6)
    margin_regression_tol = cfg.get("margin_regression_tol", 0.1)

    total_segments = 0
    new_segments = 0
    margins: List[float] = []
    pass_rates: List[float] = []
    confusion_count = 0

    for traj in holdout_trajectories:
        try:
            result = decode_fn(traj, bank_candidate)
        except Exception as exc:
            logger.warning("Quick eval decode failed: %s", exc)
            continue

        segments = result.get("segments", [])
        total_segments += len(segments)

        for seg in segments:
            label = seg.get("skill_label", "")
            if label in ("__NEW__", "NEW"):
                new_segments += 1

            margin = seg.get("margin", 0.0)
            if margin is not None:
                margins.append(float(margin))

            pr = seg.get("pass_rate")
            if pr is not None:
                pass_rates.append(float(pr))

            confusers = seg.get("confusers", [])
            confusion_count += len(confusers)

    new_rate = new_segments / max(total_segments, 1)
    mean_margin = sum(margins) / len(margins) if margins else 0.0
    mean_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 1.0

    accepted = True
    reason = ""

    if new_rate > max_new_rate:
        accepted = False
        reason = f"NEW rate {new_rate:.2f} > {max_new_rate}"

    if mean_pass_rate < min_pass_rate:
        accepted = False
        reason = f"pass rate {mean_pass_rate:.2f} < {min_pass_rate}"

    if bank_current is not None and margins:
        current_result = SkillBankQuickEvalResult()
        current_margins: List[float] = []
        for traj in holdout_trajectories:
            try:
                cr = decode_fn(traj, bank_current)
                for seg in cr.get("segments", []):
                    m = seg.get("margin")
                    if m is not None:
                        current_margins.append(float(m))
            except Exception:
                pass
        if current_margins:
            current_mean = sum(current_margins) / len(current_margins)
            if mean_margin < current_mean - margin_regression_tol:
                accepted = False
                reason = (f"margin regressed {mean_margin:.3f} vs "
                          f"current {current_mean:.3f}")

    sm = SkillBankMetrics(
        n_skills=len(getattr(bank_candidate, "skill_ids", [])),
        new_pool_size=new_segments,
        mean_pass_rate=mean_pass_rate,
        mean_margin=mean_margin,
        confusion_pairs=confusion_count,
    )

    return SkillBankQuickEvalResult(
        accepted=accepted,
        new_rate=new_rate,
        mean_margin=mean_margin,
        mean_pass_rate=mean_pass_rate,
        confusion_pairs=confusion_count,
        reason=reason,
        metrics=sm,
    )
