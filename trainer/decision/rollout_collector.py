"""
Rollout collector for the Decision Agent.

Two modes:
  1. collect_rollout / collect_batch — original single-env sequential
     collection (for standalone / eval use without VERL)
  2. VERLRolloutAdapter — bridges the Game-AI env to VERL's
     TrajectoryCollector.multi_turn_loop(), which handles batched
     multi-turn collection with vLLM/sglang actor inference.

VERL integration:
  In VERL training, rollout collection is done by:
    gen_batch_output = traj_collector.multi_turn_loop(
        gen_batch, actor_rollout_wg, envs, is_train=True
    )
  This replaces collect_batch(). The TrajectoryCollector handles:
    - Batched prompt tokenization
    - vLLM/sglang sequence generation
    - Projection (text → env action via gameai_projection)
    - Multi-turn env stepping
    - Reward accumulation into episode_rewards
  See trainer/decision/env_wrapper.py for the GameAIEnvironmentManager
  that provides the VERL env interface.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from trainer.common.metrics import RolloutRecord, RolloutStep
from trainer.common.seeds import SeedManager
from trainer.decision.reward_shaping import TrainRewardShaper

logger = logging.getLogger(__name__)


def classify_action_type(action: str) -> str:
    """Determine whether an action is primitive or a retrieval action."""
    if not action:
        return "primitive"
    upper = action.upper()
    for prefix in ("QUERY_MEM", "QUERY_SKILL", "CALL_SKILL"):
        if prefix in upper:
            return prefix
    return "primitive"


def collect_rollout(
    env_wrapper,
    policy: Any,
    reward_shaper: TrainRewardShaper,
    seed: int,
    max_steps: int = 500,
    episode_id: Optional[str] = None,
    extract_predicates: Optional[Callable[[str], Dict[str, float]]] = None,
    extract_embedding: Optional[Callable[[str], List[float]]] = None,
) -> RolloutRecord:
    """Collect one complete episode rollout (standalone mode).

    For VERL training, use TrajectoryCollector.multi_turn_loop() instead.
    """
    episode_id = episode_id or str(uuid.uuid4())[:8]
    reward_shaper.reset()

    obs, info = env_wrapper.reset(seed=seed)

    steps: List[RolloutStep] = []
    done = False
    t = 0

    while t < max_steps and not done:
        context = {
            "skill_cards": env_wrapper.state.retrieved_cards,
            "active_skill": env_wrapper.active_skill_id,
            "step": t,
        }

        policy_out = policy.sample(obs, context=context)
        action = policy_out.action
        logprob = policy_out.logprob

        action_type = classify_action_type(action)
        step_result = env_wrapper.step(action)

        r_env = step_result["r_env"]
        next_obs = step_result["obs"]
        done = step_result["done"]

        breakdown = reward_shaper.compute_reward(
            r_env=r_env,
            action_type=action_type,
            observation=next_obs,
            active_skill_id=step_result.get("active_skill_id"),
            skill_contract=step_result.get("active_skill_contract"),
        )

        predicates = {}
        if extract_predicates:
            try:
                predicates = extract_predicates(next_obs)
            except Exception:
                pass

        embedding = None
        if extract_embedding:
            try:
                embedding = extract_embedding(next_obs)
            except Exception:
                pass

        step = RolloutStep(
            step=t,
            obs_id=f"{episode_id}_obs_{t}",
            action=action,
            action_type=action_type,
            ui_events=[],
            predicates=predicates,
            embedding=embedding,
            r_env=breakdown.r_env,
            r_follow=breakdown.r_follow,
            r_cost=breakdown.r_cost,
            r_total=breakdown.r_total,
            done=done,
            episode_id=episode_id,
            traj_id=episode_id,
            seed=seed,
            active_skill_id=step_result.get("active_skill_id"),
            query_key=step_result.get("query_key"),
            logprob=logprob,
            value=policy_out.value,
        )
        steps.append(step)
        obs = next_obs
        t += 1

    record = RolloutRecord(
        episode_id=episode_id,
        traj_id=episode_id,
        seed=seed,
        steps=steps,
    )
    record.finalize()
    record.won = done
    record.score = record.total_r_env
    return record


def collect_batch(
    env_factory: Callable[[int], Any],
    policy: Any,
    skill_bank: Any,
    memory: Any,
    reward_config: Any,
    seed_manager: SeedManager,
    batch_size: int = 32,
    max_steps: int = 500,
    **kwargs,
) -> List[RolloutRecord]:
    """Collect a batch of rollouts with different seeds (standalone mode).

    For VERL training, use TrajectoryCollector.multi_turn_loop() instead,
    which handles batched multi-turn collection with distributed inference.
    """
    from trainer.decision.env_wrapper import EnvWrapper

    records: List[RolloutRecord] = []

    for i in range(batch_size):
        seed = seed_manager.next_train_seed()
        env = env_factory(seed)
        wrapper = EnvWrapper(env=env, skill_bank=skill_bank, memory=memory)
        shaper = TrainRewardShaper(config=reward_config, skill_bank=skill_bank)

        try:
            record = collect_rollout(
                env_wrapper=wrapper,
                policy=policy,
                reward_shaper=shaper,
                seed=seed,
                max_steps=max_steps,
                **kwargs,
            )
            records.append(record)
        except Exception as exc:
            logger.warning("Rollout %d (seed=%d) failed: %s", i, seed, exc)

    return records


# ---------------------------------------------------------------------------
# VERL rollout adapter
# ---------------------------------------------------------------------------
try:
    import torch
    from verl import DataProto
    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False


if _HAS_VERL:
    class VERLRolloutAdapter:
        """Adapter that converts VERL multi-turn rollout output to RolloutRecords.

        After VERL's TrajectoryCollector.multi_turn_loop() produces a
        DataProto batch, this adapter extracts per-episode RolloutRecords
        for use in co-evolution EM, logging, and evaluation.

        Usage:
            gen_batch_output = traj_collector.multi_turn_loop(...)
            adapter = VERLRolloutAdapter(tokenizer)
            records = adapter.extract_records(gen_batch_output)
        """

        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

        def extract_records(self, batch: DataProto) -> List[RolloutRecord]:
            """Extract RolloutRecords from a VERL DataProto batch.

            Reads episode_rewards, episode_lengths, and non_tensor_batch
            fields populated by the GameAIEnvironmentManager during rollout.
            """
            records = []
            ntb = batch.non_tensor_batch

            for i in range(len(batch)):
                episode_id = ntb.get("traj_uid", [f"ep_{i}"])[i] if "traj_uid" in ntb else f"ep_{i}"
                episode_reward = float(ntb["episode_rewards"][i]) if "episode_rewards" in ntb else 0.0
                episode_length = int(ntb["episode_lengths"][i]) if "episode_lengths" in ntb else 1

                steps = []
                for t in range(episode_length):
                    step = RolloutStep(
                        step=t,
                        obs_id=f"{episode_id}_obs_{t}",
                        action="",
                        action_type="primitive",
                        r_env=episode_reward / max(episode_length, 1) if t == episode_length - 1 else 0.0,
                        r_total=episode_reward / max(episode_length, 1) if t == episode_length - 1 else 0.0,
                        done=(t == episode_length - 1),
                        episode_id=str(episode_id),
                        traj_id=str(episode_id),
                        seed=0,
                    )
                    steps.append(step)

                if steps:
                    record = RolloutRecord(
                        episode_id=str(episode_id),
                        traj_id=str(episode_id),
                        seed=0,
                        steps=steps,
                    )
                    record.finalize()
                    record.won = bool(ntb.get("won", [False])[i]) if "won" in ntb else False
                    record.score = episode_reward
                    records.append(record)

            return records
