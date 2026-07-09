"""
GRPO (Group Relative Policy Optimization) trainer for the VLM Decision Agent.

Two modes:
  1. GRPOTrainer — original standalone trainer with custom group ranking
     and policy update loop (for non-VERL use / debugging).
  2. VERL integration — in VERL training, GRPO/GiGPO is handled by
     RayPPOTrainer with adv_estimator=grpo or adv_estimator=gigpo.
     The GameAITrainer subclass adds co-evolution callback support.

VERL integration:
  RayPPOTrainer.fit() handles the full pipeline:
    - Rollout collection via TrajectoryCollector.multi_turn_loop()
    - Advantage computation via compute_advantage() with GRPO/GiGPO
    - Actor update via actor_rollout_wg.update_actor()
    - KL penalty, entropy bonus, invalid action penalty
    - Distributed training via Ray + FSDP

  GameAITrainer(RayPPOTrainer) adds:
    - SkillBank co-evolution callback after each training step
    - Bank hot-swap in environment workers
    - Bank metrics logging

  Config mapping (old GRPOConfig -> VERL Hydra config):
    group_size -> env.rollout.n
    clip_ratio -> actor_rollout_ref.actor.clip_ratio (default 0.2)
    kl_coeff -> actor_rollout_ref.actor.kl_loss_coef
    lr -> actor_rollout_ref.actor.optim.lr
    gamma -> algorithm.gamma
    normalize_advantages -> algorithm.norm_adv_by_std_in_grpo
    entropy_coeff -> actor_rollout_ref.actor.entropy_coeff
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from trainer.common.metrics import (
    DecisionMetrics,
    RolloutRecord,
    aggregate_decision_metrics,
)
from trainer.decision.policy_interface import PolicyInterface
from trainer.decision.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO hyperparameters."""

    group_size: int = 8
    clip_ratio: float = 0.2
    kl_coeff: float = 0.01
    lr: float = 1e-5
    epochs_per_batch: int = 4
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coeff: float = 0.01


@dataclass
class GRPOTrainStats:
    """Statistics from one GRPO training iteration."""

    loss: float = 0.0
    policy_loss: float = 0.0
    kl_loss: float = 0.0
    entropy: float = 0.0
    mean_advantage: float = 0.0
    mean_return: float = 0.0
    grad_norm: float = 0.0
    n_episodes: int = 0
    n_steps: int = 0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


class GRPOTrainer:
    """Standalone GRPO trainer for the VLM Decision Agent.

    For VERL-based training, use GameAITrainer (below) or
    RayPPOTrainer with adv_estimator=grpo instead.
    """

    def __init__(
        self,
        policy: PolicyInterface,
        config: Optional[GRPOConfig] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        self.policy = policy
        self.cfg = config or GRPOConfig()
        self.buffer = replay_buffer or ReplayBuffer()
        self._iteration = 0
        self._total_episodes = 0

    def train_step(self, rollouts: List[RolloutRecord]) -> GRPOTrainStats:
        """Execute one GRPO training step on a batch of rollouts."""
        self._iteration += 1
        self._total_episodes += len(rollouts)

        self.buffer.add_batch(rollouts)

        groups = self._form_groups(rollouts)
        all_advantages: List[float] = []
        all_logprobs: List[float] = []
        all_old_logprobs: List[float] = []
        all_returns: List[float] = []

        for group in groups:
            returns = [r.total_reward for r in group]
            advantages = self._compute_group_advantages(returns)
            all_returns.extend(returns)
            all_advantages.extend(advantages)

            for record, adv in zip(group, advantages):
                for step in record.steps:
                    if step.logprob is not None:
                        all_old_logprobs.append(step.logprob)
                        new_lp = self.policy.logprob(
                            observation=step.obs_id,
                            action=step.action,
                        )
                        all_logprobs.append(new_lp)

        if not all_logprobs:
            return GRPOTrainStats(n_episodes=len(rollouts))

        loss, stats = self._compute_grpo_loss(
            all_logprobs, all_old_logprobs, all_advantages
        )

        update_info = self.policy.update(loss)
        stats.n_episodes = len(rollouts)
        stats.n_steps = sum(r.episode_length for r in rollouts)
        stats.mean_return = float(np.mean(all_returns)) if all_returns else 0.0
        stats.grad_norm = update_info.get("grad_norm", 0.0)

        return stats

    def _form_groups(self, rollouts):
        g = self.cfg.group_size
        groups = [rollouts[i:i + g] for i in range(0, len(rollouts), g)]
        return [grp for grp in groups if len(grp) >= 2]

    def _compute_group_advantages(self, returns):
        n = len(returns)
        if n <= 1:
            return [0.0] * n
        arr = np.array(returns, dtype=np.float64)
        if self.cfg.normalize_advantages:
            mean = arr.mean()
            std = arr.std() + 1e-8
            return ((arr - mean) / std).tolist()
        else:
            ranks = np.argsort(np.argsort(arr)).astype(np.float64)
            return (2.0 * ranks / (n - 1) - 1.0).tolist()

    def _compute_grpo_loss(self, logprobs, old_logprobs, advantages):
        lp = np.array(logprobs, dtype=np.float64)
        old_lp = np.array(old_logprobs, dtype=np.float64)
        n = min(len(lp), len(old_lp), len(advantages))
        if n == 0:
            return 0.0, GRPOTrainStats()

        lp, old_lp = lp[:n], old_lp[:n]
        adv = np.array(advantages[:n], dtype=np.float64)

        if self.cfg.normalize_advantages and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ratio = np.exp(lp - old_lp)
        clipped = np.clip(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
        policy_loss = -np.mean(np.minimum(ratio * adv, clipped * adv))

        kl = np.mean(old_lp - lp)
        kl_loss = self.cfg.kl_coeff * kl

        entropy = -np.mean(lp)
        entropy_bonus = -self.cfg.entropy_coeff * entropy

        loss = float(policy_loss + kl_loss + entropy_bonus)

        stats = GRPOTrainStats(
            loss=loss,
            policy_loss=float(policy_loss),
            kl_loss=float(kl_loss),
            entropy=float(entropy),
            mean_advantage=float(adv.mean()),
        )
        return loss, stats

    def compute_gae(self, rewards, values, dones):
        """Compute GAE for a single episode."""
        T = len(rewards)
        advantages = [0.0] * T
        gae = 0.0
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda

        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            if dones[t]:
                next_val = 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * lam * gae * (0.0 if dones[t] else 1.0)
            advantages[t] = gae

        return advantages

    @property
    def iteration(self):
        return self._iteration

    @property
    def total_episodes(self):
        return self._total_episodes


# ---------------------------------------------------------------------------
# VERL-integrated trainer (subclass of RayPPOTrainer)
# ---------------------------------------------------------------------------
try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False


if _HAS_VERL:
    class GameAITrainer(RayPPOTrainer):
        """RayPPOTrainer with integrated SkillBank co-evolution.

        Overrides fit() to inject the co-evolution callback after each
        training step.  The callback periodically runs the SkillBank
        Hard-EM pipeline and hot-swaps the bank in environment workers.

        Usage:
            from trainer.decision.grpo_trainer import GameAITrainer
            trainer = GameAITrainer(..., coevo_callback=callback)
            trainer.fit()
        """

        def __init__(self, *args, coevo_callback=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.coevo_callback = coevo_callback

        def fit(self):
            """Training loop with co-evolution callback injection.

            Extends RayPPOTrainer.fit() by inserting the co-evolution
            callback after each training step's metrics collection.
            See verl-agent/agent_system/coevolution/trainer.py for the
            reference implementation.
            """
            from omegaconf import OmegaConf
            from verl.utils.tracking import Tracking
            from verl.trainer.ppo.ray_trainer import (
                _timer,
                compute_response_mask,
                compute_advantage,
                apply_kl_penalty,
                apply_invalid_action_penalty,
                AdvantageEstimator,
            )
            from verl.trainer.ppo.core_algos import agg_loss
            from verl.trainer.ppo.metric_utils import (
                compute_data_metrics,
                compute_throughout_metrics,
                compute_timing_metrics,
            )
            from verl.trainer.ppo.reward import compute_reward, compute_reward_async
            from verl.utils.metric import reduce_metrics
            from verl import DataProto
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            from agent_system.multi_turn_rollout import adjust_batch
            import ray
            import torch
            from copy import deepcopy
            from pprint import pprint
            from tqdm import tqdm

            try:
                from gigpo import core_gigpo
            except ImportError:
                core_gigpo = None

            logger_tracker = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

            self.global_steps = 0
            self._load_checkpoint()

            if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger_tracker.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    return

            progress_bar = tqdm(
                total=self.total_training_steps,
                initial=self.global_steps,
                desc="Training Progress",
            )

            self.global_steps += 1
            last_val_metrics = None

            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    metrics = {}
                    timing_raw = {}
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                    for k in ["multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"]:
                        if k in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append(k)
                    gen_batch = batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    is_last_step = self.global_steps >= self.total_training_steps

                    with _timer("step", timing_raw):
                        with _timer("gen", timing_raw):
                            gen_batch_output = self.traj_collector.multi_turn_loop(
                                gen_batch=gen_batch,
                                actor_rollout_wg=self.actor_rollout_wg,
                                envs=self.envs,
                                is_train=True,
                            )

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with _timer("gen_max", timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        del batch
                        batch = gen_batch_output

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO and core_gigpo is not None:
                            step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                                batch=batch, gamma=self.config.algorithm.gamma,
                            )
                            batch.batch["step_rewards"] = step_rewards_tensor

                        batch = adjust_batch(self.config, batch)
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        with _timer("reward", timing_raw):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    batch, self.config, self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(
                                    batch, self.reward_fn
                                )

                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            with _timer("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer("adv", timing_raw):
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            if self.config.actor_rollout_ref.actor.get(
                                "use_invalid_action_penalty", True
                            ):
                                batch, invalid_metrics = apply_invalid_action_penalty(
                                    batch,
                                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                )
                                metrics.update(invalid_metrics)

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                                gigpo_mode=self.config.algorithm.gigpo.mode,
                                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            )

                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer("update_actor", timing_raw):
                                batch.meta_info["multi_turn"] = (
                                    self.config.actor_rollout_ref.rollout.multi_turn.enable
                                )
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # validate
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        ):
                            with _timer("testing", timing_raw):
                                val_metrics = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    # ═══════════════════════════════════════════════════
                    # SkillBank co-evolution callback (Game-AI specific)
                    # ═══════════════════════════════════════════════════
                    if self.coevo_callback is not None:
                        with _timer("coevolution", timing_raw):
                            coevo_metrics = self.coevo_callback.on_step_end(
                                global_step=self.global_steps,
                                batch=batch,
                                metrics=metrics,
                            )
                            metrics.update(coevo_metrics)

                    # Standard VERL metrics
                    metrics.update({
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    })
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(
                        compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                    )

                    logger_tracker.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
