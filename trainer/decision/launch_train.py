"""
CLI entry point for Decision Agent training.

Two modes:
  1. Standalone: python -m trainer.decision.launch_train --config ...
     Uses LLMPolicy + custom GRPO trainer (no VERL dependency).
  2. VERL mode (preferred): python -m trainer.decision.launch_train [overrides...]
     Delegates to verl-agent (https://github.com/verl-project/verl) and runs
     verl.trainer.main_gameai with Hydra, Ray, vLLM/sglang and FSDP.

VERL launch:
    python -m scripts.run_trainer --verl [overrides...]
    # or from this module (no --config):
    python -m trainer.decision.launch_train algorithm.adv_estimator=gigpo env.env_name=gameai ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Original standalone mode (no VERL)
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_grpo_config(cfg: Dict[str, Any]):
    from trainer.decision.grpo_trainer import GRPOConfig
    grpo = cfg.get("grpo", {})
    return GRPOConfig(
        group_size=grpo.get("group_size", 8),
        clip_ratio=grpo.get("clip_ratio", 0.2),
        kl_coeff=grpo.get("kl_coeff", 0.01),
        lr=grpo.get("lr", 1e-5),
        epochs_per_batch=grpo.get("epochs_per_batch", 4),
        max_grad_norm=grpo.get("max_grad_norm", 1.0),
        gamma=grpo.get("gamma", 0.99),
        gae_lambda=grpo.get("gae_lambda", 0.95),
        normalize_advantages=grpo.get("normalize_advantages", True),
        entropy_coeff=grpo.get("entropy_coeff", 0.01),
    )


def train_decision_agent(
    cfg: Dict[str, Any],
    env_factory: Any = None,
    skill_bank: Any = None,
    memory: Any = None,
) -> None:
    """Main training loop for standalone mode (no VERL)."""
    from trainer.common.eval_harness import run_decision_eval
    from trainer.common.logging import TrainLogger
    from trainer.common.metrics import aggregate_decision_metrics
    from trainer.common.seeds import SeedManager, set_global_seed
    from trainer.decision.grpo_trainer import GRPOConfig, GRPOTrainer
    from trainer.decision.policy_interface import LLMPolicy
    from trainer.decision.replay_buffer import ReplayBuffer
    from trainer.decision.rollout_collector import collect_batch

    set_global_seed(42)

    grpo_cfg = build_grpo_config(cfg)
    model_cfg = cfg.get("model", {})
    rollout_cfg = cfg.get("rollout", {})
    replay_cfg = cfg.get("replay", {})
    eval_cfg = cfg.get("eval", {})
    log_cfg = cfg.get("logging", {})
    sched_cfg = cfg.get("schedule", {})

    policy = LLMPolicy(
        model_name=model_cfg.get("name", "gpt-4o-mini"),
        lr=grpo_cfg.lr,
    )

    buffer = ReplayBuffer(
        capacity=replay_cfg.get("capacity", 10000),
        priority_alpha=replay_cfg.get("priority_alpha", 0.6),
        priority_beta=replay_cfg.get("priority_beta", 0.4),
        min_episodes=replay_cfg.get("min_episodes", 64),
    )

    trainer = GRPOTrainer(policy=policy, config=grpo_cfg, replay_buffer=buffer)

    seed_manager = SeedManager(
        base_seed=42,
        eval_seeds=eval_cfg.get("seeds", [42, 137, 256, 512]),
    )

    train_logger = TrainLogger(
        log_dir=log_cfg.get("log_dir", "runs/decision_grpo"),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project"),
    )

    from decision_agents.reward_func import RewardConfig
    costs = cfg.get("costs", {})
    follow = cfg.get("follow_shaping", {})
    reward_config = RewardConfig(
        w_follow=follow.get("w_follow", 0.1),
        query_mem_cost=costs.get("c_mem", -0.05),
        query_skill_cost=costs.get("c_skill", -0.05),
        call_skill_cost=costs.get("c_call", -0.02),
        skill_switch_cost=costs.get("c_switch", -0.10),
        follow_predicate_bonus=follow.get("predicate_bonus", 0.05),
        follow_completion_bonus=follow.get("completion_bonus", 0.20),
        follow_no_progress_penalty=follow.get("no_progress_penalty", -0.01),
    )

    total_episodes = sched_cfg.get("total_episodes", 50000)
    batch_size = rollout_cfg.get("batch_size", 32)
    max_steps = rollout_cfg.get("max_steps", 500)
    eval_interval = eval_cfg.get("interval_episodes", 50)
    log_interval = log_cfg.get("log_interval", 10)
    save_interval = log_cfg.get("save_interval", 100)

    episode_count = 0
    bank_version = 0

    logger.info("Starting Decision Agent GRPO training (total=%d)", total_episodes)

    while episode_count < total_episodes:
        rollouts = collect_batch(
            env_factory=env_factory,
            policy=policy,
            skill_bank=skill_bank,
            memory=memory,
            reward_config=reward_config,
            seed_manager=seed_manager,
            batch_size=batch_size,
            max_steps=max_steps,
        )

        stats = trainer.train_step(rollouts)
        episode_count += len(rollouts)

        if episode_count % log_interval < batch_size:
            metrics = aggregate_decision_metrics(rollouts)
            train_logger.log_decision_metrics(
                metrics, episode=episode_count,
                extra=stats.to_dict(),
            )

        if episode_count % eval_interval < batch_size:
            eval_result = run_decision_eval(
                env_factory=env_factory,
                agent=None,
                seed_manager=seed_manager,
                num_episodes=eval_cfg.get("num_eval_episodes", 10),
                max_steps=eval_cfg.get("timeout_steps", 1000),
                bank_version=bank_version,
            )
            train_logger.log_eval(
                eval_result.metrics,
                episode=episode_count,
                bank_version=bank_version,
                seeds_used=eval_result.seeds_used,
            )

        if episode_count % save_interval < batch_size:
            checkpoint_dir = Path(log_cfg.get("log_dir", "runs/decision_grpo")) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            train_logger.log_event("checkpoint", {
                "episode": episode_count,
                "iteration": trainer.iteration,
            })

    train_logger.log_event("training_complete", {"total_episodes": episode_count})
    train_logger.close()
    logger.info("Decision Agent training complete (%d episodes)", episode_count)


# ---------------------------------------------------------------------------
# VERL mode
# ---------------------------------------------------------------------------
try:
    import ray
    import hydra
    from omegaconf import OmegaConf
    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False


def _make_gameai_envs(config):
    """Build Game-AI environments for VERL training."""
    from trainer.decision.env_wrapper import (
        build_gameai_envs,
        gameai_projection,
        GameAIEnvironmentManager,
    )

    seed = config.env.get("seed", 42)
    env_num = config.data.train_batch_size
    group_n = config.env.rollout.n

    # Build env factory from game config
    env_factory = _get_env_factory(config)

    vec_env = build_gameai_envs(
        env_factory=env_factory,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
    )
    envs = GameAIEnvironmentManager(vec_env, gameai_projection, config)

    # Validation envs
    val_env_num = config.data.val_batch_size
    val_vec_env = build_gameai_envs(
        env_factory=env_factory,
        seed=seed + 10000,
        env_num=val_env_num,
        group_n=1,
    )
    val_envs = GameAIEnvironmentManager(val_vec_env, gameai_projection, config)

    return envs, val_envs


def _get_env_factory(config):
    """Get env factory from config, with fallback."""
    env_name = config.env.get("env_name", "gameai")
    max_steps = config.env.get("max_steps", 50)

    def factory(seed=0):
        try:
            from decision_agents.envs import make_game_env
            return make_game_env(env_name, seed=seed, max_steps=max_steps)
        except ImportError:
            logger.warning("Could not import game env, using dummy")
            return _DummyEnv(seed=seed)

    return factory


class _DummyEnv:
    """Minimal dummy env for testing the VERL integration."""
    def __init__(self, seed=0):
        self.seed = seed
        self._step = 0

    def reset(self):
        self._step = 0
        return "Game started. You are in a room.", {}

    def step(self, action):
        self._step += 1
        done = self._step >= 5
        return f"Step {self._step}: You did '{action}'.", 0.1 if done else 0.0, done, {"won": done}


def _make_reward_fn(config, tokenizer):
    """Build VERL reward function for Game-AI."""
    from trainer.decision.reward_shaping import GameAIRewardManager

    reward_fn = GameAIRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        normalize_by_length=False,
    )
    val_reward_fn = GameAIRewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        normalize_by_length=False,
    )
    return reward_fn, val_reward_fn


if _HAS_VERL:
    @ray.remote(num_cpus=1)
    class GameAITaskRunner:
        """VERL task runner for Game-AI training.

        Follows the same pattern as verl.trainer.main_ppo.TaskRunner
        but builds Game-AI environments and reward functions, and
        optionally uses GameAITrainer for co-evolution support.
        """

        def run(self, config):
            from pprint import pprint
            from verl.utils.fs import copy_to_local
            from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)

            local_path = copy_to_local(
                config.actor_rollout_ref.model.path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )

            # Build Game-AI envs
            envs, val_envs = _make_gameai_envs(config)

            # Tokenizer
            from verl.utils import hf_processor, hf_tokenizer
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

            # Worker classes
            if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
                from verl.single_controller.ray import RayWorkerGroup
                from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
                actor_rollout_cls = ActorRolloutRefWorker
                ray_worker_group_cls = RayWorkerGroup
            else:
                raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

            from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

            role_worker_mapping = {
                Role.ActorRollout: ray.remote(actor_rollout_cls),
                Role.Critic: ray.remote(CriticWorker),
            }

            global_pool_id = "global_pool"
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
            }

            # Reward model worker (if enabled)
            if config.reward_model.enable:
                if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                    from verl.workers.fsdp_workers import RewardModelWorker
                else:
                    raise NotImplementedError
                role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
                mapping[Role.RewardModel] = global_pool_id

            # Reference policy (for KL)
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
                mapping[Role.RefPolicy] = global_pool_id

            # Reward functions
            reward_fn, val_reward_fn = _make_reward_fn(config, tokenizer)

            resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resource_pool_spec, mapping=mapping,
            )

            # Trajectory collector
            from agent_system.multi_turn_rollout import TrajectoryCollector
            traj_collector = TrajectoryCollector(
                config=config, tokenizer=tokenizer, processor=processor,
            )

            # Datasets
            from verl.utils.dataset.rl_dataset import collate_fn
            train_dataset = _create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
            val_dataset = _create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            train_sampler = _create_rl_sampler(config.data, train_dataset)

            # Choose trainer: GameAITrainer (with co-evolution) or plain RayPPOTrainer
            coevo_callback = None
            if config.get("coevolution", {}).get("enable", False):
                coevo_callback = _build_coevo_callback(config, envs, val_envs)

            if coevo_callback is not None:
                from trainer.decision.grpo_trainer import GameAITrainer
                trainer_cls = GameAITrainer
                extra_kwargs = {"coevo_callback": coevo_callback}
            else:
                from verl.trainer.ppo.ray_trainer import RayPPOTrainer
                trainer_cls = RayPPOTrainer
                extra_kwargs = {}

            trainer = trainer_cls(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
                device_name=config.trainer.device,
                traj_collector=traj_collector,
                envs=envs,
                val_envs=val_envs,
                **extra_kwargs,
            )
            trainer.init_workers()
            trainer.fit()


def _build_coevo_callback(config, envs, val_envs):
    """Build the co-evolution callback from config.

    Initialises the SkillBankCoEvolutionCallback with all skill agent
    stages packed into the tool-calling pipeline.  The callback wraps
    boundary proposal (Stage 1), segmentation decode (Stage 2), contract
    learning (Stage 3), and bank maintenance (Stage 4) so that each stage
    runs inside the training loop and trajectory segmentations are
    persisted and updatable.
    """
    from trainer.decision.coevolution_callback import (
        CoEvolutionConfig,
        SkillBankCoEvolutionCallback,
    )
    from trainer.skillbank.stages.stage1_propose_cuts import ProposeCutsConfig
    from trainer.skillbank.stages.stage2_decode import DecodeConfig
    from trainer.skillbank.stages.stage3_contracts import ContractLearningConfig
    from trainer.skillbank.stages.stage4_update import UpdateConfig

    coevo_cfg = config.coevolution

    propose_cuts_cfg = None
    if coevo_cfg.get("propose_cuts"):
        pc = coevo_cfg.propose_cuts
        propose_cuts_cfg = ProposeCutsConfig(
            min_segment_width=pc.get("min_segment_width", 5),
            merge_radius=pc.get("merge_radius", 5),
            predicate_change_weight=pc.get("predicate_change_weight", 0.5),
            surprisal_weight=pc.get("surprisal_weight", 0.5),
        )

    decode_cfg = None
    if coevo_cfg.get("decode"):
        dc = coevo_cfg.decode
        decode_cfg = DecodeConfig(
            top_m_candidates=dc.get("top_m_candidates", 10),
            segment_min_len=dc.get("segment_min_len", 3),
            new_skill_penalty=dc.get("new_skill_penalty", 5.0),
            eff_freq=dc.get("eff_freq", 0.8),
        )

    contracts_cfg = None
    if coevo_cfg.get("contracts"):
        cc = coevo_cfg.contracts
        contracts_cfg = ContractLearningConfig(
            eff_freq=cc.get("eff_freq", 0.8),
            min_instances_per_skill=cc.get("min_instances_per_skill", 5),
        )

    update_cfg = None
    if coevo_cfg.get("update"):
        uc = coevo_cfg.update
        update_cfg = UpdateConfig(
            min_new_cluster_size=uc.get("min_new_cluster_size", 5),
            merge_jaccard_threshold=uc.get("merge_jaccard_threshold", 0.85),
            refine_delta_threshold=uc.get("refine_delta_threshold", 0.05),
        )

    callback_config = CoEvolutionConfig(
        bank_update_cadence=coevo_cfg.get("bank_update_cadence", 10),
        em_max_iterations=coevo_cfg.get("em_max_iterations", 3),
        min_pass_rate=coevo_cfg.get("min_pass_rate", 0.6),
        max_new_rate=coevo_cfg.get("max_new_rate", 0.3),
        bank_dir=coevo_cfg.get("bank_dir", "runs/skillbank"),
        segmentation_store_path=coevo_cfg.get(
            "segmentation_store_path",
            "runs/skillbank/segmentations.jsonl",
        ),
        max_trajectories_per_update=coevo_cfg.get("max_trajectories_per_update", 256),
        enable_tool_call_reward=coevo_cfg.get("enable_tool_call_reward", True),
        tool_call_reward_weight=coevo_cfg.get("tool_call_reward_weight", 0.1),
        propose_cuts=propose_cuts_cfg,
        decode=decode_cfg,
        contracts=contracts_cfg,
        update=update_cfg,
    )

    initial_bank = None
    bank_path = coevo_cfg.get("initial_bank_path")
    try:
        from skill_agents.skill_bank.bank import SkillBankMVP
        initial_bank = SkillBankMVP()
        if bank_path:
            initial_bank.load(bank_path)
            logger.info("Loaded initial skill bank from %s (%d skills)",
                        bank_path, len(initial_bank))
    except ImportError:
        logger.warning("skill_agents not available; starting with empty bank")

    # Wrap with SkillQueryEngine for richer retrieval during training
    # rollouts — same path used by run_inference.py for both GPT and Qwen.
    query_engine = None
    if initial_bank is not None and len(initial_bank) > 0:
        try:
            from skill_agents.query import SkillQueryEngine
            query_engine = SkillQueryEngine(initial_bank)
            logger.info("SkillQueryEngine initialized for training")
        except (ImportError, Exception) as exc:
            logger.debug("SkillQueryEngine not available: %s", exc)

    return SkillBankCoEvolutionCallback(
        config=callback_config,
        initial_bank=initial_bank,
        envs=envs,
        val_envs=val_envs,
    )


def _create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create an RL dataset (mirrors verl.trainer.main_ppo)."""
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        dataset_cls = RLHFDataset

    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def _create_rl_sampler(data_config, dataset):
    """Create a sampler (mirrors verl.trainer.main_ppo)."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        gen = torch.Generator()
        gen.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=gen)
    return SequentialSampler(data_source=dataset)


def run_verl_training(config) -> None:
    """Launch VERL-based Game-AI training."""
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = GameAITaskRunner.remote()
    ray.get(runner.run.remote(config))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point supporting both standalone and VERL modes.

    If no --config flag, delegates to verl-agent's main_gameai (VERL).
    With --config, runs standalone GRPO.
    """
    if _HAS_VERL and "--config" not in sys.argv:
        # VERL mode: delegate to verl-agent's verl.trainer.main_gameai
        import os
        import subprocess
        _repo_root = Path(__file__).resolve().parent.parent.parent
        _verl_agent_root = _repo_root.parent / "verl-agent"
        if not _verl_agent_root.is_dir():
            logger.error(
                "VERL mode requires verl-agent at %s. Use --config for standalone mode.",
                _verl_agent_root,
            )
            sys.exit(1)
        env = os.environ.copy()
        path_parts = [str(_repo_root), str(_verl_agent_root)]
        if env.get("PYTHONPATH"):
            path_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(path_parts)
        cmd = [sys.executable, "-m", "verl.trainer.main_gameai"] + [
            a for a in sys.argv[1:] if a != "--config"
        ]
        logger.info("Delegating to VERL: %s", " ".join(cmd))
        subprocess.run(cmd, env=env, cwd=str(_repo_root.parent))
        return
    else:
        # Standalone mode
        parser = argparse.ArgumentParser(description="Decision Agent GRPO Training")
        parser.add_argument(
            "--config", type=str,
            default="trainer/common/configs/decision_grpo.yaml",
            help="Path to GRPO config YAML",
        )
        args = parser.parse_args()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

        cfg = load_config(args.config)
        train_decision_agent(cfg)


if __name__ == "__main__":
    main()
