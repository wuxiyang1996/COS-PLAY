"""
DEPRECATED — Use ``trainer.coevolution.orchestrator.co_evolution_loop`` and
``scripts/run_coevolution.py`` instead.  This module is the old synchronous
co-evolution loop kept for backward compatibility only.

Old co-evolution orchestrator: runs Decision Agent GRPO continuously and
periodically updates the SkillBank via Hard-EM.

Training schedule:
  1. Run Decision GRPO continuously
  2. Every bank_update_cadence episodes:
     a. Freeze a rollout batch D_k
     b. Run SkillBank EM trainer → propose Bank_{k+1}
     c. Gate using fixed-seed eval (decision agent frozen during eval)
     d. If accepted, deploy Bank_{k+1} to env wrapper / retriever
     e. Continue GRPO with new bank
  3. Stability: always keep last-good bank for rollback

Usage (legacy):
    python -m trainer.launch_coevolution \\
        --decision-config trainer/common/configs/decision_grpo.yaml \\
        --skillbank-config trainer/common/configs/skillbank_em.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from trainer.common.eval_harness import run_decision_eval
from trainer.common.logging import TrainLogger
from trainer.common.metrics import aggregate_decision_metrics
from trainer.common.seeds import SeedManager, set_global_seed
from trainer.decision.grpo_trainer import GRPOConfig, GRPOTrainer
from trainer.decision.launch_train import build_grpo_config
from trainer.decision.policy_interface import LLMPolicy
from trainer.decision.replay_buffer import ReplayBuffer
from trainer.decision.rollout_collector import collect_batch
from trainer.skillbank.bank_io.bank_store import VersionedBankStore
from trainer.skillbank.bank_io.diff_logger import DiffLogger
from trainer.skillbank.em_trainer import EMConfig, EMTrainer
from trainer.skillbank.ingest_rollouts import ingest_rollouts

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_coevolution(
    decision_cfg: Dict[str, Any],
    skillbank_cfg: Dict[str, Any],
    env_factory: Optional[Callable] = None,
    initial_bank: Optional[Any] = None,
    memory: Optional[Any] = None,
) -> None:
    """Run the full co-evolution training loop.

    Args:
        decision_cfg: parsed decision_grpo.yaml
        skillbank_cfg: parsed skillbank_em.yaml
        env_factory: callable(seed) -> env instance
        initial_bank: starting SkillBankMVP (or None for empty)
        memory: EpisodicMemoryStore (or None)
    """
    set_global_seed(42)

    grpo_cfg = build_grpo_config(decision_cfg)
    model_cfg = decision_cfg.get("model", {})
    rollout_cfg = decision_cfg.get("rollout", {})
    replay_cfg = decision_cfg.get("replay", {})
    eval_cfg = decision_cfg.get("eval", {})
    sched_cfg = decision_cfg.get("schedule", {})
    log_cfg = decision_cfg.get("logging", {})

    policy = LLMPolicy(
        model_name=model_cfg.get("name", "gpt-4o-mini"),
        lr=grpo_cfg.lr,
    )
    buffer = ReplayBuffer(
        capacity=replay_cfg.get("capacity", 10000),
        priority_alpha=replay_cfg.get("priority_alpha", 0.6),
        min_episodes=replay_cfg.get("min_episodes", 64),
    )
    grpo_trainer = GRPOTrainer(policy=policy, config=grpo_cfg, replay_buffer=buffer)

    if initial_bank is None:
        from skill_agents.skill_bank.bank import SkillBankMVP
        initial_bank = SkillBankMVP()

    bank_io_cfg = skillbank_cfg.get("bank_io", {})
    bank_store = VersionedBankStore(
        bank=initial_bank,
        bank_dir=bank_io_cfg.get("bank_dir", "runs/skillbank"),
        snapshot_prefix=bank_io_cfg.get("snapshot_prefix", "bank_v"),
        max_snapshots=bank_io_cfg.get("max_snapshots", 20),
    )

    diff_logger = DiffLogger(
        diff_dir=bank_io_cfg.get("diff_log_path", "runs/skillbank/diffs")
    )

    em_cfg = _build_em_config(skillbank_cfg)
    em_trainer = EMTrainer(
        bank_store=bank_store,
        config=em_cfg,
        diff_logger=diff_logger,
    )

    lora_section = skillbank_cfg.get("lora")
    if lora_section and lora_section.get("adapter_paths"):
        try:
            from skill_agents.lora import MultiLoraSkillBankLLM, MultiLoraConfig
            lora_cfg = MultiLoraConfig.from_dict(lora_section)
            lora_llm = MultiLoraSkillBankLLM(lora_cfg)
            lora_llm.load()
            MultiLoraSkillBankLLM.set_shared_instance(lora_llm)
            logger.info("Multi-LoRA model initialized: %s", lora_llm.loaded_adapters)
        except Exception as exc:
            logger.warning("Multi-LoRA init failed (%s), EM stages will use API fallback", exc)

    seed_manager = SeedManager(
        base_seed=42,
        eval_seeds=eval_cfg.get("seeds", [42, 137, 256, 512]),
    )

    train_logger = TrainLogger(
        log_dir=log_cfg.get("log_dir", "runs/coevolution"),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project="game-ai-coevolution",
    )

    from decision_agents.reward_func import RewardConfig
    costs = decision_cfg.get("costs", {})
    follow = decision_cfg.get("follow_shaping", {})
    reward_config = RewardConfig(
        w_follow=follow.get("w_follow", 0.1),
        query_mem_cost=costs.get("c_mem", -0.05),
        query_skill_cost=costs.get("c_skill", -0.05),
        call_skill_cost=costs.get("c_call", -0.02),
        skill_switch_cost=costs.get("c_switch", -0.10),
    )

    total_episodes = sched_cfg.get("total_episodes", 50000)
    batch_size = rollout_cfg.get("batch_size", 32)
    max_steps = rollout_cfg.get("max_steps", 500)
    bank_update_cadence = sched_cfg.get("bank_update_cadence", 500)
    eval_interval = eval_cfg.get("interval_episodes", 50)

    episode_count = 0
    current_bank = bank_store.current_bank

    logger.info(
        "Starting co-evolution training "
        "(total=%d, bank_update_cadence=%d)",
        total_episodes, bank_update_cadence,
    )
    train_logger.log_event("coevolution_start", {
        "total_episodes": total_episodes,
        "bank_update_cadence": bank_update_cadence,
    })

    while episode_count < total_episodes:
        # ── Phase 1: Collect rollouts with current bank ──
        rollouts = collect_batch(
            env_factory=env_factory,
            policy=policy,
            skill_bank=current_bank,
            memory=memory,
            reward_config=reward_config,
            seed_manager=seed_manager,
            batch_size=batch_size,
            max_steps=max_steps,
        )

        # ── Phase 2: GRPO update ──
        stats = grpo_trainer.train_step(rollouts)
        episode_count += len(rollouts)

        metrics = aggregate_decision_metrics(rollouts)
        train_logger.log_decision_metrics(
            metrics, episode=episode_count,
            extra={"grpo_loss": stats.loss, "bank_version": bank_store.version},
        )

        # ── Phase 3: Periodic evaluation ──
        if episode_count % eval_interval < batch_size:
            eval_result = run_decision_eval(
                env_factory=env_factory,
                agent=None,
                seed_manager=seed_manager,
                num_episodes=eval_cfg.get("num_eval_episodes", 8),
                max_steps=eval_cfg.get("timeout_steps", 1000),
                bank_version=bank_store.version,
            )
            train_logger.log_eval(
                eval_result.metrics,
                episode=episode_count,
                bank_version=bank_store.version,
                seeds_used=eval_result.seeds_used,
            )

        # ── Phase 4: Periodic bank update via EM ──
        if episode_count % bank_update_cadence < batch_size:
            logger.info(
                "=== Bank update at episode %d (current bank v%d) ===",
                episode_count, bank_store.version,
            )
            train_logger.log_event("bank_update_start", {
                "episode": episode_count,
                "bank_version": bank_store.version,
            })

            recent_rollouts = buffer.sample_recent(batch_size * 2)
            trajectories = ingest_rollouts(recent_rollouts)

            em_result = em_trainer.run(trajectories)

            # Protocol update: synthesize/revise protocols from new sub-episodes.
            # Passes the training model so ask_model routes to the correct
            # backend (GPT or Qwen) — same path for both.
            if em_result.accepted:
                try:
                    from skill_agents.pipeline import SkillBankAgent, PipelineConfig
                    _bank = bank_store.current_bank
                    _model_name = model_cfg.get("name")
                    _pipe_cfg = PipelineConfig(
                        llm_model=_model_name,
                        extractor_model=_model_name,
                    )
                    _agent = SkillBankAgent(config=_pipe_cfg, bank=_bank)
                    n_updated = _agent.update_protocols()
                    if n_updated:
                        logger.info("Updated %d protocols after EM round.", n_updated)
                        _bank.save(bank_io_cfg.get("bank_dir", "runs/skillbank") + "/bank.jsonl")
                except Exception as exc:
                    logger.warning("Protocol update failed: %s", exc)

            if em_result.accepted:
                current_bank = bank_store.current_bank
                train_logger.log_event("bank_update_accepted", {
                    "episode": episode_count,
                    "new_version": em_result.bank_version,
                    "n_iterations": len(em_result.iterations),
                })
                if em_result.diff_report:
                    train_logger.log_bank_diff(
                        em_result.bank_version,
                        em_result.diff_report.to_dict(),
                    )

                from trainer.common.metrics import SkillBankMetrics
                sb_metrics = SkillBankMetrics(
                    n_skills=len(getattr(current_bank, "skill_ids", [])),
                    mean_pass_rate=(
                        em_result.skilleval.overall_pass_rate
                        if em_result.skilleval else 0.0
                    ),
                    mean_margin=(
                        em_result.skilleval.mean_margin
                        if em_result.skilleval else 0.0
                    ),
                )
                train_logger.log_skillbank_metrics(
                    sb_metrics, version=em_result.bank_version,
                )

                gate_eval = run_decision_eval(
                    env_factory=env_factory,
                    agent=None,
                    seed_manager=seed_manager,
                    num_episodes=4,
                    max_steps=max_steps,
                    bank_version=em_result.bank_version,
                )
                train_logger.log_eval(
                    gate_eval.metrics,
                    episode=episode_count,
                    bank_version=em_result.bank_version,
                    seeds_used=gate_eval.seeds_used,
                )
            else:
                train_logger.log_event("bank_update_rejected", {
                    "episode": episode_count,
                    "reason": em_result.rejection_reason,
                })
                logger.info("Bank update rejected: %s", em_result.rejection_reason)

    train_logger.log_event("coevolution_complete", {
        "total_episodes": episode_count,
        "final_bank_version": bank_store.version,
    })
    train_logger.close()
    logger.info(
        "Co-evolution complete: %d episodes, bank v%d",
        episode_count, bank_store.version,
    )


def _build_em_config(cfg: Dict[str, Any]) -> EMConfig:
    """Build EMConfig from skillbank YAML config."""
    from trainer.skillbank.stages.stage1_propose_cuts import ProposeCutsConfig
    from trainer.skillbank.stages.stage2_decode import DecodeConfig
    from trainer.skillbank.stages.stage3_contracts import ContractLearningConfig
    from trainer.skillbank.stages.stage4_update import UpdateConfig
    from trainer.skillbank.stages.skilleval import SkillEvalConfig

    em = cfg.get("em", {})
    pc = cfg.get("propose_cuts", {})
    dc = cfg.get("decode", {})
    ct = cfg.get("contracts", {})
    up = cfg.get("update", {})
    gt = cfg.get("gating", {})
    pred = cfg.get("predicates", {})

    return EMConfig(
        max_iterations=em.get("max_iterations", 3),
        convergence_new_rate=em.get("convergence_new_rate", 0.05),
        convergence_margin_std=em.get("convergence_margin_std", 0.5),
        holdout_fraction=em.get("holdout_fraction", 0.1),
        local_redecode=em.get("local_redecode", True),
        propose_cuts=ProposeCutsConfig(
            min_segment_width=pc.get("w", 5),
            merge_radius=pc.get("merge_radius", 5),
            predicate_change_weight=pc.get("predicate_change_weight", 0.5),
            surprisal_weight=pc.get("surprisal_weight", 0.5),
        ),
        decode=DecodeConfig(
            top_m_candidates=dc.get("top_m_candidates", 10),
            segment_min_len=dc.get("segment_min_len", 3),
            segment_max_len=dc.get("segment_max_len", 100),
            new_skill_penalty=dc.get("new_skill_penalty", 5.0),
            eff_freq=dc.get("eff_freq", 0.8),
        ),
        contracts=ContractLearningConfig(
            eff_freq=ct.get("eff_freq", 0.8) if ct.get("eff_freq") else dc.get("eff_freq", 0.8),
            min_instances_per_skill=ct.get("min_instances_per_skill", 5),
            start_end_window=ct.get("start_end_window", 5),
            pass_rate_threshold=ct.get("pass_rate_threshold", 0.6),
        ),
        update=UpdateConfig(
            min_new_cluster_size=up.get("min_new_cluster_size", 5),
            split_pass_rate_threshold=up.get("split_pass_rate_threshold", 0.7),
            child_pass_rate_threshold=up.get("child_pass_rate_threshold", 0.8),
            merge_jaccard_threshold=up.get("merge_jaccard_threshold", 0.85),
            min_child_size=up.get("min_child_size", 3),
            refine_delta_threshold=up.get("refine_delta_threshold", 0.05),
        ),
        skilleval=SkillEvalConfig(
            min_pass_rate=gt.get("min_pass_rate", 0.6),
            min_support=gt.get("min_support", 3),
            max_new_rate=gt.get("max_new_rate", 0.3),
            margin_regression_tol=gt.get("margin_regression_tol", 0.1),
            confusion_threshold=gt.get("confusion_threshold", 0.3),
        ),
        predicate_vocabulary=None,
        p_thresh=pred.get("p_thresh_vision", 0.7),
        smoothing_window=pred.get("smoothing_window", 3),
    )


def main():
    parser = argparse.ArgumentParser(description="Co-Evolution Training")
    parser.add_argument(
        "--decision-config", type=str,
        default="trainer/common/configs/decision_grpo.yaml",
    )
    parser.add_argument(
        "--skillbank-config", type=str,
        default="trainer/common/configs/skillbank_em.yaml",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    decision_cfg = load_config(args.decision_config)
    skillbank_cfg = load_config(args.skillbank_config)

    run_coevolution(decision_cfg, skillbank_cfg)


if __name__ == "__main__":
    main()
