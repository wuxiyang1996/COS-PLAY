#!/usr/bin/env python
"""Launch the co-evolution training loop.

Usage (from Game-AI-Agent root):

    # 1. Start vLLM server (on GPUs 0-3 with 5 LoRA adapters):
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-8B \\
        --tensor-parallel-size 4 \\
        --gpu-memory-utilization 0.90 \\
        --enable-lora \\
        --max-loras 5 \\
        --max-lora-rank 64 \\
        --lora-modules \\
            skill_selection=runs/lora_adapters/decision/skill_selection \\
            action_taking=runs/lora_adapters/decision/action_taking \\
            segment=runs/lora_adapters/skillbank/segment \\
            contract=runs/lora_adapters/skillbank/contract \\
            curator=runs/lora_adapters/skillbank/curator \\
        --enable-prefix-caching \\
        --enable-chunked-prefill \\
        --max-num-seqs 128 \\
        --port 8000

    # 2. Run co-evolution (GRPO on GPUs 4-7):
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    python scripts/run_coevolution.py

    # Or with custom settings:
    python scripts/run_coevolution.py \\
        --total-steps 100 \\
        --episodes-per-game 8 \\
        --checkpoint-interval 5 \\
        --wandb-project game-ai-coevolution \\
        --resume

    # Explicit run directory (otherwise auto-generated from model+timestamp):
    python scripts/run_coevolution.py \\
        --run-dir runs/Qwen3-8B_20260315_143022

    # Specific games only:
    python scripts/run_coevolution.py \\
        --games tetris twenty_forty_eight sokoban \\
        --total-steps 10

    # Resume from specific step:
    python scripts/run_coevolution.py --resume-from-step 25
"""

from __future__ import annotations

import os

# Headless mode: disable display requirements for retro/pyglet/SDL
# before any game-related imports. Ensures training runs without Xvfb.
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# HuggingFace cache — point to /workspace/huggingface so models
# (Qwen3-8B etc.) are not re-downloaded.
os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

# Force the RAG embedding model onto CPU so it does not compete with
# vLLM for GPU memory.  The orchestrator process must never allocate
# CUDA tensors on vLLM GPUs (0-3).
os.environ.setdefault("RAG_EMBEDDER_DEVICE", "cpu")

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent

for p in [
    str(CODEBASE_ROOT),
    str(CODEBASE_ROOT.parent / "GamingAgent"),
    str(CODEBASE_ROOT.parent / "AgentEvolver"),
    str(CODEBASE_ROOT.parent / "AI_Diplomacy"),
    str(CODEBASE_ROOT.parent / "Orak"),
]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from trainer.coevolution.config import (
    CoEvolutionConfig,
    CURRICULUM_PRESETS,
    GAME_MAX_STEPS,
    SKILL_BANK_GAMES,
    EVAL_ONLY_GAMES,
)
from trainer.coevolution.orchestrator import co_evolution_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Co-Evolution Training: Decision Agent + Skill Bank Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument(
        "--total-steps", type=int, default=60,
        help="Total co-evolution steps (default: 60)",
    )
    parser.add_argument(
        "--games", nargs="+", default=None,
        help=f"Games to train on (default: {len(SKILL_BANK_GAMES)} skill-bank games; {len(GAME_MAX_STEPS)} total supported)",
    )
    parser.add_argument(
        "--curriculum", type=str, default="focused",
        choices=list(CURRICULUM_PRESETS.keys()),
        help="Curriculum preset: 'focused' = 4 games then Avalon+Diplomacy, "
             "'gradual' = incrementally add games, "
             "'none' = all games from step 0 (default: focused)",
    )
    parser.add_argument(
        "--episodes-per-game", type=int, default=8,
        help="Episodes per game per step (default: 8)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64,
        help="Max concurrent episodes (default: 64)",
    )
    parser.add_argument(
        "--unified-roles", action="store_true",
        help="Enable unified multi-role rollouts for Avalon/Diplomacy. "
             "Deterministically cycles through all roles instead of random "
             "assignment. Skill banks split by side/power.",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B",
        help="Base model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max generation tokens (default: 512)",
    )

    # GPU allocation (split: vLLM on some GPUs, GRPO on others)
    parser.add_argument(
        "--vllm-gpus", nargs="+", type=int, default=[0, 1, 2, 3],
        help="GPU IDs for persistent vLLM inference servers (TP=1 each). "
             "Default: 0 1 2 3",
    )
    parser.add_argument(
        "--grpo-devices", nargs="+", type=int, default=[4, 5, 6, 7],
        help="GPU devices for GRPO FSDP training. Default: 4 5 6 7",
    )
    parser.add_argument(
        "--no-manage-vllm", action="store_true",
        help="Disable managed vLLM lifecycle. Use when running vLLM "
             "externally. In this mode, --vllm-url controls the endpoint.",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM server URL (only used with --no-manage-vllm). "
             "Default: http://localhost:8000/v1",
    )
    parser.add_argument(
        "--vllm-base-port", type=int, default=8000,
        help="Base port for managed vLLM instances (default: 8000). "
             "Instance N runs on port 8000+N.",
    )
    parser.add_argument(
        "--vllm-gpu-util", type=float, default=0.90,
        help="GPU memory utilization for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--speculative-model", type=str, default="Qwen/Qwen3-0.6B",
        help="Draft model for speculative decoding (default: Qwen/Qwen3-0.6B). "
             "Set to empty string to disable.",
    )
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=5,
        help="Number of tokens the draft model proposes per step (default: 5)",
    )

    # External opponent (Avalon / Diplomacy)
    parser.add_argument(
        "--opponent-model", type=str, default=None,
        help="External API model for opponents in Avalon/Diplomacy "
             "(e.g. gpt-5-mini). When set, non-controlled players use "
             "this model via API instead of vLLM self-play.",
    )
    parser.add_argument(
        "--opponent-api-base", type=str, default=None,
        help="Base URL for opponent API (default: https://openrouter.ai/api/v1)",
    )

    # GRPO
    parser.add_argument(
        "--no-grpo", action="store_true",
        help="Disable GRPO training (rollout + skill bank only)",
    )
    parser.add_argument(
        "--grpo-lr", type=float, default=None,
        help="Override GRPO steady-state learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--grpo-kl-coeff", type=float, default=None,
        help="Override GRPO steady-state KL coefficient (default: 0.05)",
    )
    parser.add_argument(
        "--grpo-clip-ratio", type=float, default=None,
        help="Override GRPO PPO clip ratio (default: 0.2)",
    )
    parser.add_argument(
        "--grpo-max-epochs", type=int, default=None,
        help="Max GRPO epochs per adapter per step (default: 4)",
    )
    parser.add_argument(
        "--grpo-adv-clip", type=float, default=None,
        help="Clip GRPO advantages to [-val, val] to limit outlier influence",
    )
    parser.add_argument(
        "--lora-r", type=int, default=None,
        help="Override LoRA rank for all adapters (default: 16). "
             "Used by the capacity-matched ablation runs.",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=None,
        help="Override LoRA alpha for all adapters (default: 32). "
             "Typical setting: alpha = 2 * r.",
    )
    parser.add_argument(
        "--no-skillbank", action="store_true",
        help="Ablation: disable the skill bank and skill bank agent. "
             "Rollouts run in cold-start mode (no skill retrieval), "
             "Phase B (skill discovery / bank update) is skipped, and "
             "GRPO only updates the 'action_taking' adapter — "
             "skill_selection, segment, contract, curator stay frozen. "
             "Corresponds to the paper's 'GRPO w/o skill' ablation.",
    )
    parser.add_argument(
        "--freeze-skillbank", action="store_true",
        help="Ablation: keep the skill bank ACTIVE for retrieval during "
             "rollouts but freeze its growth.  Phase B (skill discovery / "
             "bank update) is skipped, and GRPO trains BOTH 'action_taking' "
             "and 'skill_selection' (the full decision agent) — segment, "
             "contract, curator stay frozen.  Corresponds to the paper's "
             "'GRPO w/ fixed skill bank' ablation.  "
             "Mutually exclusive with --no-skillbank.",
    )

    # Training schedule
    parser.add_argument(
        "--warmup-steps", type=int, default=None,
        help="Number of warmup steps for LR/temperature/KL ramp (default: 20)",
    )
    parser.add_argument(
        "--initial-kl-coeff", type=float, default=None,
        help="KL coefficient at start of warmup (default: 0.01)",
    )
    parser.add_argument(
        "--initial-temperature", type=float, default=None,
        help="Sampling temperature at start of warmup (default: 1.0)",
    )
    parser.add_argument(
        "--steady-temperature", type=float, default=None,
        help="Sampling temperature after warmup (default: 0.7)",
    )

    # Episode control
    parser.add_argument(
        "--stuck-window", type=int, default=None,
        help="Rolling window size for stuck detection (default: 15)",
    )
    parser.add_argument(
        "--min-steps-before-stuck", type=int, default=None,
        help="Min steps before stuck detection activates (default: 20)",
    )

    # Directories
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Root run directory (default: auto-generated from model name + timestamp)",
    )
    parser.add_argument(
        "--bank-dir", type=str, default=None,
        help="Skill bank directory (default: <run-dir>/skillbank)",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default=None,
        help="LoRA adapter directory (default: <run-dir>/lora_adapters)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory (default: <run-dir>/checkpoints)",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Log directory (default: <run-dir>)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-interval", type=int, default=5,
        help="Save checkpoint every N steps (default: 5)",
    )

    # Start mode: from-scratch vs resume (mutually exclusive)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--from-scratch", action="store_true",
        help="Train from scratch: random-init all 5 LoRA adapters with "
             "gaussian weights and ignore any existing checkpoints",
    )
    start_group.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint (fail if none exists)",
    )
    parser.add_argument(
        "--resume-from-step", type=int, default=None,
        help="Resume from a specific checkpoint step (implies --resume)",
    )

    # Pre-trained adapter loading
    parser.add_argument(
        "--load-adapters-from", type=str, default=None, metavar="DIR",
        help="Load pre-trained LoRA adapters from DIR (expects sub-dirs: "
             "skill_selection, action_taking, segment, contract, curator). "
             "Missing adapters will be random-initialised.",
    )
    parser.add_argument(
        "--load-decision-adapters", type=str, default=None, metavar="DIR",
        help="Load only the 2 decision agent adapters (skill_selection, "
             "action_taking) from DIR. Skill bank adapters are random-init.",
    )
    parser.add_argument(
        "--load-skillbank-adapters", type=str, default=None, metavar="DIR",
        help="Load only the 3 skill bank adapters (segment, contract, "
             "curator) from DIR. Decision adapters are random-init.",
    )

    # W&B
    parser.add_argument(
        "--wandb-project", type=str, default="game-ai-coevolution",
        help="W&B project name (default: game-ai-coevolution)",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging",
    )

    # Workers
    parser.add_argument(
        "--thread-workers", type=int, default=64,
        help="Thread pool size (default: 64)",
    )
    parser.add_argument(
        "--process-workers", type=int, default=8,
        help="Process pool size (default: 8)",
    )

    # Skill bank seeding
    parser.add_argument(
        "--seed-bank-dir", type=str, default=None, metavar="DIR",
        help="Seed empty per-game skill banks from DIR on first launch. "
             "Expected layout: DIR/<game>/skill_bank.jsonl. "
             "Skills are only copied when the game's bank is empty.",
    )

    # Debug
    parser.add_argument(
        "--debug-io", action="store_true",
        help="Log every LLM I/O and GRPO sample to <run-dir>/debug_io/ "
             "for debugging truncation and prompt/completion inspection",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_known_games = set(GAME_MAX_STEPS)
    games = args.games if args.games else list(SKILL_BANK_GAMES)
    for g in games:
        if g not in all_known_games:
            logging.warning("Unknown game '%s', skipping", g)
    games = [g for g in games if g in all_known_games]

    if not games:
        logging.error("No valid games specified")
        sys.exit(1)

    if args.no_skillbank and args.freeze_skillbank:
        logging.error("--no-skillbank and --freeze-skillbank are mutually exclusive")
        sys.exit(1)

    # Determine start mode
    if args.from_scratch:
        start_mode = "from_scratch"
    elif args.resume or args.resume_from_step is not None:
        start_mode = "resume"
    else:
        start_mode = "auto"

    # Build pretrained_adapter_paths from CLI flags
    pretrained: Dict[str, str] = {}
    decision_names = ["skill_selection", "action_taking"]
    skillbank_names = ["segment", "contract", "curator"]

    if args.load_adapters_from:
        src = Path(args.load_adapters_from)
        for name in decision_names + skillbank_names:
            p = src / name
            if p.exists():
                pretrained[name] = str(p)
    if args.load_decision_adapters:
        src = Path(args.load_decision_adapters)
        for name in decision_names:
            p = src / name
            if p.exists():
                pretrained[name] = str(p)
    if args.load_skillbank_adapters:
        src = Path(args.load_skillbank_adapters)
        for name in skillbank_names:
            p = src / name
            if p.exists():
                pretrained[name] = str(p)

    manage_vllm = not args.no_manage_vllm

    curriculum = CURRICULUM_PRESETS[args.curriculum]

    config_kwargs = dict(
        games=games,
        episodes_per_game=args.episodes_per_game,
        unified_role_rollouts=args.unified_roles,
        max_concurrent_episodes=args.max_concurrent,
        total_steps=args.total_steps,
        curriculum_schedule=dict(curriculum) if curriculum else None,
        vllm_gpu_ids=args.vllm_gpus,
        grpo_devices=args.grpo_devices,
        manage_vllm=manage_vllm,
        vllm_base_url=args.vllm_url,
        vllm_base_port=args.vllm_base_port,
        vllm_gpu_util=args.vllm_gpu_util,
        speculative_model=args.speculative_model or None,
        num_speculative_tokens=args.num_speculative_tokens,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        grpo_enabled=not args.no_grpo,
        disable_skillbank=args.no_skillbank,
        freeze_skillbank=args.freeze_skillbank,
        checkpoint_interval=args.checkpoint_interval,
        wandb_enabled=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        start_mode=start_mode,
        resume_from_step=args.resume_from_step,
        pretrained_adapter_paths=pretrained,
        thread_workers=args.thread_workers,
        process_workers=args.process_workers,
        debug_io=args.debug_io,
    )

    if args.grpo_lr is not None:
        config_kwargs["scratch_steady_lr"] = args.grpo_lr
        config_kwargs["scratch_initial_lr"] = args.grpo_lr * 2.0
    if args.grpo_kl_coeff is not None:
        config_kwargs["scratch_steady_kl_coeff"] = args.grpo_kl_coeff
    if args.grpo_clip_ratio is not None:
        config_kwargs["grpo_clip_ratio"] = args.grpo_clip_ratio
    if args.grpo_max_epochs is not None:
        config_kwargs["grpo_max_epochs"] = args.grpo_max_epochs
    if args.lora_r is not None:
        config_kwargs["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        config_kwargs["lora_alpha"] = args.lora_alpha
    if args.grpo_adv_clip is not None:
        config_kwargs["grpo_adv_clip"] = args.grpo_adv_clip

    if args.warmup_steps is not None:
        config_kwargs["scratch_warmup_steps"] = args.warmup_steps
    if args.initial_kl_coeff is not None:
        config_kwargs["scratch_initial_kl_coeff"] = args.initial_kl_coeff
    if args.initial_temperature is not None:
        config_kwargs["scratch_initial_temperature"] = args.initial_temperature
    if args.steady_temperature is not None:
        config_kwargs["scratch_steady_temperature"] = args.steady_temperature
    if args.stuck_window is not None:
        config_kwargs["stuck_window"] = args.stuck_window
    if args.min_steps_before_stuck is not None:
        config_kwargs["min_steps_before_stuck_check"] = args.min_steps_before_stuck

    if args.unified_roles:
        config_kwargs["episodes_per_game_overrides"] = {
            g: args.episodes_per_game for g in games
        }

    if args.opponent_model is not None:
        config_kwargs["opponent_model"] = args.opponent_model
    if args.opponent_api_base is not None:
        config_kwargs["opponent_api_base"] = args.opponent_api_base

    if args.seed_bank_dir is not None:
        config_kwargs["seed_bank_dir"] = args.seed_bank_dir

    if args.run_dir is not None:
        config_kwargs["run_dir"] = args.run_dir
    if args.bank_dir is not None:
        config_kwargs["bank_dir"] = args.bank_dir
    if args.adapter_dir is not None:
        config_kwargs["adapter_dir"] = args.adapter_dir
    if args.checkpoint_dir is not None:
        config_kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        config_kwargs["log_dir"] = args.log_dir

    config = CoEvolutionConfig(**config_kwargs)
    config.resolve_paths()

    # Set up logging after paths are resolved
    log_file = Path(config.log_dir) / "coevolution.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CO-EVOLUTION TRAINING")
    print("=" * 70)
    print(f"  Run dir:      {config.run_dir}")
    print(f"  Games:        {', '.join(games)}")
    print(f"  Steps:        {config.total_steps}")
    print(f"  Eps/game:     {config.episodes_per_game}")
    print(f"  Concurrent:   {config.max_concurrent_episodes}")
    print(f"  Model:        {config.model_name}")
    if config.manage_vllm:
        print(f"  vLLM GPUs:    {config.vllm_gpu_ids} — "
              f"{len(config.vllm_gpu_ids)} × TP=1 (persistent, "
              f"ports {config.vllm_base_port}–"
              f"{config.vllm_base_port + len(config.vllm_gpu_ids) - 1})")
    else:
        print(f"  vLLM:         EXTERNAL — {config.vllm_base_url}")
    _opp = getattr(config, "opponent_model", None)
    if _opp:
        _opp_base = getattr(config, "opponent_api_base", "openrouter")
        print(f"  Opponent:     {_opp} ({_opp_base})")
    print(f"  GRPO:         {'enabled' if config.grpo_enabled else 'disabled'}")
    if config.grpo_enabled:
        print(f"    FSDP GPUs:  {config.effective_grpo_devices}")
    if getattr(config, "disable_skillbank", False):
        print("  Skill bank:   DISABLED (ablation: action_taking adapter only)")
    elif getattr(config, "freeze_skillbank", False):
        print("  Skill bank:   FROZEN (ablation: decision agent trained, bank LoRAs frozen)")
    print(f"  Bank dir:     {config.bank_dir}")
    print(f"  Adapter dir:  {config.adapter_dir}")
    print(f"  Checkpoint:   every {config.checkpoint_interval} steps → {config.checkpoint_dir}")
    print(f"  GRPO data:    {config.grpo_data_dir}")
    print(f"  Rewards:      {config.rewards_dir}")
    print(f"  TensorBoard:  {config.tensorboard_dir}")
    print(f"  Log dir:      {config.log_dir}")
    print(f"  W&B:          {'enabled' if config.wandb_enabled else 'disabled'}")
    print(f"  Debug I/O:    {'enabled → ' + config.debug_io_dir if config.debug_io else 'disabled'}")
    print(f"  Curriculum:   {config.curriculum_description()}")
    if config.start_mode == "from_scratch":
        print("  Start mode:   FROM SCRATCH (gaussian LoRA init, no checkpoint)")
        print(f"    Warmup:     {config.scratch_warmup_steps} steps "
              f"(lr {config.scratch_initial_lr:.0e}→{config.scratch_steady_lr:.0e}, "
              f"temp {config.scratch_initial_temperature}→{config.scratch_steady_temperature})")
    elif config.start_mode == "resume":
        if config.resume_from_step is not None:
            print(f"  Start mode:   RESUME from step {config.resume_from_step}")
        else:
            print("  Start mode:   RESUME from latest checkpoint")
    else:
        print("  Start mode:   AUTO (resume if checkpoint exists, else fresh)")
    if pretrained:
        print(f"  Pre-trained:  {list(pretrained.keys())}")
    print("=" * 70)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

    asyncio.run(co_evolution_loop(config))


if __name__ == "__main__":
    main()
