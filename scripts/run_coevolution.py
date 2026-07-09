#!/usr/bin/env python
"""Launch the co-evolution training loop.

Usage (from Game-AI-Agent root):

    # 1. Managed mode (default): the orchestrator launches one TP=1
    #    vLLM-serve instance per --vllm-gpus entry with the right
    #    Qwen3.5 flags (--language-model-only, --reasoning-parser qwen3,
    #    --speculative_config '{"method":"mtp","num_speculative_tokens":1}',
    #    plus all 5 LoRA adapters).  Skip step 1 entirely.
    #
    #    External / unmanaged mode:
    #    python -m vllm.entrypoints.openai.api_server \\
    #        --model Qwen/Qwen3.5-9B \\
    #        --tensor-parallel-size 4 \\
    #        --gpu-memory-utilization 0.90 \\
    #        --enable-lora --max-loras 5 --max-lora-rank 64 \\
    #        --language-model-only --reasoning-parser qwen3 \\
    #        --speculative_config '{"method":"mtp","num_speculative_tokens":1}' \\
    #        --lora-modules \\
    #            skill_selection=runs/lora_adapters/decision/skill_selection \\
    #            action_taking=runs/lora_adapters/decision/action_taking \\
    #            segment=runs/lora_adapters/skillbank/segment \\
    #            contract=runs/lora_adapters/skillbank/contract \\
    #            curator=runs/lora_adapters/skillbank/curator \\
    #        --enable-prefix-caching --enable-chunked-prefill \\
    #        --max-num-seqs 128 --port 8000

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
        --run-dir runs/Qwen3.5-9B_20260427_131800

    # Specific games only:
    python scripts/run_coevolution.py \\
        --games tetris twenty_forty_eight candy_crush \\
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
# (Qwen3.5-9B etc.) are not re-downloaded.
os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

# Force the RAG embedding model onto CPU so it does not compete with
# vLLM for GPU memory.  The orchestrator process must never allocate
# CUDA tensors on vLLM GPUs (0-3).
os.environ.setdefault("RAG_EMBEDDER_DEVICE", "cpu")

import argparse
import asyncio
import json
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
        help=(
            f"Games to train on (default: full 12-game registry — "
            f"{len(SKILL_BANK_GAMES)} games covering Phase-1 source + Phase-2 hold-out; "
            f"{len(GAME_MAX_STEPS)} total supported). "
            "For Phase-1-only training prefer "
            "'--games tetris candy_crush gymv_thunder_force_iii gymv_altered_beast "
            "gymv_columns gymv_dynamite_headdy' or use scripts/run_phase1_curriculum.sh; "
            "for single-game smoke tests, pass an explicit '--games <slug>'."
        ),
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
        help="Global default episodes per game per step (default: 8). "
             "Per-game overrides apply on top — see "
             "trainer.coevolution.config.HIGH_VARIANCE_GYMV_EPISODES "
             "for sparse-reward gymv games which default to 16. "
             "Use --episodes-per-game-overrides to customize.",
    )
    parser.add_argument(
        "--episodes-per-game-overrides", type=str, default=None,
        help="JSON map of per-game episode overrides, e.g. "
             '\'{"gymv_thunder_force_iii": 24, "tetris": 8}\'. '
             "Merged on top of the built-in HIGH_VARIANCE_GYMV_EPISODES "
             "defaults; pass an empty dict '{}' to keep only the "
             "global --episodes-per-game value across every game.",
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
        "--model", type=str, default="Qwen/Qwen3.5-9B",
        help="Base model name (default: Qwen/Qwen3.5-9B). The model that "
             "vLLM serves AND that GRPO LoRA-tunes. For inference-only "
             "Qwen3.5-35B-A3B see inference/serve_qwen35_35b_a3b.sh.",
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
        "--speculative-method", type=str, default="mtp",
        choices=["mtp", "draft_model", "none"],
        help="Speculative decoding method (default: mtp). "
             "'mtp' uses the model's built-in multi-token-prediction head "
             "(Qwen3.5 / Qwen3-Next / DeepSeek-V3); 'draft_model' uses "
             "an external small model passed via --speculative-model; "
             "'none' disables speculative decoding.",
    )
    parser.add_argument(
        "--speculative-model", type=str, default="",
        help="Draft model for --speculative-method=draft_model "
             "(e.g. Qwen/Qwen3-0.6B for plain Qwen3). Ignored when "
             "method=mtp. Default: empty (no external drafter).",
    )
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=1,
        help="Number of tokens the drafter proposes per step. "
             "Default: 1 (good for MTP); 4–5 for draft_model.",
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

    # Skill bank storage layout (per_game default, shared opt-in for the
    # cross-game / lifelong-learning experiments described in
    # ``training_notes/coevo-3phase-cross-game-ood-transfer-plan.md``).
    parser.add_argument(
        "--bank-mode", choices=("per_game", "shared"), default="per_game",
        help="Skill-bank storage layout. 'per_game' (default, legacy) "
             "writes one skill_bank.jsonl per game under "
             "<bank_dir>/<game>/skill_bank.jsonl. 'shared' writes one "
             "<bank_dir>/skill_bank.jsonl across all games and relies "
             "on the harness's task-axis veto (feasible_tasks, "
             "harness/README §22) for per-game eligibility. Pair "
             "'shared' with --seed-bank-dir and the per-phase "
             "translation step in scripts/run_phase1_curriculum.sh.",
    )

    # Phase B′: Crafter + Promotion (one-way writeback to legacy bank)
    parser.add_argument(
        "--crafter-promotion-enabled", action="store_true",
        help="Enable Phase B′: per-step Crafter (reflect on episodes → "
             "BankMutationProposals) + offline-synthetic Promotion driver "
             "(decide_promotion_gpt54.py) that writes promoted skills back "
             "into the live skill_bank.jsonl. Off by default. See "
             "implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md.",
    )
    parser.add_argument(
        "--crafter-cycle-every-k-steps", type=int, default=0,
        help="If >0, only run the Crafter every K steps "
             "(otherwise: every step). Default 0 = every step.",
    )
    parser.add_argument(
        "--crafter-outcome-failure-threshold", type=float, default=0.0,
        help="Per-episode reward below this threshold is treated as "
             "OUTCOME_FAILURE for Crafter failure synthesis. Default 0.0.",
    )
    parser.add_argument(
        "--crafter-promotion-timeout-s", type=float, default=300.0,
        help="Subprocess timeout for the offline-synthetic promotion "
             "driver (decide_promotion_gpt54.py). Default 300s.",
    )
    parser.add_argument(
        "--crafter-promotion-gate-mode",
        choices=["offline-synthetic", "offline-with-llm-judge", "live"],
        default="offline-synthetic",
        help="Forwarded to decide_promotion_gpt54.py --gate-mode. "
             "'offline-synthetic' (default): no LLM calls, all proposals "
             "land at LIMITED_PASS ⇒ PROVISIONAL. "
             "'offline-with-llm-judge': adds one BACKBONE_JUDGE_MODEL "
             "(35B-A3B) call per proposal so visibly bad proposals can "
             "be FAILed and rejected — routed via VLLM_BASE_URL_MAP. "
             "'live': calls GateService.evaluate end-to-end (diagnostic).",
    )
    # Hypothesizer fallthrough gate (post-v11 audit).  Both gates default
    # to active; lower these knobs if a workload genuinely benefits from
    # more aggressive new-skill minting.  See CoEvolutionConfig docstring
    # for the full rationale.
    parser.add_argument(
        "--crafter-hypothesize-min-recurrences", type=int, default=3,
        help="Minimum FailureMemory recurrence count before the Crafter "
             "dispatch chain falls through to the Hypothesizer "
             "(default 3 — same as hot_pattern_threshold).  Set to 1 "
             "to reproduce the pre-v11 behaviour where any single "
             "orphan failure could mint a new skill.",
    )
    parser.add_argument(
        "--crafter-hypothesize-related-skill-jaccard", type=float, default=0.30,
        help="Jaccard threshold (0..1) above which an existing bank "
             "skill is judged 'related' to the failure context, "
             "blocking the Hypothesizer fallthrough (so Patch is "
             "preferred). Default 0.30. Set to 0.0 to disable the "
             "relatedness gate (recurrence gate stays).",
    )

    # Harness wire-up (Day-10): hook the harness's eligibility +
    # validate_invocation surfaces into Phase A, and drain the rejection
    # sink into Phase B′ Crafter. See harness/README.md §22.
    parser.add_argument(
        "--harness-enabled", action="store_true",
        help="Enable Phase-A harness integration: pre-LLM "
             "select_eligible_skills filter + post-LLM "
             "validate_invocation veto, with rejection sink drained "
             "into the Crafter hook's lifecycle. Adds 0 LLM calls. "
             "Off by default. See harness/README.md §22.",
    )
    parser.add_argument(
        "--no-harness-allow-shadow", dest="harness_allow_shadow",
        action="store_false",
        help="Refuse to admit SHADOW skills via the harness eligibility "
             "filter. Default: SHADOW skills are admitted (matches "
             "HarnessConfig.allow_shadow=True).",
    )
    parser.set_defaults(harness_allow_shadow=True)

    # ── Block B — §5.5 ablation flags ──────────────────────────────────
    # Each flag is independently switchable; default values reproduce
    # the production SkillBridge behaviour.  Combine them to reproduce
    # the §5.5 ablation table without rebuilding the trainer image.
    parser.add_argument(
        "--harness-mode",
        choices=["full", "plain-text-skills", "off"],
        default="full",
        help="Block B1: harness ablation. 'full' (default) keeps "
             "eligibility filter + validate_invocation. "
             "'plain-text-skills' bypasses both — actor sees raw "
             "skill bank content as plain-text few-shot. 'off' "
             "drops candidates entirely (cold-start mode). Only "
             "consulted when --harness-enabled is set.",
    )
    parser.add_argument(
        "--no-crafter", dest="crafter_enabled", action="store_false",
        help="Block B2: w/o crafter ablation. Skips the Crafter step "
             "entirely (no patches, no LLM crafter, no hypotheses, "
             "no retire). Promotion + lifecycle still run on top of "
             "any pre-existing draft skills.",
    )
    parser.set_defaults(crafter_enabled=True)
    parser.add_argument(
        "--promotion-bypass-mode",
        choices=["gated", "permissive"],
        default="gated",
        help="Block B3: promotion gate ablation. 'gated' (default) "
             "routes through the GateService driver. 'permissive' "
             "auto-PASSes every proposal (DRAFT → ACTIVE) with no "
             "judge call, isolating the gate's contribution.",
    )
    parser.add_argument(
        "--intention-trigger",
        choices=["every-step", "sharp-shift", "disabled"],
        default="every-step",
        help="Block B4: intention loop ablation. 'every-step' "
             "(default, historical) regenerates intention LLM every "
             "inner step. 'sharp-shift' fires only on detected state "
             "delta or urgency. 'disabled' generates once at step 0 "
             "and reuses for the rest of the episode.",
    )
    parser.add_argument(
        "--actor-bank-cap-k", type=int, default=0,
        help="Block B5: cap the actor's view of the bank at top-K "
             "skills (0 = no cap, historical default). Used for the "
             "bank-size sweep K ∈ {10, 50, 200}. Implemented as a "
             "retrieval-side filter on SkillQueryEngine.select(); "
             "does not truncate the on-disk bank file.",
    )

    # Phase-start GameProfile (Path 1).  When enabled, the orchestrator
    # fires one BACKBONE_JUDGE_MODEL (35B) call per game per phase
    # boundary, parses a compact GameProfile + <state> exemplar, and
    # prepends the compact profile to the actor's SYSTEM_PROMPT and
    # SKILL_SELECTION_SYSTEM_PROMPT for the duration of the phase.
    # See trainer/coevolution/_game_schema.py.
    parser.add_argument(
        "--game-schema-enabled", action="store_true",
        help="Enable phase-start GameProfile generation (Path 1): "
             "1 × BACKBONE_JUDGE_MODEL (35B) call per game per phase "
             "produces a compact goal/win_signal/hazards/key_actions/"
             "failure_modes profile that gets injected into the "
             "actor's SYSTEM_PROMPT, plus a cached <state> exemplar "
             "for Path 2 / Path 4 to reuse. Off by default.",
    )
    parser.add_argument(
        "--game-schema-model", type=str, default="",
        help="Override model for game-schema 35B calls. Empty (default) "
             "→ BACKBONE_JUDGE_MODEL via VLLM_BASE_URL_MAP (same routing "
             "as the LLM promotion judge).",
    )
    parser.add_argument(
        "--game-schema-max-tokens", type=int, default=1024,
        help="Token budget for the 35B GameProfile response (default "
             "1024 — fits compact + <state> exemplar comfortably).",
    )
    parser.add_argument(
        "--game-schema-timeout-s", type=float, default=60.0,
        help="Hard timeout per 35B GameProfile call. On timeout we "
             "fall back to a deterministic minimum profile and "
             "continue without blocking the phase boundary.",
    )

    # Path 2 — supplemental LLM Crafter (35B-A3B teacher).
    parser.add_argument(
        "--llm-crafter-enabled", action="store_true",
        help="Enable Path 2 LLM Crafter: in addition to the rule-based "
             "Crafter, fire up to --llm-crafter-k-max parallel 35B "
             "calls per game per step (one per FailureTrace) to "
             "propose patch / hypothesize / retire BankMutationProposals. "
             "Routes through API_func.ask_model → BACKBONE_JUDGE_MODEL "
             "via VLLM_BASE_URL_MAP. Off by default.",
    )
    parser.add_argument(
        "--llm-crafter-model", type=str, default="",
        help="Override model for LLM Crafter 35B calls. Empty (default) "
             "→ BACKBONE_JUDGE_MODEL via VLLM_BASE_URL_MAP.",
    )
    parser.add_argument(
        "--llm-crafter-k-max", type=int, default=2,
        help="Hard cap on parallel LLM Crafter calls per game per step "
             "(default 2 post-v11; was 5 in v11 but the rewritten "
             "last-resort prompt + recurrence/relatedness gates make a "
             "smaller cap sufficient).",
    )
    parser.add_argument(
        "--llm-crafter-max-tokens", type=int, default=1024,
        help="Token budget per LLM Crafter response (default 1024).",
    )
    parser.add_argument(
        "--llm-crafter-temperature", type=float, default=0.3,
        help="Sampling temperature for LLM Crafter calls (default 0.3).",
    )
    parser.add_argument(
        "--llm-crafter-timeout-s", type=float, default=60.0,
        help="Hard timeout per LLM Crafter call. On timeout the trace "
             "is dropped and the deterministic proposal stream continues.",
    )
    parser.add_argument(
        "--llm-crafter-enable-thinking", action="store_true",
        help="Stage 2 (cross-domain adaptation) opt-in: forward "
             "enable_thinking=True into the 35B Crafter ask_model "
             "calls so Qwen3-A3B emits its <think> chain-of-thought "
             "before the JSON proposal. EXPERIMENTAL — observed "
             "Crafter prompts to consume >16K tokens of reasoning "
             "without emitting a final answer; you should pair this "
             "with --llm-crafter-max-tokens 16384+ AND a prompt-side "
             "'think briefly' constraint. --llm-crafter-timeout-s "
             "should also rise to >=180 s. Stage-1 in-domain "
             "training (run_phase1_curriculum.sh) keeps this OFF.",
    )

    # Path 3 — Promotion judge (35B-A3B teacher; only fires when
    # ``--crafter-promotion-gate-mode offline-with-llm-judge``).
    parser.add_argument(
        "--crafter-promotion-judge-enable-thinking",
        action="store_true",
        help="Stage 2 cross-domain opt-in for the LLM judge inside "
             "decide_promotion_gpt54.py. Forwards --enable-thinking "
             "to the driver, which threads through to "
             "_llm_skill_judge.judge_proposal's ask_model call. Pair "
             "with --crafter-promotion-judge-max-tokens 8192+ — "
             "live 35B-A3B observed to spend ~5K tokens on the "
             "<think> block before emitting the ~120-char JSON "
             "verdict; 4K truncates, 8K+ completes in ~25-40 s. "
             "No effect for any other gate mode (the driver simply "
             "ignores the flag).",
    )
    parser.add_argument(
        "--crafter-promotion-judge-max-tokens", type=int, default=256,
        help="Token budget per llm-judge response. Default 256 fits a "
             "tight JSON verdict; bump to 8192+ when --crafter-"
             "promotion-judge-enable-thinking is set (observed 5K-"
             "token <think> blocks before content emission).",
    )

    # Path 4 — LLM Harness validator (35B-A3B teacher).
    parser.add_argument(
        "--llm-harness-validator-enabled", action="store_true",
        help="Enable Path 4 LLM Harness validator: after the "
             "deterministic SkillHarness.validate_invocation admits a "
             "skill, optionally run a 35B post-validation pass. Hybrid "
             "policy: bootstrap window (--llm-harness-bootstrap-steps) "
             "always fires; afterwards only fires on uncertain cases "
             "(SHADOW status, no can_handle evidence, translation-"
             "rewritten contracts). Verdicts can ONE-WAY downgrade "
             "admit→veto. Off by default.",
    )
    parser.add_argument(
        "--llm-harness-model", type=str, default="",
        help="Override model for LLM Harness validator. Empty (default) "
             "→ BACKBONE_JUDGE_MODEL via VLLM_BASE_URL_MAP.",
    )
    parser.add_argument(
        "--llm-harness-bootstrap-steps", type=int, default=20,
        help="Trainer steps below this fire the LLM validator on EVERY "
             "admitted skill regardless of deterministic certainty "
             "(default 20).",
    )
    parser.add_argument(
        "--llm-harness-max-tokens", type=int, default=256,
        help="Token budget per LLM Harness validator response "
             "(default 256).",
    )
    parser.add_argument(
        "--llm-harness-temperature", type=float, default=0.2,
        help="Sampling temperature for LLM Harness validator calls "
             "(default 0.2).",
    )
    parser.add_argument(
        "--llm-harness-timeout-s", type=float, default=30.0,
        help="Hard timeout per LLM Harness validator call. On timeout "
             "the deterministic verdict (admit) stands.",
    )

    # Lane-(a) feature flag (T1.3a): the live trainer Crafter never mints
    # PatchProposals by default — see implementation_notes/legacy/skill-lane-decision.md.
    # The dispatcher's existing `_STATUS_NO_OP` → Hypothesizer fall-through
    # carries the failure signal through. Set this only for explicit
    # lane-(b) experiments.
    parser.add_argument(
        "--enable-protocol-patching", dest="crafter_enable_protocol_patching",
        action="store_true",
        help="Lane-(b) override: enable the Crafter Repairer / "
             "PatchProposal mint path in the live trainer. Off by "
             "default per the lane-(a) decision (skills are retrieval "
             "payloads). See implementation_notes/legacy/skill-lane-decision.md.",
    )
    parser.set_defaults(crafter_enable_protocol_patching=False)

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
        speculative_method=("none" if args.speculative_method == "none"
                            else args.speculative_method),
        speculative_model=args.speculative_model or None,
        num_speculative_tokens=args.num_speculative_tokens,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        grpo_enabled=not args.no_grpo,
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

    # ── Per-game episode overrides ──────────────────────────────────
    # Layered precedence: HIGH_VARIANCE_GYMV_EPISODES (built-in default
    # via dataclass factory) → unified-roles flat override → explicit
    # --episodes-per-game-overrides JSON.  We resolve here so the
    # final dict is what reaches CoEvolutionConfig.
    from trainer.coevolution.config import (
        EPISODES_PER_GAME_MULTIROLE as _EPS_MULTIROLE,
        HIGH_VARIANCE_GYMV_EPISODES as _EPS_HIGH_VAR,
    )
    eps_overrides: Dict[str, int] = {**_EPS_MULTIROLE, **_EPS_HIGH_VAR}
    if args.unified_roles:
        eps_overrides = {g: args.episodes_per_game for g in games}
    if args.episodes_per_game_overrides is not None:
        try:
            cli_overrides = json.loads(args.episodes_per_game_overrides)
            if not isinstance(cli_overrides, dict):
                raise ValueError(
                    "--episodes-per-game-overrides must be a JSON object"
                )
            eps_overrides = {**eps_overrides, **{
                str(k): int(v) for k, v in cli_overrides.items()
            }}
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"Invalid --episodes-per-game-overrides JSON: {exc}\n"
                f"Got: {args.episodes_per_game_overrides!r}"
            )
    if eps_overrides:
        config_kwargs["episodes_per_game_overrides"] = eps_overrides

    if args.opponent_model is not None:
        config_kwargs["opponent_model"] = args.opponent_model
    if args.opponent_api_base is not None:
        config_kwargs["opponent_api_base"] = args.opponent_api_base

    if args.seed_bank_dir is not None:
        config_kwargs["seed_bank_dir"] = args.seed_bank_dir

    if args.bank_mode is not None:
        config_kwargs["bank_mode"] = args.bank_mode

    if args.crafter_promotion_enabled:
        config_kwargs["crafter_promotion_enabled"] = True
    if args.crafter_cycle_every_k_steps:
        config_kwargs["crafter_cycle_every_k_steps"] = args.crafter_cycle_every_k_steps
    if args.crafter_outcome_failure_threshold != 0.0:
        config_kwargs["crafter_outcome_failure_threshold"] = (
            args.crafter_outcome_failure_threshold
        )
    if args.crafter_promotion_timeout_s != 300.0:
        config_kwargs["crafter_promotion_timeout_s"] = args.crafter_promotion_timeout_s
    if args.crafter_promotion_gate_mode != "offline-synthetic":
        config_kwargs["crafter_promotion_gate_mode"] = args.crafter_promotion_gate_mode
    # Hypothesizer fallthrough gate (post-v11 audit) — only emit when
    # the user explicitly diverges from the defaults so older smoke
    # tests that snapshot config_kwargs aren't tripped by a noisy
    # additive field.
    if args.crafter_hypothesize_min_recurrences != 3:
        config_kwargs["crafter_hypothesize_min_recurrences"] = (
            args.crafter_hypothesize_min_recurrences
        )
    if args.crafter_hypothesize_related_skill_jaccard != 0.30:
        config_kwargs["crafter_hypothesize_related_skill_jaccard"] = (
            args.crafter_hypothesize_related_skill_jaccard
        )

    if args.harness_enabled:
        config_kwargs["harness_enabled"] = True
    if not args.harness_allow_shadow:
        config_kwargs["harness_allow_shadow"] = False
    if args.crafter_enable_protocol_patching:
        config_kwargs["crafter_enable_protocol_patching"] = True

    # ── Block B — §5.5 ablation flags wire-up ─────────────────────────
    if args.harness_mode != "full":
        config_kwargs["harness_mode"] = args.harness_mode
    if not args.crafter_enabled:
        config_kwargs["crafter_enabled"] = False
    if args.promotion_bypass_mode != "gated":
        config_kwargs["promotion_bypass_mode"] = args.promotion_bypass_mode
    if args.intention_trigger != "every-step":
        config_kwargs["intention_trigger"] = args.intention_trigger
    if args.actor_bank_cap_k > 0:
        config_kwargs["actor_bank_cap_k"] = int(args.actor_bank_cap_k)

    if args.game_schema_enabled:
        config_kwargs["game_schema_enabled"] = True
    if args.game_schema_model:
        config_kwargs["game_schema_model"] = args.game_schema_model
    if args.game_schema_max_tokens != 1024:
        config_kwargs["game_schema_max_tokens"] = args.game_schema_max_tokens
    if args.game_schema_timeout_s != 60.0:
        config_kwargs["game_schema_timeout_s"] = args.game_schema_timeout_s

    # Path 2 — supplemental LLM Crafter (35B-A3B teacher).
    if args.llm_crafter_enabled:
        config_kwargs["llm_crafter_enabled"] = True
    if args.llm_crafter_model:
        config_kwargs["llm_crafter_model"] = args.llm_crafter_model
    if args.llm_crafter_k_max != 2:
        config_kwargs["llm_crafter_k_max"] = args.llm_crafter_k_max
    if args.llm_crafter_max_tokens != 1024:
        config_kwargs["llm_crafter_max_tokens"] = args.llm_crafter_max_tokens
    if args.llm_crafter_temperature != 0.3:
        config_kwargs["llm_crafter_temperature"] = args.llm_crafter_temperature
    if args.llm_crafter_timeout_s != 60.0:
        config_kwargs["llm_crafter_timeout_s"] = args.llm_crafter_timeout_s
    # Stage 2 cross-domain opt-in for Path 2.
    if args.llm_crafter_enable_thinking:
        config_kwargs["llm_crafter_enable_thinking"] = True

    # Path 3 — Promotion judge (Stage 2 cross-domain only).
    if args.crafter_promotion_judge_enable_thinking:
        config_kwargs[
            "crafter_promotion_judge_enable_thinking"
        ] = True
    if args.crafter_promotion_judge_max_tokens != 256:
        config_kwargs[
            "crafter_promotion_judge_max_tokens"
        ] = args.crafter_promotion_judge_max_tokens

    # Path 4 — LLM Harness validator (35B-A3B teacher).
    if args.llm_harness_validator_enabled:
        config_kwargs["llm_harness_validator_enabled"] = True
    if args.llm_harness_model:
        config_kwargs["llm_harness_model"] = args.llm_harness_model
    if args.llm_harness_bootstrap_steps != 20:
        config_kwargs["llm_harness_bootstrap_steps"] = args.llm_harness_bootstrap_steps
    if args.llm_harness_max_tokens != 256:
        config_kwargs["llm_harness_max_tokens"] = args.llm_harness_max_tokens
    if args.llm_harness_temperature != 0.2:
        config_kwargs["llm_harness_temperature"] = args.llm_harness_temperature
    if args.llm_harness_timeout_s != 30.0:
        config_kwargs["llm_harness_timeout_s"] = args.llm_harness_timeout_s

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
    print(f"  Bank dir:     {config.bank_dir}")
    print(f"  Adapter dir:  {config.adapter_dir}")
    print(f"  Checkpoint:   every {config.checkpoint_interval} steps → {config.checkpoint_dir}")
    print(f"  GRPO data:    {config.grpo_data_dir}")
    print(f"  Rewards:      {config.rewards_dir}")
    print(f"  TensorBoard:  {config.tensorboard_dir}")
    print(f"  Log dir:      {config.log_dir}")
    print(f"  W&B:          {'enabled' if config.wandb_enabled else 'disabled'}")
    if config.crafter_promotion_enabled:
        every = (config.crafter_cycle_every_k_steps
                 if config.crafter_cycle_every_k_steps > 0 else 1)
        print(f"  Crafter+Prom: enabled (every {every} step(s), "
              f"fail<{config.crafter_outcome_failure_threshold:.2f}, "
              f"timeout {config.crafter_promotion_timeout_s:.0f}s)")
    else:
        print("  Crafter+Prom: disabled")
    if config.harness_enabled:
        print(f"  Harness:      enabled (allow_shadow={config.harness_allow_shadow})")
    else:
        print("  Harness:      disabled")
    if config.game_schema_enabled:
        _gsm = config.game_schema_model or "BACKBONE_JUDGE_MODEL (env)"
        print(
            f"  GameProfile:  enabled (1×35B/game/phase via {_gsm}, "
            f"max_tokens={config.game_schema_max_tokens}, "
            f"timeout={config.game_schema_timeout_s:.0f}s)"
        )
    else:
        print("  GameProfile:  disabled")
    if getattr(config, "llm_crafter_enabled", False):
        _lcm = config.llm_crafter_model or "BACKBONE_JUDGE_MODEL (env)"
        _lc_think = (
            "  thinking=ON (Stage-2 cross-domain)"
            if getattr(config, "llm_crafter_enable_thinking", False)
            else "  thinking=OFF (Stage-1 in-domain)"
        )
        print(
            f"  LLM Crafter:  enabled (≤{config.llm_crafter_k_max} parallel "
            f"35B/game/step via {_lcm}, "
            f"max_tokens={config.llm_crafter_max_tokens}, "
            f"timeout={config.llm_crafter_timeout_s:.0f}s){_lc_think}"
        )
    else:
        print("  LLM Crafter:  disabled")
    if getattr(config, "crafter_promotion_judge_enable_thinking", False):
        print(
            f"  Promo Judge:  thinking=ON (Stage-2 cross-domain) "
            f"max_tokens={config.crafter_promotion_judge_max_tokens}"
        )
    if getattr(config, "llm_harness_validator_enabled", False):
        _lhm = config.llm_harness_model or "BACKBONE_JUDGE_MODEL (env)"
        print(
            f"  LLM Harness:  enabled (bootstrap<{config.llm_harness_bootstrap_steps} "
            f"steps via {_lhm}, "
            f"max_tokens={config.llm_harness_max_tokens}, "
            f"timeout={config.llm_harness_timeout_s:.0f}s)"
        )
    else:
        print("  LLM Harness:  disabled")
    if config.crafter_enable_protocol_patching:
        print("  Repairer:     ENABLED (lane-(b) — protocol patching live)")
    else:
        print("  Repairer:     parked (lane-(a) — patches gated off; "
              "Hypothesizer carries failure signal)")
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
