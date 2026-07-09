#!/usr/bin/env bash
#
# Train Diplomacy agent with gpt-5-mini as opponent.
#
# Uses the co-evolution loop with:
#   - gpt-5-mini opponents via OpenRouter (breaks self-play reward inflation)
#   - Very conservative hyperparameters (Diplomacy is 7-player, high variance)
#   - All checkpoints saved (checkpoint_keep_last=0)
#   - Pre-trained SFT adapters as starting point
#
# Prerequisites:
#   - 8 GPUs (0-3 for vLLM inference, 4-7 for GRPO training)
#   - OPENROUTER_API_KEY env var set (see .env.example)
#   - Qwen/Qwen3.5-9B model weights in /workspace/huggingface/
#
# Usage:
#   cd /workspace/game_agent/Game-AI-Agent
#   bash scripts/train_diplomacy_vs_gpt5mini.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/../GamingAgent:${ROOT_DIR}/../AgentEvolver:${ROOT_DIR}/../AI_Diplomacy:${PYTHONPATH:-}"
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export RAG_EMBEDDER_DEVICE=cpu

# ── SFT cold-start adapters (pre-trained from GPT-5.4 demonstrations) ──
SFT_DIR="${ROOT_DIR}/runs/sft_coldstart"

echo "============================================================"
echo "  DIPLOMACY TRAINING — gpt-5-mini opponent"
echo "============================================================"
echo "  Model:      Qwen/Qwen3.5-9B"
echo "  Opponent:   gpt-5-mini (OpenRouter)"
echo "  Steps:      35"
echo "  Episodes:   28 per step (unified roles, all 7 powers)"
echo "  LR:         1e-5 → 5e-6 (cosine)"
echo "  KL coeff:   0.02 → 0.12"
echo "  Clip ratio: 0.08"
echo "  Adapters:   from ${SFT_DIR}"
echo "============================================================"

python scripts/run_coevolution.py \
    --games diplomacy \
    --total-steps 35 \
    --episodes-per-game 28 \
    --unified-roles \
    --model "Qwen/Qwen3.5-9B" \
    --temperature 0.3 \
    --opponent-model "gpt-5-mini" \
    --opponent-api-base "https://openrouter.ai/api/v1" \
    --vllm-gpus 0 1 2 3 \
    --grpo-devices 4 5 6 7 \
    --vllm-gpu-util 0.85 \
    --speculative-model "Qwen/Qwen3-0.6B" \
    --num-speculative-tokens 5 \
    --warmup-steps 15 \
    --grpo-lr 5e-6 \
    --initial-kl-coeff 0.02 \
    --grpo-kl-coeff 0.12 \
    --grpo-clip-ratio 0.08 \
    --grpo-max-epochs 3 \
    --grpo-adv-clip 2.0 \
    --initial-temperature 0.9 \
    --steady-temperature 0.6 \
    --checkpoint-interval 1 \
    --load-adapters-from "${SFT_DIR}" \
    --curriculum none \
    --wandb-project game-ai-coevolution \
    --debug-io \
    "$@"
