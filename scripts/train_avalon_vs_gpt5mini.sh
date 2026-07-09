#!/usr/bin/env bash
#
# Train Avalon agent with gpt-5-mini as opponent.
#
# Uses the co-evolution loop with:
#   - gpt-5-mini opponents via OpenRouter (breaks self-play reward inflation)
#   - Conservative hyperparameters (low LR, high KL, tight clipping)
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
#   bash scripts/train_avalon_vs_gpt5mini.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/../GamingAgent:${ROOT_DIR}/../AgentEvolver:${PYTHONPATH:-}"
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export RAG_EMBEDDER_DEVICE=cpu

# ── SFT cold-start adapters (pre-trained from GPT-5.4 demonstrations) ──
SFT_DIR="${ROOT_DIR}/runs/sft_coldstart"

echo "============================================================"
echo "  AVALON TRAINING — gpt-5-mini opponent"
echo "============================================================"
echo "  Model:      Qwen/Qwen3.5-9B"
echo "  Opponent:   gpt-5-mini (OpenRouter)"
echo "  Steps:      30"
echo "  Episodes:   40 per step (unified roles)"
echo "  LR:         2e-5 → 1e-5 (cosine)"
echo "  KL coeff:   0.02 → 0.10"
echo "  Clip ratio: 0.10"
echo "  Adapters:   from ${SFT_DIR}"
echo "============================================================"

python scripts/run_coevolution.py \
    --games avalon \
    --total-steps 30 \
    --episodes-per-game 40 \
    --unified-roles \
    --model "Qwen/Qwen3.5-9B" \
    --temperature 0.3 \
    --opponent-model "gpt-5-mini" \
    --opponent-api-base "https://openrouter.ai/api/v1" \
    --vllm-gpus 0 1 2 3 \
    --grpo-devices 4 5 6 7 \
    --vllm-gpu-util 0.90 \
    --speculative-model "Qwen/Qwen3-0.6B" \
    --num-speculative-tokens 5 \
    --warmup-steps 10 \
    --grpo-lr 1e-5 \
    --initial-kl-coeff 0.02 \
    --grpo-kl-coeff 0.10 \
    --grpo-clip-ratio 0.10 \
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
