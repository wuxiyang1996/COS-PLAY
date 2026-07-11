#!/usr/bin/env bash
# ======================================================================
#  Text-env co-evolution pilots (sequential, each from SFT adapters):
#
#  Run 1: ALFWorld — 10 iters, train split rollouts,
#         eval_out_of_distribution eval after each iter.
#  Run 2: WebShop  — 3 iters (conservative — SFT already at 88%).
#
#  Both start FROM the SFT adapters (all 5 LoRAs) with the cold-start
#  seed skill bank; checkpoint every iteration.
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Shared settings ──────────────────────────────────────────────────
MODEL="Qwen/Qwen3-8B"
CKPT_INTERVAL=1
VLLM_GPUS="0 1 2 3"
GRPO_GPUS="4 5 6 7"
VLLM_PORT=8000
GPU_UTIL=0.82
SPEC_MODEL="Qwen/Qwen3-0.6B"
SPEC_TOKENS=5

ADAPTER_DIR="${PROJECT_ROOT}/runs/sft_textenv_v3/adapters_flat"
SEED_BANK_DIR="${PROJECT_ROOT}/labeling/output/gpt54_textenv_skillbank"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ══ Run 1: ALFWorld ═══════════════════════════════════════════════════
ALF_STEPS=10
ALF_EPISODES=48       # 8 rollouts × 6 ALFWorld categories
ALF_DIR="${PROJECT_ROOT}/runs/alfworld_coevo_${TIMESTAMP}"
mkdir -p "${ALF_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Run 1/2: ALFWorld co-evolution (from SFT, ${ALF_STEPS} iters)"
echo "  Train: train split | Eval: eval_out_of_distribution (20 eps/iter)"
echo "  Output: ${ALF_DIR}"
echo "══════════════════════════════════════════════════════════════"

ALF_EXIT=0
python scripts/run_coevolution.py \
    --games alfworld \
    --eval-games alfworld \
    --eval-episodes-per-game 20 \
    --alfworld-split train \
    --alfworld-eval-split eval_out_of_distribution \
    --run-dir "${ALF_DIR}" \
    --total-steps "${ALF_STEPS}" \
    --curriculum none \
    --episodes-per-game "${ALF_EPISODES}" \
    --checkpoint-interval "${CKPT_INTERVAL}" \
    --model "${MODEL}" \
    --load-adapters-from "${ADAPTER_DIR}" \
    --seed-bank-dir "${SEED_BANK_DIR}" \
    --vllm-gpus ${VLLM_GPUS} \
    --grpo-devices ${GRPO_GPUS} \
    --vllm-base-port "${VLLM_PORT}" \
    --vllm-gpu-util "${GPU_UTIL}" \
    --speculative-model "${SPEC_MODEL}" \
    --num-speculative-tokens "${SPEC_TOKENS}" \
    --grpo-lr 2e-5 \
    --grpo-kl-coeff 0.05 \
    --grpo-max-epochs 2 \
    --no-wandb \
    --debug-io \
    --from-scratch \
    2>&1 | tee "${ALF_DIR}/train.log" || ALF_EXIT=$?

echo
echo "Run 1 (ALFWorld) finished with exit=${ALF_EXIT}"
echo

# Make sure vLLM instances from run 1 are fully gone before run 2
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 15

# ══ Run 2: WebShop ═══════════════════════════════════════════════════
WS_STEPS=3
WS_EPISODES=24
WS_DIR="${PROJECT_ROOT}/runs/webshop_coevo_${TIMESTAMP}"
mkdir -p "${WS_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Run 2/2: WebShop co-evolution (from SFT, ${WS_STEPS} iters)"
echo "  Output: ${WS_DIR}"
echo "══════════════════════════════════════════════════════════════"

WS_EXIT=0
python scripts/run_coevolution.py \
    --games webshop \
    --run-dir "${WS_DIR}" \
    --total-steps "${WS_STEPS}" \
    --curriculum none \
    --episodes-per-game "${WS_EPISODES}" \
    --checkpoint-interval "${CKPT_INTERVAL}" \
    --model "${MODEL}" \
    --load-adapters-from "${ADAPTER_DIR}" \
    --seed-bank-dir "${SEED_BANK_DIR}" \
    --vllm-gpus ${VLLM_GPUS} \
    --grpo-devices ${GRPO_GPUS} \
    --vllm-base-port "${VLLM_PORT}" \
    --vllm-gpu-util "${GPU_UTIL}" \
    --speculative-model "${SPEC_MODEL}" \
    --num-speculative-tokens "${SPEC_TOKENS}" \
    --grpo-lr 5e-6 \
    --grpo-kl-coeff 0.1 \
    --grpo-max-epochs 1 \
    --no-wandb \
    --debug-io \
    --from-scratch \
    2>&1 | tee "${WS_DIR}/train.log" || WS_EXIT=$?

echo
echo "══════════════════════════════════════════════════════════════"
echo "  Both pilots done."
echo "  ALFWorld: exit=${ALF_EXIT}  →  ${ALF_DIR}"
echo "  WebShop:  exit=${WS_EXIT}  →  ${WS_DIR}"
echo "══════════════════════════════════════════════════════════════"
