#!/usr/bin/env bash
# ======================================================================
#  ALFWorld co-evolution from SFT v5 (SkillRL-augmented, 10,113 records)
#  with the skill-bank rollout fix (bank_available / SkillQueryEngine).
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

MODEL="Qwen/Qwen3-8B"
VLLM_GPUS="0 1 2 3"
GRPO_GPUS="4 5 6 7"
VLLM_PORT=8000
GPU_UTIL=0.82
SPEC_MODEL="Qwen/Qwen3-0.6B"
SPEC_TOKENS=5

ADAPTER_DIR="${PROJECT_ROOT}/runs/sft_textenv_v5/adapters_flat"
SEED_BANK_DIR="${PROJECT_ROOT}/labeling/output/gpt54_textenv_skillbank"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

ALF_STEPS=10
ALF_EPISODES=48
ALF_DIR="${PROJECT_ROOT}/runs/alfworld_coevo_v5_${TIMESTAMP}"
mkdir -p "${ALF_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  ALFWorld co-evolution from SFT v5 (${ALF_STEPS} iters)"
echo "  Adapters: ${ADAPTER_DIR}"
echo "  Output: ${ALF_DIR}"
echo "══════════════════════════════════════════════════════════════"

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
    --checkpoint-interval 1 \
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
    2>&1 | tee "${ALF_DIR}/train.log"
