#!/usr/bin/env bash
# Launch vLLM server(s) for co-evolution training.
#
# Two modes:
#
#   1. MANAGED (default when run from run_all.sh with MANAGE_VLLM=1):
#      The Python orchestrator manages vLLM lifecycle — this script
#      is NOT used.
#
#   2. STANDALONE / LEGACY (this script):
#      Launches a single vLLM instance. Useful for debugging or when
#      running vLLM externally from the training loop.
#
# Usage:
#   conda activate game-ai-agent
#   bash scripts/launch_vllm_coevolution.sh
#
#   # Single GPU, TP=1 (for testing):
#   VLLM_TP=1 CUDA_VISIBLE_DEVICES=0 bash scripts/launch_vllm_coevolution.sh
#
#   # Multi-GPU, TP=4 (legacy):
#   VLLM_TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_vllm_coevolution.sh
#
#   # Point to a timestamped run directory for adapters:
#   ADAPTER_DIR=runs/Qwen3-8B_20260315_143022/lora_adapters bash scripts/launch_vllm_coevolution.sh

set -euo pipefail

# Headless rendering
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# HuggingFace cache
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
ADAPTER_DIR="${ADAPTER_DIR:-runs/lora_adapters}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

echo "═══════════════════════════════════════════════════"
echo "  vLLM Co-Evolution Server"
echo "═══════════════════════════════════════════════════"
echo "  Model:    ${MODEL}"
echo "  TP:       ${TP}"
echo "  GPU Util: ${GPU_UTIL}"
echo "  Port:     ${PORT}"
echo "  Adapters: ${ADAPTER_DIR}"
echo "  Spec dec: ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo "═══════════════════════════════════════════════════"

# Build --lora-modules args (only for adapters that exist)
LORA_ARGS=""
for adapter in skill_selection action_taking; do
    adapter_path="${ADAPTER_DIR}/decision/${adapter}"
    if [ -d "${adapter_path}" ]; then
        LORA_ARGS="${LORA_ARGS} ${adapter}=${adapter_path}"
        echo "  LoRA: ${adapter} → ${adapter_path}"
    else
        echo "  LoRA: ${adapter} → (not found, will use base model)"
    fi
done
for adapter in segment contract curator; do
    adapter_path="${ADAPTER_DIR}/skillbank/${adapter}"
    if [ -d "${adapter_path}" ]; then
        LORA_ARGS="${LORA_ARGS} ${adapter}=${adapter_path}"
        echo "  LoRA: ${adapter} → ${adapter_path}"
    else
        echo "  LoRA: ${adapter} → (not found, will use base model)"
    fi
done

echo "═══════════════════════════════════════════════════"

LORA_FLAGS="--enable-lora --max-loras 5 --max-lora-rank 64"
if [ -n "${LORA_ARGS}" ]; then
    LORA_FLAGS="${LORA_FLAGS} --lora-modules ${LORA_ARGS}"
fi

SPEC_ARGS=()
if [ -n "${SPEC_MODEL}" ]; then
    SPEC_JSON="{\"model\":\"${SPEC_MODEL}\",\"num_speculative_tokens\":${SPEC_TOKENS},\"method\":\"draft_model\"}"
    SPEC_ARGS=(--speculative_config "${SPEC_JSON}")
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    ${LORA_FLAGS} \
    "${SPEC_ARGS[@]}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --port "${PORT}" \
    --trust-remote-code \
    "$@"
