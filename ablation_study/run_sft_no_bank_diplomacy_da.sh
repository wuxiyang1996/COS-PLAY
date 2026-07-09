#!/usr/bin/env bash
# ======================================================================
#  Ablation: SFT Decision Agent (No Skill Bank) vs GPT-5.4 on Diplomacy
#
#  Uses the SFT cold-start LoRA adapter for the decision agent
#  (action_taking) with NO skill bank, evaluated in 7-power Diplomacy
#  against GPT-5.4 opponents.  4 episodes per power (28 total).
#
#  Run: Qwen3-8B_20260327_062035
#
#  Usage:
#    conda activate game-ai-agent
#    bash ablation_study/run_sft_no_bank_diplomacy_da.sh
#    EVAL_GPUS=0 bash ablation_study/run_sft_no_bank_diplomacy_da.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8025/v1 bash ablation_study/run_sft_no_bank_diplomacy_da.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GAME_AI_AGENT_ENV_BIN="${GAME_AI_AGENT_ENV_BIN:-/workspace/miniconda3/envs/game-ai-agent/bin}"
if [ -x "${GAME_AI_AGENT_ENV_BIN}/python" ]; then
  export PATH="${GAME_AI_AGENT_ENV_BIN}:${PATH}"
fi

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${RUNS_DIR}/sft_coldstart/decision/action_taking/action_taking}"
SFT_ADAPTER_NAME="qwen_sft"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-4}"
EPISODES=$((NUM_POWERS * EPISODES_PER_POWER))
TEMPERATURE="${TEMPERATURE:-0.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
EVAL_GPUS="${EVAL_GPUS:-0}"
VLLM_PORT="${VLLM_PORT:-8025}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
[ -z "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY="${OPENROUTER_API_KEY:-}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[sft-no-bank-diplomacy-da] ERROR: No API key found for GPT-5.4 opponents."
    echo "  Set OPENROUTER_API_KEY or OPENAI_API_KEY, or configure api_keys.py."
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/sft_no_bank_diplomacy_da_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[sft-no-bank-diplomacy-da] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[sft-no-bank-diplomacy-da] Done."
}
trap cleanup EXIT INT TERM

if [ ! -f "${SFT_ADAPTER_PATH}/adapter_config.json" ]; then
    echo "[sft-no-bank-diplomacy-da] ERROR: SFT adapter not found at ${SFT_ADAPTER_PATH}"
    echo "  Expected: ${SFT_ADAPTER_PATH}/adapter_config.json"
    echo "  Run SFT training first, or set SFT_ADAPTER_PATH."
    exit 1
fi

echo "══════════════════════════════════════════════════════════════"
echo "  Ablation: SFT Decision Agent (No Skill Bank)"
echo "           vs ${OPPONENT_MODEL} on Diplomacy"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Base model:      ${BASE_MODEL}"
echo "  SFT adapter:     ${SFT_ADAPTER_NAME} -> ${SFT_ADAPTER_PATH}"
echo "  Skill bank:      NONE (--no-bank)"
echo "  Opponent:        ${OPPONENT_MODEL}"
echo "  Episodes:        ${EPISODES} (${EPISODES_PER_POWER} per power × ${NUM_POWERS})"
echo "  GPU(s):          ${EVAL_GPUS}"
echo "  Output:          ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ "${NO_SERVER}" = "0" ]; then
    echo "[sft-no-bank-diplomacy-da] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    echo "  Model: ${BASE_MODEL}"
    echo "  LoRA:  ${SFT_ADAPTER_NAME} (SFT cold-start)"
    echo ""

    VLLM_LOG="${OUTPUT_BASE}/vllm_server.log"

    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.90 \
            --dtype auto \
            --trust-remote-code \
            --enable-lora \
            --max-loras 2 \
            --max-lora-rank 64 \
            --lora-modules "${SFT_ADAPTER_NAME}=${SFT_ADAPTER_PATH}" \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    echo "[sft-no-bank-diplomacy-da] vLLM PID=${VLLM_PID}, log at ${VLLM_LOG}"

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[sft-no-bank-diplomacy-da] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[sft-no-bank-diplomacy-da] ERROR: vLLM server exited unexpectedly. Check ${VLLM_LOG}"
            tail -30 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[sft-no-bank-diplomacy-da] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[sft-no-bank-diplomacy-da] Using existing vLLM server at ${VLLM_BASE_URL}"
    echo "  Make sure it was started with --enable-lora and the SFT adapter!"
fi

EVAL_ARGS=(
    --games diplomacy
    --episodes "${EPISODES}"
    --temperature "${TEMPERATURE}"
    --model "${SFT_ADAPTER_NAME}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_BASE}"
    --opponent_model "${OPPONENT_MODEL}"
    --per_power
    --verbose
    --no-bank
)

echo ""
echo "[sft-no-bank-diplomacy-da] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "[sft-no-bank-diplomacy-da] Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
