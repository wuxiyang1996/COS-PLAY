#!/usr/bin/env bash
# ======================================================================
#  Ablation: Base Qwen3-8B (No LoRA, No Skill Bank) vs GPT-5.4 on Avalon
#
#  Pure baseline — the vanilla Qwen3-8B base model with NO LoRA adapter
#  and NO skill bank, evaluated as a Decision Agent in 5-player Avalon
#  against GPT-5.4 opponents.  8 episodes per player seat (40 total).
#
#  Usage:
#    conda activate game-ai-agent
#    bash ablation_study/run_base_model_avalon_da.sh
#    EVAL_GPUS=0 bash ablation_study/run_base_model_avalon_da.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8024/v1 bash ablation_study/run_base_model_avalon_da.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Ensure vLLM is on PATH (conda env with game-ai-agent)
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

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
NUM_PLAYERS=5
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-8}"
EPISODES=$((NUM_PLAYERS * EPISODES_PER_PLAYER))
TEMPERATURE="${TEMPERATURE:-0.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
EVAL_GPUS="${EVAL_GPUS:-0}"
VLLM_PORT="${VLLM_PORT:-8020}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/base_model_avalon_da_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[base-avalon-da] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[base-avalon-da] Done."
}
trap cleanup EXIT INT TERM

echo "══════════════════════════════════════════════════════════════"
echo "  Ablation: Base Qwen3-8B (No LoRA, No Skill Bank)"
echo "           vs ${OPPONENT_MODEL} on Avalon"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Base model:      ${BASE_MODEL}"
echo "  Adapter:         NONE (vanilla base model)"
echo "  Skill bank:      NONE (--no-bank)"
echo "  Opponent:        ${OPPONENT_MODEL}"
echo "  Episodes:        ${EPISODES} (${EPISODES_PER_PLAYER} per player × ${NUM_PLAYERS})"
echo "  GPU(s):          ${EVAL_GPUS}"
echo "  Output:          ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ "${NO_SERVER}" = "0" ]; then
    echo "[base-avalon-da] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    echo "  Model: ${BASE_MODEL} (no LoRA)"
    echo ""

    VLLM_LOG="${OUTPUT_BASE}/vllm_server.log"

    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90 \
            --dtype auto \
            --trust-remote-code \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    echo "[base-avalon-da] vLLM PID=${VLLM_PID}, log at ${VLLM_LOG}"

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[base-avalon-da] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[base-avalon-da] ERROR: vLLM server exited unexpectedly. Check ${VLLM_LOG}"
            tail -30 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[base-avalon-da] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[base-avalon-da] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

EVAL_ARGS=(
    --games avalon
    --episodes "${EPISODES}"
    --temperature "${TEMPERATURE}"
    --model "${BASE_MODEL}"
    --num_players "${NUM_PLAYERS}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_BASE}"
    --opponent_model "${OPPONENT_MODEL}"
    --per_role
    --verbose
    --no-bank
)

echo ""
echo "[base-avalon-da] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "[base-avalon-da] Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
