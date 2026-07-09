#!/usr/bin/env bash
# ======================================================================
#  Inference: Candy Crush with best-performing Qwen3-8B checkpoint (8 episodes)
#
#  Checkpoint: runs/Qwen3-8B_20260321_213813_(Candy_crush)/best/ (step 9)
#  Training mean_reward: 528.375  (max 653, min 469)
#
#  Launches a vLLM server for Qwen/Qwen3-8B with the best action_taking
#  LoRA adapter enabled and runs 8 inference episodes on Candy Crush
#  using the trained skill bank.
#
#  Usage:
#    conda activate game-ai-agent
#    bash inference/infer_candy_crush_best.sh
#
#    # With overrides:
#    EPISODES=16 bash inference/infer_candy_crush_best.sh
#    EVAL_GPUS=0 bash inference/infer_candy_crush_best.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8021/v1 bash inference/infer_candy_crush_best.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Best checkpoint paths ─────────────────────────────────────────────
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260321_213813_(Candy_crush)"
BEST_DIR="${RUN_DIR}/best"
ADAPTER_PATH="${BEST_DIR}/adapters/decision/action_taking"
BANK_PATH="${BEST_DIR}/banks/candy_crush/skill_bank.jsonl"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "[ERROR] Best adapter not found: ${ADAPTER_PATH}"
    exit 1
fi
if [ ! -f "${BANK_PATH}" ]; then
    echo "[ERROR] Best skill bank not found: ${BANK_PATH}"
    exit 1
fi

# ── Configurable parameters ──────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LORA_NAME="qwen3-8b-candy-crush-best"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
EVAL_GPUS="${EVAL_GPUS:-6}"
VLLM_PORT="${VLLM_PORT:-8021}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Output directory ──────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_candy_crush_best_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[infer_candy_crush_best] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[infer_candy_crush_best] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Candy Crush Inference: Qwen3-8B best checkpoint (step 9)"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:     ${BASE_MODEL}"
echo "  LoRA adapter:   ${ADAPTER_PATH}"
echo "  LoRA name:      ${LORA_NAME}"
echo "  Skill bank:     ${BANK_PATH}"
echo "  Episodes:       ${EPISODES}"
echo "  Max steps:      ${MAX_STEPS}"
echo "  Temperature:    ${TEMPERATURE}"
echo "  GPU(s):         ${EVAL_GPUS}"
echo "  Output:         ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server with LoRA ─────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer_candy_crush_best] Launching vLLM server with LoRA on ${VLLM_HOST}:${VLLM_PORT}..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --trust-remote-code \
            --enable-lora \
            --lora-modules "${LORA_NAME}=${ADAPTER_PATH}" \
            --max-lora-rank 16 \
        &
    VLLM_PID=$!

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[infer_candy_crush_best] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[infer_candy_crush_best] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[infer_candy_crush_best] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[infer_candy_crush_best] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Build evaluation command ─────────────────────────────────────────
EVAL_ARGS=(
    --games candy_crush
    --episodes "${EPISODES}"
    --max_steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --model "${LORA_NAME}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
    --bank "${BANK_PATH}"
)

echo "[infer_candy_crush_best] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

# ── Run inference ────────────────────────────────────────────────────
EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Candy Crush best-checkpoint inference COMPLETE"
else
    echo "  Candy Crush best-checkpoint inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
