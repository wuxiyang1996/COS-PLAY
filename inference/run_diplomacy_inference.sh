#!/usr/bin/env bash
# ======================================================================
#  Diplomacy inference with trained Qwen/Qwen3.5-9B decision agent
#
#  Variants:
#    da       — standard decision agent vs GPT-5.4
#    discrete — discrete action format (training-matched) vs GPT-5.4
#
#  Usage:
#    bash inference/run_diplomacy_inference.sh --variant da
#    bash inference/run_diplomacy_inference.sh --variant discrete
#    EPISODES_PER_POWER=4 bash inference/run_diplomacy_inference.sh --variant da
#    OPPONENT_MODEL=gpt-5-mini bash inference/run_diplomacy_inference.sh --variant da
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Parse arguments ──────────────────────────────────────────────────
VARIANT=""
while [[ $# -gt 0 ]]; do
    case "$1" in --variant) VARIANT="$2"; shift 2 ;; *) echo "Unknown: $1"; exit 1 ;; esac
done
[ -z "${VARIANT}" ] && echo "Usage: $0 --variant {da|discrete}" && exit 1

# ── Per-variant config ───────────────────────────────────────────────
case "${VARIANT}" in
    da)       EVAL_MODULE="inference.run_qwen3_8b_eval" ;;
    discrete) EVAL_MODULE="inference.run_diplomacy_discrete_eval" ;;
    *)        echo "[ERROR] Unknown variant: ${VARIANT}. Use: da, discrete"; exit 1 ;;
esac

# ── Checkpoint paths ─────────────────────────────────────────────────
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3.5-9B_20260327_062035"
CKPT_DIR="${RUN_DIR}/checkpoints/step_0017"
ADAPTER_PATH="${CKPT_DIR}/adapters/decision/action_taking"
BANKS_DIR="${CKPT_DIR}/banks/diplomacy"
LORA_NAME="qwen3-8b-diplomacy-da"

[ ! -d "${ADAPTER_PATH}" ] && echo "[ERROR] Adapter not found: ${ADAPTER_PATH}" && exit 1

# ── Merge per-power skill banks ──────────────────────────────────────
BANK_FILES=()
for power in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY; do
    pbank="${BANKS_DIR}/${power}/skill_bank.jsonl"
    if [ -f "${pbank}" ]; then
        BANK_FILES+=("${pbank}")
    else
        echo "[WARN] Skill bank not found for ${power}: ${pbank}"
    fi
done

COMBINED_BANK=""
if [ ${#BANK_FILES[@]} -gt 0 ]; then
    COMBINED_BANK="${CKPT_DIR}/banks/diplomacy/combined_skill_bank.jsonl"
    cat "${BANK_FILES[@]}" | sort -u > "${COMBINED_BANK}"
fi

# ── Environment ──────────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

[ -z "${OPENROUTER_API_KEY:-}" ] && echo "Warning: OPENROUTER_API_KEY not set. See .env.example."
[ -z "${OPENAI_API_KEY:-}" ] && echo "Warning: OPENAI_API_KEY not set. See .env.example."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-10}"
EPISODES=$((NUM_POWERS * EPISODES_PER_POWER))
TEMPERATURE="${TEMPERATURE:-0.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
EVAL_GPUS="${EVAL_GPUS:-6}"
VLLM_PORT="${VLLM_PORT:-8025}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"
UNCHOSEN_STRATEGY="${UNCHOSEN_STRATEGY:-hold}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_diplomacy_${VARIANT}_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do kill -0 "${VLLM_PID}" 2>/dev/null || break; sleep 1; done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Banner ───────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Diplomacy Inference (${VARIANT}) vs ${OPPONENT_MODEL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:    ${BASE_MODEL}"
echo "  Adapter:       ${ADAPTER_PATH}"
echo "  Skill bank:    ${COMBINED_BANK:-<none>}"
echo "  Eval module:   ${EVAL_MODULE}"
echo "  Episodes:      ${EPISODES} (${EPISODES_PER_POWER}/power × ${NUM_POWERS})"
[ "${VARIANT}" = "discrete" ] && echo "  Unchosen:      ${UNCHOSEN_STRATEGY}"
echo "  GPU(s):        ${EVAL_GPUS}"
echo "  Output:        ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[diplomacy-${VARIANT}] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" --max-model-len 8192 \
            --gpu-memory-utilization 0.85 --dtype auto --trust-remote-code \
            --enable-lora --lora-modules "${LORA_NAME}=${ADAPTER_PATH}" --max-lora-rank 16 \
        &
    VLLM_PID=$!

    MAX_WAIT=600; WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1 && break
        kill -0 "${VLLM_PID}" 2>/dev/null || { echo "[ERROR] vLLM exited unexpectedly."; exit 1; }
        sleep 5; WAITED=$((WAITED + 5))
    done
    [ ${WAITED} -ge ${MAX_WAIT} ] && echo "[ERROR] vLLM did not start within ${MAX_WAIT}s." && exit 1
    echo "[diplomacy-${VARIANT}] vLLM ready (${WAITED}s)."
else
    echo "[diplomacy-${VARIANT}] Using existing server at ${VLLM_BASE_URL}"
fi

# ── Build eval args ──────────────────────────────────────────────────
EVAL_ARGS=(
    --model "${LORA_NAME}" --episodes "${EPISODES}" --temperature "${TEMPERATURE}"
    --seed "${SEED}" --output_dir "${OUTPUT_DIR}" --opponent_model "${OPPONENT_MODEL}"
    --per_power --verbose
)

if [ "${EVAL_MODULE}" = "inference.run_qwen3_8b_eval" ]; then
    EVAL_ARGS=(--games diplomacy "${EVAL_ARGS[@]}")
fi

[ "${VARIANT}" = "discrete" ] && EVAL_ARGS+=(--unchosen_strategy "${UNCHOSEN_STRATEGY}")

if [ -n "${COMBINED_BANK}" ] && [ -f "${COMBINED_BANK}" ]; then
    EVAL_ARGS+=(--bank "${COMBINED_BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[diplomacy-${VARIANT}] python -m ${EVAL_MODULE} ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
python -m "${EVAL_MODULE}" "${EVAL_ARGS[@]}" || EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════════════════"
[ ${EXIT_CODE} -eq 0 ] && echo "  Diplomacy (${VARIANT}) inference COMPLETE" || echo "  Diplomacy (${VARIANT}) FAILED (exit ${EXIT_CODE})"
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
