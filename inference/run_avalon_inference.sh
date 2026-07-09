#!/usr/bin/env bash
# ======================================================================
#  Avalon inference with trained Qwen/Qwen3.5-9B decision agent
#
#  Variants:
#    best    — best checkpoint (self-play, no opponent model)
#    da      — decision agent vs GPT-5.4 opponents
#    matched — training-matched prompt format vs GPT-5.4
#
#  Usage:
#    bash inference/run_avalon_inference.sh --variant best
#    bash inference/run_avalon_inference.sh --variant da
#    bash inference/run_avalon_inference.sh --variant matched
#    EPISODES_PER_PLAYER=20 bash inference/run_avalon_inference.sh --variant da
#    OPPONENT_MODEL=gpt-5-mini bash inference/run_avalon_inference.sh --variant da
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
[ -z "${VARIANT}" ] && echo "Usage: $0 --variant {best|da|matched}" && exit 1

# ── Per-variant checkpoint config ────────────────────────────────────
case "${VARIANT}" in
    best)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_avalon_20260322_200424"
        CKPT_DIR="${RUN_DIR}/best"
        LORA_NAME="qwen3-8b-avalon-best"
        EVAL_MODULE="inference.run_qwen3_8b_eval"
        DEFAULT_EPS_PER_PLAYER=8
        USE_OPPONENT=0
        ;;
    da)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260326_215431"
        CKPT_DIR="${RUN_DIR}/checkpoints/step_0018"
        LORA_NAME="qwen3-8b-avalon-da"
        EVAL_MODULE="inference.run_qwen3_8b_eval"
        DEFAULT_EPS_PER_PLAYER=10
        USE_OPPONENT=1
        ;;
    matched)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260326_215431"
        CKPT_DIR="${RUN_DIR}/checkpoints/step_0018"
        LORA_NAME="qwen3-8b-avalon-matched"
        EVAL_MODULE="inference.run_qwen3_avalon_matched"
        DEFAULT_EPS_PER_PLAYER=10
        USE_OPPONENT=1
        ;;
    *)  echo "[ERROR] Unknown variant: ${VARIANT}. Use: best, da, matched"; exit 1 ;;
esac

ADAPTER_PATH="${CKPT_DIR}/adapters/decision/action_taking"
BANK_GOOD="${CKPT_DIR}/banks/avalon/good/skill_bank.jsonl"
BANK_EVIL="${CKPT_DIR}/banks/avalon/evil/skill_bank.jsonl"

[ ! -d "${ADAPTER_PATH}" ] && echo "[ERROR] Adapter not found: ${ADAPTER_PATH}" && exit 1

# ── Environment ──────────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
NUM_PLAYERS=5
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-${DEFAULT_EPS_PER_PLAYER}}"
EPISODES=$((NUM_PLAYERS * EPISODES_PER_PLAYER))
TEMPERATURE="${TEMPERATURE:-0.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
EVAL_GPUS="${EVAL_GPUS:-6}"
VLLM_PORT="${VLLM_PORT:-8024}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Merge good + evil skill banks ────────────────────────────────────
COMBINED_BANK=""
if [ -f "${BANK_GOOD}" ] && [ -f "${BANK_EVIL}" ]; then
    COMBINED_BANK="${CKPT_DIR}/banks/avalon/combined_skill_bank.jsonl"
    cat "${BANK_GOOD}" "${BANK_EVIL}" | sort -u > "${COMBINED_BANK}"
elif [ -f "${BANK_GOOD}" ]; then
    COMBINED_BANK="${BANK_GOOD}"
elif [ -f "${BANK_EVIL}" ]; then
    COMBINED_BANK="${BANK_EVIL}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_avalon_${VARIANT}_${TIMESTAMP}}"
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
echo "  Avalon Inference (${VARIANT})"
[ "${USE_OPPONENT}" = "1" ] && echo "  vs ${OPPONENT_MODEL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:    ${BASE_MODEL}"
echo "  Adapter:       ${ADAPTER_PATH}"
echo "  Skill bank:    ${COMBINED_BANK:-<none>}"
echo "  Eval module:   ${EVAL_MODULE}"
echo "  Episodes:      ${EPISODES} (${EPISODES_PER_PLAYER}/player × ${NUM_PLAYERS})"
echo "  GPU(s):        ${EVAL_GPUS}"
echo "  Output:        ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[avalon-${VARIANT}] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" --max-model-len 4096 \
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
    echo "[avalon-${VARIANT}] vLLM ready (${WAITED}s)."
else
    echo "[avalon-${VARIANT}] Using existing server at ${VLLM_BASE_URL}"
fi

# ── Build eval args ──────────────────────────────────────────────────
EVAL_ARGS=(
    --model "${LORA_NAME}" --episodes "${EPISODES}" --temperature "${TEMPERATURE}"
    --num_players "${NUM_PLAYERS}" --seed "${SEED}" --output_dir "${OUTPUT_DIR}"
    --per_role --verbose
)

# Only the standard eval module needs --games
if [ "${EVAL_MODULE}" = "inference.run_qwen3_8b_eval" ]; then
    EVAL_ARGS=(--games avalon "${EVAL_ARGS[@]}")
fi

[ "${USE_OPPONENT}" = "1" ] && EVAL_ARGS+=(--opponent_model "${OPPONENT_MODEL}")

if [ -n "${COMBINED_BANK}" ] && [ -f "${COMBINED_BANK}" ]; then
    EVAL_ARGS+=(--bank "${COMBINED_BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[avalon-${VARIANT}] python -m ${EVAL_MODULE} ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
python -m "${EVAL_MODULE}" "${EVAL_ARGS[@]}" || EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════════════════"
[ ${EXIT_CODE} -eq 0 ] && echo "  Avalon (${VARIANT}) inference COMPLETE" || echo "  Avalon (${VARIANT}) FAILED (exit ${EXIT_CODE})"
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
