#!/usr/bin/env bash
# ======================================================================
#  Single-player inference with best Qwen3-8B checkpoint
#
#  Launches a vLLM server with the best LoRA adapter and runs episodes
#  using the trained skill bank.
#
#  Supported games:  tetris | 2048 | candy_crush
#
#  Usage:
#    bash inference/run_single_player_inference.sh --game tetris
#    bash inference/run_single_player_inference.sh --game 2048
#    bash inference/run_single_player_inference.sh --game candy_crush
#    EPISODES=16 bash inference/run_single_player_inference.sh --game tetris
#    EVAL_GPUS=0 bash inference/run_single_player_inference.sh --game 2048
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8022/v1 \
#      bash inference/run_single_player_inference.sh --game tetris
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Parse arguments ──────────────────────────────────────────────────
GAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in --game) GAME="$2"; shift 2 ;; *) echo "Unknown: $1"; exit 1 ;; esac
done
[ -z "${GAME}" ] && echo "Usage: $0 --game {tetris|2048|candy_crush}" && exit 1

# ── Per-game checkpoint config ───────────────────────────────────────
case "${GAME}" in
    tetris)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_tetris_20260322_170438"
        GAME_KEY="tetris"
        LORA_NAME="qwen3-8b-tetris-best"
        VLLM_PORT="${VLLM_PORT:-8022}"
        ;;
    2048)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_2048_20260322_071227"
        GAME_KEY="twenty_forty_eight"
        LORA_NAME="qwen3-8b-2048-best"
        VLLM_PORT="${VLLM_PORT:-8020}"
        ;;
    candy_crush)
        RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260321_213813_(Candy_crush)"
        GAME_KEY="candy_crush"
        LORA_NAME="qwen3-8b-candy-crush-best"
        VLLM_PORT="${VLLM_PORT:-8021}"
        ;;
    *)  echo "[ERROR] Unknown game: ${GAME}. Use: tetris, 2048, candy_crush"; exit 1 ;;
esac

BEST_DIR="${RUN_DIR}/best"
ADAPTER_PATH="${BEST_DIR}/adapters/decision/action_taking"
BANK_PATH="${BEST_DIR}/banks/${GAME_KEY}/skill_bank.jsonl"

[ ! -d "${ADAPTER_PATH}" ] && echo "[ERROR] Adapter not found: ${ADAPTER_PATH}" && exit 1
[ ! -f "${BANK_PATH}" ] && echo "[ERROR] Skill bank not found: ${BANK_PATH}" && exit 1

# ── Environment ──────────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
EVAL_GPUS="${EVAL_GPUS:-6}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_${GAME}_best_${TIMESTAMP}}"
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
echo "  ${GAME} Inference: Qwen3-8B best checkpoint"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:   ${BASE_MODEL}"
echo "  LoRA adapter: ${ADAPTER_PATH}"
echo "  Skill bank:   ${BANK_PATH}"
echo "  Episodes:     ${EPISODES}    Max steps: ${MAX_STEPS}"
echo "  GPU(s):       ${EVAL_GPUS}"
echo "  Output:       ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer-${GAME}] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
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
    echo "[infer-${GAME}] vLLM ready (${WAITED}s)."
else
    echo "[infer-${GAME}] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Run inference ────────────────────────────────────────────────────
EVAL_ARGS=(
    --games "${GAME_KEY}" --episodes "${EPISODES}" --max_steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}" --model "${LORA_NAME}" --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}" --bank "${BANK_PATH}"
)

echo "[infer-${GAME}] python -m inference.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
python -m inference.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════════════════"
[ ${EXIT_CODE} -eq 0 ] && echo "  ${GAME} best-checkpoint inference COMPLETE" || echo "  ${GAME} inference FAILED (exit ${EXIT_CODE})"
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
