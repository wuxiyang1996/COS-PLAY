#!/usr/bin/env bash
# ======================================================================
#  Inference: Avalon with best-performing Qwen3-8B checkpoint (8 episodes)
#
#  Checkpoint: runs/Qwen3-8B_avalon_20260322_200424/best/ (step 5)
#  Training mean_reward: 0.995  (best step by reward)
#  Note: adapters are from final training state (step 19); the actual
#  best-reward step was 5 but per-step checkpoint adapters were empty.
#
#  Launches a vLLM server for Qwen/Qwen3-8B with the best action_taking
#  LoRA adapter enabled and runs 8 episodes per player (40 total) on
#  Avalon using the trained skill banks.
#
#  Avalon has separate skill banks for good/evil sides. This script
#  merges them into a single combined bank at runtime.
#
#  Usage:
#    conda activate game-ai-agent
#    bash inference/infer_avalon_best.sh
#
#    # With overrides:
#    EPISODES_PER_PLAYER=16 bash inference/infer_avalon_best.sh
#    EVAL_GPUS=0 bash inference/infer_avalon_best.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8024/v1 bash inference/infer_avalon_best.sh
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
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_avalon_20260322_200424"
BEST_DIR="${RUN_DIR}/best"
ADAPTER_PATH="${BEST_DIR}/adapters/decision/action_taking"
BANK_GOOD="${BEST_DIR}/banks/avalon/good/skill_bank.jsonl"
BANK_EVIL="${BEST_DIR}/banks/avalon/evil/skill_bank.jsonl"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "[ERROR] Best adapter not found: ${ADAPTER_PATH}"
    exit 1
fi

# ── Merge good + evil skill banks into a combined file ────────────────
COMBINED_BANK="${BEST_DIR}/banks/avalon/combined_skill_bank.jsonl"
if [ -f "${BANK_GOOD}" ] && [ -f "${BANK_EVIL}" ]; then
    cat "${BANK_GOOD}" "${BANK_EVIL}" | sort -u > "${COMBINED_BANK}"
    echo "[infer_avalon_best] Merged good+evil skill banks → ${COMBINED_BANK}"
elif [ -f "${BANK_GOOD}" ]; then
    cp "${BANK_GOOD}" "${COMBINED_BANK}"
elif [ -f "${BANK_EVIL}" ]; then
    cp "${BANK_EVIL}" "${COMBINED_BANK}"
else
    echo "[WARN] No skill banks found for Avalon; running without bank."
    COMBINED_BANK=""
fi

# ── Configurable parameters ──────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LORA_NAME="qwen3-8b-avalon-best"
NUM_PLAYERS=5
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-8}"
EPISODES=$((NUM_PLAYERS * EPISODES_PER_PLAYER))
TEMPERATURE="${TEMPERATURE:-0.4}"
EVAL_GPUS="${EVAL_GPUS:-6}"
VLLM_PORT="${VLLM_PORT:-8024}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Output directory ──────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_avalon_best_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[infer_avalon_best] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[infer_avalon_best] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Avalon Inference: Qwen3-8B best checkpoint (step 5)"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:         ${BASE_MODEL}"
echo "  LoRA adapter:       ${ADAPTER_PATH}"
echo "  LoRA name:          ${LORA_NAME}"
echo "  Skill bank (good):  ${BANK_GOOD}"
echo "  Skill bank (evil):  ${BANK_EVIL}"
echo "  Combined bank:      ${COMBINED_BANK:-<none>}"
echo "  Players:            ${NUM_PLAYERS} (Merlin, 2xServant, Minion, Assassin)"
echo "  Episodes/player:    ${EPISODES_PER_PLAYER}"
echo "  Total episodes:     ${EPISODES}"
echo "  Temperature:        ${TEMPERATURE}"
echo "  GPU(s):             ${EVAL_GPUS}"
echo "  Output:             ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server with LoRA ─────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer_avalon_best] Launching vLLM server with LoRA on ${VLLM_HOST}:${VLLM_PORT}..."
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
            echo "[infer_avalon_best] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[infer_avalon_best] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[infer_avalon_best] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[infer_avalon_best] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Build evaluation command ─────────────────────────────────────────
EVAL_ARGS=(
    --games avalon
    --episodes "${EPISODES}"
    --temperature "${TEMPERATURE}"
    --model "${LORA_NAME}"
    --num_players "${NUM_PLAYERS}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
)

if [ -n "${COMBINED_BANK}" ] && [ -f "${COMBINED_BANK}" ]; then
    EVAL_ARGS+=(--bank "${COMBINED_BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[infer_avalon_best] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

# ── Run inference ────────────────────────────────────────────────────
EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Avalon best-checkpoint inference COMPLETE"
    echo "  ${EPISODES} episodes (${EPISODES_PER_PLAYER} per player x ${NUM_PLAYERS} players)"
else
    echo "  Avalon best-checkpoint inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
