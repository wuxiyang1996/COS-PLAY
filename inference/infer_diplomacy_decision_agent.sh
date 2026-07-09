#!/usr/bin/env bash
# ======================================================================
#  Inference: Diplomacy Decision Agent + Skill Bank vs GPT-5.4 Opponents
#
#  Checkpoint: runs/Qwen3-8B_20260327_062035/checkpoints/step_0017
#  Training mean_reward: 4.935  (max 7.72, min 2.11, 28 episodes)
#  62 total skills across 7 powers (8+9+8+10+8+9+10)
#
#  Launches a vLLM server for Qwen/Qwen3-8B with the best action_taking
#  LoRA adapter and runs 10 episodes per power (70 total) on Diplomacy.
#  The controlled power uses the trained decision agent with skill bank
#  guidance; all opponents are controlled by GPT-5.4 via OpenRouter API.
#
#  Each episode cycles the controlled power through the 7 standard powers
#  (--per_power), so the agent is evaluated from every position.
#
#  Usage:
#    conda activate game-ai-agent
#    bash inference/infer_diplomacy_decision_agent.sh
#
#    # With overrides:
#    EPISODES_PER_POWER=20 bash inference/infer_diplomacy_decision_agent.sh
#    OPPONENT_MODEL=gpt-5-mini bash inference/infer_diplomacy_decision_agent.sh
#    EVAL_GPUS=0,1 TENSOR_PARALLEL=2 bash inference/infer_diplomacy_decision_agent.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8025/v1 bash inference/infer_diplomacy_decision_agent.sh
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
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260327_062035"
BEST_DIR="${RUN_DIR}/checkpoints/step_0017"
ADAPTER_PATH="${BEST_DIR}/adapters/decision/action_taking"
BANKS_DIR="${BEST_DIR}/banks/diplomacy"

if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "[ERROR] Best adapter not found: ${ADAPTER_PATH}"
    exit 1
fi

# ── Merge per-power skill banks into a combined file ──────────────────
COMBINED_BANK="${BANKS_DIR}/combined_skill_bank.jsonl"
BANK_FILES=()
for power in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY; do
    pbank="${BANKS_DIR}/${power}/skill_bank.jsonl"
    if [ -f "${pbank}" ]; then
        BANK_FILES+=("${pbank}")
    fi
done

if [ ${#BANK_FILES[@]} -gt 0 ]; then
    cat "${BANK_FILES[@]}" | sort -u > "${COMBINED_BANK}"
    echo "[infer_diplomacy_da] Merged ${#BANK_FILES[@]} power skill banks → ${COMBINED_BANK}"
else
    echo "[WARN] No per-power skill banks found; running without bank."
    COMBINED_BANK=""
fi

# ── API key for GPT-5.4 opponents ────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
[ -z "${OPENAI_API_KEY:-}" ] && export OPENAI_API_KEY="${OPENROUTER_API_KEY:-}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] No API key found for GPT-5.4 opponents."
    echo "  Set OPENROUTER_API_KEY or OPENAI_API_KEY, or configure api_keys.py."
    exit 1
fi

# ── Configurable parameters ──────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LORA_NAME="qwen3-8b-diplomacy-best"
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

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Output directory ──────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_diplomacy_da_vs_gpt54_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[infer_diplomacy_da] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[infer_diplomacy_da] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Diplomacy Decision Agent + Skill Bank vs ${OPPONENT_MODEL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:         ${BASE_MODEL}"
echo "  LoRA adapter:       ${ADAPTER_PATH}"
echo "  LoRA name:          ${LORA_NAME}"
echo "  Power banks dir:    ${BANKS_DIR}"
echo "  Combined bank:      ${COMBINED_BANK:-<none>}"
echo "  Opponent model:     ${OPPONENT_MODEL}"
echo "  Powers:             ${NUM_POWERS} (Austria, England, France, Germany, Italy, Russia, Turkey)"
echo "  Mode:               per_power (cycle controlled power)"
echo "  Episodes/power:     ${EPISODES_PER_POWER}"
echo "  Total episodes:     ${EPISODES}"
echo "  Temperature:        ${TEMPERATURE}"
echo "  GPU(s):             ${EVAL_GPUS}"
echo "  Output:             ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server with LoRA ─────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer_diplomacy_da] Launching vLLM server with LoRA on ${VLLM_HOST}:${VLLM_PORT}..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 8192 \
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
            echo "[infer_diplomacy_da] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[infer_diplomacy_da] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[infer_diplomacy_da] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[infer_diplomacy_da] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Build evaluation command ─────────────────────────────────────────
EVAL_ARGS=(
    --games diplomacy
    --episodes "${EPISODES}"
    --temperature "${TEMPERATURE}"
    --model "${LORA_NAME}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
    --opponent_model "${OPPONENT_MODEL}"
    --per_power
    --verbose
)

if [ -n "${COMBINED_BANK}" ] && [ -f "${COMBINED_BANK}" ]; then
    EVAL_ARGS+=(--bank "${COMBINED_BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[infer_diplomacy_da] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

# ── Run inference ────────────────────────────────────────────────────
EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Diplomacy Decision Agent vs ${OPPONENT_MODEL} — COMPLETE"
    echo "  ${EPISODES} episodes (${EPISODES_PER_POWER} per power x ${NUM_POWERS} powers)"
else
    echo "  Diplomacy Decision Agent inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
