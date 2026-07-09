#!/usr/bin/env bash
# ======================================================================
#  Inference: Avalon Decision Agent + Skill Bank vs GPT-5.4 Opponents
#
#  Checkpoint: runs/Qwen3-8B_20260326_215431/checkpoints/step_0018
#  Training mean_reward: 0.878  (best step by reward, 40 episodes)
#
#  Launches a vLLM server for Qwen/Qwen3-8B with the best action_taking
#  LoRA adapter and runs episodes on Avalon in mixed-model mode:
#  the controlled player uses the trained decision agent with skill bank
#  guidance; all opponents are controlled by GPT-5.4 via OpenRouter API.
#
#  Each episode cycles the controlled player through 0..4 (--per_role),
#  so the agent is evaluated from every seat/role.
#
#  Avalon has separate skill banks for good/evil sides. This script
#  merges them into a single combined bank at runtime.
#
#  Usage:
#    conda activate game-ai-agent
#    bash inference/infer_avalon_decision_agent.sh
#
#    # With overrides:
#    EPISODES_PER_PLAYER=20 bash inference/infer_avalon_decision_agent.sh
#    OPPONENT_MODEL=gpt-5-mini bash inference/infer_avalon_decision_agent.sh
#    EVAL_GPUS=0 bash inference/infer_avalon_decision_agent.sh
#    NO_SERVER=1 VLLM_BASE_URL=http://localhost:8024/v1 bash inference/infer_avalon_decision_agent.sh
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
RUN_DIR="${PROJECT_ROOT}/runs/Qwen3-8B_20260326_215431"
BEST_DIR="${RUN_DIR}/checkpoints/step_0018"
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
    echo "[infer_avalon_da] Merged good+evil skill banks → ${COMBINED_BANK}"
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
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-10}"
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

# ── Output directory ──────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OPP_TAG="$(echo "${OPPONENT_MODEL}" | tr -d '.' | tr -cd 'a-zA-Z0-9_')"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_avalon_da_vs_${OPP_TAG}_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[infer_avalon_da] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[infer_avalon_da] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Avalon Decision Agent vs ${OPPONENT_MODEL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:         ${BASE_MODEL}"
echo "  LoRA adapter:       ${ADAPTER_PATH}"
echo "  LoRA name:          ${LORA_NAME}"
echo "  Skill bank (good):  ${BANK_GOOD}"
echo "  Skill bank (evil):  ${BANK_EVIL}"
echo "  Combined bank:      ${COMBINED_BANK:-<none>}"
echo "  Opponent model:     ${OPPONENT_MODEL}"
echo "  Players:            ${NUM_PLAYERS} (Merlin, 2×Servant, Minion, Assassin)"
echo "  Episodes/player:    ${EPISODES_PER_PLAYER}"
echo "  Total episodes:     ${EPISODES}"
echo "  Temperature:        ${TEMPERATURE}"
echo "  Seed:               ${SEED}"
echo "  GPU(s):             ${EVAL_GPUS}"
echo "  Output:             ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server with LoRA ─────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer_avalon_da] Launching vLLM server with LoRA on ${VLLM_HOST}:${VLLM_PORT}..."
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
            echo "[infer_avalon_da] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[infer_avalon_da] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[infer_avalon_da] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[infer_avalon_da] Using existing vLLM server at ${VLLM_BASE_URL}"
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
    --opponent_model "${OPPONENT_MODEL}"
    --per_role
    --verbose
)

if [ -n "${COMBINED_BANK}" ] && [ -f "${COMBINED_BANK}" ]; then
    EVAL_ARGS+=(--bank "${COMBINED_BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[infer_avalon_da] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

# ── Run inference ────────────────────────────────────────────────────
EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Avalon DA vs ${OPPONENT_MODEL} inference COMPLETE"
    echo "  ${EPISODES} episodes (${EPISODES_PER_PLAYER} per player × ${NUM_PLAYERS} players)"
else
    echo "  Avalon DA vs ${OPPONENT_MODEL} inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
