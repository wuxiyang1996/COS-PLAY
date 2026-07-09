#!/usr/bin/env bash
# ======================================================================
#  Inference: Avalon with Qwen3-8B  (8 episodes per player = 40 total)
#
#  Launches a vLLM server for Qwen/Qwen3-8B and runs 40 inference
#  episodes on Avalon (8 episodes x 5 players).  The agent cycles
#  through all 5 roles deterministically.
#
#  Avalon game profile:
#    - 5-player social deduction game
#    - Roles: Merlin, 2x Servant (good side) vs Minion, Assassin (evil side)
#    - Good side: Merlin (knows evil), Servants (no info)
#    - Evil side: Minion (knows evil), Assassin (knows evil, can kill Merlin)
#    - 50 max steps/episode (~5 rounds of proposals + missions)
#    - Reward: win/loss outcome + role-specific bonuses
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/infer_avalon.sh
#
#    # With overrides:
#    EPISODES_PER_PLAYER=16 bash scripts/infer_avalon.sh
#    EVAL_GPUS=0,1 TENSOR_PARALLEL=2 bash scripts/infer_avalon.sh
#    NO_SERVER=1 bash scripts/infer_avalon.sh
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

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-8B}"
NUM_PLAYERS=5
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-8}"
EPISODES=$((NUM_PLAYERS * EPISODES_PER_PLAYER))
TEMPERATURE="${TEMPERATURE:-0.4}"
EVAL_GPUS="${EVAL_GPUS:-5}"
VLLM_PORT="${VLLM_PORT:-8015}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
BANK="${BANK:-}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Output directory ──────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_avalon_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[infer_avalon] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[infer_avalon] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Avalon Inference: Qwen3-8B"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:              ${MODEL}"
echo "  Players:            ${NUM_PLAYERS} (Merlin, 2xServant, Minion, Assassin)"
echo "  Episodes/player:    ${EPISODES_PER_PLAYER}"
echo "  Total episodes:     ${EPISODES}"
echo "  Temperature:        ${TEMPERATURE}"
echo "  GPU(s):             ${EVAL_GPUS}"
echo "  Output:             ${OUTPUT_DIR}"
if [ -n "${BANK}" ]; then
    echo "  Skill bank:         ${BANK}"
else
    echo "  Skill bank:         (none)"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer_avalon] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --trust-remote-code \
        &
    VLLM_PID=$!

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[infer_avalon] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[infer_avalon] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[infer_avalon] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[infer_avalon] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Build evaluation command ─────────────────────────────────────────
EVAL_ARGS=(
    --games avalon
    --episodes "${EPISODES}"
    --temperature "${TEMPERATURE}"
    --model "${MODEL}"
    --num_players "${NUM_PLAYERS}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
)

if [ -n "${BANK}" ]; then
    EVAL_ARGS+=(--bank "${BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[infer_avalon] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

# ── Run inference ────────────────────────────────────────────────────
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Avalon inference COMPLETE"
    echo "  ${EPISODES} episodes (${EPISODES_PER_PLAYER} per player x ${NUM_PLAYERS} players)"
else
    echo "  Avalon inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
