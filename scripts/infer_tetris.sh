#!/usr/bin/env bash
# ======================================================================
#  Inference: Tetris with Qwen3-32B  (8 episodes)
#
#  Launches a vLLM server for Qwen/Qwen3-32B and runs 8 inference
#  episodes on Tetris using the evaluation runner.
#
#  Tetris game profile:
#    - 6 discrete actions (left, right, rotate_cw, rotate_ccw, drop, noop)
#    - 200 max steps/episode (macro-action: placement-level)
#    - Reward: +1 per piece placed, +10 per line cleared
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/infer_tetris.sh
#
#    # With overrides:
#    EPISODES=16 bash scripts/infer_tetris.sh
#    EVAL_GPUS=1 bash scripts/infer_tetris.sh
#    VLLM_BASE_URL=http://localhost:8000/v1 NO_SERVER=1 bash scripts/infer_tetris.sh
#
#    # With a trained skill bank:
#    BANK=runs/Qwen3-32B_tetris_*/skillbank/bank.jsonl bash scripts/infer_tetris.sh
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
MODEL="${MODEL:-Qwen/Qwen3-32B}"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
EVAL_GPUS="${EVAL_GPUS:-6}"
VLLM_PORT="${VLLM_PORT:-8010}"
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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/infer_tetris_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Cleanup on exit ──────────────────────────────────────────────────
# If you see "Terminated" before this message, the eval process (foreground)
# received SIGTERM/SIGINT (manual kill, scheduler preemption, tmux close, etc.),
# not necessarily a bug in Tetris.  Abrupt vLLM kills can trigger a harmless
# PyTorch NCCL "destroy_process_group" warning; we try SIGINT first so the
# OpenAI API server can shut down cleanly.
VLLM_PID=""
cleanup() {
    # Traps run with inherited set -e; be defensive.
    set +e
    echo ""
    echo "[infer_tetris] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        # Graceful: uvicorn/vLLM often handles SIGINT better than SIGTERM for NCCL.
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..45}; do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            kill -TERM "${VLLM_PID}" 2>/dev/null || true
            sleep 3
        fi
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            kill -KILL "${VLLM_PID}" 2>/dev/null || true
        fi
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[infer_tetris] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Tetris Inference: Qwen3-32B"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:          ${MODEL}"
echo "  Episodes:       ${EPISODES}"
echo "  Max steps:      ${MAX_STEPS}"
echo "  Temperature:    ${TEMPERATURE}"
echo "  GPU(s):         ${EVAL_GPUS}"
echo "  Output:         ${OUTPUT_DIR}"
if [ -n "${BANK}" ]; then
    echo "  Skill bank:     ${BANK}"
else
    echo "  Skill bank:     (none)"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    echo "[infer_tetris] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
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
            echo "[infer_tetris] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[infer_tetris] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[infer_tetris] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[infer_tetris] Using existing vLLM server at ${VLLM_BASE_URL}"
fi

# ── Build evaluation command ─────────────────────────────────────────
EVAL_ARGS=(
    --games tetris
    --episodes "${EPISODES}"
    --max_steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --model "${MODEL}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_DIR}"
)

if [ -n "${BANK}" ]; then
    EVAL_ARGS+=(--bank "${BANK}")
else
    EVAL_ARGS+=(--no-bank)
fi

echo "[infer_tetris] Command:"
echo "  python -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

# ── Run inference ────────────────────────────────────────────────────
EXIT_CODE=0
python -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Tetris inference COMPLETE"
else
    echo "  Tetris inference FAILED (exit code ${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
