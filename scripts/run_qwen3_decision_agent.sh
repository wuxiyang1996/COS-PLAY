#!/bin/bash
# =============================================================================
# Qwen3-8B Decision Agent with Skill Bank
# =============================================================================
# Launches a vLLM server for Qwen/Qwen3-8B, then runs the decision agent
# that plays games using skills from labeling/output/gpt54_skillbank/ to guide
# its action selection.
#
# Output goes to: test_rollout/decision_agent/<game>/<timestamp>/
#
# ======================== USAGE ==============================================
#
#   # All available games, 3 episodes each (default)
#   bash scripts/run_qwen3_decision_agent.sh
#
#   # Single game
#   bash scripts/run_qwen3_decision_agent.sh --games twenty_forty_eight --episodes 5
#
#   # One episode per game (run each game once), verbose
#   bash scripts/run_qwen3_decision_agent.sh --one_per_game -v
#
#   # All games, verbose
#   bash scripts/run_qwen3_decision_agent.sh --episodes 3 -v
#
#   # Specify GPU for vLLM server
#   bash scripts/run_qwen3_decision_agent.sh --gpu 0 --one_per_game -v
#   bash scripts/run_qwen3_decision_agent.sh --gpu 0,1 --tp 2
#
#   # Without skill bank (baseline comparison)
#   bash scripts/run_qwen3_decision_agent.sh --no-bank --episodes 3
#
#   # Custom skill bank path
#   bash scripts/run_qwen3_decision_agent.sh --bank /path/to/bank --episodes 3
#
#   # Skip vLLM launch (server already running)
#   bash scripts/run_qwen3_decision_agent.sh --no-server --episodes 3
#
# =============================================================================

set -e

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /workspace/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[decision.sh] Activated conda env: $CONDA_ENV - $(python --version 2>&1)"

# ---------------------------------------------------------------------------
# PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

GAMINGAGENT_DIR="${REPO_ROOT}/../GamingAgent"
if [ -d "$GAMINGAGENT_DIR" ]; then
    export PYTHONPATH="${GAMINGAGENT_DIR}:${PYTHONPATH}"
fi

# ---------------------------------------------------------------------------
# HuggingFace cache
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HUB_CACHE"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
EVAL_GPUS="${EVAL_GPUS:-0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
LAUNCH_SERVER=true
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Parse shell-only flags before forwarding rest to Python
# ---------------------------------------------------------------------------
PYTHON_ARGS=()
_skip_next=false
for i in $(seq 1 $#); do
    arg="${!i}"
    if [ "$_skip_next" = true ]; then
        _skip_next=false
        continue
    fi
    next_i=$((i + 1))
    next_arg="${!next_i:-}"
    case "$arg" in
        --no-server)
            LAUNCH_SERVER=false
            ;;
        --gpu)
            EVAL_GPUS="$next_arg"
            _skip_next=true
            ;;
        --tp)
            TENSOR_PARALLEL="$next_arg"
            _skip_next=true
            ;;
        --model)
            MODEL="$next_arg"
            PYTHON_ARGS+=("$arg")
            ;;
        *)
            PYTHON_ARGS+=("$arg")
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Cleanup: kill vLLM on exit
# ---------------------------------------------------------------------------
VLLM_PID=""

cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[decision.sh] Shutting down vLLM server (PID=$VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Launch vLLM server
# ---------------------------------------------------------------------------
if [ "$LAUNCH_SERVER" = true ]; then
    echo "============================================"
    echo "  Launching vLLM server"
    echo "============================================"
    echo "  Model:  $MODEL"
    echo "  Host:   $VLLM_HOST:$VLLM_PORT"
    echo "  GPU(s): $EVAL_GPUS (TP=$TENSOR_PARALLEL)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --host "$VLLM_HOST" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size "$TENSOR_PARALLEL" \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --trust-remote-code \
        &
    VLLM_PID=$!

    echo "[decision.sh] vLLM server starting (PID=$VLLM_PID), waiting for ready..."
    MAX_WAIT=600
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[decision.sh] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[decision.sh] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[decision.sh] Still waiting for vLLM... ${WAITED}s / ${MAX_WAIT}s"
        fi
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[decision.sh] ERROR: vLLM server did not become ready within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "============================================"
    echo "  Skipping vLLM launch (--no-server)"
    echo "  Using VLLM_BASE_URL=$VLLM_BASE_URL"
    echo "============================================"
fi

# ---------------------------------------------------------------------------
# Run decision agent
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Running Qwen3-8B Decision Agent"
echo "  Skill Bank: labeling/output/gpt54_skillbank"
echo "  Output:     test_rollout/decision_agent"
echo "============================================"

python -m scripts.qwen3_decision_agent "${PYTHON_ARGS[@]}"

echo ""
echo "[decision.sh] Done."
