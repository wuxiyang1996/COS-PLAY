#!/bin/bash
# =============================================================================
# Qwen3-8B Skill Bank Agent — Extract Skills from GPT-5.4 Rollouts
# =============================================================================
# Launches a vLLM server for Qwen/Qwen3-8B, then runs skill extraction on
# labeled rollouts from labeling/output/gpt54/ using the SkillBankAgent
# pipeline. Same functionality as extract_skillbank_gpt54.py but using
# Qwen3-8B as the LLM backend.
#
# Input:  labeling/output/gpt54/<game>/episode_*.json
# Output: test_rollout/skillbank_agent/<game>/
#           - skill_bank.jsonl
#           - skill_catalog.json
#           - sub_episodes.json
#           - extraction_summary.json
#         test_rollout/skillbank_agent/
#           - skill_catalog_all.json
#           - skill_archetypes.json
#
# ======================== USAGE ==============================================
#
#   # All games, all episodes
#   bash scripts/run_qwen3_skillbank_agent.sh
#
#   # Quick test: one episode per game, verbose
#   bash scripts/run_qwen3_skillbank_agent.sh --one_per_game -v
#
#   # Specific games
#   bash scripts/run_qwen3_skillbank_agent.sh --games tetris twenty_forty_eight
#
#   # More episodes per game
#   bash scripts/run_qwen3_skillbank_agent.sh --max_episodes 5
#
#   # Custom input dir
#   bash scripts/run_qwen3_skillbank_agent.sh --input_dir labeling/output/gpt54
#
#   # Skip vLLM launch (server already running)
#   bash scripts/run_qwen3_skillbank_agent.sh --no-server --one_per_game -v
#
#   # Use different GPU(s)
#   bash scripts/run_qwen3_skillbank_agent.sh --gpu 1
#   bash scripts/run_qwen3_skillbank_agent.sh --gpu 0,1 --tp 2
#
#   # Dry run
#   bash scripts/run_qwen3_skillbank_agent.sh --dry_run --one_per_game
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

echo "[skillbank.sh] Activated conda env: $CONDA_ENV - $(python --version 2>&1)"

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
        echo "[skillbank.sh] Shutting down vLLM server (PID=$VLLM_PID)..."
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
            --gpu-memory-utilization 0.75 \
            --dtype auto \
            --trust-remote-code \
        &
    VLLM_PID=$!

    echo "[skillbank.sh] vLLM server starting (PID=$VLLM_PID), waiting for ready..."
    MAX_WAIT=600
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[skillbank.sh] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[skillbank.sh] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[skillbank.sh] Still waiting for vLLM... ${WAITED}s / ${MAX_WAIT}s"
        fi
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[skillbank.sh] ERROR: vLLM server did not become ready within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "============================================"
    echo "  Skipping vLLM launch (--no-server)"
    echo "  Using VLLM_BASE_URL=$VLLM_BASE_URL"
    echo "============================================"
fi

# ---------------------------------------------------------------------------
# Run skill bank extraction
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Running Qwen3-8B Skill Bank Extraction"
echo "  Input:  labeling/output/gpt54"
echo "  Output: test_rollout/skillbank_agent"
echo "============================================"

python -m scripts.qwen3_skillbank_agent "${PYTHON_ARGS[@]}"

echo ""
echo "[skillbank.sh] Done."
