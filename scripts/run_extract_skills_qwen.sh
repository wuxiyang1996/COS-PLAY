#!/bin/bash
# =============================================================================
# Qwen3-8B Skill Extraction
# =============================================================================
# Launches a vLLM server for Qwen/Qwen3-8B, then runs skill extraction on
# labeled episode trajectories using the SkillBankAgent pipeline with the
# IntentionSignalExtractor (Strategy B).
#
# Reads pre-labeled episodes (with [TAG] intentions) from:
#   labeling/output/gpt54_skills/
#
# Outputs to:
#   scripts/output/qwen_skills/<game>/skill_bank.jsonl
#   scripts/output/qwen_skills/<game>/skill_catalog.json
#   scripts/output/qwen_skills/skill_catalog_all.json
#
# ======================== USAGE ==============================================
#
#   # All games, one episode each (quick test)
#   bash scripts/run_extract_skills_qwen.sh --one_per_game -v
#
#   # All games, all episodes
#   bash scripts/run_extract_skills_qwen.sh
#
#   # Specific games
#   bash scripts/run_extract_skills_qwen.sh --games candy_crush tetris -v
#
#   # More episodes per game
#   bash scripts/run_extract_skills_qwen.sh --max_episodes 5
#
#   # Custom input (e.g. from Qwen eval rollouts)
#   bash scripts/run_extract_skills_qwen.sh --input_dir output/Qwen3-8B
#
#   # Skip vLLM launch (server already running)
#   bash scripts/run_extract_skills_qwen.sh --no-server --one_per_game -v
#
#   # Use different GPU(s)
#   bash scripts/run_extract_skills_qwen.sh --gpu 2 --one_per_game -v
#   bash scripts/run_extract_skills_qwen.sh --gpu 0,1 --tp 2
#
#   # Dry run (preview, no saves)
#   bash scripts/run_extract_skills_qwen.sh --dry_run --one_per_game
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

echo "[skills.sh] Activated conda env: $CONDA_ENV - $(python --version 2>&1)"

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
        echo "[skills.sh] Shutting down vLLM server (PID=$VLLM_PID)..."
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
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --trust-remote-code \
        &
    VLLM_PID=$!

    echo "[skills.sh] vLLM server starting (PID=$VLLM_PID), waiting for ready..."
    MAX_WAIT=600
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[skills.sh] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[skills.sh] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[skills.sh] Still waiting for vLLM... ${WAITED}s / ${MAX_WAIT}s"
        fi
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[skills.sh] ERROR: vLLM server did not become ready within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "============================================"
    echo "  Skipping vLLM launch (--no-server)"
    echo "  Using VLLM_BASE_URL=$VLLM_BASE_URL"
    echo "============================================"
fi

# ---------------------------------------------------------------------------
# Run skill extraction
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Running skill extraction"
echo "============================================"

python -m scripts.extract_skills_qwen "${PYTHON_ARGS[@]}"

echo ""
echo "[skills.sh] Done."
