#!/bin/bash
# =============================================================================
# Qwen3-8B Decision Agent Evaluation
# =============================================================================
# Launches a vLLM server for Qwen/Qwen3-8B, then runs the evaluation script
# that collects rollouts with intention + state summary annotations.
#
# Supports 8 games across 3 environment stacks:
#   LMGame-Bench:  twenty_forty_eight, sokoban, candy_crush, tetris
#   AgentEvolver:  avalon, diplomacy
#   Orak:          super_mario, pokemon_red
#
# Uses the game-ai-agent conda environment for the vLLM server and most
# games. Super Mario is automatically run in the orak-mario conda env
# (which has gym-super-mario-bros / nes_py) and reconnects to the same
# vLLM server.
#
# GPU Layout:
#   GPU 0 (default): vLLM model server
#   Override with EVAL_GPUS env var.
#
# ======================== USAGE ==============================================
#
#   # Run on all available games, 3 episodes each (default)
#   bash scripts/run_qwen3_8b_eval.sh
#
#   # LMGame-Bench games only
#   bash scripts/run_qwen3_8b_eval.sh --games twenty_forty_eight sokoban candy_crush tetris
#
#   # All 8 games (LMGame-Bench + AgentEvolver + Orak)
#   bash scripts/run_qwen3_8b_eval.sh --games twenty_forty_eight sokoban candy_crush tetris avalon diplomacy super_mario pokemon_red
#
#   # AgentEvolver games only
#   bash scripts/run_qwen3_8b_eval.sh --games avalon diplomacy --episodes 3
#
#   # Orak games only
#   bash scripts/run_qwen3_8b_eval.sh --games super_mario pokemon_red --episodes 3
#
#   # More episodes
#   bash scripts/run_qwen3_8b_eval.sh --episodes 10
#
#   # Custom model path (local checkpoint instead of HF hub)
#   bash scripts/run_qwen3_8b_eval.sh --model /path/to/checkpoint
#
#   # Run on specific GPU(s)
#   bash scripts/run_qwen3_8b_eval.sh --gpu 1
#   bash scripts/run_qwen3_8b_eval.sh --gpu 0,1 --tp 2
#
#   # Skip vLLM launch (server already running externally)
#   VLLM_BASE_URL="http://localhost:8000/v1" \
#       bash scripts/run_qwen3_8b_eval.sh --no-server
#
#   # Resume interrupted run
#   bash scripts/run_qwen3_8b_eval.sh --resume
#
#   # Verbose step-by-step output
#   bash scripts/run_qwen3_8b_eval.sh --games tetris --episodes 2 -v
#
# =============================================================================

set -e

# Require bash (arrays and other syntax used below); re-exec if run with sh
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Conda setup
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /workspace/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[eval.sh] Activated conda env: $CONDA_ENV - $(python --version 2>&1)"

# ---------------------------------------------------------------------------
# PYTHONPATH — include all env stacks so the eval script can import any wrapper
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

GAMINGAGENT_DIR="${REPO_ROOT}/../GamingAgent"
if [ -d "$GAMINGAGENT_DIR" ]; then
    export PYTHONPATH="${GAMINGAGENT_DIR}:${PYTHONPATH}"
fi

# AgentEvolver (Avalon engine)
AGENTEVOLVER_DIR="${REPO_ROOT}/../AgentEvolver"
if [ -d "$AGENTEVOLVER_DIR" ]; then
    export PYTHONPATH="${AGENTEVOLVER_DIR}:${PYTHONPATH}"
fi

# AI_Diplomacy (Diplomacy engine)
AI_DIPLOMACY_DIR="${REPO_ROOT}/../AI_Diplomacy"
if [ -d "$AI_DIPLOMACY_DIR" ]; then
    export PYTHONPATH="${AI_DIPLOMACY_DIR}:${PYTHONPATH}"
fi

# Orak (Super Mario, Pokemon Red)
ORAK_SRC_DIR="${REPO_ROOT}/../Orak/src"
if [ -d "$ORAK_SRC_DIR" ]; then
    export PYTHONPATH="${ORAK_SRC_DIR}:${PYTHONPATH}"
fi

# Headless display for Orak games that use pygame/SDL (Super Mario)
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

# ---------------------------------------------------------------------------
# HuggingFace cache — keep model weights on /workspace so they persist
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HUB_CACHE"

# Remove Qwen3-8B from the default home cache if present (avoid stale/corrupt files)
DEFAULT_HF_CACHE="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-8B"
if [ -e "$DEFAULT_HF_CACHE" ]; then
    echo "[eval.sh] Removing initial-path cache: $DEFAULT_HF_CACHE"
    rm -rf "$DEFAULT_HF_CACHE"
fi

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
# Parse shell-only flags (--no-server, --gpu, --tp) before forwarding to Python
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
        *)
            PYTHON_ARGS+=("$arg")
            ;;
    esac
done

# Extract --model from args if provided (to use for vLLM server)
for i in "${!PYTHON_ARGS[@]}"; do
    if [ "${PYTHON_ARGS[$i]}" = "--model" ]; then
        next=$((i + 1))
        if [ $next -lt ${#PYTHON_ARGS[@]} ]; then
            MODEL="${PYTHON_ARGS[$next]}"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Cleanup: kill vLLM server on exit
# ---------------------------------------------------------------------------
VLLM_PID=""

cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[eval.sh] Shutting down vLLM server (PID=$VLLM_PID)..."
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

    echo "[eval.sh] vLLM server starting (PID=$VLLM_PID), waiting for it to be ready..."
    MAX_WAIT=900
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[eval.sh] vLLM server is ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[eval.sh] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[eval.sh] Still waiting for vLLM... ${WAITED}s / ${MAX_WAIT}s"
        fi
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[eval.sh] ERROR: vLLM server did not become ready within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "============================================"
    echo "  Skipping vLLM launch (--no-server)"
    echo "  Using VLLM_BASE_URL=$VLLM_BASE_URL"
    echo "============================================"
fi

# ---------------------------------------------------------------------------
# Detect games that need the orak-mario conda env (super_mario)
# ---------------------------------------------------------------------------
# Games that MUST run in orak-mario (gym-super-mario-bros / nes_py).
ORAK_MARIO_GAMES=("super_mario")
ORAK_MARIO_ENV="orak-mario"

# Parse --games from PYTHON_ARGS to separate env-specific games.
MAIN_GAMES=()         # games for game-ai-agent env
MARIO_GAMES=()        # games for orak-mario env
HAS_GAMES_FLAG=false
IN_GAMES_SECTION=false
OTHER_ARGS=()         # non-game args to forward to both runs

_is_orak_mario_game() {
    local g="$1"
    for mg in "${ORAK_MARIO_GAMES[@]}"; do
        [ "$g" = "$mg" ] && return 0
    done
    return 1
}

for arg in "${PYTHON_ARGS[@]}"; do
    if [ "$arg" = "--games" ]; then
        HAS_GAMES_FLAG=true
        IN_GAMES_SECTION=true
        continue
    fi
    if [ "$IN_GAMES_SECTION" = true ]; then
        # A new flag means the games list ended
        if [[ "$arg" == --* ]]; then
            IN_GAMES_SECTION=false
            OTHER_ARGS+=("$arg")
        elif _is_orak_mario_game "$arg"; then
            MARIO_GAMES+=("$arg")
        else
            MAIN_GAMES+=("$arg")
        fi
    else
        OTHER_ARGS+=("$arg")
    fi
done

# When --games was NOT specified, default: run all available games in the
# main env, and additionally run super_mario in orak-mario if that env exists.
if [ "$HAS_GAMES_FLAG" = false ]; then
    MARIO_GAMES=("super_mario")
fi

# Check orak-mario env availability; skip mario games if missing
if [ ${#MARIO_GAMES[@]} -gt 0 ]; then
    if ! conda env list 2>/dev/null | grep -q "^${ORAK_MARIO_ENV} "; then
        echo "[eval.sh] WARNING: conda env '${ORAK_MARIO_ENV}' not found — skipping ${MARIO_GAMES[*]}"
        MARIO_GAMES=()
    fi
fi

# Build a shared output dir so both invocations write to the same place
MODEL_SLUG="$(echo "${MODEL##*/}" | sed 's/[^a-zA-Z0-9_.-]/_/g')"
# Storage: output/<model>/<game>/<timestamp>; Python sets timestamp at run start
SHARED_OUTPUT_DIR="${REPO_ROOT}/output/${MODEL_SLUG}"
mkdir -p "$SHARED_OUTPUT_DIR"

# If the user already passed --output_dir, honour it
for i in "${!OTHER_ARGS[@]}"; do
    if [ "${OTHER_ARGS[$i]}" = "--output_dir" ]; then
        next=$((i + 1))
        if [ $next -lt ${#OTHER_ARGS[@]} ]; then
            SHARED_OUTPUT_DIR="${OTHER_ARGS[$next]}"
        fi
    fi
done

# Inject --output_dir into OTHER_ARGS if not already present
_has_output_dir=false
for oa in "${OTHER_ARGS[@]}"; do
    [ "$oa" = "--output_dir" ] && _has_output_dir=true
done
if [ "$_has_output_dir" = false ]; then
    OTHER_ARGS+=("--output_dir" "$SHARED_OUTPUT_DIR")
fi

EXIT_CODE=0

# ---------------------------------------------------------------------------
# Run main games (game-ai-agent env)
# ---------------------------------------------------------------------------
_build_main_args() {
    local args=()
    if [ "$HAS_GAMES_FLAG" = true ] && [ ${#MAIN_GAMES[@]} -gt 0 ]; then
        args+=("--games" "${MAIN_GAMES[@]}")
    elif [ "$HAS_GAMES_FLAG" = true ] && [ ${#MAIN_GAMES[@]} -eq 0 ]; then
        # User only asked for orak-mario games — skip main run
        return 1
    fi
    # When --games was NOT specified, let the Python script discover defaults
    # (it will auto-skip super_mario since gym-super-mario-bros is missing)
    args+=("${OTHER_ARGS[@]}")
    echo "${args[@]}"
}

MAIN_RUN_ARGS="$(_build_main_args)" && RUN_MAIN=true || RUN_MAIN=false

if [ "$RUN_MAIN" = true ]; then
    echo ""
    echo "============================================"
    echo "  Qwen3-8B Evaluation  (game-ai-agent env)"
    echo "============================================"
    echo "  VLLM_BASE_URL: $VLLM_BASE_URL"
    echo "  Output:        $SHARED_OUTPUT_DIR"
    echo "  Args:          $MAIN_RUN_ARGS"
    echo "============================================"
    echo ""

    # shellcheck disable=SC2086
    python -m scripts.run_qwen3_8b_eval $MAIN_RUN_ARGS
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[eval.sh] Main evaluation exited with code $EXIT_CODE."
    fi
fi

# ---------------------------------------------------------------------------
# Run Super Mario in orak-mario env
# ---------------------------------------------------------------------------
if [ ${#MARIO_GAMES[@]} -gt 0 ]; then
    echo ""
    echo "============================================"
    echo "  Switching to conda env: $ORAK_MARIO_ENV"
    echo "  Games: ${MARIO_GAMES[*]}"
    echo "============================================"

    conda activate "$ORAK_MARIO_ENV"
    echo "[eval.sh] Activated conda env: $ORAK_MARIO_ENV - $(python --version 2>&1)"

    # Re-export PYTHONPATH for orak-mario (same layout)
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    [ -d "$GAMINGAGENT_DIR" ] && export PYTHONPATH="${GAMINGAGENT_DIR}:${PYTHONPATH}"
    [ -d "$AGENTEVOLVER_DIR" ] && export PYTHONPATH="${AGENTEVOLVER_DIR}:${PYTHONPATH}"
    [ -d "$AI_DIPLOMACY_DIR" ] && export PYTHONPATH="${AI_DIPLOMACY_DIR}:${PYTHONPATH}"
    [ -d "$ORAK_SRC_DIR" ] && export PYTHONPATH="${ORAK_SRC_DIR}:${PYTHONPATH}"

    export SDL_VIDEODRIVER=dummy

    # Ensure Xvfb is running for nes_py / pyglet
    if [ -z "${DISPLAY:-}" ]; then
        if ! pgrep -x Xvfb >/dev/null 2>&1; then
            Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
            sleep 0.5
        fi
        export DISPLAY=:99
    fi

    MARIO_RUN_ARGS=("--games" "${MARIO_GAMES[@]}" "${OTHER_ARGS[@]}")

    echo ""
    echo "============================================"
    echo "  Qwen3-8B Evaluation  ($ORAK_MARIO_ENV env)"
    echo "============================================"
    echo "  VLLM_BASE_URL: $VLLM_BASE_URL"
    echo "  Output:        $SHARED_OUTPUT_DIR"
    echo "  Args:          ${MARIO_RUN_ARGS[*]}"
    echo "============================================"
    echo ""

    python -m scripts.run_qwen3_8b_eval "${MARIO_RUN_ARGS[@]}"
    MARIO_EXIT=$?

    if [ $MARIO_EXIT -ne 0 ]; then
        echo "[eval.sh] orak-mario evaluation exited with code $MARIO_EXIT."
        [ $EXIT_CODE -eq 0 ] && EXIT_CODE=$MARIO_EXIT
    fi

    # Switch back to main env
    conda activate "$CONDA_ENV"
fi

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[eval.sh] Evaluation completed successfully."
else
    echo "[eval.sh] Evaluation exited with code $EXIT_CODE."
fi
echo "[eval.sh] Output: $SHARED_OUTPUT_DIR"

exit $EXIT_CODE
