#!/bin/bash
# =============================================================================
# Inference Runner: Decision Agent + Skill Bank → Game Environments
# =============================================================================
# Runs the trained Decision Agent with the stored Skill Bank on game
# environments and collects rollouts + evaluation metrics.
#
# Supports three usage modes:
#   1. Explicit paths — specify model checkpoint and bank file directly.
#   2. Co-evolution shortcut — point at a co-evolution output directory and
#      an iteration number; paths are resolved automatically.
#   3. VERL inference — delegates to inference.run_verl_inference for
#      vLLM/sglang-based evaluation (same env+reward as training).
#
# GPU Layout:
#   Uses 1 GPU by default (CUDA_VISIBLE_DEVICES=0).  Override with
#   INFERENCE_GPUS env var for multi-GPU tensor-parallel inference.
#
# ======================== USAGE ==============================================
#
#   # Basic: explicit model + bank
#   bash scripts/run_inference.sh \
#       --model runs/coevolution/models/decision_v3/global_step_20/actor/huggingface \
#       --bank  runs/coevolution/skillbank/bank.jsonl \
#       --games twenty_forty_eight sokoban \
#       --episodes 10
#
#   # Co-evolution shortcut (latest iteration)
#   bash scripts/run_inference.sh \
#       --coevo-dir runs/coevolution \
#       --episodes 20
#
#   # Co-evolution shortcut (specific iteration)
#   bash scripts/run_inference.sh \
#       --coevo-dir runs/coevolution \
#       --iteration 3 \
#       --episodes 20
#
#   # VERL-based inference
#   bash scripts/run_inference.sh --verl
#
#   # No skill bank (baseline)
#   bash scripts/run_inference.sh --model gpt-4o-mini --no-bank --episodes 5
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Try to add GamingAgent to PYTHONPATH if it exists
GAMINGAGENT_DIR="${REPO_ROOT}/../GamingAgent"
if [ -d "$GAMINGAGENT_DIR" ]; then
    export PYTHONPATH="${GAMINGAGENT_DIR}:${PYTHONPATH}"
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INFERENCE_GPUS="${INFERENCE_GPUS:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Check for --verl early (delegate immediately)
# ---------------------------------------------------------------------------
for arg in "$@"; do
    if [ "$arg" = "--verl" ]; then
        echo "[run_inference.sh] Delegating to VERL inference..."
        # Strip --verl from args, pass the rest
        VERL_ARGS=()
        for a in "$@"; do
            [ "$a" != "--verl" ] && VERL_ARGS+=("$a")
        done
        CUDA_VISIBLE_DEVICES="$INFERENCE_GPUS" \
            python3 -m scripts.run_inference --verl "${VERL_ARGS[@]}"
        exit $?
    fi
done

# ---------------------------------------------------------------------------
# Parse convenience env vars
# ---------------------------------------------------------------------------
COEVO_DIR="${COEVO_DIR:-}"
ITERATION="${ITERATION:-}"
MODEL="${MODEL:-}"
BANK="${BANK:-}"
GAMES="${GAMES:-}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "============================================"
echo "  Game-AI-Agent Inference"
echo "============================================"
echo "  GPU(s):     $INFERENCE_GPUS"
echo "  Repo root:  $REPO_ROOT"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Build the Python command
# ---------------------------------------------------------------------------
CMD=(python3 -m scripts.run_inference)

# Forward all CLI arguments to the Python script
CMD+=("$@")

# Also forward env-var overrides (lower priority than CLI args)
if [ -n "$MODEL" ]; then
    CMD+=(--model "$MODEL")
fi
if [ -n "$BANK" ]; then
    CMD+=(--bank "$BANK")
fi
if [ -n "$COEVO_DIR" ]; then
    CMD+=(--coevo-dir "$COEVO_DIR")
fi
if [ -n "$ITERATION" ]; then
    CMD+=(--iteration "$ITERATION")
fi
if [ -n "$GAMES" ]; then
    CMD+=(--games $GAMES)
fi
if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[run_inference.sh] Running: CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS ${CMD[*]}"
echo ""

CUDA_VISIBLE_DEVICES="$INFERENCE_GPUS" "${CMD[@]}"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[run_inference.sh] Inference completed successfully."
else
    echo "[run_inference.sh] Inference exited with code $EXIT_CODE."
fi

exit $EXIT_CODE
