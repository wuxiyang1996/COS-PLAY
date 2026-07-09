#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# SFT Cold-Start Training for Decision + Skill-Bank LoRA Adapters
# ─────────────────────────────────────────────────────────────────────
#
# Trains all 5 LoRA adapters from teacher-labelled cold-start data
# so the co-evolution GRPO loop starts from a warm checkpoint.
#
# Supports PARALLEL training: set SFT_PARALLEL=1 to train all
# adapters simultaneously, one per GPU.  With 5 GPUs this is ~5x
# faster than sequential.
#
# Adapters trained:
#   Decision:   skill_selection, action_taking
#   Skill Bank: segment, contract, curator
#
# Usage:
#   bash scripts/run_sft_coldstart.sh                    # sequential
#   SFT_PARALLEL=1 bash scripts/run_sft_coldstart.sh     # parallel
#   SFT_PARALLEL=1 SFT_GPUS="0 1 2 3 4" bash scripts/run_sft_coldstart.sh
#   SFT_ADAPTERS="segment contract" bash scripts/run_sft_coldstart.sh
#
# Then feed into co-evolution:
#   python scripts/run_coevolution.py \
#       --load-decision-adapters  runs/sft_coldstart/decision \
#       --load-skillbank-adapters runs/sft_coldstart/skillbank
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/../GamingAgent:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"

# ── Configurable via environment variables ──
MODEL="${SFT_MODEL:-Qwen/Qwen3-8B}"
OUTPUT="${SFT_OUTPUT:-runs/sft_coldstart}"
EPOCHS="${SFT_EPOCHS:-3}"
LR="${SFT_LR:-2e-4}"
BATCH="${SFT_BATCH:-4}"
GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
MAX_SEQ="${SFT_MAX_SEQ:-2048}"
ADAPTERS="${SFT_ADAPTERS:-}"    # empty = all 5
GAMES="${SFT_GAMES:-}"          # empty = all games in COLDSTART_GAMES
PARALLEL="${SFT_PARALLEL:-0}"   # 1 = parallel training
GPUS="${SFT_GPUS:-}"            # e.g. "0 1 2 3 4" (auto-detect if empty)

echo "============================================================"
echo "  SFT COLD-START TRAINING"
echo "============================================================"
echo "  Model:       ${MODEL}"
echo "  Output:      ${OUTPUT}"
echo "  Epochs:      ${EPOCHS}"
echo "  LR:          ${LR}"
echo "  Batch:       ${BATCH}"
echo "  Grad accum:  ${GRAD_ACCUM}"
echo "  Max seq len: ${MAX_SEQ}"
echo "  Adapters:    ${ADAPTERS:-all 5}"
echo "  Games:       ${GAMES:-all}"
if [ "$PARALLEL" = "1" ]; then
echo "  Mode:        PARALLEL (one adapter per GPU)"
echo "  GPUs:        ${GPUS:-auto-detect}"
else
echo "  Mode:        sequential"
fi
echo "============================================================"

ADAPTER_ARGS=""
if [ -n "$ADAPTERS" ]; then
    ADAPTER_ARGS="--adapters ${ADAPTERS}"
fi

GAMES_ARGS=""
if [ -n "$GAMES" ]; then
    GAMES_ARGS="--games ${GAMES}"
fi

PARALLEL_ARGS=""
if [ "$PARALLEL" = "1" ]; then
    PARALLEL_ARGS="--parallel"
    if [ -n "$GPUS" ]; then
        PARALLEL_ARGS="${PARALLEL_ARGS} --gpus ${GPUS}"
    fi
fi

python -m trainer.SFT.train \
    --model_name "${MODEL}" \
    --output_dir "${OUTPUT}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --batch_size "${BATCH}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_seq_length "${MAX_SEQ}" \
    ${ADAPTER_ARGS} \
    ${GAMES_ARGS} \
    ${PARALLEL_ARGS}

echo ""
echo "============================================================"
echo "  SFT TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Output adapters:"
echo "  ${OUTPUT}/decision/skill_selection/"
echo "  ${OUTPUT}/decision/action_taking/"
echo "  ${OUTPUT}/skillbank/segment/"
echo "  ${OUTPUT}/skillbank/contract/"
echo "  ${OUTPUT}/skillbank/curator/"
echo ""
echo "To use with co-evolution:"
echo "  python scripts/run_coevolution.py \\"
echo "      --load-decision-adapters  ${OUTPUT}/decision \\"
echo "      --load-skillbank-adapters ${OUTPUT}/skillbank"
echo ""
