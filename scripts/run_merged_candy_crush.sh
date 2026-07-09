#!/usr/bin/env bash
# ======================================================================
#  Merged-LoRA Candy Crush training.
#
#  Instead of 5 separate LoRA adapters (2 decision + 3 skillbank), this
#  script creates 2 unified adapters by:
#
#    1. Combining cold-start SFT training data across adapter roles:
#         decision = skill_selection + action_taking data pooled
#         skillbank = segment + contract + curator data pooled
#
#    2. Averaging LoRA weights from existing per-role SFT adapters
#       (if --source-dir is available), then optionally fine-tuning
#       the averaged weights on the combined data.
#
#    3. Deploying the unified adapters to all 5 named adapter slots
#       so the co-evolution pipeline (vLLM serving, GRPO training,
#       checkpointing) works unchanged.
#
#  The benefit is a single set of weights per agent group, preventing
#  the fragmentation and instability that comes from training 5
#  adapters on sparse per-role data.
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_merged_candy_crush.sh
#
#    # With existing SFT adapters to average first:
#    SFT_SOURCE=runs/sft_coldstart bash scripts/run_merged_candy_crush.sh
#
#    # Skip merge (already merged), just resume training:
#    SKIP_MERGE=1 bash scripts/run_merged_candy_crush.sh
#
#    # Override training params:
#    TOTAL_STEPS=20 EPISODES=12 bash scripts/run_merged_candy_crush.sh
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
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"

TOTAL_STEPS="${TOTAL_STEPS:-10}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# Merge configuration
MERGE_MODE="${MERGE_MODE:-average-and-retrain}"   # average | retrain | average-and-retrain
SFT_SOURCE="${SFT_SOURCE:-${PROJECT_ROOT}/runs/backup/sft_coldstart}"  # existing SFT adapters
CKPT_SOURCE="${CKPT_SOURCE:-}"                     # path to checkpoint adapters/ dir (alternative source)
SKIP_MERGE="${SKIP_MERGE:-0}"                      # set to 1 to skip merge step
DECISION_EPOCHS="${DECISION_EPOCHS:-5}"
SKILLBANK_EPOCHS="${SKILLBANK_EPOCHS:-8}"

# Run directory — fresh run with timestamp
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/merged_Qwen3-8B_${TIMESTAMP}}"
MERGED_DIR="${RUN_DIR}/merged_lora"
ADAPTER_DIR="${RUN_DIR}/lora_adapters"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[merged] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[merged] Done."
}
trap cleanup EXIT INT TERM

# ── Resolve source directory for averaging ───────────────────────────
SOURCE_DIR=""
if [ -n "${SFT_SOURCE}" ] && [ -d "${SFT_SOURCE}" ]; then
    SOURCE_DIR="${SFT_SOURCE}"
elif [ -n "${CKPT_SOURCE}" ] && [ -d "${CKPT_SOURCE}" ]; then
    SOURCE_DIR="${CKPT_SOURCE}"
else
    # Auto-detect: try backup, then default SFT output
    for candidate in \
        "${PROJECT_ROOT}/runs/backup/sft_coldstart" \
        "${PROJECT_ROOT}/runs/sft_coldstart"; do
        if [ -d "${candidate}/decision" ]; then
            SOURCE_DIR="${candidate}"
            break
        fi
    done
fi

# If no source for averaging, fall back to retrain-only mode
if [ -z "${SOURCE_DIR}" ] && [ "${MERGE_MODE}" != "retrain" ]; then
    echo "[merged] No source adapters found for averaging — falling back to retrain mode"
    MERGE_MODE="retrain"
fi

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  MERGED-LoRA Candy Crush Training"
echo "══════════════════════════════════════════════════════════════"
echo "  Run dir:         ${RUN_DIR}"
echo "  Model:           ${MODEL}"
echo "  Total steps:     ${TOTAL_STEPS}"
echo "  Episodes/step:   ${EPISODES}"
echo "  Checkpoint:      every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:       ${VLLM_GPUS}"
echo "  GRPO GPUs:       ${GRPO_GPUS}"
echo "  Spec decode:     ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  Merge strategy:  ${MERGE_MODE}"
if [ -n "${SOURCE_DIR}" ]; then
echo "  Source adapters:  ${SOURCE_DIR}"
fi
echo "  Merged output:   ${MERGED_DIR}"
echo "  Deploy target:   ${ADAPTER_DIR}"
echo ""
echo "  What this does:"
echo "    - Pools SFT data: skill_selection+action_taking → 1 decision adapter"
echo "    - Pools SFT data: segment+contract+curator → 1 skillbank adapter"
if [ "${MERGE_MODE}" = "average" ] || [ "${MERGE_MODE}" = "average-and-retrain" ]; then
echo "    - Averages existing LoRA weights from source adapters"
fi
if [ "${MERGE_MODE}" = "retrain" ] || [ "${MERGE_MODE}" = "average-and-retrain" ]; then
echo "    - Trains unified adapters on combined cold-start data"
fi
echo "    - Deploys to all 5 adapter slots for co-evolution compatibility"
echo "══════════════════════════════════════════════════════════════"
echo ""

mkdir -p "${RUN_DIR}" "${MERGED_DIR}" "${ADAPTER_DIR}"

# ── Phase 1: Merge LoRA Adapters ─────────────────────────────────────

if [ "${SKIP_MERGE}" = "1" ]; then
    echo "[merged] Skipping merge step (SKIP_MERGE=1)"
    echo ""
else
    echo "[merged] Phase 1: Merging LoRA adapters (mode: ${MERGE_MODE})"
    echo ""

    MERGE_ARGS=("${MERGE_MODE}")

    if [ "${MERGE_MODE}" = "average" ]; then
        MERGE_ARGS+=(
            --source-dir "${SOURCE_DIR}"
            --output-dir "${MERGED_DIR}"
            --deploy-dir "${ADAPTER_DIR}"
        )
    elif [ "${MERGE_MODE}" = "retrain" ]; then
        MERGE_ARGS+=(
            --output-dir "${MERGED_DIR}"
            --deploy-dir "${ADAPTER_DIR}"
            --model "${MODEL}"
            --decision-epochs "${DECISION_EPOCHS}"
            --skillbank-epochs "${SKILLBANK_EPOCHS}"
        )
    elif [ "${MERGE_MODE}" = "average-and-retrain" ]; then
        MERGE_ARGS+=(
            --source-dir "${SOURCE_DIR}"
            --output-dir "${MERGED_DIR}"
            --deploy-dir "${ADAPTER_DIR}"
            --model "${MODEL}"
            --decision-epochs "${DECISION_EPOCHS}"
            --skillbank-epochs "${SKILLBANK_EPOCHS}"
        )
    fi

    echo "[merged] Running: python scripts/merge_lora_adapters.py ${MERGE_ARGS[*]}"
    echo ""
    python scripts/merge_lora_adapters.py "${MERGE_ARGS[@]}"
    MERGE_EXIT=$?

    if [ ${MERGE_EXIT} -ne 0 ]; then
        echo ""
        echo "ERROR: LoRA merge failed (exit ${MERGE_EXIT})"
        exit ${MERGE_EXIT}
    fi

    echo ""
    echo "[merged] Phase 1 complete — unified adapters deployed to ${ADAPTER_DIR}"

    # Verify deployment
    echo "[merged] Verifying adapter deployment:"
    for adapter in skill_selection action_taking; do
        if [ -f "${ADAPTER_DIR}/decision/${adapter}/adapter_config.json" ]; then
            echo "  ✓ decision/${adapter}"
        else
            echo "  ✗ decision/${adapter} — MISSING"
        fi
    done
    for adapter in segment contract curator; do
        if [ -f "${ADAPTER_DIR}/skillbank/${adapter}/adapter_config.json" ]; then
            echo "  ✓ skillbank/${adapter}"
        else
            echo "  ✗ skillbank/${adapter} — MISSING"
        fi
    done
    echo ""
fi

# ── Phase 2: Co-Evolution Training ──────────────────────────────────

echo "[merged] Phase 2: Co-evolution training"
echo ""

TRAIN_ARGS=(
    --games candy_crush
    --total-steps "${TOTAL_STEPS}"
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
    --debug-io
    # shellcheck disable=SC2086
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
)

echo "[merged] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Merged-LoRA training COMPLETE"

    echo ""
    echo "  Step log:"
    if [ -f "${RUN_DIR}/step_log.jsonl" ]; then
        python -c "
import json
with open('${RUN_DIR}/step_log.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
for r in rows[-5:]:
    step = r['step']
    mr = r['mean_reward']
    ns = r.get('n_skills', '?')
    wt = r['wall_time_s'] / 60
    print(f'  Step {step:2d}: mean_reward={mr:6.1f}  skills={ns}  time={wt:.1f}m')
"
    fi
else
    echo "  Merged-LoRA training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "  Merged adapters: ${MERGED_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
