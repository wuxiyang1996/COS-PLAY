#!/usr/bin/env bash
# ======================================================================
#  Avalon + Diplomacy training — bootstrapped from a previous checkpoint.
#
#  Loads trained LoRA adapters from a prior co-evolution run (default:
#  step_0019 of the Qwen3-8B_20260320_085011 run) and focuses on the
#  two strategic/social games that weren't reached by the first run's
#  curriculum.
#
#  Usage:
#    bash scripts/run_avalon_diplomacy.sh
#
#    # Override checkpoint source:
#    CKPT_STEP=step_0024 bash scripts/run_avalon_diplomacy.sh
#
#    # Also keep training 2048 alongside:
#    EXTRA_GAMES="twenty_forty_eight" bash scripts/run_avalon_diplomacy.sh
#
#    # Custom step count:
#    TOTAL_STEPS=50 bash scripts/run_avalon_diplomacy.sh
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

# ── Source checkpoint (adapters + skill banks) ────────────────────────
SOURCE_RUN="${SOURCE_RUN:-runs/Qwen3-8B_20260320_085011}"
CKPT_STEP="${CKPT_STEP:-step_0019}"
CKPT_DIR="${SOURCE_RUN}/checkpoints/${CKPT_STEP}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_DIR}"
    echo "  Available checkpoints:"
    ls "${SOURCE_RUN}/checkpoints/" 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

TOTAL_STEPS="${TOTAL_STEPS:-40}"
EPISODES="${EPISODES_PER_GAME:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
DEBUG_IO="${DEBUG_IO:-}"

# Games: Avalon + Diplomacy, plus any extras the user wants.
# These two are the strategic/social games that need training.
EXTRA_GAMES="${EXTRA_GAMES:-}"
GAMES="avalon diplomacy ${EXTRA_GAMES}"

# ── Banner ────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Avalon + Diplomacy Co-Evolution Training"
echo "══════════════════════════════════════════════════════════════"
echo "  Source run:    ${SOURCE_RUN}"
echo "  Checkpoint:    ${CKPT_STEP}"
echo "  Model:         ${MODEL}"
echo "  Games:         ${GAMES}"
echo "  Total steps:   ${TOTAL_STEPS}"
echo "  Eps/game:      ${EPISODES}"
echo "  Checkpoint:    every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:     ${VLLM_GPUS}"
echo "  GRPO GPUs:     ${GRPO_GPUS}"
echo "  Spec decode:   ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo "══════════════════════════════════════════════════════════════"

# ── Resolve adapter paths from checkpoint ─────────────────────────────
DECISION_ADAPTERS="${CKPT_DIR}/adapters/decision"
SKILLBANK_ADAPTERS="${CKPT_DIR}/adapters/skillbank"
SEED_BANK="${CKPT_DIR}/banks"

echo ""
echo "[avalon_diplo] Loading adapters from ${CKPT_STEP}:"
echo "  Decision:  ${DECISION_ADAPTERS}"
echo "  SkillBank: ${SKILLBANK_ADAPTERS}"
echo "  Seed bank: ${SEED_BANK}"

for adapter_dir in \
    "${DECISION_ADAPTERS}/skill_selection" \
    "${DECISION_ADAPTERS}/action_taking" \
    "${SKILLBANK_ADAPTERS}/segment" \
    "${SKILLBANK_ADAPTERS}/contract" \
    "${SKILLBANK_ADAPTERS}/curator"; do
    if [ ! -f "${adapter_dir}/adapter_config.json" ]; then
        echo "[ERROR] Missing adapter: ${adapter_dir}"
        exit 1
    fi
done
echo "[avalon_diplo] All 5 adapters verified."

# ── Ensure LoRA adapters are copied into the new run ──────────────────
echo ""
echo "[avalon_diplo] Initialising new run..."

RESOLVED_RUN_DIR=$(python -c "
import sys, os
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from trainer.coevolution.config import CoEvolutionConfig, init_lora_adapters
cfg = CoEvolutionConfig(
    model_name='${MODEL}',
    pretrained_adapter_paths={
        'skill_selection': '${DECISION_ADAPTERS}/skill_selection',
        'action_taking': '${DECISION_ADAPTERS}/action_taking',
        'segment': '${SKILLBANK_ADAPTERS}/segment',
        'contract': '${SKILLBANK_ADAPTERS}/contract',
        'curator': '${SKILLBANK_ADAPTERS}/curator',
    },
)
cfg.resolve_paths()
created = init_lora_adapters(cfg)
if created:
    print(f'Loaded {len(created)} adapter(s) from checkpoint', file=sys.stderr)
else:
    print('All adapters already exist.', file=sys.stderr)
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
echo "[avalon_diplo] New run dir: ${RUN_DIR}"

# ── Build training args ───────────────────────────────────────────────
# shellcheck disable=SC2086
TRAIN_ARGS=(
    --total-steps "${TOTAL_STEPS}"
    --games ${GAMES}
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "avalon-diplo-from-${CKPT_STEP}"
    --run-dir "${RUN_DIR}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    --seed-bank-dir "${SEED_BANK}"
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
)

if [ -n "${DEBUG_IO}" ]; then
    TRAIN_ARGS+=(--debug-io)
fi

# ── Launch ────────────────────────────────────────────────────────────
echo ""
echo "[avalon_diplo] Starting co-evolution (Avalon + Diplomacy)..."
echo "[avalon_diplo] Command: python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"

echo ""
echo "[avalon_diplo] Training complete."
