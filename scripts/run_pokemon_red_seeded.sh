#!/usr/bin/env bash
# ======================================================================
#  Pokemon Red co-evolution training — seeded with GPT-5.4 skill bank.
#
#  Loads trained LoRA adapters from a prior co-evolution run and seeds
#  the skill bank with GPT-5.4-extracted skills for Pokemon Red.  This
#  gives the agent a head-start with 9 pre-extracted skills (Finish the
#  Opponent, Heal at Pokecenter, Get Starter Pokemon, Finish Oak Lab
#  Intro, Explore Opening Area, Start the Journey, Trigger Position
#  Event, Start with Squirtle, Preserve Squirtle) instead of starting
#  from an empty bank.
#
#  Pokemon Red is emulator-backed (PyBoy); episodes serialise to avoid
#  race conditions under concurrent init.
#
#  Usage:
#    bash scripts/run_pokemon_red_seeded.sh
#
#    # Override checkpoint source:
#    CKPT_STEP=step_0029 bash scripts/run_pokemon_red_seeded.sh
#
#    # Custom seed bank (default: labeling/output/gpt54_skillbank):
#    SEED_BANK_DIR=path/to/bank bash scripts/run_pokemon_red_seeded.sh
#
#    # Also keep training 2048 alongside:
#    EXTRA_GAMES="twenty_forty_eight" bash scripts/run_pokemon_red_seeded.sh
#
#    # Custom step count:
#    TOTAL_STEPS=50 bash scripts/run_pokemon_red_seeded.sh
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

# ── Source checkpoint (adapters only) ─────────────────────────────────
SOURCE_RUN="${SOURCE_RUN:-runs/Qwen3-8B_20260321_041333}"
CKPT_STEP="${CKPT_STEP:-step_0034}"
CKPT_DIR="${SOURCE_RUN}/checkpoints/${CKPT_STEP}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_DIR}"
    echo "  Available checkpoints:"
    ls "${SOURCE_RUN}/checkpoints/" 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Seed skill bank (GPT-5.4 extracted skills) ───────────────────────
SEED_BANK_DIR="${SEED_BANK_DIR:-labeling/output/gpt54_skillbank}"

if [ ! -d "${SEED_BANK_DIR}" ]; then
    echo "[ERROR] Seed bank directory not found: ${SEED_BANK_DIR}"
    exit 1
fi

SEED_SKILLS_FILE="${SEED_BANK_DIR}/pokemon_red/skill_bank.jsonl"
if [ ! -f "${SEED_SKILLS_FILE}" ]; then
    echo "[ERROR] No pokemon_red seed skills found at: ${SEED_SKILLS_FILE}"
    exit 1
fi
SEED_SKILL_COUNT=$(wc -l < "${SEED_SKILLS_FILE}")
echo "[pokemon_red] Seed skill bank: ${SEED_SKILLS_FILE} (${SEED_SKILL_COUNT} skills)"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

TOTAL_STEPS="${TOTAL_STEPS:-40}"
EPISODES="${EPISODES_PER_GAME:-4}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
DEBUG_IO="${DEBUG_IO:-true}"

EXTRA_GAMES="${EXTRA_GAMES:-}"
GAMES="pokemon_red ${EXTRA_GAMES}"

# ── Banner ────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Pokemon Red Co-Evolution Training (GPT-5.4 Seed Skills)"
echo "══════════════════════════════════════════════════════════════"
echo "  Source run:    ${SOURCE_RUN}"
echo "  Checkpoint:    ${CKPT_STEP}"
echo "  Seed bank:     ${SEED_BANK_DIR}"
echo "  Seed skills:   ${SEED_SKILL_COUNT} (pokemon_red)"
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

echo ""
echo "[pokemon_red] Loading adapters from ${CKPT_STEP}:"
echo "  Decision:  ${DECISION_ADAPTERS}"
echo "  SkillBank: ${SKILLBANK_ADAPTERS}"

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
echo "[pokemon_red] All 5 adapters verified."

# ── Initialise new run and copy adapters ──────────────────────────────
echo ""
echo "[pokemon_red] Initialising new run..."

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
echo "[pokemon_red] New run dir: ${RUN_DIR}"

# ── Pre-copy seed skills into the run's skillbank directory ───────────
# The --seed-bank-dir flag handles seeding at startup, but we also copy
# the files directly so that skills are guaranteed present even if the
# lazy seeding has already been skipped (e.g. on resume).
echo ""
echo "[pokemon_red] Pre-seeding skill banks into run directory..."
for game_dir in "${SEED_BANK_DIR}"/*/; do
    game_name="$(basename "${game_dir}")"
    src_file="${game_dir}skill_bank.jsonl"
    if [ -f "${src_file}" ]; then
        dest_dir="${RUN_DIR}/skillbank/${game_name}"
        dest_file="${dest_dir}/skill_bank.jsonl"
        if [ ! -f "${dest_file}" ] || [ ! -s "${dest_file}" ]; then
            mkdir -p "${dest_dir}"
            cp "${src_file}" "${dest_file}"
            n=$(wc -l < "${dest_file}")
            echo "  Seeded ${game_name}: ${n} skills"
        else
            echo "  Skip ${game_name}: bank already exists"
        fi
    fi
done

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
    --wandb-run-name "pokemon-red-seeded-from-${CKPT_STEP}"
    --run-dir "${RUN_DIR}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    --seed-bank-dir "${SEED_BANK_DIR}"
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
echo "[pokemon_red] Starting co-evolution (Pokemon Red, seeded)..."
echo "[pokemon_red] Command: python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"

echo ""
echo "[pokemon_red] Training complete."
