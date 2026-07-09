#!/usr/bin/env bash
# ======================================================================
#  Train Pokemon Red with LoRA adapters warm-started from SFT cold-start.
#
#  Loads pre-trained LoRA adapters (skill_selection, action_taking,
#  segment, contract, curator) from the SFT cold-start output and
#  begins co-evolution on Pokemon Red.  The Pokemon Red skill bank
#  starts empty and bootstraps from scratch.
#
#  Pokemon Red game profile (Orak env):
#    - Single-player RPG via Orak's PokemonRedEnv (PyBoy emulator)
#    - 16 actions: 8 raw buttons + 8 high-level tools
#      (move_to, warp_with_warp_point, interact_with_object,
#       continue_dialog, select_move_in_battle, switch_pkmn,
#       run_away, use_item_in_battle)
#    - 12-milestone sequential reward (Orak evaluate):
#      leave house → Oak's Lab → starter → rival battle → Viridian →
#      Oak's Parcel → deliver → Town Map/Balls/catch/Pewter/Brock
#    - Delta reward: per-step change in milestone score (0–100 scale)
#    - Parallel episodes: each PyBoy instance gets an isolated ROM copy
#      to avoid .ram/.sav file conflicts
#    - Long-horizon sparse reward — milestones are rare checkpoints
#
#  Usage:
#    conda activate game-ai-agent
#    bash scripts/run_pokemon_red.sh
#
#    # With overrides:
#    TOTAL_STEPS=30 EPISODES=6 bash scripts/run_pokemon_red.sh
#
#    # Resume from a checkpoint:
#    RESUME_FROM_STEP=8 TOTAL_STEPS=30 \
#      RUN_DIR=runs/Qwen3-8B_pokemon_red_20260322_200000 \
#      bash scripts/run_pokemon_red.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── PyBoy fast mode ───────────────────────────────────────────────────
# Default frame_time=0.01 (10ms/tick) adds ~1s per game step.
# 0 causes tick_loop to spin-lock CPU, starving vLLM I/O.
# 0.001 (1ms) is 10x faster than default while yielding CPU.
export PYBOY_FRAME_TIME="${PYBOY_FRAME_TIME:-0.001}"
export POKEMON_STARTUP_DELAY="${POKEMON_STARTUP_DELAY:-1}"

# ── Pokemon Red segmentation tuning ──────────────────────────────────
# Pokemon Red episodes can run 2000 steps with screen-state
# observations — very long trajectories with lots of exploration.
# Higher token budget for RPG context (menus, battles, map state);
# generous timeout for segment ranking over long episodes.
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-300}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-120}"

# ── CUDA memory management ───────────────────────────────────────────
# Pokemon Red screen-state descriptions can be lengthy (OCR + tile map
# + party status).  Enable expandable segments to reduce fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── GRPO memory tuning (Pokemon Red uses long prompts ~1300 tokens) ──
export GRPO_FSDP_BATCH_SIZE="${GRPO_FSDP_BATCH_SIZE:-4}"
export GRPO_REF_MICRO_BATCH="${GRPO_REF_MICRO_BATCH:-2}"
export GRPO_MAX_SEQ_LEN="${GRPO_MAX_SEQ_LEN:-1536}"

# ── Skillbank GRPO accumulation threshold ────────────────────────────
# With only 4 serial episodes/step, segment/curator sample volume is
# low.  Lower threshold so skillbank adapters actually train.
export SKILLBANK_TRAIN_THRESHOLD="${SKILLBANK_TRAIN_THRESHOLD:-8}"

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

TOTAL_STEPS="${TOTAL_STEPS:-25}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"

# ── Pokemon Red GRPO stability overrides ─────────────────────────────
# Pokemon Red is a single-player long-horizon RPG with very sparse
# reward (badges are rare, exploration is gradual).  Long episodes
# with sparse milestones produce high variance.  Conservative LR
# and strong KL prevent catastrophic forgetting between the rare
# reward signals; tight clipping stabilises updates from noisy
# exploration-heavy rollouts.
GRPO_LR="${GRPO_LR:-1e-5}"
GRPO_KL_COEFF="${GRPO_KL_COEFF:-0.08}"
GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-0.1}"
GRPO_MAX_EPOCHS="${GRPO_MAX_EPOCHS:-2}"
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-3.0}"

# ── SFT cold-start adapter paths ─────────────────────────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/pokemon_red_sft_coldstart}"
BASE_SFT_DIR="${PROJECT_ROOT}/runs/sft_coldstart"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"

if [ -z "${RESUME_FROM_STEP}" ]; then
    # Bootstrap from base sft_coldstart: copy adapters that haven't been
    # retrained yet (skill_selection, segment, contract, curator) so the
    # pokemon-specific SFT dir is self-contained.
    if [ -d "${BASE_SFT_DIR}" ] && [ ! -d "${SFT_DIR}/skillbank" ]; then
        echo "[pokemon_red] Bootstrapping ${SFT_DIR} from ${BASE_SFT_DIR}..."
        mkdir -p "${SFT_DIR}/decision" "${SFT_DIR}/skillbank"
        for adapter in skill_selection; do
            if [ -d "${BASE_SFT_DIR}/decision/${adapter}" ] && [ ! -d "${SFT_DIR}/decision/${adapter}" ]; then
                cp -r "${BASE_SFT_DIR}/decision/${adapter}" "${SFT_DIR}/decision/${adapter}"
                echo "  copied decision/${adapter}"
            fi
        done
        for adapter in segment contract curator; do
            if [ -d "${BASE_SFT_DIR}/skillbank/${adapter}" ] && [ ! -d "${SFT_DIR}/skillbank/${adapter}" ]; then
                cp -r "${BASE_SFT_DIR}/skillbank/${adapter}" "${SFT_DIR}/skillbank/${adapter}"
                echo "  copied skillbank/${adapter}"
            fi
        done
        echo ""
    fi

    if [ ! -d "${DECISION_ADAPTERS}" ]; then
        echo "ERROR: Decision adapters not found: ${DECISION_ADAPTERS}"
        echo "Run:  SFT_OUTPUT=${SFT_DIR} SFT_ADAPTERS=\"action_taking\" bash scripts/run_sft_coldstart.sh"
        exit 1
    fi
    if [ ! -d "${SKILLBANK_ADAPTERS}" ]; then
        echo "ERROR: Skill-bank adapters not found: ${SKILLBANK_ADAPTERS}"
        echo "Run:  SFT_OUTPUT=${SFT_DIR} bash scripts/run_sft_coldstart.sh"
        exit 1
    fi
fi

# ── Run directory for Pokemon Red ────────────────────────────────────
if [ -n "${RESUME_FROM_STEP}" ] && [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RESUME_FROM_STEP requires RUN_DIR to be set"
    exit 1
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_pokemon_red_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[pokemon_red] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[pokemon_red] Done."
}
trap cleanup EXIT INT TERM

# ── Print banner ─────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "  Pokemon Red: RESUME from step ${RESUME_FROM_STEP}"
else
    echo "  Pokemon Red: Warm-start from SFT cold-start LoRA adapters"
fi
echo "══════════════════════════════════════════════════════════════"
if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "  Resume step:    ${RESUME_FROM_STEP}"
else
    echo "  SFT dir:        ${SFT_DIR}"
    echo "  Decision LoRA:  ${DECISION_ADAPTERS}"
    echo "  SkillBank LoRA: ${SKILLBANK_ADAPTERS}"
fi
echo "  Run dir:        ${RUN_DIR}"
echo "  Model:          ${MODEL}"
echo "  Total steps:    ${TOTAL_STEPS}"
echo "  Episodes/step:  ${EPISODES}"
echo "  Checkpoint:     every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:      ${VLLM_GPUS}"
echo "  GRPO GPUs:      ${GRPO_GPUS}"
echo "  Spec decode:    ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  Pokemon Red game profile (Orak env):"
echo "    - Single-player RPG via Orak PokemonRedEnv (PyBoy)"
echo "    - 16 actions: 8 buttons + 8 high-level tools"
echo "    - 12-milestone sequential reward (Orak evaluate, delta-based)"
echo "    - Milestones: house→Oak→starter→battle→Viridian→...→Brock"
echo "    - Parallel episodes (per-episode ROM isolation for PyBoy)"
echo "    - Skill bank starts empty; LoRA adapters warm-started from SFT"
echo ""
echo "  Pokemon Red segmentation tuning:"
echo "    - LLM teacher max_tokens: ${SKILLBANK_LLM_TEACHER_MAX_TOKENS} (was 512)"
echo "    - Segment timeout: ${SKILLBANK_SEGMENT_TIMEOUT_S}s (was 180s)"
echo "    - Inference max_tokens: 256 (was 512)"
echo ""
echo "  GRPO stability overrides (pokemon-red-tuned):"
echo "    - LR:           ${GRPO_LR} (default 5e-5)"
echo "    - KL coeff:     ${GRPO_KL_COEFF} (default 0.05)"
echo "    - Clip ratio:   ${GRPO_CLIP_RATIO} (default 0.2)"
echo "    - Max epochs:   ${GRPO_MAX_EPOCHS} (default 4)"
echo "    - Adv clip:     ${GRPO_ADV_CLIP} (default: none)"
echo ""
echo "  Memory & skillbank tuning:"
echo "    - CUDA alloc conf:    ${PYTORCH_CUDA_ALLOC_CONF}"
echo "    - GRPO batch size:    ${GRPO_FSDP_BATCH_SIZE} (default 32)"
echo "    - GRPO ref micro-bs:  ${GRPO_REF_MICRO_BATCH} (default 8)"
echo "    - GRPO max seq len:   ${GRPO_MAX_SEQ_LEN} (default 2048)"
echo "    - Skillbank threshold: ${SKILLBANK_TRAIN_THRESHOLD} (default 32)"
echo "══════════════════════════════════════════════════════════════"
echo ""

if [ -n "${RESUME_FROM_STEP}" ]; then
    echo "[pokemon_red] Resuming from checkpoint step ${RESUME_FROM_STEP}"
    echo "  Checkpoint: ${RUN_DIR}/checkpoints/step_$(printf '%04d' "${RESUME_FROM_STEP}")"
    echo ""
else
    echo "[pokemon_red] SFT cold-start adapters:"
    for adapter_dir in "${DECISION_ADAPTERS}"/* "${SKILLBANK_ADAPTERS}"/*; do
        if [ -d "${adapter_dir}" ]; then
            name="$(basename "${adapter_dir}")"
            # adapter_config.json may be at top level or nested one level deeper
            if [ -f "${adapter_dir}/adapter_config.json" ] || \
               [ -f "${adapter_dir}/${name}/adapter_config.json" ]; then
                echo "  ✓ ${name}"
            else
                echo "  ✗ ${name} (missing adapter_config.json)"
            fi
        fi
    done
    echo ""
fi

# ── Build training command ───────────────────────────────────────────
TRAIN_ARGS=(
    --games pokemon_red
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
    --grpo-lr "${GRPO_LR}"
    --grpo-kl-coeff "${GRPO_KL_COEFF}"
    --grpo-clip-ratio "${GRPO_CLIP_RATIO}"
    --grpo-max-epochs "${GRPO_MAX_EPOCHS}"
    --grpo-adv-clip "${GRPO_ADV_CLIP}"
    --max-tokens 256
)

if [ -n "${RESUME_FROM_STEP}" ]; then
    TRAIN_ARGS+=(--resume --resume-from-step "${RESUME_FROM_STEP}")
else
    TRAIN_ARGS+=(
        --load-decision-adapters "${DECISION_ADAPTERS}"
        --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    )
fi

echo "[pokemon_red] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

# ── Run training ─────────────────────────────────────────────────────
python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Pokemon Red training COMPLETE"

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
    echo "  Pokemon Red training FAILED (exit code ${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
