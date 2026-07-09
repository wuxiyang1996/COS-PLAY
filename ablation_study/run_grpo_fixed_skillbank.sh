#!/usr/bin/env bash
# ======================================================================
#  Ablation: GRPO training of the DECISION AGENT with a FROZEN skill
#  bank.  The bank loaded at startup is consulted during rollouts (the
#  policy can read existing skills via QUERY_SKILL / CALL_SKILL), but
#  it never grows: Phase B (discovery + bank update) is skipped, and the
#  three skill-bank LoRAs (segment / contract / curator) stay frozen at
#  their SFT cold-start initialization.  GRPO trains the FULL decision
#  agent — both ``action_taking`` and ``skill_selection``.
#
#  Corresponds to the "GRPO w/ fixed skill bank" row in the paper's
#  Table 1 ablation.
#
#  Differences vs. run_grpo_no_skillbank.sh:
#    - bank used for retrieval (rollouts NOT cold-start)
#    - skill_selection IS trained
#    - need to seed the bank from a populated source
#
#  Usage:
#    conda activate game-ai-agent
#    bash ablation_study/run_grpo_fixed_skillbank.sh <game>
#
#  Where <game> is one of:
#    twenty_forty_eight | candy_crush | tetris | super_mario |
#    avalon | diplomacy
# ======================================================================
set -euo pipefail

GAME="${1:-}"
if [ -z "${GAME}" ]; then
    echo "Usage: bash ablation_study/run_grpo_fixed_skillbank.sh <game>"
    echo ""
    echo "Games:"
    echo "  twenty_forty_eight, candy_crush, tetris,"
    echo "  super_mario, avalon, diplomacy"
    exit 1
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Xvfb for Super Mario (NES) ───────────────────────────────────────
if [ "${GAME}" = "super_mario" ] && [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        XVFB_DISPLAY=":99"
        if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
            echo "[fixed-bank ablation] Starting Xvfb on ${XVFB_DISPLAY}..."
            Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    fi
fi
if [ "${GAME}" = "super_mario" ]; then
    export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"
    if [ ! -x "${ORAK_PYTHON}" ]; then
        echo "[ERROR] orak-mario Python not found at: ${ORAK_PYTHON}"
        exit 1
    fi
fi

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Shared parameters ────────────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-fixed-skillbank-ablation}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# ── SFT cold-start adapters + populated bank seeds ───────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"
SEED_BANK_DIR="${SEED_BANK_DIR:-${PROJECT_ROOT}/runs/fixed_skillbank_seeds}"

if [ ! -d "${DECISION_ADAPTERS}" ]; then
    echo "ERROR: Decision adapters not found at ${DECISION_ADAPTERS}"
    echo "Run SFT cold-start first, or symlink the existing SFT step_0000 checkpoint."
    exit 1
fi
if [ ! -d "${SEED_BANK_DIR}/${GAME}" ]; then
    echo "ERROR: Seed bank for game '${GAME}' not found at ${SEED_BANK_DIR}/${GAME}"
    echo "Expected layout: ${SEED_BANK_DIR}/<game>/[<role>/]skill_bank.jsonl"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
#  Per-game defaults (paper Appendix C, Table 3) — same as no-skillbank
#  ablation since the GRPO config is independent of bank usage.
# ──────────────────────────────────────────────────────────────────────
USE_UNIFIED_ROLES=0
GAME_EXTRA=()

case "${GAME}" in
    twenty_forty_eight|2048)
        GAME_ARG="twenty_forty_eight"
        DEFAULT_STEPS=10; DEFAULT_EPS=8; DEFAULT_CKPT=3
        DEFAULT_LR=""; DEFAULT_KL=""; DEFAULT_CLIP=""; DEFAULT_EPOCHS=""; DEFAULT_ADV=""
        ;;
    candy_crush)
        GAME_ARG="candy_crush"
        DEFAULT_STEPS=10; DEFAULT_EPS=8; DEFAULT_CKPT=3
        DEFAULT_LR=""; DEFAULT_KL=""; DEFAULT_CLIP=""; DEFAULT_EPOCHS=""; DEFAULT_ADV=""
        ;;
    tetris)
        GAME_ARG="tetris"
        DEFAULT_STEPS=7; DEFAULT_EPS=8; DEFAULT_CKPT=1
        DEFAULT_LR=2e-5; DEFAULT_KL=0.08; DEFAULT_CLIP=0.10
        DEFAULT_EPOCHS=2; DEFAULT_ADV=3.0
        ;;
    super_mario)
        GAME_ARG="super_mario"
        DEFAULT_STEPS=20; DEFAULT_EPS=8; DEFAULT_CKPT=1
        DEFAULT_LR=3e-5; DEFAULT_KL=0.04; DEFAULT_CLIP=0.15
        DEFAULT_EPOCHS=3; DEFAULT_ADV=5.0
        ;;
    avalon)
        GAME_ARG="avalon"
        USE_UNIFIED_ROLES=1
        DEFAULT_STEPS=20; DEFAULT_EPS=20; DEFAULT_CKPT=1
        DEFAULT_LR=2e-5; DEFAULT_KL=0.06; DEFAULT_CLIP=0.15
        DEFAULT_EPOCHS=2; DEFAULT_ADV=3.0
        GAME_EXTRA+=(
            --opponent-model "${OPPONENT_MODEL:-gpt-5-mini}"
            --opponent-api-base "${OPPONENT_API_BASE:-https://openrouter.ai/api/v1}"
            --temperature 0.3
            --initial-temperature 0.9
            --steady-temperature 0.6
            --warmup-steps "${WARMUP_STEPS:-10}"
            --initial-kl-coeff "${INITIAL_KL_COEFF:-0.02}"
        )
        ;;
    diplomacy)
        GAME_ARG="diplomacy"
        USE_UNIFIED_ROLES=1
        DEFAULT_STEPS=25; DEFAULT_EPS=28; DEFAULT_CKPT=1
        DEFAULT_LR=1e-5; DEFAULT_KL=0.08; DEFAULT_CLIP=0.12
        DEFAULT_EPOCHS=2; DEFAULT_ADV=3.0
        GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
        GAME_EXTRA+=(
            --opponent-model "${OPPONENT_MODEL:-gpt-5-mini}"
            --opponent-api-base "${OPPONENT_API_BASE:-https://openrouter.ai/api/v1}"
            --temperature 0.3
            --initial-temperature 0.9
            --steady-temperature 0.6
            --warmup-steps "${WARMUP_STEPS:-12}"
            --initial-kl-coeff "${INITIAL_KL_COEFF:-0.04}"
        )
        ;;
    *)
        echo "ERROR: Unknown game '${GAME}'."
        echo "Valid games: twenty_forty_eight candy_crush tetris super_mario avalon diplomacy"
        exit 1
        ;;
esac

# ── Allow env-var overrides ──────────────────────────────────────────
TOTAL_STEPS="${TOTAL_STEPS:-${DEFAULT_STEPS}}"
EPISODES="${EPISODES:-${DEFAULT_EPS}}"
CKPT_INTERVAL="${CKPT_INTERVAL:-${DEFAULT_CKPT}}"
GRPO_LR="${GRPO_LR:-${DEFAULT_LR}}"
GRPO_KL_COEFF="${GRPO_KL_COEFF:-${DEFAULT_KL}}"
GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-${DEFAULT_CLIP}}"
GRPO_MAX_EPOCHS="${GRPO_MAX_EPOCHS:-${DEFAULT_EPOCHS}}"
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-${DEFAULT_ADV}}"

# ── Run directory ────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/fixed_skillbank_Qwen3-8B_${GAME_ARG}_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[fixed-bank ablation] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[fixed-bank ablation] Done."
}
trap cleanup EXIT INT TERM

# ── Banner ───────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Ablation: GRPO w/ Fixed Skill Bank  (game=${GAME_ARG})"
echo "  Paper: 'GRPO w/ fixed skill bank' row, Table 1"
echo "  Hyperparameters: paper Appendix C, Table 3"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:           ${MODEL}"
echo "  SFT warm start:  ${SFT_DIR}"
echo "  Seed bank:       ${SEED_BANK_DIR}/${GAME_ARG}"
echo "  Run dir:         ${RUN_DIR}"
echo "  Total steps:     ${TOTAL_STEPS}"
echo "  Episodes/step:   ${EPISODES}"
echo "  Checkpoint:      every ${CKPT_INTERVAL} steps"
echo "  vLLM GPUs:       ${VLLM_GPUS}"
echo "  GRPO GPUs:       ${GRPO_GPUS}"
echo "  Spec decode:     ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo ""
echo "  Ablation gating:"
echo "    - rollouts:    USE bank for retrieval (skill IDs available)"
echo "    - Phase B:     SKIPPED (no skill discovery / bank update)"
echo "    - Phase C:     train action_taking + skill_selection"
echo "                   segment / contract / curator FROZEN"
echo ""
if [ -n "${GRPO_LR}" ]; then
    echo "  GRPO overrides:"
    echo "    - LR:          ${GRPO_LR}"
    echo "    - KL coeff:    ${GRPO_KL_COEFF}"
    echo "    - Clip ratio:  ${GRPO_CLIP_RATIO}"
    echo "    - Max epochs:  ${GRPO_MAX_EPOCHS}"
    echo "    - Adv clip:    ${GRPO_ADV_CLIP}"
else
    echo "  GRPO: using paper defaults (LR 5e-5, KL 0.05, clip 0.20, epochs 4)"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Build args ───────────────────────────────────────────────────────
TRAIN_ARGS=(
    --games "${GAME_ARG}"
    --total-steps "${TOTAL_STEPS}"
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --wandb-project "${WANDB_PROJECT}"
    --run-dir "${RUN_DIR}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --seed-bank-dir "${SEED_BANK_DIR}"
    --debug-io
    # shellcheck disable=SC2086
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
    --freeze-skillbank
)

# Optional: load skillbank LoRA slots so vLLM has all 5 adapters (frozen).
if [ -d "${SKILLBANK_ADAPTERS}" ]; then
    TRAIN_ARGS+=(--load-skillbank-adapters "${SKILLBANK_ADAPTERS}")
fi

if [ -n "${GRPO_LR}" ]; then
    TRAIN_ARGS+=(
        --grpo-lr "${GRPO_LR}"
        --grpo-kl-coeff "${GRPO_KL_COEFF}"
        --grpo-clip-ratio "${GRPO_CLIP_RATIO}"
        --grpo-max-epochs "${GRPO_MAX_EPOCHS}"
        --grpo-adv-clip "${GRPO_ADV_CLIP}"
    )
fi

if [ "${USE_UNIFIED_ROLES}" = "1" ]; then
    TRAIN_ARGS+=(--unified-roles)
fi

if [ ${#GAME_EXTRA[@]} -gt 0 ]; then
    TRAIN_ARGS+=("${GAME_EXTRA[@]}")
fi

if [ $# -gt 0 ]; then
    TRAIN_ARGS+=("$@")
fi

echo "[fixed-bank ablation] Command:"
echo "  python scripts/run_coevolution.py ${TRAIN_ARGS[*]}"
echo ""

python scripts/run_coevolution.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Fixed-skillbank ablation COMPLETE  (${GAME_ARG})"
    if [ -f "${RUN_DIR}/step_log.jsonl" ]; then
        echo ""
        echo "  Last few steps:"
        python -c "
import json
with open('${RUN_DIR}/step_log.jsonl') as f:
    rows = [json.loads(l) for l in f if l.strip()]
for r in rows[-5:]:
    step = r['step']
    mr = r['mean_reward']
    wt = r['wall_time_s'] / 60
    print(f'  Step {step:2d}: mean_reward={mr:8.2f}  time={wt:.1f}m')
"
    fi
else
    echo "  Fixed-skillbank ablation FAILED  (${GAME_ARG}, exit=${EXIT_CODE})"
    echo "  Check logs: ${RUN_DIR}/coevolution.log"
fi
echo ""
echo "  Run dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
