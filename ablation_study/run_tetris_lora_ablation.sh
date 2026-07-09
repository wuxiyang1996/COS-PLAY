#!/usr/bin/env bash
# ======================================================================
#  Tetris (macro-action) LoRA / Reward Ablation
#
#  Mirror of run_candy_crush_lora_ablation.sh on tetris with macro
#  actions. The trainer (trainer/coevolution/episode_runner.py:1308)
#  auto-wraps tetris with TetrisMacroActionWrapper, so passing
#  --games tetris is sufficient — macro-action is the training-time
#  default for tetris and we keep that here.
#
#  Cells (16): A0..A8, B1, B2, C1..C5, E1, E2 — identical to the
#  candy_crush sweep. See R2_candy_crush_lora_ablation_plan.md for
#  rationale of each cell.
#
#  Usage:
#    bash ablation_study/run_tetris_lora_ablation.sh --cell A0
#    STEPS=10 EPISODES=8 bash ablation_study/run_tetris_lora_ablation.sh --cell C5
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CELL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cell)    CELL="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ -z "${CELL}" ]; then
    echo "Usage: $0 --cell {A0|A1|...|E2}"
    exit 1
fi

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# Tetris uses the same skillbank LLM teacher tunables as candy_crush.
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-400}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-120}"

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
STEPS="${STEPS:-10}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"

if [ ! -d "${DECISION_ADAPTERS}" ] || [ ! -d "${SKILLBANK_ADAPTERS}" ]; then
    echo "ERROR: SFT cold-start adapters not found under ${SFT_DIR}"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_BASE="${SWEEP_BASE:-${PROJECT_ROOT}/ablation_study/output/tetris_sweep_${TIMESTAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${SWEEP_BASE}/${CELL}}"
mkdir -p "${OUTPUT_BASE}"

unset COSPLAY_DISABLE_ADAPTERS COSPLAY_MERGE_ADAPTERS COSPLAY_REWARD_OVERRIDES

EXTRA_ARGS=()

case "${CELL}" in
    A0) DESC="control — full system (all 5 LoRAs trained)"; ;;
    A1) DESC="freeze skill_selection (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="skill_selection" ;;
    A2) DESC="freeze action_taking (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="action_taking" ;;
    A3) DESC="freeze segment (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="segment" ;;
    A4) DESC="freeze contract (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="contract" ;;
    A5) DESC="freeze curator (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="curator" ;;
    A6) DESC="merge decision LoRAs (skill_selection+action_taking → 1)"
        export COSPLAY_MERGE_ADAPTERS="decision" ;;
    A7) DESC="merge skill-bank LoRAs (segment+contract+curator → 1)"
        export COSPLAY_MERGE_ADAPTERS="skillbank" ;;
    A8) DESC="merge all 5 LoRAs into a single shared adapter"
        export COSPLAY_MERGE_ADAPTERS="all" ;;
    B1) DESC="capacity ctrl — 5 LoRAs @ rank=4"
        EXTRA_ARGS+=(--lora-r 4 --lora-alpha 8) ;;
    B2) DESC="capacity ctrl — 1 shared LoRA @ rank=80"
        export COSPLAY_MERGE_ADAPTERS="all"
        EXTRA_ARGS+=(--lora-r 80 --lora-alpha 160) ;;
    C1) DESC="reward ablation — w_follow = 0"
        export COSPLAY_REWARD_OVERRIDES='{"w_follow": 0.0}' ;;
    C2) DESC="reward ablation — zero action/retrieval costs"
        export COSPLAY_REWARD_OVERRIDES='{"query_mem_cost": 0.0, "query_skill_cost": 0.0, "call_skill_cost": 0.0, "skill_switch_cost": 0.0}' ;;
    C3) DESC="reward ablation — zero follow predicate/completion bonus"
        export COSPLAY_REWARD_OVERRIDES='{"follow_predicate_bonus": 0.0, "follow_completion_bonus": 0.0, "follow_no_progress_penalty": 0.0}' ;;
    C4) DESC="curator gradient zero-ed (≈curator_weight=0)"
        export COSPLAY_DISABLE_ADAPTERS="curator" ;;
    C5) DESC="reward ablation — env reward only (C1 + C2 + C3)"
        export COSPLAY_REWARD_OVERRIDES='{"w_follow": 0.0, "query_mem_cost": 0.0, "query_skill_cost": 0.0, "call_skill_cost": 0.0, "skill_switch_cost": 0.0, "follow_predicate_bonus": 0.0, "follow_completion_bonus": 0.0, "follow_no_progress_penalty": 0.0}' ;;
    E1) DESC="GRPO only — decision LoRAs trained, NO skill bank at all"
        EXTRA_ARGS+=(--no-skillbank) ;;
    E2) DESC="train skill-bank LoRAs only (decision LoRAs frozen)"
        export COSPLAY_DISABLE_ADAPTERS="skill_selection,action_taking" ;;
    Z1) DESC="GRPO only — decision LoRAs trained, skill-bank FROZEN at pre-evolved bank"
        EXTRA_ARGS+=(--freeze-skillbank)
        if [ -n "${SEED_BANK_DIR:-}" ]; then
            EXTRA_ARGS+=(--seed-bank-dir "${SEED_BANK_DIR}")
        fi
        ;;
    *) echo "[ERROR] Unknown cell: ${CELL}"; exit 1 ;;
esac

echo "══════════════════════════════════════════════════════════════"
echo "  Tetris (macro) Ablation: cell=${CELL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Desc:           ${DESC}"
echo "  Steps:          ${STEPS}    Episodes/step: ${EPISODES}"
echo "  vLLM GPUs:      ${VLLM_GPUS}"
echo "  GRPO GPUs:      ${GRPO_GPUS}"
echo "  Output:         ${OUTPUT_BASE}"
echo "  COSPLAY_DISABLE_ADAPTERS = ${COSPLAY_DISABLE_ADAPTERS:-<unset>}"
echo "  COSPLAY_MERGE_ADAPTERS   = ${COSPLAY_MERGE_ADAPTERS:-<unset>}"
echo "  COSPLAY_REWARD_OVERRIDES = ${COSPLAY_REWARD_OVERRIDES:-<unset>}"
echo "  Extra CLI:      ${EXTRA_ARGS[*]:-<none>}"
echo "══════════════════════════════════════════════════════════════"

META_PATH="${OUTPUT_BASE}/ablation_meta.json"
python3 - "${META_PATH}" "${CELL}" "${DESC}" "${STEPS}" "${EPISODES}" <<'PYEOF'
import json, os, sys
out, cell, desc, steps, episodes = sys.argv[1:6]
meta = {
    "cell": cell, "description": desc,
    "steps": int(steps), "episodes_per_step": int(episodes),
    "game": "tetris", "macro_actions": True,
    "env": {k: os.environ.get(k, "") for k in (
        "COSPLAY_DISABLE_ADAPTERS", "COSPLAY_MERGE_ADAPTERS", "COSPLAY_REWARD_OVERRIDES")},
}
with open(out, "w") as f: json.dump(meta, f, indent=2)
PYEOF

cleanup() { echo "[ablation:${CELL}] cleanup"; jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true; }
trap cleanup EXIT INT TERM

RUN_DIR="${OUTPUT_BASE}/run"
LOG_FILE="${OUTPUT_BASE}/train.log"

CMD=(python scripts/run_coevolution.py
     --games          tetris
     --total-steps    "${STEPS}"
     --curriculum     none
     --episodes-per-game "${EPISODES}"
     --checkpoint-interval "${CKPT_INTERVAL}"
     --model          "${MODEL}"
     --wandb-project  "${WANDB_PROJECT}"
     --run-dir        "${RUN_DIR}"
     --load-decision-adapters "${DECISION_ADAPTERS}"
     --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
     --vllm-gpus      ${VLLM_GPUS}
     --grpo-devices   ${GRPO_GPUS}
     --vllm-base-port "${PORT}"
     --vllm-gpu-util  "${GPU_UTIL}"
     --speculative-model "${SPEC_MODEL}"
     --num-speculative-tokens "${SPEC_TOKENS}"
     "${EXTRA_ARGS[@]}"
)
echo "[ablation:${CELL}] ${CMD[*]}"
echo "[ablation:${CELL}] training log → ${LOG_FILE}"

EXIT_CODE=0
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}" || EXIT_CODE=${PIPESTATUS[0]}

echo
echo "══════════════════════════════════════════════════════════════"
if [ ${EXIT_CODE} -eq 0 ] && [ -f "${RUN_DIR}/step_log.jsonl" ]; then
    echo "  Cell ${CELL} COMPLETE — last 3 steps:"
    python3 - "${RUN_DIR}/step_log.jsonl" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for r in rows[-3:]:
    print(f"  step {r['step']:>2}  mean_reward={r.get('mean_reward', 0):.2f}  "
          f"n_skills={r.get('n_skills','?')}  wall={r.get('wall_time_s', 0):.0f}s")
PYEOF
else
    echo "  Cell ${CELL} FAILED (exit=${EXIT_CODE})"
fi
echo "  Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
