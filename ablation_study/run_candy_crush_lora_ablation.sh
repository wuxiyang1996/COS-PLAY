#!/usr/bin/env bash
# ======================================================================
#  Candy Crush LoRA / Reward Ablation (paper-aligned)
#
#  Companion to /workspace/reviews/R2_candy_crush_lora_ablation_plan.md
#  — the fine-grained LoRA + reward ablation requested by Reviewers
#  vJ13 ([Reject 1], [Reject 5]) and Fs6X (#3).
#
#  This script trains *one* ablation cell per invocation, on
#  candy_crush only, with the **paper Appendix C, Table 3 defaults**:
#     model        = Qwen/Qwen3-8B
#     total_steps  = 10
#     episodes/step = 8
#     ckpt_interval = 3
#     LoRA r/alpha = 16/32  (overridable for B1/B2)
#     GRPO LR/KL/Clip/Epochs/Adv = defaults
#  Same SFT cold-start as runs/sft_coldstart/{decision,skillbank}/.
#
#  ── Group A: LoRA structure (9 cells) ────────────────────────────
#    A0  full                  All 5 LoRAs trained (control)
#    A1  freeze-skill-sel      Freeze skill_selection at SFT init
#    A2  freeze-action-taking  Freeze action_taking at SFT init
#    A3  freeze-segment        Freeze segment at SFT init
#    A4  freeze-contract       Freeze contract at SFT init
#    A5  freeze-curator        Freeze curator at SFT init
#    A6  merge-decision        skill_selection + action_taking → 1 LoRA
#    A7  merge-skillbank       segment + contract + curator → 1 LoRA
#    A8  merge-all             All 5 roles → 1 shared LoRA
#
#  ── Group B: capacity-matched controls (2 cells) ────────────────
#    B1  rank4-five            5 LoRAs @ rank=4   (matched total params)
#    B2  rank80-shared         1 LoRA @ rank=80   (matched params, no role split)
#
#  ── Group C: reward decomposition (5 cells) ─────────────────────
#    C1  no-w-follow           w_follow = 0
#    C2  no-action-costs       query / skill / switch costs = 0
#    C3  no-follow-bonus       follow_predicate_bonus / completion_bonus = 0
#    C4  no-curator-train      curator LoRA frozen (≈ curator_weight=0)
#    C5  env-reward-only       Combine C1 + C2 + C3 (only r_env survives)
#
#  ── Group E: sequential dependence (2 cells) ────────────────────
#    E1  only-decision         --no-skillbank        (uses existing flag)
#    E2  only-skillbank        Train skill-bank LoRAs only
#
#  Usage:
#    bash ablation_study/run_candy_crush_lora_ablation.sh --cell A0
#    bash ablation_study/run_candy_crush_lora_ablation.sh --cell A3 --steps 10
#    STEPS=10 EPISODES=8 bash ablation_study/run_candy_crush_lora_ablation.sh --cell C1
#
#    # 11-cell core sweep:
#    for c in A0 A1 A2 A3 A4 A5 A6 A7 E1 C1 C4; do
#        bash ablation_study/run_candy_crush_lora_ablation.sh --cell $c
#    done
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Parse arguments ──────────────────────────────────────────────────
CELL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cell)    CELL="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,55p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ -z "${CELL}" ]; then
    echo "Usage: $0 --cell {A0|A1|...|E2}"
    echo "Run with -h for the full cell catalogue."
    exit 1
fi

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── Candy Crush segmentation tuning (matches scripts/run_candy_crush.sh) ─
export SKILLBANK_LLM_TEACHER_MAX_TOKENS="${SKILLBANK_LLM_TEACHER_MAX_TOKENS:-400}"
export SKILLBANK_SEGMENT_TIMEOUT_S="${SKILLBANK_SEGMENT_TIMEOUT_S:-120}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH (matches scripts/run_candy_crush.sh) ──────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters (paper Appendix C defaults) ──────────────
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

# ── SFT cold-start adapter paths (must exist) ────────────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
DECISION_ADAPTERS="${SFT_DIR}/decision"
SKILLBANK_ADAPTERS="${SFT_DIR}/skillbank"

if [ ! -d "${DECISION_ADAPTERS}" ] || [ ! -d "${SKILLBANK_ADAPTERS}" ]; then
    echo "ERROR: SFT cold-start adapters not found under ${SFT_DIR}"
    echo "       Expected ${DECISION_ADAPTERS} and ${SKILLBANK_ADAPTERS}."
    echo "       Run scripts/run_sft_coldstart.sh first."
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/candy_crush_${CELL}_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

# ── Reset all ablation env-vars (per cell) ──────────────────────────
unset COSPLAY_DISABLE_ADAPTERS \
      COSPLAY_MERGE_ADAPTERS \
      COSPLAY_REWARD_OVERRIDES

# Extra CLI args appended to run_coevolution.py per cell.
EXTRA_ARGS=()

case "${CELL}" in
    # ── Group A: LoRA structure ──────────────────────────────────
    A0)
        DESC="control — full system (all 5 LoRAs trained)"
        ;;
    A1)
        DESC="freeze skill_selection (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="skill_selection"
        ;;
    A2)
        DESC="freeze action_taking (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="action_taking"
        ;;
    A3)
        DESC="freeze segment (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="segment"
        ;;
    A4)
        DESC="freeze contract (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="contract"
        ;;
    A5)
        DESC="freeze curator (4 LoRAs trained)"
        export COSPLAY_DISABLE_ADAPTERS="curator"
        ;;
    A6)
        DESC="merge decision LoRAs (skill_selection+action_taking → 1)"
        export COSPLAY_MERGE_ADAPTERS="decision"
        ;;
    A7)
        DESC="merge skill-bank LoRAs (segment+contract+curator → 1)"
        export COSPLAY_MERGE_ADAPTERS="skillbank"
        ;;
    A8)
        DESC="merge all 5 LoRAs into a single shared adapter"
        export COSPLAY_MERGE_ADAPTERS="all"
        ;;

    # ── Group B: capacity-matched controls ───────────────────────
    B1)
        DESC="capacity ctrl — 5 LoRAs @ rank=4 (matched total params)"
        EXTRA_ARGS+=(--lora-r 4 --lora-alpha 8)
        ;;
    B2)
        DESC="capacity ctrl — 1 shared LoRA @ rank=80 (matched params, no split)"
        export COSPLAY_MERGE_ADAPTERS="all"
        EXTRA_ARGS+=(--lora-r 80 --lora-alpha 160)
        ;;

    # ── Group C: reward decomposition ────────────────────────────
    C1)
        DESC="reward ablation — w_follow = 0 (no skill-following shaping)"
        export COSPLAY_REWARD_OVERRIDES='{"w_follow": 0.0}'
        ;;
    C2)
        DESC="reward ablation — zero out all action / retrieval costs"
        export COSPLAY_REWARD_OVERRIDES='{"query_mem_cost": 0.0, "query_skill_cost": 0.0, "call_skill_cost": 0.0, "skill_switch_cost": 0.0}'
        ;;
    C3)
        DESC="reward ablation — zero out follow predicate / completion bonus"
        export COSPLAY_REWARD_OVERRIDES='{"follow_predicate_bonus": 0.0, "follow_completion_bonus": 0.0, "follow_no_progress_penalty": 0.0}'
        ;;
    C4)
        DESC="curator gradient zero-ed (equivalent of curator_weight=0)"
        export COSPLAY_DISABLE_ADAPTERS="curator"
        ;;
    C5)
        DESC="reward ablation — env reward only (C1 + C2 + C3)"
        export COSPLAY_REWARD_OVERRIDES='{"w_follow": 0.0, "query_mem_cost": 0.0, "query_skill_cost": 0.0, "call_skill_cost": 0.0, "skill_switch_cost": 0.0, "follow_predicate_bonus": 0.0, "follow_completion_bonus": 0.0, "follow_no_progress_penalty": 0.0}'
        ;;

    # ── Group E: sequential dependence ───────────────────────────
    E1)
        DESC="train decision LoRAs only (skill-bank frozen + Phase B skipped)"
        # Uses the existing production flag — exactly equivalent to
        # run_grpo_no_skillbank.sh, but driven from this matrix so the
        # numbers come out in the same sweep summary.
        EXTRA_ARGS+=(--no-skillbank)
        ;;
    E2)
        DESC="train skill-bank LoRAs only (decision LoRAs frozen)"
        export COSPLAY_DISABLE_ADAPTERS="skill_selection,action_taking"
        ;;

    *)
        echo "[ERROR] Unknown cell: ${CELL}"
        echo "  Valid cells: A0-A8, B1-B2, C1-C5, E1-E2"
        exit 1
        ;;
esac

# ── Banner ──────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Candy Crush Ablation (paper-aligned): cell=${CELL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Description:        ${DESC}"
echo "  Steps:              ${STEPS}    Episodes/step: ${EPISODES}"
echo "  Model:              ${MODEL}    LoRA rank/alpha: ${EXTRA_ARGS[*]:-16/32}"
echo "  vLLM GPUs:          ${VLLM_GPUS}"
echo "  GRPO GPUs:          ${GRPO_GPUS}"
echo "  Spec decode:        ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo "  SFT decision:       ${DECISION_ADAPTERS}"
echo "  SFT skillbank:      ${SKILLBANK_ADAPTERS}"
echo "  Output:             ${OUTPUT_BASE}"
echo
echo "  Ablation env vars:"
echo "    COSPLAY_DISABLE_ADAPTERS  = ${COSPLAY_DISABLE_ADAPTERS:-<unset>}"
echo "    COSPLAY_MERGE_ADAPTERS    = ${COSPLAY_MERGE_ADAPTERS:-<unset>}"
echo "    COSPLAY_REWARD_OVERRIDES  = ${COSPLAY_REWARD_OVERRIDES:-<unset>}"
echo "  Extra CLI args:             ${EXTRA_ARGS[*]:-<none>}"
echo "══════════════════════════════════════════════════════════════"
echo

# ── Persist metadata BEFORE training ────────────────────────────────
META_PATH="${OUTPUT_BASE}/ablation_meta.json"
python3 - "${META_PATH}" "${CELL}" "${DESC}" "${STEPS}" "${EPISODES}" <<'PYEOF'
import json, os, sys
out, cell, desc, steps, episodes = sys.argv[1:6]
meta = {
    "cell": cell,
    "description": desc,
    "steps": int(steps),
    "episodes_per_step": int(episodes),
    "env": {
        k: os.environ.get(k, "")
        for k in (
            "COSPLAY_DISABLE_ADAPTERS",
            "COSPLAY_MERGE_ADAPTERS",
            "COSPLAY_REWARD_OVERRIDES",
        )
    },
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
PYEOF

# ── Cleanup on exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[ablation:${CELL}] Shutting down..."
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    echo "[ablation:${CELL}] Done."
}
trap cleanup EXIT INT TERM

# ── Launch the co-evolution loop ───────────────────────────────────
RUN_DIR="${OUTPUT_BASE}/run"
LOG_FILE="${OUTPUT_BASE}/train.log"

# shellcheck disable=SC2086
CMD=(python scripts/run_coevolution.py
     --games          candy_crush
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
echo

EXIT_CODE=0
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}" || EXIT_CODE=${PIPESTATUS[0]}

# ── Tail summary ────────────────────────────────────────────────────
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
