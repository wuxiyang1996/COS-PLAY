#!/usr/bin/env bash
# ======================================================================
#  R6 — Seed-Trajectory Scaling Curve (single cell)
#
#  For a given N (number of GPT-5.4 expert episodes), produces a
#  step-0 mean-reward measurement of the SFT-only decision agent on
#  candy_crush, using:
#    • decision adapters re-trained on the N-episode subsample
#    • skill-bank adapters re-used from runs/sft_coldstart/skillbank
#      (held fixed — see writeup caveat)
#    • 1 rollout step, 8 episodes, no GRPO
#
#  Usage:
#    bash run_seed_scaling_cell.sh <N> [OUTPUT_BASE]
#
#  Special case N=60 → no subsample, no SFT; just re-uses the
#  existing runs/sft_coldstart/decision adapters and runs the eval.
#
#  Env overrides:
#    GAME           default: candy_crush
#    SFT_GPUS       default: "4 5"        (one GPU per decision adapter)
#    EPISODES       default: 8
#    EVAL_VLLM_GPUS default: "0 1 2 3"
#    EVAL_GRPO_GPUS default: "4 5 6 7"    (allocated even with --no-grpo)
#    PORT           default: 8000
#    SFT_EPOCHS     default: 3
#    SFT_DATA_SRC   default: labeling/output/gpt54_skill_labeled/grpo_coldstart
#    SFT_SB_SRC     default: runs/sft_coldstart/skillbank
#    SFT_DEC_60     default: runs/sft_coldstart/decision  (N=60 free point)
# ======================================================================
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <N> [OUTPUT_BASE]" >&2
    exit 2
fi

N="$1"
OUTPUT_BASE="${2:-${OUTPUT_BASE:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GAME="${GAME:-candy_crush}"
SFT_GPUS="${SFT_GPUS:-4 5}"
EPISODES="${EPISODES:-8}"
EVAL_VLLM_GPUS="${EVAL_VLLM_GPUS:-0 1 2 3}"
EVAL_GRPO_GPUS="${EVAL_GRPO_GPUS:-4 5 6 7}"
PORT="${PORT:-8000}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_DATA_SRC="${SFT_DATA_SRC:-${PROJECT_ROOT}/labeling/output/gpt54_skill_labeled/grpo_coldstart}"
SFT_SB_SRC="${SFT_SB_SRC:-${PROJECT_ROOT}/runs/sft_coldstart/skillbank}"
SFT_DEC_60="${SFT_DEC_60:-${PROJECT_ROOT}/runs/sft_coldstart/decision}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
GPU_UTIL="${GPU_UTIL:-0.55}"

if [ -z "${OUTPUT_BASE}" ]; then
    OUTPUT_BASE="${PROJECT_ROOT}/ablation_study/output/seed_scaling_$(date +%Y%m%d_%H%M%S)"
fi
CELL="N${N}"
CELL_DIR="${OUTPUT_BASE}/${CELL}"
DATA_DIR="${OUTPUT_BASE}/data/${CELL}"
SFT_DIR="${OUTPUT_BASE}/sft/${CELL}"
RUN_DIR="${CELL_DIR}/run"
LOG_FILE="${CELL_DIR}/cell.log"
mkdir -p "${CELL_DIR}" "${DATA_DIR}" "${SFT_DIR}" "${RUN_DIR}"

# Sanity: sources must exist.
if [ ! -d "${SFT_SB_SRC}" ]; then
    echo "[seed-cell] ERROR: SFT_SB_SRC not found: ${SFT_SB_SRC}" | tee -a "${LOG_FILE}"
    exit 1
fi
if [ ! -d "${SFT_DATA_SRC}/${GAME}" ]; then
    echo "[seed-cell] ERROR: SFT_DATA_SRC missing game: ${SFT_DATA_SRC}/${GAME}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "══════════════════════════════════════════════════════════════" | tee "${LOG_FILE}"
echo "  R6 Seed-Scaling Cell: ${CELL}  (game=${GAME})" | tee -a "${LOG_FILE}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"
echo "  N seed episodes:    ${N}" | tee -a "${LOG_FILE}"
echo "  Data subsample:     ${DATA_DIR}" | tee -a "${LOG_FILE}"
echo "  SFT decision dir:   ${SFT_DIR}/decision" | tee -a "${LOG_FILE}"
echo "  SFT skillbank src:  ${SFT_SB_SRC}  (reused, fixed)" | tee -a "${LOG_FILE}"
echo "  Eval episodes/step: ${EPISODES}  (1 step, no-grpo)" | tee -a "${LOG_FILE}"
echo "  vLLM GPUs:          ${EVAL_VLLM_GPUS}" | tee -a "${LOG_FILE}"
echo "  GRPO GPUs (alloc):  ${EVAL_GRPO_GPUS}" | tee -a "${LOG_FILE}"
echo "  Output:             ${CELL_DIR}" | tee -a "${LOG_FILE}"
echo "  Start time:         $(date -u +%FT%TZ)" | tee -a "${LOG_FILE}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${LOG_FILE}"

# ── Step 1: Subsample (or reuse N=60 source) ─────────────────────────
if [ "${N}" -ge 60 ]; then
    echo "[seed-cell] N=${N} ≥ 60: skipping subsample, will reuse full src" | tee -a "${LOG_FILE}"
    DECISION_DATA_FOR_SFT="${SFT_DATA_SRC}"
    SUBSAMPLE_SKIPPED=1
else
    echo "[seed-cell] Subsampling first ${N} episodes → ${DATA_DIR}" | tee -a "${LOG_FILE}"
    python3 "${SCRIPT_DIR}/subsample_seed_data.py" \
        --src "${SFT_DATA_SRC}" \
        --out "${DATA_DIR}" \
        --n "${N}" \
        --games "${GAME}" \
        >> "${LOG_FILE}" 2>&1
    DECISION_DATA_FOR_SFT="${DATA_DIR}"
    SUBSAMPLE_SKIPPED=0
fi

# ── Step 2: SFT decision adapters (or reuse for N=60) ───────────────
SFT_DECISION_DIR=""
if [ "${N}" -ge 60 ] && [ -d "${SFT_DEC_60}" ]; then
    echo "[seed-cell] N=${N}: reusing existing decision adapters at ${SFT_DEC_60}" | tee -a "${LOG_FILE}"
    SFT_DECISION_DIR="${SFT_DEC_60}"
    SFT_SKIPPED=1
else
    echo "[seed-cell] Launching SFT (decision adapters only)" | tee -a "${LOG_FILE}"
    SFT_START=$(date +%s)
    # shellcheck disable=SC2086
    python -m trainer.SFT.train \
        --model_name "${MODEL}" \
        --decision_data_dir "${DECISION_DATA_FOR_SFT}" \
        --output_dir "${SFT_DIR}" \
        --adapters action_taking skill_selection \
        --games "${GAME}" \
        --epochs "${SFT_EPOCHS}" \
        --parallel --gpus ${SFT_GPUS} \
        --bf16 \
        >> "${LOG_FILE}" 2>&1 || {
        echo "[seed-cell] ERROR: SFT training failed (see log)" | tee -a "${LOG_FILE}"
        exit 1
    }
    SFT_WALL=$(( $(date +%s) - SFT_START ))
    echo "[seed-cell] SFT done in ${SFT_WALL}s" | tee -a "${LOG_FILE}"
    SFT_DECISION_DIR="${SFT_DIR}/decision"
    SFT_SKIPPED=0
fi

# PEFT's get_peft_model(adapter_name=NAME) + save_pretrained creates a
# sub-folder named NAME under output_path, leaving freshly-trained
# adapters at ${SFT_DECISION_DIR}/<sub>/<sub>/adapter_model.safetensors.
# Baseline (N=60 cold-start reuse) is already flat. Materialise a flat
# symlinked view under the cell dir for --load-decision-adapters.
FLAT_DECISION_DIR="${CELL_DIR}/decision_flat"
rm -rf "${FLAT_DECISION_DIR}"
mkdir -p "${FLAT_DECISION_DIR}"
for sub in action_taking skill_selection; do
    flat="${SFT_DECISION_DIR}/${sub}/adapter_model.safetensors"
    nested="${SFT_DECISION_DIR}/${sub}/${sub}/adapter_model.safetensors"
    if [ -f "${flat}" ]; then
        ln -sfn "${SFT_DECISION_DIR}/${sub}" "${FLAT_DECISION_DIR}/${sub}"
    elif [ -f "${nested}" ]; then
        ln -sfn "${SFT_DECISION_DIR}/${sub}/${sub}" "${FLAT_DECISION_DIR}/${sub}"
    else
        echo "[seed-cell] ERROR: adapter not found for ${sub} (looked at ${flat} and ${nested})" \
            | tee -a "${LOG_FILE}"
        exit 1
    fi
done
SFT_DECISION_DIR="${FLAT_DECISION_DIR}"

# ── Write metadata BEFORE eval ──────────────────────────────────────
META_PATH="${CELL_DIR}/ablation_meta.json"
python3 - "${META_PATH}" "${CELL}" "${N}" "${DECISION_DATA_FOR_SFT}" "${SFT_DECISION_DIR}" "${SFT_SB_SRC}" \
    "${SUBSAMPLE_SKIPPED}" "${SFT_SKIPPED}" <<'PYEOF'
import json, sys
(out, cell, n, dec_data, dec_dir, sb_dir,
 sub_skipped, sft_skipped) = sys.argv[1:9]
meta = {
    "experiment": "R6_seed_scaling",
    "cell": cell,
    "n_seed_episodes": int(n),
    "decision_data_dir": dec_data,
    "decision_adapters_dir": dec_dir,
    "skillbank_adapters_dir": sb_dir,
    "subsample_skipped": bool(int(sub_skipped)),
    "sft_skipped": bool(int(sft_skipped)),
    "eval_protocol": "coevolution --total-steps 1 --no-grpo",
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
PYEOF

# ── Step 3: 1-step rollout eval (no GRPO) ───────────────────────────
echo "[seed-cell] Launching 1-step rollout eval" | tee -a "${LOG_FILE}"
EVAL_START=$(date +%s)
# shellcheck disable=SC2086
python scripts/run_coevolution.py \
    --games "${GAME}" \
    --total-steps 1 \
    --no-grpo \
    --curriculum none \
    --episodes-per-game "${EPISODES}" \
    --checkpoint-interval 9999 \
    --model "${MODEL}" \
    --run-dir "${RUN_DIR}" \
    --load-decision-adapters "${SFT_DECISION_DIR}" \
    --load-skillbank-adapters "${SFT_SB_SRC}" \
    --vllm-gpus ${EVAL_VLLM_GPUS} \
    --grpo-devices ${EVAL_GRPO_GPUS} \
    --vllm-base-port "${PORT}" \
    --vllm-gpu-util "${GPU_UTIL}" \
    --speculative-model "${SPEC_MODEL}" \
    --num-speculative-tokens "${SPEC_TOKENS}" \
    --wandb-project "game-ai-seed-scaling" \
    --wandb-run-name "seed_scaling_${CELL}" \
    --no-wandb \
    >> "${LOG_FILE}" 2>&1 || {
    echo "[seed-cell] ERROR: rollout eval failed (see log)" | tee -a "${LOG_FILE}"
    exit 1
}
EVAL_WALL=$(( $(date +%s) - EVAL_START ))
echo "[seed-cell] Eval done in ${EVAL_WALL}s" | tee -a "${LOG_FILE}"

# ── Append step-0 result to log for quick view ──────────────────────
if [ -f "${RUN_DIR}/step_log.jsonl" ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "  Cell ${CELL} COMPLETE — step 0 metrics:" | tee -a "${LOG_FILE}"
    python3 - "${RUN_DIR}/step_log.jsonl" "${CELL}" "${N}" <<'PYEOF' | tee -a "${LOG_FILE}"
import json, sys
log = sys.argv[1]
cell, n = sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(log) if l.strip()]
if not rows:
    print(f"  [{cell}] step_log.jsonl exists but empty")
else:
    r = rows[0]
    rwd  = r.get("mean_reward")
    nsk  = r.get("n_skills")
    wall = r.get("wall_time_s")
    print(f"  [{cell}] N={n}  mean_reward={rwd}  n_skills={nsk}  wall={wall}s")
PYEOF
else
    echo "[seed-cell] WARN: ${RUN_DIR}/step_log.jsonl not found" | tee -a "${LOG_FILE}"
fi

echo "[seed-cell] End time: $(date -u +%FT%TZ)" | tee -a "${LOG_FILE}"
