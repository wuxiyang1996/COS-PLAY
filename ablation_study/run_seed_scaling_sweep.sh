#!/usr/bin/env bash
# ======================================================================
#  R6 — Seed-Trajectory Scaling Curve (sweep driver)
#
#  Trains SFT-only candy_crush decision adapters from N expert episodes
#  for N ∈ {N1, N2, ...}, then runs a 1-step rollout eval per N to
#  measure SFT-only mean reward.  The N=60 cell short-circuits SFT
#  by reusing the existing runs/sft_coldstart/decision adapters.
#
#  Default sweep:    N=10 → 20 → 40 → 60   (60 is FREE)
#
#  Env overrides:
#    CELLS        e.g. "N10 N20 N40 N60"  (the sweep set)
#    OUTPUT_BASE  default: ablation_study/output/seed_scaling_<TS>
#    GAME         default: candy_crush
#    EPISODES     default: 8
#    SFT_EPOCHS   default: 3
#    SFT_GPUS     default: "4 5"
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CELLS="${CELLS:-N10 N20 N40 N60}"
GAME="${GAME:-candy_crush}"
EPISODES="${EPISODES:-8}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_GPUS="${SFT_GPUS:-4 5}"

SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/seed_scaling_${SWEEP_TS}}"
mkdir -p "${OUTPUT_BASE}"
SWEEP_LOG="${OUTPUT_BASE}/sweep.log"

echo "══════════════════════════════════════════════════════════════" | tee "${SWEEP_LOG}"
echo "  R6 Seed-Scaling Sweep on ${GAME}" | tee -a "${SWEEP_LOG}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${SWEEP_LOG}"
echo "  Cells:           ${CELLS}" | tee -a "${SWEEP_LOG}"
echo "  Episodes/cell:   ${EPISODES}  (1 step, no-grpo)" | tee -a "${SWEEP_LOG}"
echo "  SFT epochs:      ${SFT_EPOCHS}" | tee -a "${SWEEP_LOG}"
echo "  SFT GPUs:        ${SFT_GPUS}" | tee -a "${SWEEP_LOG}"
echo "  Output:          ${OUTPUT_BASE}" | tee -a "${SWEEP_LOG}"
echo "  Start time:      $(date -u +%FT%TZ)" | tee -a "${SWEEP_LOG}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${SWEEP_LOG}"

for cell in ${CELLS}; do
    N="${cell#N}"
    echo "" | tee -a "${SWEEP_LOG}"
    echo "[seed-sweep] $(date -u +%FT%TZ) ▶ Starting ${cell} (N=${N})" | tee -a "${SWEEP_LOG}"
    echo "" | tee -a "${SWEEP_LOG}"

    GAME="${GAME}" \
    EPISODES="${EPISODES}" \
    SFT_EPOCHS="${SFT_EPOCHS}" \
    SFT_GPUS="${SFT_GPUS}" \
        bash "${SCRIPT_DIR}/run_seed_scaling_cell.sh" "${N}" "${OUTPUT_BASE}" \
        >> "${SWEEP_LOG}" 2>&1 || {
        echo "[seed-sweep] ${cell} FAILED, continuing with next cell" | tee -a "${SWEEP_LOG}"
        continue
    }
    echo "[seed-sweep] $(date -u +%FT%TZ) ✓ ${cell} done" | tee -a "${SWEEP_LOG}"
done

# ── Aggregate ────────────────────────────────────────────────────────
echo "" | tee -a "${SWEEP_LOG}"
echo "[seed-sweep] $(date -u +%FT%TZ) ▶ Aggregating" | tee -a "${SWEEP_LOG}"
python3 "${SCRIPT_DIR}/aggregate_seed_scaling.py" \
    --output-root "${OUTPUT_BASE}" \
    --summary "${OUTPUT_BASE}/summary.json" \
    >> "${SWEEP_LOG}" 2>&1 || true

echo "[seed-sweep] $(date -u +%FT%TZ) ✓ Sweep complete" | tee -a "${SWEEP_LOG}"
echo "  Summary: ${OUTPUT_BASE}/summary.json" | tee -a "${SWEEP_LOG}"
echo "  Table:   ${OUTPUT_BASE}/summary.md"   | tee -a "${SWEEP_LOG}"
