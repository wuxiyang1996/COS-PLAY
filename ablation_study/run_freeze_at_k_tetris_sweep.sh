#!/usr/bin/env bash
# ======================================================================
#  Drive the full freeze-at-iter-k sweep on tetris (macro actions).
#  Mirror of run_freeze_at_k_sweep.sh but for tetris.
#
#  Default cells: F0 F2 F5 F8 F9   (matches A0's per-step checkpoints)
#  Auto-discovers A0_RUN from the latest tetris_sweep_*/A0/run.
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -z "${A0_RUN:-}" ]; then
    A0_RUN="$(ls -d "${PROJECT_ROOT}"/ablation_study/output/tetris_sweep_*/A0/run 2>/dev/null | sort | tail -1)"
fi
if [ ! -d "${A0_RUN}" ]; then
    echo "[freeze-sweep] ERROR: A0 run not found at ${A0_RUN}; set A0_RUN."
    exit 1
fi

SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/freeze_at_k_tetris_${SWEEP_TS}}"
mkdir -p "${OUTPUT_BASE}"
SWEEP_LOG="${OUTPUT_BASE}/sweep.log"

CELLS="${CELLS:-F0 F2 F5 F8 F9}"
NCONT="${NCONT:-5}"

echo "══════════════════════════════════════════════════════════════" | tee "${SWEEP_LOG}"
echo "  Freeze-at-iter-k Sweep on tetris (macro actions)" | tee -a "${SWEEP_LOG}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${SWEEP_LOG}"
echo "  A0 source:        ${A0_RUN}" | tee -a "${SWEEP_LOG}"
echo "  Cells:            ${CELLS}" | tee -a "${SWEEP_LOG}"
echo "  Continuation:     ${NCONT} steps each" | tee -a "${SWEEP_LOG}"
echo "  Output:           ${OUTPUT_BASE}" | tee -a "${SWEEP_LOG}"
echo "  Start time:       $(date -u +%FT%TZ)" | tee -a "${SWEEP_LOG}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${SWEEP_LOG}"

for cell in ${CELLS}; do
    K="${cell#F}"
    echo "" | tee -a "${SWEEP_LOG}"
    echo "[freeze-sweep] $(date -u +%FT%TZ) ▶ Starting ${cell} (freeze@step_${K})" | tee -a "${SWEEP_LOG}"
    echo "" | tee -a "${SWEEP_LOG}"

    A0_RUN="${A0_RUN}" \
    NCONT="${NCONT}" \
    OUTPUT_BASE="${OUTPUT_BASE}" \
        bash "${SCRIPT_DIR}/run_freeze_at_k_tetris.sh" "${K}" "${NCONT}" \
        >> "${SWEEP_LOG}" 2>&1 || {
        echo "[freeze-sweep] ${cell} FAILED, continuing with next cell" | tee -a "${SWEEP_LOG}"
        continue
    }

    echo "[freeze-sweep] $(date -u +%FT%TZ) ✓ ${cell} done" | tee -a "${SWEEP_LOG}"
done

echo "" | tee -a "${SWEEP_LOG}"
echo "[freeze-sweep] $(date -u +%FT%TZ) ▶ Aggregating" | tee -a "${SWEEP_LOG}"
python3 "${SCRIPT_DIR}/aggregate_freeze_at_k.py" \
    --output-root "${OUTPUT_BASE}" \
    --a0-run "${A0_RUN}" \
    --summary "${OUTPUT_BASE}/summary.json" \
    >> "${SWEEP_LOG}" 2>&1 || true

echo "[freeze-sweep] $(date -u +%FT%TZ) ✓ Sweep complete" | tee -a "${SWEEP_LOG}"
echo "  Summary: ${OUTPUT_BASE}/summary.json" | tee -a "${SWEEP_LOG}"
echo "  Table:   ${OUTPUT_BASE}/summary.md" | tee -a "${SWEEP_LOG}"
