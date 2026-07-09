#!/usr/bin/env bash
# ======================================================================
#  R9 — Cross-game skill transfer sweep (leave-one-out donor bank)
#
#  For each target game in TARGETS, runs run_r9_cross_game_cell.sh with
#  the corresponding donor_<TARGET>.jsonl bank built by build_donor_banks.py.
#  Sequential to avoid GPU contention (each cell uses 1 GPU for vLLM).
#
#  Env overrides:
#    OUTPUT_BASE    default: ablation_study/output/r9_cross_game/runs_<ts>
#    DONOR_BANK_DIR default: ablation_study/output/r9_cross_game/donor_banks
#    TARGETS        default: "candy_crush tetris twenty_forty_eight super_mario avalon diplomacy"
#                            (candy_crush first as smoke test — smallest single-player)
#    EVAL_GPUS      default: 0
#    EPISODES_SP    default: 16
#    EPISODES_MP    default: 10
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGETS="${TARGETS:-candy_crush tetris twenty_forty_eight super_mario avalon diplomacy}"
DONOR_BANK_DIR="${DONOR_BANK_DIR:-${PROJECT_ROOT}/ablation_study/output/r9_cross_game/donor_banks}"

SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/r9_cross_game/runs_${SWEEP_TS}}"
mkdir -p "${OUTPUT_BASE}"
SWEEP_LOG="${OUTPUT_BASE}/sweep.log"

echo "══════════════════════════════════════════════════════════════" | tee "${SWEEP_LOG}"
echo "  R9 Cross-Game Transfer Sweep" | tee -a "${SWEEP_LOG}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${SWEEP_LOG}"
echo "  Targets:        ${TARGETS}" | tee -a "${SWEEP_LOG}"
echo "  Donor dir:      ${DONOR_BANK_DIR}" | tee -a "${SWEEP_LOG}"
echo "  Output:         ${OUTPUT_BASE}" | tee -a "${SWEEP_LOG}"
echo "  Episodes:       single-player=${EPISODES_SP:-16}, multi-player=${EPISODES_MP:-10}" | tee -a "${SWEEP_LOG}"
echo "  Start time:     $(date -u +%FT%TZ)" | tee -a "${SWEEP_LOG}"
echo "══════════════════════════════════════════════════════════════" | tee -a "${SWEEP_LOG}"

OVERALL_RC=0
for target in ${TARGETS}; do
    donor="${DONOR_BANK_DIR}/donor_${target}.jsonl"
    echo "" | tee -a "${SWEEP_LOG}"
    echo "[r9-sweep] $(date -u +%FT%TZ) ▶ Starting target=${target}" | tee -a "${SWEEP_LOG}"
    echo "" | tee -a "${SWEEP_LOG}"

    if [ ! -f "${donor}" ]; then
        echo "[r9-sweep] WARN: donor bank not found for ${target}: ${donor} — skipping" | tee -a "${SWEEP_LOG}"
        continue
    fi

    bash "${SCRIPT_DIR}/run_r9_cross_game_cell.sh" "${target}" "${donor}" "${OUTPUT_BASE}" \
        >> "${SWEEP_LOG}" 2>&1 || {
        rc=$?
        echo "[r9-sweep] ${target} FAILED (exit ${rc}), continuing" | tee -a "${SWEEP_LOG}"
        OVERALL_RC=$((OVERALL_RC + 1))
        continue
    }
    echo "[r9-sweep] $(date -u +%FT%TZ) ✓ ${target} done" | tee -a "${SWEEP_LOG}"
done

# ── Aggregate ──────────────────────────────────────────────────────────
echo "" | tee -a "${SWEEP_LOG}"
echo "[r9-sweep] $(date -u +%FT%TZ) ▶ Aggregating" | tee -a "${SWEEP_LOG}"
python3 "${SCRIPT_DIR}/aggregate_r9_cross_game.py" \
    --output-root "${OUTPUT_BASE}" \
    --donor-manifest "${DONOR_BANK_DIR}/donor_manifest.json" \
    --summary "${OUTPUT_BASE}/summary.json" \
    >> "${SWEEP_LOG}" 2>&1 || true

echo "[r9-sweep] $(date -u +%FT%TZ) ✓ Sweep complete (failures=${OVERALL_RC})" | tee -a "${SWEEP_LOG}"
echo "  Summary: ${OUTPUT_BASE}/summary.json" | tee -a "${SWEEP_LOG}"
echo "  Table:   ${OUTPUT_BASE}/summary.md"   | tee -a "${SWEEP_LOG}"

exit ${OVERALL_RC}
