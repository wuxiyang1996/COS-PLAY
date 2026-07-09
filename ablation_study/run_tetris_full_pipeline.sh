#!/usr/bin/env bash
# ======================================================================
#  Tetris (macro) full ablation pipeline driver: R2 → R3b → R6.
#
#  - R2:  17 cells (A1..A8, B1, B2, C1..C5, E1, E2) — re-uses an existing
#         tetris_sweep_*/A0 (the smoke-test cell). Set TETRIS_SWEEP_BASE.
#  - R3b: 5 freeze-at-k cells depending on R2 A0 checkpoints.
#  - R6:  4 seed scaling cells, GAME=tetris.
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PIPELINE_LOG="${PIPELINE_LOG:-${PROJECT_ROOT}/ablation_study/output/tetris_pipeline_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${PIPELINE_LOG}")"

log() { echo "[pipeline $(date -u +%H:%M:%S)] $*" | tee -a "${PIPELINE_LOG}"; }

# ── R2: continue the active sweep (skip A0, which is the smoke test) ──
SWEEP_BASE="${TETRIS_SWEEP_BASE:-}"
if [ -z "${SWEEP_BASE}" ]; then
    SWEEP_BASE="$(ls -d "${PROJECT_ROOT}"/ablation_study/output/tetris_sweep_* 2>/dev/null | sort | tail -1)"
fi
if [ ! -d "${SWEEP_BASE}" ]; then
    log "ERROR: no tetris_sweep_* dir; set TETRIS_SWEEP_BASE."
    exit 1
fi
log "R2 sweep base: ${SWEEP_BASE}"

log "▶ R2 launch (17 cells A1..E2)"
OUTPUT_ROOT="${SWEEP_BASE}" \
CELLS="A1 A2 A3 A4 A5 A6 A7 A8 B1 B2 C1 C2 C3 C4 C5 E1 E2" \
    bash "${SCRIPT_DIR}/run_tetris_ablation_all.sh" \
    >> "${PIPELINE_LOG}" 2>&1 || log "R2 sweep returned non-zero, continuing"
log "✓ R2 done"

# ── R3b: freeze-at-k on tetris (depends on A0 checkpoints) ────────────
A0_RUN="${SWEEP_BASE}/A0/run"
if [ -d "${A0_RUN}/checkpoints" ]; then
    log "▶ R3b launch (5 cells F0/F2/F5/F8/F9), A0_RUN=${A0_RUN}"
    A0_RUN="${A0_RUN}" \
        bash "${SCRIPT_DIR}/run_freeze_at_k_tetris_sweep.sh" \
        >> "${PIPELINE_LOG}" 2>&1 || log "R3b sweep returned non-zero"
    log "✓ R3b done"
else
    log "SKIP R3b — no A0 checkpoints at ${A0_RUN}/checkpoints"
fi

# ── R6: seed scaling on tetris ───────────────────────────────────────
log "▶ R6 launch (N10/N20/N40/N60 on tetris)"
GAME=tetris \
    bash "${SCRIPT_DIR}/run_seed_scaling_sweep.sh" \
    >> "${PIPELINE_LOG}" 2>&1 || log "R6 sweep returned non-zero"
log "✓ R6 done"

log "═══ Pipeline complete ═══"
