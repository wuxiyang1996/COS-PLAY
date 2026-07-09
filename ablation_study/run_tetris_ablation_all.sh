#!/usr/bin/env bash
# ======================================================================
#  Sequentially run R2 LoRA / Reward ablation cells on tetris (macro
#  actions), then aggregate per-cell step_log.jsonl into a Markdown
#  summary. Mirror of run_candy_crush_ablation_all.sh.
#
#  Default subset (15 cells: A0 is run separately as smoke test):
#    A1..A8, B1, B2, C1..C5, E1, E2  → 17 total (override via CELLS env)
#
#  Paper-aligned defaults: STEPS=10  EPISODES=8 (Appendix C Table 3),
#  MODEL=Qwen/Qwen3-8B, vLLM GPUs 0-3, GRPO GPUs 4-7.
#
#  Usage:
#    bash ablation_study/run_tetris_ablation_all.sh
#    OUTPUT_ROOT=/path/to/sweep CELLS="A1 A2 C5" \
#        bash ablation_study/run_tetris_ablation_all.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default: all 16 cells; A0 may be run separately and skipped via CELLS.
CELLS="${CELLS:-A0 A1 A2 A3 A4 A5 A6 A7 A8 B1 B2 C1 C2 C3 C4 C5 E1 E2}"
STEPS="${STEPS:-10}"
EPISODES="${EPISODES:-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ablation_study/output/tetris_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"

echo "[sweep] CELLS=${CELLS}"
echo "[sweep] STEPS=${STEPS}  EPISODES=${EPISODES}"
echo "[sweep] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo

for CELL in ${CELLS}; do
    CELL_OUT="${OUTPUT_ROOT}/${CELL}"
    if [ -f "${CELL_OUT}/run/step_log.jsonl" ]; then
        # Idempotent skip — useful when restarting a partial sweep.
        n_done=$(wc -l < "${CELL_OUT}/run/step_log.jsonl" 2>/dev/null || echo 0)
        if [ "${n_done}" -ge "${STEPS}" ]; then
            echo "▶▶▶ Cell ${CELL}: already complete (${n_done} steps), skipping"
            continue
        fi
    fi
    echo
    echo "▶▶▶ Cell ${CELL} (output: ${CELL_OUT})"
    OUTPUT_BASE="${CELL_OUT}" STEPS="${STEPS}" EPISODES="${EPISODES}" \
        bash "${SCRIPT_DIR}/run_tetris_lora_ablation.sh" --cell "${CELL}" \
        || echo "[sweep] WARN: cell ${CELL} failed, continuing"
done

echo
echo "[sweep] All cells finished. Aggregating..."
if [ -x "${SCRIPT_DIR}/aggregate_candy_crush_ablation.py" ] \
   || [ -f "${SCRIPT_DIR}/aggregate_candy_crush_ablation.py" ]; then
    python3 "${SCRIPT_DIR}/aggregate_candy_crush_ablation.py" \
        --output-root "${OUTPUT_ROOT}" \
        --summary    "${OUTPUT_ROOT}/summary.json" \
        || echo "[sweep] aggregate script failed, summary skipped"
fi

echo "[sweep] Done. Output: ${OUTPUT_ROOT}"
