#!/usr/bin/env bash
# ======================================================================
#  Sequentially run the high-priority candy_crush ablation cells, then
#  aggregate the per-cell step_log.jsonl into a single Markdown table.
#
#  Default subset (11 runs):  A0 A1 A2 A3 A4 A5 A6 A7 E1 C1 C4
#   - A0    : control (full 5 LoRAs)
#   - A1-A5 : leave-one-LoRA-out
#   - A6/A7 : merge decision / merge skill-bank
#   - E1    : skill-bank completely off (--no-skillbank, matches
#             paper's "GRPO w/o skill" baseline)
#   - C1    : w_follow = 0
#   - C4    : curator LoRA gradient zero-ed (≈ curator_weight=0)
#
#  Paper-aligned defaults:
#     STEPS=10  EPISODES=8  (matches Appendix C Table 3)
#     MODEL=Qwen/Qwen3-8B
#     vLLM GPUs 0-3, GRPO GPUs 4-7
#
#  Usage:
#    bash ablation_study/run_candy_crush_ablation_all.sh
#    CELLS="A0 A6 A7 B1 B2" bash ablation_study/run_candy_crush_ablation_all.sh
#    STEPS=10 EPISODES=8 bash ablation_study/run_candy_crush_ablation_all.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CELLS="${CELLS:-A0 A1 A2 A3 A4 A5 A6 A7 E1 C1 C4}"
STEPS="${STEPS:-10}"
EPISODES="${EPISODES:-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ablation_study/output/candy_crush_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"

echo "[sweep] CELLS=${CELLS}"
echo "[sweep] STEPS=${STEPS}  EPISODES=${EPISODES}"
echo "[sweep] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo

for CELL in ${CELLS}; do
    CELL_OUT="${OUTPUT_ROOT}/${CELL}"
    echo
    echo "▶▶▶ Cell ${CELL} (output: ${CELL_OUT})"
    OUTPUT_BASE="${CELL_OUT}" STEPS="${STEPS}" EPISODES="${EPISODES}" \
        bash "${SCRIPT_DIR}/run_candy_crush_lora_ablation.sh" --cell "${CELL}" \
        || echo "[sweep] WARN: cell ${CELL} failed, continuing"
done

echo
echo "[sweep] All cells finished. Aggregating..."
python3 "${SCRIPT_DIR}/aggregate_candy_crush_ablation.py" \
    --output-root "${OUTPUT_ROOT}" \
    --summary    "${OUTPUT_ROOT}/summary.json"

echo "[sweep] Done. Output: ${OUTPUT_ROOT}"
