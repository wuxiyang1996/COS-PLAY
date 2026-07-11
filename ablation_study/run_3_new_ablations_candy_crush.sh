#!/usr/bin/env bash
# ======================================================================
#  Run the 3 new ablations on candy_crush with settings matching
#  the existing candy_crush_sweep (A0-E2, 2026-05-24).
#
#  Same hyper-parameters:
#    model=Qwen/Qwen3-8B, steps=10, episodes=8, ckpt_interval=3,
#    vLLM GPUs 0-3, GRPO GPUs 4-7, spec_decode=Qwen3-0.6B/5tok,
#    SFT cold-start from A0/step_0000.
#
#  Cells (sequential — all share the same GPUs):
#    NC   --no-contract
#    RDC  --raw-delta-contract
#    HS   --heuristic-segmentation
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../GamingAgent/gamingagent/envs/custom_03_candy_crush:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Settings (match candy_crush_sweep_20260524) ──────────────────────
MODEL="Qwen/Qwen3-8B"
STEPS=10
EPISODES=8
CKPT_INTERVAL=3
VLLM_GPUS="0 1 2 3"
GRPO_GPUS="4 5 6 7"
VLLM_PORT=8000
GPU_UTIL=0.82
SPEC_MODEL="Qwen/Qwen3-0.6B"
SPEC_TOKENS=5

# SFT cold-start adapters (from A0 step_0000 = pre-training init)
SFT_BASE="/workspace/Game-AI-Agent/ablation_study/output/candy_crush_sweep_20260524_025210/A0/run/checkpoints/step_0000/adapters"
DECISION_ADAPTERS="${SFT_BASE}/decision"
SKILLBANK_ADAPTERS="${SFT_BASE}/skillbank"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ablation_study/output/candy_crush_3new_${TIMESTAMP}}"
mkdir -p "${OUTPUT_ROOT}"

echo "══════════════════════════════════════════════════════════════"
echo "  3 New Ablations — Candy Crush (paper Table 2)"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:        ${MODEL}"
echo "  Steps:        ${STEPS}    Episodes: ${EPISODES}"
echo "  vLLM GPUs:    ${VLLM_GPUS}"
echo "  GRPO GPUs:    ${GRPO_GPUS}"
echo "  Spec decode:  ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
echo "  SFT decision: ${DECISION_ADAPTERS}"
echo "  SFT skillbank:${SKILLBANK_ADAPTERS}"
echo "  Output:       ${OUTPUT_ROOT}"
echo "══════════════════════════════════════════════════════════════"
echo

COMMON_ARGS=(
    --games candy_crush
    --total-steps "${STEPS}"
    --curriculum none
    --episodes-per-game "${EPISODES}"
    --checkpoint-interval "${CKPT_INTERVAL}"
    --model "${MODEL}"
    --load-decision-adapters "${DECISION_ADAPTERS}"
    --load-skillbank-adapters "${SKILLBANK_ADAPTERS}"
    --vllm-gpus ${VLLM_GPUS}
    --grpo-devices ${GRPO_GPUS}
    --vllm-base-port "${VLLM_PORT}"
    --vllm-gpu-util "${GPU_UTIL}"
    --speculative-model "${SPEC_MODEL}"
    --num-speculative-tokens "${SPEC_TOKENS}"
    --no-wandb
)

# ── Cell 1: w/o Effect Contract ──────────────────────────────────────
echo
echo "▶▶▶ Cell NC: w/o Effect Contract (--no-contract)"
echo

NC_DIR="${OUTPUT_ROOT}/NC"
mkdir -p "${NC_DIR}"

export COSPLAY_REWARD_OVERRIDES='{"follow_predicate_bonus": 0.0, "follow_completion_bonus": 0.0}'
export COSPLAY_DISABLE_ADAPTERS="contract"

NC_EXIT=0
python scripts/run_coevolution.py \
    --no-contract \
    --run-dir "${NC_DIR}/run" \
    --wandb-run-name "ablation-no-contract-candy-crush" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${NC_DIR}/train.log" || NC_EXIT=$?

unset COSPLAY_REWARD_OVERRIDES COSPLAY_DISABLE_ADAPTERS

if [ ${NC_EXIT} -eq 0 ]; then
    echo "[sweep] ✓ NC (no-contract) COMPLETE"
else
    echo "[sweep] ✗ NC (no-contract) FAILED (exit=${NC_EXIT})"
fi

echo "[sweep] Waiting 15s for GPU memory release..."
sleep 15

# ── Cell 2: Raw Delta Contract ───────────────────────────────────────
echo
echo "▶▶▶ Cell RDC: Raw Delta Contract (--raw-delta-contract)"
echo

RDC_DIR="${OUTPUT_ROOT}/RDC"
mkdir -p "${RDC_DIR}"

RDC_EXIT=0
python scripts/run_coevolution.py \
    --raw-delta-contract \
    --run-dir "${RDC_DIR}/run" \
    --wandb-run-name "ablation-raw-delta-contract-candy-crush" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${RDC_DIR}/train.log" || RDC_EXIT=$?

if [ ${RDC_EXIT} -eq 0 ]; then
    echo "[sweep] ✓ RDC (raw-delta-contract) COMPLETE"
else
    echo "[sweep] ✗ RDC (raw-delta-contract) FAILED (exit=${RDC_EXIT})"
fi

echo "[sweep] Waiting 15s for GPU memory release..."
sleep 15

# ── Cell 3: Heuristic-Only Segmentation ──────────────────────────────
echo
echo "▶▶▶ Cell HS: Heuristic-Only Segmentation (--heuristic-segmentation)"
echo

HS_DIR="${OUTPUT_ROOT}/HS"
mkdir -p "${HS_DIR}"

HS_EXIT=0
python scripts/run_coevolution.py \
    --heuristic-segmentation \
    --run-dir "${HS_DIR}/run" \
    --wandb-run-name "ablation-heuristic-segmentation-candy-crush" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "${HS_DIR}/train.log" || HS_EXIT=$?

if [ ${HS_EXIT} -eq 0 ]; then
    echo "[sweep] ✓ HS (heuristic-segmentation) COMPLETE"
else
    echo "[sweep] ✗ HS (heuristic-segmentation) FAILED (exit=${HS_EXIT})"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════════════════"
echo "  3-Cell Sweep Complete"
echo "══════════════════════════════════════════════════════════════"
echo "  NC  (no-contract):            exit=${NC_EXIT}"
echo "  RDC (raw-delta-contract):     exit=${RDC_EXIT}"
echo "  HS  (heuristic-segmentation): exit=${HS_EXIT}"
echo "  Output: ${OUTPUT_ROOT}"
echo "══════════════════════════════════════════════════════════════"

# Aggregate if all passed
if [ ${NC_EXIT} -eq 0 ] && [ ${RDC_EXIT} -eq 0 ] && [ ${HS_EXIT} -eq 0 ]; then
    echo "[sweep] All cells passed. Aggregating..."
    python3 -c "
import json, os, sys
from pathlib import Path

output_root = '${OUTPUT_ROOT}'
cells = []
for name in ['NC', 'RDC', 'HS']:
    step_log = Path(output_root) / name / 'run' / 'step_log.jsonl'
    if step_log.exists():
        rows = [json.loads(l) for l in step_log.open() if l.strip()]
        last3 = rows[-3:] if len(rows) >= 3 else rows
        mean_last3 = sum(r.get('mean_reward', 0) for r in last3) / max(len(last3), 1)
        peak = max((r.get('mean_reward', 0) for r in rows), default=0)
        cells.append({
            'cell': name,
            'n_steps': len(rows),
            'mean_reward_last3': round(mean_last3, 2),
            'peak_reward': round(peak, 2),
            'final_skills': rows[-1].get('n_skills', '?') if rows else '?',
            'total_wall_s': sum(r.get('wall_time_s', 0) for r in rows),
        })
    else:
        cells.append({'cell': name, 'error': 'step_log.jsonl not found'})

summary = {'experiment': '3_new_ablations_candy_crush', 'cells': cells}
out_path = Path(output_root) / 'summary.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[sweep] Summary written to {out_path}')
for c in cells:
    if 'error' not in c:
        print(f\"  {c['cell']}: mean_last3={c['mean_reward_last3']}  peak={c['peak_reward']}  skills={c['final_skills']}\")
    else:
        print(f\"  {c['cell']}: {c['error']}\")
" || echo "[sweep] aggregation failed"
fi

exit $(( NC_EXIT + RDC_EXIT + HS_EXIT ))
