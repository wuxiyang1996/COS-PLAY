#!/bin/bash
# =============================================================================
# Co-Evolution Framework: Main Orchestrator
# =============================================================================
# Runs the full co-evolution training loop between:
#   Decision Agent  — Qwen3-8B, trained with GRPO (via VERL)
#   Skill Bank Agent — Qwen3-8B + LoRA, trained with Hard-EM
#
# Co-evolution schedule (per iteration):
#   1. Collect rollouts: Decision Agent plays games with the current skill bank.
#   2. Skill Bank EM: Ingest trajectories → train LoRA adapters (boundary,
#      segment, contract, retrieval) → run Hard-EM → gate accept/reject.
#   3. Decision Agent GRPO: Train with updated bank on GPUs 0-3.
#   4. Repeat for NUM_ITERATIONS rounds.
#
# GPU Layout (8 GPUs total, 80GB each):
#   GPUs 0-3: Decision Agent  (FSDP training + vLLM rollout, TP=2)
#   GPUs 4-7: Skill Bank Agent (LoRA training + EM inference)
#
# ======================== MODIFY BEFORE RUNNING =============================
# 1. STORAGE_PATH          where all outputs (checkpoints, bank, rollouts) go
# 2. Decision_base_model   HF model ID or local path for the Decision Agent
# 3. SkillBank_base_model  HF model ID or local path for the Skill Bank Agent
# 4. Model_abbr            short name for experiment tracking / filenames
# ============================================================================
#
# Usage:
#   bash scripts/coevolution_train.sh
#
#   # Custom:
#   Decision_base_model=Qwen/Qwen3-8B \
#   SkillBank_base_model=Qwen/Qwen3-8B \
#   NUM_ITERATIONS=10 TRAIN_STEPS=30 \
#     bash scripts/coevolution_train.sh
#
#   # Warm-start from SFT cold-start adapters (run_sft_coldstart.sh output):
#   LOAD_DECISION_ADAPTERS=runs/sft_coldstart/decision \
#   LOAD_SKILLBANK_ADAPTERS=runs/sft_coldstart/skillbank \
#     bash scripts/coevolution_train.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --------------- (1) STORAGE PATH ---------------
export STORAGE_PATH="${STORAGE_PATH:-${REPO_ROOT}/runs/coevolution}"

# --------------- (2) BASE MODELS ---------------
Decision_base_model="${Decision_base_model:-Qwen/Qwen3-8B}"
SkillBank_base_model="${SkillBank_base_model:-Qwen/Qwen3-8B}"

# --------------- (3) MODEL ABBREVIATION ---------------
Model_abbr="${Model_abbr:-CoEvo-Decision14B-SkillBank8B}"

# --------------- (3b) WARM-START FROM SFT COLD-START ADAPTERS ---------------
# Paths to pre-trained adapters from run_sft_coldstart.sh.
# Decision:  contains sub-dirs skill_selection/, action_taking/
# SkillBank: contains sub-dirs segment/, contract/, curator/
#
# Usage:
#   LOAD_DECISION_ADAPTERS=runs/sft_coldstart/decision \
#   LOAD_SKILLBANK_ADAPTERS=runs/sft_coldstart/skillbank \
#     bash scripts/coevolution_train.sh
LOAD_DECISION_ADAPTERS="${LOAD_DECISION_ADAPTERS:-}"
LOAD_SKILLBANK_ADAPTERS="${LOAD_SKILLBANK_ADAPTERS:-}"

# --------------- (4) GPU MEMORY ---------------
export GPU_MEM="${GPU_MEM:-80}"
if [ "$GPU_MEM" != "40" ] && [ "$GPU_MEM" != "80" ]; then
    echo "ERROR: GPU_MEM must be 40 or 80 (got: $GPU_MEM)."
    exit 1
fi

# --------------- (5) TRAINING HYPERPARAMETERS ---------------
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
NUM_ITERATIONS="${NUM_ITERATIONS:-6}"

# Decision Agent
export DECISION_ROLLOUT_BATCH_SIZE="${DECISION_ROLLOUT_BATCH_SIZE:-64}"
export DECISION_GLOBAL_BATCH_SIZE="${DECISION_GLOBAL_BATCH_SIZE:-64}"
export DECISION_GROUP_SIZE="${DECISION_GROUP_SIZE:-8}"

# Skill Bank Agent
export SKILLBANK_BASE_MODEL="$SkillBank_base_model"
export EM_MAX_ITERATIONS="${EM_MAX_ITERATIONS:-3}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_LR="${LORA_LR:-2.0e-4}"
export LORA_EPOCHS="${LORA_EPOCHS:-3}"
export LORA_BATCH_SIZE="${LORA_BATCH_SIZE:-4}"
export LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-4}"

# Cold-start rollout collection
COLDSTART_EPISODES="${COLDSTART_EPISODES:-100}"
COLDSTART_MAX_STEPS="${COLDSTART_MAX_STEPS:-50}"

echo "============================================"
echo "  Co-Evolution Training"
echo "============================================"
echo "  Decision Agent:    $Decision_base_model (Qwen3-8B)"
echo "  Skill Bank Agent:  $SkillBank_base_model (Qwen3-8B + LoRA)"
echo "  Experiment:        $Model_abbr"
echo "  Storage:           $STORAGE_PATH"
echo "  GPU Memory:        ${GPU_MEM}GB"
echo "  GRPO Steps/iter:   $TRAIN_STEPS"
echo "  Num Iterations:    $NUM_ITERATIONS"
echo "  EM Iterations:     $EM_MAX_ITERATIONS"
echo "  LoRA Rank:         $LORA_RANK"
if [ -n "$LOAD_DECISION_ADAPTERS" ]; then
echo "  Decision SFT:      $LOAD_DECISION_ADAPTERS"
fi
if [ -n "$LOAD_SKILLBANK_ADAPTERS" ]; then
echo "  SkillBank SFT:     $LOAD_SKILLBANK_ADAPTERS"
fi
echo "============================================"
echo ""

mkdir -p "$STORAGE_PATH/models" \
         "$STORAGE_PATH/rollouts" \
         "$STORAGE_PATH/skillbank" \
         "$STORAGE_PATH/skillbank/diffs" \
         "$STORAGE_PATH/lora_adapters" \
         "$STORAGE_PATH/logs" \
         "$STORAGE_PATH/logs/skillbank" \
         "$STORAGE_PATH/temp_results" \
         "$STORAGE_PATH/evaluation"

# ---------------------------------------------------------------------------
# GPU cleanup between phases
# ---------------------------------------------------------------------------
cleanup_gpu_for_next_phase() {
    echo "[coevolution] Cleaning up GPU before next phase..."
    pkill -9 -f "python.*-m vllm" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "start_vllm_server\.py" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 -f "verl\.trainer\.main" 2>/dev/null || true
    pkill -9 -f "trainer\.decision" 2>/dev/null || true
    pkill -9 -f "trainer\.skillbank" 2>/dev/null || true
    pkill -9 -f "train_lora" 2>/dev/null || true
    echo "[coevolution] Killing GPU compute processes (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    echo "[coevolution] Waiting for GPU memory release..."
    sleep 10
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[coevolution] GPU cache cleared on {torch.cuda.device_count()} device(s)')
except Exception as e:
    print(f'[coevolution] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[coevolution] GPU cleanup done."
}

# ---------------------------------------------------------------------------
# Collect rollouts using the Decision Agent
# ---------------------------------------------------------------------------
collect_rollouts() {
    local decision_model=$1
    local bank_path=$2
    local rollout_name=$3

    ROLLOUT_OUT="${STORAGE_PATH}/rollouts/${rollout_name}.jsonl"
    if [ -s "$ROLLOUT_OUT" ]; then
        echo "[collect_rollouts] Already exists (non-empty): $ROLLOUT_OUT — skipping."
        return 0
    fi
    # Remove stale empty files from previous failed runs
    [ -f "$ROLLOUT_OUT" ] && [ ! -s "$ROLLOUT_OUT" ] && rm -f "$ROLLOUT_OUT"

    echo "[collect_rollouts] Generating rollouts with $decision_model..."
    BANK_ARG=""
    [ -n "$bank_path" ] && [ -d "$bank_path" ] && BANK_ARG="--bank-path $bank_path"

    # Use cold-start infrastructure to collect rollouts
    if [ -f "cold_start/run_100_rollouts.py" ]; then
        python3 cold_start/run_100_rollouts.py \
            --episodes "$COLDSTART_EPISODES" \
            --max_steps "$COLDSTART_MAX_STEPS" \
            --model "$decision_model" \
            --output_dir "${STORAGE_PATH}/rollouts/${rollout_name}" \
            --no_label --resume \
            2>&1 | tee "${STORAGE_PATH}/logs/${rollout_name}_collect.log"

        # Convert individual episode JSONs → single JSONL
        # Episodes live in per-game subdirs (e.g. rollouts_v0/tetris/episode_000.json)
        if [ -d "${STORAGE_PATH}/rollouts/${rollout_name}" ]; then
            python3 -c "
import json, glob, os
out_dir = '${STORAGE_PATH}/rollouts/${rollout_name}'
# Recursive glob: episodes are in per-game subdirectories
episodes = sorted(glob.glob(os.path.join(out_dir, '**', 'episode_*.json'), recursive=True))
episodes = [e for e in episodes if 'buffer' not in e]
with open('$ROLLOUT_OUT', 'w') as f:
    for ep_path in episodes:
        with open(ep_path) as ef:
            data = json.load(ef)
            f.write(json.dumps(data, default=str) + '\n')
print(f'Consolidated {len(episodes)} episodes → $ROLLOUT_OUT')
if not episodes:
    print('WARNING: No episode files found. Check rollout collection logs.')
"
        fi
    else
        echo "[collect_rollouts] cold_start/run_100_rollouts.py not found."
        echo "[collect_rollouts] Creating placeholder rollout file."
        echo '{}' > "$ROLLOUT_OUT"
    fi
}

# =============================================================================
# Warm-start: merge SFT cold-start adapters into initial checkpoints
# =============================================================================
Decision_init_model="$Decision_base_model"

if [ -n "$LOAD_DECISION_ADAPTERS" ]; then
    MERGED_DECISION="${STORAGE_PATH}/models/decision_sft_merged"
    if [ -f "${MERGED_DECISION}/config.json" ]; then
        echo "[warm-start] Decision SFT-merged model already exists, reusing."
    else
        echo "[warm-start] Merging Decision SFT adapters into base model..."
        mkdir -p "$MERGED_DECISION"
        python3 -c "
import sys, os, json, glob
sys.path.insert(0, '${REPO_ROOT}')

adapter_root = '${LOAD_DECISION_ADAPTERS}'
base_model   = '${Decision_base_model}'
output_dir   = '${MERGED_DECISION}'

# Discover sub-adapters (skill_selection, action_taking, etc.)
sub_dirs = sorted([
    d for d in glob.glob(os.path.join(adapter_root, '*'))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'adapter_config.json'))
])
if not sub_dirs:
    print(f'WARNING: No adapters found under {adapter_root}, using base model as-is.')
    # Symlink base model so downstream sees a valid checkpoint
    os.symlink(os.path.abspath(base_model), os.path.join(output_dir, '_base_link'))
    sys.exit(0)

print(f'Found {len(sub_dirs)} decision adapter(s): {[os.path.basename(d) for d in sub_dirs]}')

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cpu',
    )
    for adapter_path in sub_dirs:
        adapter_name = os.path.basename(adapter_path)
        print(f'  Merging adapter: {adapter_name} from {adapter_path}')
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
        model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Merged decision model saved to {output_dir}')
except ImportError as e:
    print(f'peft/transformers not available ({e}), copying base model config as fallback.')
    import shutil
    for f in ['config.json', 'generation_config.json', 'tokenizer.json',
              'tokenizer_config.json', 'special_tokens_map.json']:
        src = os.path.join(base_model, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))
" 2>&1 | tee "${STORAGE_PATH}/logs/decision_sft_merge.log"
    fi
    if [ -f "${MERGED_DECISION}/config.json" ]; then
        Decision_init_model="$MERGED_DECISION"
        echo "[warm-start] Decision Agent will start from: $Decision_init_model"
    else
        echo "[warm-start] Merge produced no checkpoint; falling back to base model."
    fi
    cleanup_gpu_for_next_phase
    sleep 3
fi

SKILLBANK_INIT_ADAPTERS=""
if [ -n "$LOAD_SKILLBANK_ADAPTERS" ]; then
    echo "[warm-start] Seeding Skill Bank LoRA adapters from SFT cold-start..."
    SKILLBANK_V1_ADAPTER_DIR="${STORAGE_PATH}/lora_adapters/${Model_abbr}_skillbank_v1"
    mkdir -p "$SKILLBANK_V1_ADAPTER_DIR"

    python3 -c "
import os, sys, shutil, json, glob
sys.path.insert(0, '${REPO_ROOT}')

src_root = '${LOAD_SKILLBANK_ADAPTERS}'
dst_root = '${SKILLBANK_V1_ADAPTER_DIR}'

# Map SFT adapter names → skillbank EM stage names
# SFT produces: segment, contract, curator
# EM expects:   boundary, segment, contract, retrieval
STAGE_MAP = {
    'segment':  'segment',
    'contract': 'contract',
    'curator':  'retrieval',
}

sub_dirs = [
    d for d in glob.glob(os.path.join(src_root, '*'))
    if os.path.isdir(d) and os.path.exists(os.path.join(d, 'adapter_config.json'))
]
print(f'Found {len(sub_dirs)} SFT skillbank adapter(s): {[os.path.basename(d) for d in sub_dirs]}')

for adapter_path in sub_dirs:
    name = os.path.basename(adapter_path)
    target_stage = STAGE_MAP.get(name, name)
    dst = os.path.join(dst_root, target_stage)
    if os.path.exists(os.path.join(dst, 'adapter_config.json')):
        print(f'  {target_stage}: already seeded, skipping.')
        continue
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(adapter_path):
        src_f = os.path.join(adapter_path, f)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, os.path.join(dst, f))
    print(f'  {name} → {target_stage}: copied to {dst}')

print('Skill Bank adapter seeding complete.')
" 2>&1 | tee "${STORAGE_PATH}/logs/skillbank_sft_seed.log"

    SKILLBANK_INIT_ADAPTERS="$SKILLBANK_V1_ADAPTER_DIR"
    echo "[warm-start] Skill Bank adapters seeded at: $SKILLBANK_INIT_ADAPTERS"
fi

# =============================================================================
# Iteration 1: Both agents start from their respective base/warm models
# =============================================================================
echo "=========================================="
echo "Starting Iteration 1"
echo "=========================================="

# --- Phase 1a: Cold-start rollouts with Decision Agent (base or SFT-merged) ---
ROLLOUT_V0="${Model_abbr}_rollouts_v0"
if [ -s "${STORAGE_PATH}/rollouts/${ROLLOUT_V0}.jsonl" ]; then
    echo "[Iter 1] Cold-start rollouts already exist, skipping..."
else
    echo "[Iter 1] Collecting cold-start rollouts with Decision Agent ($Decision_init_model)..."
    collect_rollouts "$Decision_init_model" "" "$ROLLOUT_V0"
fi

cleanup_gpu_for_next_phase
sleep 5

# --- Phase 1b: Train Skill Bank Agent v1 on cold-start rollouts ---
SKILLBANK_V1="${Model_abbr}_skillbank_v1"
if [ -f "${STORAGE_PATH}/temp_results/${SKILLBANK_V1}_em_result.json" ]; then
    echo "[Iter 1] Skill Bank v1 already trained, skipping..."
else
    echo "[Iter 1] Training Skill Bank Agent v1 (Qwen3-8B + LoRA)..."
    # If SFT adapters were seeded, export the path so skillbank_agent_train.sh
    # can detect pre-existing adapters and skip re-training those stages.
    if [ -n "$SKILLBANK_INIT_ADAPTERS" ]; then
        export SKILLBANK_INIT_ADAPTER_DIR="$SKILLBANK_INIT_ADAPTERS"
    fi
    bash scripts/skillbank_agent_train.sh \
        "${STORAGE_PATH}/rollouts/${ROLLOUT_V0}.jsonl" \
        "" \
        "$SKILLBANK_V1"
    unset SKILLBANK_INIT_ADAPTER_DIR 2>/dev/null || true
fi

cleanup_gpu_for_next_phase
sleep 5

# --- Phase 1c: Train Decision Agent v1 with Bank v1 ---
BANK_PATH="${STORAGE_PATH}/skillbank"
DECISION_V1="${Model_abbr}_decision_v1"
if [ -d "${STORAGE_PATH}/models/${DECISION_V1}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] Decision Agent v1 already exists, skipping..."
else
    echo "[Iter 1] Training Decision Agent v1 (Qwen3-8B GRPO)..."
    bash scripts/decision_agent_train.sh \
        "$Decision_init_model" \
        "$DECISION_V1" \
        "$BANK_PATH"
fi

cleanup_gpu_for_next_phase
sleep 5

# =============================================================================
# Iterations 2+: Each agent evolves from its previous version
# =============================================================================
for i in $(seq 2 $NUM_ITERATIONS); do
    prev=$((i-1))
    echo ""
    echo "=========================================="
    echo "Starting Iteration $i"
    echo "=========================================="

    DECISION_PREV="${STORAGE_PATH}/models/${Model_abbr}_decision_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface"

    # --- Phase A: Collect rollouts with Decision Agent v_{i-1} ---
    ROLLOUT_NAME="${Model_abbr}_rollouts_v${prev}"
    if [ -s "${STORAGE_PATH}/rollouts/${ROLLOUT_NAME}.jsonl" ]; then
        echo "[Iter $i] Rollouts v${prev} already exist, skipping..."
    else
        echo "[Iter $i] Collecting rollouts with Decision Agent v${prev}..."
        collect_rollouts "$DECISION_PREV" "$BANK_PATH" "$ROLLOUT_NAME"
    fi

    cleanup_gpu_for_next_phase
    sleep 5

    # --- Phase B: Train Skill Bank Agent v_i on new rollouts ---
    SKILLBANK_NAME="${Model_abbr}_skillbank_v${i}"
    if [ -f "${STORAGE_PATH}/temp_results/${SKILLBANK_NAME}_em_result.json" ]; then
        echo "[Iter $i] Skill Bank v${i} already trained, skipping..."
    else
        echo "[Iter $i] Training Skill Bank Agent v${i} (Qwen3-8B + LoRA)..."
        bash scripts/skillbank_agent_train.sh \
            "${STORAGE_PATH}/rollouts/${ROLLOUT_NAME}.jsonl" \
            "$BANK_PATH" \
            "$SKILLBANK_NAME"
    fi

    cleanup_gpu_for_next_phase
    sleep 5

    # --- Phase C: Train Decision Agent v_i with updated bank ---
    DECISION_NAME="${Model_abbr}_decision_v${i}"
    if [ -d "${STORAGE_PATH}/models/${DECISION_NAME}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] Decision Agent v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training Decision Agent v${i} (Qwen3-8B GRPO)..."
        bash scripts/decision_agent_train.sh \
            "$DECISION_PREV" \
            "$DECISION_NAME" \
            "$BANK_PATH"
    fi

    cleanup_gpu_for_next_phase
    sleep 5
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "  Co-Evolution Training Complete!"
echo "=========================================="
echo ""
echo "  Decision Agent (Qwen3-8B):"
for i in $(seq 1 $NUM_ITERATIONS); do
    DEC_PATH="${STORAGE_PATH}/models/${Model_abbr}_decision_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface"
    if [ -d "$DEC_PATH" ]; then
        echo "    v${i}: $DEC_PATH"
    else
        echo "    v${i}: (not found)"
    fi
done
echo ""
echo "  Skill Bank (Qwen3-8B + LoRA):"
echo "    Bank snapshots: ${STORAGE_PATH}/skillbank/"
echo "    LoRA adapters:  ${STORAGE_PATH}/lora_adapters/"
echo ""
echo "  Rollouts:  ${STORAGE_PATH}/rollouts/"
echo "  Logs:      ${STORAGE_PATH}/logs/"
echo ""
echo "  To run the standalone co-evolution loop (Python):"
echo "    python -m trainer.launch_coevolution \\"
echo "        --decision-config trainer/common/configs/decision_grpo.yaml \\"
echo "        --skillbank-config trainer/common/configs/skillbank_em.yaml"
echo ""
