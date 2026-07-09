#!/bin/bash
# =============================================================================
# Co-Evolution Framework: Skill Bank Agent Training (Hard-EM + LoRA)
# =============================================================================
# Trains the Skill Bank Agent (Qwen3-8B) using Hard-EM with function-
# specific LoRA adapters.
#
# The Skill Bank Agent processes trajectory rollouts from the Decision Agent
# through a 4-stage pipeline:
#   Stage 1: Boundary Proposal — detect segment boundaries in trajectories
#   Stage 2: Segmentation Decode — DP/Viterbi skill-label assignment
#   Stage 3: Contract Learning — learn and verify effect contracts
#   Stage 4: Bank Maintenance — refine, materialize, merge, split skills
#
# Each stage has its own LoRA adapter fine-tuned on the Qwen3-8B backbone.
# After EM converges, a gating eval determines whether to accept or
# reject the updated bank.
#
# GPU Layout:
#   GPUs 4-7: LoRA training + vLLM inference for EM stages
#   (GPUs 0-3 may be used by the Decision Agent concurrently)
#
# Usage:
#   bash scripts/skillbank_agent_train.sh \
#       <rollout_parquet> <bank_snapshot_path> <save_name>
#
# Arguments:
#   $1  rollout_parquet     path to rollout data from Decision Agent
#   $2  bank_snapshot_path  path to the current skill bank snapshot (or "" for empty)
#   $3  save_name           experiment name for this EM iteration
# =============================================================================

set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cleanup_skillbank_training() {
    echo "[skillbank_train] CLEANUP: Freeing GPUs and killing processes..."
    pkill -9 -f "train_lora\.py" 2>/dev/null || true
    pkill -9 -f "train_boundary_lora\.py" 2>/dev/null || true
    pkill -9 -f "train_segment_lora\.py" 2>/dev/null || true
    pkill -9 -f "train_contract_lora\.py" 2>/dev/null || true
    pkill -9 -f "train_retrieval_lora\.py" 2>/dev/null || true
    pkill -9 -f "em_trainer\.py" 2>/dev/null || true
    pkill -9 -f "python.*-m vllm" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    ray stop --force 2>/dev/null || true

    echo "[skillbank_train] Killing GPU compute processes (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done

    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[skillbank_train] GPU cache cleared on {torch.cuda.device_count()} device(s)')
except Exception as e:
    print(f'[skillbank_train] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[skillbank_train] Cleanup complete."
}
trap cleanup_skillbank_training EXIT INT TERM

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
rollout_parquet=${1:?Usage: skillbank_agent_train.sh <rollout_parquet> <bank_snapshot> <save_name>}
bank_snapshot_path=${2:-}
save_name=${3:-skillbank_v1}

echo "========================================"
echo "Skill Bank Agent Training (Hard-EM + LoRA): $save_name"
echo "  Rollout data:   $rollout_parquet"
echo "  Bank snapshot:  ${bank_snapshot_path:-none (empty bank)}"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export STORAGE_PATH="${STORAGE_PATH:-${REPO_ROOT}/runs/coevolution}"
export GPU_MEM="${GPU_MEM:-80}"
SKILLBANK_CONFIG="${SKILLBANK_CONFIG:-trainer/common/configs/skillbank_em.yaml}"
SKILLBANK_BASE_MODEL="${SKILLBANK_BASE_MODEL:-Qwen/Qwen3-8B}"
LORA_CONFIG="${LORA_CONFIG:-configs/skillbank_lora.yaml}"

EM_MAX_ITERATIONS=${EM_MAX_ITERATIONS:-3}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_LR=${LORA_LR:-2.0e-4}
LORA_EPOCHS=${LORA_EPOCHS:-3}
LORA_BATCH_SIZE=${LORA_BATCH_SIZE:-4}
LORA_GRAD_ACCUM=${LORA_GRAD_ACCUM:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}

echo "STORAGE_PATH=$STORAGE_PATH"
echo "GPU_MEM=${GPU_MEM}GB"
echo "SKILLBANK_BASE_MODEL=$SKILLBANK_BASE_MODEL"
echo "EM_MAX_ITERATIONS=$EM_MAX_ITERATIONS"

ADAPTER_DIR="${STORAGE_PATH}/lora_adapters/${save_name}"
BANK_DIR="${STORAGE_PATH}/skillbank"
LOG_DIR="${STORAGE_PATH}/logs/skillbank"

mkdir -p "$ADAPTER_DIR" \
         "$BANK_DIR" \
         "${BANK_DIR}/diffs" \
         "$LOG_DIR" \
         "$STORAGE_PATH/temp_results"

# ---------------------------------------------------------------------------
# Helper: kill GPU processes between stages
# ---------------------------------------------------------------------------
_kill_gpu_processes() {
    local label="${1:-skillbank}"
    echo "[$label] Killing GPU compute processes..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
except Exception:
    pass
" 2>/dev/null || true
    sleep 3
}

# Pre-flight cleanup
echo "[Pre-flight] Cleaning up GPU memory..."
_kill_gpu_processes "Pre-flight"

# ---------------------------------------------------------------------------
# Step 1: Ingest rollouts → TrajectoryForEM format
# ---------------------------------------------------------------------------
echo ""
echo "========== PHASE 1: INGEST ROLLOUTS =========="
echo "[Step 1] Ingesting rollouts from $rollout_parquet..."
TRAJ_FILE="${STORAGE_PATH}/temp_results/${save_name}_trajectories.json"

python3 -c "
import json, os, sys
sys.path.insert(0, '${REPO_ROOT}')

rollout_path = '$rollout_parquet'

if not os.path.exists(rollout_path):
    print(f'ERROR: Rollout file not found: {rollout_path}')
    sys.exit(1)

if os.path.getsize(rollout_path) == 0:
    print(f'ERROR: Rollout file is empty (0 bytes): {rollout_path}')
    sys.exit(1)

records = None

if rollout_path.endswith('.jsonl'):
    try:
        from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records
        episodes = load_episodes_from_jsonl(rollout_path)
        records = episodes_to_rollout_records(episodes)
    except ImportError:
        # Minimal JSONL loader — no extra deps needed
        records = []
        with open(rollout_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
elif rollout_path.endswith('.parquet'):
    import pandas as pd
    df = pd.read_parquet(rollout_path)
    records = df.to_dict(orient='records')
elif rollout_path.endswith('.json'):
    with open(rollout_path) as f:
        records = json.load(f)
else:
    print(f'ERROR: Unsupported format: {rollout_path}')
    sys.exit(1)

if not records:
    print(f'ERROR: No rollout records loaded from {rollout_path}')
    sys.exit(1)

print(f'Loaded {len(records)} rollout records')
with open('$TRAJ_FILE', 'w') as f:
    json.dump(records, f, default=str)
print(f'Saved trajectories to $TRAJ_FILE')
" 2>&1 | tee "${LOG_DIR}/${save_name}_ingest.log"

INGEST_EXIT=${PIPESTATUS[0]}
if [ "$INGEST_EXIT" -ne 0 ]; then
    echo "[Step 1] ERROR: Ingest failed (exit code $INGEST_EXIT). Cannot proceed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Train LoRA adapters for each stage on GPUs 4-7
# ---------------------------------------------------------------------------
echo ""
echo "========== PHASE 2: LORA ADAPTER TRAINING (Qwen3-8B) =========="

for STAGE in boundary segment contract retrieval; do
    STAGE_ADAPTER_DIR="${ADAPTER_DIR}/${STAGE}"

    if [ -f "${STAGE_ADAPTER_DIR}/adapter_config.json" ]; then
        echo "[Step 2] ${STAGE} adapter already exists at ${STAGE_ADAPTER_DIR}, skipping..."
        continue
    fi

    echo "[Step 2] Training ${STAGE} LoRA adapter on Qwen3-8B..."
    mkdir -p "$STAGE_ADAPTER_DIR"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -c "
import os, sys, json
sys.path.insert(0, '${REPO_ROOT}')

stage = '$STAGE'
base_model = '$SKILLBANK_BASE_MODEL'
output_dir = '$STAGE_ADAPTER_DIR'
traj_file = '$TRAJ_FILE'
bank_path = '${bank_snapshot_path}' if '${bank_snapshot_path}' else None

try:
    from trainer.skillbank.lora.train_lora import train_lora_adapter
    train_lora_adapter(
        base_model=base_model,
        task=stage,
        data_path=traj_file,
        bank_path=bank_path,
        output_dir=output_dir,
        lora_rank=$LORA_RANK,
        lora_alpha=$LORA_ALPHA,
        lr=$LORA_LR,
        num_epochs=$LORA_EPOCHS,
        batch_size=$LORA_BATCH_SIZE,
        gradient_accumulation_steps=$LORA_GRAD_ACCUM,
        max_seq_length=$MAX_SEQ_LENGTH,
    )
    print(f'LoRA adapter ({stage}) saved to {output_dir}')
except ImportError as e:
    print(f'trainer.skillbank.lora.train_lora not available: {e}')
    print(f'Using base model for {stage} (no adapter)')
    with open(os.path.join(output_dir, 'adapter_config.json'), 'w') as f:
        json.dump({'status': 'fallback_to_base', 'task': stage, 'base_model': base_model}, f, indent=2)
except Exception as e:
    print(f'ERROR training {stage} adapter: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "${LOG_DIR}/${save_name}_lora_${STAGE}.log"

    LORA_EXIT=${PIPESTATUS[0]}
    if [ "$LORA_EXIT" -ne 0 ]; then
        echo "[Step 2] ERROR: ${STAGE} LoRA training failed (exit code $LORA_EXIT)."
        exit 1
    fi

    _kill_gpu_processes "Step 2 (${STAGE})"
done

# ---------------------------------------------------------------------------
# Step 3: Run Hard-EM iterations with trained LoRA adapters
# ---------------------------------------------------------------------------
echo ""
echo "========== PHASE 3: HARD-EM (max_iterations=$EM_MAX_ITERATIONS) =========="

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -c "
import os, sys, json, logging
sys.path.insert(0, '${REPO_ROOT}')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

try:
    import yaml
    from trainer.skillbank.em_trainer import EMConfig, EMTrainer
    from trainer.skillbank.bank_io.bank_store import VersionedBankStore
    from trainer.skillbank.bank_io.diff_logger import DiffLogger
    from trainer.skillbank.ingest_rollouts import ingest_rollouts
    from trainer.launch_coevolution import _build_em_config

    with open('$SKILLBANK_CONFIG') as f:
        skillbank_cfg = yaml.safe_load(f)

    # Wire in the LoRA adapter paths we just trained
    skillbank_cfg.setdefault('lora', {})
    skillbank_cfg['lora']['base_model_name_or_path'] = '$SKILLBANK_BASE_MODEL'
    skillbank_cfg['lora']['adapter_paths'] = {
        'boundary':  '${ADAPTER_DIR}/boundary',
        'segment':   '${ADAPTER_DIR}/segment',
        'contract':  '${ADAPTER_DIR}/contract',
        'retrieval': '${ADAPTER_DIR}/retrieval',
    }

    # Initialize the multi-LoRA model so EM stages (boundary proposal,
    # segmentation decode, etc.) use the trained adapters instead of
    # falling back to the API-based ask_model.
    try:
        from skill_agents.lora import MultiLoraSkillBankLLM, MultiLoraConfig
        lora_cfg = MultiLoraConfig.from_dict(skillbank_cfg['lora'])
        lora_llm = MultiLoraSkillBankLLM(lora_cfg)
        lora_llm.load()
        MultiLoraSkillBankLLM.set_shared_instance(lora_llm)
        print(f'Multi-LoRA model initialized: {lora_llm.loaded_adapters}')
    except Exception as e:
        print(f'WARNING: Multi-LoRA init failed ({e}), EM will use API fallback')

    em_cfg = _build_em_config(skillbank_cfg)

    # Load or create skill bank
    from skill_agents.skill_bank.bank import SkillBankMVP
    bank = SkillBankMVP()
    bank_snapshot = '${bank_snapshot_path}'
    if bank_snapshot:
        bank.load(bank_snapshot)
        print(f'Loaded existing bank from {bank_snapshot}')

    bank_store = VersionedBankStore(
        bank=bank,
        bank_dir='${BANK_DIR}',
        snapshot_prefix=skillbank_cfg.get('bank_io', {}).get('snapshot_prefix', 'bank_v'),
        max_snapshots=skillbank_cfg.get('bank_io', {}).get('max_snapshots', 20),
    )
    diff_logger = DiffLogger(diff_dir='${BANK_DIR}/diffs')
    em_trainer = EMTrainer(bank_store=bank_store, config=em_cfg, diff_logger=diff_logger)

    # Load trajectories
    with open('$TRAJ_FILE') as f:
        raw = json.load(f)
    trajectories = ingest_rollouts(raw)
    print(f'Ingested {len(trajectories)} trajectories for EM')

    # Run EM
    result = em_trainer.run(trajectories)

    summary = {
        'accepted': result.accepted,
        'bank_version': result.bank_version,
        'n_iterations': len(result.iterations),
        'rejection_reason': result.rejection_reason or '',
    }
    if result.iterations:
        last = result.iterations[-1]
        summary.update({
            'n_segments': last.n_segments,
            'n_new': last.n_new,
            'new_rate': last.new_rate,
            'mean_margin': last.mean_margin,
            'mean_pass_rate': last.mean_pass_rate,
        })

    out_path = '${STORAGE_PATH}/temp_results/${save_name}_em_result.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'EM result: {json.dumps(summary, indent=2)}')

    if result.accepted:
        print(f'Bank update ACCEPTED — new version v{result.bank_version}')
        bank_store.save()
    else:
        print(f'Bank update REJECTED: {result.rejection_reason}')

except ImportError as e:
    print(f'Import error: {e}')
    print('Running in dry-run mode (skill pipeline modules not fully available)')
    out_path = '${STORAGE_PATH}/temp_results/${save_name}_em_result.json'
    with open(out_path, 'w') as f:
        json.dump({'accepted': False, 'rejection_reason': str(e)}, f, indent=2)
except Exception as e:
    print(f'ERROR during EM: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "${LOG_DIR}/${save_name}_em.log"

EM_EXIT=${PIPESTATUS[0]}
if [ "$EM_EXIT" -ne 0 ]; then
    echo "[Step 3] ERROR: Hard-EM failed (exit code $EM_EXIT)."
    exit 1
fi

_kill_gpu_processes "Step 3 done"

# ---------------------------------------------------------------------------
# Step 4: Report gating result
# ---------------------------------------------------------------------------
echo ""
echo "========== PHASE 4: GATING EVALUATION =========="
EM_RESULT="${STORAGE_PATH}/temp_results/${save_name}_em_result.json"
if [ -f "$EM_RESULT" ]; then
    ACCEPTED=$(python3 -c "import json; print(json.load(open('$EM_RESULT')).get('accepted', False))")
    if [ "$ACCEPTED" = "True" ]; then
        echo "[Step 4] Bank update ACCEPTED — new bank deployed to ${BANK_DIR}/"
    else
        REASON=$(python3 -c "import json; r=json.load(open('$EM_RESULT')); print(r.get('rejection_reason', r.get('reason', 'unknown')))")
        echo "[Step 4] Bank update REJECTED: $REASON"
        echo "[Step 4] Keeping previous bank version."
    fi
else
    echo "[Step 4] No EM result found — skipping gating."
fi

sleep 5

echo ""
echo "Skill Bank Agent training finished: $save_name"
echo "  Adapters:  ${ADAPTER_DIR}/"
echo "  Bank:      ${BANK_DIR}/"
echo "  EM result: ${EM_RESULT}"
