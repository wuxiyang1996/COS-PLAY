#!/bin/bash
# =============================================================================
# Co-Evolution Framework: Decision Agent Training (GRPO)
# =============================================================================
# Trains the Decision Agent (Qwen3-8B) using Group Relative Policy
# Optimization via VERL on game environment rollouts.
#
# The Decision Agent selects primitive game actions and tool calls
# (QUERY_SKILL, CALL_SKILL, QUERY_MEM) against the current skill bank
# maintained by the Skill Bank Agent (Qwen3-8B).
#
# GPU Layout:
#   GPUs 0-3: GRPO training (FSDP) + vLLM rollout (TP=2)
#   GPUs 4-7: Available for Skill Bank vLLM services
#
# Usage:
#   bash scripts/decision_agent_train.sh \
#       <decision_model_path> <save_name> [bank_snapshot_path]
#
# Arguments:
#   $1  decision_model_path  HF model ID or local checkpoint
#   $2  save_name            experiment name for checkpoints / wandb / logs
#   $3  bank_snapshot_path   (optional) path to a skill bank snapshot
# =============================================================================

set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Cleanup: kill all training processes on exit to free GPUs.
# ---------------------------------------------------------------------------
cleanup_decision_training() {
    echo "[decision_train] CLEANUP: Freeing GPUs and killing processes..."
    pkill -9 -f "python.*-m vllm" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 -f "verl\.trainer\.main" 2>/dev/null || true
    pkill -9 -f "trainer\.decision\.launch_train" 2>/dev/null || true

    echo "[decision_train] Killing GPU compute processes (nvidia-smi)..."
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
        print(f'[decision_train] GPU cache cleared on {torch.cuda.device_count()} device(s)')
except Exception as e:
    print(f'[decision_train] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[decision_train] Cleanup complete."
}
trap cleanup_decision_training EXIT INT TERM

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
decision_model_path=${1:-Qwen/Qwen3-8B}
save_name=${2:-decision_agent_v1}
bank_snapshot_path=${3:-}

echo "========================================"
echo "Decision Agent Training (GRPO): $save_name"
echo "  Decision model: $decision_model_path"
echo "  Bank snapshot:  ${bank_snapshot_path:-none (empty bank)}"
echo "========================================"

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
export STORAGE_PATH="${STORAGE_PATH:-${REPO_ROOT}/runs/coevolution}"
export GPU_MEM="${GPU_MEM:-80}"
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ROLLOUT_BATCH_SIZE=${DECISION_ROLLOUT_BATCH_SIZE:-64}
GLOBAL_BATCH_SIZE=${DECISION_GLOBAL_BATCH_SIZE:-64}
GROUP_SIZE=${DECISION_GROUP_SIZE:-8}

echo "STORAGE_PATH=$STORAGE_PATH"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "GPU_MEM=${GPU_MEM}GB"

mkdir -p "$STORAGE_PATH/models" \
         "$STORAGE_PATH/rollouts" \
         "$STORAGE_PATH/skillbank" \
         "$STORAGE_PATH/logs" \
         "$STORAGE_PATH/temp_results"

# Pre-flight GPU cleanup
echo "[Pre-flight] Cleaning up GPU memory..."
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
echo "[Pre-flight] GPU cleanup done."

# ---------------------------------------------------------------------------
# Build VERL overrides
# ---------------------------------------------------------------------------
EXTRA_ARGS=""
[ -n "$ROLLOUT_BATCH_SIZE" ] && EXTRA_ARGS="$EXTRA_ARGS data.rollout_batch_size=$ROLLOUT_BATCH_SIZE"
[ -n "$GLOBAL_BATCH_SIZE" ] && EXTRA_ARGS="$EXTRA_ARGS worker.actor.global_batch_size=$GLOBAL_BATCH_SIZE worker.critic.global_batch_size=$GLOBAL_BATCH_SIZE"

COEVO_ARGS=""
if [ -n "$bank_snapshot_path" ]; then
    COEVO_ARGS="coevolution.initial_bank_path=$bank_snapshot_path"
fi

# ---------------------------------------------------------------------------
# Train Decision Agent (Qwen3-8B) with GRPO on GPUs 0-3
# ---------------------------------------------------------------------------
echo "[decision_train] Starting GRPO training (max_steps=$TRAIN_STEPS, group_size=$GROUP_SIZE)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=scripts/configs/decision_agent_grpo_${GPU_MEM}gb.yaml \
    worker.actor.model.model_path=$decision_model_path \
    worker.rollout.n=$GROUP_SIZE \
    trainer.max_steps=$TRAIN_STEPS \
    trainer.save_freq=$TRAIN_STEPS \
    trainer.experiment_name=$save_name \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_name \
    trainer.total_epochs=10 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=false \
    trainer.val_freq=0 \
    $EXTRA_ARGS \
    $COEVO_ARGS

sleep 5

# ---------------------------------------------------------------------------
# Merge FSDP shards into HuggingFace format
# ---------------------------------------------------------------------------
ACTOR_DIR="${STORAGE_PATH}/models/$save_name/global_step_${TRAIN_STEPS}/actor"
if [ -d "$ACTOR_DIR" ]; then
    HUGGINGFACE_DIR="${ACTOR_DIR}/huggingface"
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "huggingface/config.json missing — copying from base model ($decision_model_path)..."
        mkdir -p "$HUGGINGFACE_DIR"
        for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
                 special_tokens_map.json added_tokens.json merges.txt vocab.json; do
            [ -f "$decision_model_path/$f" ] && cp "$decision_model_path/$f" "$HUGGINGFACE_DIR/"
        done
    fi
    echo "Merging FSDP shards..."
    python3 -c "
import subprocess, sys, os
merger = os.path.join('$REPO_ROOT', '..', '..', 'Self-Agent', 'scripts', 'model_merger.py')
if not os.path.exists(merger):
    merger = os.path.join('$REPO_ROOT', 'scripts', 'model_merger.py')
if os.path.exists(merger):
    subprocess.run([sys.executable, merger, '--local_dir', '$ACTOR_DIR'], check=True)
else:
    print('WARNING: model_merger.py not found; FSDP shards not merged.')
"
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "ERROR: Merged model not found at $HUGGINGFACE_DIR (merge may have failed)."
        exit 1
    fi
    PT_COUNT=$(find "$ACTOR_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    if [ "$PT_COUNT" -gt 0 ]; then
        echo "Removing $PT_COUNT FSDP shard .pt files to save storage..."
        rm -f "$ACTOR_DIR"/*.pt
    fi
else
    echo "ERROR: No checkpoint found at $ACTOR_DIR (training may have failed)."
    exit 1
fi

sleep 10

echo "Decision Agent training finished: $save_name"
echo "  Checkpoint: ${STORAGE_PATH}/models/$save_name/global_step_${TRAIN_STEPS}/actor/huggingface"
