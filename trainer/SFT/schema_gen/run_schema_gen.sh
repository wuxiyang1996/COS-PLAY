#!/usr/bin/env bash
# schema_gen LoRA SFT launcher (vision track).
#
# Default base model: Qwen/Qwen3.5-35B-A3B (unified vision-language MoE,
# 35B total / ~3B active).  Override with --model-name for a smaller
# variant (Qwen/Qwen3-VL-8B-Instruct fits on one A100-80GB).
#
# Hardware
# --------
# * 35B-A3B base   : ≥2× H100-80GB (or ≥4× A100-80GB) with ZeRO-3 / FSDP.
# * 32B dense      : 2× A100-80GB with deepspeed ZeRO-3.
# *  8B Instruct   : single A100-80GB with the conservative defaults.
#
# Pass --use-deepspeed to wrap the trainer in `accelerate launch
# --config_file <ds_zero3.yaml>`.  This script does not assume a
# specific accelerate config; pin one via ACCELERATE_CONFIG_FILE
# beforehand if you need a custom recipe.
#
# Pipeline
# --------
# Phase 1  Build (frame, vision_schema, env_id, …) triples from the
#          existing cold-start rollouts using
#          labeling/build_schema_gen_triples.py.  Skipped with
#          --skip-build.
# Phase 2  Inspect-only audit through trainer/SFT/schema_gen/train.py
#          (prints per-domain row counts).
# Phase 3  Launch trainer/SFT/schema_gen/train.py with the resolved
#          base model + conservative LoRA hyper-parameters.
#
# Common usage
# ------------
#   # Smoke (8B model, 64 samples per domain, audit-only).
#   bash trainer/SFT/schema_gen/run_schema_gen.sh --smoke
#
#   # 35B-A3B sharded across 2 GPUs (ZeRO-3) — pin GPUs 5,6:
#   CUDA_VISIBLE_DEVICES=5,6 bash trainer/SFT/schema_gen/run_schema_gen.sh \
#       --num-gpus 2
#
#   # 35B-A3B sharded across 3 GPUs:
#   CUDA_VISIBLE_DEVICES=5,6,7 bash trainer/SFT/schema_gen/run_schema_gen.sh \
#       --num-gpus 3
#
#   # 35B-A3B on a single GPU (only fits on a fully-free 143 GB H200):
#   CUDA_VISIBLE_DEVICES=5 bash trainer/SFT/schema_gen/run_schema_gen.sh
#
#   # Smaller variant on a single GPU (fits with vLLM workers running):
#   CUDA_VISIBLE_DEVICES=5 bash trainer/SFT/schema_gen/run_schema_gen.sh \
#       --model-name Qwen/Qwen3-VL-8B-Instruct
#
#   # Override the bundled ZeRO-3 config:
#   ACCELERATE_CONFIG_FILE=path/to/my.yaml \
#       bash trainer/SFT/schema_gen/run_schema_gen.sh --num-gpus 4
#
#   # Refresh triples, audit-only, no training.
#   bash trainer/SFT/schema_gen/run_schema_gen.sh --dry-run

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SKIP_BUILD=""
DRY_RUN=""
SMOKE=""
USE_DEEPSPEED=""
NUM_GPUS=""                # empty -> 1 (single-GPU); set with --num-gpus N
MODEL_NAME=""              # empty -> SchemaGenConfig default (Qwen3.5-35B-A3B)
RUN_ID=""                  # empty -> auto schema_gen_<ts>
DOMAINS=()                 # empty -> default (gymv + env_wrappers)
MAX_SAMPLES_PER_DOMAIN=""
EXTRA_ARGS=()

GYMV_RUN=""                # empty -> auto-pick latest
ENVW_RUN=""                # empty -> auto-pick latest

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build|--skip_build) SKIP_BUILD="1"; shift ;;
        --dry-run|--dry_run)       DRY_RUN="1"; shift ;;
        --use-deepspeed|--use_deepspeed) USE_DEEPSPEED="1"; shift ;;
        --num-gpus|--num_gpus)
            NUM_GPUS="$2"
            USE_DEEPSPEED="${USE_DEEPSPEED:-1}"   # implies --use-deepspeed
            shift 2
            ;;
        --smoke)
            SMOKE="1"
            MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
            MAX_SAMPLES_PER_DOMAIN="${MAX_SAMPLES_PER_DOMAIN:-64}"
            DRY_RUN="${DRY_RUN:-1}"   # smoke = inspect-only by default
            shift
            ;;
        --model-name|--model_name) MODEL_NAME="$2"; shift 2 ;;
        --run-id|--run_id)         RUN_ID="$2"; shift 2 ;;
        --gymv-run|--gymv_run)     GYMV_RUN="$2"; shift 2 ;;
        --envw-run|--envw_run)     ENVW_RUN="$2"; shift 2 ;;
        --max-samples-per-domain|--max_samples_per_domain)
            MAX_SAMPLES_PER_DOMAIN="$2"; shift 2 ;;
        --domains)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DOMAINS+=("$1"); shift
            done
            ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1"); shift
            ;;
    esac
done

export PYTHONPATH="${WORKSPACE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
cd "$REPO_ROOT"

echo "=================================================================="
echo "  schema_gen SFT launcher"
echo "  repo root      : $REPO_ROOT"
echo "  skip_build     : ${SKIP_BUILD:-0}"
echo "  dry_run        : ${DRY_RUN:-0}"
echo "  smoke          : ${SMOKE:-0}"
echo "  deepspeed      : ${USE_DEEPSPEED:-0}"
echo "  num_gpus       : ${NUM_GPUS:-<single-GPU>}"
echo "  model_name     : ${MODEL_NAME:-<config default Qwen3.5-35B-A3B>}"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  run_id         : ${RUN_ID:-<auto>}"
echo "  domains        : ${DOMAINS[*]:-<gymv + env_wrappers>}"
echo "  cap/domain     : ${MAX_SAMPLES_PER_DOMAIN:-<all>}"
echo "  gymv_run       : ${GYMV_RUN:-<latest>}"
echo "  envw_run       : ${ENVW_RUN:-<latest>}"
echo "  started        : $(date -Iseconds)"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Phase 1 — build triples
# ---------------------------------------------------------------------------
if [[ -z "$SKIP_BUILD" ]]; then
    echo
    echo "[Phase 1] building schema_gen triples ..."
    build_args=()
    [[ -n "$GYMV_RUN" ]] && build_args+=(--gymv-run "$GYMV_RUN")
    [[ -n "$ENVW_RUN" ]] && build_args+=(--envw-run "$ENVW_RUN")
    python labeling/build_schema_gen_triples.py "${build_args[@]}" || {
        echo "[Phase 1] FAILED" >&2
        exit 2
    }
else
    echo "[Phase 1] skipped (--skip-build)"
fi

# ---------------------------------------------------------------------------
# Phase 2 — inspect-only audit through train.py
# ---------------------------------------------------------------------------
echo
echo "[Phase 2] inspect-only audit through train.py ..."
inspect_args=(--inspect_only)
[[ -n "$MODEL_NAME" ]] && inspect_args+=(--model_name "$MODEL_NAME")
[[ -n "$MAX_SAMPLES_PER_DOMAIN" ]] && inspect_args+=(--max_samples_per_domain "$MAX_SAMPLES_PER_DOMAIN")
if [[ ${#DOMAINS[@]} -gt 0 ]]; then
    inspect_args+=(--domains "${DOMAINS[@]}")
fi
python -m trainer.SFT.schema_gen.train "${inspect_args[@]}" || {
    echo "[Phase 2] FAILED" >&2
    exit 2
}

if [[ -n "$DRY_RUN" ]]; then
    echo
    echo "[dry-run] Phase 3 (training) skipped."
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 3 — launch trainer
# ---------------------------------------------------------------------------
train_args=()
[[ -n "$MODEL_NAME" ]] && train_args+=(--model_name "$MODEL_NAME")
[[ -n "$RUN_ID" ]] && train_args+=(--run_id "$RUN_ID")
[[ -n "$MAX_SAMPLES_PER_DOMAIN" ]] && train_args+=(--max_samples_per_domain "$MAX_SAMPLES_PER_DOMAIN")
if [[ ${#DOMAINS[@]} -gt 0 ]]; then
    train_args+=(--domains "${DOMAINS[@]}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    train_args+=("${EXTRA_ARGS[@]}")
fi

echo
if [[ -n "$USE_DEEPSPEED" ]]; then
    # Resolve the accelerate / deepspeed config:
    #   1. Explicit ACCELERATE_CONFIG_FILE env-var wins.
    #   2. Otherwise use the bundled ZeRO-3 yaml in ./configs/.
    if [[ -n "${ACCELERATE_CONFIG_FILE:-}" ]]; then
        accel_cfg="$ACCELERATE_CONFIG_FILE"
    else
        accel_cfg="${SCRIPT_DIR}/configs/ds_zero3.yaml"
    fi
    if [[ ! -f "$accel_cfg" ]]; then
        echo "ERROR: accelerate config not found: $accel_cfg" >&2
        exit 2
    fi

    accel_launch_args=(--config_file "$accel_cfg")
    # If NUM_GPUS is set, override the config's num_processes.  When
    # CUDA_VISIBLE_DEVICES is also set, accelerate will only see that
    # many GPUs anyway, but passing --num_processes makes it explicit.
    if [[ -n "$NUM_GPUS" ]]; then
        accel_launch_args+=(--num_processes "$NUM_GPUS")
    fi

    echo "[Phase 3] launching via accelerate (config=${accel_cfg}, num_processes=${NUM_GPUS:-<from-config>}):"
    echo "         python -m trainer.SFT.schema_gen.train ${train_args[*]}"
    exec accelerate launch "${accel_launch_args[@]}" \
        -m trainer.SFT.schema_gen.train "${train_args[@]}"
fi

echo "[Phase 3] launching: python -m trainer.SFT.schema_gen.train ${train_args[*]}"
exec python -m trainer.SFT.schema_gen.train "${train_args[@]}"
