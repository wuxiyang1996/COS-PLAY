#!/usr/bin/env bash
# Cold-start SFT launcher for the 5 actor + skill-bank LoRAs.
#
# Pipeline
# --------
# Phase 1  Build decision-adapter JSONLs from the latest dual-axis
#          skill-actions corpus  (labeling/build_decision_sft_jsonl.py).
#          Skipped with --skip-build.
# Phase 2  Audit-load every adapter through trainer/SFT/data_loader.py
#          to confirm row counts before burning GPU.
# Phase 3  Launch trainer/SFT/train.py (sequential or --parallel).
#
# Usage
# -----
#   # Full pipeline, parallel training across as many GPUs as adapters.
#   bash trainer/SFT/run_sft.sh --parallel
#
#   # Subset (e.g. just skill-bank LoRAs).
#   bash trainer/SFT/run_sft.sh --adapters segment contract curator
#
#   # Skip the decision-data build (keeps the existing JSONLs).
#   bash trainer/SFT/run_sft.sh --skip-build --parallel
#
#   # Audit only — no training.
#   bash trainer/SFT/run_sft.sh --dry-run
#
#   # Pin a specific input run (else the latest run_* under
#   # labeling/skill_actions_out and labeling/skill_bank_out is auto-picked).
#   bash trainer/SFT/run_sft.sh \
#        --skill-actions-run labeling/skill_actions_out/run_<ts> \
#        --skill-bank-run    labeling/skill_bank_out/run_<ts>

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SKIP_BUILD=""
DRY_RUN=""
PARALLEL_FLAG=""
ADAPTERS=()
GPUS=()
EXTRA_ARGS=()

SKILL_ACTIONS_RUN=""   # auto-detect (latest)
SKILL_BANK_RUN=""      # auto-detect (latest)
DECISION_DATA_DIR=""   # auto-detect (newest decision_sft_jsonl run after Phase 1)

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build|--skip_build)        SKIP_BUILD="1"; shift ;;
        --dry-run|--dry_run)              DRY_RUN="1"; shift ;;
        --parallel)                       PARALLEL_FLAG="--parallel"; shift ;;
        --skill-actions-run|--skill_actions_run) SKILL_ACTIONS_RUN="$2"; shift 2 ;;
        --skill-bank-run|--skill_bank_run)       SKILL_BANK_RUN="$2"; shift 2 ;;
        --decision-data-dir|--decision_data_dir) DECISION_DATA_DIR="$2"; shift 2 ;;
        --adapters)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ADAPTERS+=("$1"); shift
            done
            ;;
        --gpus)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                GPUS+=("$1"); shift
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
echo "  SFT cold-start launcher"
echo "  repo root         : $REPO_ROOT"
echo "  skip_build        : ${SKIP_BUILD:-0}"
echo "  dry_run           : ${DRY_RUN:-0}"
echo "  parallel          : ${PARALLEL_FLAG:-no}"
echo "  adapters          : ${ADAPTERS[*]:-<all 5>}"
echo "  gpus              : ${GPUS[*]:-<auto>}"
echo "  skill_actions_run : ${SKILL_ACTIONS_RUN:-<latest>}"
echo "  skill_bank_run    : ${SKILL_BANK_RUN:-<latest>}"
echo "  started           : $(date -Iseconds)"
echo "=================================================================="

# ---------------------------------------------------------------------------
# Phase 1 — build decision-adapter JSONLs
# ---------------------------------------------------------------------------
if [[ -z "$SKIP_BUILD" ]]; then
    echo
    echo "[Phase 1] building decision-adapter JSONLs ..."
    build_args=(--output-dir "")
    if [[ -n "$SKILL_ACTIONS_RUN" ]]; then
        build_args=(--skill-actions-run "$SKILL_ACTIONS_RUN")
    else
        build_args=()
    fi
    python labeling/build_decision_sft_jsonl.py "${build_args[@]}" || {
        echo "[Phase 1] FAILED" >&2
        exit 2
    }
    # The converter writes to labeling/decision_sft_jsonl/run_<ts>; pick
    # the latest one for downstream use unless the user pinned it.
    if [[ -z "$DECISION_DATA_DIR" ]]; then
        DECISION_DATA_DIR="$(ls -d labeling/decision_sft_jsonl/run_* 2>/dev/null | sort | tail -n 1)"
    fi
else
    echo "[Phase 1] skipped (--skip-build)"
    if [[ -z "$DECISION_DATA_DIR" ]]; then
        DECISION_DATA_DIR="$(ls -d labeling/decision_sft_jsonl/run_* 2>/dev/null | sort | tail -n 1)"
    fi
fi

if [[ -z "$SKILL_BANK_RUN" ]]; then
    SKILL_BANK_RUN="$(ls -d labeling/skill_bank_out/run_* 2>/dev/null | sort | tail -n 1)"
fi

echo
echo "[paths] decision_data_dir = $DECISION_DATA_DIR"
echo "[paths] skillbank_data_dir = $SKILL_BANK_RUN"
[[ -d "$DECISION_DATA_DIR" ]] || {
    echo "ERROR: decision_data_dir does not exist: $DECISION_DATA_DIR" >&2
    exit 2
}
[[ -d "$SKILL_BANK_RUN" ]] || {
    echo "ERROR: skill_bank_run does not exist: $SKILL_BANK_RUN" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# Phase 2 — audit-load every adapter
# ---------------------------------------------------------------------------
echo
echo "[Phase 2] audit-load every adapter through data_loader.py ..."
python - <<PY
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
from trainer.SFT.config    import SFTConfig
from trainer.SFT.data_loader import load_adapter_dataset

cfg = SFTConfig(
    decision_data_dir="${DECISION_DATA_DIR}",
    skillbank_data_dir="${SKILL_BANK_RUN}",
)
total = 0
print(f"decision_data_dir : {cfg.decision_data_dir}")
print(f"skillbank_data_dir: {cfg.skillbank_data_dir}")
print(f"games             : {len(cfg.games)}")
for ad in ('action_taking','skill_selection','segment','contract','curator'):
    rows = load_adapter_dataset(ad, cfg)
    print(f"  {ad:18} {len(rows):>7,d} examples")
    total += len(rows)
print(f"  TOTAL: {total:,d}")
PY

if [[ -n "$DRY_RUN" ]]; then
    echo
    echo "[dry-run] Phase 3 (training) skipped."
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 3 — launch trainer
# ---------------------------------------------------------------------------
train_args=(
    --decision_data_dir "$DECISION_DATA_DIR"
    --skillbank_data_dir "$SKILL_BANK_RUN"
)
if [[ -n "$PARALLEL_FLAG" ]]; then
    train_args+=("$PARALLEL_FLAG")
fi
if [[ ${#ADAPTERS[@]} -gt 0 ]]; then
    train_args+=(--adapters "${ADAPTERS[@]}")
fi
if [[ ${#GPUS[@]} -gt 0 ]]; then
    train_args+=(--gpus "${GPUS[@]}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    train_args+=("${EXTRA_ARGS[@]}")
fi

echo
echo "[Phase 3] launching: python -m trainer.SFT.train ${train_args[*]}"
exec python -m trainer.SFT.train "${train_args[@]}"
