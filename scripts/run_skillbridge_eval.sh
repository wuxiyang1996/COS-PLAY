#!/usr/bin/env bash
# scripts/run_skillbridge_eval.sh — single entry point that runs all five
# SkillBridge cross-domain eval drivers (block C7) and aggregates results.
#
# Usage:
#   bash scripts/run_skillbridge_eval.sh \
#       --run-dir runs/skillbridge_v12 \
#       --vllm-base-url http://localhost:8000/v1 \
#       --model Qwen/Qwen3.5-9B \
#       --label skillbridge_full
#
# Optional knobs (pass after the four required flags):
#   --episodes-per-task 1
#   --max-steps 50
#   --gymv-games crafter,procgen
#   --vr-num-cases 200
#   --skip browsergym,osworld    # comma-separated list of domains to skip
#   --judge                      # enable LLM-as-judge for VR/video
#   --extra-cold-start-args "..." # forward to all cold-start subcalls
set -euo pipefail

RUN_DIR=""
MODEL="Qwen/Qwen3.5-9B"
VLLM_BASE_URL="http://localhost:8000/v1"
LABEL="skillbridge"
EPISODES_PER_TASK=1
MAX_STEPS=50
GYMV_GAMES=""
VR_NUM_CASES=200
SKIP=""
JUDGE_FLAG=""
EXTRA_COLD_START_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)              RUN_DIR="$2"; shift 2;;
        --model)                MODEL="$2"; shift 2;;
        --vllm-base-url)        VLLM_BASE_URL="$2"; shift 2;;
        --label)                LABEL="$2"; shift 2;;
        --episodes-per-task)    EPISODES_PER_TASK="$2"; shift 2;;
        --max-steps)            MAX_STEPS="$2"; shift 2;;
        --gymv-games)           GYMV_GAMES="$2"; shift 2;;
        --vr-num-cases)         VR_NUM_CASES="$2"; shift 2;;
        --skip)                 SKIP="$2"; shift 2;;
        --judge)                JUDGE_FLAG="--judge"; shift;;
        --extra-cold-start-args) EXTRA_COLD_START_ARGS="$2"; shift 2;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0;;
        *)
            echo "[run_skillbridge_eval.sh] unknown flag: $1" >&2; exit 2;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "[run_skillbridge_eval.sh] --run-dir is required" >&2
    exit 2
fi

mkdir -p "$RUN_DIR/eval"

skip_domain() {
    local d="$1"
    [[ ",${SKIP}," == *",${d},"* ]]
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA_FLAGS_ARRAY=()
if [[ -n "$EXTRA_COLD_START_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_FLAGS_ARRAY=(--cold-start-extra ${EXTRA_COLD_START_ARGS})
fi

run_browsergym() {
    if skip_domain browsergym; then return 0; fi
    echo "==[ browsergym ]==============================================="
    python -m scripts.skillbridge_eval.eval_browsergym \
        --run-dir "$RUN_DIR" \
        --model "$MODEL" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --label "$LABEL" \
        --episodes-per-task "$EPISODES_PER_TASK" \
        --max-steps "$MAX_STEPS" \
        "${EXTRA_FLAGS_ARRAY[@]}"
}

run_osworld() {
    if skip_domain osworld; then return 0; fi
    echo "==[ osworld   ]==============================================="
    python -m scripts.skillbridge_eval.eval_osworld \
        --run-dir "$RUN_DIR" \
        --model "$MODEL" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --label "$LABEL" \
        --episodes-per-task "$EPISODES_PER_TASK" \
        --max-steps "$MAX_STEPS" \
        "${EXTRA_FLAGS_ARRAY[@]}"
}

run_visual_reasoning() {
    if skip_domain visual_reasoning; then return 0; fi
    echo "==[ visual_reasoning ]========================================="
    python -m scripts.skillbridge_eval.eval_visual_reasoning \
        --run-dir "$RUN_DIR" \
        --model "$MODEL" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --label "$LABEL" \
        --num-test-cases "$VR_NUM_CASES" \
        $JUDGE_FLAG \
        "${EXTRA_FLAGS_ARRAY[@]}"
}

run_video() {
    if skip_domain video; then return 0; fi
    echo "==[ video    ]================================================="
    python -m scripts.skillbridge_eval.eval_video \
        --run-dir "$RUN_DIR" \
        --model "$MODEL" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --label "$LABEL" \
        --num-test-cases "$VR_NUM_CASES" \
        "${EXTRA_FLAGS_ARRAY[@]}"
}

run_gymv() {
    if skip_domain gymv; then return 0; fi
    echo "==[ gymv     ]================================================="
    local games_arg=""
    if [[ -n "$GYMV_GAMES" ]]; then
        games_arg="--games ${GYMV_GAMES//,/ }"
    fi
    python -m scripts.skillbridge_eval.eval_gymv \
        --run-dir "$RUN_DIR" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --episodes-per-game "$EPISODES_PER_TASK" \
        --max-steps "$MAX_STEPS" \
        $games_arg
}

run_browsergym       || echo "[run_skillbridge_eval] browsergym FAILED"
run_osworld          || echo "[run_skillbridge_eval] osworld FAILED"
run_visual_reasoning || echo "[run_skillbridge_eval] visual_reasoning FAILED"
run_video            || echo "[run_skillbridge_eval] video FAILED"
run_gymv             || echo "[run_skillbridge_eval] gymv FAILED"

echo "==[ aggregate ]================================================="
python -m scripts.skillbridge_eval.eval_aggregator \
    --run-dir "$RUN_DIR"

echo "[run_skillbridge_eval] done — results under $RUN_DIR/eval/"
