#!/usr/bin/env bash
# WebShop ablation entry point via the BrowserGym/WebShop bridge.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANTS="${VARIANTS:-full no_vision}"
NUM_GOALS="${NUM_GOALS:-50}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
RUN_ROOT="${RUN_ROOT:-runs/domain_ablation/webshop}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RUN_ROOT}"

tasks=()
for ((i = 0; i < NUM_GOALS; i++)); do
    tasks+=("browsergym/webshop.${i}")
done

for variant in ${VARIANTS}; do
    out_dir="${RUN_ROOT}/${variant}"
    extra=()
    case "${variant}" in
        full)
            ;;
        no_vision)
            extra=(--cold-start-extra --no_vision)
            ;;
        *)
            echo "Unknown WebShop variant: ${variant}" >&2
            exit 2
            ;;
    esac

    cmd=(
        python -m scripts.skillbridge_eval.eval_browsergym
        --run-dir "${out_dir}"
        --tasks "${tasks[@]}"
        --episodes-per-task "${EPISODES_PER_TASK}"
        --max-steps "${MAX_STEPS}"
        --model "${MODEL}"
        --vllm-base-url "${VLLM_BASE_URL}"
        --label "webshop_${variant}"
        "${extra[@]}"
    )

    echo ""
    echo "[webshop-ablation] variant=${variant}"
    echo "[webshop-ablation] run_dir=${out_dir}"
    printf '[webshop-ablation] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    mkdir -p "${out_dir}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${cmd[@]}"
    fi
done

