#!/usr/bin/env bash
# ALFWorld text-protocol ablation entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANTS="${VARIANTS:-with_admissible without_admissible}"
EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
SPLIT="${SPLIT:-eval_out_of_distribution}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-runs/domain_ablation/alfworld}"
DRY_RUN="${DRY_RUN:-0}"

CONDA="${CONDA:-/fs/gamma-projects/vlm-robot/conda/bin/conda}"
ALFWORLD_ENV="${ALFWORLD_ENV:-alfworld}"
ALFWORLD_PYTHON="${ALFWORLD_PYTHON:-}"

if [[ -z "${ALFWORLD_PYTHON}" ]]; then
    ALFWORLD_PYTHON="$(dirname "$(dirname "${CONDA}")")/envs/${ALFWORLD_ENV}/bin/python"
fi

if [[ "${DRY_RUN}" != "1" && ! -x "${ALFWORLD_PYTHON}" ]]; then
    echo "ERROR: ALFWorld python not found: ${ALFWORLD_PYTHON}" >&2
    echo "Create it with: bash install/install_alfworld.sh" >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${WORKSPACE_ROOT}:${PYTHONPATH:-}"

for variant in ${VARIANTS}; do
    out="${RUN_ROOT}/${variant}_seed${SEED}.json"
    cmd=(
        "${ALFWORLD_PYTHON}"
        experiments/domain_ablation/run_alfworld_ablation.py
        --variant "${variant}"
        --split "${SPLIT}"
        --episodes "${EPISODES}"
        --max-steps "${MAX_STEPS}"
        --seed "${SEED}"
        --output "${out}"
    )

    echo ""
    echo "[alfworld-ablation] variant=${variant}"
    printf '[alfworld-ablation] command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" != "1" ]]; then
        "${cmd[@]}"
    fi
done
