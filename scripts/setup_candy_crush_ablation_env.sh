#!/usr/bin/env bash
# ======================================================================
#  Create/check a conda env for Candy Crush contract ablation training.
#
#  Why this exists:
#    - Existing cluster envs are split: `vlm_benchmarks` can reset Candy
#      Crush but has no vLLM/PEFT; `swift` has vLLM/PEFT but lacks the
#      GamingAgent/Candy Crush runtime deps.
#    - We avoid `pip install tile_match_gym` because it can downgrade numpy
#      through numba. Instead the run script exposes GamingAgent's vendored
#      Candy Crush backend on PYTHONPATH.
#
#  Usage:
#    bash scripts/setup_candy_crush_ablation_env.sh
#    conda activate cosplay-candy-a100
#    bash scripts/run_candy_crush_contract_ablation.sh
#
#    # Check an existing env without installing anything:
#    CHECK_ONLY=1 ENV_NAME=vlm_benchmarks bash scripts/setup_candy_crush_ablation_env.sh
#
#  Overrides:
#    ENV_NAME=my-env CONDA=/path/to/conda bash scripts/setup_candy_crush_ablation_env.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

ENV_NAME="${ENV_NAME:-cosplay-candy-a100}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA="${CONDA:-/fs/gamma-projects/vlm-robot/conda/bin/conda}"
CHECK_ONLY="${CHECK_ONLY:-0}"

if [[ ! -x "${CONDA}" ]]; then
    echo "ERROR: conda not found/executable: ${CONDA}" >&2
    exit 1
fi

CONDA_ROOT="$(dirname "$(dirname "${CONDA}")")"
PYTHON="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_ROOT}/envs/${ENV_NAME}/bin/pip"

echo "=============================================================="
echo "  Candy Crush Ablation Conda Env"
echo "=============================================================="
echo "  conda:      ${CONDA}"
echo "  env:        ${ENV_NAME}"
echo "  python:     ${PYTHON_VERSION}"
echo "  repo:       ${PROJECT_ROOT}"
echo "  workspace:  ${WORKSPACE_ROOT}"
echo "  check only: ${CHECK_ONLY}"
echo "=============================================================="

env_exists() {
    "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
}

smoke_check() {
    if [[ ! -x "${PYTHON}" ]]; then
        echo "ERROR: env python not found: ${PYTHON}" >&2
        return 1
    fi

    echo "[setup] smoke checking imports"
    PYTHONPATH="${PROJECT_ROOT}:${WORKSPACE_ROOT}/GamingAgent:${WORKSPACE_ROOT}/GamingAgent/gamingagent/envs/custom_03_candy_crush:${WORKSPACE_ROOT}/AgentEvolver:${WORKSPACE_ROOT}/AI_Diplomacy:${WORKSPACE_ROOT}/Orak:${PYTHONPATH:-}" \
    "${PYTHON}" - <<'PY'
import importlib
import sys

print("python", sys.version)
missing = []
for name in [
    "torch",
    "vllm",
    "transformers",
    "peft",
    "accelerate",
    "gymnasium",
    "diplomacy",
    "wandb",
    "tile_match_gym",
]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        missing.append((name, repr(exc)))
        print(name, "MISSING", repr(exc))
    else:
        print(name, getattr(mod, "__version__", "imported"))

if missing:
    raise SystemExit(10)

from env_wrappers.gym_like import make_gaming_env

env = make_gaming_env("candy_crush", max_steps=2)
obs, info = env.reset()
print("candy_crush_reset", type(obs).__name__)
if hasattr(env, "close"):
    env.close()
PY
}

if [[ "${CHECK_ONLY}" == "1" ]]; then
    if ! env_exists; then
        echo "ERROR: env does not exist: ${ENV_NAME}" >&2
        exit 1
    fi
    smoke_check
    exit $?
fi

if ! env_exists; then
    "${CONDA}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
    echo "[setup] env exists: ${ENV_NAME}"
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "[setup] installing COS-PLAY training requirements"
"${PIP}" install -r "${PROJECT_ROOT}/install/requirements.txt"

echo "[setup] installing optional runtime tools"
"${PIP}" install wandb

echo "[setup] installing COS-PLAY editable"
"${PIP}" install -e "${PROJECT_ROOT}" --no-deps

if [[ -d "${WORKSPACE_ROOT}/GamingAgent" ]]; then
    echo "[setup] installing GamingAgent editable without deps"
    "${PIP}" install -e "${WORKSPACE_ROOT}/GamingAgent" --no-deps
else
    echo "WARNING: GamingAgent not found at ${WORKSPACE_ROOT}/GamingAgent" >&2
fi

if [[ ! -d "${WORKSPACE_ROOT}/AgentEvolver" ]]; then
    echo "WARNING: AgentEvolver not found at ${WORKSPACE_ROOT}/AgentEvolver" >&2
fi

smoke_check

echo ""
echo "[setup] done"
echo "Run:"
echo "  ${CONDA} activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
echo "  bash scripts/run_candy_crush_contract_ablation.sh"
