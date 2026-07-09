#!/usr/bin/env bash
# Create and configure the orak-mario conda environment.
#
# Usage:
#   bash evaluate_orak/envs/orak-mario/install.sh
#
# After install, activate with:
#   source evaluate_orak/setup_orak_mario.sh

set -euo pipefail

ENV_NAME="orak-mario"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="${SCRIPT_DIR}/requirements.txt"

echo "=== Installing ${ENV_NAME} environment ==="

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda env '${ENV_NAME}' with Python 3.11..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

CONDA_PREFIX="$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')"
PIP="${CONDA_PREFIX}/bin/pip"

echo "Installing pip packages from ${REQS}..."
"${PIP}" install -r "${REQS}"

echo ""
echo "=== ${ENV_NAME} installed ==="
echo "  Python: $("${CONDA_PREFIX}/bin/python" --version)"
echo "  NumPy:  $("${CONDA_PREFIX}/bin/python" -c 'import numpy; print(numpy.__version__)')"
echo "  nes-py: OK"
echo ""
echo "Activate with:"
echo "  source evaluate_orak/setup_orak_mario.sh"
