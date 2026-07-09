#!/usr/bin/env bash
# ==============================================================================
# install_gymv.sh — sets up the `gymv` conda env with ModalMinds/gym-v.
#
# Usage:   bash install/install_gymv.sh [GYMV_CLONE_DIR]
# Default GYMV_CLONE_DIR: /fs/gamma-projects/vlm-robot/gym-v
# ==============================================================================
set -euo pipefail

ENV_NAME="gymv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="${SCRIPT_DIR}/gymv.environment.yml"
GYMV_DIR="${1:-/fs/gamma-projects/vlm-robot/gym-v}"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/4] Creating env '$ENV_NAME' ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      Env already exists — skipping."
else
    conda env create -f "$YML"
fi

conda activate "$ENV_NAME"

echo "[2/4] Cloning gym-v to $GYMV_DIR ..."
if [[ ! -d "$GYMV_DIR" ]]; then
    git clone --depth=1 https://github.com/ModalMinds/gym-v.git "$GYMV_DIR"
else
    echo "      Already cloned."
fi

echo "[3/4] Installing gym-v editable with [games,spatial] extras ..."
pip install -e "${GYMV_DIR}[games,spatial]"

echo "[4/4] Running smoke test ..."
python "$SCRIPT_DIR/gymv_smoke.py"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
