#!/usr/bin/env bash
#
# install_alfworld.sh — create the ALFWorld conda env, install the package,
# download game/PDDL data, and write a sourceable cold_start/alfworld_env.sh.
#
# Usage:
#   bash install/install_alfworld.sh
#   ALFWORLD_EXTRAS=full bash install/install_alfworld.sh
#
# Env vars:
#   ALFWORLD_ENV_NAME   conda env name (default: alfworld)
#   ALFWORLD_DATA       data/cache directory (default: /workspace/alfworld_data)
#   ALFWORLD_EXTRAS     text | vis | full (default: text)
#   ALFWORLD_VERSION    PyPI version (default: 0.4.2)
#   ALFWORLD_NO_DOWNLOAD=1  skip alfworld-download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YML="${SCRIPT_DIR}/alfworld.environment.yml"

ENV_NAME="${ALFWORLD_ENV_NAME:-alfworld}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/workspace/alfworld_data}"
ALFWORLD_EXTRAS="${ALFWORLD_EXTRAS:-text}"
ALFWORLD_VERSION="${ALFWORLD_VERSION:-0.4.2}"
ENV_FILE="${CODEBASE_ROOT}/cold_start/alfworld_env.sh"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/5] Creating env '${ENV_NAME}' ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      Env already exists — skipping."
else
    conda env create -n "$ENV_NAME" -f "$YML"
fi

conda activate "$ENV_NAME"

echo "[2/5] Installing ALFWorld (${ALFWORLD_EXTRAS}) ..."
case "$ALFWORLD_EXTRAS" in
    text)
        pip install "alfworld==${ALFWORLD_VERSION}"
        ;;
    vis|full)
        pip install "alfworld[${ALFWORLD_EXTRAS}]==${ALFWORLD_VERSION}"
        ;;
    *)
        echo "ERROR: ALFWORLD_EXTRAS must be one of: text, vis, full"
        exit 1
        ;;
esac

echo "[3/5] Downloading ALFWorld data to ${ALFWORLD_DATA} ..."
mkdir -p "$ALFWORLD_DATA"
export ALFWORLD_DATA
if [[ "${ALFWORLD_NO_DOWNLOAD:-0}" == "1" ]]; then
    echo "      ALFWORLD_NO_DOWNLOAD=1 — skipping."
else
    alfworld-download
fi

echo "[4/5] Writing ${ENV_FILE} ..."
cat > "$ENV_FILE" <<EOF
# Source this before running ALFWorld jobs from COS-PLAY.
export ALFWORLD_DATA="${ALFWORLD_DATA}"
export ALFWORLD_ENV_NAME="${ENV_NAME}"
export PYTHONPATH="${CODEBASE_ROOT}:\${PYTHONPATH:-}"
EOF

echo "[5/5] Running smoke test ..."
python "$SCRIPT_DIR/alfworld_smoke.py"

echo
echo "Done. Activate with: conda activate ${ENV_NAME}"
echo "Source env vars with: source cold_start/alfworld_env.sh"
