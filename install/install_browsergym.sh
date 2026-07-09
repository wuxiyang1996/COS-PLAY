#!/usr/bin/env bash
# ==============================================================================
# install_browsergym.sh — sets up the `browsergym` env with ServiceNow/BrowserGym.
#
# Usage:   bash install/install_browsergym.sh [BROWSERGYM_CLONE_DIR]
# Default BROWSERGYM_CLONE_DIR: /fs/gamma-projects/vlm-robot/BrowserGym
# ==============================================================================
set -euo pipefail

ENV_NAME="browsergym"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YML="${SCRIPT_DIR}/browsergym.environment.yml"
BG_DIR="${1:-/fs/gamma-projects/vlm-robot/BrowserGym}"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[1/5] Creating env '$ENV_NAME' ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      Env already exists — skipping."
else
    conda env create -f "$YML"
fi

conda activate "$ENV_NAME"

echo "[2/5] Cloning BrowserGym to $BG_DIR ..."
if [[ ! -d "$BG_DIR" ]]; then
    git clone --depth=1 https://github.com/ServiceNow/BrowserGym.git "$BG_DIR"
else
    echo "      Already cloned."
fi

echo "[3/5] Installing BrowserGym sub-packages editable ..."
pip install -e "${BG_DIR}/browsergym/core" \
            -e "${BG_DIR}/browsergym/miniwob" \
            -e "${BG_DIR}/browsergym/webarena" \
            -e "${BG_DIR}/browsergym/visualwebarena" \
            -e "${BG_DIR}/browsergym/assistantbench" \
            -e "${BG_DIR}/browsergym/experiments"

echo "[4/5] Installing Playwright system deps + chromium ..."
# `install-deps` apt-installs libnspr4/libnss3/libatk*/libcups2/etc. on Linux.
# On hosts without sudo, this is a no-op + warning; the next line will still
# fetch the Chromium binary, but headless launches will fail until the libs
# are installed manually (see install/INSTALL_BENCHMARKS.md §Troubleshooting).
python -m playwright install-deps chromium || {
    echo "      ⚠ playwright install-deps failed (no sudo?). Install libs manually if Chromium fails to launch."
}
python -m playwright install chromium

echo "[5/5] Running smoke test ..."
python "$SCRIPT_DIR/browsergym_smoke.py"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "WorkArena (optional):  pip install --no-deps browsergym-workarena"
echo "Task counts (expected ~2,063 total):"
echo "  conda run -n $ENV_NAME python -c \"import browsergym.miniwob, browsergym.webarena, browsergym.visualwebarena, browsergym.assistantbench; import gymnasium; print(sum(1 for k in gymnasium.envs.registry if k.startswith('browsergym/')))\""
