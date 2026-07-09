#!/usr/bin/env bash
# Create and configure the orak-sc2 conda environment.
#
# This script:
#   1. Creates the conda env and installs Python packages
#   2. Downloads the SC2 4.10 headless Linux client (~3.9GB)
#   3. Extracts it and sets up the Maps symlink
#
# Usage:
#   bash evaluate_orak/envs/orak-sc2/install.sh
#
# After install, activate with:
#   source evaluate_orak/setup_orak_sc2.sh

set -euo pipefail

ENV_NAME="orak-sc2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="${SCRIPT_DIR}/requirements.txt"
SC2_DIR="/workspace/game_agent/StarCraftII"
SC2_URL="https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
SC2_ZIP_PASS="iagreetotheeula"

echo "=== Installing ${ENV_NAME} environment ==="

# ── 1. Conda env ────────────────────────────────────────────────────────
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

# ── 2. SC2 headless Linux client ────────────────────────────────────────
if [ -d "${SC2_DIR}/StarCraftII/Versions" ]; then
    echo "SC2 client already installed at ${SC2_DIR}/StarCraftII, skipping download."
else
    echo ""
    echo "Downloading SC2 4.10 headless Linux client (~3.9GB)..."
    echo "By proceeding you agree to the Blizzard AI and Machine Learning License."
    mkdir -p "${SC2_DIR}"
    wget -q --show-progress "${SC2_URL}" -O "${SC2_DIR}/SC2.4.10.zip"

    echo "Extracting (password: ${SC2_ZIP_PASS})..."
    cd "${SC2_DIR}"
    unzip -P "${SC2_ZIP_PASS}" -q SC2.4.10.zip
    rm SC2.4.10.zip
    echo "SC2 extracted to ${SC2_DIR}/StarCraftII"
fi

# ── 3. Maps symlink (burnysc2 expects lowercase 'maps') ────────────────
if [ ! -e "${SC2_DIR}/StarCraftII/maps" ]; then
    ln -s Maps "${SC2_DIR}/StarCraftII/maps"
    echo "Created lowercase 'maps' symlink."
fi

# ── 4. Verify ───────────────────────────────────────────────────────────
echo ""
echo "=== ${ENV_NAME} installed ==="
echo "  Python:   $("${CONDA_PREFIX}/bin/python" --version)"
echo "  SC2PATH:  ${SC2_DIR}/StarCraftII"
echo "  SC2 ver:  $(ls -1 "${SC2_DIR}/StarCraftII/Versions/" | grep Base)"
echo "  Maps:     $(ls -1 "${SC2_DIR}/StarCraftII/Maps/Ladder2019Season1/" | wc -l) maps in Ladder2019Season1"
echo ""
echo "Activate with:"
echo "  source evaluate_orak/setup_orak_sc2.sh"
