#!/usr/bin/env bash
# ==============================================================================
# install_osworld.sh — sets up the `osworld` env with xlang-ai/OSWorld.
#
# Usage:
#   bash install/install_osworld.sh [OSWORLD_CLONE_DIR]
#
# Default OSWORLD_CLONE_DIR: /fs/gamma-projects/vlm-robot/OSWorld
#
# Environment toggles (default = skipped):
#   WITH_DOCKER_IMAGE=1   `docker pull happysixd/osworld-docker` (~360 MB)
#   WITH_VM_DISK=1        download + unzip Ubuntu.qcow2 (~12 GB zip → 23 GB)
#   VM_DATA_DIR           where to put the qcow2; default: $PWD/docker_vm_data
#                         (OSWorld's DockerVMManager looks for
#                          ./docker_vm_data relative to cwd at runtime).
#
# Examples:
#   bash install/install_osworld.sh                                # python only
#   WITH_DOCKER_IMAGE=1 bash install/install_osworld.sh            # python + image
#   WITH_DOCKER_IMAGE=1 WITH_VM_DISK=1 bash install/install_osworld.sh
# ==============================================================================
set -euo pipefail

ENV_NAME="osworld"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YML="${SCRIPT_DIR}/osworld.environment.yml"
OSW_DIR="${1:-/fs/gamma-projects/vlm-robot/OSWorld}"
VM_DATA_DIR="${VM_DATA_DIR:-${REPO_ROOT}/docker_vm_data}"
QCOW2_URL="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"

WITH_DOCKER_IMAGE="${WITH_DOCKER_IMAGE:-0}"
WITH_VM_DISK="${WITH_VM_DISK:-0}"

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

echo "[2/5] Cloning OSWorld to $OSW_DIR ..."
if [[ ! -d "$OSW_DIR" ]]; then
    git clone --depth=1 https://github.com/xlang-ai/OSWorld.git "$OSW_DIR"
else
    echo "      Already cloned."
fi

echo "[3/5] Installing OSWorld requirements + editable package ..."
pip install -r "${OSW_DIR}/requirements.txt"
pip install -e "${OSW_DIR}"

echo "[4/5] VM backend ..."
if [[ "$WITH_DOCKER_IMAGE" == "1" ]]; then
    if command -v docker >/dev/null 2>&1; then
        echo "      Pulling happysixd/osworld-docker (~360 MB) ..."
        docker pull happysixd/osworld-docker
    else
        echo "      WARN: docker CLI not found; skipping image pull."
    fi
else
    echo "      Skipping docker pull (set WITH_DOCKER_IMAGE=1 to enable)."
fi

if [[ "$WITH_VM_DISK" == "1" ]]; then
    mkdir -p "$VM_DATA_DIR"
    if [[ -f "${VM_DATA_DIR}/Ubuntu.qcow2" ]]; then
        echo "      ${VM_DATA_DIR}/Ubuntu.qcow2 already exists — skipping."
    else
        echo "      Downloading Ubuntu.qcow2.zip (~12 GB) to ${VM_DATA_DIR} ..."
        curl -L -C - -o "${VM_DATA_DIR}/Ubuntu.qcow2.zip" "$QCOW2_URL"
        echo "      Unzipping (~23 GB output) ..."
        ( cd "$VM_DATA_DIR" && unzip -o Ubuntu.qcow2.zip && rm -f Ubuntu.qcow2.zip )
    fi
    echo "      VM disk ready: ${VM_DATA_DIR}/Ubuntu.qcow2"
    echo "      NOTE: launch jobs from ${REPO_ROOT} (or any dir whose"
    echo "      ./docker_vm_data points to this qcow2) so OSWorld's"
    echo "      DockerVMManager finds it."
else
    echo "      Skipping qcow2 download (set WITH_VM_DISK=1 to enable)."
    echo "      OSWorld will lazily download it on first reset() if missing."
fi

echo "[5/5] Running smoke test ..."
python "$SCRIPT_DIR/osworld_smoke.py"

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
echo
echo "To actually render a desktop:"
echo "  conda activate $ENV_NAME"
echo "  cd ${REPO_ROOT}"
echo "  python visual_grounding_tests/generate_osworld_text_schema.py \\"
echo "      --task_catalog ${OSW_DIR}/evaluation_examples/test_small.json \\"
echo "      --task_limit 1 --provider docker --max_steps 1 -v"
