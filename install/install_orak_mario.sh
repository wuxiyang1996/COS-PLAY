#!/usr/bin/env bash
# =============================================================================
# install_orak_mario.sh
#
# Creates the "orak-mario" conda environment for Super Mario Bros evaluation.
#
# This is a SEPARATE environment because nes-py requires numpy<2 and
# gym==0.26.2, which conflict with the main game-ai-agent environment.
#
# Prerequisites:
#   - Miniconda3 or Anaconda installed
#   - Orak repo cloned as a sibling of Multi-hop-Reasoning-VLM-Agent:
#       git clone https://github.com/krafton-ai/Orak.git <parent>/Orak
#     (the older nicholascpark/orak URL is private/unreachable)
#   - For headless servers: Xvfb (apt install xvfb)
#
# Usage:
#   bash Game-AI-Agent/install/install_orak_mario.sh [CONDA_PATH]
#
# After install:
#   source Game-AI-Agent/env_wrappers/setup_orak_mario.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="orak-mario"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"          # Game-AI-Agent/
PARENT_DIR="$(dirname "$REPO_DIR")"          # parent of all sibling repos
REQS="${SCRIPT_DIR}/requirements-orak-mario.txt"

# ---------------------------------------------------------------------------
# Locate conda
# ---------------------------------------------------------------------------
if [[ -n "${1:-}" ]]; then
    CONDA="$1"
elif command -v conda &>/dev/null; then
    CONDA="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA="$HOME/miniconda3/bin/conda"
elif [[ -x "/workspace/miniconda3/bin/conda" ]]; then
    CONDA="/workspace/miniconda3/bin/conda"
else
    echo "ERROR: conda not found. Pass the conda path as an argument or install Miniconda first."
    echo "  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash"
    exit 1
fi

CONDA_DIR="$(dirname "$(dirname "$CONDA")")"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  COS-PLAY — orak-mario Environment Installer"
echo "============================================================"
echo "  conda:       $CONDA"
echo "  env name:    $ENV_NAME"
echo "  python:      $PYTHON_VERSION"
echo "  repo dir:    $REPO_DIR"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Step 0: System libraries (libGLU + Xvfb for pyglet/nes-py rendering)
# ---------------------------------------------------------------------------
# pyglet (used by nes-py) dlopen()s libGLU.so / libGL.so at runtime; without
# them every super_mario env.reset() fails with
#     ImportError: Library "GLU" not found.
# We try a best-effort apt install; non-Debian distros / non-root users
# should install equivalents manually (see README troubleshooting).
if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ] || command -v sudo >/dev/null 2>&1; then
        SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo "
        echo "[0/4] Ensuring system GL/GLU libs + Xvfb are installed ..."
        ${SUDO}apt-get update -qq >/dev/null 2>&1 || true
        ${SUDO}apt-get install -y -qq --no-install-recommends \
            libglu1-mesa libgl1-mesa-glx libosmesa6 freeglut3-dev xvfb \
            >/dev/null 2>&1 || \
            echo "  [WARN] could not auto-install libGLU; install manually:" \
                 " apt install -y libglu1-mesa libgl1-mesa-glx xvfb"
        echo
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
if "$CONDA" env list | grep -q "^${ENV_NAME} "; then
    echo "[1/4] Conda env '$ENV_NAME' already exists — skipping creation."
else
    echo "[1/4] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    "$CONDA" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
echo

# ---------------------------------------------------------------------------
# Step 2: Install PyTorch with CUDA (for torchvision / object detection)
# ---------------------------------------------------------------------------
echo "[2/4] Installing PyTorch + torchvision with CUDA 12.x ..."
"$PIP" install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo

# ---------------------------------------------------------------------------
# Step 3: Install pip requirements
# ---------------------------------------------------------------------------
echo "[3/4] Installing pip requirements from $REQS ..."
"$PIP" install --quiet -r "$REQS"
echo

# ---------------------------------------------------------------------------
# Step 4: Verify installation
# ---------------------------------------------------------------------------
echo "[4/4] Verifying installation ..."
echo

ORAK_SRC="${PARENT_DIR}/Orak/src"
PYTHONPATH="${REPO_DIR}:${ORAK_SRC}:${PYTHONPATH:-}" \
"$PYTHON" -c "
import sys

failures = []
warnings = []

def check(label, fn, required=True):
    try:
        fn()
        print(f'  [OK]   {label}')
    except Exception as e:
        if required:
            failures.append((label, str(e)))
            print(f'  [FAIL] {label}: {e}')
        else:
            warnings.append((label, str(e)))
            print(f'  [WARN] {label}: {e}  (optional)')

print(f'Python {sys.version}')
print()

# --- Core game ---
print('Core Game:')
check('gym',                   lambda: __import__('gym'))
check('nes_py',                lambda: __import__('nes_py'))
check('gym_super_mario_bros',  lambda: __import__('gym_super_mario_bros'))
import numpy; v = tuple(int(x) for x in numpy.__version__.split('.')[:2])
print(f'  numpy version: {numpy.__version__} (must be <2.0)')
assert v[0] < 2, f'numpy {numpy.__version__} >= 2.0 — nes-py will fail'
print()

# --- Vision / ML ---
print('Vision / ML:')
check('torch',                 lambda: __import__('torch'))
check('torchvision',           lambda: __import__('torchvision'))
check('cv2 (opencv)',          lambda: __import__('cv2'))
check('skimage (scikit-image)',lambda: __import__('skimage'))
check('PIL (Pillow)',          lambda: __import__('PIL'))
print()

# --- Orak framework ---
print('Orak Framework:')
check('omegaconf',             lambda: __import__('omegaconf'))
check('gymnasium',             lambda: __import__('gymnasium'))
check('dacite',                lambda: __import__('dacite'))
check('dataclass_wizard',      lambda: __import__('dataclass_wizard'))
check('dill',                  lambda: __import__('dill'))
check('tenacity',              lambda: __import__('tenacity'))
check('pygame',                lambda: __import__('pygame'))
check('yaml (pyyaml)',         lambda: __import__('yaml'))
check('requests',              lambda: __import__('requests'))
print()

# --- API client ---
print('API Client:')
check('openai',                lambda: __import__('openai'))
print()

# --- Internal modules ---
print('Internal Modules:')
check('API_func',              lambda: __import__('API_func'), required=False)
check('env_wrappers.orak_nl_wrapper', lambda: __import__('env_wrappers.orak_nl_wrapper'), required=False)
print()

# --- Summary ---
print('=' * 50)
if failures:
    print(f'{len(failures)} check(s) FAILED:')
    for label, err in failures:
        print(f'  ✗ {label}: {err}')
    sys.exit(1)
else:
    print('All required checks passed.')
if warnings:
    print(f'{len(warnings)} optional check(s) skipped:')
    for label, err in warnings:
        print(f'  ⚠ {label}')
print('=' * 50)
"

echo
echo "============================================================"
echo "  orak-mario installation complete!"
echo "============================================================"
echo
echo "  Activate (recommended — also sets PYTHONPATH + DISPLAY):"
echo "    source $REPO_DIR/env_wrappers/setup_orak_mario.sh"
echo
echo "  Or activate manually:"
echo "    conda activate $ENV_NAME"
echo "    export PYTHONPATH=$REPO_DIR:\$PYTHONPATH"
echo
echo "  Quick test:"
echo "    python -c \"import gym_super_mario_bros; print('Mario env OK')\""
echo
echo "  Headless server (no display):"
echo "    If you see pyglet/display errors, install Xvfb:"
echo "      sudo apt install -y xvfb"
echo "    The setup script starts Xvfb automatically."
echo
echo "  Run baselines:"
echo "    bash baselines/run_super_mario_baseline.sh --model gpt-5.4"
echo
echo "  Run ablation:"
echo "    bash ablation_study/run_super_mario_ablation.sh --adapter base"
echo
echo "============================================================"
