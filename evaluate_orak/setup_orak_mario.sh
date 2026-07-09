#!/usr/bin/env bash
# Setup script for the orak-mario conda environment.
#
# Activates the environment and sets PYTHONPATH so that both
# Game-AI-Agent modules and Orak src are importable.
#
# Usage:
#   source evaluate_orak/setup_orak_mario.sh
#   python evaluate_orak/run_orak_benchmark.py --game super_mario --episodes 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(dirname "$SCRIPT_DIR")"
ORAK_SRC="${CODEBASE_ROOT}/../Orak/src"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate orak-mario

export PYTHONPATH="${CODEBASE_ROOT}:${ORAK_SRC}:${PYTHONPATH:-}"
export SDL_VIDEODRIVER=dummy

# nes_py/pyglet needs a display; start Xvfb if none exists
if [ -z "${DISPLAY:-}" ]; then
    if ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
        sleep 0.5
    fi
    export DISPLAY=:99
fi

echo "=== orak-mario environment ==="
echo "  Python:     $(python --version)"
echo "  NumPy:      $(python -c 'import numpy; print(numpy.__version__)')"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo "  Env ready for: Super Mario (Orak)"
echo ""
echo "  Quick test:"
echo "    python -c \"from evaluate_orak.orak_nl_wrapper import make_orak_env; e=make_orak_env('super_mario'); print(e.reset()[0][:200]); e.close()\""
