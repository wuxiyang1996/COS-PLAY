#!/usr/bin/env bash
# Run GPT-5.4 evaluation for Super Mario via Orak.
#
# Sets up the orak-mario conda environment, configures PYTHONPATH, and
# launches test_orak_mario_sc2_gpt54.py with --game super_mario.
#
# Usage:
#   bash evaluate_orak/run_gpt54_mario.sh                     # 3 episodes, defaults
#   bash evaluate_orak/run_gpt54_mario.sh --episodes 10       # override episodes
#   bash evaluate_orak/run_gpt54_mario.sh --max_steps 200     # override max steps
#
# All extra arguments are forwarded to the Python script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(dirname "$SCRIPT_DIR")"
ORAK_SRC="${CODEBASE_ROOT}/../Orak/src"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate orak-mario

export PYTHONPATH="${CODEBASE_ROOT}:${ORAK_SRC}:${PYTHONPATH:-}"
export SDL_VIDEODRIVER=dummy

# Mario's nes_py/pyglet needs a display; start Xvfb if no DISPLAY is set
if [ -z "${DISPLAY:-}" ]; then
    if ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
        sleep 0.5
    fi
    export DISPLAY=:99
fi

echo "=== GPT-5.4 Orak Super Mario Evaluation ==="
echo "  Python:     $(python --version)"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo ""

python "${SCRIPT_DIR}/test_orak_mario_sc2_gpt54.py" \
    --game super_mario \
    "$@"
