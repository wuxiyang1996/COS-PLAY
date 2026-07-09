#!/usr/bin/env bash
# Run GPT-5.4 evaluation for StarCraft II via Orak.
#
# Sets up the orak-sc2 conda environment, configures PYTHONPATH and SC2PATH,
# then launches test_orak_mario_sc2_gpt54.py with --game star_craft.
#
# Usage:
#   bash evaluate_orak/run_gpt54_sc2.sh                      # 3 episodes, defaults
#   bash evaluate_orak/run_gpt54_sc2.sh --episodes 5         # override episodes
#   bash evaluate_orak/run_gpt54_sc2.sh --max_steps 500      # override max steps
#
# All extra arguments are forwarded to the Python script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(dirname "$SCRIPT_DIR")"
ORAK_SRC="${CODEBASE_ROOT}/../Orak/src"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate orak-sc2

export SC2PATH="/workspace/game_agent/StarCraftII/StarCraftII"
export PYTHONPATH="${CODEBASE_ROOT}:${ORAK_SRC}:${PYTHONPATH:-}"

echo "=== GPT-5.4 Orak StarCraft II Evaluation ==="
echo "  Python:     $(python --version)"
echo "  SC2PATH:    ${SC2PATH}"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo ""
echo "  Compatible maps (SC2 4.10 Linux):"
ls -1 "${SC2PATH}/Maps/Ladder2019Season1/" 2>/dev/null | sed 's/\.SC2Map//' | sed 's/^/    /' || echo "    (maps directory not found)"
echo ""

python "${SCRIPT_DIR}/test_orak_mario_sc2_gpt54.py" \
    --game star_craft \
    "$@"
