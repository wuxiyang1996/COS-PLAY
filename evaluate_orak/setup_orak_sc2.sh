#!/usr/bin/env bash
# Setup script for the orak-sc2 conda environment.
#
# Activates the environment, sets SC2PATH, and configures PYTHONPATH
# so that both Game-AI-Agent modules and Orak src are importable.
#
# Usage:
#   source evaluate_orak/setup_orak_sc2.sh
#   python evaluate_orak/run_orak_benchmark.py --game star_craft --episodes 1
#
# Notes:
#   - SC2 headless Linux client v4.10 installed at game_agent/StarCraftII/StarCraftII/
#   - 2023 ladder maps (Orak default) are incompatible with SC2 4.10 Linux.
#     Use the provided config override or the 2019 ladder maps instead.
#   - Available compatible maps: AutomatonLE, CyberForestLE, KairosJunctionLE,
#     KingsCoveLE, NewRepugnancyLE, PortAleksanderLE, YearZeroLE, AbyssalReefLE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(dirname "$SCRIPT_DIR")"
ORAK_SRC="${CODEBASE_ROOT}/../Orak/src"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate orak-sc2

export SC2PATH="/workspace/game_agent/StarCraftII/StarCraftII"
export PYTHONPATH="${CODEBASE_ROOT}:${ORAK_SRC}:${PYTHONPATH:-}"

echo "=== orak-sc2 environment ==="
echo "  Python:     $(python --version)"
echo "  SC2PATH:    ${SC2PATH}"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo "  Env ready for: StarCraft II (Orak)"
echo ""
echo "  Compatible maps (SC2 4.10 Linux):"
ls -1 "${SC2PATH}/Maps/Ladder2019Season1/" 2>/dev/null | sed 's/\.SC2Map//' | sed 's/^/    /'
echo ""
echo "  Quick test:"
echo "    python -c \"from sc2 import maps; print(maps.get('AutomatonLE'))\""
