#!/usr/bin/env bash
# Run SFT-teacher (gpt-5.5 by default) episode labeling from the Game-AI-Agent root.
#
# Usage:
#   bash labeling/run_labeling.sh                           # all games
#   bash labeling/run_labeling.sh --games tetris            # one game
#   bash labeling/run_labeling.sh --dry_run --games tetris  # preview
#   bash labeling/run_labeling.sh --one_per_game -v          # one rollout per game
#   bash labeling/run_labeling.sh --in_place                # overwrite originals

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
GAMINGAGENT_DIR="$(dirname "$ROOT_DIR")/GamingAgent"

export PYTHONPATH="${ROOT_DIR}:${GAMINGAGENT_DIR}:${PYTHONPATH:-}"

# Pick up API key from environment or .env file
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Warning: OPENROUTER_API_KEY not set. See .env.example for required keys."
fi

echo "============================================================"
echo "  SFT-Teacher (gpt-5.5) Episode Labeling"
echo "  Root: $ROOT_DIR"
echo "  Args: $*"
echo "============================================================"

python "$SCRIPT_DIR/label_episodes_gpt54.py" "$@"
