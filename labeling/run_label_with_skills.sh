#!/usr/bin/env bash
# Run SFT-teacher (gpt-5.5 by default) episode labeling with skill selection + GRPO cold-start export.
#
# Usage:
#   bash labeling/run_label_with_skills.sh                           # all games
#   bash labeling/run_label_with_skills.sh --games tetris            # one game
#   bash labeling/run_label_with_skills.sh --one_per_game -v         # quick test
#   bash labeling/run_label_with_skills.sh --dry_run --games tetris  # preview

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
GAMINGAGENT_DIR="$(dirname "$ROOT_DIR")/GamingAgent"

export PYTHONPATH="${ROOT_DIR}:${GAMINGAGENT_DIR}:${PYTHONPATH:-}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "Warning: OPENROUTER_API_KEY not set. See .env.example for required keys."
fi

BANK_DIR="${ROOT_DIR}/skill_agents/extract_skillbank/output/gpt54_skillbank_grpo"

echo "============================================================"
echo "  SFT-Teacher (gpt-5.5) Episode Labeling + Skill Selection + GRPO Export"
echo "  Root:  $ROOT_DIR"
echo "  Bank:  $BANK_DIR"
echo "  Args:  $*"
echo "============================================================"

python "$SCRIPT_DIR/label_episodes_with_skills.py" \
    --bank "$BANK_DIR" \
    --one_per_game \
    -v \
    "$@"
