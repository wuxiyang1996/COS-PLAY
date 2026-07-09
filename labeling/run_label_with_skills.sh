#!/usr/bin/env bash
# Run GPT-5.4 episode labeling with skill selection + GRPO cold-start export.
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

if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -f "${ROOT_DIR}/api_keys.py" ]]; then
    echo "[info] OPENROUTER_API_KEY not in env; api_keys.py will be used by Python."
fi

BANK_DIR="${ROOT_DIR}/skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo"

echo "============================================================"
echo "  GPT-5.4 Episode Labeling + Skill Selection + GRPO Export"
echo "  Root:  $ROOT_DIR"
echo "  Bank:  $BANK_DIR"
echo "  Args:  $*"
echo "============================================================"

python "$SCRIPT_DIR/label_episodes_with_skills.py" \
    --bank "$BANK_DIR" \
    --one_per_game \
    -v \
    "$@"
