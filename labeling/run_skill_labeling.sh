#!/usr/bin/env bash
# Run GPT-5.4 episode labeling + skill extraction from the Game-AI-Agent root.
#
# This wraps label_and_extract_skills_gpt54.py which runs two phases:
#   Phase 1: Label episodes (summary_state, summary, intentions)
#   Phase 2: Extract skills via SkillBankAgent (segment, contract, name)
#
# Usage:
#   bash labeling/run_skill_labeling.sh                                # all games
#   bash labeling/run_skill_labeling.sh --games tetris                 # one game
#   bash labeling/run_skill_labeling.sh --dry_run --games tetris       # preview
#   bash labeling/run_skill_labeling.sh --one_per_game -v              # quick test
#   bash labeling/run_skill_labeling.sh --skip_labeling \
#        --labeled_dir labeling/output/gpt54                           # skills only
#   bash labeling/run_skill_labeling.sh --skip_skills                  # labels only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
GAMINGAGENT_DIR="$(dirname "$ROOT_DIR")/GamingAgent"

export PYTHONPATH="${ROOT_DIR}:${GAMINGAGENT_DIR}:${PYTHONPATH:-}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -f "${ROOT_DIR}/api_keys.py" ]]; then
    echo "[info] OPENROUTER_API_KEY not in env; api_keys.py will be used by Python."
fi

echo "============================================================"
echo "  GPT-5.4 Episode Labeling + Skill Extraction"
echo "  Root: $ROOT_DIR"
echo "  Args: $*"
echo "============================================================"

python "$SCRIPT_DIR/label_and_extract_skills_gpt54.py" "$@"
