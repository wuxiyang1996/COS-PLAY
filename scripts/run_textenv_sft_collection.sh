#!/usr/bin/env bash
# Collect SFT trajectories for text environments.
#   WebShop: 50 episodes
#   ALFWorld: 20 episodes per task category (6 categories)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODEBASE_ROOT"

source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODEBASE_ROOT}:${PYTHONPATH:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  OPENAI_API_KEY="$(python3 -c "import sys; sys.path.insert(0,'/workspace'); from keys import OPENAI_API_KEY; print(OPENAI_API_KEY)" 2>/dev/null || true)"
  export OPENAI_API_KEY
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY not set"
  exit 1
fi

GEN="python3 -u cold_start/generate_cold_start_textenv.py"
OUT="cold_start/output/gpt54_textenv"
COMMON=(--model gpt-5.4 --max_steps 50 --temperature 0.4 --split train --output_dir "$OUT" --resume -v)

echo "================================================================"
echo "  WebShop: 50 episodes"
echo "================================================================"
$GEN --games webshop --episodes 50 --num_products 1000 "${COMMON[@]}"

declare -A ALFWORLD_CATS=(
  [1]="pick_and_place_simple"
  [2]="look_at_obj_in_light"
  [3]="pick_clean_then_place_in_recep"
  [4]="pick_heat_then_place_in_recep"
  [5]="pick_cool_then_place_in_recep"
  [6]="pick_two_obj_and_place"
)

for tt in "${!ALFWORLD_CATS[@]}"; do
  name="${ALFWORLD_CATS[$tt]}"
  echo ""
  echo "================================================================"
  echo "  ALFWorld category ${tt}: ${name} (20 episodes)"
  echo "================================================================"
  $GEN --games alfworld --episodes 20 --task_types "$tt" \
    --game_subdir "alfworld/${name}" "${COMMON[@]}"
done

echo ""
echo "================================================================"
echo "  SFT collection complete"
echo "  Output: ${OUT}"
echo "================================================================"
