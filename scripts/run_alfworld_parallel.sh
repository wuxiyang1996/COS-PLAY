#!/usr/bin/env bash
# Collect ALFWorld SFT trajectories in parallel — one process per category.
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

GEN="python3 -u cold_start/generate_cold_start_textenv.py"
OUT="cold_start/output/gpt54_textenv"
EPISODES=${1:-20}
COMMON=(--model gpt-5.4 --max_steps 50 --temperature 0.2 --split train --output_dir "$OUT" --resume -v)

declare -A CATS=(
  [1]="pick_and_place_simple"
  [2]="look_at_obj_in_light"
  [3]="pick_clean_then_place_in_recep"
  [4]="pick_heat_then_place_in_recep"
  [5]="pick_cool_then_place_in_recep"
  [6]="pick_two_obj_and_place"
)

LOG_DIR="${CODEBASE_ROOT}/cold_start/output/gpt54_textenv/logs"
mkdir -p "$LOG_DIR"

PIDS=()
for tt in "${!CATS[@]}"; do
  name="${CATS[$tt]}"
  echo "[LAUNCH] Category ${tt}: ${name} (${EPISODES} episodes)"
  $GEN --games alfworld --episodes "$EPISODES" --task_types "$tt" \
    --game_subdir "alfworld_${name}" "${COMMON[@]}" \
    > "$LOG_DIR/${name}.log" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All 6 categories launched in parallel (PIDs: ${PIDS[*]})"
echo "Logs: ${LOG_DIR}/"
echo ""

# Monitor progress
while true; do
  sleep 60
  all_done=true
  echo "--- Progress $(date '+%H:%M:%S') ---"
  for tt in "${!CATS[@]}"; do
    name="${CATS[$tt]}"
    dir="${OUT}/alfworld_${name}"
    count=$(ls "$dir"/episode_*.json 2>/dev/null | wc -l || echo 0)
    echo "  ${name}: ${count}/${EPISODES}"
    if [[ "$count" -lt "$EPISODES" ]]; then
      all_done=false
    fi
  done
  if $all_done; then
    echo ""
    echo "All categories complete!"
    break
  fi

  # Check for crashed processes
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  [WARN] PID $pid exited"
    fi
  done
done

wait
echo "Done. Output: ${OUT}/alfworld_*/"
