#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"
export PYTHONUNBUFFERED=1

OPENAI_API_KEY=$(python3 -c "import sys; sys.path.insert(0,'/workspace'); from keys import OPENAI_API_KEY; print(OPENAI_API_KEY)")
export OPENAI_API_KEY

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PIDS=()
LABELS=()

declare -A TASK_NAMES
TASK_NAMES[1]="pick_and_place_simple"
TASK_NAMES[2]="look_at_obj_in_light"
TASK_NAMES[3]="pick_clean_then_place_in_recep"
TASK_NAMES[4]="pick_heat_then_place_in_recep"
TASK_NAMES[5]="pick_cool_then_place_in_recep"
TASK_NAMES[6]="pick_two_obj_and_place"

for TASK_TYPE in 1 2 3 4 5 6; do
    TASK_NAME=${TASK_NAMES[$TASK_TYPE]}
    SUBDIR="alfworld_${TASK_NAME}"

    echo "[$(date)] Starting ALFWorld task_type=$TASK_TYPE ($TASK_NAME) - 20 episodes"
    python3 -u "$SCRIPT_DIR/generate_cold_start_textenv.py" \
        --games alfworld \
        --episodes 20 \
        --task_types $TASK_TYPE \
        --split train \
        --game_subdir "$SUBDIR" \
        --verbose \
        > "$LOG_DIR/alfworld_${TASK_NAME}.log" 2>&1 &
    PIDS+=($!)
    LABELS+=("alfworld_${TASK_NAME}")
done

echo "[$(date)] Starting WebShop - 50 episodes (resume from existing)"
python3 -u "$SCRIPT_DIR/generate_cold_start_textenv.py" \
    --games webshop \
    --episodes 50 \
    --resume \
    --verbose \
    > "$LOG_DIR/webshop.log" 2>&1 &
PIDS+=($!)
LABELS+=("webshop")

echo ""
echo "Launched ${#PIDS[@]} processes:"
for i in "${!PIDS[@]}"; do
    echo "  [${LABELS[$i]}] PID ${PIDS[$i]}"
done
echo ""
echo "Logs in: $LOG_DIR/"
echo "Monitor: tail -f $LOG_DIR/*.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    LABEL=${LABELS[$i]}
    if ! wait "$PID"; then
        echo "[FAIL] $LABEL (PID $PID) exited with error"
        FAILED=$((FAILED + 1))
    else
        echo "[OK] $LABEL (PID $PID) completed"
    fi
done

echo ""
echo "===== ALL DONE ====="
echo "Failed: $FAILED / ${#PIDS[@]}"
echo ""

for LOG in "$LOG_DIR"/*.log; do
    echo "--- $(basename $LOG) (last 5 lines) ---"
    tail -5 "$LOG"
    echo ""
done
