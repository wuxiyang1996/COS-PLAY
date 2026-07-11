#!/usr/bin/env bash
# Parallel labeling for text-env episodes.
# Splits alfworld into 4 shards + webshop as 1 shard = 5 parallel processes.
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

INPUT_DIR="cold_start/output/gpt54_textenv"
OUTPUT_DIR="labeling/output/gpt54_textenv"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Split alfworld episodes into shards by symlinking into temp dirs
SHARD_BASE="/tmp/label_shards_$$"
rm -rf "$SHARD_BASE"

ALFWORLD_SRC="${INPUT_DIR}/alfworld"
N_ALFWORLD=$(ls "$ALFWORLD_SRC"/episode_[0-9]*.json 2>/dev/null | wc -l)
SHARDS=4
PER_SHARD=$(( (N_ALFWORLD + SHARDS - 1) / SHARDS ))

echo "================================================================"
echo "  Parallel Text-Env Labeling"
echo "  ALFWorld: $N_ALFWORLD episodes → $SHARDS shards of ~$PER_SHARD"
echo "  WebShop:  $(ls "$INPUT_DIR/webshop"/episode_[0-9]*.json 2>/dev/null | wc -l) episodes"
echo "================================================================"

PIDS=()

# Launch alfworld shards
for s in $(seq 0 $((SHARDS-1))); do
  shard_dir="${SHARD_BASE}/shard_${s}/alfworld"
  mkdir -p "$shard_dir"
  start=$((s * PER_SHARD))
  end=$(( (s+1) * PER_SHARD - 1 ))
  count=0
  for i in $(seq $start $end); do
    f="${ALFWORLD_SRC}/episode_$(printf '%03d' $i).json"
    [ -f "$f" ] && ln -s "$(realpath "$f")" "$shard_dir/" && count=$((count+1))
  done
  echo "  Shard $s: episodes ${start}-${end} ($count files)"

  python3 -u labeling/label_episodes_gpt54.py \
    --input_dir "${SHARD_BASE}/shard_${s}" \
    --output_dir "$OUTPUT_DIR" \
    --games alfworld \
    --delay 0.02 -v \
    > "$LOG_DIR/alfworld_shard_${s}.log" 2>&1 &
  PIDS+=($!)
done

# Launch webshop
echo "  WebShop: all episodes"
python3 -u labeling/label_episodes_gpt54.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --games webshop \
  --delay 0.02 -v \
  > "$LOG_DIR/webshop.log" 2>&1 &
PIDS+=($!)

echo ""
echo "Launched ${#PIDS[@]} parallel labeling processes (PIDs: ${PIDS[*]})"
echo "Logs: $LOG_DIR/"
echo ""

# Monitor
while true; do
  sleep 30
  alf_done=$(ls "$OUTPUT_DIR/alfworld"/episode_*.json 2>/dev/null | wc -l || echo 0)
  web_done=$(ls "$OUTPUT_DIR/webshop"/episode_*.json 2>/dev/null | wc -l || echo 0)
  echo "--- $(date '+%H:%M:%S') --- alfworld: ${alf_done}/120, webshop: ${web_done}/50"

  all_done=true
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      all_done=false
    fi
  done
  if $all_done; then
    echo "All labeling complete!"
    break
  fi
done

rm -rf "$SHARD_BASE"
wait
echo "Done. Output: $OUTPUT_DIR/"
