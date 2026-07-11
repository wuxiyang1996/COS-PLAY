#!/usr/bin/env bash
# Parallel skill-aware labeling for text-env episodes.
# Splits alfworld into 4 shards + webshop = 5 parallel processes.
# Each writes to a separate output dir; merged at the end.
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
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  OPENROUTER_API_KEY="$(python3 -c "import sys; sys.path.insert(0,'/workspace'); from keys import OPENROUTER_API_KEY; print(OPENROUTER_API_KEY)" 2>/dev/null || true)"
  export OPENROUTER_API_KEY
fi
export OPENAI_BASE_URL="https://us.api.openai.com/v1"

INPUT_DIR="labeling/output/gpt54_textenv"
FINAL_OUTPUT="labeling/output/gpt54_textenv_skill_labeled"
BANK_DIR="labeling/output/gpt54_textenv_skillbank"
LOG_DIR="${FINAL_OUTPUT}/logs"
mkdir -p "$LOG_DIR"

SHARD_BASE="/tmp/skill_label_shards_$$"
rm -rf "$SHARD_BASE"

N_ALFWORLD=$(ls "$INPUT_DIR/alfworld"/episode_[0-9]*.json 2>/dev/null | wc -l)
SHARDS=4
PER_SHARD=$(( (N_ALFWORLD + SHARDS - 1) / SHARDS ))

echo "================================================================"
echo "  Parallel Skill-Aware Labeling"
echo "  ALFWorld: $N_ALFWORLD episodes -> $SHARDS shards of ~$PER_SHARD"
echo "  WebShop:  $(ls "$INPUT_DIR/webshop"/episode_[0-9]*.json 2>/dev/null | wc -l) episodes"
echo "  Bank:     $BANK_DIR"
echo "================================================================"

PIDS=()

for s in $(seq 0 $((SHARDS-1))); do
  shard_input="${SHARD_BASE}/input_${s}/alfworld"
  shard_output="${SHARD_BASE}/output_${s}"
  mkdir -p "$shard_input" "$shard_output"
  
  start=$((s * PER_SHARD))
  end=$(( (s+1) * PER_SHARD - 1 ))
  count=0
  for i in $(seq $start $end); do
    f="${INPUT_DIR}/alfworld/episode_$(printf '%03d' $i).json"
    [ -f "$f" ] && ln -s "$(realpath "$f")" "$shard_input/" && count=$((count+1))
  done

  # Skip already-done episodes
  already=0
  for i in $(seq $start $end); do
    fname="episode_$(printf '%03d' $i).json"
    [ -f "${FINAL_OUTPUT}/alfworld/${fname}" ] && already=$((already+1))
  done
  todo=$((count - already))
  echo "  Shard $s: episodes ${start}-${end} ($count files, $already done, $todo todo)"

  if [ $todo -gt 0 ]; then
    python3 -u labeling/label_episodes_with_skills.py \
      --input_dir "${SHARD_BASE}/input_${s}" \
      --output_dir "$shard_output" \
      --bank "$BANK_DIR" \
      --games alfworld \
      --delay 0.01 -v \
      > "$LOG_DIR/alfworld_shard_${s}.log" 2>&1 &
    PIDS+=($!)
  fi
done

# WebShop shard
ws_output="${SHARD_BASE}/output_ws"
mkdir -p "$ws_output"
echo "  WebShop: all episodes"
python3 -u labeling/label_episodes_with_skills.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$ws_output" \
  --bank "$BANK_DIR" \
  --games webshop \
  --delay 0.01 -v \
  > "$LOG_DIR/webshop.log" 2>&1 &
PIDS+=($!)

echo ""
echo "Launched ${#PIDS[@]} parallel processes (PIDs: ${PIDS[*]})"
echo ""

# Monitor
while true; do
  sleep 20
  alf_done=0
  web_done=0
  # Count from shard outputs
  for d in ${SHARD_BASE}/output_*/alfworld; do
    [ -d "$d" ] && alf_done=$((alf_done + $(ls "$d"/episode_*.json 2>/dev/null | wc -l)))
  done
  # Add already-existing ones
  [ -d "${FINAL_OUTPUT}/alfworld" ] && alf_done=$((alf_done + $(ls "${FINAL_OUTPUT}/alfworld"/episode_*.json 2>/dev/null | wc -l)))
  for d in ${SHARD_BASE}/output_*/webshop; do
    [ -d "$d" ] && web_done=$((web_done + $(ls "$d"/episode_*.json 2>/dev/null | wc -l)))
  done
  [ -d "${FINAL_OUTPUT}/webshop" ] && web_done=$((web_done + $(ls "${FINAL_OUTPUT}/webshop"/episode_*.json 2>/dev/null | wc -l)))
  echo "--- $(date '+%H:%M:%S') --- alfworld: ${alf_done}, webshop: ${web_done}"

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

wait

# Merge shard outputs into final dir
echo ""
echo "Merging shard outputs..."

for game in alfworld webshop; do
  mkdir -p "${FINAL_OUTPUT}/${game}"
  mkdir -p "${FINAL_OUTPUT}/grpo_coldstart/${game}"
  
  for d in ${SHARD_BASE}/output_*/${game}; do
    [ -d "$d" ] && cp -n "$d"/episode_*.json "${FINAL_OUTPUT}/${game}/" 2>/dev/null || true
  done
  
  # Merge GRPO JSONL files (concatenate from all shards)
  for adapter in action_taking skill_selection; do
    for f in ${SHARD_BASE}/output_*/grpo_coldstart/${game}/${adapter}.jsonl; do
      [ -f "$f" ] && cat "$f" >> "${FINAL_OUTPUT}/grpo_coldstart/${game}/${adapter}.jsonl"
    done
  done
done

alf_final=$(ls "${FINAL_OUTPUT}/alfworld"/episode_*.json 2>/dev/null | wc -l)
web_final=$(ls "${FINAL_OUTPUT}/webshop"/episode_*.json 2>/dev/null | wc -l)
at_alf=$(wc -l < "${FINAL_OUTPUT}/grpo_coldstart/alfworld/action_taking.jsonl" 2>/dev/null || echo 0)
at_web=$(wc -l < "${FINAL_OUTPUT}/grpo_coldstart/webshop/action_taking.jsonl" 2>/dev/null || echo 0)

echo ""
echo "================================================================"
echo "  DONE"
echo "  ALFWorld: $alf_final episodes, $at_alf action_taking GRPO samples"
echo "  WebShop:  $web_final episodes, $at_web action_taking GRPO samples"
echo "  Output:   $FINAL_OUTPUT"
echo "================================================================"

rm -rf "$SHARD_BASE"
