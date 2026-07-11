#!/usr/bin/env bash
# Parallel skill-aware labeling using OpenAI direct (not OpenRouter).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODEBASE_ROOT"

source /venv/main/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODEBASE_ROOT}:${PYTHONPATH:-}"

# OpenAI direct — unset OpenRouter so ask_model routes through OpenAI
export OPENAI_API_KEY="$(python3 -c "import sys; sys.path.insert(0,'/workspace'); from keys import OPENAI_API_KEY; print(OPENAI_API_KEY)" 2>/dev/null)"
export OPENAI_BASE_URL="https://us.api.openai.com/v1"
unset OPENROUTER_API_KEY 2>/dev/null || true

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
echo "  Parallel Skill Labeling (OpenAI Direct)"
echo "  ALFWorld: $N_ALFWORLD episodes -> $SHARDS shards"
echo "  WebShop:  $(ls "$INPUT_DIR/webshop"/episode_[0-9]*.json 2>/dev/null | wc -l) episodes"
echo "================================================================"

PIDS=()

for s in $(seq 0 $((SHARDS-1))); do
  shard_input="${SHARD_BASE}/input_${s}/alfworld"
  shard_output="${SHARD_BASE}/output_${s}"
  mkdir -p "$shard_input" "$shard_output"

  start=$((s * PER_SHARD))
  end=$(( (s+1) * PER_SHARD - 1 ))
  count=0; skip=0
  for i in $(seq $start $end); do
    fname="episode_$(printf '%03d' $i).json"
    f="${INPUT_DIR}/alfworld/${fname}"
    if [ -f "${FINAL_OUTPUT}/alfworld/${fname}" ]; then
      skip=$((skip+1))
    elif [ -f "$f" ]; then
      ln -s "$(realpath "$f")" "$shard_input/"
      count=$((count+1))
    fi
  done
  echo "  Shard $s: $count todo, $skip already done"

  if [ $count -gt 0 ]; then
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

# WebShop — use separate shard output, skip already done
ws_input="${SHARD_BASE}/input_ws/webshop"
ws_output="${SHARD_BASE}/output_ws"
mkdir -p "$ws_input" "$ws_output"
ws_count=0; ws_skip=0
for f in ${INPUT_DIR}/webshop/episode_[0-9]*.json; do
  fname=$(basename "$f")
  if [ -f "${FINAL_OUTPUT}/webshop/${fname}" ]; then
    ws_skip=$((ws_skip+1))
  else
    ln -s "$(realpath "$f")" "$ws_input/"
    ws_count=$((ws_count+1))
  fi
done
echo "  WebShop: $ws_count todo, $ws_skip already done"

if [ $ws_count -gt 0 ]; then
  python3 -u labeling/label_episodes_with_skills.py \
    --input_dir "${SHARD_BASE}/input_ws" \
    --output_dir "$ws_output" \
    --bank "$BANK_DIR" \
    --games webshop \
    --delay 0.01 -v \
    > "$LOG_DIR/webshop.log" 2>&1 &
  PIDS+=($!)
fi

echo ""
echo "Launched ${#PIDS[@]} processes (PIDs: ${PIDS[*]})"
echo ""

# Monitor
while true; do
  sleep 20
  alf=0; web=0
  for d in ${SHARD_BASE}/output_*/alfworld; do
    [ -d "$d" ] && alf=$((alf + $(ls "$d"/episode_*.json 2>/dev/null | wc -l)))
  done
  [ -d "${FINAL_OUTPUT}/alfworld" ] && alf=$((alf + $(ls "${FINAL_OUTPUT}/alfworld"/episode_*.json 2>/dev/null | wc -l)))
  for d in ${SHARD_BASE}/output_*/webshop; do
    [ -d "$d" ] && web=$((web + $(ls "$d"/episode_*.json 2>/dev/null | wc -l)))
  done
  [ -d "${FINAL_OUTPUT}/webshop" ] && web=$((web + $(ls "${FINAL_OUTPUT}/webshop"/episode_*.json 2>/dev/null | wc -l)))
  procs=$(ps aux | grep "[l]abel_episodes_with_skills" | wc -l)
  echo "$(date '+%H:%M:%S') alf: ${alf}/120  web: ${web}/50  procs: $procs"
  [ "$procs" -eq 0 ] && break
done

echo ""
echo "Merging shard outputs..."
for game in alfworld webshop; do
  mkdir -p "${FINAL_OUTPUT}/${game}" "${FINAL_OUTPUT}/grpo_coldstart/${game}"
  for d in ${SHARD_BASE}/output_*/${game}; do
    [ -d "$d" ] && cp -n "$d"/episode_*.json "${FINAL_OUTPUT}/${game}/" 2>/dev/null || true
  done
  for adapter in action_taking skill_selection; do
    for f in ${SHARD_BASE}/output_*/grpo_coldstart/${game}/${adapter}.jsonl; do
      [ -f "$f" ] && cat "$f" >> "${FINAL_OUTPUT}/grpo_coldstart/${game}/${adapter}.jsonl"
    done
  done
done

alf=$(ls "${FINAL_OUTPUT}/alfworld"/episode_*.json 2>/dev/null | wc -l)
web=$(ls "${FINAL_OUTPUT}/webshop"/episode_*.json 2>/dev/null | wc -l)
at_alf=$(wc -l < "${FINAL_OUTPUT}/grpo_coldstart/alfworld/action_taking.jsonl" 2>/dev/null || echo 0)
at_web=$(wc -l < "${FINAL_OUTPUT}/grpo_coldstart/webshop/action_taking.jsonl" 2>/dev/null || echo 0)

echo "================================================================"
echo "  DONE (OpenAI Direct)"
echo "  ALFWorld: $alf episodes, $at_alf action_taking samples"
echo "  WebShop:  $web episodes, $at_web action_taking samples"
echo "  Output:   $FINAL_OUTPUT"
echo "================================================================"

rm -rf "$SHARD_BASE"
