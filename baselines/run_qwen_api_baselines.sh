#!/usr/bin/env bash
#
# baselines/run_qwen_api_baselines.sh — Qwen3.5-{9B, 35B-A3B} via OpenRouter API
#
# Drop-in API counterpart to baselines/run_qwen_vllm_baselines.sh: same
# COS-PLAY actor cold-start pipeline, same per-(model × env) layout, same
# 8-game retained Gym-V suite — every LLM call is routed through the
# OpenRouter HTTPS endpoint instead of a local vLLM server. No GPUs
# required.
#
#   Backbones (multimodal, text + image):
#     qwen/qwen3.5-9b           ($0.10 / $0.15 per 1M in/out)
#     qwen/qwen3.5-35b-a3b      ($0.16 / $1.30 per 1M in/out)
#
#   Default sweep:  2 backbones × 8 retained Gym-V envs × 16 episodes
#                   (matches the 2026-05-03 frame_skip=8 leaderboard sweep)
#
# Two run modes:
#
# 1.  Unsharded (default; --episode_shards 1):
#     Thin pass-through to baselines/run_openrouter_baselines.sh with
#     `--models qwen35-9b qwen35-35b-a3b --skip_envwrappers`. 16 cells in
#     flight (8 games × 2 backbones), each cell runs its 16 episodes
#     serially. Wall-clock = (eps × steps × per_step_latency).
#
# 2.  Sharded (--episode_shards N for N > 1):
#     Each (model × game) cell is split into N independent sub-cells with
#     contiguous seed ranges (`--seed_base offset` + `--episodes eps/N`)
#     and a per-shard output subdir. All N × 2 × 8 = 16N sub-cells run
#     in parallel (concurrency cap = max_parallel × N). After every
#     sub-cell finishes, a Python merger flattens each game's
#     `_shard_K/episode_NNN.json` files into the canonical
#     `<env>/episode_NNN.json` namespace, concatenates `rollouts.jsonl`,
#     and rebuilds `rollout_summary.json` so downstream tooling
#     (aggregate_skip8_leaderboard.py etc.) treats the run as one logical
#     sweep with the full 16-episode budget per cell.
#
#     Wall-clock improvement: ~N× (capped by OpenRouter rate limits;
#     N=4 → ~4× speedup with 64 concurrent streams, comfortably under
#     observed limits).
#
# Output layout (both modes look identical post-merge):
#
#   <codebase_root>/qwen-api-baselines-out/<run_id>/
#     qwen3.5-9b/gymv/Temporal_<Title>-v0/
#       episode_000.json … episode_015.json
#       rollouts.jsonl
#       rollout_summary.json
#       steps_stream/ep_NNN.jsonl
#     qwen3.5-35b-a3b/gymv/Temporal_<Title>-v0/...
#     _logs/<model_tag>__gymv__<env_id>[__shard<K>].log
#     _run_meta.json
#   <codebase_root>/qwen-api-baselines-out/latest -> <run_id>
#
# Usage:
#
#   # Default: both backbones × 8 retained Gym-V envs × 16 episodes,
#   # frame_skip=8, max_steps=80, vision ON, 16 in flight, NO sharding.
#   bash baselines/run_qwen_api_baselines.sh
#
#   # 4× sharding — 64 cells in flight, ~4× faster, identical artifacts:
#   bash baselines/run_qwen_api_baselines.sh --episode_shards 4
#
#   # Only the 9B leg, sharded 2×:
#   bash baselines/run_qwen_api_baselines.sh --models qwen35-9b --episode_shards 2
#
#   # Smoke run: 2 episodes, 2 envs only:
#   bash baselines/run_qwen_api_baselines.sh --episodes 2 \
#       --gymv Temporal/Airstriker-v0 Temporal/Columns-v0
#
#   # Re-include env_wrappers (sharding is gym-v-only; env_wrappers always
#   # run unsharded via the underlying runner):
#   bash baselines/run_qwen_api_baselines.sh --include_envwrappers
#
# Wrapper-only flags (consumed here, NOT forwarded to the underlying):
#   --episode_shards N         split each (model × game) into N parallel
#                              sub-cells (default: 1 — unsharded). Episodes
#                              must be divisible by N.
#   --include_envwrappers      keep env_wrappers + super_mario in the
#                              dispatch (default: skipped — gym-v only).
#                              Note: env_wrappers run UNSHARDED even when
#                              --episode_shards > 1.
#   --output_dir <path>        override base dir
#                              (default: <codebase_root>/qwen-api-baselines-out)
#
# Pass-throughs to run_openrouter_baselines.sh (most useful):
#   --models <list>            subset of {qwen35-9b, qwen35-35b-a3b}
#                              (default: both)
#   --episodes N               episodes per (model × env) (default: 16)
#   --max_parallel N           concurrent jobs cap (default: 16; in
#                              sharded mode, scaled to N × 16)
#   --max_steps_gymv N         per-episode step cap (default: 80)
#   --frame_skip_gymv N        emulator frames per agent step (default: 8)
#   --gymv <id>...             restrict gym-v env ids
#   --no_vision                disable VLM vision call
#   --resume                   resume previously-started run
#   --run_id <id>              override timestamped run id
#   --temperature_action F     sampling temp for action call (default: 0.4)
#   --temperature_schema F     sampling temp for schema call  (default: 0.2)
#   --base_url <url>           override OpenRouter URL
#   --verbose | -v             pass-through
#
# Optional env vars:
#   OPENROUTER_API_KEY         auth token (auto-loaded from api_keys.py)
#   MODEL_QWEN35_9B            override the 9B  slug (default: qwen/qwen3.5-9b)
#   MODEL_QWEN35_35B           override the 35B slug (default: qwen/qwen3.5-35b-a3b)

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# ── Resolve paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"
GYMV_ROOT="$(cd "$WORKSPACE_ROOT/gym-v" 2>/dev/null && pwd || echo "")"
GAMINGAGENT_ROOT="$(cd "$WORKSPACE_ROOT/GamingAgent" 2>/dev/null && pwd || echo "")"
ORAK_SRC="$(cd "$WORKSPACE_ROOT/Orak/src" 2>/dev/null && pwd || echo "")"
PY_GYMV="${CODEBASE_ROOT}/cold_start/generate_cold_start_actor_gymv.py"
UNDERLYING="${SCRIPT_DIR}/run_openrouter_baselines.sh"

# ── Defaults tuned to the 2026-05-03 leaderboard sweep ───────────────────
DEFAULT_OUTPUT_DIR="${CODEBASE_ROOT}/qwen-api-baselines-out"
DEFAULT_EPISODES=16
DEFAULT_MAX_PARALLEL=16
DEFAULT_MAX_STEPS_GYMV=80
DEFAULT_FRAME_SKIP_GYMV=8
DEFAULT_TEMP_ACTION=0.4
DEFAULT_TEMP_SCHEMA=0.2
DEFAULT_BASE_URL="https://openrouter.ai/api/v1"
DEFAULT_CONDA_MAIN="game-ai-agent"
DEFAULT_SEED_BASE=0  # base for all shards; shard k uses base + k*eps_per_shard

MODEL_QWEN35_9B_SLUG="${MODEL_QWEN35_9B:-qwen/qwen3.5-9b}"
MODEL_QWEN35_35B_SLUG="${MODEL_QWEN35_35B:-qwen/qwen3.5-35b-a3b}"

# Default 8-game retained Gym-V suite (must match GYMV_DEFAULT in
# run_openrouter_baselines.sh — kept in sync here so sharded mode does
# not have to read the underlying's source).
GYMV_DEFAULT=(
    Temporal/Airstriker-v0
    Temporal/AlteredBeast-v0
    Temporal/Columns-v0
    Temporal/DynamiteHeaddy-v0
    Temporal/SpaceHarrierII-v0
    Temporal/StreetsOfRage2-v0
    Temporal/Strider-v0
    Temporal/ThunderForceIII-v0
)

# ── Parse args ────────────────────────────────────────────────────────────
EPISODE_SHARDS=1
INCLUDE_ENVWRAPPERS=0
OUTPUT_DIR=""

# Wrapper-explicit knobs that we also need locally for sharded mode.
LOCAL_EPISODES=$DEFAULT_EPISODES
LOCAL_MAX_PARALLEL=$DEFAULT_MAX_PARALLEL
LOCAL_MAX_STEPS_GYMV=$DEFAULT_MAX_STEPS_GYMV
LOCAL_FRAME_SKIP_GYMV=$DEFAULT_FRAME_SKIP_GYMV
LOCAL_TEMP_ACTION=$DEFAULT_TEMP_ACTION
LOCAL_TEMP_SCHEMA=$DEFAULT_TEMP_SCHEMA
LOCAL_BASE_URL=$DEFAULT_BASE_URL
LOCAL_CONDA_MAIN=$DEFAULT_CONDA_MAIN
LOCAL_RUN_ID=""
LOCAL_NO_VISION=0
LOCAL_RESUME=0
LOCAL_VERBOSE=0
LOCAL_SAVE_FRAMES=0
LOCAL_SEED_BASE=$DEFAULT_SEED_BASE
LOCAL_MODELS=()
LOCAL_GYMV=()

PASSTHROUGH=()
USER_OVERRIDE_MODELS=0
USER_OVERRIDE_EPISODES=0
USER_OVERRIDE_PARALLEL=0
USER_OVERRIDE_STEPS=0
USER_OVERRIDE_SKIP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --episode_shards|--episode-shards)
            EPISODE_SHARDS="$2"; shift 2 ;;
        --include_envwrappers|--include-envwrappers)
            INCLUDE_ENVWRAPPERS=1; shift ;;
        --output_dir|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --models)
            USER_OVERRIDE_MODELS=1
            PASSTHROUGH+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                LOCAL_MODELS+=("$1")
                PASSTHROUGH+=("$1"); shift
            done ;;
        --episodes)
            USER_OVERRIDE_EPISODES=1
            LOCAL_EPISODES="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --max_parallel|--max-parallel)
            USER_OVERRIDE_PARALLEL=1
            LOCAL_MAX_PARALLEL="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --max_steps_gymv|--max-steps-gymv)
            USER_OVERRIDE_STEPS=1
            LOCAL_MAX_STEPS_GYMV="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --frame_skip_gymv|--frame-skip-gymv)
            USER_OVERRIDE_SKIP=1
            LOCAL_FRAME_SKIP_GYMV="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --gymv)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                LOCAL_GYMV+=("$1")
                PASSTHROUGH+=("$1") || true   # fall-through to underlying for unsharded mode
                shift
            done
            # Also forward the --gymv flag itself for unsharded mode
            # (we accumulated values above; prepend the flag once).
            ;;
        --temperature_action|--temperature-action)
            LOCAL_TEMP_ACTION="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --temperature_schema|--temperature-schema)
            LOCAL_TEMP_SCHEMA="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --base_url|--base-url)
            LOCAL_BASE_URL="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --conda_main|--conda-main)
            LOCAL_CONDA_MAIN="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --run_id|--run-id)
            LOCAL_RUN_ID="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --no_vision|--no-vision)
            LOCAL_NO_VISION=1
            PASSTHROUGH+=("$1"); shift ;;
        --vision)
            LOCAL_NO_VISION=0
            PASSTHROUGH+=("$1"); shift ;;
        --resume)
            LOCAL_RESUME=1
            PASSTHROUGH+=("$1"); shift ;;
        --verbose|-v)
            LOCAL_VERBOSE=1
            PASSTHROUGH+=("$1"); shift ;;
        --save_frames|--save-frames)
            LOCAL_SAVE_FRAMES=1
            PASSTHROUGH+=("$1"); shift ;;
        --seed_base|--seed-base)
            LOCAL_SEED_BASE="$2"
            PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        *)
            PASSTHROUGH+=("$1"); shift ;;
    esac
done

# ── Validate sharding ─────────────────────────────────────────────────────
if ! [[ "$EPISODE_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --episode_shards must be a positive integer (got '$EPISODE_SHARDS')" >&2
    exit 2
fi
if [ "$EPISODE_SHARDS" -gt 1 ]; then
    if [ $((LOCAL_EPISODES % EPISODE_SHARDS)) -ne 0 ]; then
        echo "[ERROR] episodes ($LOCAL_EPISODES) must be divisible by --episode_shards ($EPISODE_SHARDS)" >&2
        exit 2
    fi
fi

# ── If unsharded → fast-path to underlying (existing behaviour) ──────────
if [ "$EPISODE_SHARDS" -eq 1 ]; then
    if [ "$USER_OVERRIDE_MODELS" -eq 0 ]; then
        PASSTHROUGH+=(--models qwen35-9b qwen35-35b-a3b)
    fi
    if [ "$USER_OVERRIDE_EPISODES" -eq 0 ]; then
        PASSTHROUGH+=(--episodes "$DEFAULT_EPISODES")
    fi
    if [ "$USER_OVERRIDE_PARALLEL" -eq 0 ]; then
        PASSTHROUGH+=(--max_parallel "$DEFAULT_MAX_PARALLEL")
    fi
    if [ "$USER_OVERRIDE_STEPS" -eq 0 ]; then
        PASSTHROUGH+=(--max_steps_gymv "$DEFAULT_MAX_STEPS_GYMV")
    fi
    if [ "$USER_OVERRIDE_SKIP" -eq 0 ]; then
        PASSTHROUGH+=(--frame_skip_gymv "$DEFAULT_FRAME_SKIP_GYMV")
    fi
    if [ "$INCLUDE_ENVWRAPPERS" -eq 0 ]; then
        PASSTHROUGH+=(--skip_envwrappers --skip_mario)
    fi
    [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    PASSTHROUGH+=(--output_dir "$OUTPUT_DIR")

    echo "============================================================"
    echo "  Qwen3.5 API baselines (OpenRouter, UNSHARDED)"
    echo "============================================================"
    echo "  Wrapper around: $UNDERLYING"
    echo "  Output dir:     $OUTPUT_DIR"
    echo "  Mode:           $([ "$INCLUDE_ENVWRAPPERS" -eq 1 ] && echo 'env_wrappers + gym-v' || echo 'gym-v only')"
    echo "  Forward args:   ${PASSTHROUGH[*]}"
    echo "============================================================"
    echo
    exec bash "$UNDERLYING" "${PASSTHROUGH[@]}"
fi

# ── Sharded mode: dispatch directly to the python launcher ───────────────
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
[ -z "$LOCAL_RUN_ID" ] && LOCAL_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${OUTPUT_DIR}/${LOCAL_RUN_ID}"
LOG_DIR="${RUN_DIR}/_logs"
META_FILE="${RUN_DIR}/_run_meta.json"
mkdir -p "$LOG_DIR"

# Default model selection
declare -a MODEL_TAGS=()
declare -a MODEL_SLUGS=()
if [ "$USER_OVERRIDE_MODELS" -eq 1 ] && [ "${#LOCAL_MODELS[@]}" -gt 0 ]; then
    for tag in "${LOCAL_MODELS[@]}"; do
        case "$tag" in
            qwen35-9b|qwen3.5-9b|qwen-3.5-9b|9b)
                MODEL_TAGS+=("qwen3.5-9b"); MODEL_SLUGS+=("$MODEL_QWEN35_9B_SLUG") ;;
            qwen35-35b-a3b|qwen3.5-35b-a3b|qwen-3.5-35b-a3b|35b|35b-a3b)
                MODEL_TAGS+=("qwen3.5-35b-a3b"); MODEL_SLUGS+=("$MODEL_QWEN35_35B_SLUG") ;;
            *)
                echo "[ERROR] Unknown --models tag in sharded mode: '$tag' (allowed: qwen35-9b, qwen35-35b-a3b)" >&2
                exit 2 ;;
        esac
    done
else
    MODEL_TAGS=("qwen3.5-9b" "qwen3.5-35b-a3b")
    MODEL_SLUGS=("$MODEL_QWEN35_9B_SLUG" "$MODEL_QWEN35_35B_SLUG")
fi

# Default games
[ "${#LOCAL_GYMV[@]}" -eq 0 ] && LOCAL_GYMV=("${GYMV_DEFAULT[@]}")

EPS_PER_SHARD=$((LOCAL_EPISODES / EPISODE_SHARDS))
SHARDED_MAX_PARALLEL=$((LOCAL_MAX_PARALLEL * EPISODE_SHARDS))

# Conda
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not on PATH." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

# OpenRouter API key bootstrap (mirrors the underlying runner).
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    if [ -f "${WORKSPACE_ROOT}/api_keys.py" ]; then
        OPENROUTER_API_KEY="$(
            python3 - <<PYEOF
import sys, importlib.util
spec = importlib.util.spec_from_file_location("k", "${WORKSPACE_ROOT}/api_keys.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(getattr(m, "openrouter_api_key", "") or getattr(m, "OPENROUTER_API_KEY", "") or "")
PYEOF
        )"
        export OPENROUTER_API_KEY
    fi
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[ERROR] OPENROUTER_API_KEY not set and not found in api_keys.py" >&2
    exit 1
fi

# Headless rendering
export SDL_VIDEODRIVER=dummy
export PYGLET_HEADLESS=1
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1; then
        if ! pgrep -x Xvfb >/dev/null 2>&1; then
            Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
            sleep 0.5
        fi
        export DISPLAY=":99"
    fi
fi

# PYTHONPATH
PYPATH_ADD=("${CODEBASE_ROOT}")
[ -n "${GAMINGAGENT_ROOT}" ] && [ -d "${GAMINGAGENT_ROOT}" ] && PYPATH_ADD+=("${GAMINGAGENT_ROOT}")
[ -n "${ORAK_SRC}" ]         && [ -d "${ORAK_SRC}" ]         && PYPATH_ADD+=("${ORAK_SRC}")
[ -n "${GYMV_ROOT}" ]        && [ -d "${GYMV_ROOT}" ]        && PYPATH_ADD+=("${GYMV_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

sanitize() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'; }

# ── Banner ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Qwen3.5 API baselines (OpenRouter, SHARDED ×${EPISODE_SHARDS})"
echo "============================================================"
echo "  Run id:           $LOCAL_RUN_ID"
echo "  Run dir:          $RUN_DIR"
echo "  Episodes/cell:    $LOCAL_EPISODES  ($EPISODE_SHARDS shards × $EPS_PER_SHARD eps/shard)"
echo "  max_steps_gymv:   $LOCAL_MAX_STEPS_GYMV"
echo "  frame_skip_gymv:  $LOCAL_FRAME_SKIP_GYMV"
echo "  Vision:           $([ "$LOCAL_NO_VISION" -eq 1 ] && echo OFF || echo ON)"
echo "  Resume:           $([ "$LOCAL_RESUME" -eq 1 ] && echo ON || echo OFF)"
echo "  Sub-cells:        $((${#MODEL_TAGS[@]} * ${#LOCAL_GYMV[@]} * EPISODE_SHARDS))   (= ${#MODEL_TAGS[@]} models × ${#LOCAL_GYMV[@]} games × $EPISODE_SHARDS shards)"
echo "  Concurrency cap:  $SHARDED_MAX_PARALLEL"
echo "  Endpoint:         $LOCAL_BASE_URL"
echo "  API key prefix:   ${OPENROUTER_API_KEY:0:10}..."
echo
echo "  Backbones:"
for i in "${!MODEL_TAGS[@]}"; do
    printf "    [OK]  %-22s -> %s\n" "${MODEL_TAGS[$i]}" "${MODEL_SLUGS[$i]}"
done
echo
echo "  Gym-V envs (${#LOCAL_GYMV[@]}):"
for e in "${LOCAL_GYMV[@]}"; do echo "    - $e"; done
echo "============================================================"
echo

# ── Per-sub-cell launcher ─────────────────────────────────────────────────
run_shard_cell() {
    local model_tag=$1 model_slug=$2 env_id=$3 shard_idx=$4
    local model_safe;  model_safe="$(sanitize "$model_tag")"
    local env_safe;    env_safe="$(sanitize "$env_id")"
    local logfile="${LOG_DIR}/${model_safe}__gymv__${env_safe}__shard$(printf '%03d' "$shard_idx").log"
    local seed_for_shard=$((LOCAL_SEED_BASE + shard_idx * EPS_PER_SHARD))

    # Per-shard parent: launcher writes <env_safe>/episode_NNN.json under it,
    # so the on-disk layout is
    #   ${RUN_DIR}/${model_safe}/gymv/_shard_NNN/${env_safe}/episode_NNN.json
    # The merger globs ``_shard_*`` directly under ``${model_safe}/gymv/``.
    local per_shard_parent="${RUN_DIR}/${model_safe}/gymv/_shard_$(printf '%03d' "$shard_idx")"
    mkdir -p "$per_shard_parent"

    local extra=()
    [ "$LOCAL_NO_VISION"   -eq 1 ] && extra+=(--no_vision)
    [ "$LOCAL_RESUME"      -eq 1 ] && extra+=(--resume)
    [ "$LOCAL_VERBOSE"     -eq 1 ] && extra+=(--verbose)
    [ "$LOCAL_SAVE_FRAMES" -eq 1 ] && extra+=(--save_frames)

    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$LOCAL_CONDA_MAIN" --no-capture-output \
        python3 "$PY_GYMV" \
            --envs "$env_id" \
            --episodes "$EPS_PER_SHARD" \
            --max_steps "$LOCAL_MAX_STEPS_GYMV" \
            --frame_skip "$LOCAL_FRAME_SKIP_GYMV" \
            --model "$model_slug" \
            --api_key "$OPENROUTER_API_KEY" \
            --base_url "$LOCAL_BASE_URL" \
            --temperature_action "$LOCAL_TEMP_ACTION" \
            --temperature_schema "$LOCAL_TEMP_SCHEMA" \
            --seed_base "$seed_for_shard" \
            --output_dir "$per_shard_parent" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

# ── Build job list (one job per (model × env × shard)) ────────────────────
declare -a JOBS=()
for i in "${!MODEL_TAGS[@]}"; do
    for env_id in "${LOCAL_GYMV[@]}"; do
        for ((k=0; k<EPISODE_SHARDS; k++)); do
            JOBS+=("${MODEL_TAGS[$i]}|${MODEL_SLUGS[$i]}|${env_id}|${k}")
        done
    done
done

echo "Total sub-cells to dispatch: ${#JOBS[@]}"
echo "Live tail (any of these):"
for spec in "${JOBS[@]}"; do
    IFS='|' read -r tag _ env_id k <<<"$spec"
    model_safe="$(sanitize "$tag")"
    env_safe="$(sanitize "$env_id")"
    echo "  tail -f ${LOG_DIR}/${model_safe}__gymv__${env_safe}__shard$(printf '%03d' "$k").log"
done | head -16
[ "${#JOBS[@]}" -gt 16 ] && echo "  ... (+$((${#JOBS[@]} - 16)) more)"
echo

# ── Parallel dispatch with concurrency cap ────────────────────────────────
declare -A PIDS=()
declare -A RC=()
declare -a INFLIGHT=()

wait_one() {
    local first="${INFLIGHT[0]}"
    INFLIGHT=("${INFLIGHT[@]:1}")
    wait "${PIDS[$first]}"
    local rc=$?
    RC[$first]=$rc
    printf "[DONE]  %s   rc=%d   ts=%s\n" "$first" "$rc" "$(date +%H:%M:%S)"
}

START_TS="$(date +%s)"
for spec in "${JOBS[@]}"; do
    while [ "${#INFLIGHT[@]}" -ge "$SHARDED_MAX_PARALLEL" ]; do
        wait_one
    done

    IFS='|' read -r tag slug env_id k <<<"$spec"
    job_id="${tag}|${env_id}|shard${k}"

    run_shard_cell "$tag" "$slug" "$env_id" "$k" &
    PIDS["$job_id"]=$!
    INFLIGHT+=("$job_id")
    printf "[START] %s   pid=%d   ts=%s\n" "$job_id" "${PIDS[$job_id]}" "$(date +%H:%M:%S)"
done

while [ "${#INFLIGHT[@]}" -gt 0 ]; do
    wait_one
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo
echo "All sub-cells finished in ${ELAPSED}s"

# ── Merge per-shard outputs into the canonical flat namespace ─────────────
echo
echo "Merging per-shard outputs..."
python3 - <<PYEOF
"""Merge per-shard outputs of run_qwen_api_baselines.sh sharded mode.

For each (model × env), walk the K shard subdirs:
  ${RUN_DIR}/<model>/gymv/_shard_NNN/<env_safe>/...
and consolidate them into:
  ${RUN_DIR}/<model>/gymv/<env_safe>/
    episode_000.json … episode_(N-1).json    ← globally renumbered
    rollouts.jsonl                              ← concatenated in shard order
    rollout_summary.json                        ← merged episode_stats
    steps_stream/ep_NNN.jsonl                   ← copied with global names
"""
import json, re, shutil, sys
from pathlib import Path

run_dir = Path("${RUN_DIR}")
shards  = ${EPISODE_SHARDS}
eps_per = ${EPS_PER_SHARD}

merged = {"models": [], "envs": [], "shards": shards, "eps_per_shard": eps_per,
          "total_eps_per_cell": shards * eps_per, "merged_cells": 0,
          "errors": []}

for model_dir in sorted(run_dir.glob("*/gymv")):
    if not model_dir.is_dir(): continue
    model_safe = model_dir.parent.name
    if not any(model_dir.glob("_shard_*")):
        continue
    merged["models"].append(model_safe)
    # Collect env_safes seen across any shard
    env_safes = set()
    for sd in model_dir.glob("_shard_*"):
        for ed in sd.iterdir():
            if ed.is_dir() and ed.name.startswith("Temporal_"):
                env_safes.add(ed.name)
    for env_safe in sorted(env_safes):
        canonical = model_dir / env_safe
        canonical.mkdir(parents=True, exist_ok=True)
        (canonical / "steps_stream").mkdir(exist_ok=True)
        merged_summary = None
        all_rollouts_lines = []
        global_ep = 0
        for k in range(shards):
            shard_env = model_dir / f"_shard_{k:03d}" / env_safe
            if not shard_env.is_dir():
                merged["errors"].append(f"missing {shard_env}")
                continue
            # 1. Episode JSONs
            for ep_path in sorted(shard_env.glob("episode_*.json")):
                m = re.match(r"episode_(\d+)\.json$", ep_path.name)
                if not m: continue
                local_idx = int(m.group(1))
                gidx = k * eps_per + local_idx
                target = canonical / f"episode_{gidx:03d}.json"
                shutil.move(str(ep_path), str(target))
                # Patch episode_id field if present
                try:
                    d = json.loads(target.read_text())
                    if isinstance(d, dict) and "episode_id" in d:
                        d["episode_id"] = gidx
                        target.write_text(json.dumps(d))
                except Exception:
                    pass
                global_ep = max(global_ep, gidx + 1)
            # 2. steps_stream files
            ss = shard_env / "steps_stream"
            if ss.is_dir():
                for s_path in sorted(ss.glob("ep_*.jsonl")):
                    m = re.match(r"ep_(\d+)\.jsonl$", s_path.name)
                    if not m: continue
                    local_idx = int(m.group(1))
                    gidx = k * eps_per + local_idx
                    target = canonical / "steps_stream" / f"ep_{gidx:03d}.jsonl"
                    shutil.move(str(s_path), str(target))
            # 3. rollouts.jsonl (concatenated in shard order — line order
            #    will mirror global-episode order because each shard
            #    emits rollouts.jsonl in episode order).
            r_path = shard_env / "rollouts.jsonl"
            if r_path.is_file():
                with r_path.open() as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            if "episode_idx" in d:
                                d["episode_idx"] = k * eps_per + int(d["episode_idx"])
                            if "episode_id" in d:
                                d["episode_id"] = k * eps_per + int(d["episode_id"])
                            all_rollouts_lines.append(json.dumps(d) + "\n")
                        except Exception:
                            all_rollouts_lines.append(line)
            # 4. rollout_summary.json — merge episode_stats
            s_path = shard_env / "rollout_summary.json"
            if s_path.is_file():
                try:
                    sd = json.loads(s_path.read_text())
                    if merged_summary is None:
                        merged_summary = dict(sd)
                        merged_summary["episode_stats"] = []
                        merged_summary.pop("elapsed_seconds", None)
                    eps = sd.get("episode_stats") or []
                    for ep in eps:
                        gidx = k * eps_per + int(ep.get("episode_idx", 0) or 0)
                        ep = dict(ep); ep["episode_idx"] = gidx
                        merged_summary["episode_stats"].append(ep)
                    merged_summary["elapsed_seconds"] = (
                        merged_summary.get("elapsed_seconds", 0)
                        + (sd.get("elapsed_seconds") or 0)
                    )
                except Exception as exc:
                    merged["errors"].append(f"summary parse {s_path}: {exc}")
        # Write back the canonical rollouts + summary
        if all_rollouts_lines:
            (canonical / "rollouts.jsonl").write_text("".join(all_rollouts_lines))
        if merged_summary is not None:
            merged_summary["episodes_per_env"] = global_ep
            merged_summary.setdefault("env_id", env_safe.replace("_", "/", 1).replace("-v0_", "-v0").replace("Temporal_", "Temporal/"))
            (canonical / "rollout_summary.json").write_text(
                json.dumps(merged_summary, indent=2, default=str)
            )
        merged["merged_cells"] += 1

    # Optionally tear down empty shard dirs
    for sd in list(model_dir.glob("_shard_*")):
        try:
            shutil.rmtree(sd)
        except Exception:
            pass

# Write run-meta
meta = {
    "run_id": "${LOCAL_RUN_ID}",
    "mode": "sharded",
    "episode_shards": shards,
    "eps_per_shard": eps_per,
    "total_eps_per_cell": shards * eps_per,
    "max_steps": ${LOCAL_MAX_STEPS_GYMV},
    "frame_skip": ${LOCAL_FRAME_SKIP_GYMV},
    "merged": merged,
}
Path("${META_FILE}").write_text(json.dumps(meta, indent=2, default=str))
print(f"  merged {merged['merged_cells']} (model × env) cells across {merged['models']}")
print(f"  episodes per merged cell: {merged['total_eps_per_cell']}")
if merged['errors']:
    print(f"  WARN: {len(merged['errors'])} merge errors:")
    for e in merged['errors'][:5]: print(f"    - {e}")
PYEOF

# Update latest -> this run
ln -sfn "$(basename "$RUN_DIR")" "${OUTPUT_DIR}/latest" 2>/dev/null || true

echo
echo "Done.  Run dir: $RUN_DIR"
echo "       Latest -> ${OUTPUT_DIR}/latest"
