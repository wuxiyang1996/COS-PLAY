#!/usr/bin/env bash
#
# baselines/run_openrouter_baselines.sh — Frontier API baselines via OpenRouter
#
# Drives the COS-PLAY actor cold-start pipeline (visual grounding +
# schema-driven action selection) over all 17 games using three frontier
# backbones routed through OpenRouter:
#
#   anthropic/claude-4.6-sonnet-20260217  (Claude Sonnet 4.6, multimodal)
#   google/gemini-3.1-pro-preview         (Gemini 3.1 Pro,    multimodal)
#   qwen/qwen3-max                        (Qwen3 Max,         text-only → --no_vision)
#
# Reuses the same Python back-ends as cold_start/run_coldstart_actor_*.sh:
#
#   cold_start/generate_cold_start_actor.py        (env_wrappers + super_mario)
#   cold_start/generate_cold_start_actor_gymv.py   (gym-v Temporal)
#
# Game roster (12 total — 5 Gym-V games dropped 2026-05-03; see
# `baselines/README.md` § "Gym-V benchmark scope" for the decision log):
#   - env_wrappers (3): twenty_forty_eight, candy_crush, tetris
#   - Orak (1):          super_mario        (uses --conda_orak)
#   - gym-v (8):         Airstriker, AlteredBeast, Columns, DynamiteHeaddy,
#                        SpaceHarrierII, StreetsOfRage2, Strider, ThunderForceIII
#
# Total dispatched jobs: |MODELS| × 12  (default 3 × 12 = 36).
#
# Default budget: 16 episodes per (model × env), max 16 in flight.
# Vision is ON for Claude/Gemini and forced OFF for Qwen3-Max (text-only
# OpenRouter endpoint). Per the cost model in the repo, the full sweep is
# ~$1.96k all-in across the three models
# (≈$1094 Claude + $817 Gemini + $53 Qwen3-Max no_vision).
#
# Output layout:
#
#   <codebase_root>/openrouter-baselines-out/<run_id>/
#     <model_tag>/
#       env_wrappers/<game>/...          # twenty_forty_eight, candy_crush, tetris, super_mario
#       gymv/<env_id_safe>/...           # Temporal_<Title>-v0/...
#     _logs/<model_tag>__<envw|gymv>__<env>.log
#     _run_meta.json
#   <codebase_root>/openrouter-baselines-out/latest -> <run_id>
#
# Usage:
#
#   # Default: all 3 models × 17 games × 16 episodes, vision ON.
#   bash baselines/run_openrouter_baselines.sh
#
#   # Smoke run: 4 episodes, only 2 envs from gymv.
#   bash baselines/run_openrouter_baselines.sh --episodes 4 \
#       --gymv Temporal/Airstriker-v0 Temporal/Columns-v0 --skip_envwrappers
#
#   # Skip super_mario (no orak-mario conda env on this box):
#   bash baselines/run_openrouter_baselines.sh --skip_mario
#
#   # Drop a model from the sweep:
#   bash baselines/run_openrouter_baselines.sh --models claude gemini
#
#   # Resume an interrupted run (output dir reused, finished episodes skipped):
#   bash baselines/run_openrouter_baselines.sh --run_id myrun --resume
#
# Flags:
#   --models <list>          subset of {claude, gemini, qwen, qwen35-9b,
#                            qwen35-35b-a3b} (default: claude gemini qwen).
#                            See baselines/run_qwen_api_baselines.sh for a
#                            wrapper preset to the two qwen3.5-* tags.
#   --episodes N             episodes per (model × env) combo (default: 16)
#   --max_parallel N         concurrent jobs cap (default: 16)
#   --max_steps_envw N       per-episode step cap for env_wrappers (default: per-game default)
#   --max_steps_gymv N       per-episode step cap for gym-v (default: 60)
#   --frame_skip_gymv N      emulator frames per agent step on gym-v (default: 1; recommended: 8)
#   --envwrappers <g>...     restrict env_wrappers games (default: 2048, candy_crush, tetris)
#   --gymv <id>...           restrict gym-v env ids (default: 8 retained Temporal/* envs;
#                            see baselines/README.md § "Gym-V benchmark scope")
#   --skip_envwrappers       skip the env_wrappers backend entirely
#   --skip_gymv              skip the gym-v backend entirely
#   --skip_mario             drop super_mario from env_wrappers (default: included)
#   --no_vision              disable VLM vision call (text-only schema)
#   --save_frames            persist PNG frames sent to the VLM
#   --resume                 resume previously-started run (skip done episodes)
#   --run_id <id>            override timestamped run id
#   --output_dir <path>      base output dir (default: <codebase_root>/openrouter-baselines-out)
#   --conda_main <name>      main conda env (default: game-ai-agent)
#   --conda_orak <name>      mario conda env (default: orak-mario)
#   --temperature_action F   sampling temp for action call (default: 0.4)
#   --temperature_schema F   sampling temp for schema call  (default: 0.2)
#   --base_url <url>         override OpenRouter URL (default: https://openrouter.ai/api/v1)
#   --verbose | -v           pass through to the python backends
#
# Optional env vars:
#   OPENROUTER_API_KEY  — auth token. Auto-loaded from <workspace>/api_keys.py
#                         by the python bootstrap (see _bootstrap_api_keys_from_file
#                         in cold_start/generate_cold_start_actor*.py).

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
PY_ENVW="${CODEBASE_ROOT}/cold_start/generate_cold_start_actor.py"
PY_GYMV="${CODEBASE_ROOT}/cold_start/generate_cold_start_actor_gymv.py"

# ── Defaults ──────────────────────────────────────────────────────────────
EPISODES_DEFAULT=16
MAX_PARALLEL_DEFAULT=16
MAX_STEPS_GYMV_DEFAULT=60
FRAME_SKIP_GYMV_DEFAULT=1
TEMP_ACTION_DEFAULT=0.4
TEMP_SCHEMA_DEFAULT=0.2
OPENROUTER_BASE_URL_DEFAULT="https://openrouter.ai/api/v1"

# Slugs match baselines/README.md and the existing run_*_baseline.sh scripts.
#
# NOTE on per-model vision capability:
#   * Claude / Gemini are multimodal and run with vision ON by default.
#   * qwen/qwen3-max is text-only on OpenRouter (404 "No endpoints found that
#     support image input"). It also has NO dedicated "thinking" mode, which
#     means it accepts the strict tool_choice={"type":"function",...} payload
#     the actor uses. So the qwen leg is dispatched with --no_vision automatically:
#     the canonical heuristic schema (from temporal_visual_grounding /
#     env_wrappers visual utils) feeds the action call directly, no VLM round-trip.
MODEL_CLAUDE_SLUG="${MODEL_CLAUDE:-anthropic/claude-4.6-sonnet-20260217}"
MODEL_GEMINI_SLUG="${MODEL_GEMINI:-google/gemini-3.1-pro-preview}"
# OpenRouter-hosted Qwen3.5-{9B, 35B-A3B} — same model IDs as the vLLM
# self-host baseline (Qwen/Qwen3.5-9B + Qwen/Qwen3.5-35B-A3B), but routed
# through the OpenRouter inference endpoint so no GPUs are needed. Both
# slugs are multimodal (text + image), 262K-token ctx as of 2026-04-30.
# Pricing (USD per 1M tokens): 9B $0.10/$0.15 in/out; 35B $0.16/$1.30.
# Used by the dedicated wrapper baselines/run_qwen_api_baselines.sh.
MODEL_QWEN35_9B_SLUG="${MODEL_QWEN35_9B:-qwen/qwen3.5-9b}"
MODEL_QWEN35_35B_SLUG="${MODEL_QWEN35_35B:-qwen/qwen3.5-35b-a3b}"
# Default: Qwen3.5-Plus (Apr 2026 multimodal flagship — text + image + video,
# 1M context). Alibaba positions it as Qwen3-Max-class on text + GUI-agent-tuned
# multimodal. Set MODEL_QWEN=qwen/qwen3-max + MODEL_QWEN_VISION=0 to revert to
# the text-only Qwen3-Max (game pixels are then bypassed via the heuristic
# schema fallback in temporal_visual_grounding / env_wrappers utils).
# Qwen3-VL-235B-A22B-Instruct: multimodal Qwen3 flagship in Instruct (non-
# thinking) variant.  We deliberately pick the Instruct over the Thinking
# variant because OpenRouter strips ``extra_body.enable_thinking=False``
# before forwarding to Alibaba, so Qwen3.5-Plus / qwen3-vl-*-Thinking models
# reject the strict ``tool_choice={"type":"function",...}`` payload our
# actor pipeline depends on (HTTP 400 "InvalidParameter ... in thinking
# mode").  Instruct variants have no thinking layer and accept strict
# tool_choice cleanly.
#
# Override examples:
#   MODEL_QWEN=qwen/qwen3-vl-30b-a3b-instruct   # cheaper, ~1/4 the cost
#   MODEL_QWEN=qwen/qwen3.5-plus-20260420       # NEEDS tool_choice=auto, see _chat_completion
MODEL_QWEN_SLUG="${MODEL_QWEN:-qwen/qwen3-vl-235b-a22b-instruct}"
MODEL_QWEN_VISION="${MODEL_QWEN_VISION:-1}"

ENVWRAPPERS_DEFAULT=(twenty_forty_eight candy_crush tetris)

# Gym-V benchmark scope: 8 of the 13 registered Temporal envs.
#
# Dropped 2026-05-03 after the frame_skip=8 sweep showed all six tested
# backbones (GPT-5.4, Claude-4.6, Gemini-3.1-Pro, Qwen3-VL-235B,
# Qwen3.5-9B, Qwen3.5-35B-A3B) ≤8 % per-episode success on:
#   CastleOfIllusion, CastlevaniaBloodlines, GoldenAxe, KidChameleon,
#   MortalKombatII.
# Reward functions and save states are sound — the failure mode is task
# design (precise platforming / combat-block timing / multi-step combos)
# that does not emit reward density compatible with single-shot LLM
# rollouts at the current 640-frame budget. To run the full 13-game
# registry pass `--gymv` with the explicit list. See README §
# "Gym-V benchmark scope" for the full decision log.
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

CONDA_MAIN_DEFAULT="game-ai-agent"
CONDA_ORAK_DEFAULT="orak-mario"

# ── State / parsed args ───────────────────────────────────────────────────
EPISODES="$EPISODES_DEFAULT"
MAX_PARALLEL="$MAX_PARALLEL_DEFAULT"
MAX_STEPS_ENVW=""
MAX_STEPS_GYMV="$MAX_STEPS_GYMV_DEFAULT"
FRAME_SKIP_GYMV="$FRAME_SKIP_GYMV_DEFAULT"
TEMP_ACTION="$TEMP_ACTION_DEFAULT"
TEMP_SCHEMA="$TEMP_SCHEMA_DEFAULT"
INCLUDE_ENVWRAPPERS=1
INCLUDE_GYMV=1
INCLUDE_MARIO=1
USE_VISION=1
SAVE_FRAMES=0
RESUME=0
VERBOSE=0
RUN_ID=""
BASE_DIR_OVERRIDE=""
CONDA_MAIN="$CONDA_MAIN_DEFAULT"
CONDA_ORAK="$CONDA_ORAK_DEFAULT"
OPENROUTER_BASE_URL="$OPENROUTER_BASE_URL_DEFAULT"

declare -a MODEL_TAGS=("claude" "gemini" "qwen")
declare -a ENVWRAPPERS=()
declare -a GYMV=()

# ── Parse args ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --models)
            shift
            MODEL_TAGS=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                MODEL_TAGS+=("$1"); shift
            done ;;
        --episodes)
            shift; EPISODES="${1:-$EPISODES_DEFAULT}"; shift ;;
        --max_parallel|--max-parallel)
            shift; MAX_PARALLEL="${1:-$MAX_PARALLEL_DEFAULT}"; shift ;;
        --max_steps_envw|--max-steps-envw)
            shift; MAX_STEPS_ENVW="${1:-}"; shift ;;
        --max_steps_gymv|--max-steps-gymv)
            shift; MAX_STEPS_GYMV="${1:-$MAX_STEPS_GYMV_DEFAULT}"; shift ;;
        --frame_skip_gymv|--frame-skip-gymv)
            shift; FRAME_SKIP_GYMV="${1:-$FRAME_SKIP_GYMV_DEFAULT}"; shift ;;
        --temperature_action|--temperature-action)
            shift; TEMP_ACTION="${1:-$TEMP_ACTION_DEFAULT}"; shift ;;
        --temperature_schema|--temperature-schema)
            shift; TEMP_SCHEMA="${1:-$TEMP_SCHEMA_DEFAULT}"; shift ;;
        --envwrappers)
            shift
            ENVWRAPPERS=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                ENVWRAPPERS+=("$1"); shift
            done ;;
        --gymv)
            shift
            GYMV=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                GYMV+=("$1"); shift
            done ;;
        --skip_envwrappers|--skip-envwrappers)
            INCLUDE_ENVWRAPPERS=0; shift ;;
        --skip_gymv|--skip-gymv)
            INCLUDE_GYMV=0; shift ;;
        --skip_mario|--skip-mario)
            INCLUDE_MARIO=0; shift ;;
        --include_mario|--include-mario)
            INCLUDE_MARIO=1; shift ;;
        --vision)
            USE_VISION=1; shift ;;
        --no_vision|--no-vision)
            USE_VISION=0; shift ;;
        --save_frames|--save-frames)
            SAVE_FRAMES=1; shift ;;
        --resume)
            RESUME=1; shift ;;
        --verbose|-v)
            VERBOSE=1; shift ;;
        --run_id|--run-id)
            shift; RUN_ID="${1:-}"; shift ;;
        --output_dir|--output-dir)
            shift; BASE_DIR_OVERRIDE="${1:-}"; shift ;;
        --conda_main|--conda-main)
            shift; CONDA_MAIN="${1:-$CONDA_MAIN_DEFAULT}"; shift ;;
        --conda_orak|--conda-orak)
            shift; CONDA_ORAK="${1:-$CONDA_ORAK_DEFAULT}"; shift ;;
        --base_url|--base-url)
            shift; OPENROUTER_BASE_URL="${1:-$OPENROUTER_BASE_URL_DEFAULT}"; shift ;;
        -h|--help)
            sed -n '1,/^set -uo pipefail/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "        Run: bash $0 --help" >&2
            exit 2 ;;
    esac
done

[ ${#ENVWRAPPERS[@]} -eq 0 ] && ENVWRAPPERS=("${ENVWRAPPERS_DEFAULT[@]}")
[ ${#GYMV[@]} -eq 0 ]         && GYMV=("${GYMV_DEFAULT[@]}")
[ -z "$RUN_ID" ]              && RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"

BASE_DIR="${BASE_DIR_OVERRIDE:-${CODEBASE_ROOT}/openrouter-baselines-out}"
RUN_DIR="${BASE_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/_logs"
META_FILE="${RUN_DIR}/_run_meta.json"
mkdir -p "$LOG_DIR"

# ── Conda ─────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not on PATH. Cannot dispatch jobs." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

ENV_LIST="$(conda env list | awk '$1 !~ /^#/ {print $1}')"
has_env() { printf '%s\n' "${ENV_LIST}" | grep -qx "$1"; }

if ! has_env "$CONDA_MAIN"; then
    echo "[ERROR] conda env '$CONDA_MAIN' not found. Available:"
    printf '%s\n' "${ENV_LIST}" | sed 's/^/  - /'
    exit 1
fi
if [ "$INCLUDE_MARIO" -eq 1 ] && ! has_env "$CONDA_ORAK"; then
    echo "[WARN]  conda env '$CONDA_ORAK' not found — super_mario will be skipped"
    INCLUDE_MARIO=0
fi

# ── API key (the python bootstrap also loads <workspace>/api_keys.py) ─────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    if [ -f "${WORKSPACE_ROOT}/api_keys.py" ]; then
        OPENROUTER_API_KEY="$(
            python3 - <<PYEOF
import sys, importlib.util
spec = importlib.util.spec_from_file_location("k", "${WORKSPACE_ROOT}/api_keys.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(getattr(m, "openrouter_api_key", "") or "")
PYEOF
        )"
        export OPENROUTER_API_KEY
    fi
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[ERROR] OPENROUTER_API_KEY is not set and no openrouter_api_key found in api_keys.py" >&2
    exit 1
fi

# ── Headless rendering / DISPLAY (stable-retro & Genesis ROMs need SDL) ───
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

# ── PYTHONPATH (codebase + sibling repos when present) ────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}")
[ -n "${GAMINGAGENT_ROOT}" ] && [ -d "${GAMINGAGENT_ROOT}" ] && PYPATH_ADD+=("${GAMINGAGENT_ROOT}")
[ -n "${ORAK_SRC}" ]         && [ -d "${ORAK_SRC}" ]         && PYPATH_ADD+=("${ORAK_SRC}")
[ -n "${GYMV_ROOT}" ]        && [ -d "${GYMV_ROOT}" ]        && PYPATH_ADD+=("${GYMV_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Resolve model entries (tag|slug|vision_supported) ────────────────────
# vision_supported=1 → multimodal; vision_supported=0 → text-only, force --no_vision.
declare -a MODELS=()
for tag in "${MODEL_TAGS[@]}"; do
    case "$tag" in
        claude|claude-4.6|claude-sonnet-4.6|sonnet)
            MODELS+=("claude|${MODEL_CLAUDE_SLUG}|1") ;;
        gemini|gemini-3.1|gemini-3.1-pro|pro)
            MODELS+=("gemini|${MODEL_GEMINI_SLUG}|1") ;;
        qwen|qwen3|qwen3-max|qwen3.5-plus|max|plus)
            MODELS+=("qwen|${MODEL_QWEN_SLUG}|${MODEL_QWEN_VISION}") ;;
        qwen35-9b|qwen3.5-9b|qwen-3.5-9b|9b)
            # OpenRouter-hosted Qwen/Qwen3.5-9B (multimodal). Mirrors
            # the vLLM-hosted 9B leg of run_qwen_vllm_baselines.sh.
            MODELS+=("qwen3.5-9b|${MODEL_QWEN35_9B_SLUG}|1") ;;
        qwen35-35b-a3b|qwen3.5-35b-a3b|qwen-3.5-35b-a3b|35b|35b-a3b)
            # OpenRouter-hosted Qwen/Qwen3.5-35B-A3B (multimodal).
            # Mirrors the vLLM-hosted 35B-A3B leg.
            MODELS+=("qwen3.5-35b-a3b|${MODEL_QWEN35_35B_SLUG}|1") ;;
        *)
            echo "[ERROR] Unknown model tag '$tag' (allowed: claude, gemini, qwen, qwen35-9b, qwen35-35b-a3b)" >&2
            exit 2 ;;
    esac
done

sanitize() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'; }

# ── Banner ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  OpenRouter baselines (env_wrappers + gym-v)"
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Run dir:       $RUN_DIR"
echo "  Logs:          $LOG_DIR/"
echo "  Episodes/job:  $EPISODES"
echo "  Concurrency:   $MAX_PARALLEL"
echo "  Vision call:   $([ "$USE_VISION" -eq 1 ] && echo ON || echo OFF)"
echo "  Save frames:   $([ "$SAVE_FRAMES" -eq 1 ] && echo ON || echo OFF)"
echo "  Resume:        $([ "$RESUME" -eq 1 ] && echo ON || echo OFF)"
echo "  Conda main:    $CONDA_MAIN"
[ "$INCLUDE_MARIO" -eq 1 ] && echo "  Conda orak:    $CONDA_ORAK"
echo "  Endpoint:      $OPENROUTER_BASE_URL"
echo "  API key:       ${OPENROUTER_API_KEY:0:12}... (OpenRouter)"
echo
echo "  Backbones:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag slug vis <<<"$entry"
    eff_vis="$([ "$USE_VISION" -eq 1 ] && [ "$vis" -eq 1 ] && echo ON || echo OFF)"
    printf "    [OK]   %-8s %-45s vision=%s\n" "$tag" "$slug" "$eff_vis"
done
echo
if [ "$INCLUDE_ENVWRAPPERS" -eq 1 ]; then
    mario_str=""
    [ "$INCLUDE_MARIO" -eq 1 ] && mario_str=" super_mario"
    echo "  env_wrappers:  ${ENVWRAPPERS[*]}${mario_str}"
else
    echo "  env_wrappers:  (skipped)"
fi
if [ "$INCLUDE_GYMV" -eq 1 ]; then
    echo "  gym-v:         ${#GYMV[@]} env(s)"
    for e in "${GYMV[@]}"; do echo "                 - $e"; done
else
    echo "  gym-v:         (skipped)"
fi
echo "============================================================"

# ── Per-job dispatcher ────────────────────────────────────────────────────
run_envwrapper_job() {
    local model_tag=$1 slug=$2 model_vision=$3 game=$4
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local out_dir="${RUN_DIR}/${model_safe}/env_wrappers/${game}"
    local logfile="${LOG_DIR}/${model_safe}__envw__${game}.log"
    mkdir -p "$out_dir"

    local conda_env="$CONDA_MAIN"
    [ "$game" = "super_mario" ] && conda_env="$CONDA_ORAK"

    # Effective vision = global flag AND per-model capability.
    local extra=()
    if [ "$USE_VISION" -eq 0 ] || [ "$model_vision" -eq 0 ]; then
        extra+=(--no_vision)
    fi
    [ "$SAVE_FRAMES" -eq 1 ] && extra+=(--save_frames)
    [ "$RESUME" -eq 1 ]      && extra+=(--resume)
    [ "$VERBOSE" -eq 1 ]     && extra+=(--verbose)
    [ -n "$MAX_STEPS_ENVW" ] && extra+=(--max_steps "$MAX_STEPS_ENVW")

    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$conda_env" --no-capture-output \
        python3 "$PY_ENVW" \
            --games "$game" \
            --episodes "$EPISODES" \
            --model "$slug" \
            --api_key "$OPENROUTER_API_KEY" \
            --base_url "$OPENROUTER_BASE_URL" \
            --temperature_action "$TEMP_ACTION" \
            --temperature_schema "$TEMP_SCHEMA" \
            --output_dir "$out_dir" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

run_gymv_job() {
    local model_tag=$1 slug=$2 model_vision=$3 env_id=$4
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local env_safe;   env_safe="$(sanitize "$env_id")"
    local out_dir="${RUN_DIR}/${model_safe}/gymv"
    local logfile="${LOG_DIR}/${model_safe}__gymv__${env_safe}.log"
    mkdir -p "$out_dir"

    local extra=()
    if [ "$USE_VISION" -eq 0 ] || [ "$model_vision" -eq 0 ]; then
        extra+=(--no_vision)
    fi
    [ "$SAVE_FRAMES" -eq 1 ] && extra+=(--save_frames)
    [ "$RESUME" -eq 1 ]      && extra+=(--resume)
    [ "$VERBOSE" -eq 1 ]     && extra+=(--verbose)

    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$CONDA_MAIN" --no-capture-output \
        python3 "$PY_GYMV" \
            --envs "$env_id" \
            --episodes "$EPISODES" \
            --max_steps "$MAX_STEPS_GYMV" \
            --frame_skip "$FRAME_SKIP_GYMV" \
            --model "$slug" \
            --api_key "$OPENROUTER_API_KEY" \
            --base_url "$OPENROUTER_BASE_URL" \
            --temperature_action "$TEMP_ACTION" \
            --temperature_schema "$TEMP_SCHEMA" \
            --output_dir "$out_dir" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

# ── Build job list (kind|tag|slug|vision|target) ──────────────────────────
declare -a JOBS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag slug vis <<<"$entry"
    if [ "$INCLUDE_ENVWRAPPERS" -eq 1 ]; then
        for g in "${ENVWRAPPERS[@]}"; do
            JOBS+=("envw|${tag}|${slug}|${vis}|${g}")
        done
        if [ "$INCLUDE_MARIO" -eq 1 ]; then
            JOBS+=("envw|${tag}|${slug}|${vis}|super_mario")
        fi
    fi
    if [ "$INCLUDE_GYMV" -eq 1 ]; then
        for e in "${GYMV[@]}"; do
            JOBS+=("gymv|${tag}|${slug}|${vis}|${e}")
        done
    fi
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "[ERROR] No jobs to run (after applying skip flags)." >&2
    exit 2
fi

echo
echo "Total jobs to dispatch: ${#JOBS[@]}"
for spec in "${JOBS[@]}"; do
    IFS='|' read -r kind tag _ vis target <<<"$spec"
    eff_vis="$([ "$USE_VISION" -eq 1 ] && [ "$vis" -eq 1 ] && echo on || echo off)"
    printf "  - %-5s %-8s vision=%-3s %-32s\n" "$kind" "$tag" "$eff_vis" "$target"
done
echo
echo "Live tail any of:"
for spec in "${JOBS[@]}"; do
    IFS='|' read -r kind tag _ _ target <<<"$spec"
    model_safe="$(sanitize "$tag")"
    target_safe="$(sanitize "$target")"
    echo "  tail -f ${LOG_DIR}/${model_safe}__${kind}__${target_safe}.log"
done
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
    while [ "${#INFLIGHT[@]}" -ge "$MAX_PARALLEL" ]; do
        wait_one
    done

    IFS='|' read -r kind tag slug vis target <<<"$spec"
    job_id="${kind}|${tag}|${target}"

    if [ "$kind" = "envw" ]; then
        run_envwrapper_job "$tag" "$slug" "$vis" "$target" &
    else
        run_gymv_job       "$tag" "$slug" "$vis" "$target" &
    fi
    PIDS["$job_id"]=$!
    INFLIGHT+=("$job_id")
    printf "[START] %s   pid=%d   ts=%s\n" \
        "$job_id" "${PIDS[$job_id]}" "$(date +%H:%M:%S)"
done

while [ "${#INFLIGHT[@]}" -gt 0 ]; do
    wait_one
done
END_TS="$(date +%s)"
ELAPSED=$((END_TS - START_TS))

# ── Update `latest` symlink ───────────────────────────────────────────────
ln -sfn "$RUN_ID" "${BASE_DIR}/latest" 2>/dev/null || true

# ── Write meta JSON ───────────────────────────────────────────────────────
{
    printf '{\n'
    printf '  "run_id": "%s",\n'              "$RUN_ID"
    printf '  "started_at_unix": %s,\n'       "$START_TS"
    printf '  "ended_at_unix": %s,\n'         "$END_TS"
    printf '  "elapsed_seconds": %s,\n'       "$ELAPSED"
    printf '  "episodes_per_job": %s,\n'      "$EPISODES"
    printf '  "max_parallel": %s,\n'          "$MAX_PARALLEL"
    printf '  "use_vision": %s,\n'            "$([ "$USE_VISION" -eq 1 ] && echo true || echo false)"
    printf '  "include_envwrappers": %s,\n'   "$([ "$INCLUDE_ENVWRAPPERS" -eq 1 ] && echo true || echo false)"
    printf '  "include_gymv": %s,\n'          "$([ "$INCLUDE_GYMV" -eq 1 ] && echo true || echo false)"
    printf '  "include_mario": %s,\n'         "$([ "$INCLUDE_MARIO" -eq 1 ] && echo true || echo false)"
    printf '  "base_url": "%s",\n'            "$OPENROUTER_BASE_URL"
    printf '  "models": [\n'
    first=1
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag slug vis <<<"$entry"
        [ $first -eq 1 ] || printf ',\n'
        printf '    {"tag": "%s", "slug": "%s", "vision_supported": %s}' \
            "$tag" "$slug" "$([ "$vis" -eq 1 ] && echo true || echo false)"
        first=0
    done
    printf '\n  ],\n'
    printf '  "jobs": [\n'
    first=1
    for spec in "${JOBS[@]}"; do
        IFS='|' read -r kind tag _ vis target <<<"$spec"
        job_id="${kind}|${tag}|${target}"
        rc="${RC[$job_id]:-null}"
        eff_vis="$([ "$USE_VISION" -eq 1 ] && [ "$vis" -eq 1 ] && echo true || echo false)"
        [ $first -eq 1 ] || printf ',\n'
        printf '    {"kind": "%s", "tag": "%s", "target": "%s", "vision": %s, "rc": %s}' \
            "$kind" "$tag" "$target" "$eff_vis" "$rc"
        first=0
    done
    printf '\n  ]\n'
    printf '}\n'
} > "$META_FILE"

# ── Summary ───────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  OpenRouter baselines — done"
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Elapsed:       ${ELAPSED}s"
echo "  Run dir:       $RUN_DIR/"
echo "  Latest:        $BASE_DIR/latest -> $RUN_ID"
echo "  Meta:          $META_FILE"
echo

ANY_OK=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r kind tag _ _ target <<<"$spec"
    job_id="${kind}|${tag}|${target}"
    rc="${RC[$job_id]:-?}"
    model_safe="$(sanitize "$tag")"
    target_safe="$(sanitize "$target")"

    if [ "$kind" = "envw" ]; then
        out="${RUN_DIR}/${model_safe}/env_wrappers/${target}"
    else
        out="${RUN_DIR}/${model_safe}/gymv/${target_safe}"
    fi
    count=0
    if [ -d "$out" ]; then
        count=$(find "$out" -maxdepth 3 -name 'episode_*.json' \
                ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
    fi
    printf "  %-5s %-8s %-32s rc=%-3s episodes=%-3s\n" \
        "$kind" "$tag" "$target" "$rc" "$count"
    [ "$rc" = "0" ] && ANY_OK=1
done
echo "============================================================"

if [ "$ANY_OK" -eq 0 ]; then
    exit 1
fi
exit 0
