#!/usr/bin/env bash
#
# baselines/run_openrouter_transfer_baselines.sh — Frontier API baselines
# on the four TRANSFER-TARGET benchmark families covered by cold-start.
# Sibling script to run_openrouter_baselines.sh (which handles gymv +
# env_wrappers).  Together they give: 5 families × 3 models = 15 sweeps.
#
# Backbones (all routed via OpenRouter, identical convention to
# run_openrouter_baselines.sh):
#
#     anthropic/claude-4.6-sonnet-20260217   (Sonnet 4.6, multimodal)*
#     google/gemini-3.1-pro-preview          (Gemini 3.1 Pro, multimodal)
#     qwen/qwen3-max                         (Qwen3 Max, text-only on OR)
#
#   * NOTE: Claude 4.7 Sonnet does NOT exist as of Apr 2026 — only
#     Claude Opus 4.7.  Default stays on Sonnet 4.6 (cost-effective and
#     widely-tested).  To use the actual 4.7 generation:
#
#         OPENROUTER_CLAUDE_MODEL=anthropic/claude-opus-4.7 \
#             bash baselines/run_openrouter_transfer_baselines.sh
#
#     Opus 4.7 is ~3-5× the cost of Sonnet 4.6 per output token.
#
# Task families (lean-plan IDs from cold_start/evaluation_dataset/pool/):
#
#     browsergym  -> MiniWoB++ + AssistantBench   (305 tasks, 4 shards)
#     osworld     -> 10 desktop domains            (120 stratified tasks; cut from 250 on 2026-05-03)
#     vr_image    -> VisualToolBench + TIR-Bench   (~2 × 1000 samples)
#     vr_video    -> Video-Holmes + SIV-Bench      (~2 × 1000 samples)
#
# Frames / screenshots are NEVER persisted to disk:
#   - BrowserGym: --save_frames is omitted
#   - OSWorld:    --no_save_frames is forwarded
#   - VR (image/video): --save_frames is omitted
# The pipelines still feed pixels to the VLM in-memory (mandatory for
# vision-capable models — Claude / Gemini); for Qwen3-Max we force
# --no_vision so the launchers feed the canonical heuristic schema only.
#
# Output layout:
#
#   <codebase_root>/openrouter-transfer-baselines-out/<run_id>/
#     <model_tag>/                     # claude, gemini, qwen
#       browsergym/<safe_task_id>/...
#       osworld/<run_id>/<domain>/...
#       vr_image/<benchmark>/...
#       vr_video/<benchmark>/...
#     _logs/<model_tag>__<family>.log
#     _run_meta.json
#   <codebase_root>/openrouter-transfer-baselines-out/latest -> <run_id>
#
# Concurrency model (SLA target: ≤ 12 h end-to-end):
#
#   - Across (model x family) jobs: --max_parallel 8 (default).
#     12 jobs total = 3 models × 4 families.  Each upstream provider
#     (Anthropic / Google / Alibaba) sees up to ~3 concurrent jobs at
#     peak, well within typical OpenRouter tier limits.
#   - Within each job: BG shards=4, OSW max_parallel=4, VR num_workers=8.
#     Lower than the local-vLLM script because OpenRouter ↔ upstream
#     RTT is 200-600 ms and rate-limit headroom is tighter.
#
# Wall-clock budget (default settings):
#   browsergym  ~1.5 h per model
#   osworld     ~4-5 h per model  (dominates wall-clock)
#   vr_image    ~30 min per model
#   vr_video    ~30 min per model
#   ────────────────────────────────────────────
#   3 models in parallel, OSW dominates -> ~5 h end-to-end
#
# Usage:
#
#   # Default: all 3 models × all 4 families.
#   bash baselines/run_openrouter_transfer_baselines.sh
#
#   # Drop a model:
#   bash baselines/run_openrouter_transfer_baselines.sh --models claude gemini
#
#   # Restrict to a subset of families:
#   bash baselines/run_openrouter_transfer_baselines.sh --families vr_image vr_video
#
#   # Resume / smaller smoke run:
#   bash baselines/run_openrouter_transfer_baselines.sh \
#       --models gemini --families vr_image \
#       --vr_num_test_cases 5
#
# Wrapper-only flags:
#   --models <list>          subset of {claude, gemini, qwen} (default: all 3)
#   --families <list>        subset of {browsergym, osworld, vr_image,
#                            vr_video} (default: all 4)
#   --max_parallel N         (model x family) jobs in flight (default: 8)
#   # ── Per-family knobs (defaults match the cold-start lean plan) ──
#   --bg_shards N            BrowserGym shards   (default: 4)
#   --bg_max_steps N         BrowserGym per-task step cap (default: 16)
#   --osw_max_parallel N     OSWorld concurrent VMs (default: 4)
#   --osw_episodes N         OSWorld episodes per task (default: 1)
#   --osw_max_steps N        OSWorld per-episode step cap (default: 50)
#   --vr_num_test_cases N    VR sample cap per benchmark (default: 1000)
#   --vr_num_workers N       VR threadpool size (default: 8)
#   --vr_num_frames N        VR video frames/clip (default: 8)
#   --vr_judge               Enable LLM-as-judge for VTB / TIR-Bench
#   # ── Misc ────────────────────────────────────────────────────────
#   --pool_dir <path>        Directory holding the lean-plan manifests
#                            (default: cold_start/evaluation_dataset/pool)
#   --output_dir <path>      Override base output dir
#   --run_id <id>            Override timestamped run id
#   --conda_main <name>      Conda env for vr / general (default: game-ai-agent)
#   --conda_browsergym <n>   Conda env for BrowserGym   (default: browsergym)
#   --conda_osworld <n>      Conda env for OSWorld      (default: osworld)
#   --base_url <url>         OpenRouter URL (default: https://openrouter.ai/api/v1)
#   --resume                 Pass --resume to BrowserGym
#   --verbose | -v           Pass --verbose to all backends
#
# Optional env vars:
#   OPENROUTER_API_KEY        Auto-loaded from <workspace>/api_keys.py
#   OPENROUTER_CLAUDE_MODEL   Override claude slug (default: anthropic/claude-4.6-sonnet-20260217)
#   OPENROUTER_GEMINI_MODEL   Override gemini slug (default: google/gemini-3.1-pro-preview)
#   OPENROUTER_QWEN_MODEL     Override qwen   slug (default: qwen/qwen3-max)
#   OPENROUTER_QWEN_VISION    "1" to mark the qwen slug as multimodal
#                             (default: "0" because qwen3-max is text-only).
#                             Set to "1" when overriding to a multimodal slug
#                             like qwen/qwen3.5-plus-20260420 — this drops
#                             the auto --no_vision and unlocks the OSWorld leg.

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# ── Resolve paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"

PY_BG_SHARD="${CODEBASE_ROOT}/cold_start/run_coldstart_actor_browsergym_shard.sh"
PY_OSW="${CODEBASE_ROOT}/cold_start/run_coldstart_actor_osworld_all.sh"
PY_VR="${CODEBASE_ROOT}/cold_start/generate_cold_start_actor_visual_reasoning.py"

# ── Defaults ──────────────────────────────────────────────────────────────
MAX_PARALLEL_DEFAULT=8

BG_SHARDS_DEFAULT=4
BG_MAX_STEPS_DEFAULT=16
OSW_MAX_PARALLEL_DEFAULT=4
OSW_EPISODES_DEFAULT=1
OSW_MAX_STEPS_DEFAULT=50
VR_NUM_TEST_CASES_DEFAULT=1000
VR_NUM_WORKERS_DEFAULT=8
VR_NUM_FRAMES_DEFAULT=8

CONDA_MAIN_DEFAULT="game-ai-agent"
CONDA_BG_DEFAULT="browsergym"
CONDA_OSW_DEFAULT="osworld"

POOL_DIR_DEFAULT="${CODEBASE_ROOT}/cold_start/evaluation_dataset/pool"

OPENROUTER_BASE_URL_DEFAULT="https://openrouter.ai/api/v1"
MODEL_CLAUDE_SLUG="${OPENROUTER_CLAUDE_MODEL:-anthropic/claude-4.6-sonnet-20260217}"
MODEL_GEMINI_SLUG="${OPENROUTER_GEMINI_MODEL:-google/gemini-3.1-pro-preview}"
# Qwen3-VL-235B-A22B-Instruct: multimodal Qwen3 flagship in Instruct (non-
# thinking) variant.  Picked over Qwen3.5-Plus / Qwen3-VL-Thinking because
# OpenRouter strips ``extra_body.enable_thinking=False`` before forwarding
# to Alibaba — thinking-mode models then reject our strict
# ``tool_choice={"type":"function",...}`` payload with HTTP 400
# (``InvalidParameter ... in thinking mode``).  Instruct variants have no
# thinking layer and accept strict tool_choice cleanly.
#
# Smaller drop-in alternatives that also work:
#   OPENROUTER_QWEN_MODEL=qwen/qwen3-vl-30b-a3b-instruct   # cheaper, ~1/4 the cost
#   OPENROUTER_QWEN_MODEL=qwen/qwen-vl-max                 # legacy 7K-ctx, weak
MODEL_QWEN_SLUG="${OPENROUTER_QWEN_MODEL:-qwen/qwen3-vl-235b-a22b-instruct}"
MODEL_QWEN_VISION="${OPENROUTER_QWEN_VISION:-1}"

# ── State / parsed args ───────────────────────────────────────────────────
MAX_PARALLEL="$MAX_PARALLEL_DEFAULT"
BG_SHARDS="$BG_SHARDS_DEFAULT"
BG_MAX_STEPS="$BG_MAX_STEPS_DEFAULT"
OSW_MAX_PARALLEL="$OSW_MAX_PARALLEL_DEFAULT"
OSW_EPISODES="$OSW_EPISODES_DEFAULT"
OSW_MAX_STEPS="$OSW_MAX_STEPS_DEFAULT"
VR_NUM_TEST_CASES="$VR_NUM_TEST_CASES_DEFAULT"
VR_NUM_WORKERS="$VR_NUM_WORKERS_DEFAULT"
VR_NUM_FRAMES="$VR_NUM_FRAMES_DEFAULT"
VR_JUDGE=0
RESUME=0
VERBOSE=0
RUN_ID=""
BASE_DIR_OVERRIDE=""
POOL_DIR="$POOL_DIR_DEFAULT"
CONDA_MAIN="$CONDA_MAIN_DEFAULT"
CONDA_BG="$CONDA_BG_DEFAULT"
CONDA_OSW="$CONDA_OSW_DEFAULT"
OPENROUTER_BASE_URL="$OPENROUTER_BASE_URL_DEFAULT"

declare -a MODEL_TAGS=("claude" "gemini" "qwen")
declare -a FAMILIES=("browsergym" "osworld" "vr_image" "vr_video")

# ── Parse args ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --models)
            shift
            MODEL_TAGS=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                MODEL_TAGS+=("$1"); shift
            done ;;
        --families)
            shift
            FAMILIES=()
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                FAMILIES+=("$1"); shift
            done ;;
        --max_parallel|--max-parallel)
            shift; MAX_PARALLEL="${1:-$MAX_PARALLEL_DEFAULT}"; shift ;;
        --bg_shards|--bg-shards)
            shift; BG_SHARDS="${1:-$BG_SHARDS_DEFAULT}"; shift ;;
        --bg_max_steps|--bg-max-steps)
            shift; BG_MAX_STEPS="${1:-$BG_MAX_STEPS_DEFAULT}"; shift ;;
        --osw_max_parallel|--osw-max-parallel)
            shift; OSW_MAX_PARALLEL="${1:-$OSW_MAX_PARALLEL_DEFAULT}"; shift ;;
        --osw_episodes|--osw-episodes)
            shift; OSW_EPISODES="${1:-$OSW_EPISODES_DEFAULT}"; shift ;;
        --osw_max_steps|--osw-max-steps)
            shift; OSW_MAX_STEPS="${1:-$OSW_MAX_STEPS_DEFAULT}"; shift ;;
        --vr_num_test_cases|--vr-num-test-cases)
            shift; VR_NUM_TEST_CASES="${1:-$VR_NUM_TEST_CASES_DEFAULT}"; shift ;;
        --vr_num_workers|--vr-num-workers)
            shift; VR_NUM_WORKERS="${1:-$VR_NUM_WORKERS_DEFAULT}"; shift ;;
        --vr_num_frames|--vr-num-frames)
            shift; VR_NUM_FRAMES="${1:-$VR_NUM_FRAMES_DEFAULT}"; shift ;;
        --vr_judge|--vr-judge|--judge)
            VR_JUDGE=1; shift ;;
        --pool_dir|--pool-dir)
            shift; POOL_DIR="${1:-$POOL_DIR_DEFAULT}"; shift ;;
        --output_dir|--output-dir)
            shift; BASE_DIR_OVERRIDE="${1:-}"; shift ;;
        --run_id|--run-id)
            shift; RUN_ID="${1:-}"; shift ;;
        --conda_main|--conda-main)
            shift; CONDA_MAIN="${1:-$CONDA_MAIN_DEFAULT}"; shift ;;
        --conda_browsergym|--conda-browsergym)
            shift; CONDA_BG="${1:-$CONDA_BG_DEFAULT}"; shift ;;
        --conda_osworld|--conda-osworld)
            shift; CONDA_OSW="${1:-$CONDA_OSW_DEFAULT}"; shift ;;
        --base_url|--base-url)
            shift; OPENROUTER_BASE_URL="${1:-$OPENROUTER_BASE_URL_DEFAULT}"; shift ;;
        --resume)   RESUME=1; shift ;;
        --verbose|-v) VERBOSE=1; shift ;;
        -h|--help)
            sed -n '1,/^set -uo pipefail/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "        Run: bash $0 --help" >&2
            exit 2 ;;
    esac
done

[ -z "$RUN_ID" ] && RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"

BASE_DIR="${BASE_DIR_OVERRIDE:-${CODEBASE_ROOT}/openrouter-transfer-baselines-out}"
RUN_DIR="${BASE_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/_logs"
META_FILE="${RUN_DIR}/_run_meta.json"
mkdir -p "$LOG_DIR"

# ── Validate families ─────────────────────────────────────────────────────
declare -A VALID_FAMILY=( [browsergym]=1 [osworld]=1 [vr_image]=1 [vr_video]=1 )
for f in "${FAMILIES[@]}"; do
    if [ -z "${VALID_FAMILY[$f]:-}" ]; then
        echo "[ERROR] Unknown family '$f'. Allowed: browsergym, osworld, vr_image, vr_video" >&2
        exit 2
    fi
done

# ── Conda ─────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not on PATH. Cannot dispatch jobs." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

ENV_LIST="$(conda env list | awk '$1 !~ /^#/ {print $1}')"
has_env() { printf '%s\n' "${ENV_LIST}" | grep -qx "$1"; }
if ! has_env "$CONDA_MAIN"; then
    echo "[ERROR] conda env '$CONDA_MAIN' not found." >&2
    printf '%s\n' "${ENV_LIST}" | sed 's/^/  - /'
    exit 1
fi

# ── API key (auto-load from <workspace>/api_keys.py) ──────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    if [ -f "${WORKSPACE_ROOT}/api_keys.py" ]; then
        OPENROUTER_API_KEY="$(
            python3 - <<PYEOF
import importlib.util
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

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Resolve model entries (tag|slug|vision_supported) ─────────────────────
# vision_supported=1 → multimodal; vision_supported=0 → text-only, force --no_vision.
declare -a MODELS=()
for tag in "${MODEL_TAGS[@]}"; do
    case "$tag" in
        claude|claude-4.6|claude-4.7|claude-sonnet-4.6|sonnet|opus)
            MODELS+=("claude|${MODEL_CLAUDE_SLUG}|1") ;;
        gemini|gemini-3.1|gemini-3.1-pro|pro)
            MODELS+=("gemini|${MODEL_GEMINI_SLUG}|1") ;;
        qwen|qwen3|qwen3-max|max|qwen3.5-plus|qwen-vl|qwen3-vl)
            # Vision flag is opt-in via OPENROUTER_QWEN_VISION=1 because the
            # default slug (qwen3-max) is text-only; setting it to 1 also
            # implies the user has overridden OPENROUTER_QWEN_MODEL to a
            # multimodal slug.
            MODELS+=("qwen|${MODEL_QWEN_SLUG}|${MODEL_QWEN_VISION}") ;;
        *)
            echo "[ERROR] Unknown model tag '$tag' (allowed: claude, gemini, qwen)" >&2
            exit 2 ;;
    esac
done

sanitize() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'; }

# ── Banner ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  OpenRouter TRANSFER baselines (browsergym + osworld + vr)"
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Run dir:       $RUN_DIR"
echo "  Logs:          $LOG_DIR/"
echo "  Pool manifests: $POOL_DIR"
echo "  OpenRouter URL: $OPENROUTER_BASE_URL"
echo "  Concurrency:   max_parallel=$MAX_PARALLEL  (jobs in flight)"
echo "  Save frames:   OFF  (no PNGs written to disk)"
echo "  Resume:        $([ "$RESUME" -eq 1 ] && echo ON || echo OFF)"
echo "  Conda main:    $CONDA_MAIN  (vr / general)"
echo "  Conda BG:      $CONDA_BG    (BrowserGym)"
echo "  Conda OSW:     $CONDA_OSW   (OSWorld)"
echo
echo "  Backbones:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag slug vis <<<"$entry"
    vstr=$([ "$vis" = "1" ] && echo vision || echo "text-only (auto --no_vision)")
    printf "    %-8s  %-40s  %s\n" "$tag" "$slug" "$vstr"
done
echo
echo "  Families:      ${FAMILIES[*]}"
echo "  Per-family knobs:"
echo "    browsergym  shards=$BG_SHARDS              max_steps=$BG_MAX_STEPS"
echo "    osworld     max_parallel=$OSW_MAX_PARALLEL          episodes=$OSW_EPISODES   max_steps=$OSW_MAX_STEPS"
echo "    vr_image    num_test_cases=$VR_NUM_TEST_CASES   num_workers=$VR_NUM_WORKERS  judge=$([ "$VR_JUDGE" -eq 1 ] && echo ON || echo OFF)"
echo "    vr_video    num_test_cases=$VR_NUM_TEST_CASES   num_workers=$VR_NUM_WORKERS  num_frames=$VR_NUM_FRAMES"
echo "============================================================"

# ── Per-family job dispatchers ────────────────────────────────────────────

run_browsergym_job() {
    local tag=$1 model=$2 vision=$3
    local model_safe; model_safe="$(sanitize "$tag")"
    local out_dir="${RUN_DIR}/${model_safe}/browsergym"
    local logfile="${LOG_DIR}/${model_safe}__browsergym.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$RESUME" -eq 1 ]   && extra+=(--resume)
    [ "$VERBOSE" -eq 1 ]  && extra+=(-v)
    [ "$vision" = "0" ]   && extra+=(--no_vision)

    bash "$PY_BG_SHARD" \
        --num_shards "$BG_SHARDS" \
        --tasks_file "${CODEBASE_ROOT}/cold_start/task_samples/browsergym_miniwob_200.txt" \
        --tasks_file "${CODEBASE_ROOT}/cold_start/task_samples/browsergym_assistantbench_200.txt" \
        --model "$model" \
        --output_dir "$out_dir" \
        --conda_env "$CONDA_BG" \
        --max_steps "$BG_MAX_STEPS" \
        --api_key "$OPENROUTER_API_KEY" \
        --base_url "$OPENROUTER_BASE_URL" \
        "${extra[@]}" \
        > "$logfile" 2>&1
}

run_osworld_job() {
    local tag=$1 model=$2 vision=$3
    local model_safe; model_safe="$(sanitize "$tag")"
    local out_dir="${RUN_DIR}/${model_safe}/osworld"
    local logfile="${LOG_DIR}/${model_safe}__osworld.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$VERBOSE" -eq 1 ] && extra+=(-v)
    [ "$vision" = "0" ]  && extra+=(--no_vision)

    bash "$PY_OSW" \
        --max_parallel "$OSW_MAX_PARALLEL" \
        --conda_env "$CONDA_OSW" \
        --task_catalog "${POOL_DIR}/osworld_catalog.json" \
        --output_dir "$out_dir" \
        --episodes "$OSW_EPISODES" \
        --max_steps "$OSW_MAX_STEPS" \
        --no_save_frames \
        --model "$model" \
        --api_key "$OPENROUTER_API_KEY" \
        --base_url "$OPENROUTER_BASE_URL" \
        "${extra[@]}" \
        > "$logfile" 2>&1
}

_run_vr() {
    local tag=$1 model=$2 vision=$3 family=$4 benchmarks_str=$5
    local model_safe; model_safe="$(sanitize "$tag")"
    local out_dir="${RUN_DIR}/${model_safe}/${family}"
    local logfile="${LOG_DIR}/${model_safe}__${family}.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$VERBOSE" -eq 1 ] && extra+=(-v)
    [ "$VR_JUDGE" -eq 1 ] && [ "$family" = "vr_image" ] && extra+=(--judge)
    [ "$vision" = "0" ]   && extra+=(--no_vision)

    local frame_args=()
    if [ "$family" = "vr_video" ]; then
        frame_args=(--num_frames "$VR_NUM_FRAMES")
    fi

    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$CONDA_MAIN" --no-capture-output \
        python3 "$PY_VR" \
            --benchmarks $benchmarks_str \
            --sample_ids_dir "$POOL_DIR" \
            --num_test_cases "$VR_NUM_TEST_CASES" \
            --num_workers "$VR_NUM_WORKERS" \
            --model "$model" \
            --api_key "$OPENROUTER_API_KEY" \
            --base_url "$OPENROUTER_BASE_URL" \
            --output_dir "$out_dir" \
            "${frame_args[@]}" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

run_vr_image_job() { _run_vr "$1" "$2" "$3" "vr_image" "visual_toolbench tir_bench"; }
run_vr_video_job() { _run_vr "$1" "$2" "$3" "vr_video" "video_holmes siv_bench"; }

# ── Build job list (family|tag|model|vision) ──────────────────────────────
# Order: families x models so each family's 3 backbone jobs cluster
# together (one per provider — independent rate limits).
#
# CAPABILITY GATE — OSWorld + text-only models:
#   The OSWorld python launcher hard-wires ``use_vision=True`` (see
#   cold_start/generate_cold_start_actor_osworld.py: ``# vision is
#   mandatory in this pipeline``) and exposes NO --no_vision flag.
#   Sending images to a text-only OpenRouter slug (currently:
#   qwen/qwen3-max → "404 No endpoints found that support image input")
#   would crash every step. We skip such combinations here and surface
#   a clear warning rather than silently shipping a half-working run.
declare -a JOBS=()
declare -a SKIPPED=()
for fam in "${FAMILIES[@]}"; do
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag model vis <<<"$entry"
        if [ "$fam" = "osworld" ] && [ "$vis" = "0" ]; then
            SKIPPED+=("${fam}|${tag}|text-only model has no vision tower; OSWorld pipeline requires vision (no --no_vision flag)")
            continue
        fi
        JOBS+=("${fam}|${tag}|${model}|${vis}")
    done
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "[ERROR] No jobs to run." >&2
    exit 2
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo
    echo "Skipped jobs:"
    for sk in "${SKIPPED[@]}"; do
        IFS='|' read -r fam tag reason <<<"$sk"
        printf "  - %-10s %-8s  %s\n" "$fam" "$tag" "$reason"
    done
fi

echo
echo "Total jobs to dispatch: ${#JOBS[@]}"
for spec in "${JOBS[@]}"; do
    IFS='|' read -r fam tag _ vis <<<"$spec"
    vtag=$([ "$vis" = "1" ] && echo vision || echo no_vision)
    printf "  - %-10s %-8s %s\n" "$fam" "$tag" "$vtag"
done
echo
echo "Live tail any of:"
for spec in "${JOBS[@]}"; do
    IFS='|' read -r fam tag _ _ <<<"$spec"
    model_safe="$(sanitize "$tag")"
    echo "  tail -f ${LOG_DIR}/${model_safe}__${fam}.log"
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

    IFS='|' read -r fam tag model vis <<<"$spec"
    job_id="${fam}|${tag}"

    case "$fam" in
        browsergym) run_browsergym_job "$tag" "$model" "$vis" & ;;
        osworld)    run_osworld_job    "$tag" "$model" "$vis" & ;;
        vr_image)   run_vr_image_job   "$tag" "$model" "$vis" & ;;
        vr_video)   run_vr_video_job   "$tag" "$model" "$vis" & ;;
    esac
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
    printf '  "max_parallel": %s,\n'          "$MAX_PARALLEL"
    printf '  "save_frames": false,\n'
    printf '  "pool_dir": "%s",\n'            "$POOL_DIR"
    printf '  "openrouter_base_url": "%s",\n' "$OPENROUTER_BASE_URL"
    printf '  "models": [\n'
    first=1
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag slug vis <<<"$entry"
        [ $first -eq 1 ] || printf ',\n'
        printf '    {"tag": "%s", "model": "%s", "vision": %s}' \
            "$tag" "$slug" "$([ "$vis" = "1" ] && echo true || echo false)"
        first=0
    done
    printf '\n  ],\n'
    printf '  "jobs": [\n'
    first=1
    for spec in "${JOBS[@]}"; do
        IFS='|' read -r fam tag _ _ <<<"$spec"
        job_id="${fam}|${tag}"
        rc="${RC[$job_id]:-null}"
        [ $first -eq 1 ] || printf ',\n'
        printf '    {"family": "%s", "tag": "%s", "rc": %s}' \
            "$fam" "$tag" "$rc"
        first=0
    done
    printf '\n  ]\n'
    printf '}\n'
} > "$META_FILE"

# ── Summary ───────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  OpenRouter TRANSFER baselines — done"
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Elapsed:       ${ELAPSED}s ($((ELAPSED / 60)) min)"
echo "  Run dir:       $RUN_DIR/"
echo "  Latest:        $BASE_DIR/latest -> $RUN_ID"
echo "  Meta:          $META_FILE"
echo

ANY_OK=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r fam tag _ _ <<<"$spec"
    job_id="${fam}|${tag}"
    rc="${RC[$job_id]:-?}"
    model_safe="$(sanitize "$tag")"

    out="${RUN_DIR}/${model_safe}/${fam}"
    case "$fam" in
        browsergym)
            count=0
            [ -d "$out" ] && count=$(find "$out" -name 'episode_*.json' \
                ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
            unit="episodes"
            ;;
        osworld)
            count=0
            [ -d "$out" ] && count=$(find "$out" -mindepth 4 -name 'episode_*.json' \
                ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
            unit="episodes"
            ;;
        vr_image|vr_video)
            count=0
            [ -d "$out" ] && count=$(find "$out" -name 'sample_*.json' 2>/dev/null | wc -l)
            unit="samples"
            ;;
    esac
    printf "  %-10s %-8s rc=%-3s %s=%-5s\n" \
        "$fam" "$tag" "$rc" "$unit" "$count"
    [ "$rc" = "0" ] && ANY_OK=1
done
echo "============================================================"

[ "$ANY_OK" -eq 0 ] && exit 1
exit 0
