#!/usr/bin/env bash
#
# baselines/run_qwen_vllm_transfer_baselines.sh — Qwen3.5 baselines on the
# four TRANSFER-TARGET benchmark families covered by cold-start:
#
#     browsergym    -> MiniWoB++ + AssistantBench   (305 tasks, 8 shards)
#     osworld       -> 10 desktop domains            (120 stratified tasks; cut from 250 on 2026-05-03)
#     vr_image      -> VisualToolBench + TIR-Bench   (~2 × 1000 samples)
#     vr_video      -> Video-Holmes + SIV-Bench      (~2 × 1000 samples)
#
# Same vLLM server convention as run_qwen_vllm_baselines.sh:
#
#     Qwen/Qwen3.5-9B        -> http://localhost:8000/v1
#     Qwen/Qwen3.5-35B-A3B   -> http://localhost:8001/v1
#
# Reuses the EXACT same lean-plan sample IDs that cold-start uses
# (cold_start/evaluation_dataset/pool/), so the Qwen baselines compare
# directly against the gpt-5.4 cold-start runs.
#
# Frames / screenshots are NEVER persisted to disk:
#   - BrowserGym: --save_frames is omitted
#   - OSWorld:    --no_save_frames is forwarded
#   - VR (image/video): --save_frames is omitted
# The pipelines still feed pixels to the VLM in-memory; we just skip the
# PNG sidecar writes that bloat disk for a baseline comparison.
#
# Output layout:
#
#   <codebase_root>/qwen-baselines-out-transfer/<run_id>/
#     <model_tag>/                     # 9B or 35B-A3B
#       browsergym/<safe_task_id>/...
#       osworld/<run_id>/<domain>/...
#       vr_image/<benchmark>/...
#       vr_video/<benchmark>/...
#     _logs/<model_tag>__<family>.log
#     _logs/_servers/<tag>.log         # only when --launch_servers
#     _run_meta.json
#   <codebase_root>/qwen-baselines-out-transfer/latest -> <run_id>
#
# Concurrency model (SLA target: ≤ 12 h end-to-end on 8x H200):
#
#   - Across (model x family) jobs: --max_parallel 4 (default).  Both
#     vLLM endpoints serve up to 2 concurrent jobs each, which keeps
#     a TP=4 H200 in its comfort zone (~16-32 concurrent requests).
#     The natural FIFO ordering — (BG_9B, BG_35B, OSW_9B, OSW_35B,
#     VR_image_9B, VR_image_35B, VR_video_9B, VR_video_35B) — pairs
#     low-load jobs (BG=8 reqs, OSW=8 reqs) together so each backend
#     sees ~16 in-flight reqs when both are running, well within
#     server capacity.
#   - Within each job: BG shards=8, OSW max_parallel=8, VR num_workers=12.
#
# Wall-clock budget (default settings, normal/pessimistic):
#
#   Phase A: BG + OSW (4 jobs in flight)        ~2.5 h / ~3.5 h pessimistic
#   Phase B: VR image + VR video (4 jobs)       ~20 min / ~40 min pessimistic
#   ────────────────────────────────────────────────────────────────────
#   Total                                        ~3 h / ~4-5 h pessimistic
#
# 12 h SLA leaves ≥ 2.5x headroom over the pessimistic estimate.  If
# OSWorld VMs hang (rare but possible), the worst case is bounded by
# OSW_MAX_STEPS=50 × ~6 s/step × 250/8 tasks ≈ 5 h per backend, total
# ~7 h — still under 12 h.
#
# Usage:
#
#   # End-to-end on an 8x H200 box: spawn both vLLM servers (TP=4 each)
#   # and run all 8 (model x family) combos.
#   bash baselines/run_qwen_vllm_transfer_baselines.sh --launch_servers
#
#   # Default (assumes both vLLM servers are already running):
#   bash baselines/run_qwen_vllm_transfer_baselines.sh
#
#   # Only the 9B model, only image VR (smoke):
#   bash baselines/run_qwen_vllm_transfer_baselines.sh \
#       --models 9B --families vr_image
#
#   # Custom URLs (remote vLLM hosts):
#   VLLM_QWEN_9B_URL=http://gpu0:8000/v1 \
#   VLLM_QWEN_35B_URL=http://gpu1:8001/v1 \
#       bash baselines/run_qwen_vllm_transfer_baselines.sh
#
# Wrapper-only flags:
#   --launch_servers       spin up both vLLM servers in-script (kills on
#                          exit).  Defaults: 9B on GPUs 0-3 :8000, 35B on
#                          GPUs 4-7 :8001, both at TP=4.
#   --gpus_9b <csv>        GPUs for the 9B server (default: 0,1,2,3)
#   --gpus_35b <csv>       GPUs for the 35B-A3B server (default: 4,5,6,7)
#   --tp_9b N              tensor-parallel size for 9B  (default: 4)
#   --tp_35b N             tensor-parallel size for 35B (default: 4)
#   --models <list>        subset of {9B, 35B} (default: both)
#   --families <list>      subset of {browsergym, osworld, vr_image,
#                          vr_video} (default: all 4)
#   --max_parallel N       (model x family) jobs in flight (default: 4 —
#                          tuned for 12 h SLA on H200/TP=4; drop to 2
#                          if your servers struggle with > 16 concurrent
#                          requests, raise to 8 only on a beefier mesh).
#   # ── Per-family knobs (defaults match the cold-start lean plan) ──
#   --bg_shards N          BrowserGym shards   (default: 8)
#   --bg_max_steps N       BrowserGym per-task step cap (default: 16)
#   --osw_max_parallel N   OSWorld concurrent VMs (default: 8)
#   --osw_episodes N       OSWorld episodes per task (default: 1)
#   --osw_max_steps N      OSWorld per-episode step cap (default: 50)
#   --vr_num_test_cases N  VR sample cap per benchmark (default: 1000)
#   --vr_num_workers N     VR threadpool size (default: 12)
#   --vr_num_frames N      VR video frames/clip (default: 8)
#   --vr_judge             Enable LLM-as-judge for VTB / TIR-Bench
#   # ── Misc ────────────────────────────────────────────────────────
#   --pool_dir <path>      Directory holding the lean-plan manifests
#                          (default: cold_start/evaluation_dataset/pool)
#   --output_dir <path>    Override base output dir
#                          (default: <codebase_root>/qwen-baselines-out-transfer)
#   --run_id <id>          Override timestamped run id
#   --conda_main <name>    Conda env for vr / osworld python launchers
#                          (default: game-ai-agent — overridden per family)
#   --conda_browsergym <n> Conda env for BrowserGym (default: browsergym)
#   --conda_osworld <n>    Conda env for OSWorld    (default: osworld)
#   --resume               Pass --resume to BrowserGym (skip done episodes)
#   --verbose | -v         Pass --verbose / -v to all backends
#
# Optional environment variables:
#   VLLM_QWEN_9B_URL    Qwen3.5-9B  vLLM server   (default: http://localhost:8000/v1)
#   VLLM_QWEN_35B_URL   Qwen3.5-35B vLLM server   (default: http://localhost:8001/v1)
#   VLLM_API_KEY        Auth token (default: EMPTY)
#   QWEN_9B_MODEL       Override Qwen3.5-9B model id   (default: Qwen/Qwen3.5-9B)
#   QWEN_35B_MODEL      Override Qwen3.5-35B model id  (default: Qwen/Qwen3.5-35B-A3B)

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
# These are tuned for an 8x H200 box with both vLLM servers at TP=4 and
# target the 12 h SLA noted above.  See the header comment for the
# wall-clock budget breakdown and the rationale for each value.
MAX_PARALLEL_DEFAULT=4   # two jobs per backend at peak

BG_SHARDS_DEFAULT=8
BG_MAX_STEPS_DEFAULT=16
OSW_MAX_PARALLEL_DEFAULT=8
OSW_EPISODES_DEFAULT=1
OSW_MAX_STEPS_DEFAULT=50
VR_NUM_TEST_CASES_DEFAULT=1000
# 12 workers (down from 16) so two VR jobs sharing a backend stay under
# ~24 concurrent reqs, leaving headroom for the tool-call parser path.
VR_NUM_WORKERS_DEFAULT=12
VR_NUM_FRAMES_DEFAULT=8

CONDA_MAIN_DEFAULT="game-ai-agent"
CONDA_BG_DEFAULT="browsergym"
CONDA_OSW_DEFAULT="osworld"

POOL_DIR_DEFAULT="${CODEBASE_ROOT}/cold_start/evaluation_dataset/pool"

# ── vLLM server launch defaults (only used with --launch_servers) ─────────
GPUS_9B_DEFAULT="0,1,2,3"
GPUS_35B_DEFAULT="4,5,6,7"
TP_9B_DEFAULT=4
TP_35B_DEFAULT=4
SERVER_HEALTH_TIMEOUT_DEFAULT=900
SERVER_GPU_UTIL_DEFAULT=0.70
MAX_MODEL_LEN_DEFAULT=8192

QWEN_9B_MODEL="${QWEN_9B_MODEL:-Qwen/Qwen3.5-9B}"
QWEN_35B_MODEL="${QWEN_35B_MODEL:-Qwen/Qwen3.5-35B-A3B}"
VLLM_QWEN_9B_URL="${VLLM_QWEN_9B_URL:-http://localhost:8000/v1}"
VLLM_QWEN_35B_URL="${VLLM_QWEN_35B_URL:-http://localhost:8001/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

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

LAUNCH_SERVERS=0
GPUS_9B="$GPUS_9B_DEFAULT"
GPUS_35B="$GPUS_35B_DEFAULT"
TP_9B="$TP_9B_DEFAULT"
TP_35B="$TP_35B_DEFAULT"
SERVER_HEALTH_TIMEOUT="$SERVER_HEALTH_TIMEOUT_DEFAULT"

declare -a MODEL_TAGS=("9B" "35B")
declare -a FAMILIES=("browsergym" "osworld" "vr_image" "vr_video")
declare -a LAUNCHED_PIDS=()

# ── Parse args ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --launch_servers|--launch-servers)
            LAUNCH_SERVERS=1; shift ;;
        --no_launch_servers|--no-launch-servers)
            LAUNCH_SERVERS=0; shift ;;
        --gpus_9b|--gpus-9b)   shift; GPUS_9B="${1:-$GPUS_9B_DEFAULT}"; shift ;;
        --gpus_35b|--gpus-35b) shift; GPUS_35B="${1:-$GPUS_35B_DEFAULT}"; shift ;;
        --tp_9b|--tp-9b)       shift; TP_9B="${1:-$TP_9B_DEFAULT}"; shift ;;
        --tp_35b|--tp-35b)     shift; TP_35B="${1:-$TP_35B_DEFAULT}"; shift ;;
        --server_timeout|--server-timeout)
            shift; SERVER_HEALTH_TIMEOUT="${1:-$SERVER_HEALTH_TIMEOUT_DEFAULT}"; shift ;;
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

BASE_DIR="${BASE_DIR_OVERRIDE:-${CODEBASE_ROOT}/qwen-baselines-out-transfer}"
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
# Per-family env existence is checked just before dispatch (a missing env
# only fails THAT family, not the whole run).

# ── PYTHONPATH (codebase + sibling repos when present) ────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Resolve model entries ─────────────────────────────────────────────────
declare -a MODELS=()
for tag in "${MODEL_TAGS[@]}"; do
    case "$tag" in
        9B|qwen3.5-9b|Qwen3.5-9B)
            MODELS+=("9B|${QWEN_9B_MODEL}|${VLLM_QWEN_9B_URL}") ;;
        35B|35B-A3B|qwen3.5-35b-a3b|Qwen3.5-35B-A3B)
            MODELS+=("35B-A3B|${QWEN_35B_MODEL}|${VLLM_QWEN_35B_URL}") ;;
        *)
            echo "[ERROR] Unknown model tag '$tag' (allowed: 9B, 35B)" >&2
            exit 2 ;;
    esac
done

# ── TCP probe ─────────────────────────────────────────────────────────────
probe_url() {
    local url=$1 stripped host port
    stripped="${url#http://}"
    stripped="${stripped#https://}"
    stripped="${stripped%%/*}"
    if [[ "$stripped" == *:* ]]; then
        host="${stripped%:*}"
        port="${stripped##*:}"
    else
        host="$stripped"; port="80"
    fi
    if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
        exec 3<&-; exec 3>&-; return 0
    fi
    return 1
}

sanitize() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'; }

# ── vLLM server launch helpers (only with --launch_servers) ───────────────
_url_root()      { local u=$1; u="${u%/v1}"; u="${u%/}"; printf '%s' "$u"; }
_url_host_port() { local u=$1; u="${u#http://}"; u="${u#https://}"; printf '%s' "${u%%/*}"; }

_resolve_conda_python() {
    local env_name=$1 path
    path="$(conda run -n "$env_name" --no-capture-output \
                bash -c 'command -v python' 2>/dev/null | tr -d '[:space:]')"
    if [ -z "$path" ] || [ ! -x "$path" ]; then
        echo "[ERROR] could not resolve python for conda env '$env_name'" >&2
        return 1
    fi
    printf '%s' "$path"
}

# Spawn one vLLM server.  Mirrors run_qwen_vllm_baselines.sh:launch_vllm_server.
launch_vllm_server() {
    local model=$1 host_port=$2 gpus=$3 tp=$4 mlen=$5 extra=$6 logfile=$7
    local host="${host_port%:*}"
    local port="${host_port##*:}"
    mkdir -p "$(dirname "$logfile")"
    if [ -z "${CONDA_MAIN_PY:-}" ]; then
        echo "[ERROR] CONDA_MAIN_PY unset; call _resolve_conda_python first" >&2
        return 1
    fi
    local conda_bin; conda_bin="$(dirname "$CONDA_MAIN_PY")"
    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH="${conda_bin}:${PATH}" \
    nohup "$CONDA_MAIN_PY" -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --host "$host" \
            --port "$port" \
            --tensor-parallel-size "$tp" \
            --gpu-memory-utilization "$SERVER_GPU_UTIL_DEFAULT" \
            --max-model-len "$mlen" \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --trust-remote-code \
            --dtype bfloat16 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes \
            $extra \
        > "$logfile" 2>&1 &
    printf '%s' "$!"
}

wait_for_health() {
    local url=$1 pid=$2 timeout=$3 tag=$4
    local root; root="$(_url_root "$url")"
    local elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        if curl -sf "${root}/health" >/dev/null 2>&1; then
            echo "[server] ${tag} ready at ${root} after ${elapsed}s"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[server] ERROR ${tag} (pid=${pid}) exited before becoming ready" >&2
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "[server] ${tag} still warming up... ${elapsed}s / ${timeout}s"
        fi
    done
    echo "[server] ERROR ${tag} did not become healthy within ${timeout}s" >&2
    return 1
}

cleanup_servers() {
    local pid
    for pid in "${LAUNCHED_PIDS[@]:-}"; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "[server] stopping pid=${pid}"
            kill "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
if [ "$LAUNCH_SERVERS" -eq 1 ]; then
    trap cleanup_servers EXIT INT TERM
fi

# ── Optional: launch the two vLLM servers in-script ───────────────────────
if [ "$LAUNCH_SERVERS" -eq 1 ]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] --launch_servers requires curl on PATH (for /health)." >&2
        exit 1
    fi

    SERVER_LOG_DIR="${LOG_DIR}/_servers"
    mkdir -p "$SERVER_LOG_DIR"

    if ! CONDA_MAIN_PY="$(_resolve_conda_python "$CONDA_MAIN")"; then
        exit 1
    fi
    export CONDA_MAIN_PY
    echo "[server] using python: $CONDA_MAIN_PY"

    declare -A NEED_LAUNCH=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag _ url <<<"$entry"
        if probe_url "$url"; then
            echo "[server] $tag already reachable at $url — skipping launch"
            NEED_LAUNCH[$tag]=0
        else
            NEED_LAUNCH[$tag]=1
        fi
    done

    echo "============================================================"
    echo "  Launching vLLM servers (vision tower ENABLED — required by"
    echo "  browsergym / osworld / visual_reasoning pipelines)"
    echo "============================================================"
    # NOTE: do NOT pass --language-model-only here.  Unlike the gym/
    # env_wrappers baselines, our four transfer benchmarks all feed
    # pixels to the VLM at runtime (DOM screenshots, desktop captures,
    # frames extracted from clips), even though we don't persist any
    # PNGs to disk.  Disabling the vision tower would silently break
    # schema generation on every step.

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag model url <<<"$entry"
        [ "${NEED_LAUNCH[$tag]:-0}" = "1" ] || continue

        host_port="$(_url_host_port "$url")"
        case "$tag" in
            9B)
                gpus="$GPUS_9B"; tp="$TP_9B"
                extra=""
                ;;
            35B-A3B)
                gpus="$GPUS_35B"; tp="$TP_35B"
                extra=""
                if [ "$tp" -gt 1 ]; then
                    extra="--enable-expert-parallel --reasoning-parser qwen3"
                fi
                ;;
        esac

        logfile="${SERVER_LOG_DIR}/${tag}.log"
        echo "  [LAUNCH] $tag  model=$model  host=$host_port  GPUs=[$gpus]  TP=$tp"
        pid=$(launch_vllm_server "$model" "$host_port" "$gpus" "$tp" \
                                 "$MAX_MODEL_LEN_DEFAULT" "$extra" "$logfile")
        if [ -z "$pid" ]; then
            echo "[ERROR] failed to spawn vLLM server for $tag" >&2
            exit 1
        fi
        LAUNCHED_PIDS+=("$pid")
        echo "           pid=$pid  log=$logfile"
    done

    echo
    echo "Waiting for servers to become healthy (timeout=${SERVER_HEALTH_TIMEOUT}s) ..."
    HEALTH_FAIL=0
    i=0
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag _ url <<<"$entry"
        [ "${NEED_LAUNCH[$tag]:-0}" = "1" ] || continue
        pid="${LAUNCHED_PIDS[$i]}"
        i=$((i + 1))
        if ! wait_for_health "$url" "$pid" "$SERVER_HEALTH_TIMEOUT" "$tag"; then
            HEALTH_FAIL=1
        fi
    done
    if [ "$HEALTH_FAIL" -eq 1 ]; then
        echo "[ERROR] One or more launched servers failed health-check." >&2
        echo "        Inspect logs under ${SERVER_LOG_DIR}/." >&2
        exit 1
    fi
    echo
fi

# ── Reachability filter ───────────────────────────────────────────────────
declare -a SKIP_TAGS=()
ANY_REACHABLE=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag model url <<<"$entry"
    if probe_url "$url"; then
        ANY_REACHABLE=1
    else
        SKIP_TAGS+=("$tag")
    fi
done

# ── Banner ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Qwen vLLM TRANSFER baselines (browsergym + osworld + vr)"
echo "============================================================"
echo "  Run id:        $RUN_ID"
echo "  Run dir:       $RUN_DIR"
echo "  Logs:          $LOG_DIR/"
echo "  Pool manifests: $POOL_DIR"
echo "  Concurrency:   max_parallel=$MAX_PARALLEL  (jobs in flight)"
echo "  Save frames:   OFF  (no PNGs written to disk)"
echo "  Resume:        $([ "$RESUME" -eq 1 ] && echo ON || echo OFF)"
echo "  Conda main:    $CONDA_MAIN  (vr / general)"
echo "  Conda BG:      $CONDA_BG    (BrowserGym)"
echo "  Conda OSW:     $CONDA_OSW   (OSWorld)"
if [ "$LAUNCH_SERVERS" -eq 1 ]; then
    echo "  Servers:       launched in-script  (cleanup on exit)"
    echo "                 9B  -> GPUs [$GPUS_9B]  TP=$TP_9B"
    echo "                 35B -> GPUs [$GPUS_35B] TP=$TP_35B"
else
    echo "  Servers:       external (assumed already running)"
fi
echo
echo "  Backbones:"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag model url <<<"$entry"
    skip=0
    for s in "${SKIP_TAGS[@]:-}"; do
        [ "$s" = "$tag" ] && skip=1 && break
    done
    if [ "$skip" -eq 0 ]; then
        printf "    [OK]   %-8s %-30s %s\n" "$tag" "$model" "$url"
    else
        printf "    [SKIP] %-8s %-30s %s   (unreachable)\n" "$tag" "$model" "$url"
    fi
done
echo
echo "  Families:      ${FAMILIES[*]}"
echo "  Per-family knobs:"
echo "    browsergym  shards=$BG_SHARDS              max_steps=$BG_MAX_STEPS"
echo "    osworld     max_parallel=$OSW_MAX_PARALLEL          episodes=$OSW_EPISODES   max_steps=$OSW_MAX_STEPS"
echo "    vr_image    num_test_cases=$VR_NUM_TEST_CASES   num_workers=$VR_NUM_WORKERS  judge=$([ "$VR_JUDGE" -eq 1 ] && echo ON || echo OFF)"
echo "    vr_video    num_test_cases=$VR_NUM_TEST_CASES   num_workers=$VR_NUM_WORKERS  num_frames=$VR_NUM_FRAMES"
echo "============================================================"

if [ "$ANY_REACHABLE" -eq 0 ]; then
    echo "[ERROR] No vLLM endpoint is reachable. Re-run with --launch_servers" >&2
    echo "        or start the servers manually and override VLLM_QWEN_*_URL." >&2
    exit 1
fi

# Filter active models.
declare -a ACTIVE_MODELS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag _ _ <<<"$entry"
    skip=0
    for s in "${SKIP_TAGS[@]:-}"; do
        [ "$s" = "$tag" ] && skip=1 && break
    done
    [ "$skip" -eq 0 ] && ACTIVE_MODELS+=("$entry")
done

# ── Per-family job dispatchers ────────────────────────────────────────────

run_browsergym_job() {
    local model_tag=$1 model=$2 url=$3
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local out_dir="${RUN_DIR}/${model_safe}/browsergym"
    local logfile="${LOG_DIR}/${model_safe}__browsergym.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$RESUME" -eq 1 ]   && extra+=(--resume)
    [ "$VERBOSE" -eq 1 ]  && extra+=(-v)

    # Lean plan: MiniWoB (125) + AssistantBench (180) — VWA needs the
    # self-hosted Docker stack (VWA_*) which is rarely configured.
    bash "$PY_BG_SHARD" \
        --num_shards "$BG_SHARDS" \
        --tasks_file "${CODEBASE_ROOT}/cold_start/task_samples/browsergym_miniwob_200.txt" \
        --tasks_file "${CODEBASE_ROOT}/cold_start/task_samples/browsergym_assistantbench_200.txt" \
        --model "$model" \
        --output_dir "$out_dir" \
        --conda_env "$CONDA_BG" \
        --max_steps "$BG_MAX_STEPS" \
        --api_key "$VLLM_API_KEY" \
        --base_url "$url" \
        "${extra[@]}" \
        > "$logfile" 2>&1
}

run_osworld_job() {
    local model_tag=$1 model=$2 url=$3
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local out_dir="${RUN_DIR}/${model_safe}/osworld"
    local logfile="${LOG_DIR}/${model_safe}__osworld.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$VERBOSE" -eq 1 ] && extra+=(-v)

    bash "$PY_OSW" \
        --max_parallel "$OSW_MAX_PARALLEL" \
        --conda_env "$CONDA_OSW" \
        --task_catalog "${POOL_DIR}/osworld_catalog.json" \
        --output_dir "$out_dir" \
        --episodes "$OSW_EPISODES" \
        --max_steps "$OSW_MAX_STEPS" \
        --no_save_frames \
        --model "$model" \
        --api_key "$VLLM_API_KEY" \
        --base_url "$url" \
        "${extra[@]}" \
        > "$logfile" 2>&1
}

# Shared VR dispatcher.  ``benchmarks_str`` is space-separated.
_run_vr() {
    local model_tag=$1 model=$2 url=$3 family=$4 benchmarks_str=$5
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local out_dir="${RUN_DIR}/${model_safe}/${family}"
    local logfile="${LOG_DIR}/${model_safe}__${family}.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$VERBOSE" -eq 1 ] && extra+=(-v)
    [ "$VR_JUDGE" -eq 1 ] && [ "$family" = "vr_image" ] && extra+=(--judge)

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
            --api_key "$VLLM_API_KEY" \
            --base_url "$url" \
            --output_dir "$out_dir" \
            "${frame_args[@]}" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

run_vr_image_job() { _run_vr "$1" "$2" "$3" "vr_image" "visual_toolbench tir_bench"; }
run_vr_video_job() { _run_vr "$1" "$2" "$3" "vr_video" "video_holmes siv_bench"; }

# ── Build job list (family|tag|model|url) ─────────────────────────────────
# Order ensures pairs at max_parallel=2 stay one-per-backend:
# (BG_9B, BG_35B), (OSW_9B, OSW_35B), (VR_image_9B, VR_image_35B),
# (VR_video_9B, VR_video_35B).
declare -a JOBS=()
for fam in "${FAMILIES[@]}"; do
    for entry in "${ACTIVE_MODELS[@]}"; do
        IFS='|' read -r tag model url <<<"$entry"
        JOBS+=("${fam}|${tag}|${model}|${url}")
    done
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "[ERROR] No jobs to run." >&2
    exit 2
fi

echo
echo "Total jobs to dispatch: ${#JOBS[@]}"
for spec in "${JOBS[@]}"; do
    IFS='|' read -r fam tag _ _ <<<"$spec"
    printf "  - %-10s %-8s\n" "$fam" "$tag"
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

    IFS='|' read -r fam tag model url <<<"$spec"
    job_id="${fam}|${tag}"

    case "$fam" in
        browsergym) run_browsergym_job "$tag" "$model" "$url" & ;;
        osworld)    run_osworld_job    "$tag" "$model" "$url" & ;;
        vr_image)   run_vr_image_job   "$tag" "$model" "$url" & ;;
        vr_video)   run_vr_video_job   "$tag" "$model" "$url" & ;;
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
    printf '  "models": [\n'
    first=1
    for entry in "${ACTIVE_MODELS[@]}"; do
        IFS='|' read -r tag model url <<<"$entry"
        [ $first -eq 1 ] || printf ',\n'
        printf '    {"tag": "%s", "model": "%s", "base_url": "%s"}' \
            "$tag" "$model" "$url"
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
echo "  Qwen vLLM TRANSFER baselines — done"
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

    # Per-family artifact count for at-a-glance progress.
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
