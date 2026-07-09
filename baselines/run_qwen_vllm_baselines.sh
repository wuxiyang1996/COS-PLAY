#!/usr/bin/env bash
#
# baselines/run_qwen_vllm_baselines.sh — Qwen3.5 baselines via vLLM
#
# Runs the COS-PLAY actor cold-start pipeline (visual grounding +
# schema-driven action selection) using Qwen as the backbone, served
# behind a vLLM OpenAI-compatible endpoint. Reuses the same Python
# back-ends as cold_start/run_coldstart_actor_*.sh:
#
#     cold_start/generate_cold_start_actor.py        (env_wrappers)
#     cold_start/generate_cold_start_actor_gymv.py   (gym-v Temporal)
#
# Two backbones are dispatched in parallel against two distinct vLLM
# servers (defaults match inference/serve_qwen35_35b_a3b.sh + the
# Qwen3.5-9B server convention from inference/run_qwen3_8b_eval.sh):
#
#   Qwen/Qwen3.5-9B        -> http://localhost:8000/v1
#   Qwen/Qwen3.5-35B-A3B   -> http://localhost:8001/v1
#
# Each (model x env) combo runs 16 episodes by default.
#
# Output layout (default):
#
#   <codebase_root>/qwen-baselines-out/<run_id>/
#     <model_tag>/                       # 9B or 35B-A3B (both backbones share one run dir)
#       env_wrappers/<game>/...          # twenty_forty_eight, candy_crush, tetris, super_mario
#       gymv/<env_id_safe>/...           # Temporal_<Title>-v0/...
#     _logs/<model_tag>__<kind>__<target>.log
#     _run_meta.json
#   <codebase_root>/qwen-baselines-out/latest -> <run_id>
#
# Per-job (model x env) is one python3 process so a single failure
# (rate-limit, ROM missing, OOM) cannot take down the whole batch.
#
# Usage:
#
#   # End-to-end on an 8x H200 box: launch both vLLM servers (TP=4 each)
#   # and run all 22 (model x env) combos with 16 concurrent jobs.
#   # (3 env_wrappers + 8 gym-v retained envs) × 2 backbones = 22.
#   bash baselines/run_qwen_vllm_baselines.sh --launch_servers
#
#   # Default: both models, all envs, 16 episodes, max_parallel=16
#   # (assumes the two vLLM servers are already running externally)
#   bash baselines/run_qwen_vllm_baselines.sh
#
#   # Only the 9B model, only env_wrappers games:
#   bash baselines/run_qwen_vllm_baselines.sh --models 9B --skip_gymv
#
#   # Only gym-v Temporal envs, both models, episodes=8 (smoke):
#   bash baselines/run_qwen_vllm_baselines.sh --skip_envwrappers --episodes 8
#
#   # Custom vLLM endpoints (e.g. remote hosts):
#   VLLM_QWEN_9B_URL=http://gpu0:8000/v1 \
#   VLLM_QWEN_35B_URL=http://gpu1:8001/v1 \
#       bash baselines/run_qwen_vllm_baselines.sh
#
#   # Custom GPU allocation when launching servers in-script:
#   bash baselines/run_qwen_vllm_baselines.sh --launch_servers \
#       --gpus_9b 0,1 --tp_9b 2 --gpus_35b 2,3,4,5,6,7 --tp_35b 6
#
#   # Restrict to a subset of gym-v Temporal envs:
#   bash baselines/run_qwen_vllm_baselines.sh \
#       --gymv Temporal/Airstriker-v0 Temporal/Columns-v0 --skip_envwrappers
#
#   # Include super_mario (requires orak-mario conda env + nes-py):
#   bash baselines/run_qwen_vllm_baselines.sh --include_mario
#
#   # Resume an interrupted run (output dir is reused, episodes skipped):
#   bash baselines/run_qwen_vllm_baselines.sh --run_id qwen_vllm_smoke --resume
#
# Wrapper-only flags:
#   --launch_servers         spin up both vLLM servers in-process for the
#                            duration of this run (kills them on exit).
#                            Defaults: 9B on GPUs 0-3 :8000, 35B on GPUs 4-7
#                            :8001, both at TP=4.
#   --gpus_9b <csv>          GPUs for the 9B server (default: 0,1,2,3)
#   --gpus_35b <csv>         GPUs for the 35B-A3B server (default: 4,5,6,7)
#   --tp_9b N                tensor-parallel size for 9B  (default: 4)
#   --tp_35b N               tensor-parallel size for 35B (default: 4)
#   --models <list>          subset of {9B, 35B} (default: both)
#   --episodes N             episodes per (model x env) combo (default: 16)
#   --max_parallel N         concurrent jobs cap (default: 16; H200 + TP=4
#                            handles ~8 in-flight requests per server with
#                            plenty of headroom).
#   --max_steps_envw N       per-episode step cap for env_wrappers (default: backend per-game default)
#   --max_steps_gymv N       per-episode step cap for gym-v (default: 60)
#   --frame_skip_gymv N      emulator frames per agent step on gym-v (default: 1; recommended: 8)
#   --envwrappers <g>...     restrict env_wrappers games (default: 2048, candy_crush, tetris)
#   --gymv <id>...           restrict gym-v env ids (default: all 13 Temporal/* envs)
#   --skip_envwrappers       skip the env_wrappers backend entirely
#   --skip_gymv              skip the gym-v backend entirely
#   --include_mario          add super_mario (needs orak-mario conda env)
#   --vision                 enable VLM vision call (default OFF — Qwen3.5-9B / 35B-A3B
#                            are text-only in this repo's stack)
#   --save_frames            persist PNG frames to disk for debugging
#   --resume                 resume previously-started run (skip done episodes)
#   --run_id <id>            override timestamped run id
#   --output_dir <path>      base output dir (default: <codebase_root>/qwen-baselines-out)
#   --conda_main <name>      main conda env (default: game-ai-agent)
#   --conda_orak <name>      mario conda env (default: orak-mario)
#   --temperature_action F   sampling temp for action call (default: 0.4)
#   --temperature_schema F   sampling temp for schema call  (default: 0.2)
#   --verbose | -v           pass through to the python backends
#
# Optional environment variables:
#   VLLM_QWEN_9B_URL    Qwen3.5-9B  vLLM server   (default: http://localhost:8000/v1)
#   VLLM_QWEN_35B_URL   Qwen3.5-35B vLLM server   (default: http://localhost:8001/v1)
#   VLLM_API_KEY        Auth token (default: EMPTY — the canonical placeholder
#                       vLLM OpenAI-compatible servers accept).
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
GYMV_ROOT="$(cd "$WORKSPACE_ROOT/gym-v" 2>/dev/null && pwd || echo "")"
GAMINGAGENT_ROOT="$(cd "$WORKSPACE_ROOT/GamingAgent" 2>/dev/null && pwd || echo "")"
ORAK_SRC="$(cd "$WORKSPACE_ROOT/Orak/src" 2>/dev/null && pwd || echo "")"
PY_ENVW="${CODEBASE_ROOT}/cold_start/generate_cold_start_actor.py"
PY_GYMV="${CODEBASE_ROOT}/cold_start/generate_cold_start_actor_gymv.py"

# ── Defaults ──────────────────────────────────────────────────────────────
EPISODES_DEFAULT=16
# 16 jobs in flight = ~8 per server. Each H200 + TP=4 endpoint handles
# ~32 concurrent chat completions easily, so this leaves a 4x safety
# margin and still keeps both GPU pools saturated.
MAX_PARALLEL_DEFAULT=16
MAX_STEPS_GYMV_DEFAULT=60
FRAME_SKIP_GYMV_DEFAULT=1
TEMP_ACTION_DEFAULT=0.4
TEMP_SCHEMA_DEFAULT=0.2

# ── vLLM server launch defaults (used only when --launch_servers is set) ──
GPUS_9B_DEFAULT="0,1,2,3"
GPUS_35B_DEFAULT="4,5,6,7"
TP_9B_DEFAULT=4
TP_35B_DEFAULT=4
SERVER_HEALTH_TIMEOUT_DEFAULT=900    # vLLM warmup can take ~2 min on H200
# 0.75 leaves headroom for CUDA context + NCCL peer-access overhead that
# spikes when both servers' workers init across the 8-GPU mesh (~33 GiB of
# transient pre-allocation observed). Drop to 0.70 if you still hit
# "Free memory on device cudaN ... is less than desired GPU memory utilization".
SERVER_GPU_UTIL_DEFAULT=0.70
MAX_MODEL_LEN_9B_DEFAULT=8192
MAX_MODEL_LEN_35B_DEFAULT=8192

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

QWEN_9B_MODEL="${QWEN_9B_MODEL:-Qwen/Qwen3.5-9B}"
QWEN_35B_MODEL="${QWEN_35B_MODEL:-Qwen/Qwen3.5-35B-A3B}"
VLLM_QWEN_9B_URL="${VLLM_QWEN_9B_URL:-http://localhost:8000/v1}"
VLLM_QWEN_35B_URL="${VLLM_QWEN_35B_URL:-http://localhost:8001/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

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
INCLUDE_MARIO=0
USE_VISION=0
SAVE_FRAMES=0
RESUME=0
VERBOSE=0
RUN_ID=""
BASE_DIR_OVERRIDE=""
CONDA_MAIN="$CONDA_MAIN_DEFAULT"
CONDA_ORAK="$CONDA_ORAK_DEFAULT"

LAUNCH_SERVERS=0
GPUS_9B="$GPUS_9B_DEFAULT"
GPUS_35B="$GPUS_35B_DEFAULT"
TP_9B="$TP_9B_DEFAULT"
TP_35B="$TP_35B_DEFAULT"
SERVER_HEALTH_TIMEOUT="$SERVER_HEALTH_TIMEOUT_DEFAULT"

declare -a MODEL_TAGS=("9B" "35B")
declare -a ENVWRAPPERS=()
declare -a GYMV=()
declare -a LAUNCHED_PIDS=()

# ── Parse args ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --launch_servers|--launch-servers)
            LAUNCH_SERVERS=1; shift ;;
        --no_launch_servers|--no-launch-servers)
            LAUNCH_SERVERS=0; shift ;;
        --gpus_9b|--gpus-9b)
            shift; GPUS_9B="${1:-$GPUS_9B_DEFAULT}"; shift ;;
        --gpus_35b|--gpus-35b)
            shift; GPUS_35B="${1:-$GPUS_35B_DEFAULT}"; shift ;;
        --tp_9b|--tp-9b)
            shift; TP_9B="${1:-$TP_9B_DEFAULT}"; shift ;;
        --tp_35b|--tp-35b)
            shift; TP_35B="${1:-$TP_35B_DEFAULT}"; shift ;;
        --server_timeout|--server-timeout)
            shift; SERVER_HEALTH_TIMEOUT="${1:-$SERVER_HEALTH_TIMEOUT_DEFAULT}"; shift ;;
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

BASE_DIR="${BASE_DIR_OVERRIDE:-${CODEBASE_ROOT}/qwen-baselines-out}"
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

# ── Headless rendering / DISPLAY ──────────────────────────────────────────
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

# ── Probe vLLM endpoints (best-effort; missing servers are skipped) ───────
probe_url() {
    local url=$1 stripped host port
    stripped="${url#http://}"
    stripped="${stripped#https://}"
    stripped="${stripped%%/*}"
    if [[ "$stripped" == *:* ]]; then
        host="${stripped%:*}"
        port="${stripped##*:}"
    else
        host="$stripped"
        port="80"
    fi
    if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
        exec 3<&-
        exec 3>&-
        return 0
    fi
    return 1
}

# ── Sanitise filename component ───────────────────────────────────────────
sanitize() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

# ── vLLM server launch helpers (only used with --launch_servers) ──────────

# Strip "/v1" suffix to get the server root (where /health lives).
_url_root() {
    local url=$1
    url="${url%/v1}"
    url="${url%/}"
    printf '%s' "$url"
}

_url_host_port() {
    local url=$1 stripped
    stripped="${url#http://}"
    stripped="${stripped#https://}"
    stripped="${stripped%%/*}"
    printf '%s' "$stripped"
}

# Resolve the conda env's python interpreter once.  We invoke that binary
# directly (instead of `conda run …`) so $! captures the *real* python PID,
# which lets the health watchdog detect crashes immediately via kill -0.
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

# Launch one vLLM server in the background, returning its PID.
# Args: model, host_port (e.g. "127.0.0.1:8000"), gpus_csv, tp, max_model_len,
#       extra_flags_str, logfile.
launch_vllm_server() {
    local model=$1 host_port=$2 gpus=$3 tp=$4 mlen=$5 extra=$6 logfile=$7
    local host="${host_port%:*}"
    local port="${host_port##*:}"
    mkdir -p "$(dirname "$logfile")"
    if [ -z "${CONDA_MAIN_PY:-}" ]; then
        echo "[ERROR] CONDA_MAIN_PY unset; call _resolve_conda_python first" >&2
        return 1
    fi
    # IMPORTANT: do NOT put comments between the env-var prefix lines and
    # the command — a `#` after a `\` continuation terminates the logical
    # line in bash, silently dropping all the env vars (workers then run
    # without VLLM_USE_DEEP_GEMM, with deep_gemm_warmup re-enabled, etc.).
    #
    # Env vars set on this command:
    # * VLLM_USE_DEEP_GEMM=0 + VLLM_DEEP_GEMM_WARMUP=skip — bypass the
    #   vLLM warmup path that probes `deep_gemm` even on BF16 models
    #   (kernel_warmup.py:27-37 gates the call on these vars).
    # * HF_HUB_ENABLE_HF_TRANSFER=1 — faster first-time weight downloads.
    # * PATH prepended with the conda env's bin dir so torch-extension
    #   builds (FlashInfer all-reduce, GDN linear-attn JITs) can find
    #   `ninja` even though we launch python directly without
    #   `conda activate`.
    #
    # CLI:
    # * --dtype bfloat16 (not "auto") so vLLM never enters the FP8 path on
    #   these BF16 base checkpoints.  Qwen/Qwen3.5-9B and
    #   Qwen/Qwen3.5-35B-A3B are already BF16 — the FP8 variants are the
    #   separately-named *-FP8 model IDs.
    # * --enable-auto-tool-choice + --tool-call-parser hermes — required by
    #   the actor pipeline which sends tool_choice=
    #   {"type":"function","function":{"name":"choose_action"}}.  Without
    #   these, vLLM returns HTTP 400 and the actor silently falls back to
    #   random actions (action_ok=0/200 in episode logs).  `hermes` is the
    #   standard Qwen tool-call parser.
    local conda_bin="$(dirname "$CONDA_MAIN_PY")"
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

# Block until the server's /health endpoint returns 200, or timeout.
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

# Cleanup trap: kill any servers we launched in-script.
cleanup_servers() {
    local pid
    for pid in "${LAUNCHED_PIDS[@]:-}"; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "[server] stopping pid=${pid}"
            # SIGTERM first, then SIGKILL fallback after a grace period.
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

# ── Optional: launch the two vLLM servers in-script (8x H200 dual-pool) ──
if [ "$LAUNCH_SERVERS" -eq 1 ]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] --launch_servers requires curl on PATH (for /health)." >&2
        exit 1
    fi

    SERVER_LOG_DIR="${LOG_DIR}/_servers"
    mkdir -p "$SERVER_LOG_DIR"

    # Resolve interpreter once so $! is the real python PID (not the conda
    # wrapper) — required for the health watchdog to catch crashes promptly.
    if ! CONDA_MAIN_PY="$(_resolve_conda_python "$CONDA_MAIN")"; then
        exit 1
    fi
    export CONDA_MAIN_PY
    echo "[server] using python: $CONDA_MAIN_PY"

    # Reuse running servers when possible (probe before spawn).
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
    echo "  Launching vLLM servers (CUDA_VISIBLE_DEVICES per model)"
    echo "============================================================"
    # Both Qwen3.5 backbones ship a vision tower; baseline jobs default to
    # text-only (--no_vision), so we can ask vLLM to skip loading the vision
    # encoder.  This saves ~5–10 GiB and a fair chunk of warmup time.  We
    # only opt in when vision is OFF to avoid breaking --use_vision runs.
    LM_ONLY_FLAG=""
    if [ "$USE_VISION" -eq 0 ]; then
        LM_ONLY_FLAG="--language-model-only"
    fi

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r tag model url <<<"$entry"
        [ "${NEED_LAUNCH[$tag]:-0}" = "1" ] || continue

        host_port="$(_url_host_port "$url")"
        case "$tag" in
            9B)
                gpus="$GPUS_9B"
                tp="$TP_9B"
                mlen="$MAX_MODEL_LEN_9B_DEFAULT"
                extra="$LM_ONLY_FLAG"
                ;;
            35B-A3B)
                gpus="$GPUS_35B"
                tp="$TP_35B"
                mlen="$MAX_MODEL_LEN_35B_DEFAULT"
                # Qwen3.5-35B-A3B is an MoE model: enable expert-parallel
                # whenever TP > 1 to disjointly shard experts across ranks
                # (matches inference/serve_qwen35_35b_a3b.sh).
                extra="$LM_ONLY_FLAG"
                if [ "$tp" -gt 1 ]; then
                    extra="$extra --enable-expert-parallel --reasoning-parser qwen3"
                fi
                ;;
        esac

        logfile="${SERVER_LOG_DIR}/${tag}.log"
        echo "  [LAUNCH] $tag  model=$model  host=$host_port  GPUs=[$gpus]  TP=$tp"
        pid=$(launch_vllm_server "$model" "$host_port" "$gpus" "$tp" \
                                 "$mlen" "$extra" "$logfile")
        if [ -z "$pid" ]; then
            echo "[ERROR] failed to spawn vLLM server for $tag" >&2
            exit 1
        fi
        LAUNCHED_PIDS+=("$pid")
        echo "           pid=$pid  log=$logfile"
    done

    # Block on /health for everything we just spawned.
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

# ── Banner ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Qwen vLLM baselines (env_wrappers + gym-v)"
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
if [ "$LAUNCH_SERVERS" -eq 1 ]; then
    echo "  Servers:       launched in-script  (cleanup on exit)"
    echo "                 9B  -> GPUs [$GPUS_9B]  TP=$TP_9B"
    echo "                 35B -> GPUs [$GPUS_35B] TP=$TP_35B"
    echo "  Server dtype:  bfloat16  (FP8 paths disabled)"
else
    echo "  Servers:       external (assumed already running)"
fi
echo
echo "  Backbones:"
ANY_REACHABLE=0
declare -a SKIP_TAGS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag model url <<<"$entry"
    if probe_url "$url"; then
        printf "    [OK]   %-8s %-30s %s\n" "$tag" "$model" "$url"
        ANY_REACHABLE=1
    else
        printf "    [SKIP] %-8s %-30s %s   (unreachable)\n" "$tag" "$model" "$url"
        SKIP_TAGS+=("$tag")
    fi
done
echo
if [ "$INCLUDE_ENVWRAPPERS" -eq 1 ]; then
    echo "  env_wrappers:  ${ENVWRAPPERS[*]}$([ "$INCLUDE_MARIO" -eq 1 ] && echo " super_mario")"
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

if [ "$ANY_REACHABLE" -eq 0 ]; then
    echo "[ERROR] No vLLM endpoint is reachable. Start one of:"
    echo "  bash inference/serve_qwen35_35b_a3b.sh   # 35B  on :8001"
    echo "  bash inference/run_qwen3_8b_eval.sh ...  # 9B-class on :8000"
    echo "Then re-run this script (or override VLLM_QWEN_*_URL)."
    exit 1
fi

# Filter out unreachable models so we never queue a doomed job.
declare -a ACTIVE_MODELS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r tag model url <<<"$entry"
    skip=0
    for s in "${SKIP_TAGS[@]:-}"; do
        [ "$s" = "$tag" ] && skip=1 && break
    done
    [ "$skip" -eq 0 ] && ACTIVE_MODELS+=("$entry")
done

# ── Per-job dispatcher ────────────────────────────────────────────────────
run_envwrapper_job() {
    local model_tag=$1 model=$2 url=$3 game=$4
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local out_dir="${RUN_DIR}/${model_safe}/env_wrappers/${game}"
    local logfile="${LOG_DIR}/${model_safe}__envw__${game}.log"
    mkdir -p "$out_dir"

    local conda_env="$CONDA_MAIN"
    [ "$game" = "super_mario" ] && conda_env="$CONDA_ORAK"

    local extra=()
    [ "$USE_VISION" -eq 0 ] && extra+=(--no_vision)
    [ "$SAVE_FRAMES" -eq 1 ] && extra+=(--save_frames)
    [ "$RESUME" -eq 1 ]      && extra+=(--resume)
    [ "$VERBOSE" -eq 1 ]     && extra+=(--verbose)
    [ -n "$MAX_STEPS_ENVW" ] && extra+=(--max_steps "$MAX_STEPS_ENVW")

    PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
    conda run -n "$conda_env" --no-capture-output \
        python3 "$PY_ENVW" \
            --games "$game" \
            --episodes "$EPISODES" \
            --model "$model" \
            --api_key "$VLLM_API_KEY" \
            --base_url "$url" \
            --temperature_action "$TEMP_ACTION" \
            --temperature_schema "$TEMP_SCHEMA" \
            --output_dir "$out_dir" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

run_gymv_job() {
    local model_tag=$1 model=$2 url=$3 env_id=$4
    local model_safe; model_safe="$(sanitize "$model_tag")"
    local env_safe;   env_safe="$(sanitize "$env_id")"
    local out_dir="${RUN_DIR}/${model_safe}/gymv"
    local logfile="${LOG_DIR}/${model_safe}__gymv__${env_safe}.log"
    mkdir -p "$out_dir"

    local extra=()
    [ "$USE_VISION" -eq 0 ] && extra+=(--no_vision)
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
            --model "$model" \
            --api_key "$VLLM_API_KEY" \
            --base_url "$url" \
            --temperature_action "$TEMP_ACTION" \
            --temperature_schema "$TEMP_SCHEMA" \
            --output_dir "$out_dir" \
            "${extra[@]}" \
        > "$logfile" 2>&1
}

# ── Build job list (kind|tag|model|url|target) ────────────────────────────
declare -a JOBS=()
for entry in "${ACTIVE_MODELS[@]}"; do
    IFS='|' read -r tag model url <<<"$entry"
    if [ "$INCLUDE_ENVWRAPPERS" -eq 1 ]; then
        for g in "${ENVWRAPPERS[@]}"; do
            JOBS+=("envw|${tag}|${model}|${url}|${g}")
        done
        if [ "$INCLUDE_MARIO" -eq 1 ]; then
            JOBS+=("envw|${tag}|${model}|${url}|super_mario")
        fi
    fi
    if [ "$INCLUDE_GYMV" -eq 1 ]; then
        for e in "${GYMV[@]}"; do
            JOBS+=("gymv|${tag}|${model}|${url}|${e}")
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
    IFS='|' read -r kind tag model url target <<<"$spec"
    printf "  - %-5s %-8s %-32s\n" "$kind" "$tag" "$target"
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

    IFS='|' read -r kind tag model url target <<<"$spec"
    job_id="${kind}|${tag}|${target}"

    if [ "$kind" = "envw" ]; then
        run_envwrapper_job "$tag" "$model" "$url" "$target" &
    else
        run_gymv_job       "$tag" "$model" "$url" "$target" &
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
        IFS='|' read -r kind tag _ _ target <<<"$spec"
        job_id="${kind}|${tag}|${target}"
        rc="${RC[$job_id]:-null}"
        [ $first -eq 1 ] || printf ',\n'
        printf '    {"kind": "%s", "tag": "%s", "target": "%s", "rc": %s}' \
            "$kind" "$tag" "$target" "$rc"
        first=0
    done
    printf '\n  ]\n'
    printf '}\n'
} > "$META_FILE"

# ── Summary ───────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  Qwen vLLM baselines — done"
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

    # Count finished episode_*.json files for at-a-glance progress.
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
