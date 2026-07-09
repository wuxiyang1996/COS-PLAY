#!/usr/bin/env bash
#
# run_coldstart_actor_osworld_all.sh — drive ALL OSWorld domains through
# the gpt-5.5 actor cold-start, sequentially or in parallel, with per-domain
# logs grouped under one timestamped run dir.
#
# Output layout (default):
#
#   <codebase_root>/Cold-start-out-osworld/
#   ├── <run_id>/                              # YYYY-MM-DD_HH-MM-SS
#   │   ├── chrome/                            # one dir per domain
#   │   │   └── <safe_task_id>/
#   │   │       ├── episode_NNN.json
#   │   │       ├── episode_buffer.json
#   │   │       ├── rollouts.jsonl
#   │   │       ├── rollout_summary.json
#   │   │       └── frames/ep_NNN/step_NNN.png + step_NNN.json
#   │   ├── gimp/...
#   │   ├── libreoffice_calc/...
#   │   ├── libreoffice_impress/...
#   │   ├── libreoffice_writer/...
#   │   ├── multi_apps/...
#   │   ├── os/...
#   │   ├── thunderbird/...
#   │   ├── vlc/...
#   │   ├── vs_code/...
#   │   ├── _logs/<domain>.log                 # per-domain stdout/stderr
#   │   └── _run_meta.json                     # run config + per-domain rc
#   └── latest -> <run_id>                     # symlink to most recent run
#
# Each domain gets its own python3 process so that one domain's failure
# (Docker glitch, OOM, missing app, evaluator crash, etc.) cannot take
# down the others. The OSWorld VM is rebooted between tasks via
# DesktopEnv.reset(task_config=...) to amortise the ~28 s cold-boot cost.
#
# Hard-wired modes (NO opt-out, mirrored from the python launcher):
#   - VM is ALWAYS HEADLESS.
#   - The VLM (gpt-5.5 vision) is REQUIRED on every step.
#   - Frames are SAVED BY DEFAULT (PNG + step_NNN.json sidecar). Pass
#     ``--no_save_frames`` to skip when disk pressure matters.
#
# Usage:
#
#   # PARALLEL (default), all 10 OSWorld domains, full vision pipeline,
#   # 50 steps per task (the published OSWorld evaluation cap), frames
#   # saved per step. This is the closest thing to a "real" eval.
#   bash cold_start/run_coldstart_actor_osworld_all.sh \
#       --episodes 1 --max_steps 50 -v
#
#   # Proper benchmark protocol (3 episodes per task, averaged)
#   bash cold_start/run_coldstart_actor_osworld_all.sh \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 3 --max_steps 50 \
#       --parallel --max_parallel 10 -v
#
#   # SEQUENTIAL dispatch (one domain at a time)
#   bash cold_start/run_coldstart_actor_osworld_all.sh --sequential \
#       --episodes 1 --max_steps 50 -v
#
#   # Cap concurrency to 3 (Docker-friendly — VMs are RAM-hungry)
#   bash cold_start/run_coldstart_actor_osworld_all.sh --max_parallel 3 \
#       --episodes 1 --max_steps 50 -v
#
#   # Skip frame persistence (cheap on disk; rollouts.jsonl still records
#   # the schema and action — no PNGs on disk)
#   bash cold_start/run_coldstart_actor_osworld_all.sh --no_save_frames \
#       --episodes 1 --max_steps 50 -v
#
#   # Restrict to a subset of domains
#   bash cold_start/run_coldstart_actor_osworld_all.sh \
#       --domains chrome os --tasks_per_domain 2 \
#       --episodes 1 --max_steps 50 -v
#
#   # Quick pipeline sanity sweep (light budget — pipeline still validated,
#   # but expect office/gimp tasks to run out of steps; see
#   # smoke_test_osworld.sh for a labelled smoke check):
#   bash cold_start/run_coldstart_actor_osworld_all.sh \
#       --tasks_per_domain 1 --episodes 1 --max_steps 12 -v
#
#   # Custom run id (so a follow-up call lands in the same folder)
#   bash cold_start/run_coldstart_actor_osworld_all.sh \
#       --run_id smoke_2026 --episodes 1 --max_steps 50 --resume -v
#
# Wrapper-only flags (consumed here, NOT forwarded to the python launcher):
#   --parallel | -P             dispatch domains concurrently (default)
#   --sequential                dispatch domains one at a time
#   --max_parallel N            cap concurrency (default: 8 — assumes ~64 GB RAM;
#                                drop to 3-4 on smaller hosts, raise to 10+ on >=96 GB)
#   --domains <name>...         restrict to a subset (default: all 10)
#   --run_id <id>               override auto-timestamped run id
#   --output_dir <path>         override base dir
#                               (default: <repo>/Cold-start-out-osworld)
#   --conda_env <name>          conda env for the python launcher
#                               (default: ``osworld``)
#   --task_catalog <path>       OSWorld test_*.json catalog
#                               (default: /workspace/OSWorld/evaluation_examples/test_small.json)
#   --vm_data_dir <path>        directory containing Ubuntu.qcow2
#                               (default: <repo>/docker_vm_data)
#
# All other arguments are forwarded VERBATIM to the python launcher for
# every domain. The wrapper itself injects ``--domains <domain>`` and
# ``--output_dir <run_dir>`` per dispatch.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${CODEBASE_ROOT}/.." && pwd)"
PY_LAUNCHER="${SCRIPT_DIR}/generate_cold_start_actor_osworld.py"
DEFAULT_BASE_DIR="${CODEBASE_ROOT}/Cold-start-out-osworld"
DEFAULT_CONDA_ENV="osworld"
DEFAULT_TASK_CATALOG="/workspace/OSWorld/evaluation_examples/test_small.json"
DEFAULT_VM_DATA_DIR="${CODEBASE_ROOT}/docker_vm_data"
# Each KVM guest is ~6 GB RAM + 1-2 vCPU.  Default 8 matches the 10-domain
# spread (one extra slot for a re-queue on a flake) and assumes a 64 GB /
# 16-vCPU host.  Drop to 3-4 on a 32 GB box, raise to 10+ on >= 96 GB.
# Real wall-clock at 8: ~2.5 h for 250 tasks @ 30 steps each.
DEFAULT_MAX_PARALLEL=8

# All 10 registered OSWorld domains (must match ALL_OSWORLD_DOMAINS in the
# python launcher).
DEFAULT_DOMAINS=(
    chrome
    gimp
    libreoffice_calc
    libreoffice_impress
    libreoffice_writer
    multi_apps
    os
    thunderbird
    vlc
    vs_code
)

# ── Parse wrapper-only flags; forward the rest ───────────────────────────
PARALLEL=1
MAX_PARALLEL="${DEFAULT_MAX_PARALLEL}"
DOMAINS=()
USER_ARGS=()
RUN_ID=""
BASE_DIR="$DEFAULT_BASE_DIR"
CONDA_ENV="$DEFAULT_CONDA_ENV"
TASK_CATALOG="$DEFAULT_TASK_CATALOG"
VM_DATA_DIR="${OSWORLD_VM_DATA_DIR:-$DEFAULT_VM_DATA_DIR}"

while [ $# -gt 0 ]; do
    case "$1" in
        --parallel|-P)
            PARALLEL=1; shift ;;
        --sequential)
            PARALLEL=0; shift ;;
        --max_parallel|--max-parallel)
            shift; MAX_PARALLEL="${1:-${DEFAULT_MAX_PARALLEL}}"; shift ;;
        --run_id|--run-id)
            shift; RUN_ID="${1:-}"; shift ;;
        --output_dir|--output-dir)
            shift; BASE_DIR="${1:-$DEFAULT_BASE_DIR}"; shift ;;
        --conda_env|--conda-env)
            shift; CONDA_ENV="${1:-$DEFAULT_CONDA_ENV}"; shift ;;
        --task_catalog|--task-catalog)
            shift; TASK_CATALOG="${1:-$DEFAULT_TASK_CATALOG}"
            USER_ARGS+=("--task_catalog" "$TASK_CATALOG"); shift ;;
        --vm_data_dir|--vm-data-dir)
            shift; VM_DATA_DIR="${1:-$VM_DATA_DIR}"; shift ;;
        --domains)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                DOMAINS+=("$1"); shift
            done ;;
        # Eat the literal `--` separator. argparse in the python launcher
        # rejects bare `--` followed by `--flag value`, so swallow it
        # silently and forward everything that follows.
        --)
            shift; USER_ARGS+=("$@"); break ;;
        *)
            USER_ARGS+=("$1"); shift ;;
    esac
done

[ ${#DOMAINS[@]} -eq 0 ] && DOMAINS=("${DEFAULT_DOMAINS[@]}")
[ -z "$RUN_ID" ] && RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"

# Make the catalog default visible to USER_ARGS even if the user didn't pass it.
HAS_CATALOG_IN_USER_ARGS=0
for arg in "${USER_ARGS[@]:-}"; do
    case "$arg" in
        --task_catalog|--task-catalog|--task_catalog=*|--task-catalog=*)
            HAS_CATALOG_IN_USER_ARGS=1; break ;;
    esac
done
if [ "$HAS_CATALOG_IN_USER_ARGS" -eq 0 ]; then
    USER_ARGS+=("--task_catalog" "$TASK_CATALOG")
fi

RUN_DIR="${BASE_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/_logs"
META_FILE="${RUN_DIR}/_run_meta.json"

# ── Make conda usable ────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not on PATH. Cannot dispatch domains." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

ENV_LIST="$(conda env list | awk '$1 !~ /^#/ {print $1}')"
has_env() { printf '%s\n' "${ENV_LIST}" | grep -qx "$1"; }
if ! has_env "$CONDA_ENV"; then
    echo "[ERROR] conda env '$CONDA_ENV' not found. Available:"
    printf '%s\n' "${ENV_LIST}" | sed 's/^/  - /'
    echo "Install with:  bash install/install_osworld.sh"
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Headless rendering ───────────────────────────────────────────────────
export SDL_VIDEODRIVER=dummy
export PYGLET_HEADLESS=1
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1 && ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
        sleep 0.5
    fi
    if command -v Xvfb >/dev/null 2>&1; then
        export DISPLAY="${DISPLAY:-:99}"
    fi
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Docker / qcow2 preflight (warn only — OSWorld can self-heal) ─────────
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if docker image inspect happysixd/osworld-docker >/dev/null 2>&1; then
        DOCKER_STATUS="OK"
    else
        DOCKER_STATUS="WARN: image not pulled"
    fi
else
    DOCKER_STATUS="WARN: daemon unreachable"
fi
if [ -f "${VM_DATA_DIR}/Ubuntu.qcow2" ]; then
    QCOW2_STATUS="OK"
else
    QCOW2_STATUS="WARN: missing"
fi

# ── Print plan ───────────────────────────────────────────────────────────
echo "============================================================"
if [ "$PARALLEL" -eq 1 ]; then
    if [ "$MAX_PARALLEL" -gt 0 ]; then
        echo "  COS-PLAY Actor (OSWorld) — PARALLEL dispatch (cap=$MAX_PARALLEL)"
    else
        echo "  COS-PLAY Actor (OSWorld) — PARALLEL dispatch (unlimited)"
    fi
else
    echo "  COS-PLAY Actor (OSWorld) — SEQUENTIAL dispatch"
fi
echo "============================================================"
echo "  Run id:         $RUN_ID"
echo "  Run dir:        $RUN_DIR"
echo "  Per-domain logs: $LOG_DIR/<domain>.log"
echo "  Conda env:      $CONDA_ENV"
echo "  Catalog:        $TASK_CATALOG"
echo "  VM data dir:    $VM_DATA_DIR  (qcow2: $QCOW2_STATUS)"
echo "  Docker status:  $DOCKER_STATUS"
echo "  Forwarded args: ${USER_ARGS[*]:-<none>}"
echo
for domain in "${DOMAINS[@]}"; do
    printf "  %-22s -> %s/%s\n" "$domain" "$RUN_DIR" "$domain"
done
echo "============================================================"

# ── Per-domain runner ────────────────────────────────────────────────────
# We chdir into VM_PARENT_DIR so OSWorld's DockerVMManager (which looks
# for ./docker_vm_data relative to cwd at runtime) finds the qcow2.
VM_PARENT_DIR="$(dirname "${VM_DATA_DIR}")"
mkdir -p "${VM_DATA_DIR}"

run_domain() {
    local domain=$1
    local logfile="${LOG_DIR}/${domain}.log"

    (
        cd "${VM_PARENT_DIR}"
        PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGLET_HEADLESS=1 \
        conda run -n "$CONDA_ENV" --no-capture-output \
            python3 "$PY_LAUNCHER" \
                --domains "$domain" \
                --output_dir "$RUN_DIR" \
                "${USER_ARGS[@]}" \
            > "$logfile" 2>&1
    )
}

declare -A RC

# ── Dispatch ─────────────────────────────────────────────────────────────
START_TS="$(date +%s)"
if [ "$PARALLEL" -eq 1 ]; then
    declare -A PIDS
    declare -a INFLIGHT_DOMAINS
    INFLIGHT_DOMAINS=()
    wait_for_next_domain() {
        local done_domain="${INFLIGHT_DOMAINS[0]}"
        INFLIGHT_DOMAINS=("${INFLIGHT_DOMAINS[@]:1}")
        wait "${PIDS[$done_domain]}"
        local rc=$?
        RC[$done_domain]=$rc
        printf "  [DONE]   %-22s rc=%d\n" "$done_domain" "$rc"
    }
    echo
    echo "Dispatching ${#DOMAINS[@]} domain(s) in parallel:"
    for domain in "${DOMAINS[@]}"; do
        if [ "$MAX_PARALLEL" -gt 0 ]; then
            while [ "${#INFLIGHT_DOMAINS[@]}" -ge "$MAX_PARALLEL" ]; do
                wait_for_next_domain
            done
        fi

        run_domain "$domain" &
        PIDS[$domain]=$!
        INFLIGHT_DOMAINS+=("$domain")
        printf "  [START]  %-22s pid=%-8s log=%s\n" \
            "$domain" "${PIDS[$domain]}" "${LOG_DIR}/${domain}.log"
    done
    echo
    echo "Live tail (any of these in another terminal):"
    for domain in "${DOMAINS[@]}"; do
        echo "  tail -f ${LOG_DIR}/${domain}.log"
    done
    echo
    echo "Waiting for completion ..."
    while [ "${#INFLIGHT_DOMAINS[@]}" -gt 0 ]; do
        wait_for_next_domain
    done
else
    for domain in "${DOMAINS[@]}"; do
        echo
        echo ">>> [SEQUENTIAL] $domain"
        run_domain "$domain"
        rc=$?
        RC[$domain]=$rc
        tail -n 12 "${LOG_DIR}/${domain}.log" 2>/dev/null | sed 's/^/    /'
        echo "    rc=$rc"
    done
fi
END_TS="$(date +%s)"
ELAPSED=$((END_TS - START_TS))

# ── Update `latest` symlink ──────────────────────────────────────────────
ln -sfn "$RUN_ID" "${BASE_DIR}/latest" 2>/dev/null || true

# ── Write meta ───────────────────────────────────────────────────────────
{
    printf '{\n'
    printf '  "run_id": "%s",\n' "$RUN_ID"
    printf '  "started_at_unix": %s,\n' "$START_TS"
    printf '  "ended_at_unix": %s,\n' "$END_TS"
    printf '  "elapsed_seconds": %s,\n' "$ELAPSED"
    printf '  "mode": "%s",\n' "$([ "$PARALLEL" -eq 1 ] && echo parallel || echo sequential)"
    printf '  "max_parallel": %s,\n' "${MAX_PARALLEL:-0}"
    printf '  "conda_env": "%s",\n' "$CONDA_ENV"
    printf '  "task_catalog": "%s",\n' "$TASK_CATALOG"
    printf '  "vm_data_dir": "%s",\n' "$VM_DATA_DIR"
    printf '  "forwarded_args": "%s",\n' "${USER_ARGS[*]:-}"
    printf '  "domains": {\n'
    first=1
    for domain in "${DOMAINS[@]}"; do
        rc=${RC[$domain]:-null}
        [ $first -eq 1 ] || printf ',\n'
        printf '    "%s": {"rc": %s}' "$domain" "$rc"
        first=0
    done
    printf '\n  }\n'
    printf '}\n'
} > "$META_FILE"

# ── Summary ──────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  Combined run finished"
echo "============================================================"
echo "  Run id:    $RUN_ID"
echo "  Elapsed:   ${ELAPSED}s ($((ELAPSED / 60)) min)"
ANY_OK=0
for domain in "${DOMAINS[@]}"; do
    rc=${RC[$domain]:-?}
    domain_dir="$RUN_DIR/$domain"
    count=0
    tasks=0
    if [ -d "$domain_dir" ]; then
        # Count episodes across all tasks within the domain.
        count=$(find "$domain_dir" -mindepth 2 -maxdepth 2 -name 'episode_*.json' \
            ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
        tasks=$(find "$domain_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    fi
    printf "  %-22s rc=%-3s tasks=%-3s episodes=%-3s out=%s/\n" \
        "$domain" "$rc" "$tasks" "$count" "$domain"
    [ "$rc" = "0" ] && ANY_OK=1
done
echo "  Run dir:   $RUN_DIR/"
echo "  Latest:    $BASE_DIR/latest -> $RUN_ID"
echo "  Meta:      $META_FILE"
echo "============================================================"

# Non-zero only if NO domain succeeded.
if [ "$ANY_OK" -eq 0 ]; then
    exit 1
fi
exit 0
