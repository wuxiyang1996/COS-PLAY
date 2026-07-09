#!/usr/bin/env bash
# ======================================================================
#  Wait for the "GRPO w/o skill bank" sweep to finish, then launch the
#  "GRPO w/ fixed skill bank" sweep automatically.
#
#  Usage:
#    setsid nohup bash ablation_study/queue_fixed_after_no_skill.sh \
#        </dev/null >runs/ablation_queue.log 2>&1 &
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

NO_SKILL_PID_FILE="${PROJECT_ROOT}/runs/no_skillbank_ablation_logs/orchestrator.pid"
FIXED_PID_FILE="${PROJECT_ROOT}/runs/fixed_skillbank_ablation_logs/orchestrator.pid"

# ── 1. Wait for the upstream sweep to finish ─────────────────────────
if [ ! -f "${NO_SKILL_PID_FILE}" ]; then
    echo "[queue] no upstream PID file found at ${NO_SKILL_PID_FILE}"
    echo "[queue] launching fixed-bank sweep immediately."
else
    UPSTREAM_PID=$(cat "${NO_SKILL_PID_FILE}")
    echo "[queue] $(date '+%F %T') waiting for no-skillbank sweep (PID ${UPSTREAM_PID})..."

    POLL_INTERVAL=60   # seconds
    while kill -0 "${UPSTREAM_PID}" 2>/dev/null; do
        sleep "${POLL_INTERVAL}"
    done

    echo "[queue] $(date '+%F %T') no-skillbank sweep finished."
    sleep 30
fi

# ── 2. Sanity-check GPUs are free ───────────────────────────────────
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 1000' | wc -l)
if [ "${USED}" -gt 0 ]; then
    echo "[queue] WARNING: ${USED} GPUs still hold >1GB.  Cleaning..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 "${pid}" 2>/dev/null || true
    done
    sleep 15
fi

# ── 3. Launch the fixed-skillbank sweep ─────────────────────────────
mkdir -p "${PROJECT_ROOT}/runs/fixed_skillbank_ablation_logs"
ORCH_LOG="${PROJECT_ROOT}/runs/fixed_skillbank_ablation_logs/orchestrator_$(date +%Y%m%d_%H%M%S).log"
echo "[queue] launching fixed-bank sweep: log = ${ORCH_LOG}"

setsid nohup bash "${SCRIPT_DIR}/run_all_fixed_skillbank.sh" \
    </dev/null > "${ORCH_LOG}" 2>&1 &
ORCH_PID=$!
disown ${ORCH_PID} 2>/dev/null || true
echo "${ORCH_PID}" > "${FIXED_PID_FILE}"
echo "[queue] fixed-bank sweep PID: ${ORCH_PID}"
echo "[queue] $(date '+%F %T') queue handoff complete; exiting."
