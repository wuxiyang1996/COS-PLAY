#!/usr/bin/env bash
#
# install_workarena.sh — install BrowserGym's WorkArena suite.
#
# WorkArena tests an agent on enterprise ServiceNow workflows
# (forms / dashboards / catalogs / etc., 716 tasks). Unlike
# WebArena/VWA which self-host their backend, WorkArena runs against a
# real ServiceNow Personal Developer Instance (PDI) — a free hosted
# tenant you provision yourself at developer.servicenow.com. This
# script handles the local Python install; the user must supply the
# remote PDI credentials.
#
# Usage:
#   bash install/install_workarena.sh                  # python install only
#   bash install/install_workarena.sh --setup-instance  # also seed the PDI
#                                                       (requires SNOW_* env vars)
#
# Prerequisites:
#   1. ``conda activate browsergym``  (or ``CONDA_ENV=browsergym``)
#   2. For runtime (NOT install):
#        a. HuggingFace account with access granted to
#           ``ServiceNow/WorkArena-Instances`` (gated repo). Run
#           ``huggingface-cli login`` or set ``HUGGING_FACE_HUB_TOKEN``.
#        b. ServiceNow PDI provisioned at developer.servicenow.com,
#           with these env vars exported:
#             SNOW_INSTANCE_URL=https://devXXXXXX.service-now.com
#             SNOW_INSTANCE_UNAME=admin
#             SNOW_INSTANCE_PWD=YOUR_PWD
#        c. ``workarena-install`` CLI run once to seed the PDI
#           (~30 minutes; downloads catalogs, populates users, etc.).
#

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-browsergym}"
SETUP_INSTANCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --setup-instance) SETUP_INSTANCE=1 ;;
        --conda-env|--conda_env) shift; CONDA_ENV="${1:-$CONDA_ENV}" ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0 ;;
        *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# ── Activate conda env ────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda env list | awk '$1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV"; then
        conda activate "$CONDA_ENV"
        echo "[INFO] activated conda env: $CONDA_ENV"
    else
        echo "[ERROR] conda env '$CONDA_ENV' not found. Run install/install_browsergym.sh first." >&2
        exit 2
    fi
else
    echo "[WARN] conda not on PATH; using current python ($(python -V))" >&2
fi

# ── 1. Install browsergym-workarena and its non-pinned deps ──────────────
# Why ``--no-deps`` for browsergym-workarena itself: the wheel pins
# ``tqdm>=4.66.2`` which clashes with other stacks in the conda env.
# We install the OTHER required deps explicitly to avoid the conflict.
echo
echo "[1/3] Installing browsergym-workarena + missing runtime deps"
pip install --no-deps browsergym-workarena || { echo "[FAIL] pip install"; exit 3; }
pip install 'faker>=24.8.0' 'english-words>=2.0.1' || { echo "[FAIL] pip install deps"; exit 3; }

# ── 2. Verify import + task registration ─────────────────────────────────
echo
echo "[2/3] Verifying import + task registration"
python - <<'PY' || { echo "[FAIL] import / registration check"; exit 3; }
import browsergym.workarena  # noqa: F401
import gymnasium as gym
n = sum(1 for k in gym.envs.registry if k.startswith("browsergym/workarena"))
print(f"  registered tasks: {n}")
if n < 100:
    raise SystemExit(f"  expected hundreds of tasks, only got {n}")
PY

# ── 3. Optionally seed the ServiceNow PDI ────────────────────────────────
if [ "$SETUP_INSTANCE" -eq 1 ]; then
    echo
    echo "[3/3] Seeding ServiceNow PDI via 'workarena-install'"
    : "${SNOW_INSTANCE_URL:?set SNOW_INSTANCE_URL=https://devXXXXXX.service-now.com}"
    : "${SNOW_INSTANCE_UNAME:?set SNOW_INSTANCE_UNAME (usually 'admin')}"
    : "${SNOW_INSTANCE_PWD:?set SNOW_INSTANCE_PWD}"
    if ! command -v huggingface-cli >/dev/null 2>&1 || ! huggingface-cli whoami 2>&1 | grep -q '@'; then
        echo "[WARN] HuggingFace auth not configured. workarena-install will fail" >&2
        echo "       on the gated 'ServiceNow/WorkArena-Instances' dataset. Run:" >&2
        echo "         huggingface-cli login" >&2
        echo "       and grant access to the repo at" >&2
        echo "         https://huggingface.co/datasets/ServiceNow/WorkArena-Instances" >&2
    fi
    echo "  this seeds catalog + users + dashboards into your PDI (~30 min)..."
    workarena-install
else
    echo
    echo "[3/3] SKIPPED: ServiceNow PDI seeding (run with --setup-instance)"
    echo "  WorkArena tasks are registered but env.reset() will fail until you"
    echo "  provision a ServiceNow PDI and run:"
    echo "    SNOW_INSTANCE_URL=https://devXXXXXX.service-now.com \\"
    echo "    SNOW_INSTANCE_UNAME=admin \\"
    echo "    SNOW_INSTANCE_PWD=... \\"
    echo "    bash install/install_workarena.sh --setup-instance"
fi

echo
echo "================================================================"
echo "  WorkArena install: PYTHON SIDE COMPLETE"
echo "================================================================"
echo "  - browsergym-workarena $(pip show browsergym-workarena | awk '/^Version:/ {print $2}') installed"
echo "  - $(python -c 'import gymnasium as gym, browsergym.workarena; print(sum(1 for k in gym.envs.registry if k.startswith(\"browsergym/workarena\")))') tasks registered"
if [ "$SETUP_INSTANCE" -eq 0 ]; then
    echo "  - PDI: NOT yet seeded (see steps above)"
fi
