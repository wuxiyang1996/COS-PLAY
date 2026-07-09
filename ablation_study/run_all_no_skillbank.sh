#!/usr/bin/env bash
# ======================================================================
#  GRPO w/o Skill Bank — sequential runner for ALL games (one by one).
#
#  - Sources OPENROUTER_API_KEY from /workspace/keys.py so Avalon /
#    Diplomacy can talk to gpt-5-mini opponents.
#  - Warm-starts every game from the same SFT cold-start adapters at
#    Game-AI-Agent/runs/sft_coldstart/{decision,skillbank} (symlinks to
#    /workspace/main_project/runs/sft_coldstart/).
#  - Runs each game in its own log file under
#    Game-AI-Agent/runs/no_skillbank_ablation_logs/<timestamp>/.
#  - On per-game failure: continues to the next game by default.
#    Set STOP_ON_FAIL=1 to abort the whole sweep.
#
#  Usage:
#    bash ablation_study/run_all_no_skillbank.sh                 # all 6
#    bash ablation_study/run_all_no_skillbank.sh tetris avalon   # subset
#    GAMES="twenty_forty_eight tetris" \
#      bash ablation_study/run_all_no_skillbank.sh
#    STOP_ON_FAIL=1 bash ablation_study/run_all_no_skillbank.sh
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 0. Activate conda env (game-ai-agent) ─────────────────────────────
CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]; then
    if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /workspace/miniconda3/etc/profile.d/conda.sh
        conda activate "${CONDA_ENV}"
        echo "[orchestrator] activated conda env: ${CONDA_ENV} ($(python -V 2>&1))"
    else
        echo "[orchestrator] WARNING: conda.sh not found; using current python"
    fi
fi

# ── 1. Load OpenRouter API key ────────────────────────────────────────
#  Prefer /workspace/keys.py (user's location); fall back to env.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f /workspace/keys.py ]; then
    KEY="$(python - <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("keys", "/workspace/keys.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(getattr(m, "openrouter_api_key", ""))
PY
)"
    if [ -n "${KEY}" ]; then
        export OPENROUTER_API_KEY="${KEY}"
        echo "[orchestrator] OPENROUTER_API_KEY loaded from /workspace/keys.py"
    fi
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[orchestrator] WARNING: OPENROUTER_API_KEY is not set; Avalon and"
    echo "[orchestrator]          Diplomacy will fail to reach gpt-5-mini."
fi

# Same fallback for OpenAI (some opponent paths read it).
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f /workspace/keys.py ]; then
    KEY="$(python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("keys", "/workspace/keys.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(getattr(m, "openai_api_key", ""))
PY
)"
    [ -n "${KEY}" ] && export OPENAI_API_KEY="${KEY}"
fi

# ── 2. Game list ─────────────────────────────────────────────────────
ALL_GAMES=(twenty_forty_eight candy_crush tetris super_mario avalon diplomacy)
if [ $# -gt 0 ]; then
    GAMES_LIST=("$@")
elif [ -n "${GAMES:-}" ]; then
    # shellcheck disable=SC2206
    GAMES_LIST=(${GAMES})
else
    GAMES_LIST=("${ALL_GAMES[@]}")
fi

# ── 3. SFT prerequisite check ────────────────────────────────────────
SFT_DIR="${SFT_DIR:-${PROJECT_ROOT}/runs/sft_coldstart}"
for n in skill_selection action_taking; do
    f="${SFT_DIR}/decision/${n}/adapter_config.json"
    if [ ! -f "${f}" ]; then
        echo "ERROR: SFT cold-start adapter missing: ${f}" >&2
        exit 1
    fi
done

# ── 4. Log directory ────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/no_skillbank_ablation_logs/${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/SUMMARY.txt"
: > "${SUMMARY}"

STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

echo "══════════════════════════════════════════════════════════════"
echo "  GRPO w/o Skill Bank — sequential ablation sweep"
echo "══════════════════════════════════════════════════════════════"
echo "  Games:        ${GAMES_LIST[*]}"
echo "  SFT init:     ${SFT_DIR}"
echo "  Log dir:      ${LOG_ROOT}"
echo "  Stop on fail: ${STOP_ON_FAIL}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── 5. Loop over games ──────────────────────────────────────────────
OVERALL_RC=0
for GAME in "${GAMES_LIST[@]}"; do
    LOG="${LOG_ROOT}/${GAME}.log"
    START_TS="$(date +%s)"
    START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "──────────────────────────────────────────────────────────────"
    echo "  [${START_HUMAN}]  GAME = ${GAME}"
    echo "  log: ${LOG}"
    echo "──────────────────────────────────────────────────────────────"

    set +e
    bash "${SCRIPT_DIR}/run_grpo_no_skillbank.sh" "${GAME}" \
        > >(tee -a "${LOG}") 2> >(tee -a "${LOG}" >&2)
    RC=$?
    set -e

    END_TS="$(date +%s)"
    ELAPSED=$(( END_TS - START_TS ))
    H=$(( ELAPSED / 3600 ))
    M=$(( (ELAPSED % 3600) / 60 ))
    S=$(( ELAPSED % 60 ))

    if [ ${RC} -eq 0 ]; then
        STATUS="OK"
    else
        STATUS="FAIL(${RC})"
        OVERALL_RC=${RC}
    fi
    printf "%-22s  %-10s  %02d:%02d:%02d  %s\n" \
        "${GAME}" "${STATUS}" "${H}" "${M}" "${S}" "${LOG}" | tee -a "${SUMMARY}"

    if [ ${RC} -ne 0 ] && [ "${STOP_ON_FAIL}" = "1" ]; then
        echo "[orchestrator] STOP_ON_FAIL=1 and ${GAME} failed (rc=${RC}); aborting." | tee -a "${SUMMARY}"
        break
    fi

    # Brief pause so vLLM ports / GPU memory get fully released.
    sleep 10
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Sweep complete.  Summary written to:"
echo "    ${SUMMARY}"
echo "══════════════════════════════════════════════════════════════"
cat "${SUMMARY}"

exit ${OVERALL_RC}
