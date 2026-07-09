#!/usr/bin/env bash
# ======================================================================
#  Parallel baseline evaluation: gpt-5.4 as the playing agent
#  on all 6 games using BEST-STEP skill banks.
#
#  All 6 games run concurrently as independent CPU+API workers.
#  Each LLM call goes to OpenRouter (no local GPU).
#
#  Per-game harnesses & seed banks:
#    - 2048 / candy_crush / tetris  -> scripts/qwen3_decision_agent.py
#         with --bank runs/fixed_skillbank_seeds/<game>/skill_bank.jsonl
#    - super_mario                   -> evaluate_orak/test_orak_mario_sc2_gpt54.py
#         (no bank integration in this harness; runs pure gpt-5.4 baseline)
#    - avalon                        -> scripts/run_qwen3_avalon_matched.py
#         --model gpt-5.4 --opponent_model gpt-5.4 --bank combined_skill_bank.jsonl
#    - diplomacy                     -> scripts/run_diplomacy_discrete_eval.py
#         --model gpt-5.4 --opponent_model gpt-5.4 --bank combined_skill_bank.jsonl
#
#  Usage:
#    bash ablation_study/run_gpt54_eval_parallel.sh
#    GAMES="twenty_forty_eight tetris" bash ablation_study/run_gpt54_eval_parallel.sh
#    EPISODES=8 bash ablation_study/run_gpt54_eval_parallel.sh
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 0. Activate conda env ────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]; then
    if [ -f /workspace/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /workspace/miniconda3/etc/profile.d/conda.sh
        conda activate "${CONDA_ENV}"
        echo "[gpt54-parallel] activated conda env: ${CONDA_ENV}"
    fi
fi

# ── 1. Load API keys from /workspace/keys.py ─────────────────────────
load_key() {
    local var="$1" attr="$2"
    if [ -z "${!var:-}" ] && [ -f /workspace/keys.py ]; then
        local v
        v="$(python - <<PY
import importlib.util
spec = importlib.util.spec_from_file_location("keys", "/workspace/keys.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(getattr(m, "${attr}", ""))
PY
)"
        if [ -n "${v}" ]; then
            export "${var}=${v}"
            echo "[gpt54-parallel] loaded ${var} from /workspace/keys.py"
        fi
    fi
}
load_key OPENROUTER_API_KEY  openrouter_api_key
load_key OPENAI_API_KEY      openai_api_key

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[gpt54-parallel] ERROR: OPENROUTER_API_KEY not set; cannot reach gpt-5.4." >&2
    exit 1
fi

# CPU-only, no GPU contention with GRPO training
export CUDA_VISIBLE_DEVICES=""
export RAG_EMBEDDER_DEVICE="${RAG_EMBEDDER_DEVICE:-cpu}"
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

MODEL="${MODEL:-gpt-5.4}"
EPISODES="${EPISODES:-16}"
SEED_BANK_DIR="${SEED_BANK_DIR:-${PROJECT_ROOT}/runs/fixed_skillbank_seeds}"

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

# ── 3. Output root ───────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/gpt54_eval_with_bestbank/parallel_${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/SUMMARY.txt"
PIDS_FILE="${LOG_ROOT}/PIDS.txt"
: > "${SUMMARY}"
: > "${PIDS_FILE}"

echo "══════════════════════════════════════════════════════════════"
echo "  gpt-5.4 PARALLEL eval — 6 games × ${EPISODES} episodes"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:        ${MODEL}"
echo "  Games:        ${GAMES_LIST[*]}"
echo "  Seed bank:    ${SEED_BANK_DIR}"
echo "  Log root:     ${LOG_ROOT}"
echo "══════════════════════════════════════════════════════════════"

# ── 4. Build per-game command (string for nohup wrapping) ────────────
build_cmd() {
    local game="$1"
    local out_dir="$2"
    case "${game}" in
        twenty_forty_eight|candy_crush|tetris)
            local bank="${SEED_BANK_DIR}/${game}/skill_bank.jsonl"
            echo python -m scripts.qwen3_decision_agent \
                --games "${game}" \
                --episodes "${EPISODES}" \
                --model "${MODEL}" \
                --temperature 0.4 \
                --bank "${bank}" \
                --output_dir "${out_dir}" \
                -v
            ;;
        super_mario)
            echo python evaluate_orak/test_orak_mario_sc2_gpt54.py \
                --game super_mario \
                --episodes "${EPISODES}" \
                --max_steps 200 \
                --model "${MODEL}" \
                --save_episode_buffer "${out_dir}/episodes.json"
            ;;
        avalon)
            local bank="${SEED_BANK_DIR}/avalon/combined_skill_bank.jsonl"
            echo python -m scripts.run_qwen3_avalon_matched \
                --model "${MODEL}" \
                --opponent_model "${MODEL}" \
                --episodes "${EPISODES}" \
                --temperature 0.3 \
                --bank "${bank}" \
                --output_dir "${out_dir}"
            ;;
        diplomacy)
            local bank="${SEED_BANK_DIR}/diplomacy/combined_skill_bank.jsonl"
            echo python -m scripts.run_diplomacy_discrete_eval \
                --model "${MODEL}" \
                --opponent_model "${MODEL}" \
                --episodes "${EPISODES}" \
                --bank "${bank}" \
                --output_dir "${out_dir}"
            ;;
    esac
}

# ── 5. Fan out workers ───────────────────────────────────────────────
declare -A GAME_PID
for game in "${GAMES_LIST[@]}"; do
    out_dir="${LOG_ROOT}/${game}/output"
    log="${LOG_ROOT}/${game}.log"
    mkdir -p "${out_dir}"
    : > "${log}"

    cmd="$(build_cmd "${game}" "${out_dir}")"
    if [ -z "${cmd}" ]; then
        echo "[gpt54-parallel] WARN: no command for game '${game}', skipping"
        continue
    fi

    echo "[gpt54-parallel] launching ${game}"
    echo "  cmd: ${cmd}" | tee -a "${log}"
    # Wrap inside a subshell so each worker gets its own env tweaks and PID
    setsid nohup bash -c "${cmd}" </dev/null >>"${log}" 2>&1 &
    pid=$!
    GAME_PID[${game}]=${pid}
    echo "${game} ${pid}" >> "${PIDS_FILE}"
    disown ${pid} 2>/dev/null || true
done

echo ""
echo "── Worker PIDs ──"
for g in "${!GAME_PID[@]}"; do
    printf "  %-22s  PID=%s\n" "${g}" "${GAME_PID[${g}]}"
done | sort
echo ""
echo "Logs:    ${LOG_ROOT}/<game>.log"
echo "PIDS:    ${PIDS_FILE}"
echo "Summary: ${SUMMARY}  (will be filled when all workers finish)"
echo "══════════════════════════════════════════════════════════════"

# ── 6. Wait for all workers and write summary ────────────────────────
OVERALL_RC=0
for g in "${!GAME_PID[@]}"; do
    pid=${GAME_PID[${g}]}
    start_ts=$(stat -c %Y "${LOG_ROOT}/${g}.log" 2>/dev/null || date +%s)
    wait ${pid}
    rc=$?
    elapsed=$(( $(date +%s) - start_ts ))
    H=$(( elapsed / 3600 )); M=$(( (elapsed % 3600) / 60 )); S=$(( elapsed % 60 ))
    if [ ${rc} -eq 0 ]; then status="OK"; else status="FAIL(${rc})"; OVERALL_RC=${rc}; fi
    printf "%-22s  %-10s  %02d:%02d:%02d  PID=%s\n" \
        "${g}" "${status}" "${H}" "${M}" "${S}" "${pid}" | tee -a "${SUMMARY}"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Parallel eval sweep complete.  Summary:"
echo "    ${SUMMARY}"
echo "══════════════════════════════════════════════════════════════"
cat "${SUMMARY}"
exit ${OVERALL_RC}
