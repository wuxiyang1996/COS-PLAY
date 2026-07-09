#!/usr/bin/env bash
# ======================================================================
#  Baseline evaluation:  gpt-5.4 as the playing agent on all 6 games
#  using the BEST-STEP skill banks discovered by prior co-evolution.
#
#  Per game: 16 episodes (configurable via EPISODES env).
#
#  Harnesses used:
#    - 2048 / candy_crush / tetris        : scripts/qwen3_decision_agent.py
#                                            (single-player LMGAME-bench)
#    - super_mario                         : evaluate_orak/test_orak_mario_sc2_gpt54.py
#                                            (no skill-bank integration in
#                                            harness — emits gpt-5.4 baseline
#                                            without bank for mario)
#    - avalon                              : scripts/run_qwen3_avalon_matched.py
#                                            (--model gpt-5.4 --opponent_model gpt-5.4
#                                             with combined good+evil bank)
#    - diplomacy                           : scripts/run_diplomacy_discrete_eval.py
#                                            (same matched mode with combined per-power bank)
#
#  All LLM calls route through OpenRouter via API_func.ask_model;
#  OPENROUTER_API_KEY is loaded from /workspace/keys.py if not in env.
#
#  No local GPUs are used.  Safe to run alongside an active GRPO sweep.
#
#  Usage:
#    bash ablation_study/run_gpt54_eval_with_bestbank.sh
#    bash ablation_study/run_gpt54_eval_with_bestbank.sh tetris avalon
#    EPISODES=8 bash ablation_study/run_gpt54_eval_with_bestbank.sh
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
        echo "[gpt54-eval] activated conda env: ${CONDA_ENV}"
    fi
fi

# ── 1. Load API keys ─────────────────────────────────────────────────
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
            echo "[gpt54-eval] loaded ${var} from /workspace/keys.py"
        fi
    fi
}
load_key OPENROUTER_API_KEY  openrouter_api_key
load_key OPENAI_API_KEY      openai_api_key

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[gpt54-eval] ERROR: OPENROUTER_API_KEY not set; cannot reach gpt-5.4." >&2
    exit 1
fi

# Force RAG embedder to CPU so we don't compete with GRPO training for GPU memory.
export RAG_EMBEDDER_DEVICE="${RAG_EMBEDDER_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES=""           # entire script CPU-only
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

# ── 3. Log directory ────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/gpt54_eval_with_bestbank/${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/SUMMARY.txt"
: > "${SUMMARY}"

echo "══════════════════════════════════════════════════════════════"
echo "  gpt-5.4 eval over 6 games with best-step skills"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:        ${MODEL}"
echo "  Episodes:     ${EPISODES} per game"
echo "  Games:        ${GAMES_LIST[*]}"
echo "  Seed bank:    ${SEED_BANK_DIR}"
echo "  Log root:     ${LOG_ROOT}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── 4. Per-game runner ──────────────────────────────────────────────
run_game() {
    local game="$1"
    local log="$2"
    local out="${LOG_ROOT}/${game}/output"
    mkdir -p "${out}"

    case "${game}" in
        twenty_forty_eight|candy_crush|tetris)
            local bank="${SEED_BANK_DIR}/${game}/skill_bank.jsonl"
            if [ ! -f "${bank}" ]; then
                echo "[gpt54-eval] missing bank: ${bank}" >&2
                return 1
            fi
            python -m scripts.qwen3_decision_agent \
                --games "${game}" \
                --episodes "${EPISODES}" \
                --model "${MODEL}" \
                --temperature 0.4 \
                --bank "${bank}" \
                --output_dir "${out}" \
                -v \
                2>&1 | tee -a "${log}"
            return ${PIPESTATUS[0]}
            ;;

        super_mario)
            # No skill-bank-aware gpt-5.4 harness for mario — fall back to
            # the orak gpt-5.4 baseline (no bank).
            echo "[gpt54-eval] super_mario: using bank-less gpt-5.4 baseline" \
                | tee -a "${log}"
            python evaluate_orak/test_orak_mario_sc2_gpt54.py \
                --game super_mario \
                --episodes "${EPISODES}" \
                --max_steps 200 \
                --model "${MODEL}" \
                --save_episode_buffer "${out}/episodes.json" \
                2>&1 | tee -a "${log}"
            return ${PIPESTATUS[0]}
            ;;

        avalon)
            local bank="${SEED_BANK_DIR}/avalon/combined_skill_bank.jsonl"
            if [ ! -f "${bank}" ]; then
                echo "[gpt54-eval] missing bank: ${bank}" >&2
                return 1
            fi
            python -m scripts.run_qwen3_avalon_matched \
                --model "${MODEL}" \
                --opponent_model "${MODEL}" \
                --episodes "${EPISODES}" \
                --temperature 0.3 \
                --bank "${bank}" \
                --output_dir "${out}" \
                2>&1 | tee -a "${log}"
            return ${PIPESTATUS[0]}
            ;;

        diplomacy)
            local bank="${SEED_BANK_DIR}/diplomacy/combined_skill_bank.jsonl"
            if [ ! -f "${bank}" ]; then
                echo "[gpt54-eval] missing bank: ${bank}" >&2
                return 1
            fi
            python -m scripts.run_diplomacy_discrete_eval \
                --model "${MODEL}" \
                --opponent_model "${MODEL}" \
                --episodes "${EPISODES}" \
                --bank "${bank}" \
                --output_dir "${out}" \
                2>&1 | tee -a "${log}"
            return ${PIPESTATUS[0]}
            ;;

        *)
            echo "[gpt54-eval] unknown game: ${game}" >&2
            return 2
            ;;
    esac
}

# ── 5. Loop ─────────────────────────────────────────────────────────
OVERALL_RC=0
for game in "${GAMES_LIST[@]}"; do
    log="${LOG_ROOT}/${game}.log"
    : > "${log}"
    start=$(date +%s)
    when="$(date '+%F %T')"

    echo "──────────────────────────────────────────────────────────────"
    echo "  [${when}] GAME = ${game}    log = ${log}"
    echo "──────────────────────────────────────────────────────────────"

    set +e
    run_game "${game}" "${log}"
    rc=$?
    set -e

    elapsed=$(( $(date +%s) - start ))
    H=$(( elapsed / 3600 )); M=$(( (elapsed % 3600) / 60 )); S=$(( elapsed % 60 ))
    if [ ${rc} -eq 0 ]; then
        status="OK"
    else
        status="FAIL(${rc})"
        OVERALL_RC=${rc}
    fi
    printf "%-22s  %-10s  %02d:%02d:%02d  %s\n" \
        "${game}" "${status}" "${H}" "${M}" "${S}" "${log}" | tee -a "${SUMMARY}"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Eval sweep complete.  Summary:"
echo "    ${SUMMARY}"
echo "══════════════════════════════════════════════════════════════"
cat "${SUMMARY}"
exit ${OVERALL_RC}
