#!/usr/bin/env bash
# ======================================================================
#  Inference: Decision Agent WITH Skill Bank (full system)
#
#  Runs the trained decision agent (action_taking LoRA adapter) together
#  with the skill bank from the best co-evolution checkpoint.
#
#  This is the full-system counterpart to run_no_skillbank_ablation.sh.
#  Comparing the two scripts quantifies the skill bank's contribution.
#
#  Architecture:
#    1. Launch ONE vLLM server with Qwen3-8B + all 6 game adapters
#    2. For each game, prepare the skill bank (merge sub-banks if needed)
#    3. Run inference with --bank pointing to the merged skill bank
#    4. Collect results and produce a comparison summary
#
#  Usage:
#    conda activate game-ai-agent
#    bash ablation_study/run_with_skillbank.sh
#
#    # With overrides:
#    EVAL_GPUS=0 bash ablation_study/run_with_skillbank.sh
#    EPISODES=4  bash ablation_study/run_with_skillbank.sh
#    GAMES="twenty_forty_eight tetris" bash ablation_study/run_with_skillbank.sh
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── Xvfb for NES rendering (Super Mario) ─────────────────────────────
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        XVFB_DISPLAY=":99"
        if ! pgrep -f "Xvfb ${XVFB_DISPLAY}" &>/dev/null; then
            echo "[full-system] Starting Xvfb on ${XVFB_DISPLAY}..."
            Xvfb "${XVFB_DISPLAY}" -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY="${XVFB_DISPLAY}"
    fi
fi

# ── Subprocess env for Super Mario ───────────────────────────────────
export ORAK_PYTHON="${ORAK_PYTHON:-/workspace/miniconda3/envs/orak-mario/bin/python}"

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PROJECT_ROOT}/../Orak/src:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
EVAL_GPUS="${EVAL_GPUS:-0}"
VLLM_PORT="${VLLM_PORT:-8020}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-8}"
TEMPERATURE="${TEMPERATURE:-0.3}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Runs directory ───────────────────────────────────────────────────
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"

# ══════════════════════════════════════════════════════════════════════
#  Best checkpoint map
#
#  Selected by highest mean_reward among available checkpoints:
#    avalon:      step 5  (mean_reward=0.9950) — final-state adapters
#    candy_crush: step 9  (mean_reward=657.75)
#    tetris:      step 12 (mean_reward=331.50)
#    super_mario: step 11 (mean_reward=930.00)
#    diplomacy:   step 22 (mean_reward=5.0357)
#    2048:        step 5  (mean_reward=1407.50)
# ══════════════════════════════════════════════════════════════════════

declare -A GAME_ADAPTER_PATH
declare -A GAME_BANK_DIR
declare -A GAME_EVAL_NAME
declare -A GAME_EPISODES
declare -A GAME_MAX_STEPS
declare -A GAME_TEMPERATURE
declare -A GAME_EXTRA_ARGS

# --- Avalon ---
GAME_ADAPTER_PATH[avalon]="${RUNS_DIR}/Qwen3-8B_avalon_20260322_200424/best/adapters/decision/action_taking"
GAME_BANK_DIR[avalon]="${RUNS_DIR}/Qwen3-8B_avalon_20260322_200424/best/banks/avalon"
GAME_EVAL_NAME[avalon]="avalon"
GAME_EPISODES[avalon]="40"
GAME_MAX_STEPS[avalon]=""
GAME_TEMPERATURE[avalon]="0.4"
GAME_EXTRA_ARGS[avalon]="--num_players 5"

# --- Candy Crush ---
GAME_ADAPTER_PATH[candy_crush]="${RUNS_DIR}/Qwen3-8B_20260321_213813_(Candy_crush)/best/adapters/decision/action_taking"
GAME_BANK_DIR[candy_crush]="${RUNS_DIR}/Qwen3-8B_20260321_213813_(Candy_crush)/best/banks/candy_crush"
GAME_EVAL_NAME[candy_crush]="candy_crush"
GAME_EPISODES[candy_crush]="${EPISODES}"
GAME_MAX_STEPS[candy_crush]="200"
GAME_TEMPERATURE[candy_crush]="${TEMPERATURE}"
GAME_EXTRA_ARGS[candy_crush]=""

# --- Tetris ---
GAME_ADAPTER_PATH[tetris]="${RUNS_DIR}/Qwen3-8B_tetris_20260322_170438/best/adapters/decision/action_taking"
GAME_BANK_DIR[tetris]="${RUNS_DIR}/Qwen3-8B_tetris_20260322_170438/best/banks/tetris"
GAME_EVAL_NAME[tetris]="tetris"
GAME_EPISODES[tetris]="${EPISODES}"
GAME_MAX_STEPS[tetris]="200"
GAME_TEMPERATURE[tetris]="${TEMPERATURE}"
GAME_EXTRA_ARGS[tetris]="--macro-actions"

# --- Super Mario ---
GAME_ADAPTER_PATH[super_mario]="${RUNS_DIR}/Qwen3-8B_super_mario_20260323_030839/best/adapters/decision/action_taking"
GAME_BANK_DIR[super_mario]="${RUNS_DIR}/Qwen3-8B_super_mario_20260323_030839/best/banks/super_mario"
GAME_EVAL_NAME[super_mario]="super_mario"
GAME_EPISODES[super_mario]="${EPISODES}"
GAME_MAX_STEPS[super_mario]="500"
GAME_TEMPERATURE[super_mario]="${TEMPERATURE}"
GAME_EXTRA_ARGS[super_mario]=""

# --- Diplomacy ---
GAME_ADAPTER_PATH[diplomacy]="${RUNS_DIR}/Qwen3-8B_diplomacy_20260322_234548/best/adapters/decision/action_taking"
GAME_BANK_DIR[diplomacy]="${RUNS_DIR}/Qwen3-8B_diplomacy_20260322_234548/best/banks/diplomacy"
GAME_EVAL_NAME[diplomacy]="diplomacy"
GAME_EPISODES[diplomacy]="56"
GAME_MAX_STEPS[diplomacy]=""
GAME_TEMPERATURE[diplomacy]="0.4"
GAME_EXTRA_ARGS[diplomacy]=""

# --- 2048 ---
GAME_ADAPTER_PATH[twenty_forty_eight]="${RUNS_DIR}/Qwen3-8B_2048_20260322_071227/best/adapters/decision/action_taking"
GAME_BANK_DIR[twenty_forty_eight]="${RUNS_DIR}/Qwen3-8B_2048_20260322_071227/best/banks/twenty_forty_eight"
GAME_EVAL_NAME[twenty_forty_eight]="twenty_forty_eight"
GAME_EPISODES[twenty_forty_eight]="${EPISODES}"
GAME_MAX_STEPS[twenty_forty_eight]="200"
GAME_TEMPERATURE[twenty_forty_eight]="${TEMPERATURE}"
GAME_EXTRA_ARGS[twenty_forty_eight]=""

# ── Which games to run ───────────────────────────────────────────────
if [ -n "${GAMES:-}" ]; then
    ALL_GAMES=(${GAMES})
else
    ALL_GAMES=(twenty_forty_eight candy_crush tetris super_mario avalon diplomacy)
fi

# ── Output directory ─────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/with_skillbank_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[full-system] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[full-system] Done."
}
trap cleanup EXIT INT TERM

# ══════════════════════════════════════════════════════════════════════
#  Prepare skill banks (merge sub-banks for avalon & diplomacy)
# ══════════════════════════════════════════════════════════════════════

MERGED_BANK_DIR="${OUTPUT_BASE}/_merged_banks"
mkdir -p "${MERGED_BANK_DIR}"

declare -A GAME_BANK_PATH

prepare_bank() {
    local game="$1"
    local bank_dir="${GAME_BANK_DIR[$game]}"

    if [ ! -d "${bank_dir}" ]; then
        echo "    ✗ ${game}: bank directory NOT FOUND at ${bank_dir}"
        return 1
    fi

    # Check if skill_bank.jsonl exists directly in the bank dir
    if [ -f "${bank_dir}/skill_bank.jsonl" ]; then
        GAME_BANK_PATH[$game]="${bank_dir}/skill_bank.jsonl"
        local n
        n=$(wc -l < "${bank_dir}/skill_bank.jsonl")
        echo "    ✓ ${game}: ${bank_dir}/skill_bank.jsonl (${n} skills)"
        return 0
    fi

    # Multiple sub-banks (avalon: good/evil, diplomacy: 7 powers)
    # Merge all skill_bank.jsonl files into one
    local merged="${MERGED_BANK_DIR}/${game}_skill_bank.jsonl"
    local count=0
    local total_skills=0

    find "${bank_dir}" -name "skill_bank.jsonl" -type f | sort | while IFS= read -r f; do
        cat "$f"
    done > "${merged}"

    if [ -s "${merged}" ]; then
        total_skills=$(wc -l < "${merged}")
        local sub_count
        sub_count=$(find "${bank_dir}" -name "skill_bank.jsonl" -type f | wc -l)
        GAME_BANK_PATH[$game]="${merged}"
        echo "    ✓ ${game}: merged ${sub_count} sub-banks → ${total_skills} skills"
        return 0
    else
        echo "    ✗ ${game}: no skill_bank.jsonl found under ${bank_dir}"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  Validate adapters & prepare banks
# ══════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════════"
echo "  Full System: Decision Agent WITH Skill Bank"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Validating adapters and skill banks..."

LORA_MODULES=()
VALID_GAMES=()

for game in "${ALL_GAMES[@]}"; do
    adapter_path="${GAME_ADAPTER_PATH[$game]}"
    adapter_name="qwen_${game}"

    if [ ! -f "${adapter_path}/adapter_config.json" ]; then
        echo "    ✗ ${game}: adapter NOT FOUND at ${adapter_path} — skipping"
        continue
    fi

    if ! prepare_bank "${game}"; then
        echo "      Skipping ${game} (no skill bank)."
        continue
    fi

    LORA_MODULES+=("${adapter_name}=${adapter_path}")
    VALID_GAMES+=("${game}")
done

if [ ${#VALID_GAMES[@]} -eq 0 ]; then
    echo ""
    echo "[ERROR] No valid adapter+bank pairs found."
    exit 1
fi

echo ""
echo "  Base model:     ${BASE_MODEL}"
echo "  GPU(s):         ${EVAL_GPUS}"
echo "  Games:          ${VALID_GAMES[*]}"
echo "  Output:         ${OUTPUT_BASE}"
echo "  Adapters:       ${#LORA_MODULES[@]}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Launch vLLM server with LoRA adapters
# ══════════════════════════════════════════════════════════════════════
if [ "${NO_SERVER}" = "0" ]; then
    echo "[full-system] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
    echo "  Model:    ${BASE_MODEL}"
    echo "  LoRA:     ${#LORA_MODULES[@]} adapters"
    echo ""

    VLLM_LOG="${OUTPUT_BASE}/vllm_server.log"

    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" \
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TENSOR_PARALLEL}" \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90 \
            --dtype auto \
            --trust-remote-code \
            --enable-lora \
            --max-loras 6 \
            --max-lora-rank 64 \
            --lora-modules "${LORA_MODULES[@]}" \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    echo "[full-system] vLLM PID=${VLLM_PID}, log at ${VLLM_LOG}"

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[full-system] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[full-system] ERROR: vLLM server exited unexpectedly. Check ${VLLM_LOG}"
            tail -30 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[full-system] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[full-system] Using existing vLLM server at ${VLLM_BASE_URL}"
    echo "  Make sure it was started with --enable-lora and the correct adapters!"
fi

# ══════════════════════════════════════════════════════════════════════
#  Run inference per game (with skill bank)
# ══════════════════════════════════════════════════════════════════════
FAILED_GAMES=()

for game in "${VALID_GAMES[@]}"; do
    adapter_name="qwen_${game}"
    eval_name="${GAME_EVAL_NAME[$game]}"
    episodes="${GAME_EPISODES[$game]}"
    max_steps="${GAME_MAX_STEPS[$game]}"
    temperature="${GAME_TEMPERATURE[$game]}"
    extra_args="${GAME_EXTRA_ARGS[$game]}"
    bank_path="${GAME_BANK_PATH[$game]}"

    game_output="${OUTPUT_BASE}/${game}"
    mkdir -p "${game_output}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  GAME: ${game} (${episodes} episodes, adapter=${adapter_name})"
    echo "  Condition: decision agent + skill bank"
    echo "  Bank: ${bank_path}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EVAL_ARGS=(
        --games "${eval_name}"
        --episodes "${episodes}"
        --temperature "${temperature}"
        --model "${adapter_name}"
        --seed "${SEED}"
        --output_dir "${game_output}"
        --bank "${bank_path}"
    )

    if [ -n "${max_steps}" ]; then
        EVAL_ARGS+=(--max_steps "${max_steps}")
    fi

    # shellcheck disable=SC2086
    if [ -n "${extra_args}" ]; then
        EVAL_ARGS+=(${extra_args})
    fi

    # Use orak-mario python for super_mario (needs gym-super-mario-bros / numpy 1.x)
    EVAL_PYTHON="python"
    if [ "${game}" = "super_mario" ] && [ -n "${ORAK_PYTHON:-}" ] && [ -x "${ORAK_PYTHON:-}" ]; then
        EVAL_PYTHON="${ORAK_PYTHON}"
    fi

    echo "[full-system] Command:"
    echo "  ${EVAL_PYTHON} -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
    echo ""

    if ${EVAL_PYTHON} -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}"; then
        echo "[full-system] ${game}: COMPLETE"
    else
        echo "[full-system] ${game}: FAILED (exit code $?)"
        FAILED_GAMES+=("${game}")
    fi
done

# ══════════════════════════════════════════════════════════════════════
#  Generate summary
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Full System: Decision Agent WITH Skill Bank — Results"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 - "${OUTPUT_BASE}" "${RUNS_DIR}" <<'PYEOF'
import json, sys
from pathlib import Path

output_base = Path(sys.argv[1])
runs_dir = Path(sys.argv[2])

coevo_best = {
    "avalon":             {"step": 5,  "mean_reward": 0.9950},
    "candy_crush":        {"step": 9,  "mean_reward": 657.75},
    "tetris":             {"step": 12, "mean_reward": 331.50},
    "super_mario":        {"step": 11, "mean_reward": 930.00},
    "diplomacy":          {"step": 22, "mean_reward": 5.0357},
    "twenty_forty_eight": {"step": 5,  "mean_reward": 1407.50},
}

results = []
for game_dir in sorted(output_base.iterdir()):
    if not game_dir.is_dir() or game_dir.name.startswith(("_", "vllm")):
        continue
    game = game_dir.name
    summaries = list(game_dir.rglob("rollout_summary.json"))
    if not summaries:
        continue
    with open(summaries[0]) as f:
        summary = json.load(f)

    full_reward = summary.get("mean_reward", 0)
    full_episodes = summary.get("total_episodes", 0)
    full_steps = summary.get("mean_steps", 0)

    coevo = coevo_best.get(game, {})
    coevo_reward = coevo.get("mean_reward", 0)

    delta = full_reward - coevo_reward
    pct = (delta / coevo_reward * 100) if coevo_reward != 0 else 0

    results.append({
        "game": game,
        "full_system_reward": full_reward,
        "full_system_mean_steps": full_steps,
        "full_system_episodes": full_episodes,
        "coevo_train_reward": coevo_reward,
        "delta_vs_training": delta,
        "delta_pct": pct,
    })

    print(f"  {game:22s}  "
          f"reward={full_reward:10.2f}  "
          f"steps={full_steps:6.1f}  "
          f"(train={coevo_reward:10.2f}, Δ={delta:+.2f})")

if results:
    summary_path = output_base / "full_system_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "decision_agent_with_skillbank",
            "description": "Decision agent (action_taking adapter) WITH trained skill bank from best checkpoint",
            "baseline": "Co-evolution best checkpoint training reward",
            "results": results,
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
else:
    print("  No results found.")

print()
PYEOF

if [ ${#FAILED_GAMES[@]} -gt 0 ]; then
    echo "  Failed games: ${FAILED_GAMES[*]}"
fi
echo "  Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
