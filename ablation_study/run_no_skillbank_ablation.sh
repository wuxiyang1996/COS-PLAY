nnnnn#!/usr/bin/env bash
# ======================================================================
#  Ablation Study: Decision Agent WITHOUT Skill Bank
#
#  Tests the trained decision agent (action_taking LoRA adapter from
#  the best co-evolution checkpoint) with NO skill bank guidance.
#
#  This isolates the contribution of the skill bank: comparing these
#  results against the full system (decision + bank) quantifies how
#  much the skill bank improves performance.
#
#  Architecture:
#    1. Launch ONE vLLM server with Qwen3-8B + all 6 game adapters
#    2. For each game, run inference with --no-bank using the trained
#       action_taking adapter from that game's best checkpoint
#    3. Collect results and produce a comparison summary
#
#  Usage:
#    conda activate game-ai-agent
#    bash ablation_study/run_no_skillbank_ablation.sh
#
#    # With overrides:
#    EVAL_GPUS=0 bash ablation_study/run_no_skillbank_ablation.sh
#    EPISODES=4  bash ablation_study/run_no_skillbank_ablation.sh
#    GAMES="twenty_forty_eight tetris" bash ablation_study/run_no_skillbank_ablation.sh
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
            echo "[ablation] Starting Xvfb on ${XVFB_DISPLAY}..."
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
#  Best checkpoint map (game → run_dir, best_step, game_id, adapter)
#
#  Selected by highest mean_reward among available checkpoints:
#    avalon:      step 5  (mean_reward=0.9950) — final-state adapters
#    candy_crush: step 9  (mean_reward=657.75)
#    tetris:      step 12 (mean_reward=331.50)
#    super_mario: step 11 (mean_reward=930.00)
#    diplomacy:   step 22 (mean_reward=5.0357)
#    2048:        step 5  (mean_reward=1407.50)
# ══════════════════════════════════════════════════════════════════════

declare -A GAME_RUN_DIR
declare -A GAME_ADAPTER_PATH
declare -A GAME_EVAL_NAME
declare -A GAME_EPISODES
declare -A GAME_MAX_STEPS
declare -A GAME_TEMPERATURE
declare -A GAME_EXTRA_ARGS

# --- Avalon ---
# Per-step checkpoints were empty; using final-state adapters from best/
GAME_RUN_DIR[avalon]="Qwen3-8B_avalon_20260322_200424"
GAME_ADAPTER_PATH[avalon]="${RUNS_DIR}/Qwen3-8B_avalon_20260322_200424/best/adapters/decision/action_taking"
GAME_EVAL_NAME[avalon]="avalon"
GAME_EPISODES[avalon]="40"
GAME_MAX_STEPS[avalon]=""
GAME_TEMPERATURE[avalon]="0.4"
GAME_EXTRA_ARGS[avalon]="--num_players 5"

# --- Candy Crush ---
GAME_RUN_DIR[candy_crush]="Qwen3-8B_20260321_213813_(Candy_crush)"
GAME_ADAPTER_PATH[candy_crush]="${RUNS_DIR}/Qwen3-8B_20260321_213813_(Candy_crush)/best/adapters/decision/action_taking"
GAME_EVAL_NAME[candy_crush]="candy_crush"
GAME_EPISODES[candy_crush]="${EPISODES}"
GAME_MAX_STEPS[candy_crush]="200"
GAME_TEMPERATURE[candy_crush]="${TEMPERATURE}"
GAME_EXTRA_ARGS[candy_crush]=""

# --- Tetris ---
GAME_RUN_DIR[tetris]="Qwen3-8B_tetris_20260322_170438"
GAME_ADAPTER_PATH[tetris]="${RUNS_DIR}/Qwen3-8B_tetris_20260322_170438/best/adapters/decision/action_taking"
GAME_EVAL_NAME[tetris]="tetris"
GAME_EPISODES[tetris]="${EPISODES}"
GAME_MAX_STEPS[tetris]="200"
GAME_TEMPERATURE[tetris]="${TEMPERATURE}"
GAME_EXTRA_ARGS[tetris]="--macro-actions"

# --- Super Mario ---
GAME_RUN_DIR[super_mario]="Qwen3-8B_super_mario_20260323_030839"
GAME_ADAPTER_PATH[super_mario]="${RUNS_DIR}/Qwen3-8B_super_mario_20260323_030839/best/adapters/decision/action_taking"
GAME_EVAL_NAME[super_mario]="super_mario"
GAME_EPISODES[super_mario]="${EPISODES}"
GAME_MAX_STEPS[super_mario]="500"
GAME_TEMPERATURE[super_mario]="${TEMPERATURE}"
GAME_EXTRA_ARGS[super_mario]=""

# --- Diplomacy ---
GAME_RUN_DIR[diplomacy]="Qwen3-8B_diplomacy_20260322_234548"
GAME_ADAPTER_PATH[diplomacy]="${RUNS_DIR}/Qwen3-8B_diplomacy_20260322_234548/best/adapters/decision/action_taking"
GAME_EVAL_NAME[diplomacy]="diplomacy"
GAME_EPISODES[diplomacy]="56"
GAME_MAX_STEPS[diplomacy]=""
GAME_TEMPERATURE[diplomacy]="0.4"
GAME_EXTRA_ARGS[diplomacy]=""

# --- 2048 ---
GAME_RUN_DIR[twenty_forty_eight]="Qwen3-8B_2048_20260322_071227"
GAME_ADAPTER_PATH[twenty_forty_eight]="${RUNS_DIR}/Qwen3-8B_2048_20260322_071227/best/adapters/decision/action_taking"
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
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/no_skillbank_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[ablation] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[ablation] Done."
}
trap cleanup EXIT INT TERM

# ══════════════════════════════════════════════════════════════════════
#  Validate adapter paths
# ══════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════════"
echo "  Ablation Study: Decision Agent WITHOUT Skill Bank"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Validating adapter paths..."

LORA_MODULES=()
VALID_GAMES=()

for game in "${ALL_GAMES[@]}"; do
    adapter_path="${GAME_ADAPTER_PATH[$game]}"
    # Adapter name must contain "qwen" so ask_model() routes to ask_vllm()
    adapter_name="qwen_${game}"

    if [ -f "${adapter_path}/adapter_config.json" ]; then
        echo "    ✓ ${game}: ${adapter_path}"
        LORA_MODULES+=("${adapter_name}=${adapter_path}")
        VALID_GAMES+=("${game}")
    else
        echo "    ✗ ${game}: adapter NOT FOUND at ${adapter_path}"
        echo "      Skipping this game."
    fi
done

if [ ${#VALID_GAMES[@]} -eq 0 ]; then
    echo ""
    echo "[ERROR] No valid adapters found. Check that 'best/' folders exist."
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
    echo "[ablation] Launching vLLM server on ${VLLM_HOST}:${VLLM_PORT}..."
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

    echo "[ablation] vLLM PID=${VLLM_PID}, log at ${VLLM_LOG}"

    MAX_WAIT=600
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[ablation] vLLM server ready (waited ${WAITED}s)."
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[ablation] ERROR: vLLM server exited unexpectedly. Check ${VLLM_LOG}"
            tail -30 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[ablation] ERROR: vLLM server did not start within ${MAX_WAIT}s."
        exit 1
    fi
else
    echo "[ablation] Using existing vLLM server at ${VLLM_BASE_URL}"
    echo "  Make sure it was started with --enable-lora and the correct adapters!"
fi

# ══════════════════════════════════════════════════════════════════════
#  Run inference per game
# ══════════════════════════════════════════════════════════════════════
RESULTS_FILE="${OUTPUT_BASE}/ablation_results.jsonl"
FAILED_GAMES=()

for game in "${VALID_GAMES[@]}"; do
    adapter_name="qwen_${game}"
    eval_name="${GAME_EVAL_NAME[$game]}"
    episodes="${GAME_EPISODES[$game]}"
    max_steps="${GAME_MAX_STEPS[$game]}"
    temperature="${GAME_TEMPERATURE[$game]}"
    extra_args="${GAME_EXTRA_ARGS[$game]}"

    game_output="${OUTPUT_BASE}/${game}"
    mkdir -p "${game_output}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  GAME: ${game} (${episodes} episodes, adapter=${adapter_name})"
    echo "  Condition: decision agent ONLY (no skill bank)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EVAL_ARGS=(
        --games "${eval_name}"
        --episodes "${episodes}"
        --temperature "${temperature}"
        --model "${adapter_name}"
        --seed "${SEED}"
        --output_dir "${game_output}"
        --no-bank
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

    echo "[ablation] Command:"
    echo "  ${EVAL_PYTHON} -m scripts.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
    echo ""

    if ${EVAL_PYTHON} -m scripts.run_qwen3_8b_eval "${EVAL_ARGS[@]}"; then
        echo "[ablation] ${game}: COMPLETE"
    else
        echo "[ablation] ${game}: FAILED (exit code $?)"
        FAILED_GAMES+=("${game}")
    fi
done

# ══════════════════════════════════════════════════════════════════════
#  Generate summary
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Ablation Study: Decision Agent WITHOUT Skill Bank"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 - "${OUTPUT_BASE}" "${RUNS_DIR}" <<'PYEOF'
import json, sys, os
from pathlib import Path

output_base = Path(sys.argv[1])
runs_dir = Path(sys.argv[2])

# Best checkpoint rewards from co-evolution (full system: decision + bank)
coevo_best = {
    "avalon":           {"step": 5,  "mean_reward": 0.9950},
    "candy_crush":      {"step": 9,  "mean_reward": 657.75},
    "tetris":           {"step": 12, "mean_reward": 331.50},
    "super_mario":      {"step": 11, "mean_reward": 930.00},
    "diplomacy":        {"step": 22, "mean_reward": 5.0357},
    "twenty_forty_eight": {"step": 5,  "mean_reward": 1407.50},
}

results = []
for game_dir in sorted(output_base.iterdir()):
    if not game_dir.is_dir() or game_dir.name in ("vllm_server.log",):
        continue
    game = game_dir.name
    summaries = list(game_dir.rglob("rollout_summary.json"))
    if not summaries:
        continue
    with open(summaries[0]) as f:
        summary = json.load(f)

    ablation_reward = summary.get("mean_reward", 0)
    ablation_episodes = summary.get("total_episodes", 0)

    coevo = coevo_best.get(game, {})
    coevo_reward = coevo.get("mean_reward", 0)

    delta = ablation_reward - coevo_reward
    pct = (delta / coevo_reward * 100) if coevo_reward != 0 else 0

    results.append({
        "game": game,
        "ablation_no_bank_reward": ablation_reward,
        "coevo_full_reward": coevo_reward,
        "delta": delta,
        "delta_pct": pct,
        "ablation_episodes": ablation_episodes,
    })

    print(f"  {game:22s}  "
          f"no-bank={ablation_reward:10.2f}  "
          f"full={coevo_reward:10.2f}  "
          f"Δ={delta:+10.2f} ({pct:+.1f}%)")

if results:
    summary_path = output_base / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "no_skillbank_ablation",
            "description": "Decision agent with trained action_taking adapter, NO skill bank",
            "baseline": "Co-evolution best checkpoint (decision + skill bank)",
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
