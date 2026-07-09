#!/usr/bin/env bash
# ======================================================================
#  Phase-2 hold-out adaptation curriculum — 6 games × (5 + 15) steps,
#  sequential, seeded from the post-Phase-1 bank + LoRA snapshot.
#
#  Implements the plan locked in
#    training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §7
#  (held-out roster + two-budget reporting, refreshed 2026-05-03 PM).
#
#  Each Phase-2 gymv game is paired in-genre with a Phase-1 source so
#  the cross-game skill translator
#  (skill_agents/skill_bank/translate_for_target.py) has the closest
#  possible source vocabulary to re-ground onto:
#
#    Phase 1: gymv_streets_of_rage_2  ← AlteredBeast    (in-genre lift)
#    Phase 2: gymv_space_harrier_ii   ← ThunderForceIII (scale-jump test)
#    Phase 3: gymv_airstriker         ← ThunderForceIII (easier in-genre)
#    Phase 4: gymv_strider            ← DynamiteHeaddy  (partial-signal rescue)
#    Phase 5: twenty_forty_eight      ← tetris+Columns  (grid-puzzle composition)
#    Phase 6: super_mario             ← (no in-genre)   (transfer-distance bound)
#
#  Two-budget protocol (§7.2):
#    Budget A — 5 steps · INFER_ONLY=1 · frozen LoRA
#               "does the merged Phase-1 bank fire on the new game *at all*?"
#    Budget B — 15 steps · INFER_ONLY=0 · GRPO unfrozen
#               "does adding GRPO compound the bank advantage?"
#  This script defaults to Budget B (the headline result). Run with
#  ``BUDGET=A`` for the inference-only baseline.
#
#  Default carrier: shared bank + per-boundary LLM translation
#    BANK_MODE=shared
#    TRANSLATE_ON_BOUNDARY=1
#  This is the cross-game lifelong-learning pipeline landed alongside
#  the §4.1 roster refresh. To run without translation (control), set
#  TRANSLATE_ON_BOUNDARY=0; to fall back to per-game banks (pre-refresh
#  legacy), set BANK_MODE=per_game.
#
#  Prerequisites:
#    1. Phase-1 curriculum complete:
#         scripts/run_phase1_curriculum.sh
#    2. PHASE1_SNAPSHOT pointing at a Phase-1 snapshot directory
#       containing ``lora_adapters/`` and ``skillbank/skill_bank.jsonl``
#       (or ``skillbank/<game>/skill_bank.jsonl`` for per_game-mode
#       Phase-1 runs — the seed loader handles both layouts).
#    3. 35B-A3B judge endpoint reachable (auto-launched in dual_stack
#       mode by this script).
#
#  Usage:
#    PHASE1_SNAPSHOT=runs/Qwen3.5-9B_<ts>_phase1/phase_snapshots/phase_06_tetris \
#      bash scripts/run_phase2_holdout.sh
#
#    # Override per-phase iteration count:
#    ITERS_PER_PHASE=10 PHASE1_SNAPSHOT=... bash scripts/run_phase2_holdout.sh
#
#    # 5-step inference-only baseline (Budget A):
#    BUDGET=A PHASE1_SNAPSHOT=... bash scripts/run_phase2_holdout.sh
#
#    # Resume from phase 3 (e.g. after a mid-phase failure):
#    RESUME_PHASE=3 RUN_DIR=runs/<existing>_phase2 \
#      PHASE1_SNAPSHOT=... bash scripts/run_phase2_holdout.sh
#
#    # Disable per-boundary translation (test shared bank without LLM rewrite):
#    TRANSLATE_ON_BOUNDARY=0 PHASE1_SNAPSHOT=... \
#      bash scripts/run_phase2_holdout.sh
#
#    # Fall back to per-game banks (pre-refresh legacy):
#    BANK_MODE=per_game PHASE1_SNAPSHOT=... \
#      bash scripts/run_phase2_holdout.sh
#
#  Cross-refs:
#    - scripts/run_phase1_curriculum.sh (the source-GRPO curriculum)
#    - skill_agents/skill_bank/translate_for_target.py (LLM translator)
#    - trainer/coevolution/skillbank_pipeline.py:SharedSkillBankManager
#    - tests/test_shared_skill_bank.py (invariant + translator tests)
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Headless rendering ────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

# ── HuggingFace cache ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# ── PYTHONPATH (mirrors run_all.sh) ──────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
MANAGE_VLLM="${MANAGE_VLLM:-1}"

BUDGET="${BUDGET:-B}"            # A=5-step infer-only, B=15-step GRPO (default)
case "${BUDGET}" in
    A)
        ITERS_PER_PHASE="${ITERS_PER_PHASE:-5}"
        INFER_ONLY=1
        ;;
    B)
        ITERS_PER_PHASE="${ITERS_PER_PHASE:-15}"
        INFER_ONLY=0
        ;;
    *)
        echo "ERROR: unknown BUDGET=${BUDGET} (expected A or B)"
        exit 1
        ;;
esac

CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution-phase2}"
RUN_DIR="${RUN_DIR:-}"
DEBUG="${DEBUG:-}"
RESUME_PHASE="${RESUME_PHASE:-1}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# ── Shared-bank lifelong-learning mode (default ON for Phase 2) ──────
#   Phase 2 is the carrier for the cross-game transfer hypothesis we
#   built the shared-bank + translator pipeline to test, so it defaults
#   ON here. Override with BANK_MODE=per_game for the legacy ablation.
BANK_MODE="${BANK_MODE:-shared}"
TRANSLATE_ON_BOUNDARY="${TRANSLATE_ON_BOUNDARY:-1}"
case "${BANK_MODE}" in
    per_game|shared) ;;
    *)
        echo "ERROR: unknown BANK_MODE=${BANK_MODE} (expected per_game|shared)"
        exit 1
        ;;
esac
# Translation only makes sense in shared mode.
if [ "${BANK_MODE}" = "per_game" ] && [ "${TRANSLATE_ON_BOUNDARY}" = "1" ]; then
    echo "[run_phase2] BANK_MODE=per_game forces TRANSLATE_ON_BOUNDARY=0 (per-game banks aren't shared, nothing to translate)"
    TRANSLATE_ON_BOUNDARY=0
fi

# 8×H200 layout selector (mirrors run_phase1_curriculum.sh; see banner there).
LAYOUT="${LAYOUT:-dual_stack}"
JUDGE_MODE="${JUDGE_MODE:-auto}"

case "${LAYOUT}" in
    dual_stack)
        VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
        JUDGE_GPUS="${JUDGE_GPUS:-4,5}"
        JUDGE_TP="${JUDGE_TP:-2}"
        GRPO_GPUS="${GRPO_GPUS:-6 7}"
        EPISODES="${EPISODES:-8}"
        ;;
    dual_stack_fsdp4)
        VLLM_GPUS="${VLLM_GPUS:-0 1}"
        JUDGE_GPUS="${JUDGE_GPUS:-2,3}"
        JUDGE_TP="${JUDGE_TP:-2}"
        GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
        EPISODES="${EPISODES:-16}"
        ;;
    actor_only)
        VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
        JUDGE_GPUS=""
        JUDGE_TP="0"
        GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
        EPISODES="${EPISODES:-8}"
        if [ "${JUDGE_MODE}" = "auto" ]; then
            JUDGE_MODE="off"
        fi
        ;;
    *)
        echo "ERROR: unknown LAYOUT=${LAYOUT}" \
             "(expected dual_stack|dual_stack_fsdp4|actor_only)"
        exit 1
        ;;
esac

JUDGE_PORT="${JUDGE_PORT:-8004}"
JUDGE_URL="${JUDGE_URL:-http://localhost:${JUDGE_PORT}/v1}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.92}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"

# ── Phase-1 snapshot — mandatory ─────────────────────────────────────
if [ -z "${PHASE1_SNAPSHOT:-}" ]; then
    echo "ERROR: PHASE1_SNAPSHOT is required."
    echo ""
    echo "Point it at the post-Phase-1 snapshot directory, e.g.:"
    echo "  PHASE1_SNAPSHOT=runs/Qwen3.5-9B_<ts>_phase1/phase_snapshots/phase_06_tetris \\"
    echo "    bash scripts/run_phase2_holdout.sh"
    echo ""
    echo "The directory must contain lora_adapters/ and skillbank/."
    exit 1
fi
if [ ! -d "${PHASE1_SNAPSHOT}" ]; then
    echo "ERROR: PHASE1_SNAPSHOT=${PHASE1_SNAPSHOT} not found"
    exit 1
fi
PHASE1_LORA="${PHASE1_SNAPSHOT}/lora_adapters"
PHASE1_BANK_DIR="${PHASE1_SNAPSHOT}/skillbank"
if [ ! -d "${PHASE1_LORA}" ]; then
    echo "ERROR: ${PHASE1_LORA}/ missing — does this snapshot have LoRA adapters?"
    exit 1
fi
if [ ! -d "${PHASE1_BANK_DIR}" ]; then
    echo "ERROR: ${PHASE1_BANK_DIR}/ missing — does this snapshot have a skill bank?"
    exit 1
fi

# Decision adapters live under "decision/" (skill_selection + action_taking)
# and skillbank adapters under "skillbank/" (segment + contract + curator)
# in the run_phase1 snapshot layout.
PHASE1_DECISION="${PHASE1_LORA}/decision"
PHASE1_SKILLBANK="${PHASE1_LORA}/skillbank"
# Older snapshots may flat-pack all 5 LoRA dirs under lora_adapters/ —
# the trainer's prepare_adapters() handles both layouts.
if [ ! -d "${PHASE1_DECISION}" ]; then
    PHASE1_DECISION="${PHASE1_LORA}"
fi
if [ ! -d "${PHASE1_SKILLBANK}" ]; then
    PHASE1_SKILLBANK="${PHASE1_LORA}"
fi

# ── Locked Phase-2 hold-out roster (training_notes §7.1, refreshed 2026-05-03 PM)
PHASES=(
    "1:gymv_streets_of_rage_2:Streets of Rage 2"
    "2:gymv_space_harrier_ii:Space Harrier II"
    "3:gymv_airstriker:Airstriker"
    "4:gymv_strider:Strider"
    "5:twenty_forty_eight:2048"
    "6:super_mario:Super Mario Bros"
)
NUM_PHASES=${#PHASES[@]}

# In-genre Phase-1 source for each Phase-2 game (used by the per-boundary
# translator to pick the right source-game vocabulary).
declare -A IN_GENRE_SOURCE=(
    ["gymv_streets_of_rage_2"]="gymv_altered_beast"
    ["gymv_space_harrier_ii"]="gymv_thunder_force_iii"
    ["gymv_airstriker"]="gymv_thunder_force_iii"
    ["gymv_strider"]="gymv_dynamite_headdy"
    ["twenty_forty_eight"]="tetris"
    ["super_mario"]="gymv_dynamite_headdy"
)

# Per-game baseline anchor (min teacher reward across 4 frontier rows
# from new Cold-start-out-gymv/latest data — see training_notes §4.1).
declare -A BASELINE_ANCHOR=(
    ["gymv_streets_of_rage_2"]="min teacher 202 (SoR2)"
    ["gymv_space_harrier_ii"]="min teacher 14 469 (SH2 — scale outlier)"
    ["gymv_airstriker"]="min teacher 52 (Airstriker)"
    ["gymv_strider"]="min teacher 0 (partial-signal — rescue test)"
    ["twenty_forty_eight"]="paper Figure 4 (±30%)"
    ["super_mario"]="paper Figure 4 (±30%)"
)

# ── Cleanup on exit ───────────────────────────────────────────────────
VLLM_PID=""
JUDGE_PID=""
cleanup() {
    echo ""
    echo "[run_phase2] Shutting down..."
    if [ -n "${JUDGE_PID}" ] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "[run_phase2] Stopping 35B judge (PID ${JUDGE_PID})..."
        kill "${JUDGE_PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${JUDGE_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${JUDGE_PID}" 2>/dev/null || true
    fi
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_phase2] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_phase2] Done."
}
trap cleanup EXIT INT TERM

# ── 35B judge bring-up (mirrors run_phase1_curriculum.sh) ────────────
JUDGE_LOG=""
start_35b_judge() {
    if [ -z "${JUDGE_GPUS}" ] || [ "${JUDGE_TP}" = "0" ]; then
        echo "[run_phase2] LAYOUT=${LAYOUT}: 35B server not part of layout — skipping."
        return 0
    fi
    JUDGE_LOG="${RUN_DIR:-runs}/judge_35b.log"
    mkdir -p "$(dirname "${JUDGE_LOG}")"
    echo "[run_phase2] Auto-launching 35B-A3B judge:"
    echo "  GPUs:   ${JUDGE_GPUS}  (TP=${JUDGE_TP}, expert-parallel ON)"
    echo "  Port:   ${JUDGE_PORT}"
    echo "  Model:  ${JUDGE_MODEL}"
    echo "  Log:    ${JUDGE_LOG}"
    GPUS="${JUDGE_GPUS}" \
    TENSOR_PARALLEL="${JUDGE_TP}" \
    EXPERT_PARALLEL=1 \
    GPU_UTIL="${JUDGE_GPU_UTIL}" \
    PORT="${JUDGE_PORT}" \
    HOST="127.0.0.1" \
    MODEL="${JUDGE_MODEL}" \
        bash "${PROJECT_ROOT}/inference/serve_qwen35_35b_a3b.sh" \
            > "${JUDGE_LOG}" 2>&1 &
    JUDGE_PID=$!
    echo "[run_phase2] 35B server PID: ${JUDGE_PID}"

    echo "[run_phase2] Waiting for 35B server health on ${JUDGE_URL} ..."
    for _ in $(seq 1 120); do
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[run_phase2] 35B server is healthy."
            return 0
        fi
        if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
            echo "[run_phase2] 35B server died during startup — see ${JUDGE_LOG}"
            return 1
        fi
        sleep 5
    done
    echo "[run_phase2] 35B server did not become healthy within 600s — see ${JUDGE_LOG}"
    return 1
}

case "${JUDGE_MODE}" in
    auto|external)
        export VLLM_BASE_URL_MAP="${MODEL}=http://localhost:${PORT}/v1,${JUDGE_MODEL}=${JUDGE_URL}"
        export VLM_AGENT_BACKBONE_JUDGE_MODEL="${JUDGE_MODEL}"
        ;;
    off)
        echo "[run_phase2] JUDGE_MODE=off → 35B routing NOT wired" \
             "(judge will fall back to 9B; only safe for ablations)."
        ;;
    *)
        echo "ERROR: unknown JUDGE_MODE=${JUDGE_MODE}" \
             "(expected auto|external|off)"
        exit 1
        ;;
esac

# ======================================================================
# Phase 0: Resolve run directory + verify Phase-1 snapshot is consumable
# ======================================================================
echo "══════════════════════════════════════════════════════════════"
echo "  Phase-2 Hold-out Adaptation Curriculum"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:            ${MODEL}"
echo "  Budget:           ${BUDGET} (${ITERS_PER_PHASE} steps/phase, INFER_ONLY=${INFER_ONLY})"
echo "  Phases:           ${NUM_PHASES} sequential games × ${ITERS_PER_PHASE} steps"
echo "  Total iterations: $((NUM_PHASES * ITERS_PER_PHASE))"
echo "  Episodes/step:    ${EPISODES}"
echo "  Phase-1 snapshot: ${PHASE1_SNAPSHOT}"
echo "  Bank mode:        ${BANK_MODE}$([ "${BANK_MODE}" = "shared" ] && [ "${TRANSLATE_ON_BOUNDARY}" = "1" ] && echo " (translate at every phase boundary)" || true)"
echo "  Resume phase:     ${RESUME_PHASE}"
if [ "${MANAGE_VLLM}" = "1" ]; then
    echo "  GPU mode:         MANAGED (persistent vLLM + FSDP)"
    echo "  GPU layout:       LAYOUT=${LAYOUT}"
    echo "    9B vLLM GPUs:   ${VLLM_GPUS}        (TP=1 each, ports ${PORT}+)"
    if [ -n "${JUDGE_GPUS}" ]; then
        echo "    35B judge GPUs: ${JUDGE_GPUS}      (TP=${JUDGE_TP}+EP, port ${JUDGE_PORT})"
    else
        echo "    35B judge:      OFF (LAYOUT=actor_only)"
    fi
    echo "    GRPO trainer:   ${GRPO_GPUS}        (FSDP=$(echo "${GRPO_GPUS}" | wc -w))"
    echo "  Judge mode:       ${JUDGE_MODE}"
    if [ "${JUDGE_MODE}" != "off" ]; then
        echo "  Judge URL:        ${JUDGE_URL}"
        echo "  Judge model:      ${JUDGE_MODEL}"
    fi
else
    echo "  GPU mode:         LEGACY (external vLLM at port ${PORT})"
fi
echo ""
echo "  Hold-out schedule (each game paired in-genre with a Phase-1 source):"
for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r pnum game display <<< "${phase_def}"
    src="${IN_GENRE_SOURCE[${game}]:-(no in-genre source)}"
    anchor="${BASELINE_ANCHOR[${game}]:-no anchor}"
    echo "    Phase ${pnum}: ${display}  ← in-genre source: ${src}"
    echo "                    baseline: ${anchor}"
done
echo "══════════════════════════════════════════════════════════════"

# Pre-flight: confirm all Phase-2 slugs are wired in episode_runner +
# registered in trainer/coevolution/config.py:GAME_MAX_STEPS.
python -c "
from trainer.coevolution.episode_runner import _lazy_imports, GYMV_TEMPORAL_GAMES_SET
from trainer.coevolution.config import GAME_MAX_STEPS
_lazy_imports()
required_gymv = {'gymv_streets_of_rage_2', 'gymv_space_harrier_ii',
                 'gymv_airstriker', 'gymv_strider'}
required_all = required_gymv | {'twenty_forty_eight', 'super_mario'}
missing_gymv = required_gymv - GYMV_TEMPORAL_GAMES_SET
if missing_gymv:
    raise SystemExit(
        f'[run_phase2] FATAL: Phase-2 gymv slugs not wired in episode_runner: '
        f'{sorted(missing_gymv)}. Run install/install_gymv.sh + apply_patch.sh.'
    )
missing_registry = required_all - set(GAME_MAX_STEPS)
if missing_registry:
    raise SystemExit(
        f'[run_phase2] FATAL: Phase-2 slugs not in GAME_MAX_STEPS: '
        f'{sorted(missing_registry)}.'
    )
print(f'[run_phase2] Phase-2 slugs wired + registered: {sorted(required_all)}')
"

RESOLVED_RUN_DIR=$(python -c "
import os
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from pathlib import Path

from trainer.coevolution.config import CoEvolutionConfig

cfg = CoEvolutionConfig(model_name='${MODEL}')
run_dir_override = '${RUN_DIR}'
if run_dir_override:
    cfg.run_dir = run_dir_override
else:
    base = cfg.run_dir or ''
    cfg.run_dir = base.replace('runs/', 'runs/phase2_') if base else None
cfg.resolve_paths()
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
export ADAPTER_DIR="${RUN_DIR}/lora_adapters"
SNAPSHOT_DIR="${RUN_DIR}/phase_snapshots"
mkdir -p "${SNAPSHOT_DIR}" "${RUN_DIR}/skillbank"

echo "[run_phase2] Run dir:        ${RUN_DIR}"
echo "[run_phase2] Adapter dir:    ${ADAPTER_DIR}"
echo "[run_phase2] Snapshot dir:   ${SNAPSHOT_DIR}"
echo "[run_phase2] Seed bank dir:  ${PHASE1_BANK_DIR}"

# Seed the Phase-2 shared bank from the Phase-1 snapshot if shared mode.
# In shared mode the seed_bank_dir handler in SharedSkillBankManager
# concatenates every <PHASE1_BANK_DIR>/<game>/skill_bank.jsonl into one
# <RUN_DIR>/skillbank/skill_bank.jsonl with feasible_tasks=[<source_game>]
# stamped per skill (the §22 cross-contamination guard).
if [ "${BANK_MODE}" = "shared" ]; then
    # If Phase-1 was per_game, PHASE1_BANK_DIR has per-game subdirs.
    # If Phase-1 was shared, PHASE1_BANK_DIR has a single skill_bank.jsonl.
    # SharedSkillBankManager._seed_from_coldstart handles per-game subdirs;
    # for the shared-mode case we copy the file directly.
    if [ -f "${PHASE1_BANK_DIR}/skill_bank.jsonl" ]; then
        cp -f "${PHASE1_BANK_DIR}/skill_bank.jsonl" "${RUN_DIR}/skillbank/skill_bank.jsonl"
        echo "[run_phase2] Seeded shared bank from Phase-1 shared snapshot."
    fi
fi

# Start judge before curriculum loop (mirrors run_phase1_curriculum.sh).
if [ "${JUDGE_MODE}" = "auto" ]; then
    if ! start_35b_judge; then
        echo "[run_phase2] FATAL: cannot start 35B judge."
        exit 1
    fi
elif [ "${JUDGE_MODE}" = "external" ]; then
    echo "[run_phase2] JUDGE_MODE=external → assuming 35B server at ${JUDGE_URL}"
    if command -v curl >/dev/null 2>&1; then
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[run_phase2] 35B server is reachable."
        else
            echo "[run_phase2] WARNING: ${JUDGE_URL} not reachable."
        fi
    fi
fi

# ======================================================================
# Helper: phase-boundary translator (no-op outside shared mode)
#   Same shape as run_phase1_curriculum.sh::translate_bank_for_next_phase
#   but parameterised by the in-genre source from IN_GENRE_SOURCE.
# ======================================================================
translate_bank_for_target() {
    if [ "${BANK_MODE}" != "shared" ] || [ "${TRANSLATE_ON_BOUNDARY}" != "1" ]; then
        return 0
    fi

    local target_game="$1"
    local source_game="${IN_GENRE_SOURCE[${target_game}]:-}"
    if [ -z "${source_game}" ]; then
        echo "[run_phase2] translate_bank: no in-genre source for ${target_game} — skipping translation"
        return 0
    fi

    local shared_bank="${RUN_DIR}/skillbank/skill_bank.jsonl"
    if [ ! -f "${shared_bank}" ]; then
        echo "[run_phase2] translate_bank: no shared bank at ${shared_bank} yet — skipping"
        return 0
    fi

    echo ""
    echo "[run_phase2] ── Cross-game skill translation: ${source_game} → ${target_game} ──"

    local target_actions
    target_actions=$(python -c "
import os, json
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
target = '${target_game}'
acts = []
try:
    if target.startswith('gymv_'):
        from env_wrappers.gymv_temporal_nl_wrapper import make_gymv_temporal_env
        env = make_gymv_temporal_env(target)
        env.reset()
        acts = list(env.action_names)
        env.close()
    else:
        from env_wrappers.game_configs import get_game_config
        cfg = get_game_config(target)
        acts = list(getattr(cfg, 'available_actions', []) or [])
except Exception as exc:
    import sys
    print('TRANSLATOR_RESOLVE_ERROR:' + str(exc), file=sys.stderr)
    acts = []
print(json.dumps(acts))
" 2>/dev/null || echo '[]')

    if [ "${target_actions}" = "[]" ] || [ -z "${target_actions}" ]; then
        echo "[run_phase2] translate_bank: could not resolve actions for ${target_game} — skipping"
        return 0
    fi

    local translated_bank="${RUN_DIR}/skillbank/skill_bank.translated_to_${target_game}.jsonl"
    echo "[run_phase2] translate_bank: target_actions=${target_actions}"
    echo "[run_phase2] translate_bank: source_game=${source_game}"
    echo "[run_phase2] translate_bank: writing → ${translated_bank}"

    if python -m skill_agents.skill_bank.translate_for_target \
        --source-bank "${shared_bank}" \
        --target-game "${target_game}" \
        --target-actions "${target_actions}" \
        --source-game "${source_game}" \
        --output "${translated_bank}" \
        --judge-model "${JUDGE_MODEL}" \
        -v
    then
        mv -f "${translated_bank}" "${shared_bank}"
        echo "[run_phase2] translate_bank: shared bank updated with ${target_game} derivatives"
    else
        echo "[run_phase2] translate_bank: WARNING — translation failed; keeping prior shared bank"
        rm -f "${translated_bank}" 2>/dev/null || true
    fi
}

# ======================================================================
# Helper: save a Phase-2 snapshot
# ======================================================================
save_phase2_snapshot() {
    local phase_num="$1"
    local game="$2"
    local display="$3"
    local step_end="$4"

    local snap_name
    snap_name=$(printf "phase_%02d_%s" "${phase_num}" "${game}")
    local snap_path="${SNAPSHOT_DIR}/${snap_name}"

    echo ""
    echo "[run_phase2] ── Saving Phase-2 snapshot: ${snap_name} ──"
    mkdir -p "${snap_path}"

    if [ -d "${RUN_DIR}/lora_adapters" ]; then
        cp -r "${RUN_DIR}/lora_adapters" "${snap_path}/lora_adapters"
    fi
    if [ -d "${RUN_DIR}/skillbank" ]; then
        cp -r "${RUN_DIR}/skillbank" "${snap_path}/skillbank"
    fi
    local latest_ckpt=""
    if [ -d "${RUN_DIR}/checkpoints" ]; then
        latest_ckpt=$(ls -d "${RUN_DIR}/checkpoints"/step_* 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "${latest_ckpt}" ]; then
            cp -r "${latest_ckpt}" "${snap_path}/checkpoint"
        fi
    fi

    cat > "${snap_path}/phase_meta.json" <<METAEOF
{
    "phase": ${phase_num},
    "game": "${game}",
    "display_name": "${display}",
    "step_end": ${step_end},
    "iters_per_phase": ${ITERS_PER_PHASE},
    "budget": "${BUDGET}",
    "infer_only": ${INFER_ONLY},
    "bank_mode": "${BANK_MODE}",
    "translate_on_boundary": ${TRANSLATE_ON_BOUNDARY},
    "in_genre_source": "${IN_GENRE_SOURCE[${game}]:-}",
    "model": "${MODEL}",
    "phase1_snapshot": "${PHASE1_SNAPSHOT}",
    "timestamp": "$(date -Iseconds)"
}
METAEOF

    echo "[run_phase2]   Snapshot saved to: ${snap_path}"
}

# ======================================================================
# Build common training args
# ======================================================================
build_train_args() {
    local game="$1"
    local total_steps="$2"
    local is_first_phase="$3"

    local args=(
        --games "${game}"
        --total-steps "${total_steps}"
        --curriculum "none"
        --episodes-per-game "${EPISODES}"
        --checkpoint-interval "${CKPT_INTERVAL}"
        --model "${MODEL}"
        --wandb-project "${WANDB_PROJECT}"
        --run-dir "${RUN_DIR}"
        --bank-mode "${BANK_MODE}"
    )

    if [ "${is_first_phase}" = "true" ]; then
        # First phase only: load the Phase-1 snapshot LoRAs and seed
        # the bank from the Phase-1 snapshot bank dir.
        if [ -d "${PHASE1_DECISION}" ]; then
            args+=(--load-decision-adapters "${PHASE1_DECISION}")
        fi
        if [ -d "${PHASE1_SKILLBANK}" ]; then
            args+=(--load-skillbank-adapters "${PHASE1_SKILLBANK}")
        fi
        if [ -d "${PHASE1_BANK_DIR}" ]; then
            args+=(--seed-bank-dir "${PHASE1_BANK_DIR}")
        fi
    else
        args+=(--resume)
    fi

    if [ -n "${DEBUG}" ]; then
        args+=(--debug-io)
    fi

    echo "${args[@]}"
}

# ======================================================================
# Curriculum loop
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Starting Phase-2 hold-out curriculum (${NUM_PHASES} phases, sequential)"
echo "══════════════════════════════════════════════════════════════"

PHASE_FAILED=""

for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r phase_num game display <<< "${phase_def}"

    if [ "${phase_num}" -lt "${RESUME_PHASE}" ]; then
        echo ""
        echo "[run_phase2] Skipping phase ${phase_num} (${display}) — resuming from phase ${RESUME_PHASE}"
        continue
    fi

    # Translate prior shared bank onto this game's action vocabulary
    # (no-op in per_game mode or when TRANSLATE_ON_BOUNDARY=0).
    translate_bank_for_target "${game}" || true

    step_start=$(( (phase_num - 1) * ITERS_PER_PHASE ))
    step_end=$(( phase_num * ITERS_PER_PHASE ))
    is_first=$([ "${phase_num}" -eq "${RESUME_PHASE}" ] && [ "${RESUME_PHASE}" -eq 1 ] && echo "true" || echo "false")

    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  Phase ${phase_num}/${NUM_PHASES}: ${display}"
    echo "│  Slug:       ${game}"
    echo "│  In-genre:   ${IN_GENRE_SOURCE[${game}]:-(no in-genre source)}"
    echo "│  Steps:      ${step_start} → ${step_end} (${ITERS_PER_PHASE} iterations)"
    echo "│  Episodes:   ${EPISODES} rollouts per step"
    echo "│  Mode:       $([ "${is_first}" = "true" ] && echo "SEED FROM PHASE-1" || echo "RESUME (carry-over LoRA + bank)")"
    echo "│  Anchor:     ${BASELINE_ANCHOR[${game}]:-n/a}"
    echo "└──────────────────────────────────────────────────────────┘"

    PHASE_ARGS=()
    read -ra PHASE_ARGS <<< "$(build_train_args "${game}" "${step_end}" "${is_first}")"

    if [ "${MANAGE_VLLM}" = "1" ]; then
        # shellcheck disable=SC2086
        PHASE_ARGS+=(--vllm-gpus ${VLLM_GPUS})
        PHASE_ARGS+=(--grpo-devices ${GRPO_GPUS})
        PHASE_ARGS+=(--vllm-base-port "${PORT}")
        PHASE_ARGS+=(--vllm-gpu-util "${GPU_UTIL}")
        PHASE_ARGS+=(--speculative-model "${SPEC_MODEL}")
        PHASE_ARGS+=(--num-speculative-tokens "${SPEC_TOKENS}")
    else
        PHASE_ARGS+=(--no-manage-vllm)
        PHASE_ARGS+=(--vllm-url "http://localhost:${PORT}/v1")
    fi

    echo "[run_phase2] Training args: ${PHASE_ARGS[*]}"
    echo ""

    if python scripts/run_coevolution.py "${PHASE_ARGS[@]}"; then
        echo ""
        echo "[run_phase2] Phase ${phase_num} (${display}) completed successfully."
        save_phase2_snapshot "${phase_num}" "${game}" "${display}" "${step_end}"
    else
        PHASE_FAILED="${phase_num}"
        echo ""
        echo "[run_phase2] ERROR: Phase ${phase_num} (${display}) FAILED."
        save_phase2_snapshot "${phase_num}" "${game}" "${display}_FAILED" "${step_end}"
        break
    fi
done

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ -n "${PHASE_FAILED}" ]; then
    echo "  Phase-2 curriculum STOPPED at phase ${PHASE_FAILED}"
    echo "  Resume with: RESUME_PHASE=${PHASE_FAILED} RUN_DIR=${RUN_DIR} \\"
    echo "                 PHASE1_SNAPSHOT=${PHASE1_SNAPSHOT} \\"
    echo "                 bash scripts/run_phase2_holdout.sh"
else
    echo "  Phase-2 curriculum COMPLETE"
    echo "  All ${NUM_PHASES} phases finished successfully."
fi
echo ""
echo "  Run dir:        ${RUN_DIR}"
echo "  Snapshot dir:   ${SNAPSHOT_DIR}/"
echo "  Bank mode:      ${BANK_MODE}"
echo "  Translation:    ${TRANSLATE_ON_BOUNDARY}"
echo "  Budget:         ${BUDGET} (INFER_ONLY=${INFER_ONLY})"
echo "══════════════════════════════════════════════════════════════"

if [ -n "${PHASE_FAILED}" ]; then
    exit 1
fi
