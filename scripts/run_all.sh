#!/usr/bin/env bash
# ======================================================================
#  Curriculum Co-Evolution: Sequential per-game training with
#  checkpoint + skill bank snapshots between phases.
#
#  Phase 1: Candy Crush   (15 iterations, 8 rollout episodes)
#  Phase 2: 2048          (15 iterations, 8 rollout episodes)
#  Phase 3: Tetris        (15 iterations, 8 rollout episodes)
#  Phase 4: Avalon        (15 iterations, 8 rollout episodes)
#  Phase 5: Diplomacy     (15 iterations, 8 rollout episodes)
#
#  Starts from the SFT cold-start checkpoint adapters.
#  After each phase, a full snapshot (adapters + skill banks) is saved
#  under <run_dir>/phase_snapshots/phase_<N>_<game>/.
#
#  Prerequisites:
#    conda activate game-ai-agent
#    pip install wandb tensorboard peft   # one-time
#
#  Usage:
#    bash scripts/run_all.sh
#
#    # Enable debug I/O logging:
#    DEBUG=1 bash scripts/run_all.sh
#
#    # Resume from a specific phase (e.g. phase 3 = tetris):
#    RESUME_PHASE=3 bash scripts/run_all.sh
#
#    # Override episodes per game:
#    EPISODES=8 bash scripts/run_all.sh
#
#    # Override iterations per phase:
#    ITERS_PER_PHASE=10 bash scripts/run_all.sh
#
#    # Use a specific run directory (for resume across restarts):
#    RUN_DIR=runs/Qwen3-8B_20260321_curriculum bash scripts/run_all.sh
#
#    # Legacy mode (external vLLM):
#    MANAGE_VLLM=0 bash scripts/run_all.sh
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

# ── PYTHONPATH ────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Configurable parameters ──────────────────────────────────────────
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP:-4}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
MANAGE_VLLM="${MANAGE_VLLM:-1}"

ITERS_PER_PHASE="${ITERS_PER_PHASE:-15}"
EPISODES="${EPISODES:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution}"
RUN_DIR="${RUN_DIR:-}"
DEBUG="${DEBUG:-}"
RESUME_PHASE="${RESUME_PHASE:-1}"
VLLM_GPUS="${VLLM_GPUS:-0 1 2 3}"
GRPO_GPUS="${GRPO_GPUS:-4 5 6 7}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# Cold-start adapter paths (SFT-pretrained)
COLDSTART_DIR="${COLDSTART_DIR:-runs/sft_coldstart}"
COLDSTART_DECISION="${COLDSTART_DIR}/decision"
COLDSTART_SKILLBANK="${COLDSTART_DIR}/skillbank"

# ── Curriculum phase definitions ─────────────────────────────────────
# Format: "phase_number:game_name:display_name"
PHASES=(
    "1:candy_crush:Candy Crush"
    "2:twenty_forty_eight:2048"
    "3:tetris:Tetris"
    "4:avalon:Avalon"
    "5:diplomacy:Diplomacy"
)
NUM_PHASES=${#PHASES[@]}

# ── Cleanup on exit (only for legacy mode) ────────────────────────────
VLLM_PID=""
cleanup() {
    echo ""
    echo "[run_all] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_all] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_all] Done."
}
trap cleanup EXIT INT TERM

# ======================================================================
# Phase 0: Resolve run directory + ensure LoRA adapters
# ======================================================================
echo "══════════════════════════════════════════════════════════════"
echo "  Curriculum Co-Evolution Training"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:            ${MODEL}"
echo "  Phases:           ${NUM_PHASES} (${ITERS_PER_PHASE} iters each)"
echo "  Total iterations: $((NUM_PHASES * ITERS_PER_PHASE))"
echo "  Episodes/game:    ${EPISODES}"
echo "  Checkpoint:       every ${CKPT_INTERVAL} steps"
echo "  Debug I/O:        ${DEBUG:-disabled}"
echo "  Resume phase:     ${RESUME_PHASE}"
echo "  Cold-start:       ${COLDSTART_DIR}"
if [ "${MANAGE_VLLM}" = "1" ]; then
    echo "  GPU mode:         MANAGED (persistent vLLM + FSDP)"
    echo "  vLLM GPUs:        ${VLLM_GPUS}"
    echo "  GRPO GPUs:        ${GRPO_GPUS}"
    echo "  Spec decode:      ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
else
    echo "  GPU mode:         LEGACY (vLLM TP=${TP} + separate GRPO GPUs)"
    echo "  vLLM port:        ${PORT}"
fi
echo ""
echo "  Curriculum schedule:"
for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r pnum game display <<< "${phase_def}"
    step_start=$(( (pnum - 1) * ITERS_PER_PHASE ))
    step_end=$(( pnum * ITERS_PER_PHASE - 1 ))
    echo "    Phase ${pnum}: ${display} (steps ${step_start}–${step_end})"
done
echo "══════════════════════════════════════════════════════════════"

echo ""
echo "[run_all] Ensuring LoRA adapters exist (from SFT cold-start)..."

# Validate cold-start directory
if [ ! -d "${COLDSTART_DECISION}" ]; then
    echo "[run_all] ERROR: Cold-start decision adapters not found: ${COLDSTART_DECISION}"
    echo "[run_all]   Run scripts/run_sft_coldstart.sh first, or set COLDSTART_DIR"
    exit 1
fi
if [ ! -d "${COLDSTART_SKILLBANK}" ]; then
    echo "[run_all] ERROR: Cold-start skillbank adapters not found: ${COLDSTART_SKILLBANK}"
    echo "[run_all]   Run scripts/run_sft_coldstart.sh first, or set COLDSTART_DIR"
    exit 1
fi

RESOLVED_RUN_DIR=$(python -c "
import sys, os
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from trainer.coevolution.config import CoEvolutionConfig, prepare_adapters
from pathlib import Path

decision_dir = '${COLDSTART_DECISION}'
skillbank_dir = '${COLDSTART_SKILLBANK}'

pretrained = {}
for name in ['skill_selection', 'action_taking']:
    p = Path(decision_dir) / name
    if p.exists():
        pretrained[name] = str(p)
for name in ['segment', 'contract', 'curator']:
    p = Path(skillbank_dir) / name
    if p.exists():
        pretrained[name] = str(p)

cfg = CoEvolutionConfig(
    model_name='${MODEL}',
    pretrained_adapter_paths=pretrained,
)
run_dir_override = '${RUN_DIR}'
if run_dir_override:
    cfg.run_dir = run_dir_override
cfg.resolve_paths()

result = prepare_adapters(cfg)
loaded = [n for n in result if n in pretrained]
inited = [n for n in result if n not in pretrained]
if loaded:
    print(f'Loaded {len(loaded)} cold-start adapter(s): {loaded}', file=sys.stderr)
if inited:
    print(f'Random-init {len(inited)} adapter(s): {inited}', file=sys.stderr)
print(cfg.run_dir)
")

RUN_DIR="${RESOLVED_RUN_DIR}"
export ADAPTER_DIR="${RUN_DIR}/lora_adapters"
SNAPSHOT_DIR="${RUN_DIR}/phase_snapshots"
mkdir -p "${SNAPSHOT_DIR}"

echo "[run_all] Run dir:      ${RUN_DIR}"
echo "[run_all] Adapter dir:  ${ADAPTER_DIR}"
echo "[run_all] Snapshot dir: ${SNAPSHOT_DIR}"

# ======================================================================
# Helper: save a phase snapshot (checkpoint + skill bank + metadata)
# ======================================================================
save_phase_snapshot() {
    local phase_num="$1"
    local game="$2"
    local display="$3"
    local step_end="$4"

    local snap_name="phase_${phase_num}_${game}"
    local snap_path="${SNAPSHOT_DIR}/${snap_name}"

    echo ""
    echo "[run_all] ── Saving phase snapshot: ${snap_name} ──"

    mkdir -p "${snap_path}"

    # Copy LoRA adapters
    if [ -d "${RUN_DIR}/lora_adapters" ]; then
        echo "[run_all]   Copying LoRA adapters..."
        cp -r "${RUN_DIR}/lora_adapters" "${snap_path}/lora_adapters"
    fi

    # Copy skill banks (the full per-game skill bank directory)
    if [ -d "${RUN_DIR}/skillbank" ]; then
        echo "[run_all]   Copying skill banks..."
        cp -r "${RUN_DIR}/skillbank" "${snap_path}/skillbank"
    fi

    # Copy the latest checkpoint directory
    local latest_ckpt=""
    if [ -d "${RUN_DIR}/checkpoints" ]; then
        latest_ckpt=$(ls -d "${RUN_DIR}/checkpoints"/step_* 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "${latest_ckpt}" ]; then
            echo "[run_all]   Copying latest checkpoint: $(basename "${latest_ckpt}")"
            cp -r "${latest_ckpt}" "${snap_path}/checkpoint"
        fi
    fi

    # Write phase metadata
    cat > "${snap_path}/phase_meta.json" <<METAEOF
{
    "phase": ${phase_num},
    "game": "${game}",
    "display_name": "${display}",
    "step_end": ${step_end},
    "iters_per_phase": ${ITERS_PER_PHASE},
    "episodes_per_game": ${EPISODES},
    "model": "${MODEL}",
    "timestamp": "$(date -Iseconds)",
    "run_dir": "${RUN_DIR}",
    "latest_checkpoint": "${latest_ckpt:-none}"
}
METAEOF

    # Count skills per game in the snapshot
    local skill_summary=""
    if [ -d "${snap_path}/skillbank" ]; then
        for bank_file in "${snap_path}"/skillbank/*/skill_bank.jsonl; do
            if [ -f "${bank_file}" ]; then
                local gname
                gname=$(basename "$(dirname "${bank_file}")")
                local count
                count=$(wc -l < "${bank_file}" 2>/dev/null || echo 0)
                skill_summary="${skill_summary}  ${gname}=${count}"
            fi
        done
    fi

    echo "[run_all]   Phase ${phase_num} snapshot saved to: ${snap_path}"
    if [ -n "${skill_summary}" ]; then
        echo "[run_all]   Skill bank sizes:${skill_summary}"
    fi
    echo "[run_all] ── Snapshot complete ──"
    echo ""
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
    )

    if [ "${is_first_phase}" = "true" ]; then
        # First phase: load from cold-start SFT adapters
        if [ -d "${COLDSTART_DECISION}" ]; then
            args+=(--load-decision-adapters "${COLDSTART_DECISION}")
        fi
        if [ -d "${COLDSTART_SKILLBANK}" ]; then
            args+=(--load-skillbank-adapters "${COLDSTART_SKILLBANK}")
        fi
    else
        # Subsequent phases: resume from previous checkpoint
        args+=(--resume)
    fi

    if [ -n "${DEBUG}" ]; then
        args+=(--debug-io)
    fi

    echo "${args[@]}"
}

# ======================================================================
# Launch vLLM if needed (legacy mode)
# ======================================================================
if [ "${MANAGE_VLLM}" = "0" ]; then
    echo ""
    echo "[run_all] Starting vLLM server (legacy mode)..."
    bash scripts/launch_vllm_coevolution.sh &
    VLLM_PID=$!
    echo "[run_all] vLLM server PID: ${VLLM_PID}"

    echo "[run_all] Waiting for vLLM at http://localhost:${PORT}..."
    MAX_WAIT=300
    WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || \
           curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "[run_all] vLLM is ready! (waited ${WAITED}s)"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[run_all] ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $((WAITED % 30)) -eq 0 ]; then
            echo "[run_all]   ... still waiting (${WAITED}s / ${MAX_WAIT}s)"
        fi
    done

    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[run_all] ERROR: vLLM did not start within ${MAX_WAIT}s"
        exit 1
    fi
fi

# ======================================================================
# Curriculum training loop
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Starting curriculum training (${NUM_PHASES} phases)"
echo "══════════════════════════════════════════════════════════════"

PHASE_FAILED=""

for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r phase_num game display <<< "${phase_def}"

    # Skip phases before RESUME_PHASE
    if [ "${phase_num}" -lt "${RESUME_PHASE}" ]; then
        echo ""
        echo "[run_all] Skipping phase ${phase_num} (${display}) — resuming from phase ${RESUME_PHASE}"
        continue
    fi

    step_start=$(( (phase_num - 1) * ITERS_PER_PHASE ))
    step_end=$(( phase_num * ITERS_PER_PHASE ))
    is_first=$([ "${phase_num}" -eq "${RESUME_PHASE}" ] && [ "${RESUME_PHASE}" -eq 1 ] && echo "true" || echo "false")

    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  Phase ${phase_num}/${NUM_PHASES}: ${display}"
    echo "│  Game:  ${game}"
    echo "│  Steps: ${step_start} → ${step_end} (${ITERS_PER_PHASE} iterations)"
    echo "│  Episodes: ${EPISODES} rollouts per step"
    echo "│  Mode:  $([ "${is_first}" = "true" ] && echo "COLD-START" || echo "RESUME")"
    echo "└──────────────────────────────────────────────────────────┘"

    # Build phase-specific training args
    PHASE_ARGS=()
    read -ra PHASE_ARGS <<< "$(build_train_args "${game}" "${step_end}" "${is_first}")"

    # Add GPU/vLLM args based on mode
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

    echo "[run_all] Training args: ${PHASE_ARGS[*]}"
    echo ""

    # Run the training phase
    if python scripts/run_coevolution.py "${PHASE_ARGS[@]}"; then
        echo ""
        echo "[run_all] Phase ${phase_num} (${display}) completed successfully."

        # Save phase snapshot (checkpoint + skill bank)
        save_phase_snapshot "${phase_num}" "${game}" "${display}" "${step_end}"
    else
        PHASE_FAILED="${phase_num}"
        echo ""
        echo "[run_all] ERROR: Phase ${phase_num} (${display}) FAILED."
        echo "[run_all] Saving partial snapshot before aborting..."
        save_phase_snapshot "${phase_num}" "${game}" "${display}_FAILED" "${step_end}"
        break
    fi
done

# ======================================================================
# Summary
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ -n "${PHASE_FAILED}" ]; then
    echo "  Curriculum training STOPPED at phase ${PHASE_FAILED}"
    echo "  Resume with: RESUME_PHASE=${PHASE_FAILED} RUN_DIR=${RUN_DIR} bash scripts/run_all.sh"
else
    echo "  Curriculum training COMPLETE"
    echo "  All ${NUM_PHASES} phases finished successfully."
fi
echo ""
echo "  Run dir:    ${RUN_DIR}"
echo "  Snapshots:  ${SNAPSHOT_DIR}/"
if [ -d "${SNAPSHOT_DIR}" ]; then
    echo ""
    echo "  Phase snapshots:"
    for d in "${SNAPSHOT_DIR}"/phase_*; do
        if [ -d "$d" ]; then
            local_name=$(basename "$d")
            skill_count=0
            if [ -d "$d/skillbank" ]; then
                skill_count=$(find "$d/skillbank" -name "skill_bank.jsonl" -exec cat {} + 2>/dev/null | wc -l || echo 0)
            fi
            echo "    ${local_name}/ (${skill_count} skills)"
        fi
    done
fi
echo "══════════════════════════════════════════════════════════════"

if [ -n "${PHASE_FAILED}" ]; then
    exit 1
fi

echo ""
echo "[run_all] Curriculum training complete."
