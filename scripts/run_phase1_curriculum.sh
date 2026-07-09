#!/usr/bin/env bash
# ======================================================================
#  Phase-1 source-GRPO curriculum — 6 games × 10 steps, sequential, with
#  bank + LoRA carry-over between games.
#
#  Implements the plan locked in
#    training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1
#  (game roster + curriculum order, refreshed 2026-05-03 PM from the new
#  Cold-start-out-gymv/latest 4-backbone teacher table — see §13
#  Changelog).
#
#  Curriculum order (data-driven from new SFT cold-start; every game has
#  non-zero teacher reward across all 4 frontier teachers):
#    Phase 1: gymv_thunder_force_iii  (shmup,             teacher 269-750)
#    Phase 2: gymv_altered_beast      (beat-em-up,        teacher 119-425)
#    Phase 3: gymv_columns            (puzzle,            teacher  63-160)
#    Phase 4: gymv_dynamite_headdy    (action-platformer, teacher  75- 94)
#    Phase 5: candy_crush             (paper Table 3,     match-3)
#    Phase 6: tetris                  (paper Table 3,     spatial puzzle)
#
#  Phase-2 hold-out roster (run via scripts/run_phase2_holdout.sh — pairs
#  each Phase-2 gymv game in-genre with a Phase-1 source so the cross-game
#  translator has the closest possible source vocabulary):
#    SoR2          ← AlteredBeast    (in-genre lift, healthy ↔ healthy)
#    SpaceHarrierII ← ThunderForceIII (scale-jump test, ~30× reward)
#    Airstriker    ← ThunderForceIII (easier in-genre sanity)
#    Strider       ← DynamiteHeaddy  (partial-signal rescue test)
#    2048          ← tetris+Columns  (grid-puzzle composition)
#    super_mario   ← (no in-genre)   (transfer-distance bound)
#
#  Total: 60 GRPO steps. Wall-clock ~30-36 h sequential at ~30-36 min/step
#  (tightened wall-time after the v4 segmentation cap landed; previously
#  ~54 h before the SKILLBANK_MAX_SKILL_NAMES bound on segmentation cost).
#
#  Bank + LoRA carry over phase-to-phase by design (Option C in §11.2).
#  After each phase a snapshot lands at
#    <run_dir>/phase_snapshots/phase_<N>_<game>/{lora_adapters,
#    skillbank, checkpoint, phase_meta.json}
#  so any per-game-best evaluation can re-load that snapshot independently
#  of the post-curriculum state.
#
#  ── 8×H200 dual-stack layout (default, 2026-05-03) ──────────────────
#  Hosts the 9B actor AND the 35B-A3B judge simultaneously so the live
#  promotion gates (E0/E1/E2) and crafter / harness control plane hit
#  the 35B endpoint instead of silently falling back to 9B-self-judging.
#
#    GPUs 0–3 → 4× Qwen3.5-9B vLLM (TP=1 each, ports 8000–8003)
#    GPUs 4–5 → 1× Qwen3.5-35B-A3B vLLM (TP=2 + expert-parallel,
#               port 8004) — judge / crafter / harness / orchestrator.
#               Auto-launched in the background by this script when
#               JUDGE_MODE=auto (the default).
#    GPUs 6–7 → FSDP=2 GRPO trainer.
#
#  Why FSDP=2 not FSDP=4: GRPO trains LoRA only (~250 M params across
#  5 adapters); the frozen 9B base is FSDP-sharded but does no
#  backward, so per-GPU memory ≈ 12 GB on H200 141 GB.  The
#  orchestrator pipelines GRPO(N) with rollout(N+1)
#  (orchestrator.py L559/742/1093) so train_time hides inside the
#  ~25-30 min/step rollout window — FSDP=2 ends up FREE wall-clock-wise
#  while freeing 2 GPUs for the live 35B judge.
#
#  Layout overrides (env vars):
#    LAYOUT=dual_stack       → default; 4×9B + 1×35B(TP=2) + FSDP=2
#    LAYOUT=dual_stack_fsdp4 → 2×9B + 1×35B(TP=2) + FSDP=4 + EPISODES=16
#                              (use only when train_time would exceed
#                              rollout even after pipelining)
#    LAYOUT=actor_only       → 4×9B + FSDP=4, no 35B server (legacy /
#                              ablations; 9B-self-judges)
#    JUDGE_MODE=auto|external|off
#                            → control whether this script launches the
#                              35B server itself.  external=already
#                              running elsewhere (set JUDGE_URL).
#                              off=no 35B routing wired (NOT recommended).
#
#  Prerequisites:
#    conda activate game-ai-agent
#    pip install wandb tensorboard peft         # one-time
#    bash install/install_gymv.sh \
#         && bash install/gymv_temporal_patch/apply_patch.sh   # one-time
#                                                              (gym_v + Mega Drive ROMs)
#    Phase-1-source SFT cold-start adapters present at
#      runs/sft_coldstart/{decision,skillbank}/{skill_selection,
#      action_taking,segment,contract,curator}
#
#  Usage:
#    bash scripts/run_phase1_curriculum.sh
#
#    # Override per-phase iteration count (default 15; was 10 between
#    # 2026-05-04 v4 fix and v8 query-engine fix):
#    ITERS_PER_PHASE=20 bash scripts/run_phase1_curriculum.sh
#
#    # Override episodes per step (default 8 — matches paper Table 3):
#    EPISODES=4 bash scripts/run_phase1_curriculum.sh
#
#    # Resume from a specific phase (e.g. phase 4 = Strider):
#    RESUME_PHASE=4 RUN_DIR=runs/Qwen3.5-9B_<ts>_phase1 \
#      bash scripts/run_phase1_curriculum.sh
#
#    # 35B server already running on a 2nd box / port:
#    JUDGE_MODE=external JUDGE_URL=http://otherhost:8001/v1 \
#      bash scripts/run_phase1_curriculum.sh
#
#    # Single-stack (skip the 35B judge, max GRPO throughput):
#    LAYOUT=actor_only JUDGE_MODE=off bash scripts/run_phase1_curriculum.sh
#
#    # Use external vLLM instead of MANAGED 9B (legacy):
#    MANAGE_VLLM=0 VLLM_URL=http://localhost:8000/v1 \
#      bash scripts/run_phase1_curriculum.sh
#
#    # Shared-bank lifelong-learning mode (one bank file across all 6
#    # games, with cross-game LLM translation at every phase boundary):
#    BANK_MODE=shared bash scripts/run_phase1_curriculum.sh
#
#    # Shared mode without between-phase translation (just lets the
#    # harness's task-axis veto enforce per-game eligibility from the
#    # source-stamped feasible_tasks alone):
#    BANK_MODE=shared TRANSLATE_ON_BOUNDARY=0 \
#      bash scripts/run_phase1_curriculum.sh
#
#  Cross-refs:
#    - scripts/run_2048.sh                       (single-game variant of this layout)
#    - scripts/use_35b_judge.sh                  (manual VLLM_BASE_URL_MAP wiring)
#    - inference/serve_qwen35_35b_a3b.sh         (the 35B serve script this calls)
#    - env_wrappers/gymv_temporal_nl_wrapper.py  (the 4 gymv_* slugs)
#    - trainer/coevolution/episode_runner.py     (GYMV_TEMPORAL_GAMES_SET dispatch)
#    - baselines/README.md § "Gym-V benchmark scope" (per-game baseline anchors for §4.3 sanity bar)
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

# §4.1 lock (relaxed 2026-05-04 to 15 / phase after the v8 query-engine
# fix unblocked the actor-side skill loop): 15 steps per phase, giving
# the skill_selection LoRA enough GRPO iterations to overcome cold-start
# noise on each game.  Checkpoint at every step (CKPT_INTERVAL=1, was 5)
# so we have full per-step lineage for skill-bank diff analysis +
# arbitrary-step resume.  At ~250 MB LoRA + ~10 MB bank per checkpoint
# × 90 steps = ~22 GB total disk — tractable on local NVMe.  Override
# via ``CKPT_INTERVAL=5`` to restore coarser cadence.  ``EPISODES``
# default is set per LAYOUT below (8 for FSDP=2, 16 for FSDP=4).
# Override via ``ITERS_PER_PHASE`` env var.
ITERS_PER_PHASE="${ITERS_PER_PHASE:-15}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-game-ai-coevolution-phase1}"
RUN_DIR="${RUN_DIR:-}"
DEBUG="${DEBUG:-}"
RESUME_PHASE="${RESUME_PHASE:-1}"
SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"

# ── Shared-bank lifelong-learning mode (opt-in, see
#    training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §11.x) ─
#
#   BANK_MODE=per_game   (default, legacy)
#       One ``skill_bank.jsonl`` per game under
#       ``<run_dir>/skillbank/<game>/``.  Cross-game effects only at
#       the LoRA-weight level (Option C carry-over).
#
#   BANK_MODE=shared
#       One ``<run_dir>/skillbank/skill_bank.jsonl`` shared across all
#       phases.  Per-game eligibility is enforced at runtime by the
#       harness's task-axis veto (``feasible_tasks``,
#       ``harness/README §22``).  Pair with TRANSLATE_ON_BOUNDARY=1 so
#       skills mined on phase N are re-grounded onto phase N+1's action
#       vocabulary at the phase boundary (no judge cost during a phase).
#
#   TRANSLATE_ON_BOUNDARY=1
#       Between phases, run ``python -m
#       skill_agents.skill_bank.translate_for_target`` against the
#       shared bank to add cross-game-translated derivatives for the
#       upcoming phase.  No-op when BANK_MODE=per_game.
BANK_MODE="${BANK_MODE:-per_game}"
TRANSLATE_ON_BOUNDARY="${TRANSLATE_ON_BOUNDARY:-0}"
case "${BANK_MODE}" in
    per_game|shared) ;;
    *)
        echo "ERROR: unknown BANK_MODE=${BANK_MODE} (expected per_game|shared)"
        exit 1
        ;;
esac
# Auto-enable per-boundary translation when shared mode is requested
# and the user hasn't pinned TRANSLATE_ON_BOUNDARY explicitly.  Skipped
# in per_game mode (translation has no effect when banks are isolated
# per-game on disk).
if [ "${BANK_MODE}" = "shared" ] && [ -z "${TRANSLATE_ON_BOUNDARY_PINNED:-}" ] && [ "${TRANSLATE_ON_BOUNDARY}" = "0" ]; then
    TRANSLATE_ON_BOUNDARY=1
fi

# 8×H200 layout selector (see banner above for full table).
LAYOUT="${LAYOUT:-dual_stack}"
JUDGE_MODE="${JUDGE_MODE:-auto}"

# ── 35B control-plane wiring (Crafter / Promotion / Harness) ─────────
# When JUDGE_MODE != off the script wires the 35B endpoint into
# VLLM_BASE_URL_MAP (line ~348), but the orchestrator only *issues
# 35B calls* when these flags are passed to run_coevolution.py.
# Without them the 35B server idles and the run becomes a 9B-only
# ablation (mirrors the gap that broke the 2026-05-03 21:15 run).
#
# Defaults follow training_notes/coevo-3phase-cross-game-ood-transfer-plan.md
# §11.2 (live curator + crafter) and §501-502 (35B as the shared
# control-plane backbone) — all three ON when a judge is reachable.
#
#   CRAFTER_PROMOTION   1 → --crafter-promotion-enabled
#                           + --crafter-cycle-every-k-steps ${CRAFTER_CYCLE_K}
#                       0 → skip Phase B′ (legacy curator-only baseline)
#   CRAFTER_CYCLE_K     K=5 default → run Crafter every 5 steps; ~+30s/cycle.
#                       Set to 1 for paper-style every-step (≈3× the 35B
#                       budget over a phase); 0 = every step.
#   HARNESS_ENABLED     1 → --harness-enabled
#                           (eligibility filter + validate_invocation veto;
#                            0 LLM calls, pure-Python gating)
#   GAME_SCHEMA_ENABLED 1 → --game-schema-enabled (default: 1)
#                           Path 1: 1 × BACKBONE_JUDGE_MODEL (35B-A3B)
#                           call per game per phase boundary produces a
#                           compact GameProfile (goal / win_signal /
#                           hazards / key_actions / failure_modes) that
#                           gets injected into the actor's SYSTEM_PROMPT
#                           and SKILL_SELECTION_SYSTEM_PROMPT for the
#                           duration of the phase. The same call also
#                           caches a <state> exemplar at
#                             ${RUN_DIR}/phase_artifacts/<game>.schema.json
#                           for Path 2 / Path 4 to reuse as a few-shot
#                           anchor without firing their own 35B calls.
#                           Off when JUDGE_MODE=off (auto-disabled below).
#                           See trainer/coevolution/_game_schema.py.
#   GAME_SCHEMA_MAX_TOKENS  Token budget for the 35B response (default 1024).
#   GAME_SCHEMA_TIMEOUT_S   Hard timeout per 35B call (default 60s); on
#                           timeout the trainer falls back to a
#                           deterministic minimum profile and continues.
#
#   PROMOTION_GATE_MODE  Forwarded as --crafter-promotion-gate-mode.
#                       'offline-synthetic'      = no LLM calls, all
#                                                  proposals → LIMITED_PASS;
#                       'offline-with-llm-judge' = +1 BACKBONE_JUDGE_MODEL
#                                                  (35B-A3B) call per
#                                                  proposal so visibly
#                                                  bad ones can FAIL ⇒
#                                                  REJECT (DEFAULT — fires
#                                                  the 35B endpoint that
#                                                  would otherwise sit
#                                                  idle in Phase 1);
#                       'live'                   = full GateService
#                                                  (diagnostic only;
#                                                  Stage 3a FAILs without
#                                                  target adapters).
CRAFTER_PROMOTION="${CRAFTER_PROMOTION:-1}"
CRAFTER_CYCLE_K="${CRAFTER_CYCLE_K:-5}"
HARNESS_ENABLED="${HARNESS_ENABLED:-1}"
PROMOTION_GATE_MODE="${PROMOTION_GATE_MODE:-offline-with-llm-judge}"
# Path 1 (phase-start GameProfile) — 1 × BACKBONE_JUDGE_MODEL call per
# game per phase boundary.  Default ON to fire the 35B endpoint that
# would otherwise idle through phases that don't trigger a Crafter
# cycle.  See trainer/coevolution/_game_schema.py.
GAME_SCHEMA_ENABLED="${GAME_SCHEMA_ENABLED:-1}"
GAME_SCHEMA_MAX_TOKENS="${GAME_SCHEMA_MAX_TOKENS:-1024}"
GAME_SCHEMA_TIMEOUT_S="${GAME_SCHEMA_TIMEOUT_S:-60}"

# Path 2 (LLM Crafter) — supplemental 35B-A3B teacher driver for the
# Crafter.  When enabled, in addition to the deterministic rule-based
# Crafter, the trainer fires up to ${LLM_CRAFTER_K_MAX} parallel 35B
# calls per game per step (one per FailureTrace) to propose
# patch / hypothesize / retire BankMutationProposals.  Default ON so
# Phase 1 exercises the 35B endpoint on every step a failure trace is
# present; auto-disabled when JUDGE_MODE=off.  See
# trainer/coevolution/_llm_crafter.py.
LLM_CRAFTER_ENABLED="${LLM_CRAFTER_ENABLED:-1}"
LLM_CRAFTER_K_MAX="${LLM_CRAFTER_K_MAX:-2}"
LLM_CRAFTER_MAX_TOKENS="${LLM_CRAFTER_MAX_TOKENS:-1024}"
LLM_CRAFTER_TIMEOUT_S="${LLM_CRAFTER_TIMEOUT_S:-60}"

# Path 4 (LLM Harness validator) — post-LLM 35B validation pass on
# the harness's chosen skill.  Hybrid policy: bootstrap window
# (steps below ${LLM_HARNESS_BOOTSTRAP_STEPS}) always fires the LLM
# validator; afterwards the validator only fires when the
# deterministic verdict was uncertain (SHADOW status, no can_handle
# evidence, translation-rewritten contracts).  Verdicts can ONE-WAY
# downgrade admit→veto.  Default ON; auto-disabled when JUDGE_MODE=off.
# See trainer/coevolution/_llm_harness_validator.py.
LLM_HARNESS_VALIDATOR_ENABLED="${LLM_HARNESS_VALIDATOR_ENABLED:-1}"
LLM_HARNESS_BOOTSTRAP_STEPS="${LLM_HARNESS_BOOTSTRAP_STEPS:-20}"
LLM_HARNESS_MAX_TOKENS="${LLM_HARNESS_MAX_TOKENS:-256}"
LLM_HARNESS_TIMEOUT_S="${LLM_HARNESS_TIMEOUT_S:-30}"

# Auto-disable 35B-dependent flags when the judge is intentionally off
# (ablation runs).  Otherwise the orchestrator would try to POST to a
# non-existent endpoint and crash mid-step.
if [ "${JUDGE_MODE}" = "off" ]; then
    if [ "${CRAFTER_PROMOTION}" = "1" ] || [ "${HARNESS_ENABLED}" = "1" ]; then
        echo "[run_phase1] JUDGE_MODE=off → forcing CRAFTER_PROMOTION=0 + HARNESS_ENABLED=0" \
             "(35B control plane disabled; legacy curator-only path)"
        CRAFTER_PROMOTION=0
        HARNESS_ENABLED=0
    fi
    if [ "${PROMOTION_GATE_MODE}" != "offline-synthetic" ]; then
        echo "[run_phase1] JUDGE_MODE=off → forcing PROMOTION_GATE_MODE=offline-synthetic" \
             "(no 35B endpoint to query)"
        PROMOTION_GATE_MODE="offline-synthetic"
    fi
    if [ "${GAME_SCHEMA_ENABLED}" = "1" ]; then
        echo "[run_phase1] JUDGE_MODE=off → forcing GAME_SCHEMA_ENABLED=0" \
             "(no 35B endpoint to query for phase-start GameProfile)"
        GAME_SCHEMA_ENABLED=0
    fi
    if [ "${LLM_CRAFTER_ENABLED}" = "1" ]; then
        echo "[run_phase1] JUDGE_MODE=off → forcing LLM_CRAFTER_ENABLED=0" \
             "(no 35B endpoint to query for Path 2 LLM Crafter)"
        LLM_CRAFTER_ENABLED=0
    fi
    if [ "${LLM_HARNESS_VALIDATOR_ENABLED}" = "1" ]; then
        echo "[run_phase1] JUDGE_MODE=off → forcing LLM_HARNESS_VALIDATOR_ENABLED=0" \
             "(no 35B endpoint to query for Path 4 LLM Harness validator)"
        LLM_HARNESS_VALIDATOR_ENABLED=0
    fi
fi

# Defensive: validate the supplied gate mode.  Anything we don't
# recognise gets stamped back to the synthetic floor so the driver's
# argparse doesn't die mid-step.
case "${PROMOTION_GATE_MODE}" in
    offline-synthetic|offline-with-llm-judge|live) ;;
    *)
        echo "[run_phase1] WARN: unknown PROMOTION_GATE_MODE='${PROMOTION_GATE_MODE}'; " \
             "falling back to offline-synthetic" >&2
        PROMOTION_GATE_MODE="offline-synthetic"
        ;;
esac

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

# 35B-A3B judge endpoint (used by the orchestrator + skill_eval gates).
JUDGE_PORT="${JUDGE_PORT:-8004}"
JUDGE_URL="${JUDGE_URL:-http://localhost:${JUDGE_PORT}/v1}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.92}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-35B-A3B}"

# Cold-start adapter paths (SFT-pretrained); same layout as run_all.sh
COLDSTART_DIR="${COLDSTART_DIR:-runs/sft_coldstart}"
COLDSTART_DECISION="${COLDSTART_DIR}/decision"
COLDSTART_SKILLBANK="${COLDSTART_DIR}/skillbank"

# ── Locked Phase-1 curriculum (training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1)
# Format: "phase_number:game_slug:display_name"
# Slugs match env_wrappers/gymv_temporal_nl_wrapper.GYMV_TEMPORAL_GAMES
# for the 4 gymv games and env_wrappers/game_configs.py for the 2 paper games.
PHASES=(
    "1:gymv_thunder_force_iii:Thunder Force III"
    "2:gymv_altered_beast:Altered Beast"
    "3:gymv_columns:Columns"
    "4:gymv_dynamite_headdy:Dynamite Headdy"
    "5:candy_crush:Candy Crush"
    "6:tetris:Tetris"
)
NUM_PHASES=${#PHASES[@]}

# Per-game baseline anchor for §4.3 sanity bar — populated from the new
# Cold-start-out-gymv/latest 4-backbone teacher table (see
# training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1, refresh
# 2026-05-03 PM). The "min teacher reward" is the worst case across the 4
# frontier rows (GPT-5.4 / Claude-4.6-Sonnet / Gemini-3.1-Pro /
# Qwen3-VL-235B); end-of-phase actor reward should clear this floor or the
# curriculum should be re-inspected.
declare -A BASELINE_ANCHOR=(
    ["gymv_thunder_force_iii"]="min teacher 269 (TF3 across 4 frontier teachers)"
    ["gymv_altered_beast"]="min teacher 119 (AlteredBeast)"
    ["gymv_columns"]="min teacher 63 (Columns)"
    ["gymv_dynamite_headdy"]="min teacher 75 (DynamiteHeaddy)"
    ["candy_crush"]="paper Figure 4 (±30%)"
    ["tetris"]="paper Figure 4 (±30%)"
)

# ── Cleanup on exit ───────────────────────────────────────────────────
VLLM_PID=""
JUDGE_PID=""
cleanup() {
    echo ""
    echo "[run_phase1] Shutting down..."
    if [ -n "${JUDGE_PID}" ] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "[run_phase1] Stopping 35B judge (PID ${JUDGE_PID})..."
        kill "${JUDGE_PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${JUDGE_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -9 "${JUDGE_PID}" 2>/dev/null || true
    fi
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[run_phase1] Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[run_phase1] Done."
}
trap cleanup EXIT INT TERM

# ── 35B-A3B judge server (auto-launched in dual_stack*) ──────────────
JUDGE_LOG=""

start_35b_judge() {
    if [ -z "${JUDGE_GPUS}" ] || [ "${JUDGE_TP}" = "0" ]; then
        echo "[run_phase1] LAYOUT=${LAYOUT}: 35B server not part of layout — skipping."
        return 0
    fi
    JUDGE_LOG="${RUN_DIR:-runs}/judge_35b.log"
    mkdir -p "$(dirname "${JUDGE_LOG}")"
    echo "[run_phase1] Auto-launching 35B-A3B judge:"
    echo "  GPUs:   ${JUDGE_GPUS}  (TP=${JUDGE_TP}, expert-parallel ON)"
    echo "  Port:   ${JUDGE_PORT}"
    echo "  Model:  ${JUDGE_MODEL}"
    echo "  Log:    ${JUDGE_LOG}"
    # NUM_SPEC_TOKENS=5 (v10, was 1): bumps the 35B's MTP speculative
    # decoding depth.  Path 3 (promotion judge) and Path 4 (harness
    # validator) emit short JSON verdicts (256-512 tok) and Path 2
    # (LLM crafter) emits ~1K tok JSON proposals — all decode-bound
    # workloads where speculation pays off.  Acceptance rate on
    # Qwen3.5-35B-A3B with its native MTP head is empirically ~70%
    # for short structured outputs, giving ~2-3x decode speedup.
    GPUS="${JUDGE_GPUS}" \
    TENSOR_PARALLEL="${JUDGE_TP}" \
    EXPERT_PARALLEL=1 \
    GPU_UTIL="${JUDGE_GPU_UTIL}" \
    PORT="${JUDGE_PORT}" \
    HOST="127.0.0.1" \
    MODEL="${JUDGE_MODEL}" \
    NUM_SPEC_TOKENS="${JUDGE_NUM_SPEC_TOKENS:-5}" \
        bash "${PROJECT_ROOT}/inference/serve_qwen35_35b_a3b.sh" \
            > "${JUDGE_LOG}" 2>&1 &
    JUDGE_PID=$!
    echo "[run_phase1] 35B server PID: ${JUDGE_PID}"

    echo "[run_phase1] Waiting for 35B server health on ${JUDGE_URL} ..."
    for _ in $(seq 1 120); do
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[run_phase1] 35B server is healthy."
            return 0
        fi
        if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
            echo "[run_phase1] 35B server died during startup — see ${JUDGE_LOG}"
            return 1
        fi
        sleep 5
    done
    echo "[run_phase1] 35B server did not become healthy within 600s — see ${JUDGE_LOG}"
    return 1
}

# Resolved later (after RUN_DIR is set) — wire JUDGE_MODE here so the
# pre-flight banner can show the final URL/model values.
#
# vLLM URL routing (v10 fix, 2026-05-04):
#   * VLLM_BASE_URLS:    pool of 4× 9B endpoints — unmapped requests
#                        round-robin across them.  This is what the
#                        skill_bank pipeline's API_func.ask_vllm()
#                        calls (segmentation / contract / curator
#                        teacher queries) use, balancing the load
#                        evenly instead of hammering :8000 alone.
#   * VLLM_BASE_URL_MAP: ONLY pin the 35B-A3B (single endpoint at
#                        :8004).  Do NOT pin the 9B here — that would
#                        route every ``ask_vllm(model="Qwen3.5-9B")``
#                        deterministically to one port (the v9 bug).
case "${JUDGE_MODE}" in
    auto|external)
        # Build "http://localhost:PORT/v1" entries from VLLM_GPUS array.
        # Whitespace-tolerant: VLLM_GPUS may be "0 1 2 3" or "0,1,2,3".
        _VLLM_URLS_LIST=""
        _i=0
        for _g in ${VLLM_GPUS//,/ }; do
            _p=$((PORT + _i))
            if [ -z "${_VLLM_URLS_LIST}" ]; then
                _VLLM_URLS_LIST="http://localhost:${_p}/v1"
            else
                _VLLM_URLS_LIST="${_VLLM_URLS_LIST},http://localhost:${_p}/v1"
            fi
            _i=$((_i + 1))
        done
        export VLLM_BASE_URLS="${_VLLM_URLS_LIST}"
        export VLLM_BASE_URL_MAP="${JUDGE_MODEL}=${JUDGE_URL}"
        export VLM_AGENT_BACKBONE_JUDGE_MODEL="${JUDGE_MODEL}"
        unset _VLLM_URLS_LIST _i _g _p
        ;;
    off)
        # Still build VLLM_BASE_URLS so the 9B pool round-robins
        # across all ports (skill_bank pipeline's ask_vllm() honours
        # this env var; without it, every call lands on :8000 — same
        # imbalance as v9).
        _VLLM_URLS_LIST=""
        _i=0
        for _g in ${VLLM_GPUS//,/ }; do
            _p=$((PORT + _i))
            if [ -z "${_VLLM_URLS_LIST}" ]; then
                _VLLM_URLS_LIST="http://localhost:${_p}/v1"
            else
                _VLLM_URLS_LIST="${_VLLM_URLS_LIST},http://localhost:${_p}/v1"
            fi
            _i=$((_i + 1))
        done
        export VLLM_BASE_URLS="${_VLLM_URLS_LIST}"
        unset _VLLM_URLS_LIST _i _g _p
        echo "[run_phase1] JUDGE_MODE=off → 35B routing NOT wired" \
             "(judge will fall back to 9B; only safe for ablations)."
        ;;
    *)
        echo "ERROR: unknown JUDGE_MODE=${JUDGE_MODE}" \
             "(expected auto|external|off)"
        exit 1
        ;;
esac

# ======================================================================
# Phase 0: Resolve run directory + ensure LoRA adapters
# ======================================================================
echo "══════════════════════════════════════════════════════════════"
echo "  Phase-1 Source Co-Evolution Curriculum"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:            ${MODEL}"
echo "  Phases:           ${NUM_PHASES} sequential games × ${ITERS_PER_PHASE} GRPO steps"
echo "  Total iterations: $((NUM_PHASES * ITERS_PER_PHASE))"
echo "  Episodes/step:    ${EPISODES}"
echo "  Snapshot every:   ${CKPT_INTERVAL} steps within each phase"
echo "  Debug I/O:        ${DEBUG:-disabled}"
echo "  Resume phase:     ${RESUME_PHASE}"
echo "  Cold-start:       ${COLDSTART_DIR}"
echo "  Bank mode:        ${BANK_MODE}$([ "${BANK_MODE}" = "shared" ] && [ "${TRANSLATE_ON_BOUNDARY}" = "1" ] && echo " (translate at every phase boundary)" || true)"
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
    echo "  35B control plane:"
    if [ "${CRAFTER_PROMOTION}" = "1" ]; then
        echo "    Crafter+Promotion:  ON  (every ${CRAFTER_CYCLE_K} steps)"
        echo "    Promotion gate:     ${PROMOTION_GATE_MODE}"
        if [ "${PROMOTION_GATE_MODE}" = "offline-with-llm-judge" ]; then
            echo "                        (1 × 35B BACKBONE_JUDGE_MODEL call per proposal,"
            echo "                         routed via VLLM_BASE_URL_MAP → port ${JUDGE_PORT:-?})"
        fi
    else
        echo "    Crafter+Promotion:  off"
    fi
    if [ "${HARNESS_ENABLED}" = "1" ]; then
        echo "    Harness:            ON  (eligibility + validate_invocation, 0 LLM)"
    else
        echo "    Harness:            off"
    fi
    if [ "${GAME_SCHEMA_ENABLED}" = "1" ]; then
        echo "    Phase GameProfile:  ON  (1 × 35B BACKBONE_JUDGE_MODEL call /"
        echo "                              game / phase boundary, max_tokens="
        echo "                              ${GAME_SCHEMA_MAX_TOKENS}, timeout=${GAME_SCHEMA_TIMEOUT_S}s)"
    else
        echo "    Phase GameProfile:  off"
    fi
    if [ "${LLM_CRAFTER_ENABLED}" = "1" ]; then
        echo "    LLM Crafter:        ON  (≤${LLM_CRAFTER_K_MAX} parallel 35B calls /"
        echo "                              game / step, max_tokens="
        echo "                              ${LLM_CRAFTER_MAX_TOKENS}, timeout=${LLM_CRAFTER_TIMEOUT_S}s)"
    else
        echo "    LLM Crafter:        off"
    fi
    if [ "${LLM_HARNESS_VALIDATOR_ENABLED}" = "1" ]; then
        echo "    LLM Harness:        ON  (bootstrap<${LLM_HARNESS_BOOTSTRAP_STEPS} steps,"
        echo "                              hybrid post-validate, max_tokens="
        echo "                              ${LLM_HARNESS_MAX_TOKENS}, timeout=${LLM_HARNESS_TIMEOUT_S}s)"
    else
        echo "    LLM Harness:        off"
    fi
    echo "  Spec decode:      ${SPEC_MODEL} (${SPEC_TOKENS} tokens)"
else
    echo "  GPU mode:         LEGACY (external vLLM at port ${PORT})"
fi
echo ""
echo "  Curriculum schedule:"
for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r pnum game display <<< "${phase_def}"
    step_start=$(( (pnum - 1) * ITERS_PER_PHASE ))
    step_end=$(( pnum * ITERS_PER_PHASE - 1 ))
    anchor="${BASELINE_ANCHOR[${game}]:-no anchor}"
    echo "    Phase ${pnum}: ${display} (steps ${step_start}–${step_end}; baseline ${anchor})"
done
echo "══════════════════════════════════════════════════════════════"

echo ""
echo "[run_phase1] Ensuring LoRA adapters exist (from SFT cold-start)..."

if [ ! -d "${COLDSTART_DECISION}" ]; then
    echo "[run_phase1] ERROR: Cold-start decision adapters not found: ${COLDSTART_DECISION}"
    echo "[run_phase1]   Run scripts/run_sft_coldstart.sh first, or set COLDSTART_DIR"
    exit 1
fi
if [ ! -d "${COLDSTART_SKILLBANK}" ]; then
    echo "[run_phase1] ERROR: Cold-start skillbank adapters not found: ${COLDSTART_SKILLBANK}"
    echo "[run_phase1]   Run scripts/run_sft_coldstart.sh first, or set COLDSTART_DIR"
    exit 1
fi

# Pre-flight: confirm the 4 gymv slugs are wired in episode_runner.
# A missing slug would surface at runtime as a confusing
# "GAME_CONFIGS not found" error far inside the rollout loop, so
# fail loudly here.
python -c "
from trainer.coevolution.episode_runner import _lazy_imports, GYMV_TEMPORAL_GAMES_SET
from trainer.coevolution.config import GAME_MAX_STEPS
_lazy_imports()
# New Phase-1 roster (refreshed 2026-05-03 PM, data-driven from new
# Cold-start-out-gymv/latest 4-backbone teacher table).
required = {'gymv_thunder_force_iii', 'gymv_altered_beast',
            'gymv_columns', 'gymv_dynamite_headdy'}
missing_gymv = required - GYMV_TEMPORAL_GAMES_SET
if missing_gymv:
    raise SystemExit(
        f'[run_phase1] FATAL: gymv slugs not wired in episode_runner: {sorted(missing_gymv)}. '
        f'Run install/install_gymv.sh + install/gymv_temporal_patch/apply_patch.sh.'
    )
missing_registry = required - set(GAME_MAX_STEPS)
if missing_registry:
    raise SystemExit(
        f'[run_phase1] FATAL: gymv slugs not registered in trainer/coevolution/config.py:GAME_MAX_STEPS: '
        f'{sorted(missing_registry)}. Add the slug + max_steps and re-run.'
    )
print(f'[run_phase1] Gym-V slugs wired: {sorted(GYMV_TEMPORAL_GAMES_SET & required)}')
"

RESOLVED_RUN_DIR=$(python -c "
import os, sys
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
from pathlib import Path

from trainer.coevolution.config import CoEvolutionConfig, prepare_adapters

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

echo "[run_phase1] Run dir:      ${RUN_DIR}"
echo "[run_phase1] Adapter dir:  ${ADAPTER_DIR}"
echo "[run_phase1] Snapshot dir: ${SNAPSHOT_DIR}"

# Start the 35B judge BEFORE the curriculum loop so its KV cache
# allocation lands while no FSDP / vLLM-9B processes are competing
# for HBM yet.  In external mode this is a no-op; in off mode it's
# skipped entirely.
if [ "${JUDGE_MODE}" = "auto" ]; then
    if ! start_35b_judge; then
        echo "[run_phase1] FATAL: cannot start 35B judge.  Re-run with" \
             "JUDGE_MODE=external (point JUDGE_URL elsewhere) or" \
             "JUDGE_MODE=off (disable 35B routing — ablation only)."
        exit 1
    fi
elif [ "${JUDGE_MODE}" = "external" ]; then
    echo "[run_phase1] JUDGE_MODE=external → assuming 35B server at ${JUDGE_URL}"
    if command -v curl >/dev/null 2>&1; then
        if curl -fs -m 3 "${JUDGE_URL}/models" >/dev/null 2>&1; then
            echo "[run_phase1] 35B server is reachable."
        else
            echo "[run_phase1] WARNING: ${JUDGE_URL} not reachable — judge calls" \
                 "will fail at runtime.  Start the 35B server before running."
        fi
    fi
fi

# ======================================================================
# Helper: save a phase snapshot (LoRA + bank + checkpoint + metadata)
# ======================================================================
save_phase_snapshot() {
    local phase_num="$1"
    local game="$2"
    local display="$3"
    local step_end="$4"

    local snap_name
    snap_name=$(printf "phase_%02d_%s" "${phase_num}" "${game}")
    local snap_path="${SNAPSHOT_DIR}/${snap_name}"

    echo ""
    echo "[run_phase1] ── Saving phase snapshot: ${snap_name} ──"
    mkdir -p "${snap_path}"

    if [ -d "${RUN_DIR}/lora_adapters" ]; then
        echo "[run_phase1]   Copying LoRA adapters..."
        cp -r "${RUN_DIR}/lora_adapters" "${snap_path}/lora_adapters"
    fi

    if [ -d "${RUN_DIR}/skillbank" ]; then
        echo "[run_phase1]   Copying skill bank (rolling, shared across phases)..."
        cp -r "${RUN_DIR}/skillbank" "${snap_path}/skillbank"
    fi

    local latest_ckpt=""
    if [ -d "${RUN_DIR}/checkpoints" ]; then
        latest_ckpt=$(ls -d "${RUN_DIR}/checkpoints"/step_* 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "${latest_ckpt}" ]; then
            echo "[run_phase1]   Copying latest checkpoint: $(basename "${latest_ckpt}")"
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
    "episodes_per_step": ${EPISODES},
    "model": "${MODEL}",
    "timestamp": "$(date -Iseconds)",
    "run_dir": "${RUN_DIR}",
    "latest_checkpoint": "${latest_ckpt:-none}",
    "baseline_anchor": "${BASELINE_ANCHOR[${game}]:-n/a}"
}
METAEOF

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

    echo "[run_phase1]   Phase ${phase_num} snapshot saved to: ${snap_path}"
    if [ -n "${skill_summary}" ]; then
        echo "[run_phase1]   Skill bank sizes:${skill_summary}"
    fi
    echo "[run_phase1] ── Snapshot complete ──"
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
        --bank-mode "${BANK_MODE}"
    )

    # Mid-phase resume detection: if the run dir already has a
    # checkpoint, we MUST use --resume regardless of phase number.
    # This handles two flows:
    #   * RESUME_PHASE=N with N>=2 — phase N starts fresh on top of
    #     phase N-1's snapshot.  Launcher used to hard-code --resume
    #     here.
    #   * RESUME_PHASE=1 with the SFT-loaded run already partway
    #     through phase 1 (e.g. instrumentation was added mid-run).
    #     Without this branch the launcher would try to re-load SFT
    #     and overwrite the trained LoRA.
    has_existing_checkpoint=false
    if [ -d "${RUN_DIR}/checkpoints" ] \
        && ls "${RUN_DIR}/checkpoints"/step_* 1>/dev/null 2>&1; then
        has_existing_checkpoint=true
    fi

    if [ "${is_first_phase}" = "true" ] \
        && [ "${has_existing_checkpoint}" = "false" ]; then
        # First phase, fresh start: load the cold-start SFT adapters.
        # Subsequent phases inherit the rolling LoRA from --run-dir
        # (Option C carry-over).
        if [ -d "${COLDSTART_DECISION}" ]; then
            args+=(--load-decision-adapters "${COLDSTART_DECISION}")
        fi
        if [ -d "${COLDSTART_SKILLBANK}" ]; then
            args+=(--load-skillbank-adapters "${COLDSTART_SKILLBANK}")
        fi
    else
        # Either later phase OR phase 1 with existing checkpoint.
        args+=(--resume)
    fi

    # ── 35B control-plane flags ──────────────────────────────────────
    # See top-of-file env-var docs.  Without these the 35B endpoint
    # idles and the run becomes 9B-only (the bug that wasted the
    # 2026-05-03 21:15 launch).
    if [ "${CRAFTER_PROMOTION}" = "1" ]; then
        args+=(--crafter-promotion-enabled)
        args+=(--crafter-cycle-every-k-steps "${CRAFTER_CYCLE_K}")
        # Pass the gate mode through to the Promotion driver. We only
        # forward the flag when it deviates from the trainer's
        # ``offline-synthetic`` default, both to keep the launched
        # command line self-documenting and to avoid touching legacy
        # ablation scripts that may still depend on the silent default.
        if [ "${PROMOTION_GATE_MODE}" != "offline-synthetic" ]; then
            args+=(--crafter-promotion-gate-mode "${PROMOTION_GATE_MODE}")
        fi
    fi
    if [ "${HARNESS_ENABLED}" = "1" ]; then
        args+=(--harness-enabled)
    fi

    # Path 1: phase-start GameProfile (1 × 35B / game / phase).
    if [ "${GAME_SCHEMA_ENABLED}" = "1" ]; then
        args+=(--game-schema-enabled)
        if [ "${GAME_SCHEMA_MAX_TOKENS}" != "1024" ]; then
            args+=(--game-schema-max-tokens "${GAME_SCHEMA_MAX_TOKENS}")
        fi
        if [ "${GAME_SCHEMA_TIMEOUT_S}" != "60" ]; then
            args+=(--game-schema-timeout-s "${GAME_SCHEMA_TIMEOUT_S}")
        fi
    fi

    # Path 2: supplemental LLM Crafter (≤K parallel 35B / game / step).
    if [ "${LLM_CRAFTER_ENABLED}" = "1" ]; then
        args+=(--llm-crafter-enabled)
        if [ "${LLM_CRAFTER_K_MAX}" != "2" ]; then
            args+=(--llm-crafter-k-max "${LLM_CRAFTER_K_MAX}")
        fi
        if [ "${LLM_CRAFTER_MAX_TOKENS}" != "1024" ]; then
            args+=(--llm-crafter-max-tokens "${LLM_CRAFTER_MAX_TOKENS}")
        fi
        if [ "${LLM_CRAFTER_TIMEOUT_S}" != "60" ]; then
            args+=(--llm-crafter-timeout-s "${LLM_CRAFTER_TIMEOUT_S}")
        fi
    fi

    # Path 4: LLM Harness validator (post-LLM 35B veto, hybrid).
    if [ "${LLM_HARNESS_VALIDATOR_ENABLED}" = "1" ]; then
        args+=(--llm-harness-validator-enabled)
        if [ "${LLM_HARNESS_BOOTSTRAP_STEPS}" != "20" ]; then
            args+=(--llm-harness-bootstrap-steps "${LLM_HARNESS_BOOTSTRAP_STEPS}")
        fi
        if [ "${LLM_HARNESS_MAX_TOKENS}" != "256" ]; then
            args+=(--llm-harness-max-tokens "${LLM_HARNESS_MAX_TOKENS}")
        fi
        if [ "${LLM_HARNESS_TIMEOUT_S}" != "30" ]; then
            args+=(--llm-harness-timeout-s "${LLM_HARNESS_TIMEOUT_S}")
        fi
    fi

    if [ -n "${DEBUG}" ]; then
        args+=(--debug-io)
    fi

    echo "${args[@]}"
}

# ======================================================================
# Helper: invoke the cross-game skill translator at a phase boundary
# (shared-bank mode only — no-op when BANK_MODE=per_game).
#
# Reads the current shared bank ``${RUN_DIR}/skillbank/skill_bank.jsonl``,
# rewrites every skill onto the next game's action vocabulary via the
# 35B judge, and writes the union back so the next phase's harness
# admits both source-grounded *and* target-grounded variants.
#
# Failure here is non-fatal: a flaky judge call or unrecognised target
# game falls through to "no translated skills added", which leaves the
# shared bank unchanged.  The next phase still runs (it just behaves
# like Option C carry-over for skills it can't translate).
# ======================================================================
translate_bank_for_next_phase() {
    if [ "${BANK_MODE}" != "shared" ] || [ "${TRANSLATE_ON_BOUNDARY}" != "1" ]; then
        return 0
    fi

    local prev_game="$1"
    local next_game="$2"

    local shared_bank="${RUN_DIR}/skillbank/skill_bank.jsonl"
    if [ ! -f "${shared_bank}" ]; then
        echo "[run_phase1] translate_bank: no shared bank at ${shared_bank} yet — skipping"
        return 0
    fi

    echo ""
    echo "[run_phase1] ── Cross-game skill translation: ${prev_game} → ${next_game} ──"

    local target_actions
    target_actions=$(python -c "
import os, json
os.environ.setdefault('PYGLET_HEADLESS', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
target = '${next_game}'
acts = []
try:
    if target.startswith('gymv_'):
        from env_wrappers.gymv_temporal_nl_wrapper import make_gymv_temporal_env
        env = make_gymv_temporal_env(target)
        env.reset()
        acts = list(env.action_names)
        env.close()
    else:
        # env_wrappers / GYM-API non-gymv games (candy_crush, tetris,
        # 2048, super_mario): import the game-config registry to
        # extract the same action vocabulary the wrapper would surface
        # at runtime.
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
        echo "[run_phase1] translate_bank: could not resolve actions for ${next_game} — skipping translation"
        return 0
    fi

    local translated_bank="${RUN_DIR}/skillbank/skill_bank.translated_to_${next_game}.jsonl"
    echo "[run_phase1] translate_bank: target_actions=${target_actions}"
    echo "[run_phase1] translate_bank: source=${shared_bank}"
    echo "[run_phase1] translate_bank: writing → ${translated_bank}"

    if python -m skill_agents.skill_bank.translate_for_target \
        --source-bank "${shared_bank}" \
        --target-game "${next_game}" \
        --target-actions "${target_actions}" \
        --source-game "${prev_game}" \
        --output "${translated_bank}" \
        --judge-model "${JUDGE_MODEL}" \
        -v
    then
        # Atomic swap: replace the shared bank with the translated
        # union (which seeds source skills + adds target derivatives).
        # The translator's --seed-with-source flag (default ON) ensures
        # we never *lose* source skills here.
        mv -f "${translated_bank}" "${shared_bank}"
        echo "[run_phase1] translate_bank: shared bank updated with ${next_game} derivatives"
    else
        echo "[run_phase1] translate_bank: WARNING — translation failed; keeping prior shared bank"
        rm -f "${translated_bank}" 2>/dev/null || true
    fi
}

# ======================================================================
# Curriculum training loop
# ======================================================================
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Starting Phase-1 curriculum (${NUM_PHASES} phases, sequential)"
echo "══════════════════════════════════════════════════════════════"

PHASE_FAILED=""
PREV_GAME=""

for phase_def in "${PHASES[@]}"; do
    IFS=':' read -r phase_num game display <<< "${phase_def}"

    if [ "${phase_num}" -lt "${RESUME_PHASE}" ]; then
        echo ""
        echo "[run_phase1] Skipping phase ${phase_num} (${display}) — resuming from phase ${RESUME_PHASE}"
        # Track the most recent skipped game so the *first executed*
        # phase still gets a translation pass against it (when the user
        # resumes mid-curriculum in shared-bank mode).
        PREV_GAME="${game}"
        continue
    fi

    # Cross-game skill translation: re-ground prior shared bank onto the
    # upcoming phase's action vocabulary (no-op in per_game mode).
    if [ -n "${PREV_GAME}" ] && [ "${PREV_GAME}" != "${game}" ]; then
        translate_bank_for_next_phase "${PREV_GAME}" "${game}" || true
    fi

    step_start=$(( (phase_num - 1) * ITERS_PER_PHASE ))
    step_end=$(( phase_num * ITERS_PER_PHASE ))
    is_first=$([ "${phase_num}" -eq "${RESUME_PHASE}" ] && [ "${RESUME_PHASE}" -eq 1 ] && echo "true" || echo "false")

    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  Phase ${phase_num}/${NUM_PHASES}: ${display}"
    echo "│  Slug:     ${game}"
    echo "│  Steps:    ${step_start} → ${step_end} (${ITERS_PER_PHASE} iterations)"
    echo "│  Episodes: ${EPISODES} rollouts per step"
    echo "│  Mode:     $([ "${is_first}" = "true" ] && echo "COLD-START" || echo "RESUME (carry-over LoRA + bank)")"
    echo "│  Anchor:   ${BASELINE_ANCHOR[${game}]:-n/a}"
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

    echo "[run_phase1] Training args: ${PHASE_ARGS[*]}"
    echo ""

    if python scripts/run_coevolution.py "${PHASE_ARGS[@]}"; then
        echo ""
        echo "[run_phase1] Phase ${phase_num} (${display}) completed successfully."
        save_phase_snapshot "${phase_num}" "${game}" "${display}" "${step_end}"
        PREV_GAME="${game}"
    else
        PHASE_FAILED="${phase_num}"
        echo ""
        echo "[run_phase1] ERROR: Phase ${phase_num} (${display}) FAILED."
        echo "[run_phase1] Saving partial snapshot before aborting..."
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
    echo "  Phase-1 curriculum STOPPED at phase ${PHASE_FAILED}"
    echo "  Resume with: RESUME_PHASE=${PHASE_FAILED} RUN_DIR=${RUN_DIR} bash scripts/run_phase1_curriculum.sh"
else
    echo "  Phase-1 curriculum COMPLETE"
    echo "  All ${NUM_PHASES} phases finished successfully."
    echo ""
    echo "  Next: bash scripts/run_phase2_holdout.sh \\"
    echo "          PHASE1_SNAPSHOT=${SNAPSHOT_DIR}/phase_06_tetris"
fi
echo ""
echo "  Run dir:    ${RUN_DIR}"
echo "  Snapshots:  ${SNAPSHOT_DIR}/"
if [ -d "${SNAPSHOT_DIR}" ]; then
    echo ""
    echo "  Phase snapshots (sorted by phase order):"
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
echo "[run_phase1] Phase-1 curriculum complete. Post-curriculum state:"
echo "    LoRA:  ${SNAPSHOT_DIR}/phase_06_tetris/lora_adapters/"
echo "    Bank:  ${SNAPSHOT_DIR}/phase_06_tetris/skillbank/"
