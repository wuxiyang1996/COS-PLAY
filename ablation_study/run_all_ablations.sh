#!/usr/bin/env bash
# ======================================================================
#  Run all ablation experiments for a given game (or all games)
#
#  Usage:
#    bash ablation_study/run_all_ablations.sh --game diplomacy
#    bash ablation_study/run_all_ablations.sh --game avalon
#    bash ablation_study/run_all_ablations.sh --game super_mario
#    bash ablation_study/run_all_ablations.sh --game all
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GAME="all"
while [[ $# -gt 0 ]]; do
    case "$1" in --game) GAME="$2"; shift 2 ;; *) echo "Unknown: $1"; exit 1 ;; esac
done

FAILED=()
run() {
    local label="$1"; shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: ${label}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if bash "$@" 2>&1 | tee "${SCRIPT_DIR}/output/${label//[ \/]/_}.log"; then
        echo "  ✓ ${label} PASSED"
    else
        echo "  ✗ ${label} FAILED"
        FAILED+=("${label}")
    fi
}

mkdir -p "${SCRIPT_DIR}/output"

# ── Diplomacy ablations ──────────────────────────────────────────────
if [ "${GAME}" = "diplomacy" ] || [ "${GAME}" = "all" ]; then
    run "diplomacy base"             "${SCRIPT_DIR}/run_diplomacy_ablation.sh" --adapter base
    run "diplomacy sft+none"         "${SCRIPT_DIR}/run_diplomacy_ablation.sh" --adapter sft   --bank none
    run "diplomacy sft+first"        "${SCRIPT_DIR}/run_diplomacy_ablation.sh" --adapter sft   --bank first
    run "diplomacy sft+best"         "${SCRIPT_DIR}/run_diplomacy_ablation.sh" --adapter sft   --bank best
    run "diplomacy coevo+none"       "${SCRIPT_DIR}/run_diplomacy_ablation.sh" --adapter coevo --bank none
    run "diplomacy coevo+best"       "${SCRIPT_DIR}/run_diplomacy_ablation.sh" --adapter coevo --bank best
fi

# ── Avalon ablations ─────────────────────────────────────────────────
if [ "${GAME}" = "avalon" ] || [ "${GAME}" = "all" ]; then
    run "avalon base"                "${SCRIPT_DIR}/run_avalon_ablation.sh" --adapter base
    run "avalon sft+none"            "${SCRIPT_DIR}/run_avalon_ablation.sh" --adapter sft   --bank none
    run "avalon sft+first"           "${SCRIPT_DIR}/run_avalon_ablation.sh" --adapter sft   --bank first
    run "avalon sft+best"            "${SCRIPT_DIR}/run_avalon_ablation.sh" --adapter sft   --bank best
    run "avalon coevo+none"          "${SCRIPT_DIR}/run_avalon_ablation.sh" --adapter coevo --bank none
    run "avalon coevo+best"          "${SCRIPT_DIR}/run_avalon_ablation.sh" --adapter coevo --bank best
fi

# ── Super Mario ablations ────────────────────────────────────────────
if [ "${GAME}" = "super_mario" ] || [ "${GAME}" = "all" ]; then
    run "super_mario base"           "${SCRIPT_DIR}/run_super_mario_ablation.sh" --adapter base
    run "super_mario sft"            "${SCRIPT_DIR}/run_super_mario_ablation.sh" --adapter sft
fi

# ── Report ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All ablation experiments PASSED"
else
    echo "  ${#FAILED[@]} experiment(s) FAILED:"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
fi
echo "══════════════════════════════════════════════════════════════"

[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0
