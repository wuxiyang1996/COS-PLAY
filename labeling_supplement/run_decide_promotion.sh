#!/usr/bin/env bash
# Parallel dispatcher for labeling_supplement/decide_promotion_gpt54.py.
#
# For each (corpus, source) pair under --proposals-run, fans out one
# Python worker that:
#   * loads the per-source skill bank from --bank-run and seeds it as
#     CANDIDATE through the LIVE SkillLifecycleManager,
#   * walks every proposal in <proposals-run>/<corpus>/<source>/proposals.jsonl,
#   * runs the LIVE GateService inline (or consumes external verdicts
#     via --gate-verdicts-run when the Harness mirror lands),
#   * calls the LIVE PromotionOrchestrator.promote()/.rollback() —
#     no parallel reimplementation; identical to the production loop,
#   * writes one promotion_decisions.jsonl + audit.jsonl + release.json
#     + bank_snapshots/<id>.json + gate_verdicts.jsonl + a
#     defer_followups.jsonl back-edge per (corpus, source).
#
# Spec: implementation_notes/legacy/crafter-harness-orchestrator-roles.md §6.2.
# It does NOT call the Crafter and does NOT touch the source --bank-run
# directory: those are the Crafter's job and the input snapshot
# respectively.
#
# Usage (defaults pick the latest crafter + bank + actions runs):
#
#   bash labeling_supplement/run_decide_promotion.sh
#
#   bash labeling_supplement/run_decide_promotion.sh \
#        --proposals-run labeling_supplement/crafter_proposals_out/run_<ts> \
#        --bank-run      labeling/skill_bank_out/run_<ts> \
#        --actions-run   labeling/skill_actions_out/run_<ts> \
#        --output-dir    labeling_supplement/promotion_decisions_out/run_<my> \
#        --parallel 6
#
#   # Smoke test: one source pair, no rollback signal.
#   bash labeling_supplement/run_decide_promotion.sh \
#        --sources twenty_forty_eight \
#        --no-actions --parallel 1
#
#   # Tune rollback thresholds.
#   bash labeling_supplement/run_decide_promotion.sh \
#        --rollback-min-selections 5 \
#        --rollback-min-pass-rate 0.7

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROPOSALS_RUN="${REPO_ROOT}/labeling_supplement/crafter_proposals_out/run_20260430_073444"
BANK_RUN="${REPO_ROOT}/labeling/skill_bank_out/run_20260430_030637"
ACTIONS_RUN="${REPO_ROOT}/labeling/skill_actions_out/run_20260430_064325"
GATE_VERDICTS_RUN=""
OUTPUT_DIR="${REPO_ROOT}/labeling_supplement/promotion_decisions_out/run_$(date '+%Y%m%d_%H%M%S')"
PARALLEL="${PARALLEL:-6}"
NO_ACTIONS=""
CORPUS=""
SOURCES=()
SMOKE=""
DRY_RUN=""
VERBOSE=""

declare -a EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --proposals-run|--proposals_run) PROPOSALS_RUN="$2"; shift 2 ;;
        --bank-run|--bank_run)           BANK_RUN="$2"; shift 2 ;;
        --actions-run|--actions_run)     ACTIONS_RUN="$2"; shift 2 ;;
        --gate-verdicts-run|--gate_verdicts_run)
            GATE_VERDICTS_RUN="$2"; shift 2 ;;
        --output-dir|--output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --parallel)                      PARALLEL="$2"; shift 2 ;;
        --no-actions|--no_actions)       NO_ACTIONS="1"; shift ;;
        --corpus)                        CORPUS="$2"; shift 2 ;;
        --sources)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SOURCES+=("$1")
                shift
            done
            ;;
        --smoke)                         SMOKE="1"; shift ;;
        --dry-run|--dry_run)             DRY_RUN="1"; shift ;;
        -v|--verbose)                    VERBOSE="-v"; shift ;;
        --teacher-model | \
        --judge-model | \
        --rollback-min-selections | \
        --rollback-min-pass-rate)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help)
            head -n 48 "$0" | tail -n 46
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2 ;;
    esac
done

if [[ -n "$SMOKE" ]]; then
    if [[ ${#SOURCES[@]} -eq 0 ]]; then
        SOURCES=(twenty_forty_eight)
    fi
    PARALLEL="${PARALLEL:-1}"
    NO_ACTIONS="${NO_ACTIONS:-1}"
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/_dispatch_logs"
mkdir -p "$LOG_DIR"

if [[ ! -d "$PROPOSALS_RUN" ]]; then
    echo "ERROR: proposals-run does not exist: $PROPOSALS_RUN" >&2
    exit 2
fi
if [[ ! -d "$BANK_RUN" ]]; then
    echo "ERROR: bank-run does not exist: $BANK_RUN" >&2
    exit 2
fi
if [[ -z "$NO_ACTIONS" && ! -d "$ACTIONS_RUN" ]]; then
    echo "WARN:  actions-run does not exist; continuing without rollback signal: $ACTIONS_RUN" >&2
    NO_ACTIONS="1"
fi
if [[ -n "$GATE_VERDICTS_RUN" && ! -d "$GATE_VERDICTS_RUN" ]]; then
    echo "WARN:  gate-verdicts-run does not exist; falling back to inline GateService: $GATE_VERDICTS_RUN" >&2
    GATE_VERDICTS_RUN=""
fi

# ---------------------------------------------------------------------------
# Discover (corpus, source) pairs from the proposals-run layout
# ---------------------------------------------------------------------------
PAIRS=()
for corpus in gym_v env_wrappers; do
    if [[ -n "$CORPUS" && "$corpus" != "$CORPUS" ]]; then
        continue
    fi
    cdir="${PROPOSALS_RUN}/${corpus}"
    [[ -d "$cdir" ]] || continue
    for sdir in "${cdir}"/*; do
        [[ -d "$sdir" ]] || continue
        src="$(basename "$sdir")"
        [[ "$src" == _* ]] && continue
        if [[ ${#SOURCES[@]} -gt 0 ]]; then
            match=0
            for s in "${SOURCES[@]}"; do
                if [[ "$s" == "$src" ]]; then match=1; break; fi
            done
            [[ $match -eq 1 ]] || continue
        fi
        if [[ -f "${sdir}/proposals.jsonl" \
              && -f "${BANK_RUN}/${corpus}/${src}/skill_bank.jsonl" ]]; then
            PAIRS+=("${corpus}|${src}")
        fi
    done
done

if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "ERROR: no (corpus, source) pairs discovered under $PROPOSALS_RUN" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
{
    echo "=================================================================="
    echo "  Pipeline Orchestrator (offline) — labeling_supplement mirror"
    echo "  proposals_run     : $PROPOSALS_RUN"
    echo "  bank_run          : $BANK_RUN"
    echo "  actions_run       : ${NO_ACTIONS:+<disabled>}${NO_ACTIONS:-$ACTIONS_RUN}"
    echo "  gate_verdicts_run : ${GATE_VERDICTS_RUN:-<inline GateService>}"
    echo "  output_dir        : $OUTPUT_DIR"
    echo "  parallel          : $PARALLEL"
    echo "  pairs (${#PAIRS[@]}):"
    for p in "${PAIRS[@]}"; do
        echo "    - $p"
    done
    echo "  extra_args        : ${EXTRA_ARGS[*]:-<defaults>}"
    echo "  log_dir           : $LOG_DIR"
    echo "  started           : $(date -Iseconds)"
    echo "=================================================================="
} | tee "${LOG_DIR}/_dispatch.log"

if [[ -n "$DRY_RUN" ]]; then
    exit 0
fi

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
export PYTHONPATH="${WORKSPACE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

run_one() {
    local pair="$1"
    local corpus="${pair%|*}"
    local src="${pair#*|}"
    local log="${LOG_DIR}/${corpus}__${src}.log"

    local extra=()
    if [[ -n "$NO_ACTIONS" ]]; then
        extra+=(--no-actions)
    else
        extra+=(--actions-run "$ACTIONS_RUN")
    fi
    if [[ -n "$GATE_VERDICTS_RUN" ]]; then
        extra+=(--gate-verdicts-run "$GATE_VERDICTS_RUN")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        extra+=("${EXTRA_ARGS[@]}")
    fi
    if [[ -n "$VERBOSE" ]]; then
        extra+=("$VERBOSE")
    fi

    {
        echo "[worker] $(date -Iseconds) starting $corpus / $src"
        python "${SCRIPT_DIR}/decide_promotion_gpt54.py" \
            --proposals-run "$PROPOSALS_RUN" \
            --bank-run      "$BANK_RUN" \
            --output-dir    "$OUTPUT_DIR" \
            --corpus        "$corpus" \
            --source        "$src" \
            "${extra[@]}"
        rc=$?
        echo "[worker] $(date -Iseconds) finished $corpus / $src rc=$rc"
        return $rc
    } > "$log" 2>&1
}

export -f run_one
export PROPOSALS_RUN BANK_RUN ACTIONS_RUN GATE_VERDICTS_RUN OUTPUT_DIR \
       NO_ACTIONS SCRIPT_DIR LOG_DIR VERBOSE

# ---------------------------------------------------------------------------
# Schedule pool — same pattern as run_decide_skill_crafting.sh +
# run_reflect_per_episode.sh
# ---------------------------------------------------------------------------
T0=$(date +%s)
PIDS=()
RUNNING=0
declare -A PAIR_FOR_PID
declare -a RESULTS

for pair in "${PAIRS[@]}"; do
    while [[ $RUNNING -ge $PARALLEL ]]; do
        wait -n -p finished_pid
        rc=$?
        for i in "${!PIDS[@]}"; do
            if [[ "${PIDS[$i]}" == "$finished_pid" ]]; then
                completed="${PAIR_FOR_PID[$finished_pid]}"
                RESULTS+=("${completed}|${rc}")
                unset 'PIDS[i]'
                break
            fi
        done
        RUNNING=$((RUNNING - 1))
    done

    run_one "$pair" &
    pid=$!
    PIDS+=("$pid")
    PAIR_FOR_PID[$pid]="$pair"
    RUNNING=$((RUNNING + 1))
done

while [[ ${#PIDS[@]} -gt 0 ]]; do
    wait -n -p finished_pid
    rc=$?
    for i in "${!PIDS[@]}"; do
        if [[ "${PIDS[$i]}" == "$finished_pid" ]]; then
            completed="${PAIR_FOR_PID[$finished_pid]}"
            RESULTS+=("${completed}|${rc}")
            unset 'PIDS[i]'
            break
        fi
    done
done

T1=$(date +%s)
ELAPSED=$((T1 - T0))

# ---------------------------------------------------------------------------
# Aggregate run summary by re-reading per-source _promotion_summary.json
# ---------------------------------------------------------------------------
python - <<PY
import json, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

OUT = Path("${OUTPUT_DIR}")
results = []
for corpus_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name in ("gym_v", "env_wrappers")):
    for src_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        s = src_dir / "_promotion_summary.json"
        if s.exists():
            try:
                results.append(json.load(open(s)))
            except Exception as e:
                print(f"WARN: failed to read {s}: {e}", file=sys.stderr)

n_pairs = len(results)
n_ok    = sum(1 for r in results if r.get("status") == "ok")
n_props      = sum(int(r.get("n_proposals", 0))     for r in results)
n_decisions  = sum(int(r.get("n_decisions", 0))     for r in results)
n_audit_rows = sum(int(r.get("n_audit_rows", 0))    for r in results)
n_rollbacks  = sum(int(r.get("n_rollbacks", 0))     for r in results)
by_kind          = Counter()
by_verdict       = Counter()
by_decision      = Counter()
by_target_status = Counter()
for r in results:
    by_kind.update(r.get("by_kind") or {})
    by_verdict.update(r.get("by_verdict") or {})
    by_decision.update(r.get("by_decision") or {})
    by_target_status.update(r.get("by_target_status") or {})

summary = {
    "proposals_run":      "${PROPOSALS_RUN}",
    "bank_run":           "${BANK_RUN}",
    "actions_run":        None if "${NO_ACTIONS}" else "${ACTIONS_RUN}",
    "gate_verdicts_run":  "${GATE_VERDICTS_RUN}" or None,
    "output_root":        "${OUTPUT_DIR}",
    "n_pairs":            n_pairs,
    "n_pairs_ok":         n_ok,
    "n_proposals":        n_props,
    "n_decisions":        n_decisions,
    "n_audit_rows":       n_audit_rows,
    "n_rollbacks":        n_rollbacks,
    "by_kind":            dict(by_kind),
    "by_verdict":         dict(by_verdict),
    "by_decision":        dict(by_decision),
    "by_target_status":   dict(by_target_status),
    "per_pair":           results,
    "completed_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    "elapsed_sec":        ${ELAPSED},
}
out = OUT / "_run_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\\n[dispatcher] run summary -> {out}")
print(f"[dispatcher] {n_ok}/{n_pairs} pairs ok, {n_props} proposal(s), "
      f"{n_decisions} decision(s), rollbacks={n_rollbacks}, "
      f"by_decision={dict(by_decision)}, elapsed=${ELAPSED}s")
PY

# ---------------------------------------------------------------------------
# Per-pair return-code summary
# ---------------------------------------------------------------------------
echo
echo "================== per-pair status ==================" | tee -a "${LOG_DIR}/_dispatch.log"
ok=0
fail=0
for r in "${RESULTS[@]}"; do
    pair="${r%|*}"
    rc="${r##*|}"
    if [[ "$rc" == "0" ]]; then
        echo "  OK   $pair" | tee -a "${LOG_DIR}/_dispatch.log"
        ok=$((ok+1))
    else
        echo "  FAIL $pair (rc=$rc)" | tee -a "${LOG_DIR}/_dispatch.log"
        fail=$((fail+1))
    fi
done
echo "------------------------------------------------------" | tee -a "${LOG_DIR}/_dispatch.log"
echo "  ${ok} ok, ${fail} failed, ${#RESULTS[@]} total"      | tee -a "${LOG_DIR}/_dispatch.log"
echo "  output dir: $OUTPUT_DIR"                              | tee -a "${LOG_DIR}/_dispatch.log"
echo "  finished:   $(date -Iseconds)"                        | tee -a "${LOG_DIR}/_dispatch.log"
echo "  elapsed:    ${ELAPSED}s"                              | tee -a "${LOG_DIR}/_dispatch.log"

[[ $fail -eq 0 ]]
