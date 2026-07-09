#!/usr/bin/env bash
# Parallel dispatcher for labeling_supplement/dump_harness_io_gpt54.py.
#
# For each (corpus, source) pair under --bank-run, fans out one Python
# worker that:
#   * loads the per-source skill bank into a fresh, temp four-store
#     SkillRepository (skills promoted to PROVISIONAL by default so the
#     harness eligibility filter — which gates on
#     status ∈ {ACTIVE, SHADOW, PROVISIONAL} — actually returns them),
#   * builds an AdapterRegistry with all five domain adapters
#     registered, then a SkillHarness + GateService around it,
#   * dumps the I/O of:
#       online surface  — every step of every actor episode, through
#                         harness.select_eligible_skills /
#                         harness.validate_invocation (stub, see
#                         harness/README.md §9.1) /
#                         harness.run_skill (opt-in)
#       offline surface — every typed BankMutationProposal, through
#                         GateService.evaluate(...) with replay seeds /
#                         shadow log / non-regression scalars
#                         synthesised from the cold-start corpus.
#
# This is the offline mirror of the **harness validation surfaces** in
# the live runtime. It does NOT mutate the bank or write SkillStatus
# (those are PromotionOrchestrator's jobs). Read alongside
# implementation_notes/legacy/crafter-harness-orchestrator-roles.md and
# harness/README.md.
#
# Usage (defaults pick the latest bank + actions runs):
#
#   bash labeling_supplement/run_dump_harness_io.sh
#
#   bash labeling_supplement/run_dump_harness_io.sh \
#        --bank-run               labeling/skill_bank_out/run_<ts> \
#        --actions-run            labeling/skill_actions_out/run_<ts> \
#        --crafter-proposals-run  labeling_supplement/crafter_proposals_out/run_<ts> \
#        --reflections-run        labeling_supplement/episode_reflections_out/run_<ts> \
#        --output-dir             labeling_supplement/harness_io_out/run_<my> \
#        --parallel 6
#
#   # Smoke test: one source pair, two episodes / five steps / three proposals.
#   bash labeling_supplement/run_dump_harness_io.sh \
#        --sources twenty_forty_eight \
#        --smoke
#
#   # Online only (no offline gate calls, no proposals required).
#   bash labeling_supplement/run_dump_harness_io.sh \
#        --surface online --max-episodes 3 --max-steps 10
#
#   # Enable real harness.run_skill (opt-in: expensive on stub adapters
#   # and currently meaningful only for the gymv adapter).
#   bash labeling_supplement/run_dump_harness_io.sh --run-skill --max-steps 5

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BANK_RUN="${REPO_ROOT}/labeling/skill_bank_out/run_20260430_030637"
ACTIONS_RUN="${REPO_ROOT}/labeling/skill_actions_out/run_20260430_064325"
CRAFTER_PROPOSALS_RUN=""
REFLECTIONS_RUN=""
OUTPUT_DIR="${REPO_ROOT}/labeling_supplement/harness_io_out/run_$(date '+%Y%m%d_%H%M%S')"
SURFACE="both"           # online | offline | both
PARALLEL="${PARALLEL:-6}"
CORPUS=""
SOURCES=()
SMOKE=""
DRY_RUN=""
VERBOSE=""

# Forwarded-verbatim flags.
declare -a EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bank-run|--bank_run)                     BANK_RUN="$2"; shift 2 ;;
        --actions-run|--actions_run)               ACTIONS_RUN="$2"; shift 2 ;;
        --crafter-proposals-run|--crafter_proposals_run)
                                                   CRAFTER_PROPOSALS_RUN="$2"; shift 2 ;;
        --reflections-run|--reflections_run)       REFLECTIONS_RUN="$2"; shift 2 ;;
        --output-dir|--output_dir)                 OUTPUT_DIR="$2"; shift 2 ;;
        --surface)                                 SURFACE="$2"; shift 2 ;;
        --parallel)                                PARALLEL="$2"; shift 2 ;;
        --corpus)                                  CORPUS="$2"; shift 2 ;;
        --sources)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SOURCES+=("$1")
                shift
            done
            ;;
        --smoke)                                   SMOKE="1"; shift ;;
        --dry-run|--dry_run)                       DRY_RUN="1"; shift ;;
        -v|--verbose)                              VERBOSE="-v"; shift ;;
        --run-skill)                               EXTRA_ARGS+=("$1"); shift ;;
        --no-force-runnable)                       EXTRA_ARGS+=("$1"); shift ;;
        --max-episodes | \
        --max-steps | \
        --max-proposals | \
        --max-replay-seeds | \
        --max-shadow-episodes)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help)
            head -n 60 "$0" | tail -n 58
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
    EXTRA_ARGS+=(--max-episodes 2 --max-steps 5 --max-proposals 3)
fi

if [[ "$SURFACE" != "online" && "$SURFACE" != "offline" && "$SURFACE" != "both" ]]; then
    echo "ERROR: --surface must be one of online|offline|both (got: $SURFACE)" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/_dispatch_logs"
mkdir -p "$LOG_DIR"

if [[ ! -d "$BANK_RUN" ]]; then
    echo "ERROR: bank-run does not exist: $BANK_RUN" >&2
    exit 2
fi
if [[ ! -d "$ACTIONS_RUN" ]]; then
    echo "ERROR: actions-run does not exist: $ACTIONS_RUN" >&2
    exit 2
fi
if [[ -n "$CRAFTER_PROPOSALS_RUN" && ! -d "$CRAFTER_PROPOSALS_RUN" ]]; then
    echo "ERROR: crafter-proposals-run does not exist: $CRAFTER_PROPOSALS_RUN" >&2
    exit 2
fi
if [[ -n "$REFLECTIONS_RUN" && ! -d "$REFLECTIONS_RUN" ]]; then
    echo "ERROR: reflections-run does not exist: $REFLECTIONS_RUN" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Discover (corpus, source) pairs from the bank-run layout
# ---------------------------------------------------------------------------
PAIRS=()
for corpus in gym_v env_wrappers; do
    if [[ -n "$CORPUS" && "$corpus" != "$CORPUS" ]]; then
        continue
    fi
    cdir="${BANK_RUN}/${corpus}"
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
        if [[ -f "${sdir}/skill_bank.jsonl" ]]; then
            adir="${ACTIONS_RUN}/${corpus}/${src}"
            if [[ -d "$adir" ]] && compgen -G "${adir}/episode_*.json" > /dev/null; then
                PAIRS+=("${corpus}|${src}")
            fi
        fi
    done
done

if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "ERROR: no (corpus, source) pairs discovered" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
{
    echo "=================================================================="
    echo "  Harness I/O dump — labeling_supplement validation surface"
    echo "  bank_run                : $BANK_RUN"
    echo "  actions_run             : $ACTIONS_RUN"
    echo "  crafter_proposals_run   : ${CRAFTER_PROPOSALS_RUN:-<unset>}"
    echo "  reflections_run         : ${REFLECTIONS_RUN:-<unset>}"
    echo "  output_dir              : $OUTPUT_DIR"
    echo "  surface                 : $SURFACE"
    echo "  parallel                : $PARALLEL"
    echo "  pairs (${#PAIRS[@]}):"
    for p in "${PAIRS[@]}"; do
        echo "    - $p"
    done
    echo "  extra_args              : ${EXTRA_ARGS[*]:-<defaults>}"
    echo "  log_dir                 : $LOG_DIR"
    echo "  started                 : $(date -Iseconds)"
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
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        extra+=("${EXTRA_ARGS[@]}")
    fi
    if [[ -n "$VERBOSE" ]]; then
        extra+=("$VERBOSE")
    fi
    if [[ -n "$CRAFTER_PROPOSALS_RUN" ]]; then
        extra+=(--crafter-proposals-run "$CRAFTER_PROPOSALS_RUN")
    fi
    if [[ -n "$REFLECTIONS_RUN" ]]; then
        extra+=(--reflections-run "$REFLECTIONS_RUN")
    fi

    {
        echo "[worker] $(date -Iseconds) starting $corpus / $src"
        python "${SCRIPT_DIR}/dump_harness_io_gpt54.py" \
            --bank-run     "$BANK_RUN" \
            --actions-run  "$ACTIONS_RUN" \
            --output-dir   "$OUTPUT_DIR" \
            --surface      "$SURFACE" \
            --corpus       "$corpus" \
            --source       "$src" \
            "${extra[@]}"
        rc=$?
        echo "[worker] $(date -Iseconds) finished $corpus / $src rc=$rc"
        return $rc
    } > "$log" 2>&1
}

export -f run_one
export BANK_RUN ACTIONS_RUN CRAFTER_PROPOSALS_RUN REFLECTIONS_RUN \
       OUTPUT_DIR SCRIPT_DIR LOG_DIR SURFACE VERBOSE

# ---------------------------------------------------------------------------
# Schedule pool — same pattern as run_reflect_per_episode.sh
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
# Aggregate run summary by re-reading per-source _source_summary.json files
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
        s = src_dir / "_source_summary.json"
        if s.exists():
            try:
                results.append(json.load(open(s)))
            except Exception as e:
                print(f"WARN: failed to read {s}: {e}", file=sys.stderr)

n_pairs = len(results)
n_ok    = sum(1 for r in results if r.get("status") == "ok")

# Online roll-up
n_online_pairs   = sum(1 for r in results if r.get("online"))
n_episodes       = sum(int(((r.get("online") or {}).get("n_episodes") or 0)) for r in results)
n_steps_dumped   = sum(int(((r.get("online") or {}).get("n_steps_dumped") or 0)) for r in results)
agree_total: Counter = Counter()
elig_hist : Counter = Counter()
for r in results:
    on = r.get("online") or {}
    agree_total.update(on.get("agreement_histogram") or {})
    for k, v in (on.get("n_eligible_histogram") or {}).items():
        elig_hist[k] += v

# Offline roll-up
n_offline_pairs  = sum(1 for r in results if r.get("offline"))
n_proposals      = sum(int(((r.get("offline") or {}).get("n_proposals") or 0)) for r in results)
n_skill_missing  = sum(int(((r.get("offline") or {}).get("n_skill_missing") or 0)) for r in results)
n_gate_error     = sum(int(((r.get("offline") or {}).get("n_gate_error") or 0)) for r in results)
final_hist: Counter = Counter()
stage_hist: dict = {}
for r in results:
    off = r.get("offline") or {}
    final_hist.update(off.get("by_final_verdict") or {})
    for stage, hist in (off.get("per_stage_verdict_histogram") or {}).items():
        d = stage_hist.setdefault(stage, Counter())
        d.update(hist)

summary = {
    "bank_run":               "${BANK_RUN}",
    "actions_run":            "${ACTIONS_RUN}",
    "crafter_proposals_run":  "${CRAFTER_PROPOSALS_RUN}" or None,
    "reflections_run":        "${REFLECTIONS_RUN}" or None,
    "output_root":            "${OUTPUT_DIR}",
    "surface":                "${SURFACE}",
    "n_pairs":                n_pairs,
    "n_pairs_ok":             n_ok,
    "online": {
        "n_pairs":                  n_online_pairs,
        "n_episodes_dumped":        n_episodes,
        "n_steps_dumped":           n_steps_dumped,
        "agreement_histogram":      dict(agree_total),
        "n_eligible_histogram":     dict(elig_hist),
    },
    "offline": {
        "n_pairs":                       n_offline_pairs,
        "n_proposals":                   n_proposals,
        "n_skill_missing":               n_skill_missing,
        "n_gate_error":                  n_gate_error,
        "by_final_verdict":              dict(final_hist),
        "per_stage_verdict_histogram":   {k: dict(v) for k, v in stage_hist.items()},
    },
    "completed_at":           datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    "elapsed_sec":            ${ELAPSED},
    "per_pair":               results,
}
out = OUT / "_run_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\\n[dispatcher] run summary -> {out}")
print(f"[dispatcher] {n_ok}/{n_pairs} pairs ok | online: {n_episodes} ep, {n_steps_dumped} steps | "
      f"offline: {n_proposals} props ({n_skill_missing} missing skill, {n_gate_error} gate err) "
      f"final={dict(final_hist)} | elapsed=${ELAPSED}s")
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
