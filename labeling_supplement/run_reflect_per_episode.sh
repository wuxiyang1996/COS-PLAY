#!/usr/bin/env bash
# Parallel dispatcher for labeling_supplement/reflect_per_episode_gpt54.py.
#
# For each (corpus, source) pair under --bank-run, fans out one Python
# worker that:
#   * loads the per-source skill bank as the seeded CANDIDATE state,
#   * walks every episode_*.json under --actions-run/<corpus>/<source>/,
#   * pairs each episode with its
#     skill_bank_run/<corpus>/<source>/per_episode_bank_management/episode_<i>/bank_management_io.json,
#   * builds an EpisodeReflection (failure traces + new candidates),
#   * calls SkillCrafterService.reflect_on_episode() — the LIVE
#     two-tier-trigger reactive pass — and writes the resulting
#     PatchProposals/RetireProposals/HypothesisProposals to disk.
#
# This is the offline mirror of the **per-episode reactive pass** in
# the live runtime (see crafter/__init__.py and
# implementation_notes/legacy/crafter-harness-orchestrator-roles.md). It does
# NOT run gates and does NOT mutate the bank: those are the Harness's
# and Orchestrator's jobs respectively.
#
# Usage (defaults pick the latest bank + actions runs):
#
#   bash labeling_supplement/run_reflect_per_episode.sh
#
#   bash labeling_supplement/run_reflect_per_episode.sh \
#        --bank-run     labeling/skill_bank_out/run_<ts> \
#        --actions-run  labeling/skill_actions_out/run_<ts> \
#        --output-dir   labeling_supplement/episode_reflections_out/run_<my> \
#        --parallel 6
#
#   # Smoke test: one source pair, two episodes each.
#   bash labeling_supplement/run_reflect_per_episode.sh \
#        --sources twenty_forty_eight Temporal_Airstriker-v0 \
#        --max-episodes 2 --parallel 2
#
#   # Tune failure synthesis thresholds.
#   bash labeling_supplement/run_reflect_per_episode.sh \
#        --low-applicability 0.5 \
#        --max-failures-per-episode 16

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BANK_RUN="${REPO_ROOT}/labeling/skill_bank_out/run_20260430_030637"
ACTIONS_RUN="${REPO_ROOT}/labeling/skill_actions_out/run_20260430_064325"
OUTPUT_DIR="${REPO_ROOT}/labeling_supplement/episode_reflections_out/run_$(date '+%Y%m%d_%H%M%S')"
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
        --bank-run|--bank_run)         BANK_RUN="$2"; shift 2 ;;
        --actions-run|--actions_run)   ACTIONS_RUN="$2"; shift 2 ;;
        --output-dir|--output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --parallel)                    PARALLEL="$2"; shift 2 ;;
        --corpus)                      CORPUS="$2"; shift 2 ;;
        --sources)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SOURCES+=("$1")
                shift
            done
            ;;
        --smoke)                       SMOKE="1"; shift ;;
        --dry-run|--dry_run)           DRY_RUN="1"; shift ;;
        -v|--verbose)                  VERBOSE="-v"; shift ;;
        --max-episodes | \
        --low-applicability | \
        --max-failures-per-episode)
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help)
            head -n 41 "$0" | tail -n 39
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2 ;;
    esac
done

if [[ -n "$SMOKE" ]]; then
    if [[ ${#SOURCES[@]} -eq 0 ]]; then
        SOURCES=(twenty_forty_eight Temporal_Airstriker-v0)
    fi
    PARALLEL="${PARALLEL:-2}"
    EXTRA_ARGS+=(--max-episodes 2)
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
    echo "  Skill Crafter (per-episode reactive) — labeling_supplement mirror"
    echo "  bank_run     : $BANK_RUN"
    echo "  actions_run  : $ACTIONS_RUN"
    echo "  output_dir   : $OUTPUT_DIR"
    echo "  parallel     : $PARALLEL"
    echo "  pairs (${#PAIRS[@]}):"
    for p in "${PAIRS[@]}"; do
        echo "    - $p"
    done
    echo "  extra_args   : ${EXTRA_ARGS[*]:-<defaults>}"
    echo "  log_dir      : $LOG_DIR"
    echo "  started      : $(date -Iseconds)"
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

    {
        echo "[worker] $(date -Iseconds) starting $corpus / $src"
        python "${SCRIPT_DIR}/reflect_per_episode_gpt54.py" \
            --bank-run     "$BANK_RUN" \
            --actions-run  "$ACTIONS_RUN" \
            --output-dir   "$OUTPUT_DIR" \
            --corpus       "$corpus" \
            --source       "$src" \
            "${extra[@]}"
        rc=$?
        echo "[worker] $(date -Iseconds) finished $corpus / $src rc=$rc"
        return $rc
    } > "$log" 2>&1
}

export -f run_one
export BANK_RUN ACTIONS_RUN OUTPUT_DIR SCRIPT_DIR LOG_DIR VERBOSE

# ---------------------------------------------------------------------------
# Schedule pool — same pattern as run_decide_skill_crafting.sh
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
n_episodes  = sum(int(r.get("n_episodes", 0))  for r in results)
n_proposals = sum(int(r.get("n_proposals", 0)) for r in results)
n_subsumes  = sum(int(r.get("n_subsumption_retires", 0)) for r in results)
n_coalesced = sum(int(r.get("n_patches_coalesced", 0)) for r in results)
n_cooldown  = sum(int(r.get("n_patches_skipped_cooldown", 0)) for r in results)
by_kind     = Counter()
by_proposer = Counter()
for r in results:
    by_kind.update(r.get("by_kind") or {})
    by_proposer.update(r.get("by_proposer") or {})

summary = {
    "bank_run":                  "${BANK_RUN}",
    "actions_run":               "${ACTIONS_RUN}",
    "output_root":               "${OUTPUT_DIR}",
    "n_pairs":                   n_pairs,
    "n_pairs_ok":                n_ok,
    "n_episodes":                n_episodes,
    "n_proposals":               n_proposals,
    "n_subsumption_retires":     n_subsumes,
    "n_patches_coalesced":       n_coalesced,
    "n_patches_skipped_cooldown":n_cooldown,
    "by_kind":                   dict(by_kind),
    "by_proposer":               dict(by_proposer),
    "per_pair":                  results,
    "completed_at":              datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    "elapsed_sec":               ${ELAPSED},
}
out = OUT / "_run_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\\n[dispatcher] run summary -> {out}")
print(f"[dispatcher] {n_ok}/{n_pairs} pairs ok, {n_episodes} ep(s), "
      f"{n_proposals} proposal(s), subsumes={n_subsumes}, "
      f"coalesced={n_coalesced}, cooldown={n_cooldown}, "
      f"by_kind={dict(by_kind)}, elapsed=${ELAPSED}s")
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
