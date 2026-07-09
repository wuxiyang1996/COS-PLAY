#!/usr/bin/env bash
# Parallel dispatcher for labeling/label_skill_actions_gpt54.py.
#
# For each (corpus, game) pair under --intentions-run, fans out one
# Python worker that:
#   * loads the per-game skill bank from --bank-run,
#   * iterates every step in every episode_*.json,
#   * queries the bank with (intentions, summary_state) -> top_k,
#   * writes the labelled episode (input + skills + skill_query) under
#     ${OUTPUT_DIR}/<corpus>/<game>/episode_*.json.
#
# This is **REPLAY-only**: original rollout actions are preserved
# verbatim; only the new skills + skill_query fields are added.
# No LLM calls per step, no harness wired in (see the script docstring
# in label_skill_actions_gpt54.py for the rationale).
#
# Usage (defaults pick the dual-axis intentions + the 030637 bank run):
#
#   bash labeling/run_label_skill_actions.sh
#
#   bash labeling/run_label_skill_actions.sh \
#        --intentions-run labeling/intentions_out/run_dualaxis_<ts> \
#        --bank-run       labeling/skill_bank_out/run_<ts> \
#        --output-dir     labeling/skill_actions_out/run_<my> \
#        --top-k 5 --parallel 8
#
#   # Smoke test: one gym-v env + one env_wrappers game, 1 ep each.
#   bash labeling/run_label_skill_actions.sh \
#        --games Temporal_Airstriker-v0 tetris \
#        --limit-episodes 1 --parallel 2

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
INTENTIONS_RUN="${REPO_ROOT}/labeling/intentions_out/run_dualaxis_20260429_224917"
BANK_RUN="${REPO_ROOT}/labeling/skill_bank_out/run_20260430_030637"
OUTPUT_DIR="${REPO_ROOT}/labeling/skill_actions_out/run_$(date '+%Y%m%d_%H%M%S')"
TOP_K="${TOP_K:-5}"
PARALLEL="${PARALLEL:-8}"
LIMIT_EPISODES=""
SMOKE=""
GAMES=()
CORPUS=""

# ---------------------------------------------------------------------------
# Arg parse
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --intentions-run|--intentions_run) INTENTIONS_RUN="$2"; shift 2 ;;
        --bank-run|--bank_run)             BANK_RUN="$2"; shift 2 ;;
        --output-dir|--output_dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --top-k|--top_k)                   TOP_K="$2"; shift 2 ;;
        --parallel)                        PARALLEL="$2"; shift 2 ;;
        --limit-episodes|--limit_episodes) LIMIT_EPISODES="$2"; shift 2 ;;
        --corpus)                          CORPUS="$2"; shift 2 ;;
        --smoke)                           SMOKE="1"; shift ;;
        --games)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                GAMES+=("$1")
                shift
            done
            ;;
        -h|--help)
            head -n 30 "$0" | tail -n 28
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -n "$SMOKE" ]]; then
    LIMIT_EPISODES="${LIMIT_EPISODES:-1}"
    if [[ ${#GAMES[@]} -eq 0 ]]; then
        GAMES=(Temporal_Airstriker-v0 tetris)
    fi
    PARALLEL="${PARALLEL:-2}"
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/_dispatch_logs"
mkdir -p "$LOG_DIR"

if [[ ! -d "$INTENTIONS_RUN" ]]; then
    echo "ERROR: intentions-run does not exist: $INTENTIONS_RUN" >&2
    exit 2
fi
if [[ ! -d "$BANK_RUN" ]]; then
    echo "ERROR: bank-run does not exist: $BANK_RUN" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Discover (corpus, game) pairs
# ---------------------------------------------------------------------------
PAIRS=()
for corpus in gym_v env_wrappers; do
    if [[ -n "$CORPUS" && "$corpus" != "$CORPUS" ]]; then
        continue
    fi
    cdir="${INTENTIONS_RUN}/${corpus}"
    [[ -d "$cdir" ]] || continue
    for gdir in "${cdir}"/*; do
        [[ -d "$gdir" ]] || continue
        game="$(basename "$gdir")"
        if [[ ${#GAMES[@]} -gt 0 ]]; then
            match=0
            for g in "${GAMES[@]}"; do
                if [[ "$g" == "$game" ]]; then match=1; break; fi
            done
            [[ $match -eq 1 ]] || continue
        fi
        # require at least one episode_*.json
        if compgen -G "${gdir}/episode_*.json" > /dev/null; then
            PAIRS+=("${corpus}|${game}")
        fi
    done
done

if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "ERROR: no (corpus, game) pairs discovered under $INTENTIONS_RUN" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
{
    echo "=================================================================="
    echo "  Skill-conditioned action labeling (replay-only)"
    echo "  intentions_run : $INTENTIONS_RUN"
    echo "  bank_run       : $BANK_RUN"
    echo "  output_dir     : $OUTPUT_DIR"
    echo "  top_k          : $TOP_K"
    echo "  parallel       : $PARALLEL"
    echo "  limit_episodes : ${LIMIT_EPISODES:-<all>}"
    echo "  pairs (${#PAIRS[@]}):"
    for p in "${PAIRS[@]}"; do
        echo "    - $p"
    done
    echo "  log dir        : $LOG_DIR"
    echo "  started        : $(date -Iseconds)"
    echo "=================================================================="
} | tee "${LOG_DIR}/_dispatch.log"

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
export PYTHONPATH="${WORKSPACE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

run_one() {
    local pair="$1"
    local corpus="${pair%|*}"
    local game="${pair#*|}"
    local log="${LOG_DIR}/${corpus}__${game}.log"

    local extra_args=()
    if [[ -n "$LIMIT_EPISODES" ]]; then
        extra_args+=(--limit-episodes "$LIMIT_EPISODES")
    fi

    {
        echo "[worker] $(date -Iseconds) starting $corpus / $game"
        python "${SCRIPT_DIR}/label_skill_actions_gpt54.py" \
            --intentions-run "$INTENTIONS_RUN" \
            --bank-run       "$BANK_RUN" \
            --output-dir     "$OUTPUT_DIR" \
            --top-k          "$TOP_K" \
            --corpus         "$corpus" \
            --game           "$game" \
            "${extra_args[@]}"
        rc=$?
        echo "[worker] $(date -Iseconds) finished $corpus / $game rc=$rc"
        return $rc
    } > "$log" 2>&1
}

export -f run_one
export INTENTIONS_RUN BANK_RUN OUTPUT_DIR TOP_K LIMIT_EPISODES SCRIPT_DIR LOG_DIR

# ---------------------------------------------------------------------------
# Schedule pool
# ---------------------------------------------------------------------------
T0=$(date +%s)
PIDS=()
RUNNING=0
declare -a RESULTS  # "corpus|game|rc"

for pair in "${PAIRS[@]}"; do
    while [[ $RUNNING -ge $PARALLEL ]]; do
        wait -n -p finished_pid
        rc=$?
        # find which pair completed
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

# Wait for the rest
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
# Aggregate run summary by re-invoking the labeler in --all mode with
# every game already done (it just rewrites _run_summary.json from the
# per-game _skill_actions_summary.json files).
# ---------------------------------------------------------------------------
python - <<PY
import json, sys
from pathlib import Path
from datetime import datetime

OUT = Path("${OUTPUT_DIR}")
results = []
for corpus_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name in ("gym_v", "env_wrappers")):
    for game_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        s = game_dir / "_skill_actions_summary.json"
        if s.exists():
            try:
                results.append(json.load(open(s)))
            except Exception as e:
                print(f"WARN: failed to read {s}: {e}", file=sys.stderr)

n_episodes = sum(r.get("n_episodes", 0) for r in results)
n_steps    = sum(r.get("n_steps", 0)    for r in results)
n_with     = sum(r.get("n_with_skill",0) for r in results)

by_corpus = {}
for r in results:
    c = r["corpus"]
    b = by_corpus.setdefault(c, {"games":0,"episodes":0,"steps":0,"with_skill":0,"distinct_skills":set()})
    b["games"]      += 1
    b["episodes"]   += r.get("n_episodes",0)
    b["steps"]      += r.get("n_steps",0)
    b["with_skill"] += r.get("n_with_skill",0)
    for sid in (r.get("selection_histogram") or {}).keys():
        b["distinct_skills"].add(sid)
for v in by_corpus.values():
    v["distinct_skills"] = sorted(v["distinct_skills"])
    v["n_distinct_skills"] = len(v["distinct_skills"])
    v["coverage"] = (v["with_skill"] / v["steps"]) if v["steps"] else 0.0

summary = {
    "intentions_run": "${INTENTIONS_RUN}",
    "bank_run":       "${BANK_RUN}",
    "output_root":    "${OUTPUT_DIR}",
    "top_k":          ${TOP_K},
    "n_pairs":        len(results),
    "n_pairs_ok":     sum(1 for r in results if r.get("status") == "ok"),
    "n_episodes":     n_episodes,
    "n_steps":        n_steps,
    "n_with_skill":   n_with,
    "coverage":       (n_with / n_steps) if n_steps else 0.0,
    "by_corpus":      by_corpus,
    "per_pair":       results,
    "completed_at":   datetime.utcnow().isoformat() + "Z",
    "elapsed_sec":    ${ELAPSED},
}
out = OUT / "_run_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\\n[dispatcher] run summary -> {out}")
print(f"[dispatcher] {summary['n_pairs_ok']}/{summary['n_pairs']} pairs ok, "
      f"{n_episodes} eps, {n_steps} steps, "
      f"coverage={summary['coverage']:.2%}, elapsed=${ELAPSED}s")
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
