#!/usr/bin/env bash
# scripts/run_offline_promotion_cycle.sh — fire the offline promotion
# loop ONCE to populate ``bank.runnable()`` (T1.2 — the §17 keystone
# for the harness wire).
#
# Spec sources:
#   * implementation_notes/pre-training-readiness-audit.md §0.1 (T1.2)
#   * plans/05-harness/PLAN-HARNESS.md §17    (the keystone)
#   * skill_bank/repository.py:51-57          (`runnable()` filter)
#   * labeling_supplement/decide_promotion_gpt54.py
#   * skill_bank/legacy_writeback.py
#
# Why this exists: the live trainer's actor will only see skills that
# pass ``SkillRepository.runnable()``, which filters to ACTIVE / SHADOW.
# Cold-start banks are minted as CANDIDATE — they need at least one
# offline promotion pass *before* the trainer can use them. This script
# drives that single pass end-to-end:
#
#   1. ``decide_promotion_gpt54.py`` (Phase-1 ``offline-synthetic``
#      gate) — emits ``bank_snapshots/snap-*.json`` per (corpus,source).
#   2. ``skill_bank.legacy_writeback.writeback_promotion`` — projects
#      the promoted skills back into each per-game ``skill_bank.jsonl``
#      so AsyncSkillBankPipeline picks them up via the legacy loader.
#
# The pipeline is LLM-free, GPU-free, and ``runs/``-free. It only
# touches the cold-start bank corpus and the per-pair skill banks.
#
# Usage (defaults pick the latest crafter + bank + actions runs):
#
#     bash scripts/run_offline_promotion_cycle.sh
#
#     # Smoke run on one source pair, no actor metrics.
#     bash scripts/run_offline_promotion_cycle.sh \
#         --sources twenty_forty_eight \
#         --no-actions --parallel 1
#
#     # Custom output / writeback target.
#     bash scripts/run_offline_promotion_cycle.sh \
#         --output-dir /tmp/promotion_run_42 \
#         --writeback-bank-run labeling/skill_bank_out/run_20260430_030637
#
#     # Dry-run the legacy_writeback step (skip the JSONL mutation).
#     bash scripts/run_offline_promotion_cycle.sh --dry-run-writeback

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults — pick the latest run we know about; override via CLI.
# ---------------------------------------------------------------------------
PROPOSALS_RUN="${REPO_ROOT}/labeling_supplement/crafter_proposals_out/run_20260430_073444"
BANK_RUN="${REPO_ROOT}/labeling/skill_bank_out/run_20260430_030637"
ACTIONS_RUN="${REPO_ROOT}/labeling/skill_actions_out/run_20260430_064325"
WRITEBACK_BANK_RUN=""                      # defaults to BANK_RUN (in place)
GATE_VERDICTS_RUN=""
GATE_MODE="offline-synthetic"
OUTPUT_DIR="${REPO_ROOT}/labeling_supplement/promotion_decisions_out/run_offline_cycle_$(date '+%Y%m%d_%H%M%S')"
PARALLEL="${PARALLEL:-6}"
NO_ACTIONS=""
CORPUS=""
SOURCES=()
DRY_RUN_WRITEBACK=""
SKIP_WRITEBACK=""
VERBOSE=""

declare -a EXTRA_DRIVER_ARGS=()

usage() {
    sed -n '1,46p' "$0"
}

# ---------------------------------------------------------------------------
# Arg parse — mirrors run_decide_promotion.sh where shared.
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --proposals-run|--proposals_run)        PROPOSALS_RUN="$2"; shift 2 ;;
        --bank-run|--bank_run)                  BANK_RUN="$2"; shift 2 ;;
        --actions-run|--actions_run)            ACTIONS_RUN="$2"; shift 2 ;;
        --writeback-bank-run|--writeback_bank_run)
            WRITEBACK_BANK_RUN="$2"; shift 2 ;;
        --gate-verdicts-run|--gate_verdicts_run)
            GATE_VERDICTS_RUN="$2"; shift 2 ;;
        --gate-mode|--gate_mode)                GATE_MODE="$2"; shift 2 ;;
        --output-dir|--output_dir)              OUTPUT_DIR="$2"; shift 2 ;;
        --parallel)                             PARALLEL="$2"; shift 2 ;;
        --no-actions|--no_actions)              NO_ACTIONS="1"; shift ;;
        --corpus)                               CORPUS="$2"; shift 2 ;;
        --sources)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SOURCES+=("$1")
                shift
            done
            ;;
        --dry-run-writeback|--dry_run_writeback) DRY_RUN_WRITEBACK="1"; shift ;;
        --skip-writeback|--skip_writeback)       SKIP_WRITEBACK="1"; shift ;;
        -v|--verbose)                            VERBOSE="-v"; shift ;;
        --teacher-model | \
        --judge-model | \
        --rollback-min-selections | \
        --rollback-min-pass-rate)
            EXTRA_DRIVER_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

WRITEBACK_BANK_RUN="${WRITEBACK_BANK_RUN:-$BANK_RUN}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${OUTPUT_DIR}/_run.log"

{
    echo "=================================================================="
    echo "Offline promotion cycle (T1.2 — §17 keystone)"
    echo "  proposals-run    : ${PROPOSALS_RUN}"
    echo "  bank-run         : ${BANK_RUN}"
    echo "  actions-run      : ${ACTIONS_RUN}"
    echo "  writeback-bank   : ${WRITEBACK_BANK_RUN}"
    echo "  output-dir       : ${OUTPUT_DIR}"
    echo "  gate-mode        : ${GATE_MODE}"
    echo "  parallel         : ${PARALLEL}"
    echo "  no-actions       : ${NO_ACTIONS:-0}"
    echo "  corpus filter    : ${CORPUS:-<all>}"
    echo "  sources filter   : ${SOURCES[*]:-<all>}"
    echo "  dry-run-writebk  : ${DRY_RUN_WRITEBACK:-0}"
    echo "  skip-writeback   : ${SKIP_WRITEBACK:-0}"
    echo "=================================================================="
} | tee -a "${RUN_LOG}"

# ---------------------------------------------------------------------------
# Step 1 — decide_promotion_gpt54.py over the latest cold-start corpus.
# ---------------------------------------------------------------------------
PYTHONPATH="${WORKSPACE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

DRIVER_ARGS=(
    --proposals-run "${PROPOSALS_RUN}"
    --bank-run      "${BANK_RUN}"
    --output-dir    "${OUTPUT_DIR}"
    --gate-mode     "${GATE_MODE}"
)
if [[ -z "$NO_ACTIONS" ]]; then
    DRIVER_ARGS+=(--actions-run "${ACTIONS_RUN}")
fi
if [[ -n "$GATE_VERDICTS_RUN" ]]; then
    DRIVER_ARGS+=(--gate-verdicts-run "${GATE_VERDICTS_RUN}")
fi
if [[ -n "$CORPUS" ]]; then
    DRIVER_ARGS+=(--corpus "${CORPUS}")
fi
if [[ ${#SOURCES[@]} -gt 0 ]]; then
    DRIVER_ARGS+=(--sources "${SOURCES[@]}")
fi
DRIVER_ARGS+=("${EXTRA_DRIVER_ARGS[@]}")
if [[ -n "$VERBOSE" ]]; then
    DRIVER_ARGS+=(-v)
fi

{
    echo
    echo "── Step 1: decide_promotion_gpt54.py ─────────────────────────────"
    echo "Args: ${DRIVER_ARGS[*]}"
} | tee -a "${RUN_LOG}"

if ! python "${REPO_ROOT}/labeling_supplement/decide_promotion_gpt54.py" \
        "${DRIVER_ARGS[@]}" 2>&1 \
        | tee -a "${RUN_LOG}"; then
    echo "!! decide_promotion_gpt54.py failed — see ${RUN_LOG}" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Step 2 — legacy_writeback.writeback_promotion per pair.
# ---------------------------------------------------------------------------
if [[ -n "$SKIP_WRITEBACK" ]]; then
    {
        echo
        echo "── Step 2: legacy_writeback ── SKIPPED (--skip-writeback)"
    } | tee -a "${RUN_LOG}"
    echo "Done. Promotion artifacts under: ${OUTPUT_DIR}"
    exit 0
fi

WRITEBACK_REPORT="${OUTPUT_DIR}/_writeback_summary.json"
WRITEBACK_LOG="${OUTPUT_DIR}/_writeback.log"

{
    echo
    echo "── Step 2: legacy_writeback.writeback_promotion ──────────────────"
    echo "Output    : ${OUTPUT_DIR}"
    echo "Bank-run  : ${WRITEBACK_BANK_RUN}"
    echo "Dry-run   : ${DRY_RUN_WRITEBACK:-0}"
} | tee -a "${RUN_LOG}"

PYTHONPATH="${WORKSPACE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}" \
PROMOTION_OUT="${OUTPUT_DIR}" \
BANK_RUN_DIR="${WRITEBACK_BANK_RUN}" \
WB_REPORT_PATH="${WRITEBACK_REPORT}" \
DRY_RUN_FLAG="${DRY_RUN_WRITEBACK:-0}" \
python - <<'PY' 2>&1 | tee -a "${WRITEBACK_LOG}" | tee -a "${RUN_LOG}"
"""Inline driver — calls writeback_promotion per (corpus, source) pair.

Mirrors the aggregator pattern at
``labeling_supplement/run_decide_promotion.sh:290-349``. Reads the
promotion-decision artifacts the previous step just emitted and walks
each pair to push its latest snapshot into the trainer-side bank.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROMOTION_OUT = Path(os.environ["PROMOTION_OUT"]).resolve()
BANK_RUN_DIR = Path(os.environ["BANK_RUN_DIR"]).resolve()
WB_REPORT_PATH = Path(os.environ["WB_REPORT_PATH"]).resolve()
DRY_RUN = bool(int(os.environ.get("DRY_RUN_FLAG", "0")))

# Library API only — no CLI on legacy_writeback.
from skill_bank.legacy_writeback import (  # type: ignore
    find_latest_snapshot,
    writeback_promotion,
)


def _walk_pairs(root: Path):
    """Yield (corpus, source, pair_dir) triples for every promotion pair."""
    for corpus_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if corpus_dir.name.startswith("_"):
            continue
        for source_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
            if source_dir.name.startswith("_"):
                continue
            yield corpus_dir.name, source_dir.name, source_dir


summary = {
    "promotion_out": str(PROMOTION_OUT),
    "writeback_bank_run": str(BANK_RUN_DIR),
    "dry_run": DRY_RUN,
    "pairs": [],
    "n_pairs": 0,
    "n_pairs_writeback_ok": 0,
    "n_pairs_skipped": 0,
    "n_pairs_failed": 0,
    "n_inserted_total": 0,
    "n_updated_total": 0,
    "n_skipped_status_total": 0,
}

for corpus, source, pair_dir in _walk_pairs(PROMOTION_OUT):
    summary["n_pairs"] += 1
    snap = find_latest_snapshot(pair_dir)
    if snap is None:
        print(f"  · {corpus}/{source}: no snapshot — skipping", flush=True)
        summary["n_pairs_skipped"] += 1
        summary["pairs"].append({
            "corpus": corpus, "source": source, "snapshot": None,
            "status": "no_snapshot",
        })
        continue

    legacy_path = (
        BANK_RUN_DIR / corpus / source / "skill_bank.jsonl"
    )
    if not legacy_path.is_file():
        print(
            f"  ! {corpus}/{source}: no legacy bank at {legacy_path} — skipping",
            flush=True,
        )
        summary["n_pairs_skipped"] += 1
        summary["pairs"].append({
            "corpus": corpus, "source": source,
            "snapshot": str(snap), "legacy_bank": str(legacy_path),
            "status": "no_legacy_bank",
        })
        continue

    try:
        report = writeback_promotion(
            snapshot_path=snap,
            legacy_bank_path=legacy_path,
            dry_run=DRY_RUN,
        )
    except Exception as exc:                                  # noqa: BLE001
        print(
            f"  ✗ {corpus}/{source}: writeback FAILED — "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        summary["n_pairs_failed"] += 1
        summary["pairs"].append({
            "corpus": corpus, "source": source,
            "snapshot": str(snap), "legacy_bank": str(legacy_path),
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
        })
        continue

    # ``writeback_promotion`` returns a ``WritebackReport`` dataclass —
    # use ``to_dict()`` for the JSON-serialisable view and attribute
    # access for the counters.
    report_dict = report.to_dict()
    n_ins = int(report.n_inserted)
    n_upd = int(report.n_updated)
    n_skip = int(report.n_skipped_status)
    summary["n_pairs_writeback_ok"] += 1
    summary["n_inserted_total"] += n_ins
    summary["n_updated_total"] += n_upd
    summary["n_skipped_status_total"] += n_skip
    summary["pairs"].append({
        "corpus": corpus,
        "source": source,
        "snapshot": str(snap),
        "legacy_bank": str(legacy_path),
        "status": "dry_run" if DRY_RUN else "ok",
        "report": report_dict,
    })
    tag = "DRY" if DRY_RUN else "OK "
    print(
        f"  ✓ [{tag}] {corpus}/{source}: +{n_ins} new, ~{n_upd} updated, "
        f"-{n_skip} non-eligible (snapshot={snap.name})",
        flush=True,
    )

WB_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
WB_REPORT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print()
print(f"Writeback summary → {WB_REPORT_PATH}")
print(
    f"  pairs: {summary['n_pairs_writeback_ok']}/{summary['n_pairs']} ok, "
    f"{summary['n_pairs_skipped']} skipped, {summary['n_pairs_failed']} failed"
)
print(
    f"  totals: +{summary['n_inserted_total']} inserted, "
    f"~{summary['n_updated_total']} updated, "
    f"-{summary['n_skipped_status_total']} non-eligible"
)
sys.exit(0 if summary["n_pairs_failed"] == 0 else 4)
PY
WB_RC=$?

# ---------------------------------------------------------------------------
# Step 3 — assert §17 post-condition: at least one bank now has a
# runnable skill on disk. Probe via direct JSONL inspection (we never
# import the trainer's heavy SkillRepository here).
# ---------------------------------------------------------------------------
{
    echo
    echo "── Step 3: §17 post-condition check ──────────────────────────────"
} | tee -a "${RUN_LOG}"

PYTHONPATH="${WORKSPACE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}" \
BANK_RUN_DIR="${WRITEBACK_BANK_RUN}" \
python - <<'PY' 2>&1 | tee -a "${RUN_LOG}"
"""Probe each ``skill_bank.jsonl`` for at least one entry with a
runnable promotion marker — the §17 keystone post-condition.

The lifecycle status emitted by ``legacy_writeback._project_to_legacy_envelope``
lives at ``row['skill']['_writeback_status']`` (a round-trip-safe
annotation; the legacy ``SkillBankMVP.load()`` round-trips ``report``
through ``VerificationReport.from_dict`` which would reject any extra
field). When the trainer hydrates a ``SkillRepository`` from this
JSONL, ``_writeback_status`` is the field that drives whether a
record lands in the active store / shadow shelf vs. the candidate /
draft stores. So this is the field we probe to assert
``bank.runnable() != []`` after promotion.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BANK_RUN_DIR = Path(os.environ["BANK_RUN_DIR"]).resolve()
RUNNABLE_STATUSES = {"active", "provisional", "shadow"}

n_pairs = 0
n_runnable = 0
n_total = 0
status_counter: dict[str, int] = {}
for corpus_dir in sorted(p for p in BANK_RUN_DIR.iterdir() if p.is_dir()):
    for source_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        bank = source_dir / "skill_bank.jsonl"
        if not bank.is_file():
            continue
        n_pairs += 1
        n_runnable_here = 0
        n_total_here = 0
        for line in bank.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            n_total_here += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            skill = row.get("skill") or {}
            # Promotion-projected entries carry ``_writeback_status``;
            # cold-start originals carry only ``status`` (typically
            # "draft"). Prefer the promotion marker when present so a
            # post-promotion run reflects the new lifecycle even when
            # the legacy ``status`` slot is unmodified.
            status = str(
                skill.get("_writeback_status") or skill.get("status") or ""
            ).lower()
            status_counter[status] = status_counter.get(status, 0) + 1
            if status in RUNNABLE_STATUSES:
                n_runnable_here += 1
        n_runnable += n_runnable_here
        n_total += n_total_here
        marker = "✓" if n_runnable_here > 0 else "·"
        print(
            f"  {marker} {corpus_dir.name}/{source_dir.name}: "
            f"{n_runnable_here}/{n_total_here} runnable",
            flush=True,
        )

print()
print(
    f"§17 keystone status: {n_runnable}/{n_total} runnable across "
    f"{n_pairs} pair(s) under {BANK_RUN_DIR}"
)
print(f"  status breakdown: {status_counter}")
sys.exit(0 if n_runnable > 0 else 5)
PY
POST_RC=$?

{
    echo
    echo "Done. Final status:"
    echo "  writeback exit_code     = ${WB_RC}"
    echo "  §17 keystone exit_code  = ${POST_RC}"
    if [[ ${POST_RC} -eq 0 ]]; then
        echo "  bank.runnable() != [] → trainer launch unblocked"
    else
        echo "  bank.runnable() == []  → re-check decide_promotion gate verdicts"
    fi
    echo "  artifacts: ${OUTPUT_DIR}"
} | tee -a "${RUN_LOG}"

if [[ ${WB_RC} -ne 0 ]]; then
    exit ${WB_RC}
fi
exit ${POST_RC}
