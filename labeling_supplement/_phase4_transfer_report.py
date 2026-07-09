#!/usr/bin/env python
"""Phase-5/6 -- Stage 6 unified transfer report generator.

Sibling of `labeling_supplement/_phase4_transfer_matrix.py`. Reads the
matrix driver's `cells.json` AND Stage 0's `upper_bounds.csv`, and
emits the unified `_report.md` that closes the Phase-5/6 cross-domain
measurement plan
(implementation_notes/legacy/phase5-cross-domain-measurement.md, sections 9
and 11.5.4 / 11.5.6).

I/O contract:

    Inputs
    ------
    --cells-json PATH
        Path to a Stage 6 cells.json. Default: most recent
        cross_domain_results/_final/*/cells.json on disk.
    --upper-bounds-csv PATH
        Path to Stage 0's upper_bounds.csv. Default:
        cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv.
    --out-path PATH
        Where to write the report. Default: <cells_json_dir>/_report.md.
    --slack FLOAT
        Tolerance applied to the Stage 0 upper-bound comparison
        (memo section 9 / 11.5.6). Default: 0.10.

    Output
    ------
    Markdown report at `--out-path` with seven sections:

      1. Run metadata               -- config + cell count + elapsed.
      2. Experiment A               -- game-source admit rates.
      3. Experiment B               -- within-VR/video 4x4.
      4. Experiment C               -- cross-cluster cells.
      5. Stage 0 upper-bound diff   -- per-cell measured vs upper_bound.
      6. Acceptance gates           -- G1..G6 PASS/FAIL/N-A verdicts.
      7. Per-skill rationale        -- top-10 admits + top diagnostic
                                       labels among rejects.

Usage::

    python -m labeling_supplement._phase4_transfer_report
    python -m labeling_supplement._phase4_transfer_report \\
        --cells-json cross_domain_results/_final/run_<ts>/cells.json \\
        --upper-bounds-csv cross_domain_results/_phase0/phase0_canonical/upper_bounds.csv

The driver writes to disk and prints the report path; it never exits
non-zero on gate failure -- the gate verdicts are reported in-band so
downstream automation can grep them.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("phase4_transfer_report")


DEFAULT_FINAL_GLOB = REPO_ROOT / "cross_domain_results" / "_final"
DEFAULT_UPPER_BOUNDS_CSV = (
    REPO_ROOT / "cross_domain_results" / "_phase0" / "phase0_canonical"
    / "upper_bounds.csv"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _find_latest_cells_json() -> Optional[Path]:
    """Return the most-recently-modified cells.json under _final/."""
    if not DEFAULT_FINAL_GLOB.exists():
        return None
    candidates = sorted(
        DEFAULT_FINAL_GLOB.glob("*/cells.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_cells(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_upper_bounds(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return ``{(source_corpus, target_domain): row_dict}``."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("source_corpus", ""), row.get("target_domain", ""))
            try:
                row["upper_bound_admit_rate"] = float(
                    row.get("upper_bound_admit_rate") or 0.0
                )
            except (TypeError, ValueError):
                row["upper_bound_admit_rate"] = 0.0
            out[key] = row
    return out


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _format_pct(rate: float, n_admit: int, n_total: int) -> str:
    return f"{rate:.0%} ({n_admit}/{n_total})"


def _matrix_table(
    cells: Sequence[Dict[str, Any]],
    *,
    sources: Sequence[str],
    targets: Sequence[str],
) -> List[str]:
    """Build a markdown table mapping (source, target) -> admit rate."""
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {
        (c["source_corpus"], c["target_corpus"]): c for c in cells
    }
    lines: List[str] = []
    if not sources or not targets:
        lines.append("(no cells in this partition)")
        return lines
    header = "| source \\\\ target | " + " | ".join(targets) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(targets) + 1)) + "|")
    for src in sources:
        row = [src]
        for tgt in targets:
            c = by_pair.get((src, tgt))
            if c is None:
                row.append("--")
            elif c.get("error"):
                row.append(f"ERR ({c['n_admit']}/{c['n_total']})")
            else:
                row.append(_format_pct(
                    c["admit_rate"], c["n_admit"], c["n_total"]
                ))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _filter_cells(
    cells: Sequence[Dict[str, Any]],
    *,
    source_clusters: Optional[Sequence[str]] = None,
    target_clusters: Optional[Sequence[str]] = None,
    same_cluster: Optional[bool] = None,
    diagonal: Optional[bool] = None,
    skip_errored: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in cells:
        if skip_errored and c.get("error"):
            continue
        sc = c.get("source_cluster", "unknown")
        tc = c.get("target_cluster", "unknown")
        if source_clusters is not None and sc not in source_clusters:
            continue
        if target_clusters is not None and tc not in target_clusters:
            continue
        if same_cluster is True and sc != tc:
            continue
        if same_cluster is False and sc == tc:
            continue
        if diagonal is True and c["source_corpus"] != c["target_corpus"]:
            continue
        if diagonal is False and c["source_corpus"] == c["target_corpus"]:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Section emitters
# ---------------------------------------------------------------------------


def _section_metadata(payload: Dict[str, Any]) -> List[str]:
    cfg = payload.get("config", {})
    lines: List[str] = []
    lines.append("## 1. Run metadata\n")
    lines.append(f"- run_id: `{payload.get('run_id', '?')}`")
    lines.append(f"- timestamp: `{payload.get('timestamp', '?')}`")
    lines.append(f"- n_cells: {payload.get('n_cells', '?')}")
    lines.append(f"- elapsed_s: {payload.get('elapsed_s', '?')}")
    lines.append("- config:")
    lines.append("  ```")
    for k in sorted(cfg.keys()):
        lines.append(f"  {k}: {cfg[k]}")
    lines.append("  ```")
    lines.append(
        f"- source_corpora ({len(payload.get('source_corpora') or [])}): "
        f"{payload.get('source_corpora')}"
    )
    lines.append(
        f"- target_corpora ({len(payload.get('target_corpora') or [])}): "
        f"{payload.get('target_corpora')}"
    )
    lines.append("")
    return lines


def _section_experiment_a(cells: Sequence[Dict[str, Any]]) -> List[str]:
    """Game-source admit rates -- N source rows x M target columns."""
    lines: List[str] = []
    lines.append("## 2. Experiment A -- game-source admit rates\n")
    lines.append(
        "Filter: `source_cluster == \"game\"`. Rows are game source "
        "banks; columns are every measured target corpus.\n"
    )
    game_cells = _filter_cells(cells, source_clusters=("game",), skip_errored=False)
    if not game_cells:
        lines.append("(no game-source cells in this run)")
        lines.append("")
        return lines
    sources = sorted({c["source_corpus"] for c in game_cells})
    targets = sorted({c["target_corpus"] for c in game_cells})
    lines.extend(_matrix_table(game_cells, sources=sources, targets=targets))
    lines.append("")
    return lines


def _section_experiment_b(cells: Sequence[Dict[str, Any]]) -> List[str]:
    """Within-VR/video 4x4 matrix."""
    lines: List[str] = []
    lines.append("## 3. Experiment B -- within-VR/video 4x4\n")
    lines.append(
        "Filter: `source_cluster in {image,video}` AND "
        "`target_cluster in {image,video}`. This is the same cell "
        "partition Stage 5's `_phase5_matrix.py` measures, here folded "
        "into the unified report.\n"
    )
    vr_cells = _filter_cells(
        cells,
        source_clusters=("image", "video"),
        target_clusters=("image", "video"),
        skip_errored=False,
    )
    if not vr_cells:
        lines.append("(no VR/video cells in this run)")
        lines.append("")
        return lines
    sources = sorted({c["source_corpus"] for c in vr_cells})
    targets = sorted({c["target_corpus"] for c in vr_cells})
    lines.extend(_matrix_table(vr_cells, sources=sources, targets=targets))
    lines.append("")
    return lines


def _section_experiment_c(cells: Sequence[Dict[str, Any]]) -> List[str]:
    """Cross-cluster cells, grouped by (source_cluster, target_cluster)."""
    lines: List[str] = []
    lines.append("## 4. Experiment C -- cross-cluster cells\n")
    lines.append(
        "Filter: `source_cluster != target_cluster`. Sub-tables grouped "
        "by `(source_cluster, target_cluster)`.\n"
    )
    xc = _filter_cells(cells, same_cluster=False, skip_errored=False)
    if not xc:
        lines.append("(no cross-cluster cells in this run)")
        lines.append("")
        return lines
    by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for c in xc:
        by_pair[(c["source_cluster"], c["target_cluster"])].append(c)
    for (sc, tc) in sorted(by_pair.keys()):
        sub = by_pair[(sc, tc)]
        lines.append(f"### {sc} -> {tc}\n")
        sources = sorted({c["source_corpus"] for c in sub})
        targets = sorted({c["target_corpus"] for c in sub})
        lines.extend(_matrix_table(sub, sources=sources, targets=targets))
        lines.append("")
    return lines


def _section_upper_bound(
    cells: Sequence[Dict[str, Any]],
    upper_bounds: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    slack: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Compare measured admit_rate to Stage 0 upper bound + slack."""
    lines: List[str] = []
    lines.append(
        "## 5. Stage 0 upper-bound comparison "
        f"(slack={slack:.2f})\n"
    )
    lines.append(
        "Stage 0 keys upper bounds on `(source_corpus, target_domain)` "
        "rather than `target_corpus`, so multiple target corpora that "
        "share a target_domain are compared to the same upper bound. "
        "`violation = YES` iff measured > upper_bound + slack.\n"
    )
    lines.append(
        "| source_corpus | target_corpus | target_domain | measured | "
        "upper_bound | delta | violation |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    rows: List[Dict[str, Any]] = []
    for c in cells:
        if c.get("error"):
            continue
        td = c.get("target_domain")
        sc = c.get("source_corpus")
        ub_row = upper_bounds.get((sc, td))
        ub_val = (
            ub_row.get("upper_bound_admit_rate") if ub_row is not None else None
        )
        measured = c["admit_rate"]
        if ub_val is None:
            delta = None
            violation = "no-ub"
        else:
            delta = measured - ub_val
            violation = "YES" if measured > ub_val + slack else "no"
        rows.append({
            "source_corpus": sc,
            "target_corpus": c["target_corpus"],
            "target_domain": td,
            "measured": measured,
            "upper_bound": ub_val,
            "delta": delta,
            "violation": violation,
        })

    rows.sort(key=lambda r: (r["violation"] != "YES", r["source_corpus"], r["target_corpus"]))
    n_violations = sum(1 for r in rows if r["violation"] == "YES")
    n_no_ub = sum(1 for r in rows if r["violation"] == "no-ub")

    for r in rows:
        ub_disp = "?" if r["upper_bound"] is None else f"{r['upper_bound']:.2%}"
        delta_disp = "?" if r["delta"] is None else f"{r['delta']:+.2%}"
        lines.append(
            f"| {r['source_corpus']} | {r['target_corpus']} | "
            f"{r['target_domain']} | {r['measured']:.2%} | {ub_disp} | "
            f"{delta_disp} | {r['violation']} |"
        )
    lines.append("")
    lines.append(
        f"**Total violations**: {n_violations} / {len(rows)} cells "
        f"(no-ub rows: {n_no_ub})."
    )
    if n_violations:
        viol = [
            f"{r['source_corpus']}->{r['target_corpus']}"
            for r in rows if r['violation'] == "YES"
        ]
        lines.append(f"Violation cells: {viol}")
    lines.append("")
    return lines, rows


def _section_acceptance_gates(
    cells: Sequence[Dict[str, Any]],
    upper_bound_rows: Sequence[Dict[str, Any]],
) -> List[str]:
    """Memo section 11.5.6 acceptance gates (G1..G6)."""
    lines: List[str] = []
    lines.append("## 6. Acceptance gates (memo section 11.5.6)\n")
    lines.append("| Gate | Result | Cells evaluated | Note |")
    lines.append("|---|---|---|---|")

    measurable = [c for c in cells if not c.get("error")]

    def _avg_rate(items: Sequence[Dict[str, Any]]) -> Optional[float]:
        if not items:
            return None
        return sum(c["admit_rate"] for c in items) / len(items)

    def _format_cells(items: Sequence[Dict[str, Any]], cap: int = 5) -> str:
        if not items:
            return "no cells"
        sample = items[:cap]
        suffix = "" if len(items) <= cap else f", +{len(items) - cap} more"
        return ", ".join(
            f"{c['source_corpus']}->{c['target_corpus']}={c['admit_rate']:.0%}"
            for c in sample
        ) + suffix

    # G1: diagonal cells (source_corpus == target_corpus) >= 80%
    diag = _filter_cells(measurable, diagonal=True, skip_errored=False)
    if not diag:
        g1 = "N-A"
        g1_note = "no diagonal cells measured"
    else:
        ok = all(c["admit_rate"] >= 0.80 for c in diag)
        g1 = "PASS" if ok else "FAIL"
        g1_note = _format_cells(diag)
    lines.append(f"| G1 (diagonal >=80%) | {g1} | {len(diag)} | {g1_note} |")

    # G2: within-cluster off-diagonal >= 30%
    within_off = [
        c for c in _filter_cells(measurable, same_cluster=True)
        if c["source_corpus"] != c["target_corpus"]
    ]
    if not within_off:
        g2 = "N-A"
        g2_note = "no within-cluster off-diag cells"
    else:
        ok = all(c["admit_rate"] >= 0.30 for c in within_off)
        g2 = "PASS" if ok else "FAIL"
        g2_note = _format_cells(within_off)
    lines.append(
        f"| G2 (within-cluster off-diag >=30%) | {g2} | "
        f"{len(within_off)} | {g2_note} |"
    )

    # G3: cross-cluster game <-> image-VR in [15%, 35%]
    g3_cells = [
        c for c in measurable
        if (c["source_cluster"] == "game" and c["target_cluster"] == "image")
        or (c["source_cluster"] == "image" and c["target_cluster"] == "game")
    ]
    if not g3_cells:
        g3 = "N-A"
        g3_note = "no game<->image-VR cells"
    else:
        ok = all(0.15 <= c["admit_rate"] <= 0.35 for c in g3_cells)
        g3 = "PASS" if ok else "FAIL"
        g3_note = _format_cells(g3_cells)
    lines.append(
        f"| G3 (game<->image-VR in [15%,35%]) | {g3} | "
        f"{len(g3_cells)} | {g3_note} |"
    )

    # G4: cross-cluster game <-> video-VR in [15%, 30%]
    g4_cells = [
        c for c in measurable
        if (c["source_cluster"] == "game" and c["target_cluster"] == "video")
        or (c["source_cluster"] == "video" and c["target_cluster"] == "game")
    ]
    if not g4_cells:
        g4 = "N-A"
        g4_note = "no game<->video-VR cells"
    else:
        ok = all(0.15 <= c["admit_rate"] <= 0.30 for c in g4_cells)
        g4 = "PASS" if ok else "FAIL"
        g4_note = _format_cells(g4_cells)
    lines.append(
        f"| G4 (game<->video-VR in [15%,30%]) | {g4} | "
        f"{len(g4_cells)} | {g4_note} |"
    )

    # G5: QA-source -> game-target near-zero (<5%); soft FAIL >=5%
    g5_cells = [
        c for c in measurable
        if c["source_cluster"] in ("image", "video")
        and c["target_cluster"] == "game"
    ]
    if not g5_cells:
        g5 = "N-A"
        g5_note = "no QA->game cells"
    else:
        rates = [c["admit_rate"] for c in g5_cells]
        max_rate = max(rates)
        if max_rate < 0.05:
            g5 = "PASS"
        else:
            g5 = "soft-FAIL"
        g5_note = (
            f"max={max_rate:.0%}; "
            + _format_cells(g5_cells)
        )
    lines.append(
        f"| G5 (QA->game <5%, informative) | {g5} | "
        f"{len(g5_cells)} | {g5_note} |"
    )

    # G6: all measured rates <= upper_bound + slack (no violations)
    if not upper_bound_rows:
        g6 = "N-A"
        g6_note = "upper_bounds.csv missing"
    else:
        n_viol = sum(1 for r in upper_bound_rows if r["violation"] == "YES")
        n_total = len(upper_bound_rows)
        if n_viol == 0:
            g6 = "PASS"
        else:
            g6 = "FAIL"
        g6_note = (
            f"{n_viol}/{n_total} violations" if n_viol == 0
            else f"violators: " + ", ".join(
                f"{r['source_corpus']}->{r['target_corpus']} "
                f"(measured={r['measured']:.0%}, ub={r['upper_bound']:.0%})"
                for r in upper_bound_rows if r["violation"] == "YES"
            )[:160]
        )
    lines.append(f"| G6 (measured <= upper_bound + slack) | {g6} | "
                 f"{len(upper_bound_rows)} | {g6_note} |")
    lines.append("")
    return lines


def _section_per_skill(cells: Sequence[Dict[str, Any]]) -> List[str]:
    """Top-10 admits + diagnostic-label rejects."""
    lines: List[str] = []
    lines.append("## 7. Per-skill admit / reject rationale\n")

    flat: List[Dict[str, Any]] = []
    for c in cells:
        if c.get("error"):
            continue
        for v in c["verdicts"]:
            flat.append({
                "source_corpus": c["source_corpus"],
                "target_corpus": c["target_corpus"],
                "skill_id": v["skill_id"],
                "skill_type": v["skill_type"],
                "pass_rate": v["pass_rate"],
                "n_success": v["n_success"],
                "n_total": v["n_demos_used"] or 0,
                "success": v["success"],
                "diagnostic_label": v.get("diagnostic_label"),
            })

    admits = [f for f in flat if f["success"]]
    rejects = [f for f in flat if not f["success"]]

    lines.append("### Top-10 admits (by pass_rate, then n_success desc)\n")
    if not admits:
        lines.append("(no admitted skills in this run)")
        lines.append("")
    else:
        admits_sorted = sorted(
            admits,
            key=lambda x: (-x["pass_rate"], -x["n_success"]),
        )[:10]
        lines.append(
            "| source_corpus | target_corpus | skill_id | "
            "pass_rate | n_success/n_total |"
        )
        lines.append("|---|---|---|---|---|")
        for a in admits_sorted:
            lines.append(
                f"| {a['source_corpus']} | {a['target_corpus']} | "
                f"`{a['skill_id'][:48]}` | {a['pass_rate']:.2f} | "
                f"{a['n_success']}/{a['n_total']} |"
            )
        lines.append("")

    lines.append("### Top-10 rejects -- by diagnostic_label frequency\n")
    if not rejects:
        lines.append("(no rejected skills in this run)")
        lines.append("")
        return lines
    diag_counter: Counter = Counter()
    samples_by_diag: Dict[str, List[str]] = defaultdict(list)
    for r in rejects:
        d = r.get("diagnostic_label") or "(none)"
        diag_counter[d] += 1
        if len(samples_by_diag[d]) < 3:
            samples_by_diag[d].append(r["skill_id"])

    lines.append("| diagnostic_label | count | sample skill_ids |")
    lines.append("|---|---|---|")
    for d, n in diag_counter.most_common(10):
        sample = ", ".join(f"`{s[:32]}`" for s in samples_by_diag[d])
        lines.append(f"| {d} | {n} | {sample} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cells-json",
        default=None,
        help=("Path to a Stage 6 cells.json. Default: most recent "
              "cross_domain_results/_final/*/cells.json."),
    )
    p.add_argument(
        "--upper-bounds-csv",
        default=str(DEFAULT_UPPER_BOUNDS_CSV),
        help=("Path to Stage 0's upper_bounds.csv. Default: "
              "cross_domain_results/_phase0/phase0_canonical/"
              "upper_bounds.csv."),
    )
    p.add_argument(
        "--out-path",
        default=None,
        help="Where to write the report. Default: <cells_json_dir>/_report.md.",
    )
    p.add_argument(
        "--slack",
        type=float,
        default=0.10,
        help="Stage 0 upper-bound slack. Default: 0.10.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cells_json:
        cells_json_path = Path(args.cells_json)
    else:
        latest = _find_latest_cells_json()
        if latest is None:
            raise SystemExit(
                f"no cells.json found under {DEFAULT_FINAL_GLOB}; pass "
                f"--cells-json explicitly or run "
                f"`python -m labeling_supplement._phase4_transfer_matrix` first."
            )
        cells_json_path = latest
    if not cells_json_path.exists():
        raise SystemExit(f"cells.json missing: {cells_json_path}")

    upper_bounds_csv_path = Path(args.upper_bounds_csv)
    if not upper_bounds_csv_path.exists():
        logger.warning(
            "upper_bounds.csv missing: %s (G6 + Section 5 will be N-A)",
            upper_bounds_csv_path,
        )

    payload = _load_cells(cells_json_path)
    cells = payload.get("cells") or []
    upper_bounds = _load_upper_bounds(upper_bounds_csv_path)

    out_path = (
        Path(args.out_path) if args.out_path
        else cells_json_path.parent / "_report.md"
    )

    body: List[str] = []
    body.append(
        f"# Phase-5/6 Stage-6 unified transfer report\n\n"
        f"Source `cells.json`: `{cells_json_path}`\n"
        f"Stage 0 upper bounds: `{upper_bounds_csv_path}` "
        f"({len(upper_bounds)} rows loaded)\n"
        f"Slack: {args.slack:.2f}\n"
    )

    body.extend(_section_metadata(payload))
    body.extend(_section_experiment_a(cells))
    body.extend(_section_experiment_b(cells))
    body.extend(_section_experiment_c(cells))
    section5_lines, ub_rows = _section_upper_bound(cells, upper_bounds, slack=args.slack)
    body.extend(section5_lines)
    body.extend(_section_acceptance_gates(cells, ub_rows))
    body.extend(_section_per_skill(cells))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body))

    print()
    print(f"=== Phase-5/6 Stage-6 unified report ===")
    print(f"cells.json:        {cells_json_path}")
    print(f"upper_bounds.csv:  {upper_bounds_csv_path} "
          f"({len(upper_bounds)} rows)")
    print(f"slack:             {args.slack:.2f}")
    print(f"_report.md:        {out_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
