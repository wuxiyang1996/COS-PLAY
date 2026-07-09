"""aggregate_osworld_baseline.py — collapse 4-provider OSWorld 50-task
runs into a publication-ready comparison table.

Reads each ``runs/osworld_baseline_50/<provider>/`` subdirectory
(written by ``run_osworld_multimodel.sh`` with ``--output_dir``), pulls
per-task ``rollout_summary.json`` files via the existing
``cold_start.load_rollouts.aggregate_run_pass_at_1`` helper (so the
``eval_score is None`` → 0 convention stays consistent with the rest
of the codebase), then writes:

  ``compare/pass_at_1_overall.csv``    one row per provider with
                                       global pass@1 + Wilson 95% CI.
  ``compare/pass_at_1_per_domain.csv`` provider × domain matrix.
  ``compare/per_task.csv``             task × provider 0/1 matrix +
                                       overall agreement column.
  ``compare/readme.md``                human-readable rendering.

The script is *purely additive* — it never edits the per-provider
output dirs. Run it as many times as you want; later providers
arriving via rsync will trigger an updated comparison without
touching earlier data.

Usage:
    python scripts/aggregate_osworld_baseline.py
    python scripts/aggregate_osworld_baseline.py --root runs/osworld_baseline_50
    python scripts/aggregate_osworld_baseline.py --legacy-mean   # exclude null
    python scripts/aggregate_osworld_baseline.py --providers gpt5 claude-sonnet
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is importable when the script is run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cold_start.load_rollouts import aggregate_run_pass_at_1  # noqa: E402


# Provider tag → human-readable label. Keep in sync with
# ``cold_start/run_osworld_multimodel.sh:provider_to_model``.
DEFAULT_PROVIDER_LABELS = {
    "gpt5":           "GPT-5.4",
    "gpt5-or":        "GPT-5.4 (OR)",
    "claude-sonnet":  "Claude Sonnet 4.6",
    "claude-opus":    "Claude Opus 4.7",
    "gemini-pro":     "Gemini 2.5 Pro",
    "gemini-3-pro":   "Gemini 3.1 Pro Preview",
    "qwen3-vl":       "Qwen3-VL 235B-A22B Instruct",
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _wilson_95(k: int, n: int) -> Tuple[float, float, float]:
    """Wilson score interval at 95% confidence.

    Returns ``(rate, lower, upper)``. Falls back to ``(0,0,0)`` for
    ``n=0``. We use Wilson rather than Clopper-Pearson because the
    project's ``evaluation/scoreboard.py`` already uses Wilson and
    consistency is more valuable than a few-decimal-points difference.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054  # 95% two-sided
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Per-provider scan
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    """One provider's full scan over all per-task rollout summaries."""
    provider: str
    label: str
    run_dir: Path
    overall_n: int = 0
    overall_solved: int = 0
    overall_unscored: int = 0
    overall_errored: int = 0
    per_domain: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_task: Dict[Tuple[str, str], Optional[float]] = field(default_factory=dict)
    found: bool = False
    note: str = ""


def _scan_provider(
    root: Path,
    provider: str,
    *,
    label: str,
    treat_null_as_zero: bool,
) -> ProviderResult:
    """Aggregate a single provider's run dir into a ProviderResult."""
    run_dir = root / provider
    res = ProviderResult(provider=provider, label=label, run_dir=run_dir)

    if not run_dir.is_dir():
        res.note = "provider dir missing — has the run finished + been rsync'd?"
        return res

    # Reuse the existing aggregator for the global + per-domain numbers
    # (it knows the rollout_summary.json shape and applies the
    # ``eval_score is None`` → 0 rule consistently).
    agg = aggregate_run_pass_at_1(
        str(run_dir), treat_null_as_zero=treat_null_as_zero,
    )
    res.found = True
    res.overall_n = int(agg.get("total_tasks", 0))
    res.overall_solved = int(agg.get("solved", 0))
    res.overall_unscored = int(agg.get("unscored", 0))
    res.overall_errored = int(agg.get("errored", 0))
    res.per_domain = dict(agg.get("per_domain", {}))

    # Walk per-task to build the (provider × task) matrix used in
    # ``per_task.csv`` and the agreement column.
    for domain_dir in sorted(run_dir.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
            continue
        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            rs = task_dir / "rollout_summary.json"
            if not rs.is_file():
                continue
            try:
                summary = json.loads(rs.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            best: Optional[float] = None
            for ep in (summary.get("episode_stats") or []):
                if ep.get("error"):
                    continue
                sc = ep.get("eval_score")
                if isinstance(sc, (int, float)):
                    best = max(best or 0.0, float(sc))
            res.per_task[(domain_dir.name, task_dir.name)] = best

    return res


# ---------------------------------------------------------------------------
# CSV emitters
# ---------------------------------------------------------------------------

def _write_overall_csv(out: Path, results: List[ProviderResult]) -> None:
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "provider", "label", "n", "solved", "unscored", "errored",
            "pass@1", "wilson95_lo", "wilson95_hi", "run_dir", "note",
        ])
        for r in results:
            rate, lo, hi = _wilson_95(r.overall_solved, r.overall_n)
            w.writerow([
                r.provider, r.label, r.overall_n, r.overall_solved,
                r.overall_unscored, r.overall_errored,
                f"{rate:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                str(r.run_dir), r.note,
            ])


def _write_per_domain_csv(out: Path, results: List[ProviderResult]) -> None:
    domains = sorted({d for r in results for d in r.per_domain})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["domain"]
            + [f"{r.label} (pass@1)" for r in results]
            + [f"{r.label} (n)" for r in results]
        )
        for dom in domains:
            row: List[Any] = [dom]
            rates: List[Any] = []
            ns: List[Any] = []
            for r in results:
                d = r.per_domain.get(dom)
                if d is None:
                    rates.append("")
                    ns.append(0)
                else:
                    rates.append(f"{d['solved'] / d['tasks']:.4f}"
                                 if d["tasks"] else "")
                    ns.append(d["tasks"])
            w.writerow(row + rates + ns)


def _write_per_task_csv(out: Path, results: List[ProviderResult]) -> None:
    """One row per (domain, task_id). Each provider column is 1 (best
    eval_score > 0), 0 (eval_score == 0), or '' (provider did not run
    this task / unscored / errored).
    """
    all_tasks = sorted({k for r in results for k in r.per_task})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "task_id"]
                   + [r.label for r in results]
                   + ["n_solved_across_providers"])
        for (dom, tid) in all_tasks:
            cells: List[Any] = []
            n_solved = 0
            for r in results:
                v = r.per_task.get((dom, tid))
                if v is None:
                    cells.append("")
                else:
                    cell = 1 if v > 0 else 0
                    cells.append(cell)
                    n_solved += cell
            w.writerow([dom, tid, *cells, n_solved])


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _format_pct(rate: float, lo: float, hi: float) -> str:
    return f"{rate * 100:5.1f}%  ({lo * 100:4.1f}% – {hi * 100:4.1f}%)"


def _write_readme(
    out: Path,
    results: List[ProviderResult],
    *,
    treat_null_as_zero: bool,
    root: Path,
) -> None:
    domains = sorted({d for r in results for d in r.per_domain})

    lines: List[str] = []
    lines.append("# OSWorld 4-model baseline — comparison report")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from `{root}`. eval_score=None → "
        f"{'0 (treat as failure)' if treat_null_as_zero else 'excluded'}._"
    )
    lines.append("")
    lines.append("## Overall pass@1 (Wilson 95% CI)")
    lines.append("")
    lines.append("| Provider | n | solved | unscored | errored | pass@1 | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if not r.found:
            lines.append(f"| {r.label} | — | — | — | — | _missing_ | _{r.note}_ |")
            continue
        rate, lo, hi = _wilson_95(r.overall_solved, r.overall_n)
        lines.append(
            f"| {r.label} | {r.overall_n} | {r.overall_solved} | "
            f"{r.overall_unscored} | {r.overall_errored} | "
            f"{rate * 100:.1f}% | {lo * 100:.1f}% – {hi * 100:.1f}% |"
        )
    lines.append("")

    lines.append("## Per-domain pass@1")
    lines.append("")
    header = "| Domain | " + " | ".join(r.label for r in results) + " |"
    sep = "|" + "|".join("---" for _ in range(len(results) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for dom in domains:
        cells: List[str] = []
        for r in results:
            d = r.per_domain.get(dom)
            if d is None or d["tasks"] == 0:
                cells.append("—")
            else:
                rate, lo, hi = _wilson_95(d["solved"], d["tasks"])
                cells.append(f"{rate * 100:.0f}% ({d['solved']}/{d['tasks']})")
        lines.append(f"| {dom} | " + " | ".join(cells) + " |")
    lines.append("")

    # 4-way agreement breakdown
    if all(r.found for r in results) and len(results) >= 2:
        all_tasks = sorted({k for r in results for k in r.per_task})
        agree_solved = 0
        agree_failed = 0
        disagree = 0
        unknown = 0
        for k in all_tasks:
            outcomes = []
            for r in results:
                v = r.per_task.get(k)
                outcomes.append(None if v is None else (1 if v > 0 else 0))
            if any(o is None for o in outcomes):
                unknown += 1
            elif all(o == 1 for o in outcomes):
                agree_solved += 1
            elif all(o == 0 for o in outcomes):
                agree_failed += 1
            else:
                disagree += 1

        n_full = agree_solved + agree_failed + disagree
        lines.append("## Cross-provider agreement (over tasks where every provider finished)")
        lines.append("")
        lines.append(f"- All providers solved:   **{agree_solved} / {n_full}**")
        lines.append(f"- All providers failed:   **{agree_failed} / {n_full}**")
        lines.append(f"- Mixed outcome:          **{disagree} / {n_full}**")
        if unknown:
            lines.append(f"- (Excluded for incomplete coverage: {unknown})")
        lines.append("")
        lines.append(
            "The mixed-outcome row tasks are the most informative ones "
            "for debugging: they expose the slice where one provider "
            "differs from the rest."
        )
        lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `pass_at_1_overall.csv`   — global numbers per provider.")
    lines.append("- `pass_at_1_per_domain.csv` — per-domain × per-provider grid.")
    lines.append("- `per_task.csv`             — per-task × per-provider 0/1 matrix.")
    lines.append("- `readme.md`                — this file.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root", type=Path,
        default=_PROJECT_ROOT / "runs" / "osworld_baseline_50",
        help=("root that contains one subdirectory per provider "
              f"(default: {_PROJECT_ROOT / 'runs' / 'osworld_baseline_50'})"),
    )
    p.add_argument(
        "--providers", nargs="+",
        default=list(DEFAULT_PROVIDER_LABELS.keys()),
        help=("provider tags to include (default: all known). "
              "Missing dirs are reported but do not abort the run."),
    )
    p.add_argument(
        "--treat-null-as-zero", dest="treat_null_as_zero",
        action="store_true", default=True,
        help="Count eval_score=None as 0 (default).",
    )
    p.add_argument(
        "--legacy-mean", dest="treat_null_as_zero",
        action="store_false",
        help="Match the old buggy mean — exclude unscored episodes.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Write CSVs / readme under this directory (default: <root>/compare/).",
    )
    args = p.parse_args()

    root = args.root.expanduser().resolve()
    out_dir = (args.out or (root / "compare")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"  root:  {root}")
    print(f"  out:   {out_dir}")
    print(f"  null-as-zero: {args.treat_null_as_zero}")
    print(f"  providers: {' '.join(args.providers)}")
    print("=" * 64)

    results: List[ProviderResult] = []
    for tag in args.providers:
        label = DEFAULT_PROVIDER_LABELS.get(tag, tag)
        r = _scan_provider(
            root, tag,
            label=label, treat_null_as_zero=args.treat_null_as_zero,
        )
        results.append(r)
        if r.found:
            rate, lo, hi = _wilson_95(r.overall_solved, r.overall_n)
            print(
                f"  [{tag:<14}] n={r.overall_n:>3d}  "
                f"pass@1={rate * 100:5.1f}%  CI=({lo * 100:4.1f}-{hi * 100:4.1f})%  "
                f"unscored={r.overall_unscored}"
            )
        else:
            print(f"  [{tag:<14}] MISSING — {r.note}")

    if not any(r.found for r in results):
        print("\n[ERROR] no provider directories found; nothing to write.")
        return 2

    # Filter out missing providers from the output tables.
    found = [r for r in results if r.found]

    _write_overall_csv(out_dir / "pass_at_1_overall.csv", found)
    _write_per_domain_csv(out_dir / "pass_at_1_per_domain.csv", found)
    _write_per_task_csv(out_dir / "per_task.csv", found)
    _write_readme(
        out_dir / "readme.md", found,
        treat_null_as_zero=args.treat_null_as_zero, root=root,
    )

    print("=" * 64)
    print(f"  wrote {out_dir}/pass_at_1_overall.csv")
    print(f"  wrote {out_dir}/pass_at_1_per_domain.csv")
    print(f"  wrote {out_dir}/per_task.csv")
    print(f"  wrote {out_dir}/readme.md")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
