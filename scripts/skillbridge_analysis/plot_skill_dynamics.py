"""Plot skill-bank dynamics over training (block E1).

Inputs:

* ``<run-dir>/lifecycle_log/transitions.jsonl``        (skill status transitions)
* ``<run-dir>/promotion_decisions_out/*_run_summary.json`` (per-decision audit)
* ``<run-dir>/audit.jsonl``                             (crafter mutation events)

Produces two figures:

1. ``skill_dynamics_curves.png`` — cumulative counts of
   {promoted, rejected, deprecated, retired} skills against trainer step.
2. ``crafter_mutation_pie.png``  — pie chart of crafter mutation tags.

Also writes ``skill_dynamics_summary.json`` with the underlying counts
so downstream LaTeX macros can quote exact numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib output (still writes the JSON summary).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _aggregate_lifecycle(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    cumulative_by_step: Dict[str, List[tuple[int, int]]] = defaultdict(list)
    by_step_running = Counter()

    rows = sorted(rows, key=lambda r: (r.get("step") or 0, r.get("ts") or 0.0))
    for r in rows:
        to_status = (r.get("to_status") or "").upper()
        counts[to_status] += 1
        step = int(r.get("step") or 0)
        by_step_running[to_status] += 1
        cumulative_by_step[to_status].append((step, by_step_running[to_status]))

    return {
        "total_transitions": len(rows),
        "counts_by_to_status": dict(counts),
        "cumulative_by_step": {k: v for k, v in cumulative_by_step.items()},
    }


def _aggregate_crafter_mutations(audit_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    mutation_counts: Counter = Counter()
    for r in audit_rows:
        if (r.get("kind") or "") != "crafter_mutation":
            continue
        for tag in r.get("mutation_tags") or []:
            mutation_counts[str(tag)] += 1
        op = r.get("operation") or r.get("mutation_type")
        if op:
            mutation_counts[str(op)] += 1
    return dict(mutation_counts)


def _maybe_plot_curves(
    lifecycle: Dict[str, Any], out_path: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s) — skipping plot", exc)
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for status, points in lifecycle["cumulative_by_step"].items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, label=status, linewidth=1.6)
    ax.set_xlabel("trainer step")
    ax.set_ylabel("cumulative count")
    ax.set_title("Skill-bank dynamics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _maybe_plot_pie(mutations: Dict[str, int], out_path: Path) -> None:
    if not mutations:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s) — skipping pie", exc)
        return
    items = sorted(mutations.items(), key=lambda kv: -kv[1])[:10]
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Crafter mutation operations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = args.out_dir or (args.run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    lifecycle_rows = _read_jsonl(
        args.run_dir / "lifecycle_log" / "transitions.jsonl"
    )
    lifecycle = _aggregate_lifecycle(lifecycle_rows)

    audit_rows = _read_jsonl(args.run_dir / "audit.jsonl")
    mutations = _aggregate_crafter_mutations(audit_rows)

    summary = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "lifecycle": {
            k: v for k, v in lifecycle.items() if k != "cumulative_by_step"
        },
        "crafter_mutation_counts": mutations,
    }
    summary_path = out_dir / "skill_dynamics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", summary_path)

    if not args.no_plot:
        _maybe_plot_curves(
            lifecycle, out_dir / "skill_dynamics_curves.png"
        )
        _maybe_plot_pie(
            mutations, out_dir / "crafter_mutation_pie.png"
        )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
