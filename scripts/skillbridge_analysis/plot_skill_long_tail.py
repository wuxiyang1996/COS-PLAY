"""Per-skill retrieval frequency long-tail plot (block E2).

Counts how many times each ``skill_id`` was successfully validated
(``ok=true`` rows) in ``<run-dir>/harness_log/validate.jsonl`` and
plots the resulting long-tail distribution (head log-log).

Falls back to ``reward_log.jsonl`` (``chosen_skill_id`` field) when the
new instrumentation log is empty (e.g. for runs that pre-date Block A).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--include-rejected", action="store_true",
        help="Also count ``ok=False`` validate-events as retrievals.",
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _count_from_validate(
    path: Path, include_rejected: bool
) -> Counter:
    counts: Counter = Counter()
    for r in _iter_jsonl(path):
        sid = r.get("skill_id")
        if not sid:
            continue
        if not include_rejected and not r.get("ok"):
            continue
        counts[str(sid)] += 1
    return counts


def _count_from_reward_log(path: Path) -> Counter:
    counts: Counter = Counter()
    for r in _iter_jsonl(path):
        sid = r.get("chosen_skill_id") or r.get("skill_id")
        if not sid:
            continue
        counts[str(sid)] += 1
    return counts


def _maybe_plot(
    counts: Counter, out_path: Path, *, top_n: int = 200
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s) — skipping plot", exc)
        return
    if not counts:
        logger.info("no retrieval counts — nothing to plot")
        return
    items = counts.most_common(top_n)
    ranks = list(range(1, len(items) + 1))
    freqs = [c for _, c in items]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].bar(ranks, freqs, width=0.9, color="#3a86ff", edgecolor="none")
    axes[0].set_xlabel("skill rank (by retrieval count)")
    axes[0].set_ylabel("retrievals (validated)")
    axes[0].set_title("Skill retrieval long tail")
    axes[0].grid(alpha=0.25)

    axes[1].plot(ranks, freqs, marker=".", linestyle="-", color="#fb5607")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("rank (log)")
    axes[1].set_ylabel("retrievals (log)")
    axes[1].set_title("Long tail (log-log)")
    axes[1].grid(alpha=0.25, which="both")

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

    counts = _count_from_validate(
        args.run_dir / "harness_log" / "validate.jsonl",
        include_rejected=args.include_rejected,
    )
    source = "harness_validate"
    if not counts:
        # Fallback to legacy reward_log.
        counts = _count_from_reward_log(args.run_dir / "reward_log.jsonl")
        source = "reward_log"

    summary = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "source": source,
        "n_unique_skills": len(counts),
        "n_total_retrievals": int(sum(counts.values())),
        "top_50": counts.most_common(50),
        "include_rejected": bool(args.include_rejected),
    }
    summary_path = out_dir / "skill_long_tail_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s (n=%d unique, %d total)",
                summary_path,
                summary["n_unique_skills"],
                summary["n_total_retrievals"])

    if not args.no_plot:
        _maybe_plot(counts, out_dir / "skill_long_tail.png")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
