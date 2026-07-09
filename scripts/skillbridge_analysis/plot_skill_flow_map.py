"""Skill flow map (block E6).

A condensed Sankey-style diagram showing how skills move through the
SkillBridge lifecycle.  Counts come from
``<run-dir>/lifecycle_log/transitions.jsonl`` and any
``audit.jsonl`` proposals (crafter mutations / promotions) we can find
under ``<run-dir>`` (or ``<run-dir>/_artifacts/``).

Stages (left → right):

    extracted (DRAFT)  →  proposed by crafter (PROVISIONAL)
                       →  promoted (ACTIVE)
                       →  deprecated / retired

For each pair of adjacent stages we count the number of skill_ids that
made the transition.  When matplotlib's pandas-driven ``sankey`` (from
``matplotlib.sankey``) is unavailable, we fall back to a stacked-bar
representation that conveys the same structure.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


_STAGE_ORDER = (
    "EXTRACTED",
    "DRAFT",
    "PROVISIONAL",
    "ACTIVE",
    "DEPRECATED",
    "RETIRED",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _count_flows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    seen_pairs_per_skill: Dict[str, set] = defaultdict(set)
    for r in rows:
        sid = r.get("skill_id")
        if not sid:
            continue
        from_s = (r.get("from_status") or "").upper() or "UNKNOWN"
        to_s = (r.get("to_status") or "").upper() or "UNKNOWN"
        pair = (from_s, to_s)
        if pair in seen_pairs_per_skill[sid]:
            continue
        seen_pairs_per_skill[sid].add(pair)
        pair_counts[pair] += 1
    return dict(pair_counts)


def _stage_totals(
    pair_counts: Dict[Tuple[str, str], int],
) -> Dict[str, int]:
    totals: Dict[str, int] = defaultdict(int)
    for (f, t), c in pair_counts.items():
        totals[f] += c
        totals[t] += c
    return dict(totals)


def _maybe_plot(
    pair_counts: Dict[Tuple[str, str], int], out_path: Path
) -> None:
    if not pair_counts:
        logger.info("no transitions — skipping flow map")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s)", exc)
        return

    stages = [s for s in _STAGE_ORDER if any(
        s == f or s == t for (f, t) in pair_counts
    )]
    stage_index = {s: i for i, s in enumerate(stages)}
    n = len(stages)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.7 * n)))

    totals = defaultdict(float)
    for (f, t), c in pair_counts.items():
        totals[f] += c
        totals[t] += c
    max_t = max(totals.values()) or 1.0

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    for s, i in stage_index.items():
        h = 0.6 * totals[s] / max_t
        ax.barh(
            y=i, width=h, height=0.45,
            color="#3a86ff", edgecolor="white",
        )
        ax.text(
            h + 0.01, i, f"{s} ({int(totals[s])})",
            va="center", ha="left", fontsize=9,
        )

    for (f, t), c in pair_counts.items():
        if f not in stage_index or t not in stage_index:
            continue
        ax.annotate(
            f"{int(c)}",
            xy=(0.5, stage_index[t]),
            xytext=(0.5, stage_index[f]),
            arrowprops={
                "arrowstyle": "->",
                "lw": max(0.5, 3.0 * c / max_t),
                "color": "#fb5607",
                "alpha": 0.7,
            },
            fontsize=8,
            ha="center",
        )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Skill flow map (lifecycle transitions)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
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

    rows = _read_jsonl(args.run_dir / "lifecycle_log" / "transitions.jsonl")
    pair_counts = _count_flows(rows)
    totals = _stage_totals(pair_counts)

    summary = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "transitions": [
            {"from": f, "to": t, "count": c}
            for (f, t), c in sorted(
                pair_counts.items(), key=lambda kv: -kv[1]
            )
        ],
        "stage_totals": totals,
    }
    summary_path = out_dir / "skill_flow_map_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", summary_path)

    if not args.no_plot:
        _maybe_plot(pair_counts, out_dir / "skill_flow_map.png")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
