"""Per-component runtime overhead plot (block E5).

Reads ``<run-dir>/runtime_log/component_timings.jsonl`` (one row per
component per trainer step) and produces:

* ``runtime_overhead_summary.json`` — per-component aggregate
  (calls, tokens, total ms, ms/call, tokens/call) plus % of wall-clock.
* ``runtime_overhead_bar.png``      — stacked / grouped bar of total
  ms per component (top axis) and tokens per component (bottom axis).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _aggregate(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"per_component": {}, "n_rows": 0}
    per_component: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "n_calls": 0,
            "total_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "n_steps_observed": 0,
        }
    )
    n_rows = 0
    seen_steps = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            comp = r.get("component") or "unknown"
            step = int(r.get("step") or 0)
            seen_steps[comp].add(step)
            bucket = per_component[comp]
            bucket["n_calls"] += int(r.get("n_calls") or 0)
            bucket["total_ms"] += float(r.get("total_ms") or 0.0)
            bucket["prompt_tokens"] += int(r.get("prompt_tokens") or 0)
            bucket["completion_tokens"] += int(r.get("completion_tokens") or 0)
            n_rows += 1

    for comp, bucket in per_component.items():
        bucket["n_steps_observed"] = len(seen_steps[comp])
        n = bucket["n_calls"] or 1
        bucket["ms_per_call"] = bucket["total_ms"] / n
        bucket["prompt_tokens_per_call"] = bucket["prompt_tokens"] / n
        bucket["completion_tokens_per_call"] = bucket["completion_tokens"] / n

    total_ms = sum(b["total_ms"] for b in per_component.values()) or 1.0
    for bucket in per_component.values():
        bucket["pct_of_total_ms"] = 100.0 * bucket["total_ms"] / total_ms

    return {
        "per_component": dict(per_component),
        "n_rows": n_rows,
        "total_ms": total_ms,
    }


def _maybe_plot(summary: Dict[str, Any], out_path: Path) -> None:
    per = summary.get("per_component") or {}
    if not per:
        logger.info("no per-component runtime rows — skipping plot")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s)", exc)
        return
    items = sorted(per.items(), key=lambda kv: -kv[1]["total_ms"])
    components = [c for c, _ in items]
    total_ms = [b["total_ms"] / 1000.0 for _, b in items]
    tok_total = [b["prompt_tokens"] + b["completion_tokens"] for _, b in items]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].bar(components, total_ms, color="#3a86ff", edgecolor="white")
    axes[0].set_ylabel("total wall (s)")
    axes[0].set_title("Per-component runtime (cumulative)")
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(components, tok_total, color="#fb5607", edgecolor="white")
    axes[1].set_ylabel("total tokens")
    axes[1].set_xlabel("component")
    axes[1].grid(alpha=0.25, axis="y")
    plt.setp(axes[1].get_xticklabels(), rotation=35, ha="right")

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

    summary = _aggregate(args.run_dir / "runtime_log" / "component_timings.jsonl")
    summary_path = out_dir / "runtime_overhead_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s (%d components)", summary_path, len(summary["per_component"]))

    if not args.no_plot:
        _maybe_plot(summary, out_dir / "runtime_overhead_bar.png")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
