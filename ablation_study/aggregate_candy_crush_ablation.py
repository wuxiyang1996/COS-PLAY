#!/usr/bin/env python3
"""Aggregate candy_crush ablation cells into a single comparison table.

For each cell directory under ``--output-root`` this script reads:
    - ``ablation_meta.json``      (env vars + cell description)
    - ``run/step_log.jsonl``      (per-step metrics emitted by the
                                   co-evolution orchestrator)

and produces:
    - A wide ``summary.json`` keyed by cell name.
    - A printed Markdown table written to stdout AND saved as
      ``summary.md`` next to ``summary.json``.

The numbers reported per cell are:
    - mean candy_crush reward over the **last K=3 training steps**
      (with paper STEPS=10 this is the natural converged window).
    - 95 % bootstrap CI on that last-K mean.
    - peak candy_crush reward across the entire run.
    - final skill-bank size + total new skills minted.
    - total wall-clock time (s).

Usage:
    python ablation_study/aggregate_candy_crush_ablation.py \\
        --output-root ablation_study/output/candy_crush_sweep_<ts> \\
        --summary    ablation_study/output/candy_crush_sweep_<ts>/summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_LAST_K = 3
_BOOTSTRAP_N = 2_000
_GAME = "candy_crush"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _bootstrap_ci(values: List[float], n: int = _BOOTSTRAP_N) -> float:
    if len(values) < 2:
        return 0.0
    rng = random.Random(0xC0DE)
    means: List[float] = []
    for _ in range(n):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (hi - lo) / 2.0


def _candy_reward_per_step(step_records: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for rec in step_records:
        per_game = rec.get("reward_per_game", {})
        game = per_game.get(_GAME)
        if isinstance(game, dict) and "mean_reward" in game:
            out.append(float(game["mean_reward"]))
        elif "mean_reward" in rec and len(per_game) == 1:
            out.append(float(rec["mean_reward"]))
    return out


def _summarize_cell(cell_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = cell_dir / "ablation_meta.json"
    if not meta_path.exists():
        meta_path = next(cell_dir.rglob("ablation_meta.json"), None)
    step_log = cell_dir / "run" / "step_log.jsonl"
    if not step_log.exists():
        step_log = next(cell_dir.rglob("step_log.jsonl"), None)
    if step_log is None or not step_log.exists():
        return None

    meta: Dict[str, Any] = {}
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    records = _read_jsonl(step_log)
    rewards = _candy_reward_per_step(records)
    if not rewards:
        return None

    last_k = rewards[-_LAST_K:] if len(rewards) >= _LAST_K else rewards
    mean_last = statistics.mean(last_k)
    ci_last = _bootstrap_ci(last_k)
    peak = max(rewards)
    n_steps = len(rewards)
    final_skills = int(records[-1].get("n_skills", 0)) if records else 0
    total_new_skills = sum(int(r.get("n_new_skills", 0)) for r in records)
    total_wall = sum(float(r.get("wall_time_s", 0.0)) for r in records)

    return {
        "cell": meta.get("cell", cell_dir.name),
        "description": meta.get("description", ""),
        "env": meta.get("env", {}),
        "n_steps": n_steps,
        "mean_reward_last_k": mean_last,
        "ci95_last_k": ci_last,
        "peak_reward": peak,
        "final_skills": final_skills,
        "total_new_skills": total_new_skills,
        "total_wall_time_s": total_wall,
    }


def _render_markdown(cells: List[Dict[str, Any]],
                     control: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(
        "| Cell | Description | n_steps | Last-{k} reward (±95% CI) | "
        "Δ vs A0 | Peak | Final skills | Wall (s) |".format(k=_LAST_K)
    )
    lines.append(
        "|------|-------------|---------|--------------------------"
        "|---------|------|--------------|----------|"
    )
    base = control["mean_reward_last_k"] if control else None
    for c in cells:
        delta_str = "—"
        if base is not None and c["cell"] != "A0":
            d = c["mean_reward_last_k"] - base
            pct = (d / base * 100.0) if abs(base) > 1e-9 else float("nan")
            delta_str = f"{d:+.2f} ({pct:+.1f}%)"
        lines.append(
            "| {cell} | {desc} | {n} | {mean:.2f} ± {ci:.2f} | {dlt} | "
            "{peak:.2f} | {sk} | {wall:.0f} |".format(
                cell=c["cell"], desc=c["description"], n=c["n_steps"],
                mean=c["mean_reward_last_k"], ci=c["ci95_last_k"],
                dlt=delta_str, peak=c["peak_reward"],
                sk=c["final_skills"], wall=c["total_wall_time_s"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True,
                   help="Directory containing per-cell sub-directories.")
    p.add_argument("--summary", default=None,
                   help="Where to write the JSON summary.")
    args = p.parse_args()

    root = Path(args.output_root).resolve()
    if not root.exists():
        print(f"[ERROR] output-root not found: {root}", file=sys.stderr)
        return 1

    cells: List[Dict[str, Any]] = []
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        summary = _summarize_cell(cell_dir)
        if summary is None:
            print(f"[WARN] no step_log.jsonl under {cell_dir.name}, skipping",
                  file=sys.stderr)
            continue
        cells.append(summary)

    if not cells:
        print("[ERROR] No cells with usable logs found.", file=sys.stderr)
        return 2

    cells.sort(key=lambda c: c["cell"])
    control = next((c for c in cells if c["cell"] == "A0"), None)
    md = _render_markdown(cells, control)
    print(md)

    summary_path = Path(args.summary) if args.summary else root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"control_cell": "A0", "cells": cells}, indent=2),
        encoding="utf-8",
    )
    (summary_path.with_suffix(".md")).write_text(md + "\n", encoding="utf-8")
    print(f"\n[aggregator] JSON  → {summary_path}", file=sys.stderr)
    print(f"[aggregator] Markdown → {summary_path.with_suffix('.md')}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
