#!/usr/bin/env python3
"""Aggregate R10 candy_crush state-summary noise sweep results.

Reads each cell's rollout_summary.json + cell_meta.json from the sweep
output dir and produces summary.json + summary.md with per-cell
mean ± 95% CI (per-episode, n=16, Student-t df=15), Δ vs N0 control.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


# Student-t critical value at alpha=0.025, df=n-1
T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
           18: 2.101, 19: 2.093, 20: 2.086}


def t_crit(df: int) -> float:
    if df <= 0:
        return 0.0
    if df in T_TABLE:
        return T_TABLE[df]
    return 1.96  # large-sample fallback


def ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    m = mean(values)
    if n == 1:
        return (m, 0.0)
    s = stdev(values)
    se = s / math.sqrt(n)
    return (m, t_crit(n - 1) * se)


def find_rollout(cell_dir: Path) -> Optional[Path]:
    matches = list(cell_dir.rglob("rollout_summary.json"))
    if not matches:
        return None
    return matches[0]


def load_cell(cell_dir: Path) -> Optional[Dict]:
    meta_path = cell_dir / "cell_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    rs = find_rollout(cell_dir)
    if rs is None:
        meta["status"] = "missing_rollout"
        return meta
    d = json.loads(rs.read_text())
    rewards = [s["total_reward"] for s in d.get("episode_stats", []) if "total_reward" in s]
    if not rewards:
        meta["status"] = "no_episode_stats"
        return meta
    m, ci = ci95(rewards)
    n_steps = [s.get("steps") for s in d.get("episode_stats", []) if s.get("steps") is not None]
    meta.update({
        "status": "ok",
        "n_episodes": len(rewards),
        "rewards": rewards,
        "mean_reward": m,
        "ci95": ci,
        "std_reward": stdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_steps": mean(n_steps) if n_steps else None,
        "rollout_summary_path": str(rs),
    })
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()
    root = Path(args.output_root)

    cell_order = ["N0", "N2", "N3", "F-board", "NUM"]
    cells: List[Dict] = []
    for name in cell_order:
        d = root / name
        if not d.exists():
            continue
        info = load_cell(d)
        if info is None:
            continue
        info["cell"] = name
        cells.append(info)

    control: Optional[Dict] = next((c for c in cells if c["cell"] == "N0" and c.get("status") == "ok"), None)

    summary_json = {
        "experiment": "R10_candy_crush_summary_noise",
        "control_cell": "N0",
        "cells": cells,
    }
    (root / "summary.json").write_text(json.dumps(summary_json, indent=2))

    md = ["# R10 — Candy Crush State-Summary Noise Robustness",
          "",
          "Each cell evaluates the paper's best `candy_crush` adapter on 16 episodes "
          "with perturbed `summary_state`. Perturbations are injected by env var "
          "`COSPLAY_SUMMARY_NOISE` consumed by `decision_agents.agent_helper:build_rag_summary`. "
          "CI is Student-t 95 % over per-episode rewards (n=16, df=15, t=2.131).",
          "",
          "| Cell | Noise spec | Mean ± 95 % CI | Δ vs N0 | std | [min, max] | mean_steps | wall(s) |",
          "|---|---|---:|---:|---:|---|---:|---:|"]
    for c in cells:
        if c.get("status") != "ok":
            md.append(f"| {c['cell']} | `{c.get('noise_spec','')}` | FAILED ({c.get('status','?')}) | — | — | — | — | — |")
            continue
        m = c["mean_reward"]
        ci = c["ci95"]
        delta = ""
        if control is not None and c["cell"] != control["cell"]:
            dm = m - control["mean_reward"]
            pct = 100.0 * dm / control["mean_reward"] if control["mean_reward"] else 0.0
            delta = f"{dm:+.2f} ({pct:+.1f}%)"
        elif c["cell"] == "N0":
            delta = "—"
        md.append(
            f"| **{c['cell']}** | `{c.get('noise_spec','')}` | "
            f"{m:.2f} ± {ci:.2f} | {delta} | "
            f"{c['std_reward']:.2f} | "
            f"[{c['min_reward']:.0f}, {c['max_reward']:.0f}] | "
            f"{c['mean_steps']:.1f if c['mean_steps'] is not None else 'NA'} | "
            f"{c.get('cell_wall_s','?')} |"
        )
    md.append("")
    md.append("## Interpretation guide")
    md.append("- Flat (Δ within CI) under dropout p=0.25 → agent robust to moderate noise.")
    md.append("- Single-field drops larger than dropout-25 → that field is essential.")
    md.append("- NUM noise small Δ → agent uses ordinal/relative signals, not exact numerics.")
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[r10-aggregate] wrote {root/'summary.json'} and {root/'summary.md'}")
    for c in cells:
        if c.get("status") == "ok":
            print(f"  {c['cell']:>8s}  mean={c['mean_reward']:7.2f}  ci95=±{c['ci95']:6.2f}  n={c['n_episodes']}")
        else:
            print(f"  {c['cell']:>8s}  {c.get('status','?')}")


if __name__ == "__main__":
    main()
