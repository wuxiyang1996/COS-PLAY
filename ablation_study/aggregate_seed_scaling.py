#!/usr/bin/env python3
"""Aggregate R6 seed-scaling sweep results.

For each N cell, reads:
  - ablation_meta.json   (n_seed_episodes, etc.)
  - run/step_log.jsonl   (step-0 mean_reward, n_skills, wall_time_s)

Outputs:
  summary.json     machine-readable
  summary.md       markdown scaling table for rebuttal
  curve.png        scaling curve (N → step-0 mean reward)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def collect_cell(cell_dir: Path) -> Optional[Dict]:
    meta = _read_json(cell_dir / "ablation_meta.json")
    if not meta:
        return None
    log = _read_jsonl(cell_dir / "run" / "step_log.jsonl")
    step0 = log[0] if log else {}
    return dict(
        cell=cell_dir.name,
        n_seed_episodes=meta.get("n_seed_episodes"),
        subsample_skipped=meta.get("subsample_skipped"),
        sft_skipped=meta.get("sft_skipped"),
        step0_mean_reward=step0.get("mean_reward"),
        step0_n_skills=step0.get("n_skills"),
        step0_wall_s=step0.get("wall_time_s"),
        eval_protocol=meta.get("eval_protocol"),
    )


def collect_cells(output_root: Path) -> List[Dict]:
    rows = []
    for cell_dir in sorted(output_root.glob("N*")):
        if not cell_dir.is_dir():
            continue
        if not (cell_dir / "ablation_meta.json").exists():
            continue
        row = collect_cell(cell_dir)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: r.get("n_seed_episodes") or 0)
    return rows


def build_markdown(cells: List[Dict]) -> str:
    lines = []
    lines.append("# R6 — Seed-Trajectory Scaling Curve (candy_crush)")
    lines.append("")
    lines.append("Each row trains the decision agent's 2 LoRA adapters")
    lines.append("(action_taking + skill_selection) on the first N GPT-5.4")
    lines.append("expert episodes, then runs a single rollout step")
    lines.append("(`--total-steps 1 --no-grpo`, 8 episodes) with the SFT")
    lines.append("skill-bank adapters held fixed at the 60-episode baseline.")
    lines.append("")
    lines.append("This isolates the decision agent's sensitivity to expert")
    lines.append("demonstration quantity (vJ13 Q5).")
    lines.append("")
    if not cells:
        lines.append("_(no cells aggregated yet)_")
        return "\n".join(lines)

    lines.append("| Cell | N seeds | rows AT / SS | step-0 mean reward | n_skills | wall(s) | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in cells:
        n = c.get("n_seed_episodes")
        rwd = c.get("step0_mean_reward")
        nsk = c.get("step0_n_skills")
        wall = c.get("step0_wall_s")
        rwd_s = f"{rwd:.3f}" if isinstance(rwd, (int, float)) else "—"
        nsk_s = f"{nsk}" if nsk is not None else "—"
        wall_s = f"{wall:.0f}" if isinstance(wall, (int, float)) else "—"
        rows = n * 50 if n else None  # ≈50 rows / episode
        rows_s = f"{rows} / {rows}" if rows else "—"
        note_bits = []
        if c.get("subsample_skipped"):
            note_bits.append("full src")
        if c.get("sft_skipped"):
            note_bits.append("reused SFT (60-ep baseline)")
        note = ", ".join(note_bits) or "subsampled + SFT-trained"
        lines.append(f"| {c['cell']} | {n} | {rows_s} | {rwd_s} | {nsk_s} | {wall_s} | {note} |")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("- **monotone increase**: more expert seeds → better SFT cold-start.")
    lines.append("- **plateau by N=40**: diminishing returns; ~40 episodes suffice.")
    lines.append("- **N=10 ≪ N=60**: method depends critically on expert data.")
    lines.append("- **N=10 ≈ N=60**: low data efficiency floor; SFT does not benefit "
                 "from more demos beyond a small set.")
    return "\n".join(lines)


def maybe_plot(cells: List[Dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped (matplotlib not available): {e}")
        return
    xs, ys = [], []
    for c in cells:
        n = c.get("n_seed_episodes")
        r = c.get("step0_mean_reward")
        if n is None or r is None:
            continue
        xs.append(n)
        ys.append(r)
    if not xs:
        print("[plot] no points to plot")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(xs, ys, marker="o", label="SFT step-0 mean reward")
    ax.set_xlabel("# expert seed episodes (N)")
    ax.set_ylabel("Candy Crush step-0 mean reward")
    ax.set_title("R6 — Seed-trajectory scaling on candy_crush")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    cells = collect_cells(args.output_root)

    summary = dict(experiment="R6_seed_scaling",
                   game="candy_crush",
                   cells=cells)
    summary_path = args.summary or args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[seed-aggregate] wrote {summary_path}")

    md_path = args.output_root / "summary.md"
    md_path.write_text(build_markdown(cells))
    print(f"[seed-aggregate] wrote {md_path}")

    if cells:
        maybe_plot(cells, args.output_root / "curve.png")


if __name__ == "__main__":
    main()
