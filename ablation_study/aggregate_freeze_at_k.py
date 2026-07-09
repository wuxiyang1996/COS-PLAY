#!/usr/bin/env python3
"""Aggregate freeze-at-iter-k sweep results.

For each freeze cell, compute:
  - reward_at_freeze   (from A0 ckpt metadata.json)
  - reward_after_continuation_last  (last step of the continuation run)
  - reward_after_continuation_best  (best step of the continuation run)
  - reward_delta = best_after - reward_at_freeze
  - n_skills_at_freeze, n_skills_final  (should be equal under freeze)
  - wall_time

Compare against A0 (full co-evolution, control) and the existing
fixed_skillbank_Qwen3-8B_candy_crush_* run (= freeze@iter=1 from
SFT cold-start).

Outputs:
  summary.json    -- machine-readable
  summary.md      -- markdown table for rebuttal
  curve.png       -- matplotlib plot (freeze_step vs final reward)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


def _read_step_log(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _summarise_run(step_log: List[Dict]) -> Dict[str, Optional[float]]:
    rewards = [r.get("mean_reward") for r in step_log if r.get("mean_reward") is not None]
    n_skills = [r.get("n_skills") for r in step_log if r.get("n_skills") is not None]
    if not rewards:
        return dict(n_steps=0, last_reward=None, best_reward=None,
                    mean_last3=None, n_skills_first=None, n_skills_last=None,
                    total_wall_s=None)
    return dict(
        n_steps=len(step_log),
        last_reward=rewards[-1],
        best_reward=max(rewards),
        mean_last3=statistics.mean(rewards[-3:]),
        n_skills_first=n_skills[0] if n_skills else None,
        n_skills_last=n_skills[-1] if n_skills else None,
        total_wall_s=sum(r.get("wall_time_s", 0) for r in step_log),
    )


def _read_ckpt_meta(ckpt_dir: Path) -> Dict:
    p = ckpt_dir / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _read_ablation_meta(cell_dir: Path) -> Dict:
    p = cell_dir / "ablation_meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def collect_cells(output_root: Path) -> List[Dict]:
    rows = []
    for cell_dir in sorted(output_root.glob("F*")):
        if not cell_dir.is_dir():
            continue
        meta = _read_ablation_meta(cell_dir)
        step_log = _read_step_log(cell_dir / "run" / "step_log.jsonl")
        summary = _summarise_run(step_log)
        rows.append(dict(
            cell=cell_dir.name,
            freeze_at_step=meta.get("freeze_at_step"),
            source_a0_run=meta.get("source_a0_run"),
            continuation_steps=meta.get("continuation_steps"),
            **summary,
        ))
    return rows


def collect_a0_control(a0_run: Path) -> Dict:
    step_log = _read_step_log(a0_run / "step_log.jsonl")
    summary = _summarise_run(step_log)
    # Per-step reward map for the curve
    per_step = {
        r["step"]: r["mean_reward"]
        for r in step_log if r.get("mean_reward") is not None
    }
    return dict(
        run="A0_control_full_coevolution",
        run_dir=str(a0_run),
        per_step=per_step,
        **summary,
    )


def collect_freeze_at_iter1(project_root: Path) -> Optional[Dict]:
    """The existing R3 candy_crush run = freeze at iter=1 from SFT cold-start."""
    candidates = sorted(
        (project_root / "runs").glob("fixed_skillbank_Qwen3-8B_candy_crush_*")
    )
    if not candidates:
        return None
    r = candidates[-1]
    step_log = _read_step_log(r / "step_log.jsonl")
    summary = _summarise_run(step_log)
    return dict(run="R3_freeze_at_iter1_from_SFT", run_dir=str(r), **summary)


def reward_at_freeze(cell: Dict, a0_per_step: Dict[int, float]) -> Optional[float]:
    k = cell.get("freeze_at_step")
    if k is None:
        return None
    return a0_per_step.get(k)


def build_markdown(cells: List[Dict], a0: Dict, r3: Optional[Dict]) -> str:
    a0_per_step = a0.get("per_step", {})
    lines = []
    lines.append("# Freeze-at-iter-k sweep — candy_crush")
    lines.append("")
    lines.append("Each cell loads ALL 5 LoRA adapters and the skill bank state from")
    lines.append("the A0 control's checkpoint at step K, then continues for "
                 f"{cells[0].get('continuation_steps') if cells else '?'} more steps "
                 "with `--freeze-skillbank` (decision LoRAs train, skill bank LoRAs + bank state frozen).")
    lines.append("")
    lines.append("**Reference (A0 = full co-evolution control, no freezing):**")
    lines.append(f"- 10-step run, best reward = `{a0['best_reward']:.2f}`, "
                 f"mean(last 3) = `{a0['mean_last3']:.2f}`, "
                 f"final n_skills = `{a0['n_skills_last']}`")
    if r3:
        lines.append(f"- R3 (freeze-at-iter=1 from SFT cold-start): "
                     f"best = `{r3['best_reward']:.2f}`, "
                     f"mean(last 3) = `{r3['mean_last3']:.2f}`")
    lines.append("")
    if not cells:
        lines.append("_(no freeze cells aggregated yet)_")
        return "\n".join(lines)
    lines.append("| Cell | Freeze@step | r@freeze | best after | last after | mean(last3) | Δ best-r@freeze | n_skills f→l | wall(s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        rfreeze = reward_at_freeze(c, a0_per_step)
        rfreeze_s = f"{rfreeze:.2f}" if rfreeze is not None else "—"
        delta = (c["best_reward"] - rfreeze) if (rfreeze is not None and c["best_reward"] is not None) else None
        delta_s = f"{delta:+.2f}" if delta is not None else "—"
        best_s = f"{c['best_reward']:.2f}" if c["best_reward"] is not None else "—"
        last_s = f"{c['last_reward']:.2f}" if c["last_reward"] is not None else "—"
        ml3_s = f"{c['mean_last3']:.2f}" if c["mean_last3"] is not None else "—"
        nsk = f"{c['n_skills_first']}→{c['n_skills_last']}"
        wall = f"{c['total_wall_s']:.0f}" if c["total_wall_s"] else "—"
        lines.append(f"| {c['cell']} | {c['freeze_at_step']} | {rfreeze_s} | {best_s} | {last_s} | {ml3_s} | {delta_s} | {nsk} | {wall} |")

    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("- **monotone increase F0→F9** → bank improves with iteration, no collapse / no cycle.")
    lines.append("- **F9 ≈ A0** → frozen-late ≈ full co-evolution; co-evolution converges.")
    lines.append("- **F0/F2 ≫ A0** ⚠ → early freeze beats co-evolution; co-evolution may be harmful.")
    lines.append("- **All Δ < 0** → decision-only fine-tune cannot improve once bank is frozen.")
    return "\n".join(lines)


def maybe_plot(cells: List[Dict], a0: Dict, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped (matplotlib not available): {e}")
        return
    xs, ys_best, ys_last, ys_ml3, ys_rfreeze = [], [], [], [], []
    a0_per_step = a0.get("per_step", {})
    for c in sorted(cells, key=lambda c: c.get("freeze_at_step") or 0):
        if c.get("freeze_at_step") is None:
            continue
        xs.append(c["freeze_at_step"])
        ys_best.append(c["best_reward"])
        ys_last.append(c["last_reward"])
        ys_ml3.append(c["mean_last3"])
        ys_rfreeze.append(a0_per_step.get(c["freeze_at_step"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, ys_best, marker="o", label="best after continuation")
    ax.plot(xs, ys_ml3,  marker="s", label="mean(last 3)")
    ax.plot(xs, ys_rfreeze, marker="x", linestyle="--",
            label="reward at freeze (no continuation)")
    ax.axhline(a0["best_reward"], color="gray", linestyle=":",
               label=f"A0 full co-evolution (best={a0['best_reward']:.0f})")
    ax.set_xlabel("Freeze point k (training step)")
    ax.set_ylabel("Candy Crush mean reward")
    ax.set_title("Freeze-at-iter-k vs full co-evolution")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--a0-run", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--project-root", type=Path,
                    default=Path("/workspace/Game-AI-Agent"))
    args = ap.parse_args()

    cells = collect_cells(args.output_root)
    a0    = collect_a0_control(args.a0_run)
    r3    = collect_freeze_at_iter1(args.project_root)

    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = dict(cells=cells, a0_control=a0, r3_baseline=r3)
    summary_path = args.summary or args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[freeze-aggregate] wrote {summary_path}")

    md_path = args.output_root / "summary.md"
    md_path.write_text(build_markdown(cells, a0, r3))
    print(f"[freeze-aggregate] wrote {md_path}")

    if cells:
        maybe_plot(cells, a0, args.output_root / "curve.png")
    else:
        print("[freeze-aggregate] no cells to plot yet")


if __name__ == "__main__":
    main()
