#!/usr/bin/env python3
"""
summarize_qwen_vllm_baselines.py — aggregate per-(model x env) stats produced
by ``baselines/run_qwen_vllm_baselines.sh``.

Reads every ``rollout_summary.json`` under
``<codebase_root>/qwen-baselines-out/<run_id>/<model_tag>/{env_wrappers,gymv}/<env>/``
and emits:

  - A flat per-row table printed to stdout (one row per model x env).
  - Aggregated mean reward / steps / completion rate per model.
  - A combined ``qwen_vllm_summary.json`` written into the run dir.

Usage:

    # Default: pick up "latest" run under <codebase_root>/qwen-baselines-out/
    python baselines/summarize_qwen_vllm_baselines.py

    # Specific run id
    python baselines/summarize_qwen_vllm_baselines.py \
        --run_dir <codebase_root>/qwen-baselines-out/2026-04-29_04-05-30

    # Custom output filename
    python baselines/summarize_qwen_vllm_baselines.py --out /tmp/qwen_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE = CODEBASE_ROOT / "qwen-baselines-out"


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _ci95(xs: Iterable[float]) -> float:
    """Half-width of a 95% normal-approx CI for the mean."""
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    return 1.96 * _std(xs) / math.sqrt(len(xs))


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] failed to read {path}: {exc}", file=sys.stderr)
        return None


def _collect_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Walk run_dir for per-env rollout_summary.json files."""
    rows: List[Dict[str, Any]] = []
    if not run_dir.is_dir():
        return rows

    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if model_dir.name.startswith("_"):
            continue
        model_tag = model_dir.name
        for kind_name in ("env_wrappers", "gymv"):
            kind_dir = model_dir / kind_name
            if not kind_dir.is_dir():
                continue
            for env_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
                summary_path = env_dir / "rollout_summary.json"
                if not summary_path.is_file():
                    # Some Python backends nest one level deeper (game subdir).
                    nested = list(env_dir.glob("**/rollout_summary.json"))
                    if not nested:
                        continue
                    summary_path = nested[0]
                summary = _load_summary(summary_path)
                if summary is None:
                    continue
                ep_stats = summary.get("episode_stats") or []
                rewards = [
                    float(s["total_reward"])
                    for s in ep_stats
                    if "error" not in s and "total_reward" in s
                ]
                steps = [
                    float(s["steps"])
                    for s in ep_stats
                    if "error" not in s and "steps" in s
                ]
                row = {
                    "model_tag": model_tag,
                    "kind": "env_wrappers" if kind_name == "env_wrappers" else "gymv",
                    "env": summary.get("game") or summary.get("env_id") or env_dir.name,
                    "target_episodes": summary.get("target_episodes"),
                    "completed_episodes": summary.get("completed_episodes", len(rewards)),
                    "mean_reward": _mean(rewards),
                    "max_reward": max(rewards) if rewards else 0.0,
                    "min_reward": min(rewards) if rewards else 0.0,
                    "std_reward": _std(rewards),
                    "ci95_reward": _ci95(rewards),
                    "mean_steps": _mean(steps),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "use_vision": summary.get("use_vision"),
                    "model": summary.get("model"),
                    "model_routed": summary.get("model_routed"),
                    "summary_path": str(summary_path),
                }
                rows.append(row)
    return rows


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rollout_summary.json files found)")
        return

    header = (
        f"{'model':<10} {'kind':<13} {'env':<32} "
        f"{'eps':>5} {'mean_R':>9} {'+-95%':>7} "
        f"{'min_R':>8} {'max_R':>8} {'mean_steps':>11}"
    )
    print(header)
    print("-" * len(header))
    rows_sorted = sorted(rows, key=lambda r: (r["model_tag"], r["kind"], r["env"]))
    for r in rows_sorted:
        eps = r.get("completed_episodes", 0) or 0
        print(
            f"{r['model_tag']:<10} "
            f"{r['kind']:<13} "
            f"{r['env'][:32]:<32} "
            f"{eps:>5} "
            f"{r['mean_reward']:>9.2f} "
            f"{r['ci95_reward']:>7.2f} "
            f"{r['min_reward']:>8.2f} "
            f"{r['max_reward']:>8.2f} "
            f"{r['mean_steps']:>11.1f}"
        )


def _aggregate_per_model(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model_tag"], []).append(r)
    out: List[Dict[str, Any]] = []
    for tag, lst in sorted(by_model.items()):
        means = [r["mean_reward"] for r in lst]
        steps = [r["mean_steps"] for r in lst if r["mean_steps"]]
        completed = sum(int(r.get("completed_episodes") or 0) for r in lst)
        out.append({
            "model_tag": tag,
            "n_envs": len(lst),
            "completed_episodes_total": completed,
            "macro_mean_reward": _mean(means),
            "macro_mean_steps": _mean(steps),
            "model": (lst[0].get("model") if lst else None),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Qwen vLLM baseline results across models / envs.",
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Run dir under <codebase_root>/qwen-baselines-out/. "
             "Default: <base>/latest (resolved via symlink).",
    )
    parser.add_argument(
        "--base_dir", type=str, default=str(DEFAULT_BASE),
        help=f"Base output dir (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Where to write the combined summary JSON. "
             "Default: <run_dir>/qwen_vllm_summary.json",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        latest = base / "latest"
        if latest.exists():
            run_dir = latest.resolve()
        else:
            run_dirs = sorted(
                (p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ) if base.is_dir() else []
            if not run_dirs:
                print(f"[ERROR] No run dir under {base}", file=sys.stderr)
                return 2
            run_dir = run_dirs[0]
    if not run_dir.is_dir():
        print(f"[ERROR] Run dir not found: {run_dir}", file=sys.stderr)
        return 2

    rows = _collect_rows(run_dir)
    print(f"Run dir:   {run_dir}")
    print(f"Found:     {len(rows)} (model x env) results")
    print()
    _print_table(rows)
    print()

    aggregates = _aggregate_per_model(rows)
    if aggregates:
        print("Per-model macro-averages (over envs):")
        for a in aggregates:
            print(
                f"  {a['model_tag']:<10}  envs={a['n_envs']:<3} "
                f"completed_eps={a['completed_episodes_total']:<5} "
                f"macro_mean_reward={a['macro_mean_reward']:>8.2f}  "
                f"macro_mean_steps={a['macro_mean_steps']:>7.1f}"
            )

    out_path = Path(args.out) if args.out else (run_dir / "qwen_vllm_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "n_results": len(rows),
                "results": rows,
                "per_model": aggregates,
            },
            f, indent=2, ensure_ascii=False, default=str,
        )
    print()
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
