"""OSWorld held-out evaluation driver (block C3).

Thin wrapper around :mod:`cold_start.generate_cold_start_actor_osworld`.
Identical strategy to :mod:`scripts.skillbridge_eval.eval_browsergym`:
launch the cold-start subprocess pointing at a vLLM endpoint that has
the SkillBridge LoRAs pre-loaded, then read the resulting
``batch_rollout_summary.json`` and roll it up.

Note: OSWorld requires a virtualised desktop provider (vmware/docker/
azure) — the same provider env vars expected by the cold-start script
must be set before invoking this driver.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--task-catalog", type=Path, default=None,
        help="OSWorld test_*.json catalog. Defaults to whatever the "
             "cold-start script defaults to (typically test_small.json).",
    )
    p.add_argument(
        "--domains", nargs="+", default=None,
        help="Restrict to specific OSWorld domains.",
    )
    p.add_argument(
        "--task-ids", nargs="+", default=None,
        help="Restrict to specific UUID-string task ids.",
    )
    p.add_argument(
        "--tasks-per-domain", type=int, default=None,
        help="Max tasks per domain (smoke-eval knob).",
    )
    p.add_argument("--episodes-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--vllm-base-url", type=str,
        default="http://localhost:8000/v1",
    )
    p.add_argument("--label", type=str, default="skillbridge")
    p.add_argument(
        "--provider-name", type=str, default=None,
        help="OSWorld desktop provider (vmware/docker/azure/...).",
    )
    p.add_argument("--os-type", type=str, default=None)
    p.add_argument(
        "--no-som", action="store_true",
        help="Disable Set-of-Marks. Set ONLY for the raw-pixel ablation.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
    )
    p.add_argument(
        "--cold-start-extra", nargs="*", default=[],
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _build_cmd(
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cold_start" / "generate_cold_start_actor_osworld.py"),
        "--episodes", str(args.episodes_per_task),
        "--max_steps", str(args.max_steps),
        "--model", args.model,
        "--output_dir", str(out_dir),
    ]
    if args.task_catalog:
        cmd += ["--task_catalog", str(args.task_catalog)]
    if args.domains:
        cmd += ["--domains", *args.domains]
    if args.task_ids:
        cmd += ["--task_ids", *args.task_ids]
    if args.tasks_per_domain is not None:
        cmd += ["--tasks_per_domain", str(args.tasks_per_domain)]
    if args.provider_name:
        cmd += ["--provider_name", args.provider_name]
    if args.os_type:
        cmd += ["--os_type", args.os_type]
    if args.no_som:
        cmd += ["--no_som"]
    cmd.extend(args.cold_start_extra)
    return cmd


def _aggregate(out_dir: Path) -> Dict[str, Any]:
    master = out_dir / "batch_rollout_summary.json"
    per_task: Dict[str, Dict[str, Any]] = {}
    if not master.exists():
        logger.warning("no batch_rollout_summary.json under %s", out_dir)
        return {"per_task": per_task, "overall": {}}

    try:
        data = json.loads(master.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse %s: %s", master, exc)
        return {"per_task": per_task, "overall": {}}

    for entry in data.get("per_task_summaries", []) or []:
        key = entry.get("task_id") or entry.get("task") or entry.get("id")
        if not key:
            continue
        per_task[key] = {
            "domain": entry.get("domain"),
            "n_episodes": int(entry.get("completed_episodes", 0)),
            "success_rate": float(entry.get("success_rate", 0.0)),
            "mean_reward": float(entry.get("mean_reward", 0.0)),
            "mean_eval_score": float(entry.get("mean_eval_score", 0.0)),
            "mean_steps": float(entry.get("mean_steps", 0.0)),
            "skipped": bool(entry.get("skipped", False)),
        }

    valid = [p for p in per_task.values() if not p.get("skipped")]
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for p in valid:
        by_domain.setdefault(p["domain"] or "_unknown", []).append(p)

    overall = {
        "n_tasks": len(per_task),
        "n_tasks_completed": len(valid),
        "success_rate_macro": (
            sum(p["success_rate"] for p in valid) / len(valid)
            if valid else 0.0
        ),
        "mean_eval_score_macro": (
            sum(p["mean_eval_score"] for p in valid) / len(valid)
            if valid else 0.0
        ),
        "by_domain": {
            d: {
                "n_tasks": len(items),
                "success_rate": sum(i["success_rate"] for i in items) / len(items),
                "mean_eval_score": sum(i["mean_eval_score"] for i in items) / len(items),
            }
            for d, items in by_domain.items()
        },
    }
    return {"per_task": per_task, "overall": overall}


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.run_dir.exists():
        logger.error("run-dir %s missing", args.run_dir)
        return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.run_dir / "eval" / f"osworld_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_cmd(args=args, out_dir=out_dir)
    env = os.environ.copy()
    env["VLLM_BASE_URL"] = args.vllm_base_url
    env.setdefault(
        "VLLM_BASE_URL_MAP",
        json.dumps({args.model: args.vllm_base_url}),
    )

    logger.info("eval_osworld: $ %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    wall = time.time() - t0
    logger.info("eval_osworld: cold-start exit=%d wall=%.1fs",
                proc.returncode, wall)

    agg = _aggregate(out_dir)
    result = {
        "schema_version": 1,
        "domain": "osworld",
        "label": args.label,
        "run_dir": str(args.run_dir),
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "cold_start_returncode": int(proc.returncode),
        "out_dir": str(out_dir),
        "wall_seconds": round(wall, 2),
        **agg,
    }

    out_path = args.output or (
        args.run_dir / "eval" / f"osworld_result_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("eval_osworld: wrote %s", out_path)

    print("\n=== osworld eval summary ===")
    print(f"label   : {args.label}")
    print(f"tasks   : {result['overall'].get('n_tasks', 0)}")
    print(f"success : {result['overall'].get('success_rate_macro', 0.0):.2%}")
    print(f"score   : {result['overall'].get('mean_eval_score_macro', 0.0):.3f}")
    print(f"out     : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
