"""BrowserGym held-out evaluation driver (block C2).

Thin wrapper around :mod:`cold_start.generate_cold_start_actor_browsergym`:
the cold-start script already drives BrowserGym tasks with an arbitrary
``--model`` slug, so the SkillBridge eval just needs to point that slug
at a vLLM endpoint with the trained LoRA pre-loaded and translate the
per-task outputs into a uniform :file:`eval_result.json`.

Usage::

    python -m scripts.skillbridge_eval.eval_browsergym \\
        --run-dir runs/.../ \\
        --tasks-file cold_start/task_samples/browsergym_assistantbench_test_feasible.txt \\
        --episodes-per-task 1 \\
        --model Qwen/Qwen3.5-9B \\
        --vllm-base-url http://localhost:8000/v1 \\
        --output runs/.../eval/browsergym_result.json

The ``--vllm-base-url`` must point at a server where the LoRA adapter
``action_taking`` (and optionally ``skill_selection``) has been loaded
via vLLM's multi-LoRA module API.
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
        "--tasks-file", type=Path, default=None,
        help="Path to a newline-delimited BrowserGym env-id list. "
             "Defaults to assistantbench feasible.",
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        help="Inline list of BrowserGym env ids "
             "(e.g. 'browsergym/miniwob.click-button').",
    )
    p.add_argument("--episodes-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--vllm-base-url", type=str,
        default="http://localhost:8000/v1",
    )
    p.add_argument(
        "--label", type=str, default="skillbridge",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Defaults to "
             "<run-dir>/eval/browsergym_result_<ts>.json.",
    )
    p.add_argument(
        "--cold-start-extra", nargs="*", default=[],
        help="Pass-through args to generate_cold_start_actor_browsergym.py.",
    )
    p.add_argument(
        "--limit-tasks", type=int, default=None,
        help="If set, only run the first N tasks from --tasks-file. "
             "Used for quick smoke checks.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _load_tasks(args: argparse.Namespace) -> List[str]:
    if args.tasks:
        return list(args.tasks)
    if args.tasks_file:
        path = args.tasks_file
    else:
        path = (
            REPO_ROOT
            / "cold_start"
            / "task_samples"
            / "browsergym_assistantbench_test_feasible.txt"
        )
    if not path.exists():
        raise FileNotFoundError(f"tasks file {path} not found")
    tasks = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    return tasks


def _run_cold_start(
    *,
    args: argparse.Namespace,
    tasks: List[str],
    out_dir: Path,
) -> int:
    """Invoke the cold-start BrowserGym driver as a subprocess."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cold_start" / "generate_cold_start_actor_browsergym.py"),
        "--tasks", *tasks,
        "--episodes", str(args.episodes_per_task),
        "--max_steps", str(args.max_steps),
        "--model", args.model,
        "--output_dir", str(out_dir),
    ]
    cmd.extend(args.cold_start_extra)

    env = os.environ.copy()
    env["VLLM_BASE_URL"] = args.vllm_base_url
    # Cold-start scripts route through API_func.ask_vllm; the
    # VLLM_BASE_URL_MAP env knob is checked first.  We pin every model
    # we might call to the same endpoint so the trained LoRA always
    # serves the action_taking calls.
    env.setdefault(
        "VLLM_BASE_URL_MAP",
        json.dumps({args.model: args.vllm_base_url}),
    )

    logger.info("eval_browsergym: $ %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    wall = time.time() - t0
    logger.info("eval_browsergym: cold-start exit=%d wall=%.1fs",
                proc.returncode, wall)
    return int(proc.returncode)


def _aggregate_outputs(out_dir: Path) -> Dict[str, Any]:
    """Read the cold-start ``batch_rollout_summary.json`` and roll
    per-target stats up into a uniform schema."""
    master = out_dir / "batch_rollout_summary.json"
    per_task: Dict[str, Dict[str, Any]] = {}

    if master.exists():
        try:
            data = json.loads(master.read_text())
            for entry in data.get("per_target_summaries", []) or []:
                task = (
                    entry.get("env_id")
                    or entry.get("task")
                    or entry.get("url")
                    or entry.get("payload")
                )
                if not task:
                    continue
                per_task[task] = {
                    "n_episodes": int(entry.get("completed_episodes", 0)),
                    "success_rate": float(entry.get("success_rate", 0.0)),
                    "mean_reward": float(entry.get("mean_reward", 0.0)),
                    "mean_steps": float(entry.get("mean_steps", 0.0)),
                    "skipped": bool(entry.get("skipped", False)),
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to parse %s: %s", master, exc)

    if not per_task:
        for summary_path in out_dir.glob("**/rollout_summary.json"):
            try:
                data = json.loads(summary_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("skip %s: %s", summary_path, exc)
                continue
            task = (
                data.get("env_id")
                or data.get("task")
                or summary_path.parent.name
            )
            per_task[task] = {
                "n_episodes": int(data.get("completed_episodes", 0)),
                "success_rate": float(data.get("success_rate", 0.0)),
                "mean_reward": float(data.get("mean_reward", 0.0)),
                "mean_steps": float(data.get("mean_steps", 0.0)),
            }

    valid = [p for p in per_task.values() if not p.get("skipped")]
    overall = {
        "n_tasks": len(per_task),
        "n_tasks_completed": len(valid),
        "success_rate_macro": (
            sum(p["success_rate"] for p in valid) / len(valid)
            if valid else 0.0
        ),
        "mean_reward_macro": (
            sum(p["mean_reward"] for p in valid) / len(valid)
            if valid else 0.0
        ),
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

    try:
        tasks = _load_tasks(args)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.run_dir / "eval" / f"browsergym_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = _run_cold_start(args=args, tasks=tasks, out_dir=out_dir)
    agg = _aggregate_outputs(out_dir)

    result = {
        "schema_version": 1,
        "domain": "browsergym",
        "label": args.label,
        "run_dir": str(args.run_dir),
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "n_tasks": len(tasks),
        "episodes_per_task": args.episodes_per_task,
        "max_steps": args.max_steps,
        "cold_start_returncode": rc,
        "out_dir": str(out_dir),
        **agg,
    }

    out_path = args.output or (
        args.run_dir / "eval" / f"browsergym_result_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("eval_browsergym: wrote %s", out_path)

    print("\n=== browsergym eval summary ===")
    print(f"label    : {args.label}")
    print(f"tasks    : {len(tasks)}")
    print(f"sr_macro : {result['overall']['success_rate_macro']:.2%}")
    print(f"out      : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
