"""Few-shot scaling sweep driver (block D2).

For each target domain, evaluate the SkillBridge actor with k in
``{0, 1, 4, 16, 64}`` target-domain demonstrations.  We currently model
"k demonstrations" as **k pre-population episodes that the actor runs
before the held-out evaluation pass**, so the skill bank picks up
target-domain skills at the configured rate.  Per-k post-evaluation
results are aggregated into a single JSON.

Important: this driver re-uses the existing per-domain eval drivers,
launching them as subprocesses.  For ``k>0``, it also kicks off a short
"warm-up" run on the target domain (using ``run_phase1_curriculum``-
style invocations) to seed the bank with k episodes.  The warm-up step
is gated behind ``--enable-warmup``; when disabled, the sweep simply
reports the same eval over and over so users can wire in their own
warm-up policy.

Example::

    python -m scripts.skillbridge_eval.run_few_shot_sweep \\
        --run-dir runs/skillbridge_v12 \\
        --domain visual_reasoning \\
        --ks 0 1 4 16 64 \\
        --vllm-base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DRIVER_MODULE = {
    "browsergym":       "scripts.skillbridge_eval.eval_browsergym",
    "osworld":          "scripts.skillbridge_eval.eval_osworld",
    "visual_reasoning": "scripts.skillbridge_eval.eval_visual_reasoning",
    "video":            "scripts.skillbridge_eval.eval_video",
    "gymv":             "scripts.skillbridge_eval.eval_gymv",
}

_PRIMARY_PATH = {
    "browsergym":       ("overall", "success_rate_macro"),
    "osworld":          ("overall", "success_rate_macro"),
    "visual_reasoning": ("overall", "accuracy_micro"),
    "video":            ("overall", "accuracy_micro"),
    "gymv":             ("overall", "mean_reward_macro"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--domain", required=True, choices=list(_DRIVER_MODULE))
    p.add_argument(
        "--ks", type=int, nargs="+", default=[0, 1, 4, 16, 64],
        help="Sweep over these k values (target-domain demonstrations).",
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--vllm-base-url", type=str,
        default="http://localhost:8000/v1",
    )
    p.add_argument(
        "--enable-warmup", action="store_true",
        help="If set, run k target-domain warm-up episodes before each "
             "eval pass (writes to a per-k checkpoint inside "
             "<run-dir>/few_shot/k=<k>/).",
    )
    p.add_argument(
        "--warmup-cmd-template", type=str, default=None,
        help="Shell command template to run the k-shot warm-up. "
             "Substitutions: ``{k}``, ``{domain}``, ``{run_dir}``, "
             "``{model}``. Used only when --enable-warmup is set.",
    )
    p.add_argument("--episodes-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--vr-num-cases", type=int, default=200)
    p.add_argument("--gymv-games", nargs="+", default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _run_warmup(
    *,
    args: argparse.Namespace,
    k: int,
) -> int:
    if not args.enable_warmup or k == 0 or not args.warmup_cmd_template:
        return 0
    cmd = args.warmup_cmd_template.format(
        k=k,
        domain=args.domain,
        run_dir=str(args.run_dir),
        model=args.model,
    )
    logger.info("few-shot warm-up k=%d: $ %s", k, cmd)
    return subprocess.run(cmd, shell=True).returncode


def _run_eval(*, args: argparse.Namespace, k: int) -> Optional[Path]:
    module = _DRIVER_MODULE[args.domain]
    eval_dir = args.run_dir / "eval" / "few_shot" / args.domain
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"k={k}_{ts}.json"

    cmd = [
        sys.executable, "-m", module,
        "--run-dir", str(args.run_dir),
        "--model", args.model,
        "--vllm-base-url", args.vllm_base_url,
        "--label", f"few_shot:k={k}",
        "--output", str(out_path),
    ]
    if args.domain in ("browsergym", "osworld"):
        cmd += [
            "--episodes-per-task", str(args.episodes_per_task),
            "--max-steps", str(args.max_steps),
        ]
    if args.domain in ("visual_reasoning", "video"):
        cmd += ["--num-test-cases", str(args.vr_num_cases)]
    if args.domain == "gymv":
        cmd += [
            "--episodes-per-game", str(args.episodes_per_task),
            "--max-steps", str(args.max_steps),
        ]
        if args.gymv_games:
            cmd += ["--games", *args.gymv_games]

    env = os.environ.copy()
    env["VLLM_BASE_URL"] = args.vllm_base_url
    env["SKILLBRIDGE_FEWSHOT_K"] = str(k)

    logger.info("few-shot k=%d eval: $ %s", k, " ".join(cmd))
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        logger.warning("few-shot eval k=%d exit=%d", k, rc)
    return out_path if out_path.exists() else None


def _read_primary(path: Path, domain: str) -> Optional[float]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not parse %s: %s", path, exc)
        return None
    bucket, key = _PRIMARY_PATH[domain]
    val = (data.get(bucket) or {}).get(key)
    return float(val) if isinstance(val, (int, float)) else None


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.run_dir.exists():
        logger.error("run-dir %s missing", args.run_dir)
        return 1

    sweep: List[Dict[str, Any]] = []
    for k in args.ks:
        warm_rc = _run_warmup(args=args, k=k)
        result_path = _run_eval(args=args, k=k)
        primary = _read_primary(result_path, args.domain) if result_path else None
        sweep.append({
            "k": k,
            "warmup_returncode": warm_rc,
            "result_path": str(result_path) if result_path else None,
            "primary": primary,
        })

    out_path = args.output or (
        args.run_dir / "eval" / f"few_shot_{args.domain}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "domain": args.domain,
                "run_dir": str(args.run_dir),
                "model": args.model,
                "ks": args.ks,
                "sweep": sweep,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("wrote few-shot sweep %s", out_path)

    print(f"\n=== few-shot sweep ({args.domain}) ===")
    print(f"{'k':>4} | {'primary':>10}")
    print("-" * 18)
    for entry in sweep:
        v = entry["primary"]
        cell = f"{v:.4f}" if v is not None else "—"
        print(f"{entry['k']:>4} | {cell:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
