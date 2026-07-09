"""Video held-out evaluation driver (block C5).

Wraps :mod:`cold_start.generate_cold_start_actor_visual_reasoning`
restricted to the two video benchmarks (``video_holmes`` and
``siv_bench``).  Reuses the same vision -> schema -> actor pipeline
under the hood and re-emits the cold-start ``batch_summary.json`` in
the SkillBridge uniform schema.
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
from typing import Any, Dict

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


_BENCHMARKS = ("video_holmes", "siv_bench")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--benchmarks", nargs="+", default=list(_BENCHMARKS),
        help=f"Video benchmark subset (default: {_BENCHMARKS}).",
    )
    p.add_argument("--num-test-cases", type=int, default=200)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--vllm-base-url", type=str,
        default="http://localhost:8000/v1",
    )
    p.add_argument(
        "--sample-ids-dir", type=Path, default=None,
    )
    p.add_argument("--label", type=str, default="skillbridge")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--cold-start-extra", nargs="*", default=[])
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _build_cmd(*, args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(
            REPO_ROOT
            / "cold_start"
            / "generate_cold_start_actor_visual_reasoning.py"
        ),
        "--benchmarks", *args.benchmarks,
        "--num_test_cases", str(args.num_test_cases),
        "--num_frames", str(args.num_frames),
        "--num_workers", str(args.num_workers),
        "--model", args.model,
        "--output_dir", str(out_dir),
    ]
    if args.sample_ids_dir:
        cmd += ["--sample_ids_dir", str(args.sample_ids_dir)]
    cmd.extend(args.cold_start_extra)
    return cmd


def _aggregate(out_dir: Path) -> Dict[str, Any]:
    master = out_dir / "batch_summary.json"
    per_bench: Dict[str, Dict[str, Any]] = {}

    if not master.exists():
        logger.warning("no batch_summary.json under %s", out_dir)
        return {"per_benchmark": per_bench, "overall": {}}

    try:
        data = json.loads(master.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse %s: %s", master, exc)
        return {"per_benchmark": per_bench, "overall": {}}

    for entry in data.get("per_benchmark", []) or []:
        name = entry.get("benchmark") or entry.get("name")
        if not name:
            continue
        attempted = int(entry.get("samples_attempted", 0))
        completed = int(entry.get("samples_completed", 0))
        correct = entry.get("correct_ok") or 0
        with_gold = entry.get("correct_total_with_gold") or 0
        per_bench[name] = {
            "samples_attempted": attempted,
            "samples_completed": completed,
            "correct_ok": int(correct),
            "correct_total_with_gold": int(with_gold),
            "accuracy": (correct / with_gold) if with_gold else None,
            "num_frames": entry.get("num_frames"),
        }

    micro_correct = sum(p["correct_ok"] for p in per_bench.values())
    micro_total = sum(p["correct_total_with_gold"] for p in per_bench.values())
    accs = [p["accuracy"] for p in per_bench.values() if p["accuracy"] is not None]
    overall = {
        "n_benchmarks": len(per_bench),
        "samples_completed_total": sum(
            p["samples_completed"] for p in per_bench.values()
        ),
        "accuracy_micro": (micro_correct / micro_total) if micro_total else None,
        "accuracy_macro": (sum(accs) / len(accs)) if accs else None,
    }
    return {"per_benchmark": per_bench, "overall": overall}


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
    out_dir = args.run_dir / "eval" / f"video_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_cmd(args=args, out_dir=out_dir)
    env = os.environ.copy()
    env["VLLM_BASE_URL"] = args.vllm_base_url
    env.setdefault(
        "VLLM_BASE_URL_MAP",
        json.dumps({args.model: args.vllm_base_url}),
    )

    logger.info("eval_video: $ %s", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    wall = time.time() - t0
    logger.info("eval_video: cold-start exit=%d wall=%.1fs",
                proc.returncode, wall)

    agg = _aggregate(out_dir)
    result = {
        "schema_version": 1,
        "domain": "video",
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
        args.run_dir / "eval" / f"video_result_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("eval_video: wrote %s", out_path)

    print("\n=== video eval summary ===")
    print(f"label    : {args.label}")
    print(f"benches  : {result['overall'].get('n_benchmarks', 0)}")
    print(f"acc μ    : {result['overall'].get('accuracy_micro')}")
    print(f"acc macro: {result['overall'].get('accuracy_macro')}")
    print(f"out      : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
