"""Evaluate a co-evolution checkpoint on MMLU-Pro and Math-500.

Measures whether game-RL fine-tuning causes catastrophic forgetting on
general reasoning benchmarks.  Runs two conditions:

  1. **Baseline** – vanilla Qwen3-8B (no adapter)
  2. **Adapter**  – Qwen3-8B + the best LoRA adapter from a training run

Both use the vLLM backend in lm-evaluation-harness for fast inference on
multi-GPU nodes.

Usage (from repo root, inside the game-ai-agent conda env):

    # Full run — Candy Crush best checkpoint, action_taking adapter
    python -m inference.run_academic_benchmarks

    # Quick smoke-test (5 samples per task)
    python -m inference.run_academic_benchmarks --limit 5

    # Custom adapter / run
    python -m inference.run_academic_benchmarks \
        --adapter-path runs/<run>/best/adapters/decision/action_taking \
        --run-tag my_experiment

    # Skip baseline (only run the adapter condition)
    python -m inference.run_academic_benchmarks --skip-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _pin_gpu(gpu_id: int) -> None:
    """Pin this process and all children to a single physical GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = (
    REPO_ROOT
    / "runs"
    / "Qwen3-8B_20260321_213813_(Candy_crush)"
    / "best"
    / "adapters"
    / "decision"
    / "action_taking"
)
BASE_MODEL = "Qwen/Qwen3-8B"
TASKS = ["mmlu_pro", "minerva_math500"]


def _run_eval(
    *,
    model_args: str,
    tasks: list[str],
    label: str,
    output_dir: Path,
    batch_size: str = "auto",
    limit: int | None = None,
):
    """Run lm_eval.simple_evaluate and persist results."""
    import lm_eval

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  model_args : {model_args}")
    print(f"  tasks      : {', '.join(tasks)}")
    print(f"{'=' * 70}\n")

    kwargs: dict = dict(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=True,
    )
    if limit is not None:
        kwargs["limit"] = limit

    t0 = time.time()
    results = lm_eval.simple_evaluate(**kwargs)
    elapsed = time.time() - t0

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    print(f"\n{'─' * 70}")
    print(f"  Results for: {label}  ({elapsed:.0f}s)")
    print(f"{'─' * 70}")
    for task_name, metrics in sorted(results["results"].items()):
        summary[task_name] = {}
        print(f"\n  {task_name}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                summary[task_name][k] = v
                if "stderr" not in k:
                    fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                    print(f"    {k:40s} {fmt}")
    print()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    full = {
        "label": label,
        "model_args": model_args,
        "elapsed_s": elapsed,
        "results": results.get("results", {}),
        "n-shot": results.get("n-shot", {}),
    }
    (output_dir / "full_results.json").write_text(
        json.dumps(full, indent=2, default=str)
    )
    return summary


def _print_comparison(baseline: dict, adapter: dict):
    """Side-by-side delta table."""
    print(f"\n{'=' * 70}")
    print("  Comparison (adapter − baseline)")
    print(f"{'=' * 70}")
    all_tasks = sorted(set(baseline) | set(adapter))
    for task in all_tasks:
        bm = baseline.get(task, {})
        am = adapter.get(task, {})
        print(f"\n  {task}:")
        all_keys = sorted(set(bm) | set(am))
        for k in all_keys:
            if "stderr" in k:
                continue
            bv = bm.get(k)
            av = am.get(k)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                delta = av - bv
                sign = "+" if delta >= 0 else ""
                print(f"    {k:40s}  base={bv:.4f}  adapter={av:.4f}  Δ={sign}{delta:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MMLU-Pro & Math-500 eval for game-RL checkpoints",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(DEFAULT_ADAPTER),
        help="Path to the LoRA adapter directory (adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--tasks", nargs="+", default=TASKS,
        help="lm-eval task names (default: mmlu_pro minerva_math500)",
    )
    parser.add_argument("--gpu", type=int, default=7, help="Physical GPU index to use")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor-parallel size for vLLM")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per task (for testing)")
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "eval_results" / "academic_benchmarks"),
    )
    parser.add_argument("--run-tag", default="candy_crush_best", help="Subdirectory tag")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-adapter", action="store_true")
    args = parser.parse_args()

    _pin_gpu(args.gpu)
    print(f"Pinned to physical GPU {args.gpu}")

    adapter_path = Path(args.adapter_path)
    if not args.skip_adapter and not (adapter_path / "adapter_config.json").exists():
        parser.error(f"adapter_config.json not found in {adapter_path}")

    out_root = Path(args.output_dir) / args.run_tag
    common_vllm = (
        f"pretrained={args.base_model},"
        f"tensor_parallel_size={args.tp_size},"
        "dtype=bfloat16,"
        "gpu_memory_utilization=0.85,"
        "max_model_len=4096,"
        "enable_thinking=False"
    )

    baseline_summary = None
    adapter_summary = None

    if not args.skip_baseline:
        baseline_summary = _run_eval(
            model_args=common_vllm,
            tasks=args.tasks,
            label=f"Baseline: {args.base_model}",
            output_dir=out_root / "baseline",
            batch_size=args.batch_size,
            limit=args.limit,
        )

    if not args.skip_adapter:
        adapter_vllm = (
            f"{common_vllm},"
            f"lora_local_path={adapter_path},"
            f"max_lora_rank=16"
        )
        adapter_summary = _run_eval(
            model_args=adapter_vllm,
            tasks=args.tasks,
            label=f"Adapter: {adapter_path.parent.name}/{adapter_path.name}",
            output_dir=out_root / "adapter_action_taking",
            batch_size=args.batch_size,
            limit=args.limit,
        )

    if baseline_summary and adapter_summary:
        _print_comparison(baseline_summary, adapter_summary)

    print(f"All results saved under: {out_root}")


if __name__ == "__main__":
    main()
