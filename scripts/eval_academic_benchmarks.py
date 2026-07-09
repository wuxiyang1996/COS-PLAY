"""
Evaluate base Qwen3-8B and game-RL fine-tuned checkpoints on academic benchmarks
(MMLU-Pro, Math-500) to measure catastrophic forgetting / reasoning transfer.

Usage:
    # Baseline (no adapter)
    python scripts/eval_academic_benchmarks.py --mode base

    # With LoRA adapter from best checkpoint
    python scripts/eval_academic_benchmarks.py --mode adapter \
        --adapter-path runs/Qwen3-8B_diplomacy_20260322_234548/checkpoints/step_99999/adapters/decision/action_taking

    # Merge adapter and eval with vLLM (fastest)
    python scripts/eval_academic_benchmarks.py --mode merged \
        --adapter-path runs/Qwen3-8B_diplomacy_20260322_234548/checkpoints/step_99999/adapters/decision/action_taking
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

def merge_adapter(base_model: str, adapter_path: str, output_dir: str):
    """Merge a PEFT LoRA adapter into the base model and save."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.")
    return output_dir


def run_lm_eval(model_path: str, tasks: list[str], output_dir: str,
                backend: str = "vllm", tp_size: int = 1,
                batch_size: str = "auto", num_fewshot: int | None = None,
                limit: int | None = None):
    """Run lm-evaluation-harness."""
    import lm_eval

    task_str = ",".join(tasks)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Tasks: {task_str}")
    print(f"Backend: {backend}, TP: {tp_size}")
    print(f"{'='*60}\n")

    model_args_parts = [f"pretrained={model_path}"]

    if backend == "vllm":
        model_args_parts.extend([
            f"tensor_parallel_size={tp_size}",
            "dtype=bfloat16",
            "gpu_memory_utilization=0.85",
            "max_model_len=4096",
        ])
    else:
        model_args_parts.extend([
            "dtype=bfloat16",
        ])

    model_args = ",".join(model_args_parts)

    eval_kwargs = dict(
        model=backend,
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=True,
    )
    if num_fewshot is not None:
        eval_kwargs["num_fewshot"] = num_fewshot
    if limit is not None:
        eval_kwargs["limit"] = limit

    results = lm_eval.simple_evaluate(**eval_kwargs)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n  {task_name}:")
        for metric, value in sorted(task_results.items()):
            if isinstance(value, (int, float)) and "stderr" not in metric:
                print(f"    {metric}: {value:.4f}" if isinstance(value, float) else f"    {metric}: {value}")

    os.makedirs(output_dir, exist_ok=True)
    results_file = Path(output_dir) / "summary.json"
    summary = {}
    for task_name, task_results in results["results"].items():
        summary[task_name] = {
            k: v for k, v in task_results.items()
            if isinstance(v, (int, float, str))
        }
    results_file.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {results_file}")

    samples_file = Path(output_dir) / "full_results.json"
    serializable = {
        "results": results.get("results", {}),
        "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
        "n-shot": results.get("n-shot", {}),
    }
    samples_file.write_text(json.dumps(serializable, indent=2, default=str))

    return results


def main():
    parser = argparse.ArgumentParser(description="Academic benchmark evaluation")
    parser.add_argument("--mode", choices=["base", "adapter", "merged"],
                        default="base",
                        help="base: eval raw Qwen3-8B; "
                             "adapter: eval with PEFT adapter (hf backend); "
                             "merged: merge adapter then eval with vLLM")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter directory (required for adapter/merged modes)")
    parser.add_argument("--adapter-name", type=str, default="action_taking",
                        help="Name of adapter (for output directory naming)")
    parser.add_argument("--tasks", nargs="+",
                        default=["mmlu_pro", "minerva_math500"],
                        help="lm-eval task names")
    parser.add_argument("--output-dir", type=str,
                        default="eval_results/academic_benchmarks")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per task (for quick testing)")
    parser.add_argument("--merged-model-dir", type=str, default=None,
                        help="Directory to save merged model (temp dir if not set)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "base":
        run_name = "base_qwen3_8b"
        out_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)
        run_lm_eval(
            model_path=args.base_model,
            tasks=args.tasks,
            output_dir=out_dir,
            backend="vllm",
            tp_size=args.tp_size,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

    elif args.mode == "adapter":
        if not args.adapter_path:
            parser.error("--adapter-path is required for adapter mode")
        run_name = f"adapter_{args.adapter_name}"
        out_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)

        model_args = f"pretrained={args.base_model},peft={args.adapter_path},dtype=bfloat16"
        import lm_eval
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=args.tasks,
            batch_size=args.batch_size,
            output_path=out_dir,
            log_samples=True,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY (adapter mode)")
        print("=" * 60)
        for task_name, task_results in results["results"].items():
            print(f"\n  {task_name}:")
            for metric, value in sorted(task_results.items()):
                if isinstance(value, (int, float)) and "stderr" not in metric:
                    print(f"    {metric}: {value:.4f}" if isinstance(value, float)
                          else f"    {metric}: {value}")

    elif args.mode == "merged":
        if not args.adapter_path:
            parser.error("--adapter-path is required for merged mode")

        merged_dir = args.merged_model_dir
        use_temp = merged_dir is None
        if use_temp:
            merged_dir = tempfile.mkdtemp(prefix="merged_qwen3_8b_")

        try:
            merge_adapter(args.base_model, args.adapter_path, merged_dir)
            run_name = f"merged_{args.adapter_name}"
            out_dir = os.path.join(args.output_dir, run_name)
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, "eval_config.json"), "w") as f:
                json.dump({
                    "base_model": args.base_model,
                    "adapter_path": args.adapter_path,
                    "adapter_name": args.adapter_name,
                    "tasks": args.tasks,
                    "mode": "merged",
                }, f, indent=2)

            run_lm_eval(
                model_path=merged_dir,
                tasks=args.tasks,
                output_dir=out_dir,
                backend="vllm",
                tp_size=args.tp_size,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
            )
        finally:
            if use_temp and os.path.exists(merged_dir):
                print(f"Cleaning up temp merged model: {merged_dir}")
                shutil.rmtree(merged_dir)


if __name__ == "__main__":
    main()
