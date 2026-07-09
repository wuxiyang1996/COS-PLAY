#!/usr/bin/env python3
"""SFT cold-start trainer for all 5 LoRA adapters.

Supports **parallel training** across multiple GPUs — each adapter gets
its own GPU and process, so all 5 train simultaneously.  With 5+ GPUs
the total wall-clock time equals the slowest single adapter rather than
the sum of all five.

Output checkpoints are written in the exact layout expected by the
co-evolution GRPO pipeline::

    <output_dir>/
    ├── decision/
    │   ├── skill_selection/   # adapter_config.json + adapter_model.safetensors
    │   └── action_taking/
    └── skillbank/
        ├── segment/
        ├── contract/
        └── curator/

Usage::

    # Sequential (1 GPU, adapters trained one after another)
    python -m trainer.SFT.train

    # Parallel (each adapter on a separate GPU, all at once)
    python -m trainer.SFT.train --parallel

    # Parallel on specific GPUs
    python -m trainer.SFT.train --parallel --gpus 0 1 2 3 4

    # Subset + parallel
    python -m trainer.SFT.train --parallel --adapters segment contract curator
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT cold-start training for decision + skill-bank LoRA adapters",
    )
    p.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-8B",
        help="Base model (must match co-evolution config)",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory (default: runs/sft_coldstart)",
    )
    p.add_argument(
        "--decision_data_dir", type=str, default=None,
        help="Path to gpt54_skill_labeled/grpo_coldstart",
    )
    p.add_argument(
        "--skillbank_data_dir", type=str, default=None,
        help="Path to gpt54_skillbank_grpo",
    )
    p.add_argument(
        "--adapters", type=str, nargs="*", default=None,
        help="Subset of adapters to train (default: all 5)",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--eval_fraction", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument(
        "--games", type=str, nargs="*", default=None,
        help="Subset of games to include in training data",
    )
    p.add_argument(
        "--parallel", action="store_true", default=False,
        help="Train adapters in parallel, one per GPU",
    )
    p.add_argument(
        "--gpus", type=int, nargs="*", default=None,
        help="GPU IDs for parallel training (default: 0..N-1 where N = #adapters)",
    )
    p.add_argument(
        "--gpu", type=int, default=None,
        help="(internal) Pin this process to a specific GPU (used by parallel launcher)",
    )
    return p.parse_args()


def _build_config(args: argparse.Namespace):
    from trainer.SFT.config import SFTConfig
    kwargs = {
        "model_name": args.model_name,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "warmup_ratio": args.warmup_ratio,
        "eval_fraction": args.eval_fraction,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": args.bf16,
        "adapters": args.adapters,
    }
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.decision_data_dir:
        kwargs["decision_data_dir"] = args.decision_data_dir
    if args.skillbank_data_dir:
        kwargs["skillbank_data_dir"] = args.skillbank_data_dir
    if args.games:
        kwargs["games"] = args.games
    return SFTConfig(**kwargs)


def format_for_sft(examples: list, tokenizer) -> list:
    """Convert prompt/completion dicts to chat-formatted text for SFT.

    Uses the model's chat template (``apply_chat_template``) when
    available so the SFT data matches the format the model sees during
    GRPO inference.
    """
    formatted = []
    for ex in examples:
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "")
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                text = f"{prompt}\n{completion}"
        else:
            text = f"{prompt}\n{completion}"
        formatted.append({"text": text})
    return formatted


def train_single_adapter(
    adapter_name: str,
    examples: list,
    base_model,
    tokenizer,
    config,
) -> str:
    """Train one LoRA adapter via HuggingFace Trainer and return save path.

    The base model is wrapped with PEFT, trained, saved, then unwrapped
    so the same base model instance can be reused for the next adapter.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    params = config.effective_params(adapter_name)
    output_path = config.adapter_output_path(adapter_name)
    output_path.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path)

    logger.info(
        "=== Training LoRA adapter '%s' === (%d examples, lr=%.2e, epochs=%d)",
        adapter_name, len(examples), params["lr"], params["epochs"],
    )

    formatted = format_for_sft(examples, tokenizer)

    n_eval = max(1, int(len(formatted) * config.eval_fraction))
    eval_data = formatted[:n_eval]
    train_data = formatted[n_eval:]
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=params["max_seq_length"],
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    target_modules = config.resolve_target_modules()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config, adapter_name=adapter_name)
    peft_model.enable_input_require_grads()

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    hf_output = str(output_path / "hf_trainer")
    training_args = TrainingArguments(
        output_dir=hf_output,
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        learning_rate=params["lr"],
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    peft_model.save_pretrained(out_str)
    tokenizer.save_pretrained(out_str)

    meta = {
        "adapter_name": adapter_name,
        "base_model": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": target_modules,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "epochs": params["epochs"],
        "lr": params["lr"],
        "training_type": "sft_coldstart",
    }
    with open(output_path / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved adapter '%s' to %s", adapter_name, out_str)

    # Unwrap to reuse base model for next adapter
    base_model_unwrapped = peft_model.unload()
    base_model_unwrapped.config.use_cache = False

    del trainer, peft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_str


def train_all_adapters(config=None, gpu: Optional[int] = None, **kwargs) -> dict:
    """Train all requested LoRA adapters from cold-start data.

    Parameters
    ----------
    config : SFTConfig, optional
        If not given, a default config is created.
    gpu : int, optional
        Pin training to a specific GPU (sets ``CUDA_VISIBLE_DEVICES``
        before loading the model).
    **kwargs
        Override fields on ``SFTConfig``.

    Returns
    -------
    dict
        ``{adapter_name: output_path}`` for each trained adapter.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trainer.SFT.config import SFTConfig
    from trainer.SFT.data_loader import load_all_adapter_datasets

    if config is None:
        config = SFTConfig(**kwargs)

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logger.info("Pinned to GPU %d (CUDA_VISIBLE_DEVICES=%s)", gpu, gpu)

    logger.info("SFT cold-start config: model=%s, output=%s", config.model_name, config.output_dir)
    logger.info("Adapters to train: %s", config.adapters_to_train)

    t0 = time.time()

    datasets = load_all_adapter_datasets(config)

    empty = [name for name, data in datasets.items() if not data]
    if empty:
        logger.warning("No training data for adapters: %s — skipping", empty)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    logger.info("Loading base model '%s' (dtype=%s) …", config.model_name, dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base_model = base_model.to("cuda")
    base_model.config.use_cache = False
    logger.info(
        "Model loaded on %s — %.1f GB GPU memory allocated",
        next(base_model.parameters()).device,
        torch.cuda.memory_allocated() / 1e9,
    )

    results: dict = {}
    for adapter_name in config.adapters_to_train:
        data = datasets.get(adapter_name, [])
        if not data:
            logger.warning("Skipping '%s' — no data", adapter_name)
            continue

        save_path = train_single_adapter(
            adapter_name=adapter_name,
            examples=data,
            base_model=base_model,
            tokenizer=tokenizer,
            config=config,
        )
        results[adapter_name] = save_path

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info(
        "SFT cold-start training complete: %d adapters in %.1f min",
        len(results), elapsed / 60,
    )
    logger.info("Adapter paths:")
    for name, path in results.items():
        logger.info("  %s → %s", name, path)

    summary_path = Path(config.output_dir) / "sft_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "adapters": results,
            "model_name": config.model_name,
            "elapsed_min": round(elapsed / 60, 2),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.resolve_target_modules(),
        }, f, indent=2)

    return results


def _print_progress(processes: list, t0: float):
    """Tail the last progress line from each GPU's log file."""
    import re
    elapsed = time.time() - t0
    parts = [f"[{elapsed/60:.0f}m]"]
    for gpu, adapter_list, proc, _lf, log_path in processes:
        status = "running" if proc.poll() is None else (
            "done" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        )
        progress = ""
        try:
            with open(log_path, "r") as f:
                content = f.read()
            last_incomplete = None
            last_any = None
            for m in re.finditer(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", content):
                last_any = m
                if int(m.group(1)) < 100:
                    last_incomplete = m
            best = last_incomplete or last_any
            if best:
                progress = f" {best.group(1)}% ({best.group(2)}/{best.group(3)})"
        except Exception:
            pass
        names = "+".join(adapter_list)
        parts.append(f"GPU{gpu}[{names}]:{status}{progress}")
    logger.info("  ".join(parts))


def _train_parallel(config, gpu_ids: List[int]) -> dict:
    """Launch one subprocess per adapter, each pinned to a different GPU.

    Each subprocess runs ``python -m trainer.SFT.train --adapters <name>
    --gpu <id>`` so it loads the base model only on that GPU and trains
    independently.  Wall-clock time = slowest adapter instead of sum.
    """
    from trainer.SFT.config import SFTConfig

    adapters = config.adapters_to_train
    n_adapters = len(adapters)
    n_gpus = len(gpu_ids)

    if n_gpus < n_adapters:
        logger.warning(
            "Only %d GPUs for %d adapters — some GPUs will train multiple adapters sequentially",
            n_gpus, n_adapters,
        )

    gpu_assignment: Dict[int, List[str]] = {g: [] for g in gpu_ids}
    for i, adapter in enumerate(adapters):
        gpu = gpu_ids[i % n_gpus]
        gpu_assignment[gpu].append(adapter)

    logger.info("Parallel training plan:")
    for gpu, adapter_list in sorted(gpu_assignment.items()):
        if adapter_list:
            logger.info("  GPU %d: %s", gpu, ", ".join(adapter_list))

    base_cmd = [sys.executable, "-m", "trainer.SFT.train"]

    shared_args = [
        "--model_name", config.model_name,
        "--output_dir", config.output_dir,
        "--lr", str(config.lr),
        "--epochs", str(config.epochs),
        "--batch_size", str(config.batch_size),
        "--grad_accum", str(config.grad_accum),
        "--max_seq_length", str(config.max_seq_length),
        "--lora_r", str(config.lora_r),
        "--lora_alpha", str(config.lora_alpha),
        "--lora_dropout", str(config.lora_dropout),
        "--warmup_ratio", str(config.warmup_ratio),
        "--eval_fraction", str(config.eval_fraction),
        "--logging_steps", str(config.logging_steps),
        "--save_steps", str(config.save_steps),
    ]
    if config.bf16:
        shared_args.append("--bf16")
    else:
        shared_args.append("--no_bf16")
    if config.games:
        shared_args.extend(["--games"] + config.games)

    t0 = time.time()
    processes: List[tuple] = []

    for gpu, adapter_list in sorted(gpu_assignment.items()):
        if not adapter_list:
            continue
        cmd = base_cmd + shared_args + [
            "--adapters",
        ] + adapter_list + [
            "--gpu", str(gpu),
        ]

        log_path = Path(config.output_dir) / f"sft_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        logger.info("Launching GPU %d: %s → %s", gpu, adapter_list, log_path)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        processes.append((gpu, adapter_list, proc, log_file, log_path))

    results: dict = {}
    failed: List[str] = []

    active = list(processes)
    while active:
        still_running = []
        for gpu, adapter_list, proc, log_file, log_path in active:
            ret = proc.poll()
            if ret is None:
                still_running.append((gpu, adapter_list, proc, log_file, log_path))
            else:
                log_file.close()
                if ret == 0:
                    logger.info("GPU %d finished: %s", gpu, adapter_list)
                    for adapter in adapter_list:
                        results[adapter] = str(config.adapter_output_path(adapter))
                else:
                    logger.error(
                        "GPU %d FAILED (exit %d): %s — see %s",
                        gpu, ret, adapter_list, log_path,
                    )
                    failed.extend(adapter_list)
        active = still_running
        if not active:
            break
        _print_progress(processes, t0)
        time.sleep(15)

    elapsed = time.time() - t0

    if failed:
        logger.error("Failed adapters: %s", failed)
    logger.info(
        "Parallel SFT complete: %d/%d adapters in %.1f min (%.1fx vs sequential estimate)",
        len(results), n_adapters, elapsed / 60,
        max(1.0, n_adapters / max(1, min(n_adapters, n_gpus))),
    )

    summary_path = Path(config.output_dir) / "sft_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "adapters": results,
            "failed": failed,
            "model_name": config.model_name,
            "elapsed_min": round(elapsed / 60, 2),
            "parallel": True,
            "gpu_assignment": {str(g): a for g, a in gpu_assignment.items() if a},
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.resolve_target_modules(),
        }, f, indent=2)

    return results


def _detect_gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def main():
    args = parse_args()
    config = _build_config(args)

    if args.parallel:
        adapters = config.adapters_to_train
        if args.gpus:
            gpu_ids = args.gpus
        else:
            n_gpus = _detect_gpu_count()
            if n_gpus == 0:
                logger.error("--parallel requested but no GPUs detected; falling back to sequential")
                train_all_adapters(config)
                return
            gpu_ids = list(range(min(n_gpus, len(adapters))))
        logger.info(
            "Parallel mode: %d adapters across GPUs %s",
            len(adapters), gpu_ids,
        )
        _train_parallel(config, gpu_ids)
    else:
        train_all_adapters(config, gpu=args.gpu)


if __name__ == "__main__":
    main()
