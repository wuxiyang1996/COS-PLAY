#!/usr/bin/env python3
"""
Unified LoRA training script for skill-bank function adapters.

Trains one adapter at a time on the shared Qwen3-8B base model.
Other adapters stay frozen / unused.

Usage::

    # Train boundary adapter
    python -m trainer.skillbank.lora.train_lora \\
        --skill_function boundary \\
        --data_path runs/datasets/boundary_train.jsonl \\
        --output_dir runs/lora_adapters/boundary

    # Or use the convenience wrapper:
    python -m trainer.skillbank.lora.train_boundary_lora \\
        --data_path runs/datasets/boundary_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a function-specific LoRA adapter")
    p.add_argument(
        "--skill_function", type=str, required=True,
        choices=["boundary", "segment", "contract", "retrieval"],
        help="Which adapter to train",
    )
    p.add_argument("--data_path", type=str, required=True, help="JSONL training data")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Defaults to runs/lora_adapters/{skill_function}")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, nargs="*", default=None,
                    help="Linear layers to adapt (default: auto-detect)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--eval_fraction", type=float, default=0.05,
                    help="Fraction of data for evaluation")
    return p.parse_args()


def load_data(path: str):
    """Load JSONL dataset with prompt/completion pairs."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def format_for_sft(examples: list, tokenizer) -> list:
    """Convert prompt/completion dicts to text strings for SFT.

    Uses the Qwen chat template if available, otherwise simple concatenation.
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


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from skill_agents.lora.skill_function import SkillFunction

    fn = SkillFunction.from_str(args.skill_function)
    output_dir = args.output_dir or f"runs/lora_adapters/{fn.value}"

    logger.info("=== Training LoRA adapter for %s ===", fn.value)
    logger.info("Base model:  %s", args.base_model)
    logger.info("Output dir:  %s", output_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format data
    raw_examples = load_data(args.data_path)
    formatted = format_for_sft(raw_examples, tokenizer)

    # Train/eval split
    n_eval = max(1, int(len(formatted) * args.eval_fraction))
    train_data = formatted[n_eval:]
    eval_data = formatted[:n_eval]
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Load base model
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Configure LoRA — only the selected adapter
    target_modules = args.target_modules
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config, adapter_name=fn.adapter_name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info("Starting training …")
    trainer.train()

    # Save only the adapter (not the full model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved adapter to %s", output_dir)

    # Write metadata
    meta = {
        "skill_function": fn.value,
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
    }
    meta_path = Path(output_dir) / "adapter_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", meta_path)


def train_lora_adapter(
    base_model: str,
    task: str,
    data_path: str,
    output_dir: str,
    bank_path: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
) -> None:
    """Programmatic entry point for skill-bank LoRA training.

    Called by ``skillbank_agent_train.sh`` instead of going through the CLI.
    """
    args = argparse.Namespace(
        skill_function=task,
        data_path=data_path,
        base_model=base_model,
        output_dir=output_dir,
        lora_r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=None,
        lr=lr,
        epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        eval_fraction=0.05,
    )
    train(args)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
