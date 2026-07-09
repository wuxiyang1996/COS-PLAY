#!/usr/bin/env python3
"""Merge multiple LoRA adapters into unified decision + skillbank adapters.

Creates 2 unified LoRA adapters (1 decision + 1 skillbank) from the
existing 5 per-role adapters.  Two strategies, usable independently or
together:

  1. **Weight averaging** — element-wise mean of safetensor weights
     across adapters within each group.
  2. **Combined-data SFT** — pool cold-start training data from all
     adapters in a group and train a single LoRA.  Can optionally start
     from averaged weights as initialization.

After merging, the unified adapter is copied to every slot in its group
so the existing co-evolution pipeline sees 5 named adapters backed by
only 2 sets of weights.

Usage examples::

    # Average existing SFT adapter weights (fastest, no GPU needed for
    # the averaging step itself)
    python scripts/merge_lora_adapters.py average \
        --source-dir runs/sft_coldstart \
        --deploy-dir runs/Qwen3-8B_20260321_213813/lora_adapters

    # Train unified adapters from combined cold-start data
    python scripts/merge_lora_adapters.py retrain \
        --deploy-dir runs/Qwen3-8B_20260321_213813/lora_adapters

    # Average first, then fine-tune the averaged weights on combined data
    python scripts/merge_lora_adapters.py average-and-retrain \
        --source-dir runs/sft_coldstart \
        --deploy-dir runs/Qwen3-8B_20260321_213813/lora_adapters

    # Average from a specific checkpoint
    python scripts/merge_lora_adapters.py average \
        --source-dir runs/Qwen3-8B_20260321_213813/checkpoints/step_0005/adapters \
        --deploy-dir runs/Qwen3-8B_20260321_213813/lora_adapters
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

DECISION_ADAPTERS = ["skill_selection", "action_taking"]
SKILLBANK_ADAPTERS = ["segment", "contract", "curator"]
DECISION_SUBDIR = "decision"
SKILLBANK_SUBDIR = "skillbank"


# ── Weight Averaging ────────────────────────────────────────────────────


def _find_adapter_dirs(source_dir: str, adapter_names: List[str], subdir: str) -> List[str]:
    """Locate adapter directories under *source_dir*, trying nested and flat layouts.

    PEFT's ``save_pretrained`` creates a subdirectory named after the adapter,
    so the actual weights may live at ``<subdir>/<name>/<name>/adapter_config.json``.
    We try multiple candidate paths in priority order.
    """
    found: List[str] = []
    base = Path(source_dir)
    for name in adapter_names:
        candidates = [
            base / subdir / name,
            base / subdir / name / name,       # PEFT nested layout
            base / name,
            base / name / name,                 # PEFT nested flat layout
        ]
        resolved = False
        for p in candidates:
            if (p / "adapter_config.json").exists():
                found.append(str(p))
                resolved = True
                break
        if not resolved:
            # Last resort: scan one level of subdirs for adapter_config.json
            parent = base / subdir / name
            if parent.is_dir():
                for child in parent.iterdir():
                    if child.is_dir() and (child / "adapter_config.json").exists():
                        found.append(str(child))
                        resolved = True
                        break
        if not resolved:
            logger.warning("Adapter '%s' not found under %s", name, source_dir)
    return found


def average_lora_weights(adapter_dirs: List[str], output_dir: str) -> str:
    """Element-wise average of LoRA safetensor weights from multiple adapters.

    All adapters must share identical LoRA config (rank, alpha, target
    modules).  The averaged weights are saved alongside the config from
    the first adapter.
    """
    import torch
    from safetensors.torch import load_file, save_file

    logger.info(
        "Averaging LoRA weights from %d adapters → %s",
        len(adapter_dirs), output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)

    state_dicts: List[Dict[str, Any]] = []
    for d in adapter_dirs:
        sf = os.path.join(d, "adapter_model.safetensors")
        bn = os.path.join(d, "adapter_model.bin")
        if os.path.exists(sf):
            sd = load_file(sf)
        elif os.path.exists(bn):
            sd = torch.load(bn, map_location="cpu", weights_only=True)
        else:
            logger.warning("No adapter weights in %s — skipping", d)
            continue
        state_dicts.append(sd)
        logger.info("  Loaded %d tensors from %s", len(sd), d)

    if not state_dicts:
        raise FileNotFoundError("No adapter weights found to average")

    all_keys = sorted(state_dicts[0].keys())
    averaged: Dict[str, Any] = {}
    for key in all_keys:
        tensors = [sd[key] for sd in state_dicts if key in sd]
        averaged[key] = torch.stack(tensors).mean(dim=0).to(torch.bfloat16)

    logger.info("Averaged %d tensor keys across %d adapters", len(averaged), len(state_dicts))

    save_file(averaged, os.path.join(output_dir, "adapter_model.safetensors"))

    cfg_src = os.path.join(adapter_dirs[0], "adapter_config.json")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, output_dir)

    return output_dir


def run_averaging(source_dir: str, output_dir: str) -> Dict[str, str]:
    """Average decision adapters → unified_decision, skillbank → unified_skillbank."""
    results: Dict[str, str] = {}

    d_dirs = _find_adapter_dirs(source_dir, DECISION_ADAPTERS, DECISION_SUBDIR)
    if d_dirs:
        dst = os.path.join(output_dir, "unified_decision")
        average_lora_weights(d_dirs, dst)
        results["decision"] = dst
        logger.info(
            "Decision: averaged %d adapters (%s) → %s",
            len(d_dirs), ", ".join(DECISION_ADAPTERS[:len(d_dirs)]), dst,
        )
    else:
        logger.error("No decision adapters found — cannot average")

    s_dirs = _find_adapter_dirs(source_dir, SKILLBANK_ADAPTERS, SKILLBANK_SUBDIR)
    if s_dirs:
        dst = os.path.join(output_dir, "unified_skillbank")
        average_lora_weights(s_dirs, dst)
        results["skillbank"] = dst
        logger.info(
            "Skillbank: averaged %d adapters (%s) → %s",
            len(s_dirs), ", ".join(SKILLBANK_ADAPTERS[:len(s_dirs)]), dst,
        )
    else:
        logger.error("No skillbank adapters found — cannot average")

    return results


# ── Combined-Data SFT Training ─────────────────────────────────────────


def _load_combined_data(group: str, config: Any) -> List[Dict[str, str]]:
    """Pool cold-start SFT data for all adapters in a group."""
    from trainer.SFT.data_loader import load_adapter_dataset

    adapters = DECISION_ADAPTERS if group == "decision" else SKILLBANK_ADAPTERS
    combined: List[Dict[str, str]] = []
    for adapter_name in adapters:
        try:
            data = load_adapter_dataset(adapter_name, config)
            logger.info(
                "  [%s] %s: %d examples", group, adapter_name, len(data),
            )
            combined.extend(data)
        except Exception as exc:
            logger.warning("  [%s] %s: failed to load — %s", group, adapter_name, exc)

    logger.info("[%s] Combined: %d total examples", group, len(combined))
    return combined


def train_unified_adapter(
    adapter_label: str,
    examples: List[Dict[str, str]],
    output_dir: str,
    model_name: str = "Qwen/Qwen3-8B",
    init_adapter_dir: Optional[str] = None,
    lr: float = 2e-4,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bf16: bool = True,
) -> str:
    """Train a single unified LoRA adapter from combined data.

    If *init_adapter_dir* is provided, LoRA weights are initialised from
    those averaged weights instead of random init.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    logger.info(
        "=== Training unified '%s' adapter === "
        "(%d examples, lr=%.2e, epochs=%d, init=%s)",
        adapter_label, len(examples), lr, epochs,
        init_adapter_dir or "random",
    )

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = []
    for ex in examples:
        prompt, completion = ex.get("prompt", ""), ex.get("completion", "")
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

    n_eval = max(1, int(len(formatted) * 0.05))
    train_data, eval_data = formatted[n_eval:], formatted[:n_eval]
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True,
            max_length=max_seq_length, padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    arch = getattr(model_cfg, "model_type", "")
    if "qwen" in arch.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    dtype = torch.bfloat16 if bf16 else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to("cuda")
    base_model.config.use_cache = False

    if init_adapter_dir and (Path(init_adapter_dir) / "adapter_config.json").exists():
        logger.info("Initialising from averaged weights: %s", init_adapter_dir)
        peft_model = PeftModel.from_pretrained(
            base_model, init_adapter_dir, adapter_name=adapter_label,
        )
        peft_model.set_adapter(adapter_label)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config, adapter_name=adapter_label)

    peft_model.enable_input_require_grads()

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info("Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    hf_out = os.path.join(output_dir, "hf_trainer")
    training_args = TrainingArguments(
        output_dir=hf_out,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "adapter_label": adapter_label,
        "base_model": model_name,
        "source_adapters": (
            DECISION_ADAPTERS if "decision" in adapter_label else SKILLBANK_ADAPTERS
        ),
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "epochs": epochs,
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "init_from_average": init_adapter_dir is not None,
        "training_type": "merged_sft",
    }
    with open(os.path.join(output_dir, "adapter_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved unified '%s' adapter → %s", adapter_label, output_dir)

    del trainer, peft_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir


def run_combined_sft(
    output_dir: str,
    model_name: str = "Qwen/Qwen3-8B",
    init_decision: Optional[str] = None,
    init_skillbank: Optional[str] = None,
    decision_epochs: int = 5,
    skillbank_epochs: int = 8,
) -> Dict[str, str]:
    """Train 2 unified adapters from combined cold-start data."""
    from trainer.SFT.config import SFTConfig

    sft_config = SFTConfig(model_name=model_name)
    results: Dict[str, str] = {}

    decision_data = _load_combined_data("decision", sft_config)
    if decision_data:
        dst = os.path.join(output_dir, "unified_decision")
        train_unified_adapter(
            "unified_decision", decision_data, dst,
            model_name=model_name, init_adapter_dir=init_decision,
            epochs=decision_epochs,
        )
        results["decision"] = dst

    skillbank_data = _load_combined_data("skillbank", sft_config)
    if skillbank_data:
        dst = os.path.join(output_dir, "unified_skillbank")
        train_unified_adapter(
            "unified_skillbank", skillbank_data, dst,
            model_name=model_name, init_adapter_dir=init_skillbank,
            epochs=skillbank_epochs,
        )
        results["skillbank"] = dst

    return results


# ── Deployment ──────────────────────────────────────────────────────────


def deploy_to_adapter_slots(
    decision_dir: Optional[str],
    skillbank_dir: Optional[str],
    target_dir: str,
) -> None:
    """Copy unified adapters to all 5 named adapter slots.

    After deployment the directory layout matches what vLLM and the
    co-evolution GRPO pipeline expect::

        <target_dir>/
        ├── decision/
        │   ├── skill_selection/   ← copy of unified_decision
        │   └── action_taking/     ← copy of unified_decision
        └── skillbank/
            ├── segment/           ← copy of unified_skillbank
            ├── contract/          ← copy of unified_skillbank
            └── curator/           ← copy of unified_skillbank
    """
    if decision_dir:
        for name in DECISION_ADAPTERS:
            dst = os.path.join(target_dir, DECISION_SUBDIR, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(decision_dir, dst)
            logger.info("Deployed unified_decision → %s/%s", DECISION_SUBDIR, name)

    if skillbank_dir:
        for name in SKILLBANK_ADAPTERS:
            dst = os.path.join(target_dir, SKILLBANK_SUBDIR, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(skillbank_dir, dst)
            logger.info("Deployed unified_skillbank → %s/%s", SKILLBANK_SUBDIR, name)


# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA adapters into unified decision + skillbank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # --- average ---
    avg = sub.add_parser("average", help="Average existing SFT adapter weights")
    avg.add_argument("--source-dir", required=True, help="Dir with existing adapters (SFT output or checkpoint/adapters)")
    avg.add_argument("--output-dir", default="runs/merged_lora", help="Where to write unified adapters")
    avg.add_argument("--deploy-dir", default=None, help="If set, copy to all 5 adapter slots here")

    # --- retrain ---
    rt = sub.add_parser("retrain", help="Train from combined cold-start data")
    rt.add_argument("--output-dir", default="runs/merged_lora", help="Where to write unified adapters")
    rt.add_argument("--deploy-dir", default=None, help="If set, copy to all 5 adapter slots here")
    rt.add_argument("--model", default="Qwen/Qwen3-8B")
    rt.add_argument("--decision-epochs", type=int, default=5)
    rt.add_argument("--skillbank-epochs", type=int, default=8)

    # --- average-and-retrain ---
    ar = sub.add_parser("average-and-retrain", help="Average weights, then fine-tune on combined data")
    ar.add_argument("--source-dir", required=True, help="Dir with existing adapters")
    ar.add_argument("--output-dir", default="runs/merged_lora", help="Where to write unified adapters")
    ar.add_argument("--deploy-dir", default=None, help="If set, copy to all 5 adapter slots here")
    ar.add_argument("--model", default="Qwen/Qwen3-8B")
    ar.add_argument("--decision-epochs", type=int, default=3)
    ar.add_argument("--skillbank-epochs", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    results: Dict[str, str] = {}

    if args.mode == "average":
        results = run_averaging(args.source_dir, args.output_dir)

    elif args.mode == "retrain":
        results = run_combined_sft(
            args.output_dir,
            model_name=args.model,
            decision_epochs=args.decision_epochs,
            skillbank_epochs=args.skillbank_epochs,
        )

    elif args.mode == "average-and-retrain":
        avg_results = run_averaging(args.source_dir, args.output_dir)
        results = run_combined_sft(
            args.output_dir,
            model_name=args.model,
            init_decision=avg_results.get("decision"),
            init_skillbank=avg_results.get("skillbank"),
            decision_epochs=args.decision_epochs,
            skillbank_epochs=args.skillbank_epochs,
        )

    deploy_dir = getattr(args, "deploy_dir", None)
    if deploy_dir:
        deploy_to_adapter_slots(
            results.get("decision"),
            results.get("skillbank"),
            deploy_dir,
        )

    elapsed = time.time() - t0
    logger.info("Done in %.1f min", elapsed / 60)

    summary = {
        "mode": args.mode,
        "results": results,
        "deploy_dir": deploy_dir,
        "elapsed_min": round(elapsed / 60, 2),
    }
    summary_path = os.path.join(args.output_dir, "merge_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
