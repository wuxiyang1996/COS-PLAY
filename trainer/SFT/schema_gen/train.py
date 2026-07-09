#!/usr/bin/env python
"""Train the Qwen3-VL-8B ``schema_gen`` LoRA adapter.

Phase-1 of PLAN-VISUAL-GROUNDING-MILESTONES (§5.1).  The adapter learns
to map ``(image[s] + goal/question)`` → ``<state>…</state>`` schema
across all four collected domains (Gym-V, BrowserGym, image-QA,
video-QA) using teacher labels collected by the Phase-0 scripts +
benchmark parsers.

Usage::

    # Sanity check on 64 samples per domain, no GPU required for the dry
    # run (--inspect_only just lists what would be trained).
    python -m trainer.SFT.schema_gen.train \\
        --max_samples_per_domain 64 --inspect_only

    # Real run on a single A100 80 GB
    python -m trainer.SFT.schema_gen.train \\
        --domains gymv browser image_qa \\
        --epochs 2 --batch_size 1 --grad_accum 16

    # Heuristic-only ablation (PLAN-V-G-MILESTONES §9 ablation A2)
    python -m trainer.SFT.schema_gen.train \\
        --target_source heuristic \\
        --output_dir runs/sft_schema_gen_heuristic

Output checkpoint layout::

    <output_dir>/<run_id>/
        adapter_config.json      # peft adapter spec
        adapter_model.safetensors
        train_config.json        # full SchemaGenConfig dump
        training_args.json
        checkpoint-NNNN/         # intermediate checkpoints

The resulting adapter is loaded directly by
``vlm_wrapper.ground.cascaded_ground`` when the env var
``SCHEMA_GEN_ADAPTER_DIR`` is set to its directory — it then becomes
the Path-A "vision head".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("trainer.SFT.schema_gen.train")

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trainer.SFT.schema_gen.config import SchemaGenConfig  # noqa: E402
from trainer.SFT.schema_gen.data_loader import (  # noqa: E402
    SchemaGenSample,
    load_schema_gen_dataset,
)
from vlm_wrapper.schema import (  # noqa: E402
    SCHEMA_VERSION,
    build_adaptive_system_prompt,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Qwen3-VL-8B schema_gen LoRA adapter",
    )
    p.add_argument("--model_name", default=None,
                   help=(
                       "Override base VLM (default = SchemaGenConfig.model_name "
                       "= Qwen/Qwen3.5-35B-A3B, the unified vision-language MoE).  "
                       "Pass Qwen/Qwen3-VL-8B-Instruct for fast single-A100 "
                       "smoke runs, Qwen/Qwen3-VL-32B for the dense sibling, or "
                       "Qwen/Qwen3-VL-235B-A22B for the larger MoE teacher."
                   ))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument(
        "--domains", nargs="+", default=None,
        choices=["gymv", "env_wrappers", "browser", "image_qa", "video_qa"],
        help=(
            "Subset of domains to train on.  Default (from SchemaGenConfig) "
            "is gymv + env_wrappers — the two corpora populated from the "
            "current cold-start rollouts."
        ),
    )
    p.add_argument(
        "--target_source", default="vision",
        choices=["vision", "heuristic", "auto"],
        help="Which schema to use as the SFT target.",
    )
    p.add_argument("--max_samples_per_domain", type=int, default=None)
    p.add_argument("--include_hard_cases", action="store_true",
                   help="Override the default drop_hard_cases=True")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--inspect_only", action="store_true",
        help="Print dataset stats and exit without loading the model.",
    )
    p.add_argument(
        "--dump_dataset_jsonl", default=None,
        help="If set, write the assembled dataset (sample_id + prompt + "
             "target) to this JSONL and exit.  Useful for prompt review "
             "before launching a real run.",
    )
    # ── Speed / kernel knobs (T2.11 closure) ─────────────────────────
    p.add_argument(
        "--use_liger_kernel", dest="use_liger_kernel",
        action="store_true", default=True,
        help="Apply liger-kernel fused Qwen3.5 patches (default: True).",
    )
    p.add_argument(
        "--no_liger_kernel", dest="use_liger_kernel", action="store_false",
        help="Disable liger-kernel even if installed.",
    )
    p.add_argument(
        "--no_gradient_checkpointing", dest="no_gradient_checkpointing",
        action="store_true", default=False,
        help="Disable activation checkpointing (faster, more memory).",
    )
    p.add_argument(
        "--optim", type=str, default=None,
        help=(
            "HF Trainer optim string (default: 'paged_adamw_8bit' if "
            "bitsandbytes is installed, else 'adamw_torch_fused')."
        ),
    )
    p.add_argument(
        "--dataloader_workers", type=int, default=4,
        help="DataLoader worker count (default: 4).",
    )
    p.add_argument(
        "--strict_lora_coverage", action="store_true", default=False,
        help=(
            "Abort if any required projection has zero LoRA-wrapped layers "
            "(catches T2.11-style recipe drift)."
        ),
    )
    return p.parse_args()


def make_config(args: argparse.Namespace) -> SchemaGenConfig:
    cfg = SchemaGenConfig()
    if args.model_name:
        cfg.model_name = args.model_name
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_id:
        cfg.run_id = args.run_id
    if args.domains:
        cfg.domains = list(args.domains)
    if args.target_source:
        cfg.target_source = args.target_source
    if args.max_samples_per_domain is not None:
        cfg.max_samples_per_domain = args.max_samples_per_domain
    if args.include_hard_cases:
        cfg.drop_hard_cases = False
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    if args.max_seq_length is not None:
        cfg.max_seq_length = args.max_seq_length
    if args.seed is not None:
        cfg.seed = args.seed
    # ── Speed knobs ──────────────────────────────────────────────────
    if hasattr(args, "use_liger_kernel"):
        cfg.use_liger_kernel = args.use_liger_kernel
    if getattr(args, "no_gradient_checkpointing", False):
        cfg.gradient_checkpointing = False
    if args.optim is not None:
        cfg.optim = args.optim
    if hasattr(args, "dataloader_workers") and args.dataloader_workers is not None:
        cfg.dataloader_num_workers = args.dataloader_workers
    if getattr(args, "strict_lora_coverage", False):
        cfg.strict_lora_coverage = True
    return cfg


# ======================================================================
# Dataset assembly + chat-template formatting
# ======================================================================

def _system_prompt_for(domain: str, max_entities: int = 25) -> str:
    """Use the canonical adaptive system prompt builder.

    Keeping the system prompt identical to what the inference pipeline
    sends (``vlm_wrapper.ground._build_messages``) is critical — the
    student must see the *same* schema spec at train time and at
    rollout time, otherwise the cascaded eval will mis-bucket
    schema-format failures as model regressions.
    """
    return build_adaptive_system_prompt(
        domain=domain, max_entities=max_entities,
    )


def _to_chat_record(
    sample: SchemaGenSample, max_entities: int = 25,
) -> dict[str, Any]:
    """Turn a ``SchemaGenSample`` into the dict the trainer collator expects.

    Shape::

        {
            "messages": [<system>, <user with image>, <assistant target>],
            "images":   ["/abs/path/to/frame.png", ...],
            "domain":   "gymv",
            "source":   "vision",
            "sample_id": "...",
        }
    """
    sys_prompt = _system_prompt_for(sample.domain, max_entities=max_entities)
    user_content: list[dict[str, Any]] = []
    for path in sample.images:
        user_content.append({"type": "image", "image": path})
    user_content.append({"type": "text", "text": sample.prompt})
    # Wrap *every* turn's content as a list of typed parts.  Mixing
    # plain str + list inside the same ``messages`` column makes
    # pyarrow's ``Dataset.from_list`` raise
    # ``ArrowInvalid: cannot mix list and non-list, non-null values``.
    # Qwen3-VL's processor.apply_chat_template handles either form, so
    # uniform list-of-parts is safe at train and eval time.
    return {
        "messages": [
            {"role": "system",
             "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": user_content},
            {"role": "assistant",
             "content": [{"type": "text", "text": sample.target_schema}]},
        ],
        "images": sample.images,
        "domain": sample.domain,
        "source": sample.source,
        "sample_id": sample.sample_id,
    }


# ======================================================================
# Training driver
# ======================================================================

def _summarise(samples: list[SchemaGenSample]) -> dict[str, Any]:
    """Return a compact stats dict for logging / inspect_only mode."""
    by_domain: dict[str, int] = {}
    by_source: dict[str, int] = {}
    n_video = 0
    for s in samples:
        by_domain[s.domain] = by_domain.get(s.domain, 0) + 1
        by_source[s.source] = by_source.get(s.source, 0) + 1
        if s.extra_context.get("is_video"):
            n_video += 1
    return {
        "n_total": len(samples),
        "by_domain": by_domain,
        "by_source": by_source,
        "n_video_samples": n_video,
        "schema_version": SCHEMA_VERSION,
    }


def main() -> int:
    args = parse_args()
    cfg = make_config(args)
    logger.info("SchemaGenConfig: %s", json.dumps(cfg.to_dict(), indent=2))

    samples = load_schema_gen_dataset(cfg)
    stats = _summarise(samples)
    logger.info("Dataset stats: %s", json.dumps(stats, indent=2))
    if not samples:
        logger.error(
            "No samples loaded — check that the Phase-0 collection ran "
            "and the configured paths exist."
        )
        return 1

    if args.dump_dataset_jsonl:
        out_path = Path(args.dump_dataset_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({
                    "sample_id": s.sample_id,
                    "domain": s.domain,
                    "source": s.source,
                    "images": s.images,
                    "prompt": s.prompt,
                    "target_schema": s.target_schema,
                }, ensure_ascii=False) + "\n")
        logger.info("Wrote dataset preview to %s", out_path)
        return 0

    if args.inspect_only:
        return 0

    # ------------------------------------------------------------------
    # Lazy heavy-deps import — keeps --inspect_only working without GPU
    # / transformers installed.
    # ------------------------------------------------------------------
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor
        # Vision auto-class was renamed in transformers 5.x:
        #   transformers >=5.0 → ``AutoModelForImageTextToText``
        #   transformers  4.x → ``AutoModelForVision2Seq``
        # Try the new name first, fall back to the legacy alias so old
        # envs keep working.
        try:
            from transformers import AutoModelForImageTextToText as _AutoVisionModel
        except ImportError:  # transformers <5
            from transformers import AutoModelForVision2Seq as _AutoVisionModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        logger.error(
            "Training dependencies missing (%s).  Activate the "
            "`game-ai-agent` env (INSTALL.md §) or run "
            "`pip install -U trl peft transformers accelerate datasets`.",
            exc,
        )
        return 2

    # ------------------------------------------------------------------
    # Build HF Dataset
    # ------------------------------------------------------------------
    chat_rows = [_to_chat_record(s) for s in samples]
    ds = Dataset.from_list(chat_rows)

    # Eval split — small fraction held out so the trainer reports
    # generalisation metrics even though we don't have a gold dev set
    # yet.  PLAN-V-G-MILESTONES §8 Week 4 will replace this with the
    # benchmark-specific eval harness.
    if cfg.eval_fraction > 0 and len(ds) > 32:
        split = ds.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None
    logger.info(
        "Train size=%d, eval size=%s",
        len(train_ds), len(eval_ds) if eval_ds is not None else "n/a",
    )

    # ------------------------------------------------------------------
    # Load processor + model
    # ------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code,
    )

    # ── T2.11 speed-up scaffolding ────────────────────────────────────
    # TF32 + liger-kernel must be wired *before* the model loads since
    # liger-kernel monkey-patches class-level methods.
    from trainer.SFT.speed_utils import (
        apply_liger_kernel,
        enable_tf32,
        pick_optim,
    )
    enable_tf32()
    if cfg.use_liger_kernel:
        try:
            from transformers import AutoConfig as _AutoCfg
            _probe_cfg = _AutoCfg.from_pretrained(
                cfg.model_name, trust_remote_code=cfg.trust_remote_code,
            )
            _probe_arch = (
                getattr(getattr(_probe_cfg, "text_config", _probe_cfg),
                        "model_type", "") or ""
            ).lower()
            # ``fused_loss=False`` because schema_gen uses TRL's
            # ``SFTTrainer`` whose ``compute_loss`` reads
            # ``outputs.logits[..., :-1, :]`` directly — liger's
            # default ``fused_linear_cross_entropy=True`` patch
            # nullifies ``outputs.logits`` and trips a TypeError
            # at the first training step.  RMSNorm + SwiGLU fusions
            # still fire, retaining the bulk of the speedup.
            apply_liger_kernel(_probe_arch, fused_loss=False)
        except Exception as exc:
            logger.warning(
                "liger-kernel probe failed: %s — proceeding without it.",
                exc,
            )

    # Resolve attention implementation with a graceful fallback chain:
    #   1. flash_attention_2  — fastest, but requires `flash-attn` which
    #      lacks a prebuilt wheel for torch 2.11+cu130 at the moment.
    #   2. sdpa               — PyTorch's built-in scaled-dot-product
    #      attention (uses Hopper-optimised Flash kernels under the hood
    #      on H200 with bf16); no extra dep.
    #   3. eager              — last-resort reference implementation.
    def _resolve_attn_impl() -> str:
        if cfg.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash-attn not installed; falling back to "
                    "attn_implementation='sdpa' (fast on H200/bf16)."
                )
        return "sdpa"

    attn_impl = _resolve_attn_impl()
    logger.info("Using attn_implementation=%s", attn_impl)
    model = _AutoVisionModel.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=attn_impl,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Resolve target_modules for the *actual* base model ────────────
    # Single source of truth in ``trainer.SFT.lora_targets`` — the
    # Qwen3.5 hybrid stack needs ``in_proj_z/b/a`` legs that the older
    # classic-7 list silently missed (T2.11).
    resolved_targets = cfg.resolve_target_modules()
    logger.info(
        "LoRA target_modules (resolved for %s): %s",
        cfg.model_name, resolved_targets,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=resolved_targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── T2.11 fail-fast on recipe drift ───────────────────────────────
    from trainer.SFT.lora_targets import assert_lora_coverage
    text_cfg = getattr(model.config, "text_config", model.config)
    text_arch = (getattr(text_cfg, "model_type", "") or "").lower()
    assert_lora_coverage(
        model,
        model_arch=text_arch,
        require_strict=cfg.strict_lora_coverage,
        logger_=logger,
    )

    # ------------------------------------------------------------------
    # Collator: format messages → tokenised input ids (mask the user
    # tokens out of the loss so we only train on the schema completion).
    # ------------------------------------------------------------------
    def _clean_msgs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip pyarrow-injected null fields from message content.

        ``Dataset.from_list`` unions all dict keys across rows, so a row
        like ``{"type": "text", "text": "..."}`` becomes
        ``{"type": "text", "text": "...", "image": None}``.  Qwen3-VL's
        chat template then sees ``'image' in item`` and (if the role is
        ``system``) raises *"System message cannot contain images."*.
        Drop the null keys so the template sees the original shape.
        """
        cleaned: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                content = [
                    {k: v for k, v in part.items() if v is not None}
                    if isinstance(part, dict)
                    else part
                    for part in content
                ]
            cleaned.append({**msg, "content": content})
        return cleaned

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        images_per_sample: list[list[Any]] = []
        for ex in batch:
            text = processor.apply_chat_template(
                _clean_msgs(ex["messages"]),
                tokenize=False, add_generation_prompt=False,
            )
            texts.append(text)
            from PIL import Image as _Image
            imgs = []
            for path in ex["images"]:
                if path.lower().endswith((".mp4", ".mov", ".webm", ".avi")):
                    # Sample frames at train time for video samples.
                    from visual_reasoning_wrapper.benchmarks.video_holmes import (
                        sample_video_frames,
                    )
                    frames, _, _ = sample_video_frames(
                        path, num_frames=cfg.video_num_frames,
                        max_side=cfg.image_max_side,
                    )
                    imgs.extend(frames)
                else:
                    img = _Image.open(path).convert("RGB")
                    if max(img.size) > cfg.image_max_side:
                        scale = cfg.image_max_side / max(img.size)
                        img = img.resize(
                            (int(img.size[0] * scale),
                             int(img.size[1] * scale)),
                        )
                    imgs.append(img)
            images_per_sample.append(imgs)

        encoded = processor(
            text=texts, images=images_per_sample,
            padding=True, truncation=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        # Mask pad tokens out of the loss.
        labels[labels == processor.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    out_dir = cfg.adapter_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # NOTE: We construct an `SFTConfig` directly (rather than a plain
    # `TrainingArguments`) because TRL 0.17.x's `SFTTrainer.__init__` runs a
    # legacy compatibility branch when it receives `TrainingArguments` that
    # calls `.pop("push_to_hub_token")` — a field that no longer exists in
    # transformers 5.x, raising `KeyError: 'push_to_hub_token'`.  Passing
    # `SFTConfig` skips that branch entirely.  `SFTConfig` is a subclass of
    # `TrainingArguments` so all standard fields below are still valid.
    resolved_optim = cfg.optim or pick_optim(prefer_8bit=True)
    logger.info(
        "Trainer optimizer = %s, gradient_checkpointing=%s, dataloader_workers=%d",
        resolved_optim, cfg.gradient_checkpointing, cfg.dataloader_num_workers,
    )

    train_args = SFTConfig(
        output_dir=str(out_dir),
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps if eval_ds is not None else None,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # Qwen3.5-MoE expert routing produces different per-expert
        # token counts on the forward pass vs the backward recompute.
        # The default ``use_reentrant=False`` checkpoint then fails the
        # strict shape-match sanity check with
        #   ``torch.utils.checkpoint.CheckpointError: Recomputed values
        #     for the following tensors have different metadata...``
        # Switching to legacy reentrant mode skips that check.
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim=resolved_optim,
        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        # ---------------------------------------------------------------
        # Skip TRL's built-in dataset preprocessing.  TRL 0.17.x would
        # otherwise call ``tokenizer.apply_chat_template`` on every row,
        # which trips Qwen3-VL's chat template:
        #   ``System message cannot contain images.``
        # PyArrow unifies the dict schema across all ``content`` items in
        # the ``messages`` column; that union adds a (null) ``image``
        # field to the system text dict, which the Jinja template
        # interprets as "this system message has an image".
        # We already build inputs ourselves in ``collate``, so we simply
        # opt out of TRL's preprocessing.
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )
    logger.info("Starting Qwen3-VL schema_gen LoRA training → %s", out_dir)
    trainer.train()
    trainer.save_model(str(out_dir))
    logger.info("Training complete.  Adapter saved at %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
