"""FSDP-parallel GRPO trainer for multi-GPU LoRA training.

Spawns one process per GPU using ``torch.multiprocessing.spawn``.
Each process loads the base model, applies the target LoRA adapter,
wraps with FSDP (sharding frozen base weights across ranks), and
trains on its slice of the data.  FSDP handles gradient
synchronisation automatically via AllReduce during ``.backward()``.

Typical 8-GPU FSDP memory per rank (14B bf16):
  - Sharded frozen params:  ~3.5 GB  (28 GB / 8)
  - Full LoRA params:       ~0.25 GB
  - LoRA optimizer states:  ~0.5 GB
  - Activations (grad ckpt): ~10-20 GB
  Total: ~14-24 GB of 80 GB  →  plenty of room

Usage::

    from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo

    stats = run_fsdp_grpo(
        gpu_ids=[4, 5, 6, 7],
        model_name="Qwen/Qwen3-8B",
        adapter_dir="runs/lora_adapters/decision/skill_selection",
        adapter_name="skill_selection",
        prompts=prompts,
        completions=completions,
        advantages=advantages,
    )
"""

from __future__ import annotations

import gc
import json
import logging
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_GRPO_MAX_SEQ_LEN = int(os.environ.get("GRPO_MAX_SEQ_LEN", "2048"))
_GRPO_REF_MICRO_BATCH = int(os.environ.get("GRPO_REF_MICRO_BATCH", "8"))
_NCCL_TIMEOUT_S = int(os.environ.get("GRPO_NCCL_TIMEOUT_S", "600"))


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _detect_wrap_cls(model):
    """Return the transformer decoder-layer class for FSDP auto-wrap."""
    for module in model.modules():
        cls_name = type(module).__name__
        if "DecoderLayer" in cls_name:
            return {type(module)}
    return set()


def _wrap_for_chat(tokenizer, prompts, completions):
    """Apply the model's chat template to (prompt, completion) pairs.

    Returns ``(chat_full_texts, chat_prompts)`` where each full text is
    ``[user: prompt][assistant: completion]`` and each chat prompt is
    ``[user: prompt]<generation_prompt>`` — matching the SFT training
    format.  Falls back to raw concatenation when the tokenizer has no
    ``apply_chat_template``.
    """
    has_template = hasattr(tokenizer, "apply_chat_template")
    chat_full: list = []
    chat_prompts: list = []
    for p, c in zip(prompts, completions):
        if has_template:
            try:
                full = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p},
                     {"role": "assistant", "content": c}],
                    tokenize=False, add_generation_prompt=False,
                )
                cprompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                chat_full.append(full)
                chat_prompts.append(cprompt)
                continue
            except Exception:
                pass
        chat_full.append(p + c)
        chat_prompts.append(p)
    return chat_full, chat_prompts


def _run_grpo_training_loop(
    model,
    tokenizer,
    rank: int,
    device: "torch.device",
    is_main: bool,
    world_size: int,
    adapter_name: str,
    prompts: list,
    completions: list,
    advantages: list,
    lr: float,
    epochs: int,
    batch_size: int,
    clip_ratio: float,
    kl_coeff: float,
    accumulation_steps: int = 4,
) -> "Optional[Dict[str, Any]]":
    """Execute GRPO training on an already-wrapped FSDP model.

    Handles data sharding, tokenization, reference log-prob computation,
    and the PPO-style training loop.  The caller is responsible for model
    lifecycle (loading, FSDP wrapping, saving weights, cleanup).

    Returns a dict of training statistics, or ``None`` if training was
    skipped (no valid samples).
    """
    import torch
    import torch.distributed as dist

    n_total = len(prompts)
    per_rank = n_total // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else n_total
    my_prompts = prompts[start_idx:end_idx]
    my_completions = completions[start_idx:end_idx]
    my_advantages = advantages[start_idx:end_idx]
    n_my = len(my_prompts)

    if is_main:
        logger.info(
            "FSDP GRPO [%s]: %d samples → %d/rank, %d epochs, bs=%d",
            adapter_name, n_total, n_my, epochs, batch_size,
        )

    max_seq = _GRPO_MAX_SEQ_LEN
    t_ref = time.time()
    tokenized: list = []
    if n_my > 0:
        chat_full, chat_prompts = _wrap_for_chat(
            tokenizer, my_prompts, my_completions,
        )
        enc_batch = tokenizer(
            chat_full, truncation=True, max_length=max_seq,
            padding=False, return_tensors=None,
        )
        penc_batch = tokenizer(
            chat_prompts, truncation=True, max_length=max_seq,
            padding=False, return_tensors=None,
        )
        for i in range(n_my):
            ids = enc_batch["input_ids"][i]
            mask = enc_batch["attention_mask"][i]
            plen = len(penc_batch["input_ids"][i])
            if plen >= len(ids):
                continue
            tokenized.append({
                "input_ids": torch.tensor([ids]),
                "attn_mask": torch.tensor([mask]),
                "plen": plen,
                "adv": my_advantages[i],
            })

    n_valid_local = torch.tensor(
        len(tokenized), device=device, dtype=torch.long,
    )
    n_valid_max = n_valid_local.clone()
    dist.all_reduce(n_valid_max, op=dist.ReduceOp.MAX)
    n_valid_max_val = int(n_valid_max.item())

    if n_valid_max_val == 0:
        if is_main:
            logger.warning("All samples filtered out, skipping training")
        return None

    dummy_entry = tokenized[0] if tokenized else {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attn_mask": torch.ones(1, 2, dtype=torch.long),
        "plen": 1,
        "adv": 0.0,
    }
    while len(tokenized) < n_valid_max_val:
        tokenized.append(None)

    # ── Reference log-probabilities (memory-safe micro-batching) ──
    ref_data: list = [None] * len(tokenized)
    pad_id = tokenizer.pad_token_id or 0
    ref_batch = min(_GRPO_REF_MICRO_BATCH, batch_size)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(tokenized), ref_batch):
            batch_end = min(batch_start + ref_batch, len(tokenized))
            batch_items = tokenized[batch_start:batch_end]
            real_flags = [tk is not None for tk in batch_items]
            entries = [tk if tk is not None else dummy_entry
                       for tk in batch_items]

            max_len = max(e["input_ids"].shape[1] for e in entries)
            bsz = len(entries)
            batch_ids = torch.full(
                (bsz, max_len), pad_id, dtype=torch.long,
            )
            batch_attn = torch.zeros(bsz, max_len, dtype=torch.long)
            for j, e in enumerate(entries):
                slen = e["input_ids"].shape[1]
                batch_ids[j, :slen] = e["input_ids"][0]
                batch_attn[j, :slen] = e["attn_mask"][0]

            batch_ids = batch_ids.to(device)
            batch_attn = batch_attn.to(device)

            try:
                out = model(input_ids=batch_ids, attention_mask=batch_attn)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if is_main:
                    logger.warning(
                        "OOM in ref log-prob batch (bsz=%d, seq=%d), "
                        "falling back to sample-by-sample",
                        bsz, max_len,
                    )
                for j, e in enumerate(entries):
                    if not real_flags[j]:
                        continue
                    slen = e["input_ids"].shape[1]
                    one_ids = e["input_ids"].to(device)
                    one_attn = e["attn_mask"].to(device)
                    try:
                        one_out = model(
                            input_ids=one_ids, attention_mask=one_attn,
                        )
                        plen = e["plen"]
                        logits_j = one_out.logits[0, plen - 1:slen - 1, :]
                        target_j = one_ids[0, plen:slen]
                        lp = torch.log_softmax(logits_j, dim=-1)
                        per_tok = lp.gather(
                            -1, target_j.unsqueeze(-1),
                        ).squeeze(-1)
                        del one_out, logits_j, lp
                        if per_tok.numel() > 0:
                            ref_data[batch_start + j] = {
                                "input_ids": e["input_ids"],
                                "attn_mask": e["attn_mask"],
                                "plen": plen,
                                "ref_lp": per_tok.cpu(),
                                "adv": e["adv"],
                            }
                        del per_tok
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if is_main:
                            logger.warning(
                                "OOM on single sample (seq=%d), skipping",
                                slen,
                            )
                    finally:
                        del one_ids, one_attn
                        torch.cuda.empty_cache()
                continue

            for j, e in enumerate(entries):
                if not real_flags[j]:
                    continue
                plen = e["plen"]
                slen = e["input_ids"].shape[1]
                logits_j = out.logits[j, plen - 1:slen - 1, :]
                target_j = batch_ids[j, plen:slen]
                lp = torch.log_softmax(logits_j, dim=-1)
                per_tok = lp.gather(
                    -1, target_j.unsqueeze(-1),
                ).squeeze(-1)
                del logits_j, lp

                if per_tok.numel() == 0:
                    continue
                ref_data[batch_start + j] = {
                    "input_ids": e["input_ids"],
                    "attn_mask": e["attn_mask"],
                    "plen": plen,
                    "ref_lp": per_tok.cpu(),
                    "adv": e["adv"],
                }
            del out, batch_ids, batch_attn
            torch.cuda.empty_cache()
    model.train()

    ref_data = [rd for rd in ref_data if rd is not None]

    n_train_local = torch.tensor(
        len(ref_data), device=device, dtype=torch.long,
    )
    n_train_max = n_train_local.clone()
    dist.all_reduce(n_train_max, op=dist.ReduceOp.MAX)
    n_train_max_val = int(n_train_max.item())

    if n_train_max_val == 0:
        if is_main:
            logger.warning("No valid ref data, skipping training")
        return None

    if is_main:
        logger.info(
            "Reference log-probs: %d/%d valid (%.1fs)",
            len(ref_data), n_my, time.time() - t_ref,
        )

    dummy_rd = ref_data[0] if ref_data else {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attn_mask": torch.ones(1, 2, dtype=torch.long),
        "plen": 1,
        "ref_lp": torch.zeros(1),
        "adv": 0.0,
    }
    while len(ref_data) < n_train_max_val:
        ref_data.append(None)

    # ── Training loop (with gradient accumulation + OOM resilience) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    accum = max(1, accumulation_steps)

    # Adaptive batch size: cap based on median sequence length to prevent
    # OOM on long-context games like Diplomacy (~2 KB prompts).
    # Heuristic: safe_bs ~ 80GB_budget / (seq_len * per_token_cost).
    seq_lengths = [rd["input_ids"].shape[1] for rd in ref_data if rd is not None]
    if seq_lengths:
        median_seq = sorted(seq_lengths)[len(seq_lengths) // 2]
        safe_bs = max(1, int(24_000 / max(median_seq, 1)))
        eff_batch_size = min(batch_size, safe_bs)
        if eff_batch_size < batch_size and is_main:
            logger.info(
                "Adaptive batch size: %d→%d (median_seq=%d)",
                batch_size, eff_batch_size, median_seq,
            )
    else:
        eff_batch_size = batch_size

    total_loss = 0.0
    total_tokens = 0
    n_my_train = len(ref_data)
    t_train = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_samples = 0
        t_epoch = time.time()

        n_mini = max(1, (n_my_train + eff_batch_size - 1) // eff_batch_size)
        for mb in range(n_mini):
            mb_start = mb * eff_batch_size
            mb_end = min(mb_start + eff_batch_size, n_my_train)
            batch_items = ref_data[mb_start:mb_end]
            real_flags = [rd is not None for rd in batch_items]
            entries = [rd if rd is not None else dummy_rd
                       for rd in batch_items]

            max_len = max(e["input_ids"].shape[1] for e in entries)
            bsz = len(entries)
            batch_ids = torch.full(
                (bsz, max_len), pad_id, dtype=torch.long,
            )
            batch_attn = torch.zeros(bsz, max_len, dtype=torch.long)
            for j, e in enumerate(entries):
                slen = e["input_ids"].shape[1]
                batch_ids[j, :slen] = e["input_ids"][0]
                batch_attn[j, :slen] = e["attn_mask"][0]

            batch_ids = batch_ids.to(device)
            batch_attn = batch_attn.to(device)

            try:
                with torch.enable_grad():
                    out = model(
                        input_ids=batch_ids,
                        attention_mask=batch_attn,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                old_bs = eff_batch_size
                eff_batch_size = max(1, eff_batch_size // 2)
                if is_main:
                    logger.warning(
                        "OOM in training fwd (bsz=%d, seq=%d), "
                        "halving batch %d→%d and skipping mini-batch",
                        bsz, max_len, old_bs, eff_batch_size,
                    )
                optimizer.zero_grad()
                continue

            mb_losses: list = []
            for j, e in enumerate(entries):
                if not real_flags[j]:
                    mb_losses.append(out.logits[j, 0, 0] * 0.0)
                    continue

                plen = e["plen"]
                slen = e["input_ids"].shape[1]
                old_lp = e["ref_lp"].to(device)

                logits_j = out.logits[j, plen - 1:slen - 1, :]
                target_j = batch_ids[j, plen:slen]
                lp = torch.log_softmax(logits_j, dim=-1)
                per_tok = lp.gather(
                    -1, target_j.unsqueeze(-1),
                ).squeeze(-1)
                del logits_j, lp

                ratio = torch.exp(per_tok - old_lp)
                clipped = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio,
                )
                adv_t = torch.tensor(
                    e["adv"], device=device, dtype=per_tok.dtype,
                )
                if not torch.isfinite(adv_t).all():
                    mb_losses.append(out.logits[j, 0, 0] * 0.0)
                    continue
                surr = torch.min(ratio * adv_t, clipped * adv_t)
                loss = -surr.mean() + kl_coeff * (old_lp - per_tok).mean()
                if not torch.isfinite(loss):
                    mb_losses.append(out.logits[j, 0, 0] * 0.0)
                    continue

                mb_losses.append(loss)
                epoch_loss += loss.item()
                epoch_samples += 1
                total_tokens += per_tok.numel()

            if mb_losses:
                (torch.stack(mb_losses).mean() / accum).backward()

            del out, batch_ids, batch_attn
            torch.cuda.empty_cache()

            is_accum_boundary = ((mb + 1) % accum == 0) or (mb == n_mini - 1)
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += epoch_loss / max(epoch_samples, 1)

        if is_main:
            elapsed = time.time() - t_epoch
            rate = n_my / elapsed if elapsed > 0 else 0
            logger.info(
                "  FSDP [%s] epoch %d/%d: loss=%.4f (%.1f samples/s, %.1fs, accum=%d, bs=%d)",
                adapter_name, epoch + 1, epochs,
                epoch_loss / max(epoch_samples, 1), rate, elapsed, accum,
                eff_batch_size,
            )

    train_time = time.time() - t_train
    del optimizer, trainable_params, ref_data, tokenized

    return {
        "n_samples": n_total,
        "n_tokens": total_tokens,
        "mean_loss": total_loss / max(epochs, 1),
        "epochs": epochs,
        "train_time_s": train_time,
    }


def _save_lora_under_fsdp(model, adapter_name, save_dir, is_main):
    """Save LoRA weights efficiently using ``summon_full_params``.

    Only gathers parameters temporarily (no CPU offload of the full 16 GB
    base model), then copies only the small LoRA tensors to CPU for saving.
    """
    import re

    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    t0 = time.time()
    with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
        if is_main:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            _adapter_key_re = re.compile(
                r"(lora_[AB])\." + re.escape(adapter_name) + r"\.(weight)"
            )
            lora_state = {}
            for name, param in model.named_parameters():
                if "lora_" not in name:
                    continue
                clean_key = _adapter_key_re.sub(r"\1.\2", name)
                lora_state[clean_key] = param.detach().cpu().clone()

            if lora_state:
                try:
                    from safetensors.torch import save_file
                    save_file(
                        lora_state,
                        str(save_path / "adapter_model.safetensors"),
                    )
                except ImportError:
                    torch.save(
                        lora_state,
                        str(save_path / "adapter_model.bin"),
                    )
                logger.info(
                    "Saved %d LoRA tensors → %s (%.1fs)",
                    len(lora_state), save_path, time.time() - t0,
                )
            else:
                logger.warning("No LoRA parameters found to save!")


def _load_or_init_lora_under_fsdp(
    model, adapter_dir, adapter_name, device, is_main,
):
    """Swap LoRA weights inside an FSDP-wrapped model.

    Uses ``summon_full_params(writeback=True)`` to temporarily unshard all
    parameters, copies new LoRA values from disk (or re-initialises them),
    then re-shards on context exit.
    """
    import math
    import re

    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    t0 = time.time()
    adapter_path = Path(adapter_dir)
    has_saved = (adapter_path / "adapter_config.json").exists()

    saved_state: dict = {}
    if has_saved:
        sf = adapter_path / "adapter_model.safetensors"
        bf = adapter_path / "adapter_model.bin"
        if sf.exists():
            from safetensors.torch import load_file
            saved_state = load_file(str(sf), device="cpu")
        elif bf.exists():
            saved_state = torch.load(str(bf), map_location="cpu")

    _adapter_key_re = re.compile(
        r"(lora_[AB])\." + re.escape(adapter_name) + r"\.(weight)"
    )

    with FSDP.summon_full_params(model, writeback=True, rank0_only=False):
        for name, param in model.named_parameters():
            if "lora_" not in name:
                continue
            clean = _adapter_key_re.sub(r"\1.\2", name)
            if clean in saved_state:
                param.data.copy_(
                    saved_state[clean].to(param.dtype).to(param.device)
                )
            elif "lora_A" in name:
                torch.nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
            elif "lora_B" in name:
                torch.nn.init.zeros_(param.data)

    if is_main:
        src = "disk" if saved_state else "fresh init"
        logger.info(
            "LoRA weights swapped to '%s' (%s, %.1fs)",
            adapter_name, src, time.time() - t0,
        )


def _train_one_adapter(
    rank: int,
    device: "torch.device",
    is_main: bool,
    world_size: int,
    model_name: str,
    job: Dict[str, Any],
) -> None:
    """Train a single LoRA adapter with FSDP.

    Assumes NCCL process group is already initialized.
    Loads model, trains, saves, and frees GPU memory.
    """
    import functools

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = job["adapter_dir"]
    adapter_name = job["adapter_name"]
    prompts = job["prompts"]
    completions = job["completions"]
    advantages = job["advantages"]
    lr = job["lr"]
    epochs = job["epochs"]
    batch_size = job["batch_size"]
    clip_ratio = job["clip_ratio"]
    kl_coeff = job["kl_coeff"]
    save_dir = job["save_dir"]
    result_file = job["result_file"]
    io_log_dir = job.get("io_log_dir")

    t0 = time.time()

    if is_main:
        logger.info(
            "FSDP rank 0: loading %s (bf16) onto %d GPUs",
            model_name, world_size,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": rank},
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    adapter_path = Path(adapter_dir)
    adapter_config_file = adapter_path / "adapter_config.json"

    if adapter_config_file.exists():
        model = PeftModel.from_pretrained(
            model, str(adapter_path),
            adapter_name=adapter_name,
            is_trainable=True,
        )
        if is_main:
            logger.info("Loaded adapter '%s' from %s", adapter_name, adapter_path)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj",
            ],
            inference_mode=False,
        )
        model = get_peft_model(model, lora_cfg, adapter_name=adapter_name)
        if is_main:
            logger.info("Created fresh adapter '%s' (r=16, alpha=32)", adapter_name)

    for n, p in model.named_parameters():
        is_lora = "lora" in n.lower()
        p.requires_grad = is_lora
        if is_lora and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    model.train()

    wrap_cls = _detect_wrap_cls(model)
    if wrap_cls:
        auto_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=wrap_cls,
        )
    else:
        auto_policy = None

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=bf16_policy,
        device_id=rank,
        use_orig_params=True,
    )

    load_time = time.time() - t0
    if is_main:
        n_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            "FSDP model ready (%d GPUs, %s trainable params, %.1fs load)",
            world_size, f"{n_trainable:,}", load_time,
        )

    n_total = len(prompts)
    per_rank = n_total // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank < world_size - 1 else n_total
    my_prompts = prompts[start_idx:end_idx]
    my_completions = completions[start_idx:end_idx]
    my_advantages = advantages[start_idx:end_idx]
    n_my = len(my_prompts)

    if is_main:
        logger.info(
            "FSDP GRPO [%s]: %d samples → %d/rank, %d epochs, bs=%d",
            adapter_name, n_total, n_my, epochs, batch_size,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    max_seq = _GRPO_MAX_SEQ_LEN
    t_ref = time.time()
    tokenized: list = []
    if n_my > 0:
        chat_full, chat_prompts = _wrap_for_chat(
            tokenizer, my_prompts, my_completions,
        )
        enc_batch = tokenizer(
            chat_full, truncation=True, max_length=max_seq,
            padding=False, return_tensors=None,
        )
        penc_batch = tokenizer(
            chat_prompts, truncation=True, max_length=max_seq,
            padding=False, return_tensors=None,
        )
        for i in range(n_my):
            ids = enc_batch["input_ids"][i]
            mask = enc_batch["attention_mask"][i]
            plen = len(penc_batch["input_ids"][i])
            if plen >= len(ids):
                continue
            tokenized.append({
                "input_ids": torch.tensor([ids]),
                "attn_mask": torch.tensor([mask]),
                "plen": plen,
                "adv": my_advantages[i],
            })

    n_valid_local = torch.tensor(
        len(tokenized), device=device, dtype=torch.long,
    )
    n_valid_max = n_valid_local.clone()
    dist.all_reduce(n_valid_max, op=dist.ReduceOp.MAX)
    n_valid_max_val = int(n_valid_max.item())

    if n_valid_max_val == 0:
        if is_main:
            logger.warning("All samples filtered out, skipping training")
            result = {
                "n_samples": 0, "n_tokens": 0,
                "mean_loss": 0.0, "epochs": 0,
                "train_time_s": 0.0, "load_time_s": load_time,
                "n_gpus": world_size, "throughput": 0.0,
            }
            with open(result_file, "w") as f:
                json.dump(result, f)
        del model, optimizer, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return

    dummy_entry = tokenized[0] if tokenized else {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attn_mask": torch.ones(1, 2, dtype=torch.long),
        "plen": 1,
        "adv": 0.0,
    }
    while len(tokenized) < n_valid_max_val:
        tokenized.append(None)

    ref_data: list = [None] * len(tokenized)
    pad_id = tokenizer.pad_token_id or 0
    ref_batch = min(_GRPO_REF_MICRO_BATCH, batch_size)

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(tokenized), ref_batch):
            batch_end = min(batch_start + ref_batch, len(tokenized))
            batch_items = tokenized[batch_start:batch_end]
            real_flags = [tk is not None for tk in batch_items]
            entries = [tk if tk is not None else dummy_entry
                       for tk in batch_items]

            max_len = max(e["input_ids"].shape[1] for e in entries)
            bsz = len(entries)
            batch_ids = torch.full(
                (bsz, max_len), pad_id, dtype=torch.long,
            )
            batch_attn = torch.zeros(bsz, max_len, dtype=torch.long)
            for j, e in enumerate(entries):
                slen = e["input_ids"].shape[1]
                batch_ids[j, :slen] = e["input_ids"][0]
                batch_attn[j, :slen] = e["attn_mask"][0]

            batch_ids = batch_ids.to(device)
            batch_attn = batch_attn.to(device)

            try:
                out = model(input_ids=batch_ids, attention_mask=batch_attn)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if is_main:
                    logger.warning(
                        "OOM in ref log-prob batch (bsz=%d, seq=%d), "
                        "falling back to sample-by-sample",
                        bsz, max_len,
                    )
                for j, e in enumerate(entries):
                    if not real_flags[j]:
                        continue
                    slen = e["input_ids"].shape[1]
                    one_ids = e["input_ids"].to(device)
                    one_attn = e["attn_mask"].to(device)
                    try:
                        one_out = model(
                            input_ids=one_ids, attention_mask=one_attn,
                        )
                        plen = e["plen"]
                        logits_j = one_out.logits[0, plen - 1:slen - 1, :]
                        target_j = one_ids[0, plen:slen]
                        lp = torch.log_softmax(logits_j, dim=-1)
                        per_tok = lp.gather(
                            -1, target_j.unsqueeze(-1),
                        ).squeeze(-1)
                        del one_out, logits_j, lp
                        if per_tok.numel() > 0:
                            ref_data[batch_start + j] = {
                                "input_ids": e["input_ids"],
                                "attn_mask": e["attn_mask"],
                                "plen": plen,
                                "ref_lp": per_tok.cpu(),
                                "adv": e["adv"],
                            }
                        del per_tok
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if is_main:
                            logger.warning(
                                "OOM on single sample (seq=%d), skipping",
                                slen,
                            )
                    finally:
                        del one_ids, one_attn
                        torch.cuda.empty_cache()
                continue

            for j, e in enumerate(entries):
                if not real_flags[j]:
                    continue
                plen = e["plen"]
                slen = e["input_ids"].shape[1]
                logits_j = out.logits[j, plen - 1:slen - 1, :]
                target_j = batch_ids[j, plen:slen]
                lp = torch.log_softmax(logits_j, dim=-1)
                per_tok = lp.gather(
                    -1, target_j.unsqueeze(-1),
                ).squeeze(-1)
                del logits_j, lp

                if per_tok.numel() == 0:
                    continue
                ref_data[batch_start + j] = {
                    "input_ids": e["input_ids"],
                    "attn_mask": e["attn_mask"],
                    "plen": plen,
                    "ref_lp": per_tok.cpu(),
                    "adv": e["adv"],
                }
            del out, batch_ids, batch_attn
            torch.cuda.empty_cache()
    model.train()

    ref_data = [rd for rd in ref_data if rd is not None]

    n_train_local = torch.tensor(
        len(ref_data), device=device, dtype=torch.long,
    )
    n_train_max = n_train_local.clone()
    dist.all_reduce(n_train_max, op=dist.ReduceOp.MAX)
    n_train_max_val = int(n_train_max.item())

    if n_train_max_val == 0:
        if is_main:
            logger.warning("No valid ref data, skipping training")
            result = {
                "n_samples": 0, "n_tokens": 0,
                "mean_loss": 0.0, "epochs": 0,
                "train_time_s": 0.0, "load_time_s": load_time,
                "n_gpus": world_size, "throughput": 0.0,
            }
            with open(result_file, "w") as f:
                json.dump(result, f)
        del model, optimizer, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return

    if is_main:
        logger.info(
            "Reference log-probs: %d/%d valid (%.1fs)",
            len(ref_data), n_my, time.time() - t_ref,
        )

    dummy_rd = ref_data[0] if ref_data else {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attn_mask": torch.ones(1, 2, dtype=torch.long),
        "plen": 1,
        "ref_lp": torch.zeros(1),
        "adv": 0.0,
    }
    while len(ref_data) < n_train_max_val:
        ref_data.append(None)

    accum_single = max(1, 4)
    eff_batch_size = batch_size
    total_loss = 0.0
    total_tokens = 0
    n_my_train = len(ref_data)
    t_train = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_samples = 0
        t_epoch = time.time()

        n_mini = max(1, (n_my_train + eff_batch_size - 1) // eff_batch_size)
        for mb in range(n_mini):
            mb_start = mb * eff_batch_size
            mb_end = min(mb_start + eff_batch_size, n_my_train)
            batch_items = ref_data[mb_start:mb_end]
            real_flags = [rd is not None for rd in batch_items]
            entries = [rd if rd is not None else dummy_rd
                       for rd in batch_items]

            max_len = max(e["input_ids"].shape[1] for e in entries)
            bsz = len(entries)
            batch_ids = torch.full(
                (bsz, max_len), pad_id, dtype=torch.long,
            )
            batch_attn = torch.zeros(bsz, max_len, dtype=torch.long)
            for j, e in enumerate(entries):
                slen = e["input_ids"].shape[1]
                batch_ids[j, :slen] = e["input_ids"][0]
                batch_attn[j, :slen] = e["attn_mask"][0]

            batch_ids = batch_ids.to(device)
            batch_attn = batch_attn.to(device)

            try:
                with torch.enable_grad():
                    out = model(
                        input_ids=batch_ids,
                        attention_mask=batch_attn,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                old_bs = eff_batch_size
                eff_batch_size = max(1, eff_batch_size // 2)
                if is_main:
                    logger.warning(
                        "OOM in training fwd (bsz=%d, seq=%d), "
                        "halving batch %d→%d and skipping mini-batch",
                        bsz, max_len, old_bs, eff_batch_size,
                    )
                optimizer.zero_grad()
                continue

            mb_losses: list = []
            for j, e in enumerate(entries):
                if not real_flags[j]:
                    mb_losses.append(out.logits[j, 0, 0] * 0.0)
                    continue

                plen = e["plen"]
                slen = e["input_ids"].shape[1]
                old_lp = e["ref_lp"].to(device)

                logits_j = out.logits[j, plen - 1:slen - 1, :]
                target_j = batch_ids[j, plen:slen]
                lp = torch.log_softmax(logits_j, dim=-1)
                per_tok = lp.gather(
                    -1, target_j.unsqueeze(-1),
                ).squeeze(-1)
                del logits_j, lp

                ratio = torch.exp(per_tok - old_lp)
                clipped = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio,
                )
                adv_t = torch.tensor(
                    e["adv"], device=device, dtype=per_tok.dtype,
                )
                if not torch.isfinite(adv_t).all():
                    mb_losses.append(out.logits[j, 0, 0] * 0.0)
                    continue
                surr = torch.min(ratio * adv_t, clipped * adv_t)
                loss = -surr.mean() + kl_coeff * (old_lp - per_tok).mean()
                if not torch.isfinite(loss):
                    mb_losses.append(out.logits[j, 0, 0] * 0.0)
                    continue

                mb_losses.append(loss)
                epoch_loss += loss.item()
                epoch_samples += 1
                total_tokens += per_tok.numel()

            if mb_losses:
                (torch.stack(mb_losses).mean() / accum_single).backward()

            del out, batch_ids, batch_attn
            torch.cuda.empty_cache()

            is_accum_boundary = ((mb + 1) % accum_single == 0) or (mb == n_mini - 1)
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += epoch_loss / max(epoch_samples, 1)

        if is_main:
            elapsed = time.time() - t_epoch
            rate = n_my / elapsed if elapsed > 0 else 0
            logger.info(
                "  FSDP [%s] epoch %d/%d: loss=%.4f (%.1f samples/s, %.1fs, bs=%d)",
                adapter_name, epoch + 1, epochs,
                epoch_loss / max(epoch_samples, 1), rate, elapsed,
                eff_batch_size,
            )

    train_time = time.time() - t_train

    save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True,
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, save_policy,
    ):
        state_dict = model.state_dict()

    if is_main:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        import re
        _adapter_key_re = re.compile(
            r"(lora_[AB])\." + re.escape(adapter_name) + r"\.(weight)"
        )
        lora_state = {}
        for k, v in state_dict.items():
            if "lora_" not in k:
                continue
            clean_key = _adapter_key_re.sub(r"\1.\2", k)
            lora_state[clean_key] = v

        if lora_state:
            try:
                from safetensors.torch import save_file
                save_file(lora_state, str(save_path / "adapter_model.safetensors"))
            except ImportError:
                torch.save(lora_state, str(save_path / "adapter_model.bin"))

            logger.info(
                "Saved %d LoRA tensors → %s", len(lora_state), save_path,
            )
        else:
            logger.warning("No LoRA parameters found to save!")

        mean_loss = total_loss / max(epochs, 1)
        throughput = n_total * epochs / train_time if train_time > 0 else 0
        logger.info(
            "FSDP GRPO [%s] done: %.1fs train (%.1fs load), "
            "%d tokens, loss=%.4f, %.1f samples/s",
            adapter_name, train_time, load_time,
            total_tokens, mean_loss, throughput,
        )

        result = {
            "n_samples": n_total,
            "n_tokens": total_tokens,
            "mean_loss": mean_loss,
            "epochs": epochs,
            "train_time_s": train_time,
            "load_time_s": load_time,
            "n_gpus": world_size,
            "throughput": throughput,
        }
        with open(result_file, "w") as f:
            json.dump(result, f)

    if io_log_dir and is_main:
        _write_debug_io(
            io_log_dir, adapter_name, prompts, completions, advantages,
        )

    del model, optimizer, tokenizer, state_dict
    gc.collect()
    torch.cuda.empty_cache()


def _fsdp_train_worker(rank: int, args: Dict[str, Any]) -> None:
    """Per-GPU FSDP training worker (spawned by :func:`run_fsdp_grpo`).

    Thin wrapper that initialises the NCCL process group, delegates to
    :func:`_train_one_adapter`, then tears down the group.
    """
    import datetime

    import torch
    import torch.distributed as dist

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args["master_port"])
    dist.init_process_group(
        "nccl", rank=rank, world_size=args["world_size"],
        timeout=datetime.timedelta(seconds=_NCCL_TIMEOUT_S),
    )

    try:
        _train_one_adapter(
            rank, device, rank == 0, args["world_size"],
            args["model_name"], args,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _fsdp_train_worker_multi(rank: int, args: Dict[str, Any]) -> None:
    """Per-GPU worker that trains multiple adapters with a persistent model.

    The base model is loaded and FSDP-wrapped **once**.  For each adapter
    job, LoRA weights are swapped in-place via ``summon_full_params`` and
    saved efficiently (only LoRA tensors, no full state-dict gather).

    This eliminates per-adapter overhead of model loading (~12 s),
    FSDP wrapping (~60-80 s) and full state-dict save (~100 s), reducing
    skill-bank Phase C.2 from ~12 min to ~2-3 min.
    """
    import datetime
    import functools

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args["master_port"])
    dist.init_process_group(
        "nccl", rank=rank, world_size=args["world_size"],
        timeout=datetime.timedelta(seconds=_NCCL_TIMEOUT_S),
    )

    is_main = rank == 0
    world_size = args["world_size"]
    model_name = args["model_name"]
    jobs = args["jobs"]

    try:
        # ── Phase 1: load model + FSDP wrap (done ONCE) ──────────────
        t_init = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": rank},
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        first_job = jobs[0]
        first_adapter = first_job["adapter_name"]
        adapter_path = Path(first_job["adapter_dir"])
        adapter_config_file = adapter_path / "adapter_config.json"

        if adapter_config_file.exists():
            model = PeftModel.from_pretrained(
                model, str(adapter_path),
                adapter_name=first_adapter,
                is_trainable=True,
            )
            if is_main:
                logger.info(
                    "Loaded adapter '%s' from %s", first_adapter, adapter_path,
                )
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj",
                ],
                inference_mode=False,
            )
            model = get_peft_model(model, lora_cfg, adapter_name=first_adapter)
            if is_main:
                logger.info(
                    "Created fresh adapter '%s' (r=16, alpha=32)",
                    first_adapter,
                )

        for n, p in model.named_parameters():
            is_lora = "lora" in n.lower()
            p.requires_grad = is_lora
            if is_lora and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        model.train()

        wrap_cls = _detect_wrap_cls(model)
        if wrap_cls:
            auto_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=wrap_cls,
            )
        else:
            auto_policy = None

        bf16_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=bf16_policy,
            device_id=rank,
            use_orig_params=True,
        )

        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.suppress_errors = True

        init_time = time.time() - t_init
        if is_main:
            n_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(
                "Persistent FSDP model ready (%d GPUs, %s trainable, "
                "%.1fs init)",
                world_size, f"{n_trainable:,}", init_time,
            )

        # ── Phase 2: train each adapter ──────────────────────────────
        for job_idx, job in enumerate(jobs):
            adapter_name = job["adapter_name"]

            if is_main:
                logger.info(
                    "Persistent job %d/%d: '%s' (%d samples)",
                    job_idx + 1, len(jobs), adapter_name,
                    len(job.get("prompts", [])),
                )

            if job_idx > 0:
                _load_or_init_lora_under_fsdp(
                    model, job["adapter_dir"], first_adapter,
                    device, is_main,
                )

            t_job = time.time()
            stats = _run_grpo_training_loop(
                model, tokenizer, rank, device, is_main, world_size,
                adapter_name,
                job["prompts"], job["completions"], job["advantages"],
                job["lr"], job["epochs"], job["batch_size"],
                job["clip_ratio"], job["kl_coeff"],
                accumulation_steps=job.get("accumulation_steps", 4),
            )

            if stats is None:
                stats = {
                    "n_samples": 0, "n_tokens": 0,
                    "mean_loss": 0.0, "epochs": 0, "train_time_s": 0.0,
                }

            _save_lora_under_fsdp(
                model, first_adapter, job["save_dir"], is_main,
            )

            job_time = time.time() - t_job
            stats["wall_time_s"] = job_time
            stats["load_time_s"] = init_time if job_idx == 0 else 0.0
            stats["n_gpus"] = world_size
            if stats["train_time_s"] > 0:
                stats["throughput"] = (
                    stats["n_samples"] * stats["epochs"]
                    / stats["train_time_s"]
                )
            else:
                stats["throughput"] = 0.0

            if is_main:
                result_file = job["result_file"]
                with open(result_file, "w") as f:
                    json.dump(stats, f)

                io_log_dir = job.get("io_log_dir")
                if io_log_dir:
                    _write_debug_io(
                        io_log_dir, adapter_name,
                        job["prompts"], job["completions"],
                        job["advantages"],
                    )

                logger.info(
                    "FSDP GRPO [%s] done: %.1fs (train %.1fs)",
                    adapter_name, job_time,
                    stats.get("train_time_s", 0),
                )

            dist.barrier()

            # Free optimizer states, ref tensors, and CUDA cache between
            # adapter jobs to prevent OOM when training 3+ adapters.
            gc.collect()
            torch.cuda.empty_cache()

        # ── Phase 3: cleanup ─────────────────────────────────────────
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _write_debug_io(
    io_log_dir: str,
    adapter_name: str,
    prompts: List[str],
    completions: List[str],
    advantages: List[float],
) -> None:
    try:
        io_dir = Path(io_log_dir) / "grpo"
        io_dir.mkdir(parents=True, exist_ok=True)
        fname = f"fsdp_{adapter_name}.jsonl"
        with open(io_dir / fname, "w") as f:
            for i, (p, c, a) in enumerate(zip(prompts, completions, advantages)):
                record = {
                    "ts": time.time(),
                    "adapter": adapter_name,
                    "sample_idx": i,
                    "prompt": p,
                    "prompt_len_chars": len(p),
                    "completion": c,
                    "completion_len_chars": len(c),
                    "advantage": a,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("FSDP debug I/O write failed: %s", exc)


def run_fsdp_grpo(
    gpu_ids: List[int],
    model_name: str,
    adapter_dir: str,
    adapter_name: str,
    prompts: List[str],
    completions: List[str],
    advantages: List[float],
    *,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    clip_ratio: float = 0.2,
    kl_coeff: float = 0.05,
    save_dir: Optional[str] = None,
    io_log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch FSDP GRPO training for one adapter across multiple GPUs.

    Blocks until training completes.  Returns training stats dict.

    Parameters
    ----------
    gpu_ids : list[int]
        Physical GPU indices (e.g. ``[4, 5, 6, 7]``).
    model_name : str
        HuggingFace model id (e.g. ``"Qwen/Qwen3-8B"``).
    adapter_dir : str
        Directory containing the LoRA adapter (or where a fresh one
        will be created).
    adapter_name : str
        PEFT adapter name (e.g. ``"skill_selection"``).
    prompts, completions, advantages :
        Parallel lists of training data.
    """
    import torch.multiprocessing as mp

    if not prompts:
        logger.warning("No training data for adapter '%s'", adapter_name)
        return {"n_samples": 0, "skipped": True}

    if save_dir is None:
        save_dir = adapter_dir

    world_size = len(gpu_ids)
    master_port = _find_free_port()

    result_fd, result_file = tempfile.mkstemp(
        suffix=".json", prefix="fsdp_result_",
    )
    os.close(result_fd)

    args = {
        "world_size": world_size,
        "master_port": master_port,
        "gpu_ids": gpu_ids,
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "adapter_name": adapter_name,
        "prompts": prompts,
        "completions": completions,
        "advantages": advantages,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "clip_ratio": clip_ratio,
        "kl_coeff": kl_coeff,
        "save_dir": save_dir,
        "result_file": result_file,
        "io_log_dir": io_log_dir,
    }

    logger.info(
        "Launching FSDP GRPO '%s' on %d GPUs %s (%d samples, %d epochs)",
        adapter_name, world_size, gpu_ids, len(prompts), epochs,
    )

    # Remap CUDA_VISIBLE_DEVICES so rank 0..N-1 map to the target GPUs
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    t0 = time.time()
    try:
        mp.spawn(
            _fsdp_train_worker,
            nprocs=world_size,
            args=(args,),
            join=True,
        )
    except Exception:
        try:
            import torch as _torch
            for _gid in gpu_ids:
                with _torch.cuda.device(_gid):
                    _torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        raise
    finally:
        if original_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    elapsed = time.time() - t0

    try:
        with open(result_file) as f:
            result = json.load(f)
        result["wall_time_s"] = elapsed
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to read FSDP result: %s", exc)
        result = {"error": str(exc), "wall_time_s": elapsed}
    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass

    logger.info(
        "FSDP GRPO '%s' complete: %.1fs wall time", adapter_name, elapsed,
    )
    return result


def run_fsdp_grpo_multi(
    gpu_ids: List[int],
    model_name: str,
    jobs: List[Dict[str, Any]],
    *,
    io_log_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Train multiple LoRA adapters in a single process spawn.

    The NCCL process group and spawned workers are reused across all
    adapter jobs, and subsequent model loads benefit from OS page cache
    (the 14B model is only read from disk on the first job).

    Parameters
    ----------
    gpu_ids : list[int]
        Physical GPU indices (e.g. ``[4, 5, 6, 7]``).
    model_name : str
        HuggingFace model id.
    jobs : list[dict]
        Each dict must contain: ``adapter_dir``, ``adapter_name``,
        ``prompts``, ``completions``, ``advantages``, ``lr``,
        ``epochs``, ``batch_size``, ``clip_ratio``, ``kl_coeff``,
        ``save_dir``.

    Returns
    -------
    list[dict]
        One result dict per job (same order as *jobs*).
    """
    import torch.multiprocessing as mp

    valid_jobs = [j for j in jobs if j.get("prompts")]
    if not valid_jobs:
        logger.warning("No jobs with training data, nothing to do")
        return [{"n_samples": 0, "skipped": True} for _ in jobs]

    world_size = len(gpu_ids)
    master_port = _find_free_port()

    result_files = []
    for j in valid_jobs:
        fd, rfile = tempfile.mkstemp(suffix=".json", prefix="fsdp_result_")
        os.close(fd)
        j["result_file"] = rfile
        j.setdefault("io_log_dir", io_log_dir)
        j.setdefault("save_dir", j["adapter_dir"])
        result_files.append(rfile)

    args = {
        "world_size": world_size,
        "master_port": master_port,
        "model_name": model_name,
        "jobs": valid_jobs,
    }

    adapter_names = [j["adapter_name"] for j in valid_jobs]
    logger.info(
        "Launching FSDP GRPO multi (%s) on %d GPUs %s",
        ", ".join(adapter_names), world_size, gpu_ids,
    )

    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    t0 = time.time()
    try:
        mp.spawn(
            _fsdp_train_worker_multi,
            nprocs=world_size,
            args=(args,),
            join=True,
        )
    except Exception:
        # On spawn failure (SIGABRT, OOM), force-clear GPU memory on
        # all training GPUs so the next step starts with a clean slate.
        try:
            import torch as _torch
            for _gid in gpu_ids:
                with _torch.cuda.device(_gid):
                    _torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        raise
    finally:
        if original_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    elapsed = time.time() - t0

    results = []
    for rfile in result_files:
        try:
            with open(rfile) as f:
                r = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Failed to read FSDP result: %s", exc)
            r = {"error": str(exc)}
        finally:
            try:
                os.unlink(rfile)
            except OSError:
                pass
        results.append(r)

    # Insert skipped placeholders for jobs that had no data
    full_results = []
    valid_iter = iter(results)
    for j in jobs:
        if j.get("prompts"):
            full_results.append(next(valid_iter))
        else:
            full_results.append({"n_samples": 0, "skipped": True})

    logger.info(
        "FSDP GRPO multi complete: %d adapters in %.1fs wall time",
        len(valid_jobs), elapsed,
    )
    return full_results
