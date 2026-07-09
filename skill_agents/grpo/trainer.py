"""
GRPO LoRA trainer: reads the buffer, computes advantages, performs
policy-gradient updates on LoRA adapter weights.

Uses HuggingFace native PEFT + PyTorch optimizer.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from skill_agents.grpo.buffer import GRPOBuffer, GRPOSample
from skill_agents.grpo.config import GRPOConfig, StageGRPOConfig
from skill_agents.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


class GRPOLoRATrainer:
    """Train LoRA adapters using GRPO samples from the buffer.

    One trainer instance handles all adapters. Each ``train_step`` call
    processes one adapter's buffer, computes the GRPO loss, and performs
    a gradient step on the adapter's parameters.

    Parameters
    ----------
    llm : MultiLoraSkillBankLLM
        The shared model — provides ``log_probs()`` for gradient computation.
    config : GRPOConfig
        Per-stage hyperparameters.
    """

    def __init__(
        self,
        llm: Any,
        config: Optional[GRPOConfig] = None,
        io_log_dir: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._config = config or GRPOConfig()
        self._optimizers: Dict[str, Any] = {}
        self._step_counts: Dict[str, int] = {}
        self._io_log_dir = io_log_dir

    def _get_optimizer(self, adapter: SkillFunction, cfg: StageGRPOConfig) -> Any:
        """Lazily create an AdamW optimizer for the adapter's LoRA parameters."""
        import torch

        key = adapter.value
        if key not in self._optimizers:
            self._llm._activate_adapter(adapter)
            if self._llm._is_peft_model:
                params = [
                    p for n, p in self._llm._model.named_parameters()
                    if p.requires_grad and "lora" in n.lower()
                ]
            else:
                params = [p for p in self._llm._model.parameters() if p.requires_grad]

            if not params:
                logger.warning("No trainable parameters found for adapter %s", key)
                return None

            self._optimizers[key] = torch.optim.AdamW(params, lr=cfg.lr)
            self._step_counts[key] = 0
            logger.info(
                "Created optimizer for %s: %d params, lr=%.2e",
                key, len(params), cfg.lr,
            )
        return self._optimizers[key]

    def train_step(self, buffer: GRPOBuffer) -> Dict[str, Any]:
        """Run one GRPO training step per adapter that has data.

        Returns per-adapter training stats.
        """
        import torch

        stats: Dict[str, Any] = {}

        for adapter in buffer.adapters_with_data():
            if not self._config.is_enabled(adapter):
                continue

            samples = buffer.samples_for(adapter)
            if not samples:
                continue

            cfg = self._config.for_stage(adapter)
            adapter_stats = self._train_adapter(adapter, samples, cfg)
            stats[adapter.value] = adapter_stats

        return stats

    def _train_adapter(
        self,
        adapter: SkillFunction,
        samples: List[GRPOSample],
        cfg: StageGRPOConfig,
    ) -> Dict[str, Any]:
        """Train a single adapter on its buffered samples.

        Uses gradient accumulation with mini-batches of
        ``cfg.grad_accum_steps`` samples.  Each mini-batch computes its
        loss and immediately calls ``.backward()``, freeing the autograd
        graph before the next mini-batch.  This keeps peak GPU memory
        proportional to ``grad_accum_steps`` rather than the total sample
        count.
        """
        import torch

        optimizer = self._get_optimizer(adapter, cfg)
        if optimizer is None:
            return {"error": "no_trainable_params"}

        prompts: List[str] = []
        completions: List[str] = []
        advantages: List[float] = []

        for sample in samples:
            if not sample.prompt or not sample.completions:
                continue
            from skill_agents.grpo.advantage_utils import compute_grpo_group_advantages

            group_advantages = compute_grpo_group_advantages(
                sample.rewards, completions=sample.completions,
            )
            for comp, adv in zip(sample.completions, group_advantages):
                if comp:
                    prompts.append(sample.prompt)
                    completions.append(comp)
                    advantages.append(adv)

        if not prompts:
            return {"n_samples": 0, "skipped": True}

        self._dump_training_io(adapter, prompts, completions, advantages, samples)

        total_loss = 0.0
        n_tokens = 0
        n_samples = len(prompts)
        batch_size = max(cfg.grad_accum_steps, 1)
        n_batches = (n_samples + batch_size - 1) // batch_size
        t_train_start = time.time()
        has_batch_fn = hasattr(self._llm, "log_probs_batch")

        logger.info(
            "GRPO training [%s]: %d samples × %d epochs, batch_size=%d (%d batches/epoch)",
            adapter.value, n_samples, cfg.epochs_per_batch, batch_size, n_batches,
        )

        for epoch in range(cfg.epochs_per_batch):
            optimizer.zero_grad()
            epoch_loss_scalar = 0.0
            samples_in_epoch = 0
            t_epoch = time.time()

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_samples)
                b_prompts = prompts[start:end]
                b_completions = completions[start:end]
                b_advantages = advantages[start:end]

                if has_batch_fn:
                    log_ps = self._llm.log_probs_batch(
                        adapter, b_prompts, b_completions,
                    )
                else:
                    log_ps = [
                        self._llm.log_probs(adapter, p, c)
                        for p, c in zip(b_prompts, b_completions)
                    ]

                batch_loss_parts: List[torch.Tensor] = []
                for log_p, adv in zip(log_ps, b_advantages):
                    if log_p.numel() == 0:
                        continue

                    with torch.no_grad():
                        old_log_p = log_p.detach().clone()

                    ratio = torch.exp(log_p - old_log_p)
                    clipped = torch.clamp(
                        ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio,
                    )
                    adv_tensor = torch.tensor(adv, device=log_p.device)

                    surrogate = torch.min(ratio * adv_tensor, clipped * adv_tensor)
                    token_loss = -surrogate.mean()

                    kl = (old_log_p - log_p).mean()
                    token_loss = token_loss + cfg.kl_coeff * kl

                    batch_loss_parts.append(token_loss)
                    epoch_loss_scalar += token_loss.item()
                    samples_in_epoch += 1
                    n_tokens += log_p.numel()

                if batch_loss_parts:
                    batch_loss = torch.stack(batch_loss_parts).mean() / n_batches
                    batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._llm._model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                if (b + 1) % max(n_batches // 5, 1) == 0 or b == n_batches - 1:
                    elapsed = time.time() - t_epoch
                    rate = (end) / elapsed if elapsed > 0 else 0
                    avg_loss = epoch_loss_scalar / max(samples_in_epoch, 1)
                    eta = (n_samples - end) / rate if rate > 0 else 0
                    logger.info(
                        "  [%s] epoch %d/%d  batch %d/%d (%d samples, %.1f/s)  "
                        "loss=%.4f  eta=%.0fs",
                        adapter.value, epoch + 1, cfg.epochs_per_batch,
                        b + 1, n_batches, end, rate, avg_loss, eta,
                    )

            if samples_in_epoch > 0:
                total_loss += epoch_loss_scalar / n_samples

        train_elapsed = time.time() - t_train_start
        logger.info(
            "GRPO training [%s] done: %.1fs total (%.1f samples/s)",
            adapter.value, train_elapsed,
            n_samples * cfg.epochs_per_batch / max(train_elapsed, 1),
        )

        key = adapter.value
        self._step_counts[key] = self._step_counts.get(key, 0) + 1

        stats = {
            "n_samples": n_samples,
            "n_tokens": n_tokens,
            "epochs": cfg.epochs_per_batch,
            "batch_size": batch_size,
            "mean_loss": total_loss / max(cfg.epochs_per_batch, 1),
            "step": self._step_counts[key],
        }
        logger.info("GRPO train [%s] step %d: %s", key, self._step_counts[key], stats)
        return stats

    def _dump_training_io(
        self,
        adapter: SkillFunction,
        prompts: List[str],
        completions: List[str],
        advantages: List[float],
        samples: List[GRPOSample],
    ) -> None:
        if self._io_log_dir is None:
            return
        try:
            step = self._step_counts.get(adapter.value, 0)
            out_dir = Path(self._io_log_dir) / "grpo"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"step_{step:04d}_{adapter.value}.jsonl"
            path = out_dir / fname
            with open(path, "w", encoding="utf-8") as f:
                for i, (p, c, a) in enumerate(zip(prompts, completions, advantages)):
                    record = {
                        "ts": time.time(),
                        "adapter": adapter.value,
                        "grpo_step": step,
                        "sample_idx": i,
                        "prompt": p,
                        "prompt_len_chars": len(p),
                        "completion": c,
                        "completion_len_chars": len(c),
                        "advantage": a,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "Debug I/O: wrote %d GRPO samples for %s → %s",
                len(prompts), adapter.value, path,
            )
        except Exception as exc:
            logger.warning("Failed to write GRPO debug I/O: %s", exc)

