"""Shared SFT speed-up helpers (liger-kernel, optimizer choice, TF32).

Two callers — :mod:`trainer.SFT.train` (cold-start adapters) and
:mod:`trainer.SFT.schema_gen.train` (visual-grounding adapter) — both
need the same answer to "which Liger patch fits this base model?" and
"which 8-bit optimizer is available?".  This module centralises those
decisions so a single bug fix lands everywhere.

Design notes
------------
* **Liger-kernel** ships dedicated patches for ``qwen3_5`` and
  ``qwen3_5_moe`` (verified on liger-kernel ≥0.6).  Calling the patch
  *before* loading the model replaces ``F.cross_entropy``, ``RMSNorm``,
  ``RoPE`` and a few other hot kernels with fused Triton equivalents.
  Empirically worth ~30-40 % SFT throughput.
* **bitsandbytes** ``paged_adamw_8bit`` cuts optimizer memory ~4×,
  freeing room to disable ``gradient_checkpointing`` (which itself
  costs 30-40 % throughput).  The 8-bit moments don't measurably hurt
  loss for SFT.
* **TF32** matmul on Hopper (H200) gives a "free" 1.5-2× speedup over
  fp32 for the residual fp32 ops (norm reductions etc.) that haven't
  been folded into bf16.
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF32 / matmul precision
# ---------------------------------------------------------------------------

def enable_tf32() -> None:
    """Set TF32 matmul precision to ``"high"`` — H200 / Ampere safe.

    A single line in production; kept as a helper so the same banner
    line goes to the log, making it visible in run summaries.
    """
    try:
        import torch

        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(
            "TF32 enabled (matmul_precision='high', cuda.matmul.allow_tf32=True)",
        )
    except Exception as exc:
        logger.warning("Could not enable TF32: %s", exc)


# ---------------------------------------------------------------------------
# Liger-kernel auto-application
# ---------------------------------------------------------------------------

#: Mapping from HF ``model_type`` → liger-kernel monkey-patch function.
#: Order matters — we longest-prefix-match against the live arch so
#: ``qwen3_5_moe`` resolves before ``qwen3_5``.
_LIGER_PATCHES = (
    ("qwen3_5_moe", "apply_liger_kernel_to_qwen3_5_moe"),
    ("qwen3_5", "apply_liger_kernel_to_qwen3_5"),
    ("qwen3_vl_moe", "apply_liger_kernel_to_qwen3_vl_moe"),
    ("qwen3_vl", "apply_liger_kernel_to_qwen3_vl"),
    ("qwen3_moe", "apply_liger_kernel_to_qwen3_moe"),
    ("qwen3", "apply_liger_kernel_to_qwen3"),
    ("qwen2_5_vl", "apply_liger_kernel_to_qwen2_5_vl"),
    ("qwen2_vl", "apply_liger_kernel_to_qwen2_vl"),
    ("qwen2", "apply_liger_kernel_to_qwen2"),
)


def liger_available() -> bool:
    """Return True iff liger-kernel can be imported."""
    try:
        importlib.import_module("liger_kernel.transformers")
        return True
    except Exception:
        return False


def apply_liger_kernel(
    model_arch: str,
    *,
    fused_loss: bool = True,
) -> Optional[str]:
    """Apply the liger-kernel monkey-patch matching *model_arch*.

    Must be called **before** the base model is instantiated (the patch
    swaps class-level methods).  Returns the name of the patch that
    fired, or ``None`` if no match / liger not installed.

    Parameters
    ----------
    model_arch
        Lower-case HF ``model_type`` string (e.g. ``"qwen3_5"``,
        ``"qwen3_5_moe"``).
    fused_loss
        When True (default, used by HF :class:`Trainer` callers like
        cold-start), also enable
        ``fused_linear_cross_entropy=True`` — this fuses
        ``lm_head`` + cross-entropy and **drops the materialised
        ``logits`` tensor from the model output** to save memory.
        When False (used by TRL ``SFTTrainer`` callers like
        :mod:`trainer.SFT.schema_gen.train`), the fused-CE patch is
        skipped because TRL's ``compute_loss`` reads
        ``outputs.logits[..., :-1, :]`` directly and crashes with
        ``TypeError: 'NoneType' object is not subscriptable`` when
        logits is None.  RMSNorm + SwiGLU fusions still apply, so
        most of the speedup is retained.
    """
    if not liger_available():
        logger.info("liger-kernel not installed — skipping kernel fusion.")
        return None

    arch = (model_arch or "").lower()
    if not arch:
        logger.info("apply_liger_kernel: empty model_arch, skipping.")
        return None

    try:
        from liger_kernel.transformers import monkey_patch as mp
    except Exception as exc:
        logger.warning("liger-kernel import failed: %s", exc)
        return None

    patch_kwargs = {
        "rope": False,                             # default off — small
        "cross_entropy": False,                    # let trainer own CE
        "fused_linear_cross_entropy": bool(fused_loss),
        "rms_norm": True,
        "swiglu": True,
    }

    for token, patch_name in _LIGER_PATCHES:
        if token in arch:
            patch = getattr(mp, patch_name, None)
            if patch is None:
                logger.warning(
                    "Liger patch '%s' not found for arch=%s; "
                    "liger-kernel version may be too old.",
                    patch_name, arch,
                )
                return None
            try:
                patch(**patch_kwargs)
                logger.info(
                    "Applied liger-kernel patch: %s (fused_loss=%s)",
                    patch_name, fused_loss,
                )
                return patch_name
            except TypeError:
                # Older liger-kernel versions don't expose every kwarg.
                # Fall back to the default invocation rather than skipping.
                try:
                    patch()
                    logger.info(
                        "Applied liger-kernel patch (default kwargs): %s",
                        patch_name,
                    )
                    return patch_name
                except Exception as exc:
                    logger.warning(
                        "Liger patch '%s' failed: %s — continuing without it.",
                        patch_name, exc,
                    )
                    return None
            except Exception as exc:
                logger.warning(
                    "Liger patch '%s' failed: %s — continuing without it.",
                    patch_name, exc,
                )
                return None

    logger.info("No liger-kernel patch matches arch=%s", arch)
    return None


# ---------------------------------------------------------------------------
# Optimizer pick
# ---------------------------------------------------------------------------

def bitsandbytes_available() -> bool:
    """Return True iff bitsandbytes can be imported."""
    try:
        importlib.import_module("bitsandbytes")
        return True
    except Exception:
        return False


def pick_optim(prefer_8bit: bool = True) -> str:
    """Pick a string compatible with HuggingFace ``TrainingArguments(optim=...)``.

    ``paged_adamw_8bit`` saves ~4× optimizer memory and lets us either
    increase batch size or disable gradient checkpointing.  Falls back
    to ``adamw_torch_fused`` (PyTorch's fused AdamW, ~10-15 % faster
    than ``adamw_torch``) when bitsandbytes isn't available.
    """
    if prefer_8bit and bitsandbytes_available():
        return "paged_adamw_8bit"
    return "adamw_torch_fused"


__all__ = [
    "enable_tf32",
    "liger_available",
    "apply_liger_kernel",
    "bitsandbytes_available",
    "pick_optim",
]
