"""Single source of truth for LoRA ``target_modules`` across Qwen families.

Background — T2.11 (LoRA target-modules drift)
----------------------------------------------
Qwen3.5 is the first Qwen family that ships a **hybrid** decoder block.
Each layer is one of two types, picked by ``config.layer_types[i]``:

* ``"linear_attention"`` — uses ``Qwen3_5GatedDeltaNet`` whose linears are
  ``linear_attn.{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj}``.
* ``"full_attention"``  — uses ``Qwen3_5Attention`` whose linears are
  ``self_attn.{q_proj, k_proj, v_proj, o_proj}``.

Both branches share an MLP with ``mlp.{gate_proj, up_proj, down_proj}``.

Older "classic-7" target lists carried forward from Qwen2/Qwen3 only
contained ``q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`` —
**none** of which match the linear-attention block.  PEFT's substring
matching is silent on a miss, so the trained adapter ends up with only
the full-attention + MLP delta and the linear-attention legs are never
wrapped.  This produced 23 % LoRA coverage on Qwen3.5-35B-A3B
(``schema_gen``) and 87 % on Qwen3.5-9B (cold-start) at load time.

The complete Qwen3.5 list adds **the three GatedDeltaNet gating legs**
(``in_proj_z/b/a``) and the fused ``in_proj_qkv`` + ``out_proj`` legs:

    QWEN3_5_LORA_TARGETS = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

For Qwen2 / Qwen2.5 / Qwen3 / Qwen3-MoE / Qwen3-VL / Qwen3-VL-MoE the
classic list is correct because every layer is full-attention.

This module is the **only** place where these lists should live.  Both
``trainer.SFT.config`` (cold-start adapters) and
``trainer.SFT.schema_gen.config`` (visual-grounding adapter) import from
here so a future fix can't drift again.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical target-module sets
# ---------------------------------------------------------------------------

#: Full Qwen3.5 LoRA target list — matches every named ``nn.Linear`` in
#: ``Qwen3_5DecoderLayer`` regardless of the layer's ``layer_type`` value.
QWEN3_5_LORA_TARGETS: Tuple[str, ...] = (
    # full_attention layer (Qwen3_5Attention)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # linear_attention layer (Qwen3_5GatedDeltaNet)
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    # MLP (both layer types share Qwen3_5MLP)
    "gate_proj", "up_proj", "down_proj",
)

#: Classic seven-projection list — correct for Qwen2 / Qwen2.5 / Qwen3
#: dense / Qwen3 MoE / Qwen3-VL / Qwen3-VL-MoE.  PEFT substring matching
#: catches MoE expert variants (e.g. ``mlp.experts.<j>.gate_proj``)
#: automatically.
QWEN_CLASSIC_LORA_TARGETS: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

#: Substrings of ``model.config.model_type`` (or ``text_config.model_type``)
#: that select the Qwen3.5 list.  We match on the full text_arch so
#: ``qwen3_5`` and ``qwen3_5_moe`` both resolve to the hybrid list.
_QWEN3_5_ARCH_TOKENS: Tuple[str, ...] = ("qwen3_5",)


def is_qwen3_5_arch(text_arch: str) -> bool:
    """Return True when *text_arch* is a Qwen3.5-family architecture."""
    arch = (text_arch or "").lower()
    return any(tok in arch for tok in _QWEN3_5_ARCH_TOKENS)


def resolve_target_modules(
    *,
    model_name_or_arch: Optional[str] = None,
    text_arch: Optional[str] = None,
    explicit: Optional[Iterable[str]] = None,
) -> List[str]:
    """Resolve LoRA ``target_modules`` for a Qwen-family base model.

    Precedence:

    1. If *explicit* is given, return it as a list (caller-provided override).
    2. If *text_arch* is provided, use it directly.  Cheapest path.
    3. Else, AutoConfig-introspect *model_name_or_arch* (HF id or local path).

    The function never raises on a probe failure — it falls back to the
    classic-7 list with a warning, since that's the safe behaviour for
    every Qwen family except 3.5.
    """
    if explicit is not None:
        return list(explicit)

    arch = (text_arch or "").lower()
    if not arch and model_name_or_arch:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(
                model_name_or_arch, trust_remote_code=True,
            )
            text_cfg = getattr(cfg, "text_config", cfg)
            arch = (getattr(text_cfg, "model_type", "") or "").lower()
        except Exception as exc:
            logger.warning(
                "lora_targets: AutoConfig probe failed for %r (%s); "
                "falling back to classic-7 list.",
                model_name_or_arch, exc,
            )

    if is_qwen3_5_arch(arch):
        return list(QWEN3_5_LORA_TARGETS)
    if "qwen" in arch:
        return list(QWEN_CLASSIC_LORA_TARGETS)
    return ["q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# Post-wrap sanity check (catches future recipe drift)
# ---------------------------------------------------------------------------

#: For each layer-type in Qwen3.5, the projections that should appear
#: somewhere inside the wrapped module set.  Used by
#: :func:`assert_lora_coverage` to abort if the recipe lost a leg.
_REQUIRED_PROJECTIONS_PER_LAYER_TYPE = {
    "full_attention": (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
    "linear_attention": (
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
}


def summarise_wrapped_modules(peft_model) -> dict:
    """Count how many LoRA layers wrap each projection name.

    Returns a dict ``{projection_name: n_wrapped}``.  Reads names off
    ``peft_model.named_modules()`` looking for the ``LoraLayer`` marker
    so the count is accurate regardless of which PEFT version injected
    the adapter.
    """
    counts: dict = {}
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        # Older peft layouts — fall back to substring detection.
        LoraLayer = None  # type: ignore

    for full_name, module in peft_model.named_modules():
        is_lora = False
        if LoraLayer is not None and isinstance(module, LoraLayer):
            is_lora = True
        else:
            is_lora = "lora" in module.__class__.__name__.lower()
        if not is_lora:
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def assert_lora_coverage(
    peft_model,
    *,
    model_arch: str,
    require_strict: bool = False,
    logger_: Optional[logging.Logger] = None,
) -> dict:
    """Verify that every architecturally-required projection got LoRA-wrapped.

    Returns the summary dict from :func:`summarise_wrapped_modules` and
    logs a warning (or raises in strict mode) if any required projection
    is missing.  Designed to fail-fast on a recipe drift like T2.11 so
    no future SFT run silently ships a 23 % adapter.

    Parameters
    ----------
    peft_model:
        Output of :func:`peft.get_peft_model`.
    model_arch:
        ``model.config.text_config.model_type`` (or ``model.config.model_type``).
    require_strict:
        When True, raise ``RuntimeError`` if any required leg is missing.
        When False (default), only warn.
    logger_:
        Logger to emit messages on.  Defaults to this module's logger.
    """
    log = logger_ or logger
    summary = summarise_wrapped_modules(peft_model)

    if is_qwen3_5_arch(model_arch):
        required = set(_REQUIRED_PROJECTIONS_PER_LAYER_TYPE["full_attention"]) | set(
            _REQUIRED_PROJECTIONS_PER_LAYER_TYPE["linear_attention"],
        )
    else:
        required = set(QWEN_CLASSIC_LORA_TARGETS)

    missing = sorted(name for name in required if summary.get(name, 0) == 0)
    log.info(
        "LoRA wrap summary (%s): %s",
        model_arch or "unknown",
        ", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
    )
    if missing:
        msg = (
            f"LoRA recipe drift detected on '{model_arch}': "
            f"required projection(s) {missing} have ZERO wrapped layers. "
            f"Update target_modules in trainer/SFT/lora_targets.py."
        )
        if require_strict:
            raise RuntimeError(msg)
        log.warning(msg)
    else:
        log.info(
            "LoRA wrap coverage OK — all %d required projection legs reached.",
            len(required),
        )
    return summary


__all__ = [
    "QWEN3_5_LORA_TARGETS",
    "QWEN_CLASSIC_LORA_TARGETS",
    "is_qwen3_5_arch",
    "resolve_target_modules",
    "summarise_wrapped_modules",
    "assert_lora_coverage",
]
