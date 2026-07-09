"""Tests for :mod:`trainer.SFT.lora_targets` (T2.11 closure).

These tests cover the two key invariants that block T2.11 from
recurring:

1. The Qwen3.5 hybrid list **always** contains the GatedDeltaNet gating
   legs (``in_proj_z``, ``in_proj_b``, ``in_proj_a``).  A future
   refactor that drops them would silently regress ~13 % of cold-start
   adapter coverage and ~77 % of schema_gen coverage.
2. The classic-7 list is returned for every Qwen family that uses pure
   full-attention (Qwen2 / Qwen2.5 / Qwen3 / Qwen3-MoE / Qwen3-VL /
   Qwen3-VL-MoE).  Adding the GatedDeltaNet legs to those families is
   harmless (substring matches nothing) but the test keeps the
   expected output list documented and stable for downstream callers.
3. Explicit caller overrides win over auto-resolution.

The resolver also probes ``transformers.AutoConfig`` when only
``model_name_or_arch`` is given, but those code paths are mocked here
to keep the test offline.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ensure_target_module_importable():
    # The helper has no side-effects on import; smoke-check it once
    # so a failure here surfaces as a clean fixture error.
    import trainer.SFT.lora_targets as _ltargets  # noqa: F401


def test_qwen3_5_arch_detection():
    from trainer.SFT.lora_targets import is_qwen3_5_arch

    assert is_qwen3_5_arch("qwen3_5")
    assert is_qwen3_5_arch("qwen3_5_moe")
    assert is_qwen3_5_arch("qwen3_5_text")
    # Negative cases — every other Qwen family is full-attention.
    for arch in [
        "qwen2", "qwen2_5", "qwen2_5_vl",
        "qwen3", "qwen3_moe",
        "qwen3_vl", "qwen3_vl_moe",
        "llama", "mistral", "",
    ]:
        assert not is_qwen3_5_arch(arch), f"unexpected match on {arch!r}"


def test_qwen3_5_target_modules_contain_all_gated_delta_net_legs():
    """Regression guard against re-introducing the T2.11 drift."""
    from trainer.SFT.lora_targets import (
        QWEN3_5_LORA_TARGETS,
        resolve_target_modules,
    )

    # All five GatedDeltaNet linears + the four full_attention legs +
    # the three MLP legs must be present.
    required = {
        # full_attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # linear_attention (the one that bit us — make sure all five stick)
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        # mlp
        "gate_proj", "up_proj", "down_proj",
    }
    assert required.issubset(set(QWEN3_5_LORA_TARGETS))

    for arch in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text"):
        resolved = resolve_target_modules(text_arch=arch)
        assert required.issubset(set(resolved)), (
            f"resolve_target_modules({arch!r}) must include every "
            f"GatedDeltaNet leg; missing={required - set(resolved)}"
        )


def test_classic_qwen_target_modules_for_full_attention_families():
    """Qwen2 / Qwen3 / Qwen3-VL all share the seven-projection list."""
    from trainer.SFT.lora_targets import (
        QWEN_CLASSIC_LORA_TARGETS,
        resolve_target_modules,
    )

    expected = set(QWEN_CLASSIC_LORA_TARGETS)
    classic_archs = (
        "qwen2", "qwen2_5", "qwen2_5_vl",
        "qwen3", "qwen3_moe",
        "qwen3_vl", "qwen3_vl_moe",
    )
    for arch in classic_archs:
        resolved = set(resolve_target_modules(text_arch=arch))
        assert resolved == expected, (
            f"{arch!r} should resolve to the classic-7 list; got {sorted(resolved)}"
        )


def test_explicit_override_wins_over_auto_resolution():
    """A caller-provided explicit list short-circuits introspection."""
    from trainer.SFT.lora_targets import resolve_target_modules

    explicit = ["custom_proj", "another_proj"]
    # Even when the arch suggests Qwen3.5, the explicit list is returned
    # verbatim — the caller is responsible for knowing what they wrap.
    out = resolve_target_modules(text_arch="qwen3_5", explicit=explicit)
    assert out == explicit


def test_unknown_arch_falls_back_to_minimal_list():
    """Unknown architectures get a safe minimal target list."""
    from trainer.SFT.lora_targets import resolve_target_modules

    out = resolve_target_modules(text_arch="brand_new_architecture")
    # The minimal fallback used for non-Qwen families.
    assert out == ["q_proj", "v_proj"]


def test_assert_lora_coverage_passes_when_all_legs_wrapped():
    """Synthetic peft model exposing every required leg name → no warning."""
    import logging
    import types

    from trainer.SFT.lora_targets import assert_lora_coverage

    # Fake "wrapped LoRA layer" — just any object whose class name
    # contains 'lora' (case-insensitive).  The helper accepts both
    # PEFT's ``LoraLayer`` instances and substring-matched fallbacks.
    class FakeLoraLayer:  # noqa: D401 — test fixture
        pass
    FakeLoraLayer.__name__ = "FakeLoraLayer"

    leg_names = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    fake_modules = [
        (f"model.layers.0.self_attn.{name}", FakeLoraLayer())
        for name in leg_names
    ]
    fake_model = types.SimpleNamespace(
        named_modules=lambda: fake_modules,
    )

    summary = assert_lora_coverage(
        fake_model,
        model_arch="qwen3_5",
        require_strict=False,
        logger_=logging.getLogger("test"),
    )
    for name in leg_names:
        assert summary.get(name, 0) >= 1


def test_assert_lora_coverage_strict_raises_when_legs_missing():
    """Strict mode raises if any architecturally-required leg is missing."""
    import types

    from trainer.SFT.lora_targets import assert_lora_coverage

    class FakeLoraLayer:
        pass
    FakeLoraLayer.__name__ = "FakeLoraLayer"

    # Only the classic-7 wrapped — every GatedDeltaNet gating leg
    # missing.  This is exactly the T2.11 manifest.
    classic_only = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    fake_modules = [
        (f"model.layers.0.self_attn.{name}", FakeLoraLayer())
        for name in classic_only
    ]
    fake_model = types.SimpleNamespace(
        named_modules=lambda: fake_modules,
    )

    with pytest.raises(RuntimeError, match=r"recipe drift detected"):
        assert_lora_coverage(
            fake_model,
            model_arch="qwen3_5",
            require_strict=True,
        )
