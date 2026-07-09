#!/usr/bin/env python3
"""Re-key cold-start LoRA safetensors so they match the VLM-loader path.

Problem (T2.13, 2026-05-03)
---------------------------
``trainer/SFT/train.py`` loads the base via ``AutoModelForCausalLM`` which
returns the language-only sub-model ``Qwen3_5ForCausalLM``.  PEFT then
attaches LoRA at ``model.layers.<i>.<...>`` and the saved checkpoint keys
look like::

    base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight

But the production loaders — ``trainer/coevolution/prepare_adapters``,
``trainer/SFT/schema_gen/train.py``, ``evaluation/smoke_load_sft_adapters``
and the vLLM ``--model Qwen/Qwen3.5-9B`` server — all use
``AutoModelForImageTextToText`` which returns ``Qwen3_5ForConditionalGeneration``
where the language tower lives under ``model.language_model.layers.<i>``.
PEFT therefore expects keys like::

    base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.lora_A.weight

The structural prefix mismatch causes ``PeftModel.from_pretrained`` to
emit a "Found missing adapter keys" warning and silently keep the LoRA
slots at their initialization (``lora_B`` ≡ 0 → delta = 0).  In other
words: **vLLM serves the base model with no LoRA effect, so all the
cold-start SFT signal is dropped at the boundary.**

The 35B-A3B ``schema_gen`` adapter is unaffected because its training
script already uses ``AutoModelForImageTextToText``.

Fix
---
This script rewrites the saved ``adapter_model.safetensors`` keys::

    base_model.model.model.layers.<i>.<rest>
        →
    base_model.model.model.language_model.layers.<i>.<rest>

It preserves dtypes / shapes / values, and writes a backup
``adapter_model.safetensors.pre_vlm_remap`` so the original is recoverable.
``adapter_config.json`` does not need editing — ``target_modules`` are
LEAF NAME PATTERNS (e.g. ``q_proj``), not full paths.

Usage::

    # Scratch / dry-run: copy first, transform copy, leave original alone.
    python evaluation/fix_lora_keys_for_vlm_loader.py \\
        --adapter runs/sft_coldstart_20260502_025737/decision/action_taking/action_taking \\
        --output  /tmp/action_taking_remapped --copy

    # In-place fix on all 5 cold-start adapters (creates .pre_vlm_remap backup):
    python evaluation/fix_lora_keys_for_vlm_loader.py \\
        --in-place \\
        --adapter runs/sft_coldstart_20260502_025737/decision/action_taking/action_taking \\
        --adapter runs/sft_coldstart_20260502_025737/decision/skill_selection/skill_selection \\
        --adapter runs/sft_coldstart_20260502_025737/skillbank/segment/segment \\
        --adapter runs/sft_coldstart_20260502_025737/skillbank/contract/contract \\
        --adapter runs/sft_coldstart_20260502_025737/skillbank/curator/curator
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from safetensors import safe_open
from safetensors.torch import save_file

OLD_PREFIX = "base_model.model.model.layers."
NEW_PREFIX = "base_model.model.model.language_model.layers."

# ``adapter_config.json`` may also store ``modules_to_save`` paths and
# legacy meta — peek at them for diagnostic purposes only.
SAFETENSORS_FILE = "adapter_model.safetensors"
BACKUP_SUFFIX = ".pre_vlm_remap"


def _classify_keys(keys: List[str]) -> Tuple[int, int, int]:
    """Return (n_lm_only, n_vlm, n_other)."""
    lm_only = sum(1 for k in keys if k.startswith(OLD_PREFIX))
    vlm = sum(1 for k in keys if k.startswith(NEW_PREFIX))
    other = len(keys) - lm_only - vlm
    return lm_only, vlm, other


def remap_one(adapter_dir: Path, *, output_dir: Path | None, in_place: bool) -> Dict[str, object]:
    """Rewrite the safetensors of one adapter directory.

    Returns a per-adapter report dict.
    """
    src = adapter_dir / SAFETENSORS_FILE
    if not src.is_file():
        return {"adapter_dir": str(adapter_dir), "ok": False, "error": f"missing {SAFETENSORS_FILE}"}

    with safe_open(str(src), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        # Snapshot tensor data before any rename.
        data: Dict[str, "torch.Tensor"] = {k: f.get_tensor(k) for k in all_keys}
        meta: Dict[str, str] = dict(f.metadata() or {})

    n_lm, n_vlm, n_other = _classify_keys(all_keys)
    n_total = len(all_keys)

    if n_lm == 0 and n_vlm == n_total:
        # Already VLM-keyed — schema_gen is in this state.  No-op.
        return {
            "adapter_dir": str(adapter_dir),
            "ok": True,
            "skipped": True,
            "reason": "already VLM-keyed (no remap needed)",
            "n_total": n_total,
            "n_lm_only": n_lm,
            "n_vlm": n_vlm,
            "n_other": n_other,
        }
    if n_lm == 0 and n_vlm == 0:
        return {
            "adapter_dir": str(adapter_dir),
            "ok": False,
            "error": (
                f"no keys match either prefix — unfamiliar layout "
                f"(sample={all_keys[0] if all_keys else '<empty>'})"
            ),
        }
    if n_other > 0:
        # Some keys (e.g. embed_tokens) live outside .layers.* and don't
        # need remapping — keep them as-is and warn so we can audit.
        pass

    # Remap.
    new_data: Dict[str, "torch.Tensor"] = {}
    n_renamed = 0
    for k, t in data.items():
        if k.startswith(OLD_PREFIX):
            new_k = NEW_PREFIX + k[len(OLD_PREFIX):]
            new_data[new_k] = t
            n_renamed += 1
        else:
            new_data[k] = t

    # Choose destination.
    if in_place:
        # Backup first.
        bak = src.with_suffix(src.suffix + BACKUP_SUFFIX)
        if not bak.exists():
            shutil.copy2(src, bak)
        dst = src
    else:
        if output_dir is None:
            return {"adapter_dir": str(adapter_dir), "ok": False, "error": "non-in-place run requires --output"}
        output_dir.mkdir(parents=True, exist_ok=True)
        # Copy companion files (adapter_config.json, README, etc.) first.
        for sibling in adapter_dir.iterdir():
            if sibling.name == SAFETENSORS_FILE:
                continue
            target = output_dir / sibling.name
            if sibling.is_dir():
                shutil.copytree(sibling, target, dirs_exist_ok=True)
            else:
                shutil.copy2(sibling, target)
        dst = output_dir / SAFETENSORS_FILE

    save_file(new_data, str(dst), metadata=meta or None)

    return {
        "adapter_dir": str(adapter_dir),
        "ok": True,
        "skipped": False,
        "in_place": in_place,
        "output": str(dst),
        "n_total": n_total,
        "n_lm_only_before": n_lm,
        "n_vlm_before": n_vlm,
        "n_other_before": n_other,
        "n_renamed": n_renamed,
        "backup": str(src.with_suffix(src.suffix + BACKUP_SUFFIX)) if in_place else None,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-key SFT cold-start LoRA adapters for the VLM loader path.",
    )
    parser.add_argument(
        "--adapter", action="append", required=True, type=Path,
        help="Adapter directory (containing adapter_model.safetensors). May be repeated.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (only used when --in-place is NOT set; "
             "for multi-adapter runs in non-in-place mode, a sub-dir per adapter is created).",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Rewrite the safetensors in place (creates a .pre_vlm_remap backup).",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Convenience alias for non-in-place mode (the default).",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Write a JSON report to this path.",
    )
    args = parser.parse_args(argv)

    if args.in_place and args.output:
        print("ERROR: --in-place and --output are mutually exclusive.", file=sys.stderr)
        return 2

    if not args.in_place and not args.output:
        print("ERROR: pass --in-place or --output <DIR>.", file=sys.stderr)
        return 2

    reports = []
    for i, adapter_dir in enumerate(args.adapter):
        if args.in_place:
            rep = remap_one(adapter_dir, output_dir=None, in_place=True)
        else:
            # If multiple adapters in non-in-place mode, fan out under args.output/<basename>.
            if len(args.adapter) > 1:
                out = args.output / adapter_dir.name
            else:
                out = args.output
            rep = remap_one(adapter_dir, output_dir=out, in_place=False)
        reports.append(rep)
        status = "OK" if rep.get("ok") else "FAIL"
        skipped = rep.get("skipped")
        action = (
            "skipped (already VLM-keyed)"
            if skipped
            else f"renamed {rep.get('n_renamed', 0)}/{rep.get('n_total', 0)} keys"
        )
        print(f"[{status}] {adapter_dir.name}: {action}", file=sys.stderr)
        if not rep.get("ok"):
            print(f"        error: {rep.get('error')}", file=sys.stderr)

    summary = {
        "n_adapters": len(reports),
        "n_ok": sum(1 for r in reports if r.get("ok")),
        "n_fail": sum(1 for r in reports if not r.get("ok")),
        "n_skipped": sum(1 for r in reports if r.get("skipped")),
        "results": reports,
    }
    text = json.dumps(summary, indent=2, default=str)
    if args.report:
        args.report.write_text(text)
    else:
        print(text)
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
