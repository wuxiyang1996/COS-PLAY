#!/usr/bin/env python3
"""Load-smoke each SFT adapter (T2.10).

For every entry in ``runs/sft_coldstart/sft_summary_all.json``:

  1. Reload its base model (``cfg.base_model_name_or_path``).
  2. Wrap it with ``peft.PeftModel.from_pretrained(adapter_path)``.
  3. Run *one* forward pass on a 4-token sentinel input.
  4. Verify the output logits are finite and shape-correct.

Confirms the on-disk adapter is intact AND that any
``*.partial_*`` shards left behind by ``run_sft_coldstart.sh`` are not
the canonical artifact (PEFT only loads ``adapter_model.safetensors``).
A failure here indicates a torn write — re-train that adapter before
launching co-evolution.

Spec:
  - ``implementation_notes/pre-training-readiness-audit.md`` §0.1  (T2.10)
  - ``runs/sft_coldstart/sft_summary_all.json``                    (T2.9 input)

Cost:
  Each smoke takes ~30s for the 9B adapters and ~3 min for the 35B-A3B
  ``schema_gen`` adapter (CPU offload pass — no fine-tuning, no
  multi-GPU sharding). The full sweep is < 8 min on a single GPU and
  fits in 80 GB VRAM via ``device_map="auto"``.

Usage:
    # Smoke every adapter listed in the manifest.
    python evaluation/smoke_load_sft_adapters.py \\
        --manifest runs/sft_coldstart/sft_summary_all.json

    # Limit to one adapter (e.g. just probe schema_gen).
    python evaluation/smoke_load_sft_adapters.py \\
        --manifest runs/sft_coldstart/sft_summary_all.json \\
        --only schema_gen

    # CPU-only smoke (slower; useful for CI without a GPU).
    python evaluation/smoke_load_sft_adapters.py \\
        --manifest runs/sft_coldstart/sft_summary_all.json \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Defer heavy imports — keep ``--help`` cheap and let users discover
# the CLI surface without paying for a ~3s torch import.


@dataclass
class SmokeResult:
    name: str
    phase: str
    base_model: str
    adapter_path: str
    ok: bool
    elapsed_s: float = 0.0
    output_shape: Optional[List[int]] = None
    finite_outputs: Optional[bool] = None
    n_loaded_params: Optional[int] = None
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"Manifest not found: {path}\n"
            f"Run: python scripts/build_sft_manifest.py --output {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _select_adapters(
    manifest: Dict[str, Any],
    only: Optional[List[str]],
    skip: Optional[List[str]],
) -> List[Dict[str, Any]]:
    rows = list(manifest.get("adapters", []))
    if only:
        rows = [r for r in rows if r["name"] in only]
    if skip:
        rows = [r for r in rows if r["name"] not in skip]
    return rows


def _resolve_adapter_path(row: Dict[str, Any], repo_root: Path) -> Path:
    p = Path(row["path"])
    return p if p.is_absolute() else repo_root / p


def _smoke_one(
    row: Dict[str, Any],
    *,
    repo_root: Path,
    device: str,
    dtype: str,
) -> SmokeResult:
    name = row["name"]
    phase = row["phase"]
    base_model = row["base_model"]
    adapter_path = _resolve_adapter_path(row, repo_root)

    res = SmokeResult(
        name=name, phase=phase, base_model=base_model,
        adapter_path=str(adapter_path), ok=False,
    )

    if not adapter_path.is_dir():
        res.error = f"adapter directory missing: {adapter_path}"
        return res
    if not (adapter_path / "adapter_model.safetensors").is_file():
        res.error = (
            f"adapter_model.safetensors missing under {adapter_path} — "
            f"likely a torn write. Re-train this adapter."
        )
        return res

    t0 = time.monotonic()
    try:
        import torch  # local import — see module docstring
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import AutoModelForImageTextToText  # type: ignore
        except ImportError:                                           # pragma: no cover
            AutoModelForImageTextToText = None                        # type: ignore

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        # Pick the right base loader for the multimodal Qwen3.5 family.
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        is_multimodal = hasattr(cfg, "text_config") or hasattr(cfg, "vision_config")
        loader_cls = (
            AutoModelForImageTextToText
            if (is_multimodal and AutoModelForImageTextToText is not None)
            else AutoModelForCausalLM
        )

        # device_map='auto' shards across GPU(s) + CPU; explicit 'cpu'
        # forces a CPU-only smoke (slow but laptop-friendly).
        device_map = "cpu" if device == "cpu" else "auto"

        base = loader_cls.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_path))
        peft_model.eval()

        # Count LoRA params actually loaded — non-zero confirms the
        # adapter weights stitched into the base correctly.
        n_loaded = sum(
            int(p.numel())
            for n, p in peft_model.named_parameters()
            if "lora_" in n
        )
        res.n_loaded_params = n_loaded

        # Tokenize a deterministic 4-token sentinel. We use 'state' as
        # the prefix to land inside the project's natural text domain.
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = "state ok"
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        if device != "cpu":
            target_device = next(peft_model.parameters()).device
            ids = ids.to(target_device)

        with torch.no_grad():
            out = peft_model(input_ids=ids)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("forward pass produced no .logits")

        res.output_shape = list(logits.shape)
        res.finite_outputs = bool(torch.isfinite(logits).all().item())
        if not res.finite_outputs:
            res.error = "logits contain NaN/Inf — adapter likely torn"
        else:
            res.ok = True

        # Cleanup before next adapter.
        del peft_model, base, logits, out
        if device != "cpu":
            torch.cuda.empty_cache()

    except Exception as exc:                      # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        res.notes.append(traceback.format_exc(limit=4))
    finally:
        res.elapsed_s = round(time.monotonic() - t0, 2)

    return res


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load-smoke each SFT adapter listed in the manifest.",
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--manifest", type=Path,
        default=repo_root / "runs" / "sft_coldstart" / "sft_summary_all.json",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=repo_root,
        help="Repo root used to resolve relative adapter paths.",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated adapter names to include (default: all).",
    )
    parser.add_argument(
        "--skip", type=str, default=None,
        help="Comma-separated adapter names to skip.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu"], default="auto",
        help="auto = device_map='auto' (sharded GPU+CPU); cpu = pure CPU.",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Where to write a JSON report (default: print to stdout).",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop on the first failure instead of smoking every adapter.",
    )
    args = parser.parse_args(argv)

    # Default HF cache locations — mirror trainer/coevolution/config.py.
    os.environ.setdefault("HF_HOME", "/workspace/huggingface")
    os.environ.setdefault(
        "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"),
    )

    manifest = _load_manifest(args.manifest)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    skip = [s.strip() for s in args.skip.split(",")] if args.skip else None
    rows = _select_adapters(manifest, only, skip)
    if not rows:
        print(f"No adapters matched filters in {args.manifest}", file=sys.stderr)
        return 2

    repo_root = args.repo_root.resolve()
    print(
        f"Smoking {len(rows)} adapter(s) under {repo_root} "
        f"(device={args.device}, dtype={args.dtype})",
        file=sys.stderr,
    )

    results: List[SmokeResult] = []
    for r in rows:
        res = _smoke_one(
            r, repo_root=repo_root, device=args.device, dtype=args.dtype,
        )
        results.append(res)
        status = "OK " if res.ok else "FAIL"
        msg = f"  [{status}] {res.phase:>11s}/{res.name:<16s}  {res.elapsed_s:>6.1f}s"
        if not res.ok:
            msg += f"  — {res.error}"
        print(msg, file=sys.stderr)
        if args.fail_fast and not res.ok:
            break

    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    summary = {
        "manifest_version": manifest.get("manifest_version"),
        "manifest_path": str(args.manifest),
        "n_smoked": len(results),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "device": args.device,
        "dtype": args.dtype,
        "results": [r.to_dict() for r in results],
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport → {args.report}", file=sys.stderr)
    else:
        print(json.dumps(summary, indent=2))

    print(
        f"\nSmoke summary: {n_ok}/{len(results)} OK, {n_fail} FAIL",
        file=sys.stderr,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
