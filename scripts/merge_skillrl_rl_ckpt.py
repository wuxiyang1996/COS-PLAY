#!/usr/bin/env python3
"""Merge SkillRL's verl FSDP actor shards into a HuggingFace checkpoint.

Their scripts/model_merger.py breaks on torch 2.13 (pickled DeviceMesh has no
``_layout``). Every tensor in these shards is a DTensor with placement
Shard(dim=0), so the merge is a dim-0 concat of the local tensors per key.

Usage:
    python scripts/merge_skillrl_rl_ckpt.py \
        --snap /path/to/actor \
        --out /workspace/models/webshop-7b-rl-hf \
        --world-size 8
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="Path to actor/ directory")
    ap.add_argument("--out", required=True, help="Output HF model directory")
    ap.add_argument("--world-size", type=int, required=True)
    args = ap.parse_args()

    snap = Path(args.snap)
    out = Path(args.out)
    world_size = args.world_size
    out.mkdir(parents=True, exist_ok=True)

    shards = []
    for rank in range(world_size):
        p = snap / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"loading {p.name} ...", flush=True)
        shards.append(torch.load(p, map_location="cpu", weights_only=False))

    from torch.distributed.tensor import Replicate, Shard

    merged = {}
    for key in shards[0]:
        parts = []
        for sd in shards:
            v = sd[key]
            t = v._local_tensor if hasattr(v, "_local_tensor") else v
            placements = getattr(v, "placements", None)
            if placements is not None:
                p = placements[0]
                if isinstance(p, Replicate):
                    parts = [t]
                    break
                assert isinstance(p, Shard) and p.dim == 0, (
                    f"{key}: unexpected placement {placements}"
                )
            parts.append(t)
        full = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        merged[key] = full.to(torch.bfloat16)

    total = sum(v.numel() for v in merged.values())
    print(f"merged {len(merged)} tensors, {total/1e9:.2f}B params")

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(snap)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(merged, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    if getattr(config, "tie_word_embeddings", False) and "lm_head.weight" not in merged:
        model.tie_weights()

    model.save_pretrained(out, safe_serialization=True)

    for f in snap.glob("*"):
        if f.suffix in (".json", ".txt", ".jinja", ".model") and "extra_state" not in f.name:
            shutil.copy(f, out / f.name)
    print("saved to", out)


if __name__ == "__main__":
    main()
