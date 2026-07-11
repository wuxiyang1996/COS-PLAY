#!/usr/bin/env python3
"""Sequentially download WebShop-7B-RL actor shards with retries."""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/huggingface/hub")

from huggingface_hub import hf_hub_download

REPO = "Jianwen/Webshop-7B-RL"
META = [
    "actor/config.json",
    "actor/generation_config.json",
    "actor/tokenizer.json",
    "actor/tokenizer_config.json",
    "actor/special_tokens_map.json",
    "actor/vocab.json",
    "actor/merges.txt",
    "actor/added_tokens.json",
]
SHARDS = [f"actor/model_world_size_8_rank_{i}.pt" for i in range(8)]


def download_one(path: str) -> str:
    last = None
    for attempt in range(8):
        try:
            print(f"DL {path} attempt={attempt+1}", flush=True)
            out = hf_hub_download(REPO, path)
            print(f"OK {path} -> {out}", flush=True)
            return out
        except Exception as exc:
            last = exc
            wait = min(60, 8 * (attempt + 1))
            print(f"ERR {path}: {exc} | sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"FAILED {path}: {last}")


def main() -> None:
    for f in META + SHARDS:
        download_one(f)
    print("DONE sequential", flush=True)


if __name__ == "__main__":
    main()
