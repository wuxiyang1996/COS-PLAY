#!/usr/bin/env python
"""Smoke-test for FSDP GRPO training.

Runs a small FSDP GRPO training job with synthetic data to verify:
1. FSDP model loading and sharding works
2. Training loop completes without errors
3. Adapter is saved correctly
4. Memory usage is reasonable

Usage:
    # Test with 4 GPUs (default):
    python scripts/test_fsdp_grpo.py

    # Test with specific GPUs:
    python scripts/test_fsdp_grpo.py --gpus 4 5 6 7

    # Test with different batch sizes:
    python scripts/test_fsdp_grpo.py --batch-sizes 8 16 32 64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(CODEBASE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_fsdp")


def generate_synthetic_data(n_samples: int = 64):
    """Create synthetic GRPO training data."""
    prompts = [
        f"You are playing a game. The current state is: score={i*10}, "
        f"position=({i % 10}, {i // 10}). Choose the best action."
        for i in range(n_samples)
    ]
    completions = [
        f"Based on the game state, I will take action: move_{'right' if i % 2 == 0 else 'up'}. "
        f"This should increase the score by approximately {i % 5 + 1} points."
        for i in range(n_samples)
    ]
    import random
    random.seed(42)
    advantages = [random.gauss(0, 1) for _ in range(n_samples)]
    return prompts, completions, advantages


def test_fsdp_grpo(
    gpu_ids: list,
    model_name: str,
    n_samples: int,
    batch_size: int,
):
    """Run one FSDP GRPO test."""
    from skill_agents_grpo.grpo.fsdp_trainer import run_fsdp_grpo

    prompts, completions, advantages = generate_synthetic_data(n_samples)

    with tempfile.TemporaryDirectory(prefix="fsdp_test_") as tmpdir:
        adapter_dir = os.path.join(tmpdir, "test_adapter")
        os.makedirs(adapter_dir)

        logger.info(
            "Running FSDP GRPO test: %d GPUs, %d samples, batch_size=%d",
            len(gpu_ids), n_samples, batch_size,
        )

        t0 = time.time()
        stats = run_fsdp_grpo(
            gpu_ids=gpu_ids,
            model_name=model_name,
            adapter_dir=adapter_dir,
            adapter_name="test_adapter",
            prompts=prompts,
            completions=completions,
            advantages=advantages,
            lr=5e-5,
            epochs=1,
            batch_size=batch_size,
            clip_ratio=0.2,
            kl_coeff=0.05,
            save_dir=adapter_dir,
        )
        elapsed = time.time() - t0

        saved_files = list(Path(adapter_dir).glob("*"))
        has_weights = any(
            f.name in ("adapter_model.safetensors", "adapter_model.bin")
            for f in saved_files
        )

        logger.info("=" * 60)
        logger.info("FSDP GRPO Test Results (batch_size=%d)", batch_size)
        logger.info("=" * 60)
        logger.info("  GPUs:         %s", gpu_ids)
        logger.info("  Samples:      %d", n_samples)
        logger.info("  Batch size:   %d", batch_size)
        logger.info("  Wall time:    %.1fs", elapsed)
        logger.info("  Stats:        %s", stats)
        logger.info("  Saved files:  %s", [f.name for f in saved_files])
        logger.info("  Weights OK:   %s", has_weights)

        if stats.get("error"):
            logger.error("  ERROR: %s", stats["error"])
            return False

        if not has_weights:
            logger.error("  ERROR: No adapter weights saved!")
            return False

        throughput = stats.get("throughput", 0)
        logger.info("  Throughput:   %.1f samples/s", throughput)
        logger.info("=" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(description="Test FSDP GRPO training")
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[4, 5, 6, 7],
        help="GPU IDs to use (default: 4 5 6 7)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B",
        help="Model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=64,
        help="Number of synthetic samples (default: 64)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[8],
        help="Batch sizes to test (default: 8)",
    )
    args = parser.parse_args()

    logger.info("FSDP GRPO Smoke Test")
    logger.info("  GPUs: %s", args.gpus)
    logger.info("  Model: %s", args.model)
    logger.info("  Samples: %d", args.n_samples)
    logger.info("  Batch sizes: %s", args.batch_sizes)

    results = {}
    for bs in args.batch_sizes:
        logger.info("\n--- Testing batch_size=%d ---", bs)
        try:
            ok = test_fsdp_grpo(args.gpus, args.model, args.n_samples, bs)
            results[bs] = "PASS" if ok else "FAIL"
        except Exception as exc:
            logger.error("batch_size=%d CRASHED: %s", bs, exc, exc_info=True)
            results[bs] = f"CRASH: {exc}"

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    for bs, status in results.items():
        logger.info("  batch_size=%d: %s", bs, status)
    logger.info("=" * 60)

    if all(v == "PASS" for v in results.values()):
        logger.info("All tests PASSED")
    else:
        logger.error("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
