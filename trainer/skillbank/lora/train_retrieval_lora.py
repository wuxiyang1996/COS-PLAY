#!/usr/bin/env python3
"""Convenience wrapper: train the **retrieval** LoRA adapter.

Forwards all arguments to ``train_lora.py`` with ``--skill_function retrieval``.

Usage::

    python -m trainer.skillbank.lora.train_retrieval_lora \
        --data_path runs/datasets/retrieval_train.jsonl \
        --output_dir runs/lora_adapters/retrieval
"""

import sys

sys.argv.insert(1, "--skill_function")
sys.argv.insert(2, "retrieval")

from trainer.skillbank.lora.train_lora import main  # noqa: E402

if __name__ == "__main__":
    main()
