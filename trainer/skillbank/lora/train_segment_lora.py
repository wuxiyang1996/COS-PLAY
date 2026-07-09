#!/usr/bin/env python3
"""Convenience wrapper: train the **segment** LoRA adapter.

Forwards all arguments to ``train_lora.py`` with ``--skill_function segment``.

Usage::

    python -m trainer.skillbank.lora.train_segment_lora \
        --data_path runs/datasets/segment_train.jsonl \
        --output_dir runs/lora_adapters/segment
"""

import sys

sys.argv.insert(1, "--skill_function")
sys.argv.insert(2, "segment")

from trainer.skillbank.lora.train_lora import main  # noqa: E402

if __name__ == "__main__":
    main()
