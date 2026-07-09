#!/usr/bin/env python3
"""Convenience wrapper: train the **boundary** LoRA adapter.

Forwards all arguments to ``train_lora.py`` with ``--skill_function boundary``.

Usage::

    python -m trainer.skillbank.lora.train_boundary_lora \
        --data_path runs/datasets/boundary_train.jsonl \
        --output_dir runs/lora_adapters/boundary
"""

import sys

sys.argv.insert(1, "--skill_function")
sys.argv.insert(2, "boundary")

from trainer.skillbank.lora.train_lora import main  # noqa: E402

if __name__ == "__main__":
    main()
