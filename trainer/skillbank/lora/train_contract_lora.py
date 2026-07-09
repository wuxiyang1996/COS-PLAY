#!/usr/bin/env python3
"""Convenience wrapper: train the **contract** LoRA adapter.

Forwards all arguments to ``train_lora.py`` with ``--skill_function contract``.

Usage::

    python -m trainer.skillbank.lora.train_contract_lora \
        --data_path runs/datasets/contract_train.jsonl \
        --output_dir runs/lora_adapters/contract
"""

import sys

sys.argv.insert(1, "--skill_function")
sys.argv.insert(2, "contract")

from trainer.skillbank.lora.train_lora import main  # noqa: E402

if __name__ == "__main__":
    main()
