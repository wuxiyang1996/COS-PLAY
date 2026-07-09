"""SFT cold-start training for decision and skill-bank LoRA adapters.

Trains all 5 LoRA adapters (skill_selection, action_taking, segment,
contract, curator) from teacher-labelled cold-start data so that the
co-evolution GRPO loop starts from a non-random checkpoint.
"""

from trainer.SFT.config import SFTConfig
from trainer.SFT.data_loader import load_all_adapter_datasets
from trainer.SFT.train import train_all_adapters

__all__ = ["SFTConfig", "load_all_adapter_datasets", "train_all_adapters"]
