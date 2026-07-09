"""
Configuration for the multi-LoRA skill-bank LLM (GRPO edition).

One shared Qwen3-8B backbone, 3 GRPO-trained LoRA adapters
(segment, contract, curator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from skill_agents_grpo.lora.skill_function import SkillFunction


@dataclass
class MultiLoraConfig:
    """All settings for the shared base model + GRPO-trained LoRA adapters.

    Parameters
    ----------
    base_model_name_or_path : str
        HuggingFace model id or local path for Qwen3-8B.
    adapter_paths : dict
        ``{function_name: path_to_adapter}``.  Active GRPO keys:
        ``"segment"``, ``"contract"``, ``"curator"``.
        Legacy keys ``"boundary"`` and ``"retrieval"`` are accepted
        but not GRPO-trained.
    default_function : str
        Fallback function used when none is specified.
    device : str
        ``"auto"`` for accelerate device-map, or ``"cuda:0"`` etc.
    dtype : str
        ``"bfloat16"`` | ``"float16"`` | ``"float32"``.
    max_new_tokens : int
        Default generation length.
    temperature : float
        Default sampling temperature.
    top_p : float
        Default nucleus sampling parameter.
    allow_fallback_to_base_model : bool
        If True, generate with the base model when the requested adapter
        is missing.  If False, raise an error.
    devices : list[int] | None
        GPU indices to spread the model across.  When provided, overrides
        ``device`` and builds a ``max_memory`` map so that
        ``from_pretrained(device_map="auto", max_memory=...)`` distributes
        layers across exactly these GPUs.
    gradient_checkpointing : bool
        Enable gradient checkpointing to trade compute for memory during
        the backward pass.  Recommended for GRPO training.
    """

    base_model_name_or_path: str = "Qwen/Qwen3-8B"
    adapter_paths: Dict[str, str] = field(default_factory=dict)
    default_function: str = "boundary"
    device: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    allow_fallback_to_base_model: bool = True
    devices: Optional[List[int]] = None
    gradient_checkpointing: bool = False

    def adapter_path_for(self, fn: SkillFunction) -> Optional[str]:
        return self.adapter_paths.get(fn.value)

    def has_adapter(self, fn: SkillFunction) -> bool:
        path = self.adapter_path_for(fn)
        return path is not None and Path(path).exists()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiLoraConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_name_or_path": self.base_model_name_or_path,
            "adapter_paths": dict(self.adapter_paths),
            "default_function": self.default_function,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "allow_fallback_to_base_model": self.allow_fallback_to_base_model,
        }


@dataclass
class LoraTrainingConfig:
    """Hyperparameters for training a single function-specific LoRA adapter.

    Parameters
    ----------
    skill_function : str
        Which adapter to train (``"segment"`` | ``"contract"`` |
        ``"curator"`` for GRPO; ``"boundary"`` | ``"retrieval"``
        also accepted for legacy SFT training).
    base_model_name_or_path : str
        Same base model as inference.
    output_dir : str
        Where to save the trained adapter.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha scaling.
    lora_dropout : float
        LoRA dropout rate.
    target_modules : list[str] | None
        Which linear layers to adapt (None = PEFT auto-detect for Qwen).
    learning_rate : float
    num_train_epochs : int
    per_device_train_batch_size : int
    gradient_accumulation_steps : int
    max_seq_length : int
    warmup_ratio : float
    logging_steps : int
    save_steps : int
    bf16 : bool
    """

    skill_function: str = "boundary"
    base_model_name_or_path: str = "Qwen/Qwen3-8B"
    output_dir: str = "runs/lora_adapters/boundary"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list] = None
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 200
    bf16: bool = True

    @property
    def function_enum(self) -> SkillFunction:
        return SkillFunction.from_str(self.skill_function)
