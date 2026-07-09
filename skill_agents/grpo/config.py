"""Per-stage GRPO hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from skill_agents.lora.skill_function import SkillFunction


@dataclass
class StageGRPOConfig:
    """GRPO hyperparameters for a single stage / adapter."""

    group_size: int = 4
    clip_ratio: float = 0.2
    kl_coeff: float = 0.05
    lr: float = 5e-5
    epochs_per_batch: int = 2
    temperature: float = 0.7
    max_buffer_size: int = 256
    grad_accum_steps: int = 8
    enabled: bool = True


@dataclass
class GRPOConfig:
    """Top-level GRPO configuration for all wrapped stages."""

    stage_configs: Dict[str, StageGRPOConfig] = field(default_factory=lambda: {
        SkillFunction.CONTRACT.value: StageGRPOConfig(
            group_size=4, kl_coeff=0.05, lr=5e-5, epochs_per_batch=2,
        ),
        SkillFunction.CURATOR.value: StageGRPOConfig(
            group_size=4, kl_coeff=0.05, lr=5e-5, epochs_per_batch=2,
        ),
        SkillFunction.SEGMENT.value: StageGRPOConfig(
            group_size=4, kl_coeff=0.02, lr=3e-5, epochs_per_batch=3,
        ),
    })

    def for_stage(self, function: SkillFunction) -> StageGRPOConfig:
        return self.stage_configs.get(function.value, StageGRPOConfig())

    def is_enabled(self, function: SkillFunction) -> bool:
        cfg = self.stage_configs.get(function.value)
        return cfg is not None and cfg.enabled
