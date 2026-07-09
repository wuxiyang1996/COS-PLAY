"""
GRPO wrapper infrastructure for skill-bank LoRA adapters.

Wraps existing LLM call points with a generic GRPO sampler:
  - ``GRPOBuffer``: stores (adapter, prompt, completions, rewards) tuples.
  - ``GRPOCallWrapper``: intercepts an LLM function, samples G times,
    evaluates via a reward function, stores results, returns the best.
  - ``GRPOLoRATrainer``: reads the buffer, computes GRPO advantages,
    updates LoRA adapter weights via policy-gradient loss.
"""

from skill_agents.grpo.buffer import GRPOBuffer, GRPOSample
from skill_agents.grpo.grpo_outputs import SkillBankLLMOutput, default_grpo_training_completion
from skill_agents.grpo.wrapper import GRPOCallWrapper
from skill_agents.grpo.trainer import GRPOLoRATrainer
from skill_agents.grpo.config import GRPOConfig

__all__ = [
    "GRPOBuffer",
    "GRPOSample",
    "GRPOCallWrapper",
    "GRPOLoRATrainer",
    "GRPOConfig",
    "SkillBankLLMOutput",
    "default_grpo_training_completion",
]
