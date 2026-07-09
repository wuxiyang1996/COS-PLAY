"""
Multi-LoRA infrastructure for the skill-bank agent (GRPO edition).

One shared Qwen3-8B backbone + 3 GRPO-trained LoRA adapters
(segment, contract, curator).  BOUNDARY and RETRIEVAL enum values
are retained for backward compatibility but are not GRPO-trained.

GRPO training uses ``log_probs()`` on ``MultiLoraSkillBankLLM`` to
compute per-token log-probabilities with gradients, enabling
policy-gradient updates on LoRA adapter weights.

Quick start::

    from skill_agents_grpo.lora import MultiLoraSkillBankLLM, MultiLoraConfig, SkillFunction

    cfg = MultiLoraConfig(adapter_paths={"contract": "path/to/adapter"})
    llm = MultiLoraSkillBankLLM(cfg)
    text = llm.generate(SkillFunction.CONTRACT, "Your prompt here")
"""

from skill_agents_grpo.lora.skill_function import SkillFunction
from skill_agents_grpo.lora.config import MultiLoraConfig, LoraTrainingConfig
from skill_agents_grpo.lora.model import MultiLoraSkillBankLLM

__all__ = [
    "SkillFunction",
    "MultiLoraConfig",
    "LoraTrainingConfig",
    "MultiLoraSkillBankLLM",
]
