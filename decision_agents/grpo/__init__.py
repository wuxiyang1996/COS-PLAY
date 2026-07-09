"""Qwen3-VL-8B-Instruct actor for online inference + GRPO + LoRA.

The "online policy" flavour of the Actor Agent.  Built for:

* **inference at rollout time** — ``Qwen/Qwen3-VL-8B-Instruct`` served
  via :class:`trainer.coevolution.vllm_client.AsyncVLLMClient` with
  multi-LoRA hot-swap (the same vLLM instance used by the GRPO loop);
* **GRPO training** — emits per-step
  :class:`trainer.common.metrics.RolloutStep` records that the
  decision-agent GRPO trainer (:mod:`trainer.coevolution.grpo_training`)
  ingests directly.

Companion sub-package: :mod:`decision_agents.SFT` (GPT-4o-driven
collector that produces the cold-start data this actor is fine-tuned
on, before GRPO kicks in).
"""

from __future__ import annotations

from decision_agents.grpo.actor_qwen_vl import (
    DEFAULT_QWEN_VL_MODEL,
    QwenVLActor,
)
from decision_agents.grpo.rollout_logger import GRPORolloutLogger

__all__ = [
    "QwenVLActor",
    "DEFAULT_QWEN_VL_MODEL",
    "GRPORolloutLogger",
]
