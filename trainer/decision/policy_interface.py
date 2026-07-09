"""
Policy interface for the VLM Decision Agent.

Two modes:
  1. LLMPolicy — original API-based policy (for standalone / evaluation use)
  2. VERLActorProxy — thin proxy around VERL's distributed actor worker group
     (vLLM/sglang-backed).  In VERL training, the actor is managed by
     RayPPOTrainer; this proxy is only used if Game-AI code needs to call
     the actor outside the normal VERL rollout loop (e.g. evaluation).

VERL integration:
  During training, VERL's TrajectoryCollector handles all action sampling
  and logprob computation via actor_rollout_wg.generate_sequences() and
  actor_rollout_wg.compute_log_prob().  The custom PolicyInterface is NOT
  used in the hot path — VERL manages everything.

  The VERLActorProxy is provided for evaluation / standalone scenarios
  where the caller wants a PolicyInterface-compatible API but the model
  is served by VERL's vLLM/sglang inference engine.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PolicyOutput:
    """Output of a single policy forward pass."""

    action: str
    logprob: float
    entropy: float = 0.0
    value: Optional[float] = None
    raw_response: str = ""
    metadata: Optional[Dict[str, Any]] = None


class PolicyInterface(ABC):
    """Abstract interface for a trainable policy.

    Concrete implementations wrap a specific LLM/VLM backend and provide:
      - sample(): stochastic action sampling with logprobs
      - logprob(): compute logprob of a given action under current policy
      - update(): apply a gradient step given GRPO loss
    """

    @abstractmethod
    def sample(
        self,
        observation: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> PolicyOutput:
        """Sample an action from the policy."""

    @abstractmethod
    def logprob(
        self,
        observation: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute log-probability of a given action under current policy."""

    @abstractmethod
    def batch_logprobs(
        self,
        observations: List[str],
        actions: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """Compute log-probabilities for a batch of (obs, action) pairs."""

    @abstractmethod
    def update(self, loss: float, grads: Optional[Any] = None) -> Dict[str, float]:
        """Apply one gradient step. Returns training stats."""

    @abstractmethod
    def get_parameters(self) -> Any:
        """Return current model parameters (for checkpointing)."""

    @abstractmethod
    def load_parameters(self, params: Any) -> None:
        """Load model parameters (for checkpoint restoration)."""


class LLMPolicy(PolicyInterface):
    """Concrete policy implementation wrapping an LLM API.

    Used for standalone training / evaluation without VERL.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        prompt_builder: Any = None,
        lr: float = 1e-5,
    ):
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.lr = lr
        self._parameters: Dict[str, Any] = {}
        self._step_count = 0

    def sample(
        self,
        observation: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> PolicyOutput:
        try:
            from API_func import ask_model
        except ImportError:
            return PolicyOutput(action="no-op", logprob=-1.0, raw_response="")

        prompt = self._build_prompt(observation, context)
        response = ask_model(
            prompt, model=self.model_name,
            temperature=temperature, max_tokens=400,
        )
        action = self._extract_action(response or "")
        lp = self._estimate_logprob(prompt, action, temperature)

        return PolicyOutput(
            action=action, logprob=lp,
            raw_response=response or "",
            metadata={"prompt_len": len(prompt)},
        )

    def logprob(self, observation, action, context=None):
        prompt = self._build_prompt(observation, context)
        return self._estimate_logprob(prompt, action, temperature=0.0)

    def batch_logprobs(self, observations, actions, contexts=None):
        contexts = contexts or [None] * len(observations)
        return [
            self.logprob(obs, act, ctx)
            for obs, act, ctx in zip(observations, actions, contexts)
        ]

    def update(self, loss, grads=None):
        self._step_count += 1
        return {"loss": loss, "step": self._step_count, "lr": self.lr}

    def get_parameters(self):
        return dict(self._parameters)

    def load_parameters(self, params):
        if isinstance(params, dict):
            self._parameters = params

    def _build_prompt(self, observation, context):
        if self.prompt_builder is not None:
            return self.prompt_builder(observation, context)
        parts = [f"Observation: {observation[:2000]}"]
        if context:
            if context.get("skill_cards"):
                parts.append(f"Available skills: {context['skill_cards']}")
            if context.get("active_skill"):
                parts.append(f"Active skill: {context['active_skill']}")
        parts.append("Choose ONE action:")
        return "\n".join(parts)

    @staticmethod
    def _extract_action(response: str) -> str:
        import re
        m = re.search(r'"action"\s*:\s*"([^"]+)"', response)
        if m:
            return m.group(1)
        m = re.search(r"TOOL:\s*take_action.*?\"action\":\s*\"([^\"]+)\"", response, re.DOTALL)
        if m:
            return m.group(1)
        words = response.strip().split()
        return words[-1] if words else "no-op"

    @staticmethod
    def _estimate_logprob(prompt, action, temperature):
        action_len = max(len(action.split()), 1)
        return -0.5 * action_len - 0.1 * math.log(max(temperature, 0.01))


# ---------------------------------------------------------------------------
# VERL-compatible actor proxy
# ---------------------------------------------------------------------------
try:
    import torch
    from verl import DataProto
    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False


if _HAS_VERL:
    class VERLActorProxy(PolicyInterface):
        """Policy proxy that delegates to a VERL actor worker group.

        This wraps VERL's distributed actor (vLLM/sglang) behind the
        PolicyInterface API.  Primarily useful for evaluation loops that
        want a PolicyInterface but should use the VERL-managed model.

        Note: During VERL training, the TrajectoryCollector and
        RayPPOTrainer call the actor directly — this proxy is NOT used
        in the training hot path.
        """

        def __init__(
            self,
            actor_rollout_wg,
            tokenizer,
            temperature: float = 0.7,
            max_tokens: int = 512,
        ):
            self.actor_wg = actor_rollout_wg
            self.tokenizer = tokenizer
            self.temperature = temperature
            self.max_tokens = max_tokens

        def sample(self, observation, context=None, temperature=None):
            """Sample an action by calling VERL actor's generate_sequences."""
            temp = temperature or self.temperature
            prompt_text = observation
            if context:
                if context.get("skill_cards"):
                    prompt_text += f"\nAvailable skills: {context['skill_cards']}"

            input_ids = self.tokenizer.encode(
                prompt_text, return_tensors="pt",
                truncation=True, max_length=4096,
            )
            attention_mask = torch.ones_like(input_ids)

            batch = DataProto.from_single_dict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": torch.arange(input_ids.shape[-1]).unsqueeze(0),
            })
            batch.meta_info["do_sample"] = True
            batch.meta_info["temperature"] = temp
            batch.meta_info["max_new_tokens"] = self.max_tokens

            output = self.actor_wg.generate_sequences(batch)
            response_ids = output.batch["responses"][0]
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True,
            )

            # Extract action from <action> tags
            import re
            match = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)
            action = match.group(1).strip() if match else response_text.strip()

            logprob = 0.0
            if "old_log_probs" in output.batch:
                mask = output.batch.get("response_mask", torch.ones_like(response_ids, dtype=torch.float32))
                logprob = float((output.batch["old_log_probs"][0] * mask).sum())

            return PolicyOutput(
                action=action, logprob=logprob,
                raw_response=response_text,
                metadata={"prompt_len": input_ids.shape[-1]},
            )

        def logprob(self, observation, action, context=None):
            prompt_text = observation
            full_text = prompt_text + action
            input_ids = self.tokenizer.encode(
                full_text, return_tensors="pt",
                truncation=True, max_length=4096,
            )
            attention_mask = torch.ones_like(input_ids)
            batch = DataProto.from_single_dict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": torch.arange(input_ids.shape[-1]).unsqueeze(0),
            })
            output = self.actor_wg.compute_log_prob(batch)
            if "old_log_probs" in output.batch:
                return float(output.batch["old_log_probs"].sum())
            return 0.0

        def batch_logprobs(self, observations, actions, contexts=None):
            return [
                self.logprob(o, a) for o, a in zip(observations, actions)
            ]

        def update(self, loss, grads=None):
            # In VERL, updates are handled by RayPPOTrainer.
            return {"note": "updates handled by VERL RayPPOTrainer"}

        def get_parameters(self):
            return None  # Managed by VERL FSDP

        def load_parameters(self, params):
            pass  # Managed by VERL checkpoint system
