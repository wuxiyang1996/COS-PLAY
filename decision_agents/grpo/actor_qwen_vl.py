"""Qwen3-VL-8B-Instruct Actor — online inference + GRPO + LoRA.

This is the "deployed" flavour of the Actor Agent: a subclass of
:class:`decision_agents.actor_agent.ActorAgent` that swaps the LLM
backend to ``Qwen/Qwen3-VL-8B-Instruct`` served by vLLM (multi-LoRA,
hot-swappable adapters) and emits per-step
:class:`trainer.common.metrics.RolloutStep` records that the GRPO
trainer in :mod:`trainer.coevolution.grpo_training` consumes directly.

Two contracts matter here:

1. **vLLM backend** — :class:`trainer.coevolution.vllm_client.\
   AsyncVLLMClient` already runs the multi-LoRA server used by the
   co-evolution loop.  The actor reuses it (`generate_chat`) so:

   * one vLLM instance powers both the GRPO trainer's rollouts and any
     external runner instantiating this actor;
   * the ``adapter`` argument routes between the LoRA adapters
     (`skill_selection`, `action_taking`, …) the SFT cold-start
     trainer wrote to ``runs/sft_coldstart/decision/<adapter>``.

2. **Sync veneer over async** — :class:`ActorAgent.step` is sync (it's
   called from gym-style runners and from offline tests).
   :meth:`QwenVLActor._call_llm` therefore wraps the async
   ``AsyncVLLMClient.generate_chat`` in :func:`asyncio.run` /
   ``loop.run_until_complete``.  Power users that drive the actor from
   an async context can call :meth:`step_async` to skip the wrap.

Companion sub-package: :mod:`decision_agents.SFT` (the GPT-4o-driven
collector that produces the cold-start data this actor is fine-tuned
on, before GRPO kicks in).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from decision_agents.actor_agent import (
    ActorAgent,
    ActorDecision,
)
from decision_agents.core.harness import Harness
from decision_agents.core.multimodal import (
    VisualInput,
    build_qwen_vl_messages,
)
from decision_agents.grpo.rollout_logger import GRPORolloutLogger
from decision_agents.reward_func import RewardConfig, RewardResult
from decision_agents.schema_parser import StateSchema
from decision_agents.skill_interface import SkillProvider

try:
    from trainer.coevolution.vllm_client import AsyncVLLMClient
except ImportError:  # pragma: no cover — light tests
    AsyncVLLMClient = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
"""Per ``vlm_wrapper/README.md`` the deployed student is the 8B dense
Qwen3-VL Instruct edition.  The escalation rung (235B-A22B-Thinking)
lives in ``vlm_wrapper`` itself, not here — this actor stays pinned to
the size that GRPO can afford to update."""

DEFAULT_QWEN_VL_SYSTEM_PROMPT = (
    "You are an Actor Agent for a multimodal task. You will be shown a "
    "screenshot together with a structured state summary and a list of "
    "valid environment actions. Choose exactly ONE valid action.\n"
    "\n"
    "Output STRICTLY in this format (no extra prose):\n"
    "SUBGOAL: [TAG] <your immediate objective in <=15 words>\n"
    "REASONING: <1-2 sentences citing what you saw in the screenshot>\n"
    "ACTION: <one valid action, copied verbatim, or its 1-based number>"
)

#: LoRA adapter the actor selects when picking an environment action.
#: Matches the cold-start adapter name in ``trainer.SFT.config.\
#: DECISION_ADAPTERS`` and the runtime adapter slot in
#: ``trainer.coevolution.vllm_client.ADAPTER_MAP``.
ACTION_ADAPTER = "action_taking"


# ──────────────────────────────────────────────────────────────────────
# QwenVLActor
# ──────────────────────────────────────────────────────────────────────


class QwenVLActor(ActorAgent):
    """Online-policy actor backed by Qwen3-VL-8B + vLLM multi-LoRA.

    Parameters (in addition to :class:`ActorAgent`)
    -----------------------------------------------
    vllm_client
        Pre-built :class:`AsyncVLLMClient`.  Required for live use; can
        be ``None`` for offline tests that monkeypatch ``_call_llm`` /
        ``_call_llm_async``.
    rollout_logger
        Optional :class:`GRPORolloutLogger` that the actor pokes after
        each :meth:`observe_result`.  When set, every call to
        :meth:`step` + :meth:`observe_result` lands as one
        :class:`~trainer.common.metrics.RolloutStep` on the active
        episode without the runner having to plumb anything.
    adapter
        LoRA adapter name to invoke for the action-taking call.
        Defaults to :data:`ACTION_ADAPTER` (``"action_taking"``); set
        to ``None`` to use the bare base model (useful for ablations).
    system_prompt
        System message prepended to every chat call.  Defaults to
        :data:`DEFAULT_QWEN_VL_SYSTEM_PROMPT`.
    """

    def __init__(
        self,
        *,
        vllm_client: Optional[Any] = None,
        rollout_logger: Optional[GRPORolloutLogger] = None,
        adapter: Optional[str] = ACTION_ADAPTER,
        system_prompt: str = DEFAULT_QWEN_VL_SYSTEM_PROMPT,
        skill_provider: Optional[SkillProvider] = None,
        harness: Optional[Harness] = None,
        reward_config: Optional[RewardConfig] = None,
        model: str = DEFAULT_QWEN_VL_MODEL,
        stall_window: int = 4,
        # Deprecated kwargs forwarded to ActorAgent (which warns + ignores).
        hop_policy: Any = None,
        max_hops_per_step: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            skill_provider=skill_provider,
            harness=harness,
            reward_config=reward_config,
            stall_window=stall_window,
            hop_policy=hop_policy,
            max_hops_per_step=max_hops_per_step,
        )
        self.vllm_client = vllm_client
        self.rollout_logger = rollout_logger
        self.adapter = adapter
        self.system_prompt = system_prompt
        self._step_images: List[VisualInput] = []
        # Last-step bookkeeping so observe_result can correctly attribute
        # the RolloutStep to the right ActorDecision instance.
        self._pending_decision: Optional[ActorDecision] = None

    # ── per-step image plumbing ──────────────────────────────────────

    def set_step_images(self, images: Optional[Sequence[VisualInput]]) -> None:
        self._step_images = list(images) if images else []

    # ── sync step ────────────────────────────────────────────────────

    def step(  # type: ignore[override]
        self,
        *,
        observation: str,
        schema_text: Optional[str] = None,
        schema: Optional[StateSchema] = None,
        task: str = "",
        valid_actions: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
        images: Optional[Sequence[VisualInput]] = None,
    ) -> ActorDecision:
        """Run one outer step (sync).

        Same contract as :meth:`ActorAgent.step` plus an optional
        *images* keyword forwarded to :meth:`_call_llm`.
        """
        if images is not None:
            self.set_step_images(images)
        decision = super().step(
            observation=observation,
            schema_text=schema_text,
            schema=schema,
            task=task,
            valid_actions=valid_actions,
            info=info,
        )
        self._pending_decision = decision
        # Drop staged images once the step closes; runners that want
        # sticky images call set_step_images() between steps explicitly.
        self._step_images = []
        return decision

    async def step_async(
        self,
        *,
        observation: str,
        schema_text: Optional[str] = None,
        schema: Optional[StateSchema] = None,
        task: str = "",
        valid_actions: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
        images: Optional[Sequence[VisualInput]] = None,
    ) -> ActorDecision:
        """Async variant for callers already on an event loop.

        Internally we still run :meth:`ActorAgent.step` synchronously
        — only the LLM call inside ``_pick_action`` is awaitable, and
        :meth:`_call_llm` already bridges that.  Exposed primarily for
        symmetry with :class:`AsyncVLLMClient`'s API surface; runners
        that drive multiple actors concurrently can ``await`` this
        method to avoid blocking the loop while the parent class
        processes the schema.
        """
        # The parent class is sync; we wrap in run_in_executor so we
        # don't block the event loop on schema parsing / scratchpad
        # bookkeeping.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.step(
                observation=observation,
                schema_text=schema_text,
                schema=schema,
                task=task,
                valid_actions=valid_actions,
                info=info,
                images=images,
            ),
        )

    # ── observe_result with rollout logging ──────────────────────────

    def observe_result(  # type: ignore[override]
        self,
        decision: ActorDecision,
        *,
        reward: float,
        next_observation: str = "",
        next_schema_text: Optional[str] = None,
        next_schema: Optional[StateSchema] = None,
        done: bool = False,
    ) -> RewardResult:
        """Forward to the parent and log a :class:`RolloutStep`.

        The :class:`GRPORolloutLogger` is poked here (not in
        :meth:`step`) because :class:`RolloutStep` needs the realised
        reward decomposition, which only exists *after* the env stepped.
        """
        rr = super().observe_result(
            decision,
            reward=reward,
            next_observation=next_observation,
            next_schema_text=next_schema_text,
            next_schema=next_schema,
            done=done,
        )
        if self.rollout_logger is not None:
            try:
                self.rollout_logger.log_step(
                    decision=decision,
                    reward_result=rr,
                    done=done,
                )
            except Exception as exc:
                _LOGGER.warning("rollout logger failed for step: %s", exc)
        self._pending_decision = None
        return rr

    # ── LLM seam override (multimodal vLLM call) ─────────────────────

    def _call_llm(
        self,
        prompt: str,
        *,
        images: Optional[List[Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> str:
        """Bridge the sync ActorAgent pipeline to the async vLLM client.

        Runs the underlying coroutine via :func:`asyncio.run` when
        called from a sync context.  When called from inside a running
        event loop (e.g. a runner that already awaited
        :meth:`step_async`), defers to a fresh loop in a worker thread
        — this is rare in practice but cheap to handle.
        """
        attached: List[VisualInput] = list(images) if images else list(self._step_images)

        coro = self._call_llm_async(
            prompt,
            images=attached,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is None:
            return asyncio.run(coro)

        # Already inside an event loop — run on a side thread so we
        # don't deadlock the caller's loop.
        import concurrent.futures
        import threading

        result: Dict[str, str] = {}
        exc_box: Dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["v"] = asyncio.run(coro)
            except BaseException as exc:  # pragma: no cover — defensive
                exc_box["e"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "e" in exc_box:
            _LOGGER.warning("vLLM bridge thread raised: %s", exc_box["e"])
            return ""
        return result.get("v", "")

    async def _call_llm_async(
        self,
        prompt: str,
        *,
        images: Optional[Sequence[VisualInput]] = None,
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> str:
        """Async multimodal Qwen3-VL chat completion via vLLM.

        Falls back to an empty reply (never raises) so the parent's
        deterministic ``_pick_action`` fallback can still drive the
        episode forward when vLLM hiccups — the GRPO trainer prefers a
        noisy but completed episode over a crashed one.
        """
        if self.vllm_client is None:
            _LOGGER.debug("QwenVLActor has no vllm_client; emitting empty reply")
            return ""
        messages = build_qwen_vl_messages(
            prompt=prompt,
            images=images,
            system=self.system_prompt,
        )
        try:
            res = await self.vllm_client.generate_chat(
                messages,
                adapter=self.adapter,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            _LOGGER.warning("AsyncVLLMClient.generate_chat failed: %s", exc)
            return ""
        return getattr(res, "text", "") or ""


__all__ = [
    "QwenVLActor",
    "DEFAULT_QWEN_VL_MODEL",
    "DEFAULT_QWEN_VL_SYSTEM_PROMPT",
    "ACTION_ADAPTER",
]
