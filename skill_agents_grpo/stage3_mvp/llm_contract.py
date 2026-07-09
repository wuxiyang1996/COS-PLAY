"""
LLM-assisted contract learning via the CONTRACT LoRA adapter.

Provides ``llm_summarize_contract`` which uses the trained CONTRACT adapter
to generate effect summaries from segment observations, enriching the
purely algorithmic frequency-based contract learning in Stage 3.

When GRPO is active, calls are intercepted by ``GRPOCallWrapper``:
G samples are generated, evaluated via ``contract_reward()``, and the
best is returned. The EM pipeline is unaware of the wrapping.

When the multi-LoRA model is not configured, falls back gracefully so
the algorithmic path continues to work.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Dynamic reward context (thread-safe) ─────────────────────────────
# The pipeline sets this *before* each skill's LLM call so the GRPO
# reward function can read it at scoring time rather than relying on
# static partial-binding at enable time.

_reward_ctx = threading.local()


def set_contract_reward_context(
    *,
    consensus_add: Optional[Set[str]] = None,
    consensus_del: Optional[Set[str]] = None,
    holdout_instances: Optional[list] = None,
    verify_config: Optional[Any] = None,
    instance_rewards: Optional[List[float]] = None,
) -> None:
    """Update the per-thread contract reward context.

    Called from the Stage 3 pipeline right before ``llm_summarize_contract``
    so that the GRPO reward function uses the real verification path.

    Parameters
    ----------
    instance_rewards : list[float], optional
        Per-holdout-instance cumulative game reward, parallel to
        *holdout_instances*.  Used to weight verification so effects
        holding in high-reward instances matter more.
    """
    _reward_ctx.data = {
        "consensus_add": consensus_add,
        "consensus_del": consensus_del,
        "holdout_instances": holdout_instances,
        "verify_config": verify_config,
        "instance_rewards": instance_rewards,
    }


def _get_contract_reward_context() -> dict:
    return getattr(_reward_ctx, "data", {})

_CONTRACT_PROMPT_TEMPLATE = """\
You are analyzing skill effects from game trajectory segments.

Skill: {skill_id}
Number of instances: {n_instances}

Representative segment observations:
{segment_observations}

State predicates at segment start: {predicates_start}
State predicates at segment end: {predicates_end}

Summarize the effects of this skill as a JSON object:
{{"eff_add": ["predicates that become true"], "eff_del": ["predicates that become false"], "description": "one-line description"}}"""


def _get_contract_ask_fn() -> Optional[Callable[..., str]]:
    """Return a CONTRACT-routed ask function.

    Resolution order:
      1. CONTRACT LoRA adapter via ``MultiLoraSkillBankLLM``
      2. Local vLLM (``ask_vllm``) — avoids OpenRouter rate limits
      3. ``ask_model`` (routes through OpenRouter)

    The returned callable is wrapped for reasoning-model compatibility
    (Qwen3 ``/no_think``, think-tag stripping).
    """
    from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models

    _hint = "Qwen/Qwen3-8B"
    try:
        from skill_agents_grpo.lora import MultiLoraSkillBankLLM, SkillFunction
        llm = MultiLoraSkillBankLLM.get_shared_instance()
        if llm is not None:
            return wrap_ask_for_reasoning_models(
                llm.as_ask_fn(SkillFunction.CONTRACT), model_hint=_hint,
            )
    except Exception:
        pass
    try:
        from API_func import ask_vllm, _probe_vllm
        if _probe_vllm():
            logger.debug("CONTRACT fallback: using local vLLM")
            return wrap_ask_for_reasoning_models(ask_vllm, model_hint=_hint)
    except Exception:
        pass
    from API_func import ask_model
    return wrap_ask_for_reasoning_models(ask_model, model_hint=_hint)


def _build_contract_prompt(
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
) -> str:
    """Build the CONTRACT prompt from arguments (exposed for GRPO prompt extraction)."""
    return _CONTRACT_PROMPT_TEMPLATE.format(
        skill_id=skill_id,
        n_instances=n_instances,
        segment_observations="; ".join(segment_observations[:5]),
        predicates_start=json.dumps(sorted(predicates_start)),
        predicates_end=json.dumps(sorted(predicates_end)),
    )


def llm_summarize_contract(
    skill_id: str,
    segment_observations: List[str],
    predicates_start: Set[str],
    predicates_end: Set[str],
    n_instances: int = 0,
    *,
    temperature: float = 0.1,
    model: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Use the CONTRACT adapter (or ask_model fallback) to generate an effect summary.

    Returns a dict ``{"eff_add": [...], "eff_del": [...], "description": "..."}``
    or None if no LLM backend is available or the call fails.

    Resolution order for the LLM backend:
      1. CONTRACT LoRA adapter (via ``MultiLoraSkillBankLLM``)
      2. ``ask_model`` fallback (routes to GPT / Qwen via OpenRouter)

    The fallback ensures cold-start I/O is always collected even when no
    LoRA adapter is configured, providing training data for the CONTRACT
    adapter's initial fine-tuning.

    When GRPO wrapping is active (via ``enable_contract_grpo``), the
    wrapper intercepts this call, generates G samples at higher temperature,
    evaluates each with ``contract_reward()``, stores data for training,
    and returns the best.
    """
    import time as _time
    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

    ask_fn = _get_contract_ask_fn()
    if ask_fn is None:
        return None

    resolved_model = model or ""

    prompt = _build_contract_prompt(
        skill_id, segment_observations, predicates_start, predicates_end, n_instances,
    )

    try:
        from skill_agents_grpo._llm_retry import sync_ask_with_retry

        t0 = _time.time()
        call_kwargs: Dict[str, Any] = {"temperature": temperature}
        if resolved_model:
            call_kwargs["model"] = resolved_model
        raw = sync_ask_with_retry(
            ask_fn,
            prompt,
            log_label=f"CONTRACT:{skill_id}",
            **call_kwargs,
        )
        elapsed = _time.time() - t0
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = None
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])

        record_io(ColdStartRecord(
            module="stage3_contract",
            function="contract_summary",
            prompt=prompt,
            response=raw or "",
            parsed=parsed,
            model=resolved_model,
            temperature=temperature,
            elapsed_s=round(elapsed, 3),
            skill_id=skill_id,
            extra={"n_instances": n_instances},
            error=None if parsed else "parse_failed",
        ))

        if parsed is not None:
            from skill_agents_grpo.grpo.grpo_outputs import SkillBankLLMOutput

            return SkillBankLLMOutput(dict(parsed), raw_completion=raw or "")
        return None
    except Exception as exc:
        logger.warning("CONTRACT call failed for %s: %s", skill_id, exc)

    return None


# ── GRPO integration ──────────────────────────────────────────────────

_grpo_original_fn: Optional[Callable] = None


def enable_contract_grpo(
    buffer: Any,
    group_size: int = 4,
    temperature: float = 0.7,
) -> None:
    """Activate GRPO wrapping on ``llm_summarize_contract``.

    After calling this, every ``llm_summarize_contract()`` invocation
    generates G samples, evaluates with ``contract_reward()``, stores
    data in *buffer*, and returns the best.

    Reward context (holdout, consensus, verify_config) is read dynamically
    from the thread-local ``_reward_ctx`` at scoring time.  Call
    :func:`set_contract_reward_context` before each skill's LLM call.
    """
    import skill_agents_grpo.stage3_mvp.llm_contract as _mod
    from skill_agents_grpo.grpo.rewards import contract_reward
    from skill_agents_grpo.grpo.wrapper import GRPOCallWrapper
    from skill_agents_grpo.lora.skill_function import SkillFunction

    global _grpo_original_fn

    if _grpo_original_fn is not None:
        logger.warning("Contract GRPO already enabled — skipping re-enable")
        return

    _grpo_original_fn = _mod.llm_summarize_contract

    def _dynamic_contract_reward(llm_output, *args, **kwargs):
        ctx = _get_contract_reward_context()
        passthrough = {
            k: v for k, v in kwargs.items()
            if k not in ("consensus_add", "consensus_del",
                         "holdout_instances", "verify_config",
                         "instance_rewards")
        }
        return contract_reward(
            llm_output, *args,
            consensus_add=ctx.get("consensus_add"),
            consensus_del=ctx.get("consensus_del"),
            holdout_instances=ctx.get("holdout_instances"),
            verify_config=ctx.get("verify_config"),
            instance_rewards=ctx.get("instance_rewards"),
            **passthrough,
        )

    def _prompt_extractor(
        skill_id: str,
        segment_observations: List[str],
        predicates_start: Set[str],
        predicates_end: Set[str],
        n_instances: int = 0,
        **kw: Any,
    ) -> str:
        return _build_contract_prompt(
            skill_id, segment_observations, predicates_start, predicates_end, n_instances,
        )

    def _metadata_extractor(skill_id: str, *a: Any, **kw: Any) -> Dict[str, Any]:
        return {"skill_id": skill_id}

    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.CONTRACT,
        reward_fn=_dynamic_contract_reward,
        buffer=buffer,
        group_size=group_size,
        temperature=temperature,
        prompt_extractor=_prompt_extractor,
        metadata_extractor=_metadata_extractor,
    )

    _mod.llm_summarize_contract = wrapper.wrap(_grpo_original_fn)
    logger.info("Contract GRPO enabled: G=%d, temp=%.2f", group_size, temperature)


def disable_contract_grpo() -> None:
    """Deactivate GRPO wrapping, restore original function."""
    import skill_agents_grpo.stage3_mvp.llm_contract as _mod

    global _grpo_original_fn
    if _grpo_original_fn is not None:
        _mod.llm_summarize_contract = _grpo_original_fn
        _grpo_original_fn = None
        logger.info("Contract GRPO disabled")
