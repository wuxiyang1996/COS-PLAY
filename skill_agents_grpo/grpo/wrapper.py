"""
Generic GRPO call wrapper: intercepts an LLM function, samples G times,
evaluates via a reward function, stores results in a buffer, and returns
the best sample to the calling pipeline.

The EM pipeline runs unchanged — it just gets better LLM outputs as
GRPO training progresses.

Usage::

    from skill_agents_grpo.grpo import GRPOCallWrapper, GRPOBuffer
    from skill_agents_grpo.lora import SkillFunction

    buffer = GRPOBuffer()
    wrapper = GRPOCallWrapper(
        adapter=SkillFunction.CONTRACT,
        reward_fn=contract_reward,
        buffer=buffer,
        group_size=4,
        temperature=0.7,
    )

    # Monkey-patch or inject
    original_fn = llm_summarize_contract
    llm_summarize_contract = wrapper.wrap(original_fn)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from skill_agents_grpo.grpo.buffer import GRPOBuffer, GRPOSample
from skill_agents_grpo.grpo.grpo_outputs import default_grpo_training_completion
from skill_agents_grpo.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


class GRPOCallWrapper:
    """Wrap an LLM call to generate G samples, evaluate, store, return best.

    Parameters
    ----------
    adapter : SkillFunction
        Which LoRA adapter this wrapper targets.
    reward_fn : callable
        ``reward_fn(sample_output, *original_args, **original_kwargs) -> float``
        Evaluates a single sample. Must be CPU-only and deterministic.
    buffer : GRPOBuffer
        Shared buffer that accumulates samples for deferred training.
    group_size : int
        Number of samples per call (G).
    temperature : float
        Sampling temperature injected during rollout.
    prompt_extractor : callable, optional
        ``prompt_extractor(*args, **kwargs) -> str``
        Extracts the prompt string from the original function's arguments.
        Needed for log-prob recomputation during training. If None, the
        prompt is stored as empty (training will skip these samples).
    metadata_extractor : callable, optional
        ``metadata_extractor(*args, **kwargs) -> dict``
        Extracts context metadata (e.g. skill_id) for diagnostics.
    training_completion_fn : callable, optional
        ``training_completion_fn(sample_output) -> str``
        String stored per GRPO sample for FSDP log-prob training.  Defaults to
        :func:`default_grpo_training_completion` (raw LLM body when available).
    """

    def __init__(
        self,
        adapter: SkillFunction,
        reward_fn: Callable[..., float],
        buffer: GRPOBuffer,
        group_size: int = 4,
        temperature: float = 0.7,
        prompt_extractor: Optional[Callable[..., str]] = None,
        metadata_extractor: Optional[Callable[..., Dict[str, Any]]] = None,
        training_completion_fn: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self.adapter = adapter
        self.reward_fn = reward_fn
        self.buffer = buffer
        self.group_size = group_size
        self.temperature = temperature
        self.prompt_extractor = prompt_extractor
        self.metadata_extractor = metadata_extractor
        self.training_completion_fn = (
            training_completion_fn or default_grpo_training_completion
        )
        self._call_count = 0
        self._total_samples = 0

    def wrap(self, original_fn: Callable) -> Callable:
        """Return a wrapped version of *original_fn* that does GRPO sampling.

        The wrapped function has the same signature as the original.
        It generates G samples, evaluates rewards, stores the group in
        the buffer, and returns the best sample to the caller.
        """
        wrapper_self = self

        @wraps(original_fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return wrapper_self._run(original_fn, args, kwargs)

        wrapped._grpo_wrapper = wrapper_self  # type: ignore[attr-defined]
        return wrapped

    _PARALLEL_SAMPLES = os.environ.get(
        "GRPO_PARALLEL_SAMPLES", "1"
    ).strip().lower() not in ("0", "false", "no")

    def _generate_one(
        self,
        original_fn: Callable,
        args: tuple,
        kwargs_sample: dict,
        idx: int,
    ) -> Tuple[int, Any, str, bool]:
        """Generate one GRPO sample. Returns (idx, output, completion, ok)."""
        try:
            out = original_fn(*args, **kwargs_sample)
            try:
                tc = self.training_completion_fn(out)
            except Exception:
                tc = str(out) if out is not None else ""
            return (idx, out, tc if isinstance(tc, str) else str(tc), True)
        except Exception as exc:
            if idx == 0:
                logger.warning(
                    "GRPO[%s] sample %d failed: %s", self.adapter.value, idx, exc,
                )
            return (idx, None, "", False)

    def _run(
        self,
        original_fn: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Core sampling logic — generates G samples in parallel when possible."""
        self._call_count += 1

        kwargs_sample = {**kwargs, "temperature": self.temperature}

        samples: List[Any] = [None] * self.group_size
        completions: List[str] = [""] * self.group_size
        n_failures = 0

        if self._PARALLEL_SAMPLES and self.group_size > 1:
            with ThreadPoolExecutor(max_workers=self.group_size) as pool:
                futures = [
                    pool.submit(
                        self._generate_one, original_fn, args, kwargs_sample, i,
                    )
                    for i in range(self.group_size)
                ]
                for fut in as_completed(futures):
                    idx, out, tc, ok = fut.result()
                    if ok:
                        samples[idx] = out
                        completions[idx] = tc
                    else:
                        n_failures += 1
        else:
            for i in range(self.group_size):
                idx, out, tc, ok = self._generate_one(
                    original_fn, args, kwargs_sample, i,
                )
                if ok:
                    samples[idx] = out
                    completions[idx] = tc
                else:
                    n_failures += 1

        if n_failures == self.group_size:
            logger.warning(
                "GRPO[%s] all %d samples failed — falling back to unwrapped call",
                self.adapter.value, self.group_size,
            )
            try:
                return original_fn(*args, **kwargs)
            except Exception as exc:
                logger.error(
                    "GRPO[%s] fallback also failed: %s", self.adapter.value, exc,
                )
                return None

        rewards: List[float] = []
        for sample in samples:
            try:
                r = self.reward_fn(sample, *args, **kwargs)
                rewards.append(float(r))
            except Exception:
                rewards.append(0.0)

        prompt = ""
        if self.prompt_extractor is not None:
            try:
                prompt = self.prompt_extractor(*args, **kwargs)
            except Exception:
                pass

        metadata: Dict[str, Any] = {}
        if self.metadata_extractor is not None:
            try:
                metadata = self.metadata_extractor(*args, **kwargs)
            except Exception:
                pass

        grpo_sample = GRPOSample(
            adapter=self.adapter,
            prompt=prompt,
            completions=completions,
            rewards=rewards,
            metadata=metadata,
        )
        self.buffer.add(grpo_sample)
        self._total_samples += self.group_size

        non_empty = sum(1 for c in completions if c)
        n_unique = len({c for c in completions if c})
        r_mean = sum(rewards) / max(len(rewards), 1)
        r_var = sum((r - r_mean) ** 2 for r in rewards) / max(len(rewards), 1)
        all_equal = r_var < 1e-12
        logger.info(
            "GRPO[%s] call #%d: %d/%d non-empty, %d unique, rewards=%s, "
            "r_mean=%.4f, r_var=%.6f, tied=%s, prompt_len=%d",
            self.adapter.value, self._call_count, non_empty, len(completions),
            n_unique, [f"{r:.3f}" for r in rewards],
            r_mean, r_var, all_equal, len(prompt),
        )
        if non_empty == 0 and self.adapter.value in ("contract", "curator"):
            logger.warning(
                "GRPO[%s] all samples empty → reward 0.0 (often vLLM connection "
                "errors or parse failures). Retries: set SKILLBANK_LLM_RETRIES / "
                "SKILLBANK_LLM_RETRY_DELAY_S; vLLM: VLLM_OPENAI_MAX_RETRIES.",
                self.adapter.value,
            )

        best_idx = grpo_sample.best_index
        return samples[best_idx]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter.value,
            "calls": self._call_count,
            "total_samples": self._total_samples,
            "buffer_size": self.buffer.size(self.adapter),
        }
