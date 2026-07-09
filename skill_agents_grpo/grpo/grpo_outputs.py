"""Structured LLM outputs for skill-bank GRPO.

Parsed dict/list bodies must stay compatible with the rest of the pipeline, but
GRPO training needs the **actual** model-generated text for log-prob alignment.
We attach that as a normal Python attribute (not a JSON key) on a ``dict``
subclass.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SkillBankLLMOutput(dict):
    """``dict`` of parsed LLM JSON plus ``_grpo_raw_completion`` for GRPO/FSDP.

    The raw string is **not** inserted as a dict key, so ``json.dumps(dict(x))``
    and callers that only iterate mapping keys behave like a plain parse result.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        *,
        raw_completion: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(data or {}, **kwargs)
        self._grpo_raw_completion: str = raw_completion or ""


def default_grpo_training_completion(sample: Any) -> str:
    """Text to store as GRPO ``completions[i]`` for policy-gradient training.

    Priority:
      1. ``sample._grpo_raw_completion`` (contract / curator single-turn JSON).
      2. ``sample.raw_rollouts`` joined (segment: one ranking JSON per segment).
      3. Fallback ``str(sample)`` (legacy / debugging).
    """
    if sample is None:
        return ""
    raw = getattr(sample, "_grpo_raw_completion", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    rollouts = getattr(sample, "raw_rollouts", None)
    if rollouts:
        return "\n---\n".join(str(x) for x in rollouts)
    return str(sample) if sample is not None else ""
