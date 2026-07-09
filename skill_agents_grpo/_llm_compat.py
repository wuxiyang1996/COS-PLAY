"""Compatibility helpers for reasoning-model LLM calls (Qwen3, QwQ, etc.).

Reasoning models like Qwen3-8B default to an internal "thinking" mode that
emits ``<think>…</think>`` blocks *before* the actual answer.  These blocks
consume the ``max_tokens`` budget and often leave nothing for the structured
output we need (JSON, rankings, protocols).

Two mitigations, applied transparently via ``wrap_ask_for_reasoning_models``:

1. Append ``/no_think`` to every prompt — disables the thinking mode so the
   full token budget goes to actual output.
2. Strip any residual ``<think>`` blocks from the response.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

_REASONING_MODEL_PATTERNS = ("qwen3", "qwen-3", "qwq")
_THINK_COMPLETE_RE = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks from reasoning-model output.

    Handles both complete and truncated (opened but never closed) blocks
    that occur when ``max_tokens`` cuts off mid-thought.
    """
    if not text or "<think>" not in text:
        return text
    result = _THINK_COMPLETE_RE.sub("", text)
    result = _THINK_OPEN_RE.sub("", result)
    return result.strip()


def is_reasoning_model(model_name: Optional[str]) -> bool:
    """Return True if *model_name* looks like a reasoning model."""
    if not model_name:
        return False
    lower = model_name.lower()
    return any(p in lower for p in _REASONING_MODEL_PATTERNS)


def wrap_ask_for_reasoning_models(
    ask_fn: Callable,
    model_hint: Optional[str] = None,
) -> Callable:
    """Wrap an ``ask_model`` callable to handle reasoning-model quirks.

    For models like Qwen3:
      - Appends ``/no_think`` to prompts so the full token budget
        goes to structured output.
      - Strips ``<think>`` tags from responses.

    If the effective model (from *model_hint* or ``kwargs["model"]``)
    doesn't look like a reasoning model, the prompt is passed through
    unchanged but think-tag stripping is still applied defensively.

    Parameters
    ----------
    ask_fn : callable
        Original ``ask_model(prompt, **kwargs) -> str``.
    model_hint : str, optional
        Default model name if the caller doesn't pass ``model=`` in kwargs.
    """
    def wrapped(prompt: str, **kwargs) -> str:
        effective_model = kwargs.get("model") or model_hint
        if is_reasoning_model(effective_model):
            prompt = prompt.rstrip() + "\n/no_think"
        result = ask_fn(prompt, **kwargs)
        if result and isinstance(result, str):
            result = strip_think_tags(result)
        return result

    return wrapped
