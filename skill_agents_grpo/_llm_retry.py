"""Retry helper for synchronous skill-bank LLM calls.

vLLM / HTTP occasionally returns empty bodies or ``Error calling vLLM...``
text instead of JSON.  That makes ``llm_summarize_contract`` /
``filter_candidates`` return ``None``; GRPO then logs ``0/4 non-empty`` and
reward 0.0 for every sample in the group.

Environment variables
---------------------
SKILLBANK_LLM_RETRIES
    Max attempts per call (default ``5``).
SKILLBANK_LLM_RETRY_DELAY_S
    Base backoff in seconds; delay doubles each retry: base, 2×base, … (default ``1.0``).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _is_llm_error_payload(text: str) -> bool:
    """True if *text* is empty or clearly an HTTP/API failure message."""
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    if s.startswith("Error"):
        return True
    low = s.lower()

    # Only check error patterns in short responses (valid LLM JSON is typically
    # longer than 60 chars).  Game data often embeds numbers like "score=500"
    # that would false-positive against bare HTTP status codes.
    if len(s) < 60:
        short_needles = ("503", "502", "500")
        if any(n in low for n in short_needles):
            return True

    needles = (
        "connection error",
        "connection refused",
        "error calling vllm",
        "error calling openrouter",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "http 503",
        "http 502",
        "http 500",
        "status 503",
        "status 502",
        "status 500",
        "econnreset",
        "eof occurred",
        "broken pipe",
        "remote end closed",
    )
    return any(n in low for n in needles)


def sync_ask_with_retry(
    ask_fn: Callable[..., str],
    prompt: str,
    *,
    log_label: str = "skillbank_llm",
    max_attempts: int | None = None,
    base_delay_s: float | None = None,
    **call_kwargs: Any,
) -> str:
    """Call *ask_fn* with exponential backoff until success or attempts exhausted.

    Success means a non-empty string that does not look like an API error payload.
    """
    attempts = max_attempts
    if attempts is None:
        attempts = int(os.environ.get("SKILLBANK_LLM_RETRIES", "5"))
    delay0 = base_delay_s
    if delay0 is None:
        delay0 = float(os.environ.get("SKILLBANK_LLM_RETRY_DELAY_S", "1.0"))

    last_raw = ""
    for attempt in range(attempts):
        try:
            raw = ask_fn(prompt, **call_kwargs)
            last_raw = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        except Exception as exc:
            last_raw = ""
            if attempt + 1 >= attempts:
                logger.warning("%s: exhausted retries after exception: %s", log_label, exc)
                return ""
            delay = delay0 * (2**attempt)
            logger.warning(
                "%s: attempt %d/%d raised %s — retry in %.1fs",
                log_label,
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            continue

        if _is_llm_error_payload(last_raw):
            if attempt + 1 >= attempts:
                preview = (last_raw[:200] + "…") if len(last_raw) > 200 else last_raw
                logger.warning(
                    "%s: exhausted %d attempts; last payload: %s",
                    log_label,
                    attempts,
                    preview or "(empty)",
                )
                return last_raw
            delay = delay0 * (2**attempt)
            preview = (last_raw[:120] + "…") if len(last_raw) > 120 else last_raw
            logger.warning(
                "%s: attempt %d/%d unusable — %s — retry in %.1fs",
                log_label,
                attempt + 1,
                attempts,
                preview or "(empty)",
                delay,
            )
            time.sleep(delay)
            continue

        return last_raw

    return last_raw
