"""Async vLLM client wrapper for co-evolution inference.

Wraps ``openai.AsyncOpenAI`` to provide adapter-aware completions via the
vLLM multi-LoRA server.  Both decision agent and skill bank agent call the
same vLLM instance; the ``adapter`` parameter selects the active LoRA.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

ADAPTER_MAP = {
    "skill_selection": "skill_selection",
    "action_taking": "action_taking",
    "segment": "segment",
    "contract": "contract",
    "curator": "curator",
    "base": None,
}

# HTTP 400/404 status codes that indicate a missing LoRA adapter
_ADAPTER_NOT_FOUND_CODES = {400, 404}


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    adapter: Optional[str] = None


def _is_adapter_missing(exc: Exception) -> bool:
    """Check if an OpenAI API error indicates the LoRA adapter is not loaded."""
    status = getattr(exc, "status_code", None)
    if status in _ADAPTER_NOT_FOUND_CODES:
        return True
    msg = str(exc).lower()
    return "not found" in msg or "does not exist" in msg or "unknown model" in msg


def _is_connection_error(exc: Exception) -> bool:
    """True if *exc* is a network-level failure (refused, reset, timeout)."""
    from openai import APIConnectionError, APITimeoutError
    if isinstance(exc, (APIConnectionError, APITimeoutError, ConnectionError, OSError)):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("connection", "refused", "reset", "timed out", "eof"))


# How long (seconds) a client stays in the "suspect" pool after a connection failure.
_HEALTH_PENALTY_S = float(os.environ.get("VLLM_HEALTH_PENALTY_S", "30"))


class AsyncVLLMClient:
    """Thin async wrapper over vLLM's OpenAI-compatible API with multi-LoRA.

    Supports multiple vLLM backends with health-aware load balancing
    (one per GPU in the phase-swapped architecture).  When a request
    to one instance fails with a connection error, the request is
    retried on the next healthy instance before giving up.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        base_urls: Optional[List[str]] = None,
        model: str = "Qwen/Qwen3-8B",
        default_temperature: float = 0.3,
        default_max_tokens: int = 512,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        import httpx

        urls = base_urls if base_urls else [base_url]
        n = len(urls)
        per_client_conns = max(32, 256 // n)
        per_client_keepalive = max(16, 128 // n)

        self._clients: List[AsyncOpenAI] = []
        self._client_urls: List[str] = list(urls)
        for url in urls:
            client = AsyncOpenAI(
                base_url=url,
                api_key="EMPTY",
                timeout=timeout,
                max_retries=max_retries,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=per_client_conns,
                        max_keepalive_connections=per_client_keepalive,
                    ),
                    timeout=httpx.Timeout(timeout, connect=30.0),
                ),
            )
            self._clients.append(client)

        self._rr_counter = 0
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        self._io_log_dir: Optional[str] = None
        self._io_step: int = 0
        self._io_lock = asyncio.Lock()

        # Health tracking: maps client index → timestamp of last connection failure.
        # Clients with recent failures are deprioritised by _next_client().
        self._last_fail: Dict[int, float] = {}

    def _mark_failed(self, idx: int) -> None:
        self._last_fail[idx] = time.monotonic()

    def _mark_ok(self, idx: int) -> None:
        self._last_fail.pop(idx, None)

    def _is_suspect(self, idx: int) -> bool:
        ts = self._last_fail.get(idx)
        if ts is None:
            return False
        return (time.monotonic() - ts) < _HEALTH_PENALTY_S

    def _next_client_idx(self) -> int:
        """Pick the next client, preferring instances not recently failed."""
        n = len(self._clients)
        if n == 1:
            return 0
        start = self._rr_counter % n
        self._rr_counter += 1
        if not self._is_suspect(start):
            return start
        for offset in range(1, n):
            candidate = (start + offset) % n
            if not self._is_suspect(candidate):
                return candidate
        return start  # all suspect — use round-robin anyway

    def _next_client(self) -> AsyncOpenAI:
        """Round-robin across vLLM instances, skipping recently-failed ones."""
        return self._clients[self._next_client_idx()]

    def enable_io_logging(self, log_dir: str, step: int = 0) -> None:
        """Enable debug I/O logging: every LLM call is written to disk."""
        self._io_log_dir = log_dir
        self._io_step = step
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Debug I/O logging enabled: %s (step %d)", log_dir, step)

    def set_io_step(self, step: int) -> None:
        self._io_step = step

    async def _log_io(self, record: Dict[str, Any]) -> None:
        if self._io_log_dir is None:
            return
        adapter = record.get("adapter") or "base"
        fname = f"step_{self._io_step:04d}_{adapter}.jsonl"
        path = os.path.join(self._io_log_dir, fname)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        async with self._io_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    async def generate(
        self,
        prompt: str,
        *,
        adapter: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerateResult:
        """Generate a completion via vLLM.

        Parameters
        ----------
        prompt : str
            Full prompt text (system + user rolled into one string).
        adapter : str | None
            LoRA adapter name (e.g. ``"action_taking"``).  ``None`` = base model.
        """
        t0 = time.monotonic()
        temp = temperature if temperature is not None else self.default_temperature
        mtok = max_tokens if max_tokens is not None else self.default_max_tokens

        model_id = self.model
        used_adapter = adapter
        if adapter and adapter in ADAPTER_MAP and ADAPTER_MAP[adapter] is not None:
            model_id = ADAPTER_MAP[adapter]

        n_clients = len(self._clients)
        last_exc: Optional[Exception] = None

        for attempt in range(n_clients):
            idx = self._next_client_idx()
            client = self._clients[idx]
            try:
                resp = await client.completions.create(
                    model=model_id,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=mtok,
                    stop=stop,
                )
                self._mark_ok(idx)
                break
            except Exception as exc:
                last_exc = exc
                if adapter and model_id != self.model and _is_adapter_missing(exc):
                    logger.info(
                        "Adapter '%s' not loaded in vLLM, falling back to base model",
                        adapter,
                    )
                    used_adapter = None
                    try:
                        resp = await client.completions.create(
                            model=self.model,
                            prompt=prompt,
                            temperature=temp,
                            max_tokens=mtok,
                            stop=stop,
                        )
                        self._mark_ok(idx)
                        break
                    except Exception as exc2:
                        last_exc = exc2

                if _is_connection_error(exc) and attempt + 1 < n_clients:
                    self._mark_failed(idx)
                    logger.debug(
                        "vLLM instance %d connection error, trying next (adapter=%s)",
                        idx, adapter,
                    )
                    continue
                else:
                    self._mark_failed(idx)
                    logger.warning("vLLM call failed (adapter=%s): %s", adapter, exc)
                    return GenerateResult(text="", adapter=used_adapter)
        else:
            logger.warning(
                "All %d vLLM instances failed (adapter=%s): %s",
                n_clients, adapter, last_exc,
            )
            return GenerateResult(text="", adapter=used_adapter)

        text = resp.choices[0].text if resp.choices else ""
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        elapsed = (time.monotonic() - t0) * 1000
        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        await self._log_io({
            "ts": time.time(),
            "call_id": self._call_count,
            "mode": "completion",
            "adapter": used_adapter,
            "prompt": prompt,
            "prompt_len_chars": len(prompt),
            "completion": text,
            "completion_len_chars": len(text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "max_tokens_requested": mtok,
            "finish_reason": finish_reason,
            "temperature": temp,
            "latency_ms": round(elapsed, 1),
            "possibly_truncated": finish_reason == "length",
        })

        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            adapter=used_adapter,
        )

    _DEFAULT_EXTRA_BODY: Dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        adapter: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Chat-completion variant that lets vLLM apply the chat template.

        Thinking is disabled by default to avoid wasting tokens on
        ``<think>`` blocks.  Pass ``extra_body={}`` to override.
        """
        t0 = time.monotonic()
        temp = temperature if temperature is not None else self.default_temperature
        mtok = max_tokens if max_tokens is not None else self.default_max_tokens

        model_id = self.model
        used_adapter = adapter
        if adapter and adapter in ADAPTER_MAP and ADAPTER_MAP[adapter] is not None:
            model_id = ADAPTER_MAP[adapter]

        effective_extra = extra_body if extra_body is not None else self._DEFAULT_EXTRA_BODY
        extra_kwargs: Dict[str, Any] = {}
        if stop:
            extra_kwargs["stop"] = stop
        if effective_extra:
            extra_kwargs["extra_body"] = effective_extra

        n_clients = len(self._clients)
        last_exc: Optional[Exception] = None

        for attempt in range(n_clients):
            idx = self._next_client_idx()
            client = self._clients[idx]
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mtok,
                    **extra_kwargs,
                )
                self._mark_ok(idx)
                break
            except Exception as exc:
                last_exc = exc
                if adapter and model_id != self.model and _is_adapter_missing(exc):
                    logger.info(
                        "Adapter '%s' not loaded in vLLM (chat), falling back to base model",
                        adapter,
                    )
                    used_adapter = None
                    try:
                        resp = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temp,
                            max_tokens=mtok,
                            **extra_kwargs,
                        )
                        self._mark_ok(idx)
                        break
                    except Exception as exc2:
                        last_exc = exc2

                if _is_connection_error(exc) and attempt + 1 < n_clients:
                    self._mark_failed(idx)
                    logger.debug(
                        "vLLM instance %d connection error (chat), trying next (adapter=%s)",
                        idx, adapter,
                    )
                    continue
                else:
                    self._mark_failed(idx)
                    logger.warning("vLLM chat call failed (adapter=%s): %s", adapter, exc)
                    return GenerateResult(text="", adapter=used_adapter)
        else:
            logger.warning(
                "All %d vLLM instances failed (chat, adapter=%s): %s",
                n_clients, adapter, last_exc,
            )
            return GenerateResult(text="", adapter=used_adapter)

        text = resp.choices[0].message.content if resp.choices else ""
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        elapsed = (time.monotonic() - t0) * 1000
        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        await self._log_io({
            "ts": time.time(),
            "call_id": self._call_count,
            "mode": "chat",
            "adapter": used_adapter,
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "completion": text,
            "completion_len_chars": len(text) if text else 0,
            "completion_tokens": completion_tokens,
            "max_tokens_requested": mtok,
            "finish_reason": finish_reason,
            "temperature": temp,
            "latency_ms": round(elapsed, 1),
            "possibly_truncated": finish_reason == "length",
        })

        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            adapter=used_adapter,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "call_count": self._call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }

    def reset_stats(self) -> None:
        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    async def health_check(self) -> bool:
        """Check if all vLLM instances are reachable."""
        try:
            for client in self._clients:
                resp = await client.models.list()
                if len(resp.data) == 0:
                    return False
            return True
        except Exception:
            return False
