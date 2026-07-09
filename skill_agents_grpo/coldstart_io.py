"""
Centralized cold-start I/O recording for Qwen+GRPO-connected functions.

Every LLM-calling function that will be replaced or augmented by Qwen+GRPO
records its prompt/response through this module.  The collected records
serve as:
  1. Supervised fine-tuning data for Qwen3-8B cold-start.
  2. Reference outputs for GRPO reward comparison.

Usage in any LLM-calling module::

    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

    t0 = time.time()
    response = ask_fn(prompt, ...)
    record_io(ColdStartRecord(
        module="boundary_proposal",
        function="predicate_extraction",
        prompt=prompt,
        response=response or "",
        parsed=parsed_result,
        model=model_name,
        elapsed_s=round(time.time() - t0, 3),
    ))

The main pipeline flushes records per-episode and at the end via
``flush()`` → writes to ``coldstart_io.jsonl``.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColdStartRecord:
    """One LLM call with full prompt/response for cold-start replay."""

    module: str
    function: str
    prompt: str = ""
    response: str = ""
    parsed: Optional[Dict[str, Any]] = None
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    elapsed_s: float = 0.0
    # Context (all optional, set what's available)
    skill_id: Optional[str] = None
    skill_names: List[str] = field(default_factory=list)
    segment_start: Optional[int] = None
    segment_end: Optional[int] = None
    n_steps: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d = {
            k: v
            for k, v in d.items()
            if v is not None and v != "" and v != [] and v != 0 and v != 0.0 and v != {}
        }
        d["module"] = self.module
        d["function"] = self.function
        return d


# ── Global thread-safe buffer ────────────────────────────────────────

_records: List[ColdStartRecord] = []
_lock = threading.Lock()


def record_io(rec: ColdStartRecord) -> None:
    """Thread-safe append to the module-level recording buffer."""
    with _lock:
        _records.append(rec)


def flush() -> List[dict]:
    """Return all accumulated records as dicts and clear the buffer."""
    with _lock:
        out = [r.to_dict() for r in _records]
        _records.clear()
        return out


def get_records() -> List[dict]:
    """Return all accumulated records as dicts (non-destructive)."""
    with _lock:
        return [r.to_dict() for r in _records]


def reset() -> None:
    """Clear all accumulated records without returning them."""
    with _lock:
        _records.clear()


def count() -> int:
    """Return the number of buffered records."""
    with _lock:
        return len(_records)
