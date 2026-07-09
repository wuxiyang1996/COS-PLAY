"""
GRPO replay buffer: stores rollout data for deferred training.

During Phase 1 (rollout), ``GRPOCallWrapper`` generates G samples per
LLM call, evaluates rewards, and pushes a ``GRPOSample`` here.  During
Phase 2 (training), ``GRPOLoRATrainer`` reads the buffer, computes
log-probs with gradients, and performs policy-gradient updates.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from skill_agents_grpo.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


@dataclass
class GRPOSample:
    """One GRPO group: G completions for the same prompt.

    Attributes
    ----------
    adapter : SkillFunction
        Which LoRA adapter produced these samples.
    prompt : str
        The input text (for log-prob recomputation during training).
    completions : list[str]
        G raw completion strings.
    rewards : list[float]
        Per-completion reward values (same length as completions).
    metadata : dict
        Stage-specific context (e.g. skill_id, segment boundaries).
    """

    adapter: SkillFunction
    prompt: str
    completions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def group_size(self) -> int:
        return len(self.completions)

    @property
    def best_index(self) -> int:
        if not self.rewards:
            return 0
        return max(range(len(self.rewards)), key=lambda i: self.rewards[i])

    @property
    def best_completion(self) -> Optional[str]:
        if not self.completions:
            return None
        return self.completions[self.best_index]


class GRPOBuffer:
    """Thread-safe accumulator for GRPO rollout samples.

    Partitioned by adapter so ``GRPOLoRATrainer`` can train each
    adapter independently.
    """

    def __init__(self, max_size_per_adapter: int = 256) -> None:
        self._samples: Dict[str, List[GRPOSample]] = {}
        self._max_size = max_size_per_adapter
        self._lock = threading.Lock()

    def add(self, sample: GRPOSample) -> None:
        adapter = sample.adapter
        if isinstance(adapter, str) and not hasattr(adapter, 'value'):
            adapter = SkillFunction(adapter)
            sample.adapter = adapter
        key = adapter.value
        with self._lock:
            if key not in self._samples:
                self._samples[key] = []
            buf = self._samples[key]
            buf.append(sample)
            if len(buf) > self._max_size:
                buf.pop(0)

    def samples_for(self, adapter: SkillFunction) -> List[GRPOSample]:
        with self._lock:
            return list(self._samples.get(adapter.value, []))

    def size(self, adapter: Optional[SkillFunction] = None) -> int:
        with self._lock:
            if adapter is not None:
                return len(self._samples.get(adapter.value, []))
            return sum(len(v) for v in self._samples.values())

    def clear(self, adapter: Optional[SkillFunction] = None) -> None:
        with self._lock:
            if adapter is not None:
                self._samples.pop(adapter.value, None)
            else:
                self._samples.clear()

    def adapters_with_data(self) -> List[SkillFunction]:
        with self._lock:
            result = []
            for key, samples in self._samples.items():
                if samples:
                    try:
                        result.append(SkillFunction(key))
                    except ValueError:
                        pass
            return result

    def __repr__(self) -> str:
        with self._lock:
            parts = [f"{k}: {len(v)}" for k, v in self._samples.items() if v]
        return f"GRPOBuffer({', '.join(parts) or 'empty'})"
