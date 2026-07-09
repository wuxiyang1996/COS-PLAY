"""`BudgetController` (PLAN-PIPELINE-ORCHESTRATOR §8).

Tracks per-episode and per-inner-MDP-tick resource usage. Hard caps are
exceeded ⇒ raise `BudgetExceeded`; soft caps are observed and emitted to
the audit log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from orchestrator.config import BudgetLimits


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class _Counters:
    outer_steps: int = 0
    inner_steps: int = 0
    skill_invocations: int = 0
    tokens: int = 0
    grounding_escalations: int = 0
    teacher_calls: int = 0
    started_at: float = field(default_factory=time.time)


class BudgetController:
    def __init__(self, limits: Optional[BudgetLimits] = None) -> None:
        self._limits = limits or BudgetLimits()
        self._counters = _Counters()

    # -- accounting --------------------------------------------------------

    def add_outer_step(self) -> None:
        self._counters.outer_steps += 1
        self._enforce()

    def add_inner_step(self) -> None:
        self._counters.inner_steps += 1
        self._enforce()

    def add_skill_invocation(self) -> None:
        self._counters.skill_invocations += 1
        self._enforce()

    def add_tokens(self, n: int) -> None:
        self._counters.tokens += int(n)
        self._enforce()

    def add_grounding_escalation(self) -> None:
        self._counters.grounding_escalations += 1
        self._enforce()

    def add_teacher_call(self) -> None:
        self._counters.teacher_calls += 1
        self._enforce()

    # -- queries -----------------------------------------------------------

    def remaining(self) -> Dict[str, float]:
        return {
            "outer_steps": self._limits.max_outer_steps - self._counters.outer_steps,
            "inner_steps": self._limits.max_inner_steps - self._counters.inner_steps,
            "skill_invocations": self._limits.max_skill_invocations - self._counters.skill_invocations,
            "tokens": self._limits.max_tokens - self._counters.tokens,
            "ms": self._limits.max_ms - (time.time() - self._counters.started_at) * 1000,
            "grounding_escalations": self._limits.max_grounding_escalations - self._counters.grounding_escalations,
            "teacher_calls": self._limits.max_teacher_calls - self._counters.teacher_calls,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "outer_steps": self._counters.outer_steps,
            "inner_steps": self._counters.inner_steps,
            "skill_invocations": self._counters.skill_invocations,
            "tokens": self._counters.tokens,
            "grounding_escalations": self._counters.grounding_escalations,
            "teacher_calls": self._counters.teacher_calls,
            "elapsed_ms": (time.time() - self._counters.started_at) * 1000,
        }

    # -- enforcement -------------------------------------------------------

    def _enforce(self) -> None:
        c, l = self._counters, self._limits
        if c.outer_steps > l.max_outer_steps:
            raise BudgetExceeded(f"outer_steps>{l.max_outer_steps}")
        if c.inner_steps > l.max_inner_steps:
            raise BudgetExceeded(f"inner_steps>{l.max_inner_steps}")
        if c.skill_invocations > l.max_skill_invocations:
            raise BudgetExceeded(f"skill_invocations>{l.max_skill_invocations}")
        if c.tokens > l.max_tokens:
            raise BudgetExceeded(f"tokens>{l.max_tokens}")
        if c.grounding_escalations > l.max_grounding_escalations:
            raise BudgetExceeded(f"grounding_escalations>{l.max_grounding_escalations}")
        if c.teacher_calls > l.max_teacher_calls:
            raise BudgetExceeded(f"teacher_calls>{l.max_teacher_calls}")
        elapsed_ms = (time.time() - c.started_at) * 1000
        if elapsed_ms > l.max_ms:
            raise BudgetExceeded(f"ms>{l.max_ms}")


__all__ = ["BudgetController", "BudgetExceeded"]
