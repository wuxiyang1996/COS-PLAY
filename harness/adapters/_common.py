"""Helpers shared by every adapter.

Adapters all need the same boilerplate:
  - Walk `skill.protocol` hop by hop.
  - Bind hop slots from the runtime context.
  - Honor budget (hops, ms).
  - Honor `dry_run` for the gate replay path.

We expose two helpers — `iter_hops` and `apply_budget` — so adapter bodies
stay focused on the one thing only they can do (talk to the env).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from data_structure.extensions.skill_record import SkillRecord


@dataclass
class HopBindings:
    bindings: Dict[str, Any]
    state_facts: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, value: Any) -> Any:
        """Replace `${name}` slots in strings with bound values."""
        if not isinstance(value, str) or "${" not in value:
            return value
        out = value
        for k, v in {**self.state_facts, **self.bindings}.items():
            out = out.replace(f"${{{k}}}", str(v))
        return out

    def resolve_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self.resolve(v) for k, v in payload.items()}


def iter_hops(skill: SkillRecord) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for i, hop in enumerate(skill.protocol):
        if not isinstance(hop, dict):
            continue
        yield i, hop


@dataclass
class BudgetGuard:
    max_hops: int = 8
    max_ms: float = 30_000.0
    started_at: float = field(default_factory=time.time)

    def check(self, hop_index: int) -> Optional[str]:
        if hop_index >= self.max_hops:
            return f"budget:max_hops={self.max_hops}"
        if (time.time() - self.started_at) * 1000 > self.max_ms:
            return f"budget:max_ms={self.max_ms}"
        return None


def normalize_hop_action(hop: Dict[str, Any]) -> str:
    """Map a hop's `op` / `action` field to a canonical action_type."""
    return str(hop.get("action") or hop.get("op") or hop.get("type") or "STEP").upper()


__all__ = ["BudgetGuard", "HopBindings", "iter_hops", "normalize_hop_action"]
