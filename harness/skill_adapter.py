"""`SkillAdapter` protocol.

PLAN-HARNESS §5.4 — every domain that wants to *execute* a skill must
register one. Adapters do not own the skill protocol: they receive a
`SkillRecord` plus a runtime context and return a structured run result.

Adapters are the *only* place that talk to a concrete env / tool surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from common.enums import SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord


@dataclass
class AdapterRunContext:
    """Per-invocation context passed by the harness."""

    state: StateSchema
    bindings: Dict[str, Any] = field(default_factory=dict)   # slot fills
    parent_run_id: Optional[str] = None
    parent_episode_id: Optional[str] = None
    budget: Dict[str, float] = field(default_factory=dict)   # tokens, hops, ms
    seed: Optional[int] = None
    dry_run: bool = False                                    # gate replay mode


@dataclass
class AdapterRunResult:
    """What an adapter returns to the harness.

    The harness will wrap this into a `SkillEpisode` and persist it; the
    adapter does not write artifacts directly.
    """

    success: bool
    contract_satisfied: bool
    final_state: Optional[StateSchema] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)   # raw step records
    new_evidence: List[EvidenceRef] = field(default_factory=list)
    answer: Optional[Any] = None
    score: Optional[float] = None
    abort_reason: Optional[str] = None
    cost: Dict[str, float] = field(default_factory=dict)
    diagnostic_label: Optional[str] = None                       # PLAN-HARNESS §6.4
    extra: Dict[str, Any] = field(default_factory=dict)


class SkillAdapter(ABC):
    """Domain-specific executor for a `SkillRecord`."""

    #: Unique name (e.g. "gymv", "browser", "osworld"); also the domain key.
    name: str = ""
    #: Which `SkillType`s this adapter can execute.
    supported_types: tuple[SkillType, ...] = (SkillType.ACTION, SkillType.MIXED)

    @abstractmethod
    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool:
        """Cheap, side-effect-free admissibility check."""

    @abstractmethod
    def run(self, skill: SkillRecord, ctx: AdapterRunContext) -> AdapterRunResult:
        """Execute the skill. May not throw on contract violation — return
        `success=False` and an `abort_reason` instead."""


class _AdapterProtocol(Protocol):
    """Structural typing for adapters that don't subclass `SkillAdapter`."""

    name: str
    supported_types: tuple[SkillType, ...]

    def can_handle(self, skill: SkillRecord, state: StateSchema) -> bool: ...

    def run(self, skill: SkillRecord, ctx: AdapterRunContext) -> AdapterRunResult: ...


__all__ = [
    "AdapterRunContext",
    "AdapterRunResult",
    "SkillAdapter",
    "_AdapterProtocol",
]
