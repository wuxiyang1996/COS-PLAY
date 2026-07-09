"""
Central function identifier for LoRA adapter routing.

Each skill-bank function maps to exactly one LoRA adapter.
Use ``SkillFunction`` as the routing key everywhere.

Three active GRPO-trained adapters: SEGMENT, CONTRACT, CURATOR.
BOUNDARY and RETRIEVAL are retained for backward compat with
``skill_agents`` base code but are **not** GRPO-trained here
(BOUNDARY reward is too indirect; RETRIEVAL uses the existing
decision-agent GRPO trainer).
"""

from __future__ import annotations

from enum import Enum


class SkillFunction(str, Enum):
    """Skill-bank functions, each optionally backed by a LoRA adapter.

    Active GRPO adapters: SEGMENT, CONTRACT, CURATOR.
    Legacy (kept for compat): BOUNDARY, RETRIEVAL.
    """

    BOUNDARY = "boundary"           # legacy — not GRPO-trained in this module
    SEGMENT = "segment"             # GRPO-trained
    CONTRACT = "contract"           # GRPO-trained
    RETRIEVAL = "retrieval"         # legacy — uses decision-agent GRPO
    CURATOR = "curator"             # GRPO-trained
    ACTION_TAKING = "action_taking"       # decision-agent GRPO
    SKILL_SELECTION = "skill_selection"   # decision-agent GRPO

    @classmethod
    def from_str(cls, value: str) -> "SkillFunction":
        """Case-insensitive lookup, e.g. ``SkillFunction.from_str("Boundary")``."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown skill function {value!r}. Choose from: {valid}")

    @property
    def adapter_name(self) -> str:
        """Name used when registering / selecting a PEFT adapter."""
        return self.value
