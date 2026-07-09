"""Skill Bank — typed, four-store, lifecycle-managed skill registry.

This module is the canonical implementation of PLAN-UNIFIED-SKILL-GATE
§5 (storage split) and §6 (lifecycle ownership).

It coexists with the legacy `skill_agents/skill_bank/` package — that
module is the historical Stage-3 MVP and continues to power existing
mining flows. The new `skill_bank/` package is the *target* layout that
Harness/Orchestrator/Crafter import from. A small adapter
(`skill_bank.legacy_bridge`) provides a one-way migration path for
existing Stage-3 records.

Architectural rule (mechanically enforced):
  * `lifecycle.SkillLifecycleManager` is the **only** symbol that may
    write to any of the four stores. Direct writes are blocked by a
    `_locked` sentinel inside `stores.SkillStore`.
"""

from skill_bank.lifecycle import LifecycleError, SkillLifecycleManager
from skill_bank.repository import SkillRepository
from skill_bank.stores import SkillStore, StoreLockedError, StoreName

__all__ = [
    "LifecycleError",
    "SkillLifecycleManager",
    "SkillRepository",
    "SkillStore",
    "StoreLockedError",
    "StoreName",
]
