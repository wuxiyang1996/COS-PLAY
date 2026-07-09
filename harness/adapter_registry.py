"""`AdapterRegistry` — central registry of `SkillAdapter` instances.

PLAN-HARNESS §5.4: there is exactly one registry per process. Adapters
are looked up by *(domain, skill_type)*; if no adapter is registered for
a (domain, type) pair the harness reports skill ineligible — it does not
silently fall back.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from common.enums import SkillType
from harness.skill_adapter import SkillAdapter


_AdapterKey = Tuple[str, SkillType]


class AdapterRegistry:
    def __init__(self) -> None:
        self._by_key: Dict[_AdapterKey, SkillAdapter] = {}
        self._by_name: Dict[str, SkillAdapter] = {}

    # -- registration ------------------------------------------------------

    def register(self, adapter: SkillAdapter) -> None:
        if not adapter.name:
            raise ValueError("SkillAdapter.name must be a non-empty string")
        if adapter.name in self._by_name:
            raise ValueError(f"Adapter {adapter.name!r} already registered")
        self._by_name[adapter.name] = adapter
        for skill_type in adapter.supported_types:
            key = (adapter.name, skill_type)
            if key in self._by_key:
                raise ValueError(f"Adapter key {key!r} already registered")
            self._by_key[key] = adapter

    def unregister(self, name: str) -> None:
        adapter = self._by_name.pop(name, None)
        if adapter is None:
            return
        for skill_type in adapter.supported_types:
            self._by_key.pop((name, skill_type), None)

    # -- lookup ------------------------------------------------------------

    def get(self, domain: str, skill_type: SkillType) -> Optional[SkillAdapter]:
        return self._by_key.get((domain, skill_type))

    def get_by_name(self, name: str) -> Optional[SkillAdapter]:
        return self._by_name.get(name)

    def domains(self) -> List[str]:
        return sorted({d for (d, _t) in self._by_key.keys()})

    def all(self) -> Iterable[SkillAdapter]:
        return list(self._by_name.values())


__all__ = ["AdapterRegistry"]
