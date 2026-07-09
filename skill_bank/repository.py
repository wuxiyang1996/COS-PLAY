"""`SkillRepository` — read-only multi-store query surface.

The Harness, Actor, and Crafter all read through this class. Mutations
go through `SkillLifecycleManager` instead.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from common.enums import SkillStatus
from data_structure.extensions.skill_record import SkillRecord
from skill_bank.stores import SkillStore, StoreName


class SkillRepository:
    def __init__(
        self,
        *,
        draft_store: SkillStore,
        candidate_store: SkillStore,
        active_store: SkillStore,
        archive_store: SkillStore,
    ) -> None:
        assert draft_store.name == StoreName.DRAFT
        assert candidate_store.name == StoreName.CANDIDATE
        assert active_store.name == StoreName.ACTIVE
        assert archive_store.name == StoreName.ARCHIVE
        self.draft = draft_store
        self.candidate = candidate_store
        self.active = active_store
        self.archive = archive_store

    # -- single-skill lookup ----------------------------------------------

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        for store in self._stores():
            r = store.get(skill_id)
            if r is not None:
                return r
        return None

    def store_of(self, skill_id: str) -> Optional[StoreName]:
        for store in self._stores():
            if skill_id in store:
                return store.name
        return None

    # -- bulk views --------------------------------------------------------

    def runnable(self, *, include_shadow: bool = True) -> List[SkillRecord]:
        out: List[SkillRecord] = []
        for r in self.active.all():
            if r.status == SkillStatus.SHADOW and not include_shadow:
                continue
            out.append(r)
        return out

    def candidates(self) -> List[SkillRecord]:
        return self.candidate.all()

    def drafts(self) -> List[SkillRecord]:
        return self.draft.all()

    def archive_records(self) -> List[SkillRecord]:
        return self.archive.all()

    def all(self) -> List[SkillRecord]:
        out: List[SkillRecord] = []
        for store in self._stores():
            out.extend(store.all())
        return out

    def by_status(self, status: SkillStatus) -> List[SkillRecord]:
        return [r for r in self.all() if r.status == status]

    def by_domain(self, domain: str) -> List[SkillRecord]:
        return [r for r in self.all() if domain in r.feasible_domains]

    def _stores(self) -> Iterable[SkillStore]:
        return (self.active, self.candidate, self.draft, self.archive)


__all__ = ["SkillRepository"]
