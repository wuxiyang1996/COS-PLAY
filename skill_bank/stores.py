"""Four physical skill stores (PLAN-UNIFIED-SKILL-GATE §5).

The split is mechanical:
  * `draft_store`     — DRAFT, REJECTED
  * `candidate_store` — CANDIDATE
  * `active_store`    — SHADOW, PROVISIONAL, ACTIVE
  * `archive_store`   — DEPRECATED, ROLLED_BACK

Mechanical isolation enforces "no promotion without gate": a CANDIDATE
record physically does not exist in `active_store`, so the runtime
cannot accidentally execute it as if it were ACTIVE.

External callers may *read* any store. Mutations are gated by a
`_locked` sentinel that only `SkillLifecycleManager` knows how to clear.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

from common.enums import SkillStatus
from data_structure.extensions.skill_record import SkillRecord


class StoreName(str, Enum):
    DRAFT = "draft_store"
    CANDIDATE = "candidate_store"
    ACTIVE = "active_store"
    ARCHIVE = "archive_store"


# Mapping from canonical SkillStatus -> physical store.
_STATUS_TO_STORE = {
    SkillStatus.DRAFT: StoreName.DRAFT,
    SkillStatus.REJECTED: StoreName.DRAFT,
    SkillStatus.CANDIDATE: StoreName.CANDIDATE,
    SkillStatus.SHADOW: StoreName.ACTIVE,
    SkillStatus.PROVISIONAL: StoreName.ACTIVE,
    SkillStatus.ACTIVE: StoreName.ACTIVE,
    SkillStatus.DEPRECATED: StoreName.ARCHIVE,
    SkillStatus.ROLLED_BACK: StoreName.ARCHIVE,
}


def store_for_status(status: SkillStatus) -> StoreName:
    return _STATUS_TO_STORE[status]


class StoreLockedError(RuntimeError):
    """Raised when something other than `SkillLifecycleManager` tries
    to write to a store directly."""


@dataclass
class _Cell:
    record: SkillRecord
    raw_path: Optional[str] = None


class SkillStore:
    """A single physical store backed by a directory of JSON files."""

    def __init__(self, name: StoreName, root: Optional[str] = None) -> None:
        self.name = name
        self._root = root
        self._lock = threading.RLock()
        self._cells: Dict[str, _Cell] = {}
        self._mutation_token: Optional[object] = None  # set by lifecycle manager
        if root:
            os.makedirs(root, exist_ok=True)
            self._load_from_disk()

    # ---- read API -------------------------------------------------------

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        with self._lock:
            cell = self._cells.get(skill_id)
            return cell.record if cell else None

    def all(self) -> List[SkillRecord]:
        with self._lock:
            return [c.record for c in self._cells.values()]

    def __contains__(self, skill_id: str) -> bool:  # type: ignore[override]
        with self._lock:
            return skill_id in self._cells

    def __len__(self) -> int:
        with self._lock:
            return len(self._cells)

    # ---- gated write API (callable only via context manager) -----------

    @contextlib.contextmanager
    def _unlocked(self, token: object) -> Iterator[None]:
        with self._lock:
            prev = self._mutation_token
            self._mutation_token = token
            try:
                yield
            finally:
                self._mutation_token = prev

    def put(self, record: SkillRecord, *, token: object) -> None:
        if token is not self._mutation_token:
            raise StoreLockedError(
                f"Direct write to {self.name.value} is forbidden — go through "
                f"SkillLifecycleManager.transition()."
            )
        expected_store = store_for_status(record.status)
        if expected_store != self.name:
            raise StoreLockedError(
                f"Record {record.skill_id} (status={record.status.value}) "
                f"belongs in {expected_store.value}, not {self.name.value}."
            )
        with self._lock:
            self._cells[record.skill_id] = _Cell(record=record, raw_path=self._path_for(record))
            self._persist(record)

    def remove(self, skill_id: str, *, token: object) -> Optional[SkillRecord]:
        if token is not self._mutation_token:
            raise StoreLockedError(
                f"Direct delete from {self.name.value} is forbidden."
            )
        with self._lock:
            cell = self._cells.pop(skill_id, None)
            if cell and cell.raw_path and os.path.exists(cell.raw_path):
                os.remove(cell.raw_path)
            return cell.record if cell else None

    # ---- persistence helpers --------------------------------------------

    def _path_for(self, record: SkillRecord) -> Optional[str]:
        if not self._root:
            return None
        return os.path.join(self._root, f"{record.skill_id}.json")

    def _persist(self, record: SkillRecord) -> None:
        path = self._path_for(record)
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record.to_json(), fh, indent=2)

    def _load_from_disk(self) -> None:
        assert self._root is not None
        for fn in os.listdir(self._root):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self._root, fn)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            try:
                record = _record_from_json(data)
            except Exception:                                  # noqa: BLE001
                continue
            if store_for_status(record.status) != self.name:
                # Misplaced record: skip; lifecycle manager will rebalance.
                continue
            self._cells[record.skill_id] = _Cell(record=record, raw_path=path)


# ---------------------------------------------------------------- helpers


def _record_from_json(data: Dict[str, Any]) -> SkillRecord:
    from common.enums import SkillSourceType, SkillStatus, SkillType
    from data_structure.extensions.skill_record import SkillContract

    contract_raw = data.get("contract", {}) or {}
    contract = SkillContract(
        preconditions=list(contract_raw.get("preconditions", [])),
        effects_add=list(contract_raw.get("effects_add", [])),
        effects_del=list(contract_raw.get("effects_del", [])),
        belief_progress=list(contract_raw.get("belief_progress", [])),
        grounding_progress=list(contract_raw.get("grounding_progress", [])),
        expected_evidence_roles=list(contract_raw.get("expected_evidence_roles", [])),
        success_criteria=list(contract_raw.get("success_criteria", [])),
        abort_criteria=list(contract_raw.get("abort_criteria", [])),
    )
    return SkillRecord(
        skill_id=str(data["skill_id"]),
        name=str(data["name"]),
        skill_type=SkillType(data["skill_type"]),
        source_type=SkillSourceType(data["source_type"]),
        status=SkillStatus(data["status"]),
        version=str(data.get("version", "v1")),
        feasible_domains=list(data.get("feasible_domains", [])),
        source_domains=list(data.get("source_domains", [])),
        transfer_target_domains=list(data.get("transfer_target_domains", [])),
        verified_domains=list(data.get("verified_domains", [])),
        feasible_tasks=list(data.get("feasible_tasks", [])),
        verified_tasks=list(data.get("verified_tasks", [])),
        adapter_history=[dict(x) for x in data.get("adapter_history", [])],
        false_binding_patterns=[dict(x) for x in data.get("false_binding_patterns", [])],
        protocol=list(data.get("protocol", [])),
        contract=contract,
        parent_skill_ids=list(data.get("parent_skill_ids", [])),
        proposal_id=data.get("proposal_id"),
        crafted_at=data.get("crafted_at"),
        last_evaluation_id=data.get("last_evaluation_id"),
        metrics=dict(data.get("metrics", {})),
        notes=str(data.get("notes", "")),
        tags=list(data.get("tags", [])),
    )


__all__ = [
    "SkillStore",
    "StoreLockedError",
    "StoreName",
    "store_for_status",
]
