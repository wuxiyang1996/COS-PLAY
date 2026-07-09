"""`SnapshotManager` — immutable bank+config snapshots for releases.

PLAN-PIPELINE-ORCHESTRATOR §7. A snapshot is a JSON payload containing
enough state to reconstitute the runtime: every active SkillRecord, the
adapter registry signature, and the active config hash.

Snapshots are content-addressed; identical bank state hashes to the same
snapshot id.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from common.ids import new_snapshot_id, schema_hash
from data_structure.extensions.skill_record import SkillRecord
from orchestrator.artifact_store import ArtifactStore


class SnapshotManager:
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    def take(
        self,
        *,
        active_records: Iterable[SkillRecord],
        adapter_signature: List[str],
        config_payload: Dict[str, Any],
        notes: str = "",
    ) -> Dict[str, Any]:
        records = [r.to_json() for r in active_records]
        body = {
            "skills": records,
            "adapter_signature": list(adapter_signature),
            "config_payload": config_payload,
        }
        body_hash = schema_hash(body)
        snapshot_id = new_snapshot_id()
        payload = {
            "snapshot_id": snapshot_id,
            "created_at": time.time(),
            "notes": notes,
            "body_hash": body_hash,
            "body": body,
        }
        self._store.put_snapshot(snapshot_id, payload)
        return payload

    def load(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self._store.root, "snapshots", f"{snapshot_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


__all__ = ["SnapshotManager"]
