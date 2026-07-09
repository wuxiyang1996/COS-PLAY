"""`RunRelease` — frozen, atomically-promoted runtime snapshot.

Spec: PLAN-PIPELINE-ORCHESTRATOR §7 (PromotionOrchestrator atomic promotion).

A run release captures the *exact* (bank ⊕ adapters ⊕ config) bundle the
runtime is currently using. Rollback is just `current_release_id ←
previous_release_id`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from common.ids import new_snapshot_id, schema_hash


@dataclass
class RunRelease:
    release_id: str = field(default_factory=new_snapshot_id)
    parent_release_id: Optional[str] = None
    bank_snapshot_path: str = ""
    adapter_snapshot_paths: Dict[str, str] = field(default_factory=dict)
    config_snapshot_path: str = ""
    promoted_skill_ids: List[str] = field(default_factory=list)
    deprecated_skill_ids: List[str] = field(default_factory=list)
    gate_evaluation_ids: List[str] = field(default_factory=list)
    eval_suite_id: str = ""
    scoreboard_path: str = ""
    notes: str = ""
    created_at: Optional[float] = None

    def content_hash(self) -> str:
        return schema_hash(
            {
                "bank_snapshot_path": self.bank_snapshot_path,
                "adapter_snapshot_paths": dict(sorted(self.adapter_snapshot_paths.items())),
                "config_snapshot_path": self.config_snapshot_path,
                "promoted_skill_ids": sorted(self.promoted_skill_ids),
                "deprecated_skill_ids": sorted(self.deprecated_skill_ids),
                "eval_suite_id": self.eval_suite_id,
                "scoreboard_path": self.scoreboard_path,
            }
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "parent_release_id": self.parent_release_id,
            "bank_snapshot_path": self.bank_snapshot_path,
            "adapter_snapshot_paths": dict(self.adapter_snapshot_paths),
            "config_snapshot_path": self.config_snapshot_path,
            "promoted_skill_ids": list(self.promoted_skill_ids),
            "deprecated_skill_ids": list(self.deprecated_skill_ids),
            "gate_evaluation_ids": list(self.gate_evaluation_ids),
            "eval_suite_id": self.eval_suite_id,
            "scoreboard_path": self.scoreboard_path,
            "notes": self.notes,
            "created_at": self.created_at,
            "content_hash": self.content_hash(),
        }


__all__ = ["RunRelease"]
