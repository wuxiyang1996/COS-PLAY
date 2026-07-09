"""ID minting and content-hash helpers (PLAN-PIPELINE-ORCHESTRATOR §6 schemas).

Every artifact written to disk gets a stable, monotonically increasing ID.
Hashes are content-addressed (SHA-256 of canonical JSON) so the gate /
promotion path can verify "the proposal we approved is the proposal we ship".
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any


def _ts_prefix() -> str:
    # Millisecond-resolution sortable prefix.
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _short_uuid() -> str:
    return uuid.uuid4().hex[:10]


def new_episode_id() -> str:
    return f"ep-{_ts_prefix()}-{_short_uuid()}"


def new_run_id() -> str:
    return f"run-{_ts_prefix()}-{_short_uuid()}"


def new_skill_id(family: str = "skill") -> str:
    return f"{family}-{_short_uuid()}"


def new_proposal_id() -> str:
    return f"prop-{_ts_prefix()}-{_short_uuid()}"


def new_snapshot_id() -> str:
    return f"snap-{_ts_prefix()}-{_short_uuid()}"


def new_span_id() -> str:
    return f"span-{_short_uuid()}"


def schema_hash(payload: Any) -> str:
    """Deterministic content hash for a JSON-serialisable payload.

    Used by the gate to bind a `SkillEvaluationRecord` to the *exact*
    `SkillRecord` that was evaluated — promotion is rejected if the
    content hash drifts (PLAN-UNIFIED-SKILL-GATE §6 PromotionOrchestrator).
    """

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    # Tolerate dataclasses, enums, sets, and tuples in record payloads.
    if hasattr(value, "to_json"):
        return value.to_json()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    raise TypeError(f"Cannot JSON-serialise {type(value).__name__}")


__all__ = [
    "new_episode_id",
    "new_proposal_id",
    "new_run_id",
    "new_skill_id",
    "new_snapshot_id",
    "new_span_id",
    "schema_hash",
]
