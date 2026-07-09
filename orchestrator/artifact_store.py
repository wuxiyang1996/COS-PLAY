"""`ArtifactStore` — local, file-backed storage for system artifacts.

PLAN-PIPELINE-ORCHESTRATOR §6 specifies the artifact taxonomy. Every
artifact lives in a typed sub-directory of `artifact_root`:

    {artifact_root}/episodes/<episode_id>.json
    {artifact_root}/skill_episodes/<episode_id>.json
    {artifact_root}/proposals/<proposal_id>.json
    {artifact_root}/evaluations/<evaluation_id>.json
    {artifact_root}/failures/<failure_id>.json
    {artifact_root}/snapshots/<snapshot_id>.json
    {artifact_root}/releases/<release_id>.json
    {artifact_root}/audit.jsonl

Reads are cheap (just glob/load). Writes are atomic via tmp-then-rename.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    proposal_to_json,
)
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.run_release import RunRelease
from data_structure.extensions.skill_episode import SkillEpisode
from data_structure.extensions.skill_evaluation import SkillEvaluationRecord


_AUDIT_LOG = "audit.jsonl"


class ArtifactStore:
    SUBDIRS = (
        "episodes",
        "skill_episodes",
        "proposals",
        "evaluations",
        "failures",
        "snapshots",
        "releases",
    )

    def __init__(self, root: str) -> None:
        self.root = root
        self._lock = threading.Lock()
        os.makedirs(root, exist_ok=True)
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(root, sub), exist_ok=True)

    # -- writes ------------------------------------------------------------

    def put_skill_episode(self, ep: SkillEpisode) -> str:
        return self._put_json(("skill_episodes", f"{ep.episode_id}.json"), ep.to_json())

    def put_proposal(self, proposal: BankMutationProposal) -> str:
        return self._put_json(("proposals", f"{proposal.proposal_id}.json"), proposal_to_json(proposal))

    def put_evaluation(self, ev: SkillEvaluationRecord) -> str:
        return self._put_json(("evaluations", f"{ev.evaluation_id}.json"), ev.to_json())

    def put_failure(self, ft: FailureTrace) -> str:
        return self._put_json(("failures", f"{ft.failure_id}.json"), ft.to_json())

    def put_release(self, rel: RunRelease) -> str:
        return self._put_json(("releases", f"{rel.release_id}.json"), rel.to_json())

    def put_snapshot(self, snapshot_id: str, payload: Dict[str, Any]) -> str:
        return self._put_json(("snapshots", f"{snapshot_id}.json"), payload)

    def put_episode(self, episode_id: str, payload: Dict[str, Any]) -> str:
        return self._put_json(("episodes", f"{episode_id}.json"), payload)

    def append_audit(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", time.time())
        path = os.path.join(self.root, _AUDIT_LOG)
        with self._lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")

    # -- reads -------------------------------------------------------------

    def get_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        return self._get_json(("evaluations", f"{evaluation_id}.json"))

    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        return self._get_json(("proposals", f"{proposal_id}.json"))

    def list_skill_episodes(self) -> Iterable[Dict[str, Any]]:
        return self._list_json("skill_episodes")

    def list_failures(self) -> Iterable[Dict[str, Any]]:
        return self._list_json("failures")

    def list_evaluations(self) -> Iterable[Dict[str, Any]]:
        return self._list_json("evaluations")

    def list_releases(self) -> List[Dict[str, Any]]:
        return list(self._list_json("releases"))

    # -- internals ---------------------------------------------------------

    def _put_json(self, parts: tuple[str, str], payload: Dict[str, Any]) -> str:
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # atomic tmp-then-rename
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=os.path.dirname(path), suffix=".tmp", encoding="utf-8"
        ) as tmp:
            json.dump(payload, tmp, indent=2, default=str)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
        return path

    def _get_json(self, parts: tuple[str, str]) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.root, *parts)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _list_json(self, sub: str) -> Iterable[Dict[str, Any]]:
        sub_dir = os.path.join(self.root, sub)
        if not os.path.exists(sub_dir):
            return []
        out = []
        for fn in sorted(os.listdir(sub_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(sub_dir, fn), "r", encoding="utf-8") as fh:
                try:
                    out.append(json.load(fh))
                except json.JSONDecodeError:
                    continue
        return out


__all__ = ["ArtifactStore"]
