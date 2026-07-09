"""
Versioned bank storage with transactional commit/rollback.

Provides:
  - query_skill(key) -> topK SkillCards
  - Snapshot-based versioning (Bank_k -> Bank_{k+1})
  - Atomic commit/rollback: build Bank' in-memory, validate, then commit or discard
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillCard:
    """Compact skill representation returned by query_skill()."""

    skill_id: str
    effects: Dict[str, Any] = field(default_factory=dict)
    typical_len: int = 0
    confusers: List[str] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class VersionedBankStore:
    """Versioned wrapper around SkillBankMVP with transactional semantics.

    Supports:
      - Creating a candidate bank in-memory (fork)
      - Committing or rolling back the candidate
      - Snapshot persistence with version tracking
      - Query API returning SkillCards
    """

    def __init__(
        self,
        bank: Any,
        bank_dir: str = "runs/skillbank",
        snapshot_prefix: str = "bank_v",
        max_snapshots: int = 20,
    ):
        self.bank = bank
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_prefix = snapshot_prefix
        self.max_snapshots = max_snapshots

        self._version = 0
        self._candidate: Optional[Any] = None
        self._last_good_bank: Optional[Any] = None

    @property
    def version(self) -> int:
        return self._version

    @property
    def current_bank(self) -> Any:
        return self.bank

    def fork(self) -> Any:
        """Create an in-memory candidate bank (deep copy of current).

        The candidate can be modified freely; changes are only applied
        to the main bank on commit().
        """
        self._candidate = copy.deepcopy(self.bank)
        return self._candidate

    @property
    def candidate(self) -> Optional[Any]:
        return self._candidate

    def commit(self, rebuild_indices: bool = True) -> int:
        """Accept the candidate bank as the new version.

        Saves a snapshot, advances the version counter, and optionally
        rebuilds retrieval indices.

        Returns the new version number.
        """
        if self._candidate is None:
            logger.warning("commit() called with no candidate — nothing to do")
            return self._version

        self._last_good_bank = copy.deepcopy(self.bank)
        self.bank = self._candidate
        self._candidate = None
        self._version += 1

        snapshot_path = self.bank_dir / f"{self.snapshot_prefix}{self._version}.jsonl"
        try:
            self.bank.save(str(snapshot_path))
            logger.info("Committed bank v%d → %s", self._version, snapshot_path)
        except Exception as exc:
            logger.error("Failed to save bank snapshot: %s", exc)

        self._prune_old_snapshots()

        if rebuild_indices:
            self._rebuild_indices()

        return self._version

    def rollback(self) -> None:
        """Discard the candidate bank and keep the current version."""
        self._candidate = None
        logger.info("Rolled back candidate — keeping bank v%d", self._version)

    def rollback_to_last_good(self) -> None:
        """Revert to the last committed bank version."""
        if self._last_good_bank is not None:
            self.bank = self._last_good_bank
            self._last_good_bank = None
            logger.info("Reverted to last good bank")
        else:
            logger.warning("No last-good bank available for rollback")

    def query_skill(self, key: str, top_k: int = 3) -> List[SkillCard]:
        """Query skills by natural-language key.

        Returns topK SkillCards from the current bank.
        """
        try:
            from skill_agents.query import SkillQueryEngine
            engine = SkillQueryEngine(self.bank)
            results = engine.query(key, top_k=top_k)
        except ImportError:
            results = []

        cards: List[SkillCard] = []
        for r in results:
            contract = r.get("contract", {})
            card = SkillCard(
                skill_id=r.get("skill_id", ""),
                effects={
                    "eff_add": contract.get("eff_add", []),
                    "eff_del": contract.get("eff_del", []),
                    "eff_event": contract.get("eff_event", []),
                },
                typical_len=contract.get("n_instances", 0),
                score=r.get("score", 0.0),
            )
            cards.append(card)
        return cards

    def load_version(self, version: int) -> bool:
        """Load a specific bank version from snapshot."""
        snapshot_path = self.bank_dir / f"{self.snapshot_prefix}{version}.jsonl"
        if not snapshot_path.exists():
            logger.warning("Snapshot v%d not found at %s", version, snapshot_path)
            return False

        try:
            self.bank.load(str(snapshot_path))
            self._version = version
            logger.info("Loaded bank v%d from %s", version, snapshot_path)
            return True
        except Exception as exc:
            logger.error("Failed to load bank v%d: %s", version, exc)
            return False

    def get_snapshot_versions(self) -> List[int]:
        """List available snapshot version numbers."""
        versions = []
        for p in self.bank_dir.glob(f"{self.snapshot_prefix}*.jsonl"):
            try:
                v = int(p.stem.replace(self.snapshot_prefix, ""))
                versions.append(v)
            except ValueError:
                pass
        return sorted(versions)

    def _prune_old_snapshots(self) -> None:
        """Remove oldest snapshots beyond max_snapshots."""
        versions = self.get_snapshot_versions()
        while len(versions) > self.max_snapshots:
            oldest = versions.pop(0)
            path = self.bank_dir / f"{self.snapshot_prefix}{oldest}.jsonl"
            try:
                path.unlink()
                logger.debug("Pruned old snapshot v%d", oldest)
            except Exception:
                pass

    def _rebuild_indices(self) -> None:
        """Rebuild retrieval indices for the current bank.

        Currently a no-op — SkillQueryEngine rebuilds on construction.
        Hook for future index persistence.
        """
        pass
