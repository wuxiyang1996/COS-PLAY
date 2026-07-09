"""`RewardLogger` — central sink for SkillEpisode + reward events.

PLAN-HARNESS §5.6: a single sink so the orchestrator, gate, and
training cadence all read from the same source of truth.

This implementation is a thin file-backed JSONL log; the orchestrator's
`ArtifactStore` provides longer-lived storage. We intentionally keep
this module dependency-free so tests can use it standalone — the
teacher-anchored normalization helper from
``common.reward_anchors`` is imported lazily inside the writers so
``harness/`` continues to have no module-level non-stdlib import surface.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from data_structure.extensions.skill_episode import SkillEpisode


@dataclass
class RewardLogEntry:
    episode_id: str
    skill_id: str
    skill_version: str
    domain: str
    success: bool
    score: Optional[float]
    cost: Dict[str, float] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    transfer_label: Optional[str] = None
    timestamp: float = 0.0
    # Teacher-anchored normalized reward (training_notes §4.5). Computed
    # via ``common.reward_anchors.normalize_reward(score, domain)``;
    # ``None`` here means "no anchor available" (vs ``0.0`` which means
    # "scored zero against a known anchor"). Additive only — ``score``
    # is unchanged so legacy readers see no behavior change. Required
    # by Layer-D and W&B aggregates that compare across phases / games
    # with wildly different reward scales.
    reward_normalized: Optional[float] = None
    # Discriminator for the multi-kind JSONL (T2.4). ``"skill_episode"``
    # is the original aggregate-per-episode entry shape; ``GRPOStepLogEntry``
    # carries ``"grpo_step"`` and is emitted by the trainer at the per-step
    # reward attach site (``trainer/coevolution/episode_runner.py``).
    kind: str = "skill_episode"

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "episode_id": self.episode_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "domain": self.domain,
            "success": self.success,
            "score": self.score,
            "reward_normalized": self.reward_normalized,
            "cost": dict(self.cost),
            "parent_run_id": self.parent_run_id,
            "transfer_label": self.transfer_label,
            "timestamp": self.timestamp,
        }


@dataclass
class GRPOStepLogEntry:
    """Per-step training-reward entry (T2.4 single-sink invariant).

    The trainer's GRPO emit site (``episode_runner.py`` action_taking
    + skill_selection branches) attaches a scalar reward to a
    ``GRPORecord``; the same scalar — together with the adapter,
    step index, and metadata — is mirrored here so eval and training
    read from one source. This entry is *additive*: existing
    ``RewardLogger.entries()`` continue to return only
    ``skill_episode`` rows; ``grpo_step_entries()`` returns the new
    rows. Both share the same JSONL file (kind-discriminated).
    """

    kind: str = "grpo_step"
    episode_id: str = ""
    game: str = ""
    adapter: str = ""           # "action_taking" | "skill_selection"
    step: int = 0
    reward: float = 0.0
    # Teacher-anchored normalized reward (training_notes §4.5).
    # See ``RewardLogEntry.reward_normalized`` for semantics. Additive
    # only — ``reward`` is unchanged so the GRPO advantage path
    # (which reads raw rewards) is untouched.
    reward_normalized: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "episode_id": self.episode_id,
            "game": self.game,
            "adapter": self.adapter,
            "step": self.step,
            "reward": self.reward,
            "reward_normalized": self.reward_normalized,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }


class RewardLogger:
    """Append-only JSONL sink. Thread-safe via a single mutex.

    Two record kinds share one log: per-``SkillEpisode`` aggregates
    (the original) and per-step GRPO training rewards (T2.4). They
    are kind-discriminated in JSONL but live in separate in-memory
    buffers for fast typed reads.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        *,
        episode_dir: Optional[str] = None,
        clock: Callable[[], float] = time.time,
        reward_anchors: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        self._log_path = log_path
        self._episode_dir = episode_dir
        self._clock = clock
        self._lock = threading.Lock()
        self._memory: List[RewardLogEntry] = []
        self._grpo_memory: List[GRPOStepLogEntry] = []
        # Teacher-anchored normalization table (training_notes §4.5).
        # ``None`` means "use the static fallback from
        # ``common.reward_anchors.TEACHER_REWARD_ANCHORS``"; pass an
        # explicit dict to override (e.g. orchestrator may have already
        # auto-derived from the cold-start ``rollout_summary.json``).
        # Each value may itself be ``None`` to mark "no anchor" for
        # that game (downstream emits ``reward_normalized=None``).
        self._reward_anchors: Optional[Dict[str, Optional[float]]] = (
            dict(reward_anchors) if reward_anchors is not None else None
        )
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if episode_dir:
            os.makedirs(episode_dir, exist_ok=True)

    def set_reward_anchors(
        self, anchors: Optional[Dict[str, Optional[float]]]
    ) -> None:
        """Late-bind the per-game teacher anchor table.

        Useful when the orchestrator constructs ``RewardLogger`` before
        cold-start auto-derivation has run (so the static fallback is
        used initially), then upgrades to the auto-derived table once
        ready.
        """

        with self._lock:
            self._reward_anchors = (
                dict(anchors) if anchors is not None else None
            )

    def _normalize(self, raw: Optional[float], game: str) -> Optional[float]:
        """Apply teacher-anchored normalization (lazy import to keep
        ``harness/`` standalone)."""

        if raw is None or not game:
            return None
        try:
            from common.reward_anchors import normalize_reward
        except Exception:  # pragma: no cover  (harness might be vendored)
            return None
        try:
            return normalize_reward(raw, game, anchors=self._reward_anchors)
        except Exception:  # pragma: no cover  (defensive)
            return None

    # -- writes ------------------------------------------------------------

    def log_episode(self, episode: SkillEpisode) -> RewardLogEntry:
        outcome = episode.outcome
        score = outcome.score if outcome else None
        entry = RewardLogEntry(
            episode_id=episode.episode_id,
            skill_id=episode.skill_id,
            skill_version=episode.skill_version,
            domain=episode.domain,
            success=bool(outcome and outcome.success),
            score=score,
            reward_normalized=self._normalize(score, episode.domain),
            cost=dict(episode.cost),
            parent_run_id=episode.parent_run_id,
            transfer_label=episode.transfer_label,
            timestamp=self._clock(),
        )
        with self._lock:
            self._memory.append(entry)
            if self._log_path:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_json()) + "\n")
            if self._episode_dir:
                ep_path = os.path.join(self._episode_dir, f"{episode.episode_id}.json")
                with open(ep_path, "w", encoding="utf-8") as fh:
                    json.dump(episode.to_json(), fh, indent=2)
        return entry

    def log_grpo_record(
        self,
        *,
        episode_id: str,
        adapter: str,
        step: int,
        reward: float,
        game: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GRPOStepLogEntry:
        """Mirror a per-step GRPO reward into the unified JSONL sink.

        Called from ``trainer.coevolution.episode_runner`` immediately
        after ``grpo_records.append(GRPORecord(...))`` so eval and
        training share the same source of truth (T2.4 single-sink
        invariant). Primitives only — ``GRPORecord`` is *not* imported
        here so ``harness/`` keeps zero dependence on ``trainer/``.
        """

        entry = GRPOStepLogEntry(
            episode_id=episode_id,
            game=game,
            adapter=adapter,
            step=int(step),
            reward=float(reward),
            reward_normalized=self._normalize(float(reward), game),
            metadata=dict(metadata or {}),
            timestamp=self._clock(),
        )
        with self._lock:
            self._grpo_memory.append(entry)
            if self._log_path:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_json()) + "\n")
        return entry

    # -- reads -------------------------------------------------------------

    def entries(self) -> List[RewardLogEntry]:
        with self._lock:
            return list(self._memory)

    def grpo_step_entries(self) -> List[GRPOStepLogEntry]:
        """All per-step GRPO entries written via ``log_grpo_record``."""

        with self._lock:
            return list(self._grpo_memory)

    def filter(
        self,
        *,
        skill_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Iterable[RewardLogEntry]:
        for e in self.entries():
            if skill_id and e.skill_id != skill_id:
                continue
            if domain and e.domain != domain:
                continue
            yield e


__all__ = ["GRPOStepLogEntry", "RewardLogEntry", "RewardLogger"]
