"""
SkillBank co-evolution callback for VERL-based training.

Integrates all skill agent stages (boundary proposal, segmentation decode,
contract learning, bank maintenance) into the training loop as a unified
tool-calling pipeline.  Runs after each actor update at a configurable
cadence, converting rollout trajectories into the EM pipeline, updating
the skill bank transactionally, and persisting trajectory segmentations
so they can be refined across training steps.

Usage::

    from trainer.decision.coevolution_callback import (
        CoEvolutionConfig,
        SkillBankCoEvolutionCallback,
    )

    callback = SkillBankCoEvolutionCallback(
        config=CoEvolutionConfig(bank_update_cadence=10),
        initial_bank=bank,
        envs=envs,
        val_envs=val_envs,
    )
    # Inside GameAITrainer.fit():
    #   coevo_metrics = callback.on_step_end(global_step, batch, metrics)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from trainer.common.metrics import RolloutRecord, SkillBankMetrics
from trainer.skillbank.bank_io.bank_store import VersionedBankStore
from trainer.skillbank.bank_io.diff_logger import DiffLogger
from trainer.skillbank.em_trainer import EMConfig, EMTrainer
from trainer.skillbank.ingest_rollouts import TrajectoryForEM, ingest_rollouts
from trainer.skillbank.stages.stage1_propose_cuts import CutCandidate, ProposeCutsConfig, propose_cuts
from trainer.skillbank.stages.stage2_decode import DecodeConfig, DecodeResult, decode_batch, decode_trajectory
from trainer.skillbank.stages.stage3_contracts import (
    ContractLearningConfig,
    ContractLearningResult,
    apply_contracts_to_bank,
    learn_contracts,
)
from trainer.skillbank.stages.stage4_update import UpdateConfig, UpdateResult, run_update

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoEvolutionConfig:
    """Configuration for the SkillBank co-evolution callback."""

    bank_update_cadence: int = 10
    em_max_iterations: int = 3
    min_pass_rate: float = 0.6
    max_new_rate: float = 0.3
    bank_dir: str = "runs/skillbank"
    segmentation_store_path: str = "runs/skillbank/segmentations.jsonl"

    max_trajectories_per_update: int = 256
    enable_tool_call_reward: bool = True
    tool_call_reward_weight: float = 0.1

    propose_cuts: Optional[ProposeCutsConfig] = None
    decode: Optional[DecodeConfig] = None
    contracts: Optional[ContractLearningConfig] = None
    update: Optional[UpdateConfig] = None

    predicate_vocabulary: Optional[List[str]] = None
    p_thresh: float = 0.7
    smoothing_window: int = 3


# ---------------------------------------------------------------------------
# Persistent segmentation store
# ---------------------------------------------------------------------------

@dataclass
class SegmentationEntry:
    """Stored segmentation for one trajectory, updatable across training."""

    traj_id: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    bank_version: int = 0
    global_step: int = 0
    timestamp: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "segments": self.segments,
            "bank_version": self.bank_version,
            "global_step": self.global_step,
            "timestamp": self.timestamp,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentationEntry":
        return cls(
            traj_id=d["traj_id"],
            segments=d.get("segments", []),
            bank_version=d.get("bank_version", 0),
            global_step=d.get("global_step", 0),
            timestamp=d.get("timestamp", 0.0),
            tool_calls=d.get("tool_calls", []),
        )


class SegmentationStore:
    """Persistent store for trajectory segmentations.

    Segmentations are keyed by trajectory ID and updated each time the
    EM pipeline re-segments that trajectory.  Older segmentations are
    kept for history; the latest version is returned by ``get()``.
    """

    def __init__(self, path: str = "runs/skillbank/segmentations.jsonl"):
        self._path = Path(path)
        self._entries: Dict[str, SegmentationEntry] = {}
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def update(
        self,
        traj_id: str,
        segments: List[Dict[str, Any]],
        bank_version: int,
        global_step: int = 0,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update the segmentation for a trajectory."""
        self._entries[traj_id] = SegmentationEntry(
            traj_id=traj_id,
            segments=segments,
            bank_version=bank_version,
            global_step=global_step,
            timestamp=time.time(),
            tool_calls=tool_calls or [],
        )
        self._version += 1

    def get(self, traj_id: str) -> Optional[SegmentationEntry]:
        return self._entries.get(traj_id)

    def get_all(self) -> Dict[str, SegmentationEntry]:
        return dict(self._entries)

    def get_segments_for_traj(self, traj_id: str) -> List[Dict[str, Any]]:
        entry = self._entries.get(traj_id)
        return entry.segments if entry else []

    def __len__(self) -> int:
        return len(self._entries)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        logger.debug("Saved %d segmentations to %s", len(self._entries), self._path)

    def load(self) -> None:
        if not self._path.exists():
            return
        self._entries.clear()
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                entry = SegmentationEntry.from_dict(d)
                self._entries[entry.traj_id] = entry
        logger.info("Loaded %d segmentations from %s", len(self._entries), self._path)


# ---------------------------------------------------------------------------
# Skill agent tool pipeline
# ---------------------------------------------------------------------------

class SkillAgentToolPipeline:
    """Wraps all four skill agent stages as a callable tool pipeline.

    Each stage is registered as a named tool that can be invoked independently
    or sequentially through ``run_full_pipeline()``.  This ensures that every
    skill agent stage participates in the training loop and that segmentations
    are stored/updatable.

    Tools:
        boundary_proposal   — Stage 1: propose candidate cut points
        segmentation_decode — Stage 2: DP/Viterbi skill-label assignment
        contract_learning   — Stage 3: learn and verify effect contracts
        bank_maintenance    — Stage 4: refine, materialize, merge, split
    """

    TOOLS: Dict[str, str] = {
        "boundary_proposal": "Propose candidate segment boundaries (Stage 1)",
        "segmentation_decode": "Decode optimal skill sequence via DP (Stage 2)",
        "contract_learning": "Learn and verify skill effect contracts (Stage 3)",
        "bank_maintenance": "Refine, materialize, merge, split skills (Stage 4)",
    }

    def __init__(
        self,
        em_trainer: EMTrainer,
        segmentation_store: SegmentationStore,
        config: Optional[CoEvolutionConfig] = None,
    ):
        self._em = em_trainer
        self._seg_store = segmentation_store
        self._cfg = config or CoEvolutionConfig()

    def list_tools(self) -> Dict[str, str]:
        return dict(self.TOOLS)

    # -- Individual stage tools -------------------------------------------

    def call_boundary_proposal(
        self,
        trajectories: List[TrajectoryForEM],
    ) -> Dict[str, List[CutCandidate]]:
        """Stage 1: propose boundary cuts for each trajectory."""
        cuts_per_traj: Dict[str, List[CutCandidate]] = {}
        for traj in trajectories:
            cuts = propose_cuts(traj, config=self._cfg.propose_cuts)
            cuts_per_traj[traj.traj_id] = cuts
        logger.info(
            "[Tool:boundary_proposal] %d trajectories → %d total cuts",
            len(trajectories),
            sum(len(c) for c in cuts_per_traj.values()),
        )
        return cuts_per_traj

    def call_segmentation_decode(
        self,
        trajectories: List[TrajectoryForEM],
        cuts_per_traj: Dict[str, List[CutCandidate]],
        bank: Any,
    ) -> List[DecodeResult]:
        """Stage 2: decode skill labels for each trajectory's segments."""
        results = decode_batch(trajectories, cuts_per_traj, bank, config=self._cfg.decode)
        total_segs = sum(len(dr.segments) for dr in results)
        total_new = sum(dr.n_new for dr in results)
        logger.info(
            "[Tool:segmentation_decode] %d segments (%d NEW)",
            total_segs, total_new,
        )
        return results

    def call_contract_learning(
        self,
        decode_results: List[DecodeResult],
        bank: Any,
    ) -> ContractLearningResult:
        """Stage 3: learn and verify skill contracts from decoded segments."""
        result = learn_contracts(decode_results, bank, config=self._cfg.contracts)
        logger.info(
            "[Tool:contract_learning] %d contracts learned, mean_pass_rate=%.3f",
            len(result.contracts),
            (
                sum(result.per_skill_pass_rates.values())
                / max(len(result.per_skill_pass_rates), 1)
            ),
        )
        return result

    def call_bank_maintenance(
        self,
        decode_results: List[DecodeResult],
        contracts: Dict,
        bank: Any,
    ) -> UpdateResult:
        """Stage 4: refine, materialize, merge, split."""
        result = run_update(decode_results, contracts, bank, config=self._cfg.update)
        logger.info(
            "[Tool:bank_maintenance] refine=%d materialize=%d merge=%d split=%d",
            result.n_refine, result.n_materialize, result.n_merge, result.n_split,
        )
        return result

    # -- Full pipeline ----------------------------------------------------

    def run_full_pipeline(
        self,
        trajectories: List[TrajectoryForEM],
        bank_store: VersionedBankStore,
        global_step: int = 0,
    ) -> "PipelineResult":
        """Run all four stages sequentially with segmentation persistence.

        Delegates to the EMTrainer for the core loop, then stores the
        resulting segmentations.
        """
        em_result = self._em.run(trajectories)

        if em_result.accepted:
            self._persist_segmentations(
                trajectories, bank_store, em_result, global_step,
            )

        stage_metrics = {}
        if em_result.iterations:
            last_iter = em_result.iterations[-1]
            stage_metrics = {
                "boundary_proposal/n_cuts": last_iter.n_segments,
                "segmentation_decode/n_new": last_iter.n_new,
                "segmentation_decode/new_rate": last_iter.new_rate,
                "segmentation_decode/mean_margin": last_iter.mean_margin,
                "contract_learning/mean_pass_rate": last_iter.mean_pass_rate,
                "bank_maintenance/update_events": last_iter.update_events,
            }

        return PipelineResult(
            accepted=em_result.accepted,
            bank_version=em_result.bank_version,
            n_iterations=len(em_result.iterations),
            em_result=em_result,
            stage_metrics=stage_metrics,
            rejection_reason=em_result.rejection_reason,
        )

    def run_stages_individually(
        self,
        trajectories: List[TrajectoryForEM],
        bank: Any,
        bank_store: VersionedBankStore,
        global_step: int = 0,
    ) -> "PipelineResult":
        """Run each stage as an explicit tool call (for fine-grained control).

        Unlike ``run_full_pipeline`` which delegates to EMTrainer, this
        method calls each stage tool independently, giving callers
        visibility into intermediate results.
        """
        from trainer.skillbank.stages.stage0_predicates import enrich_trajectory_predicates

        for traj in trajectories:
            enrich_trajectory_predicates(
                traj,
                predicate_vocabulary=self._cfg.predicate_vocabulary,
                threshold=self._cfg.p_thresh,
                smoothing_window=self._cfg.smoothing_window,
            )

        cuts_per_traj = self.call_boundary_proposal(trajectories)
        decode_results = self.call_segmentation_decode(trajectories, cuts_per_traj, bank)
        contract_result = self.call_contract_learning(decode_results, bank)

        apply_contracts_to_bank(contract_result, bank, only_verified=True)

        update_result = self.call_bank_maintenance(
            decode_results, contract_result.contracts, bank,
        )

        total_segs = sum(len(dr.segments) for dr in decode_results)
        total_new = sum(dr.n_new for dr in decode_results)

        for traj in trajectories:
            dr_match = next((dr for dr in decode_results if dr.traj_id == traj.traj_id), None)
            if dr_match is None:
                continue
            seg_dicts = [s.to_dict() for s in dr_match.segments]
            self._seg_store.update(
                traj_id=traj.traj_id,
                segments=seg_dicts,
                bank_version=bank_store.version,
                global_step=global_step,
            )

        stage_metrics = {
            "boundary_proposal/n_cuts": sum(len(c) for c in cuts_per_traj.values()),
            "segmentation_decode/n_segments": total_segs,
            "segmentation_decode/n_new": total_new,
            "segmentation_decode/new_rate": total_new / max(total_segs, 1),
            "contract_learning/n_contracts": len(contract_result.contracts),
            "bank_maintenance/n_refine": update_result.n_refine,
            "bank_maintenance/n_materialize": update_result.n_materialize,
            "bank_maintenance/n_merge": update_result.n_merge,
            "bank_maintenance/n_split": update_result.n_split,
        }

        return PipelineResult(
            accepted=True,
            bank_version=bank_store.version,
            n_iterations=1,
            stage_metrics=stage_metrics,
        )

    # -- Helpers ----------------------------------------------------------

    def _persist_segmentations(
        self,
        trajectories: List[TrajectoryForEM],
        bank_store: VersionedBankStore,
        em_result: Any,
        global_step: int,
    ) -> None:
        """Store segmentation results from the last EM decode pass."""
        last_decode = self._em.last_decode_results
        if last_decode is None:
            last_decode = self._em._decode_all(
                trajectories, bank_store.current_bank,
            )
        for dr in last_decode:
            seg_dicts = [s.to_dict() for s in dr.segments]
            tool_calls = self._extract_tool_calls_from_traj(
                dr.traj_id, trajectories,
            )
            self._seg_store.update(
                traj_id=dr.traj_id,
                segments=seg_dicts,
                bank_version=bank_store.version,
                global_step=global_step,
                tool_calls=tool_calls,
            )

    @staticmethod
    def _extract_tool_calls_from_traj(
        traj_id: str,
        trajectories: List[TrajectoryForEM],
    ) -> List[Dict[str, Any]]:
        """Pull tool call records from a trajectory's frames."""
        for traj in trajectories:
            if traj.traj_id != traj_id:
                continue
            calls = []
            for frame in traj.frames:
                if frame.action_type in ("QUERY_SKILL", "QUERY_MEM", "CALL_SKILL"):
                    calls.append({
                        "t": frame.t,
                        "action_type": frame.action_type,
                        "action": frame.action,
                        "active_skill_id": frame.active_skill_id,
                    })
            return calls
        return []


@dataclass
class PipelineResult:
    """Aggregated result from running the skill agent tool pipeline."""

    accepted: bool = True
    bank_version: int = 0
    n_iterations: int = 0
    em_result: Any = None
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "bank_version": self.bank_version,
            "n_iterations": self.n_iterations,
            "rejection_reason": self.rejection_reason,
            **self.stage_metrics,
        }


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------

class SkillBankCoEvolutionCallback:
    """VERL training callback for SkillBank co-evolution.

    Called by ``GameAITrainer.fit()`` after each training step.  At the
    configured cadence:

    1. Extracts rollout records from the VERL batch.
    2. Converts them to ``TrajectoryForEM`` objects.
    3. Runs the full skill agent tool pipeline (Stages 1-4) via EMTrainer.
    4. Persists trajectory segmentations in a ``SegmentationStore``.
    5. Hot-swaps the updated bank into environment workers.
    6. Optionally computes tool-call reward metrics.
    7. Returns metrics dict for VERL logging.
    """

    def __init__(
        self,
        config: CoEvolutionConfig,
        initial_bank: Any,
        envs: Any,
        val_envs: Optional[Any] = None,
    ):
        self._config = config
        self._envs = envs
        self._val_envs = val_envs

        self._bank_store = VersionedBankStore(
            bank=initial_bank,
            bank_dir=config.bank_dir,
        )

        self._seg_store = SegmentationStore(
            path=config.segmentation_store_path,
        )
        self._seg_store.load()

        em_config = EMConfig(
            max_iterations=config.em_max_iterations,
            convergence_new_rate=config.max_new_rate,
            propose_cuts=config.propose_cuts,
            decode=config.decode,
            contracts=config.contracts,
            update=config.update,
            predicate_vocabulary=config.predicate_vocabulary,
            p_thresh=config.p_thresh,
            smoothing_window=config.smoothing_window,
        )

        diff_logger = DiffLogger(
            diff_dir=str(Path(config.bank_dir) / "diffs"),
        )

        self._em_trainer = EMTrainer(
            bank_store=self._bank_store,
            config=em_config,
            diff_logger=diff_logger,
        )

        self._pipeline = SkillAgentToolPipeline(
            em_trainer=self._em_trainer,
            segmentation_store=self._seg_store,
            config=config,
        )

        self._rollout_adapter = None
        self._tool_reward_config = None
        if config.enable_tool_call_reward:
            try:
                from skill_agents.tool_call_reward import ToolCallRewardConfig
                self._tool_reward_config = ToolCallRewardConfig()
            except ImportError:
                logger.warning("skill_agents.tool_call_reward not available")

        self._update_count = 0

    # -- Public API -------------------------------------------------------

    def on_step_end(
        self,
        global_step: int,
        batch: Any,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Called after each VERL training step.

        Returns a dict of metrics to merge into the training log.
        """
        if global_step % self._config.bank_update_cadence != 0:
            return {}

        records = self._extract_rollouts(batch)
        if not records:
            logger.debug("Co-evolution step %d: no rollouts to process", global_step)
            return {"coevo/skipped": 1}

        trajectories = ingest_rollouts(
            records,
            max_trajectories=self._config.max_trajectories_per_update,
        )
        if not trajectories:
            return {"coevo/skipped": 1}

        result = self._pipeline.run_full_pipeline(
            trajectories,
            bank_store=self._bank_store,
            global_step=global_step,
        )

        if result.accepted:
            self._hot_swap_bank()
            self._update_count += 1
            self._seg_store.save()

        tool_metrics = {}
        if self._config.enable_tool_call_reward and self._tool_reward_config:
            tool_metrics = self._compute_tool_call_metrics(records)

        coevo_metrics = self._build_metrics(result, global_step)
        coevo_metrics.update(tool_metrics)
        return coevo_metrics

    @property
    def bank_store(self) -> VersionedBankStore:
        return self._bank_store

    @property
    def segmentation_store(self) -> SegmentationStore:
        return self._seg_store

    @property
    def pipeline(self) -> SkillAgentToolPipeline:
        return self._pipeline

    @property
    def current_bank(self) -> Any:
        return self._bank_store.current_bank

    # -- Internal ---------------------------------------------------------

    def _extract_rollouts(self, batch: Any) -> List[RolloutRecord]:
        """Convert VERL DataProto batch to RolloutRecords."""
        if self._rollout_adapter is None:
            try:
                from trainer.decision.rollout_collector import VERLRolloutAdapter
                self._rollout_adapter = VERLRolloutAdapter()
            except ImportError:
                return []

        try:
            return self._rollout_adapter.extract_records(batch)
        except Exception as exc:
            logger.warning("Failed to extract rollouts: %s", exc)
            return []

    def _hot_swap_bank(self) -> None:
        """Push updated bank into all environment workers."""
        new_bank = self._bank_store.current_bank
        if hasattr(self._envs, "update_skill_bank"):
            self._envs.update_skill_bank(new_bank)
        if self._val_envs is not None and hasattr(self._val_envs, "update_skill_bank"):
            self._val_envs.update_skill_bank(new_bank)
        logger.info(
            "Hot-swapped bank v%d (%d skills) into env workers",
            self._bank_store.version,
            len(getattr(new_bank, "skill_ids", [])),
        )

    def _compute_tool_call_metrics(
        self, records: List[RolloutRecord],
    ) -> Dict[str, float]:
        """Compute aggregate tool-call reward metrics over rollouts."""
        try:
            from skill_agents.tool_call_reward import (
                compute_tool_call_reward,
                ToolCallRewardConfig,
            )
        except ImportError:
            return {}

        cfg = self._tool_reward_config or ToolCallRewardConfig()
        bank = self._bank_store.current_bank
        rewards_all = []
        relevance_all = []
        utility_all = []
        n_tool_calls = 0

        for record in records:
            for step in record.steps:
                if step.action_type not in ("QUERY_SKILL", "QUERY_MEM", "CALL_SKILL"):
                    continue

                tool_name = {
                    "QUERY_SKILL": "query_skill",
                    "QUERY_MEM": "query_memory",
                    "CALL_SKILL": "call_skill",
                }.get(step.action_type, "take_action")

                rr = compute_tool_call_reward(
                    tool_name=tool_name,
                    tool_args={"key": step.query_key or ""},
                    context_observation=step.obs_id,
                    skill_bank=bank,
                    retrieved_skill_id=step.active_skill_id,
                    config=cfg,
                )
                rewards_all.append(rr.r_total)
                relevance_all.append(rr.r_relevance)
                utility_all.append(rr.r_utility)
                n_tool_calls += 1

        if not rewards_all:
            return {"coevo/tool_call_count": 0}

        return {
            "coevo/tool_call_count": n_tool_calls,
            "coevo/tool_call_reward_mean": sum(rewards_all) / len(rewards_all),
            "coevo/tool_call_relevance_mean": sum(relevance_all) / len(relevance_all),
            "coevo/tool_call_utility_mean": sum(utility_all) / len(utility_all),
        }

    def _build_metrics(
        self, result: PipelineResult, global_step: int,
    ) -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "coevo/accepted": int(result.accepted),
            "coevo/bank_version": result.bank_version,
            "coevo/n_iterations": result.n_iterations,
            "coevo/n_segmentations_stored": len(self._seg_store),
            "coevo/update_count": self._update_count,
            "coevo/global_step": global_step,
        }
        if result.rejection_reason:
            m["coevo/rejection_reason"] = result.rejection_reason

        for key, val in result.stage_metrics.items():
            m[f"coevo/{key}"] = val

        bank = self._bank_store.current_bank
        n_skills = len(getattr(bank, "skill_ids", []))
        m["coevo/n_skills"] = n_skills
        return m
