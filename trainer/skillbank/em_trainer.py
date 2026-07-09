"""
Hard-EM loop driver for the SkillBank Agent.

Iterates: propose cuts → decode → contracts → update → SkillEval gating.
Manages transactional bank updates with commit/rollback semantics.

Usage::

    from trainer.skillbank.em_trainer import EMTrainer, EMConfig

    trainer = EMTrainer(bank_store=bank_store, config=cfg)
    result = trainer.run(trajectories)
    # result.accepted, result.bank_version, result.diff_report
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.skillbank.bank_io.bank_store import VersionedBankStore
from trainer.skillbank.bank_io.diff_logger import BankDiffReport, DiffLogger, compute_bank_diff
from trainer.skillbank.bank_io.indices import KeywordIndex
from trainer.skillbank.ingest_rollouts import TrajectoryForEM, split_holdout
from trainer.skillbank.stages.skilleval import (
    SkillEvalConfig,
    SkillEvalResult,
    evaluate_bank_update,
)
from trainer.skillbank.stages.stage0_predicates import enrich_trajectory_predicates
from trainer.skillbank.stages.stage1_propose_cuts import (
    CutCandidate,
    ProposeCutsConfig,
    propose_cuts,
)
from trainer.skillbank.stages.stage2_decode import (
    DecodeConfig,
    DecodeResult,
    decode_batch,
    decode_trajectory,
)
from trainer.skillbank.stages.stage3_contracts import (
    ContractLearningConfig,
    ContractLearningResult,
    apply_contracts_to_bank,
    learn_contracts,
)
from trainer.skillbank.stages.stage4_update import UpdateConfig, UpdateResult, run_update

logger = logging.getLogger(__name__)


@dataclass
class EMConfig:
    """Configuration for the Hard-EM trainer."""

    max_iterations: int = 3
    convergence_new_rate: float = 0.05
    convergence_margin_std: float = 0.5
    holdout_fraction: float = 0.1
    local_redecode: bool = True

    propose_cuts: Optional[ProposeCutsConfig] = None
    decode: Optional[DecodeConfig] = None
    contracts: Optional[ContractLearningConfig] = None
    update: Optional[UpdateConfig] = None
    skilleval: Optional[SkillEvalConfig] = None

    predicate_vocabulary: Optional[List[str]] = None
    p_thresh: float = 0.7
    smoothing_window: int = 3


@dataclass
class EMIterationResult:
    """Result of one EM iteration."""

    iteration: int = 0
    n_segments: int = 0
    n_new: int = 0
    new_rate: float = 0.0
    mean_margin: float = 0.0
    mean_pass_rate: float = 0.0
    update_events: int = 0
    skilleval_accepted: bool = True


@dataclass
class EMRunResult:
    """Result of a complete EM run (possibly multiple iterations)."""

    accepted: bool = True
    bank_version: int = 0
    iterations: List[EMIterationResult] = field(default_factory=list)
    diff_report: Optional[BankDiffReport] = None
    skilleval: Optional[SkillEvalResult] = None
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "bank_version": self.bank_version,
            "n_iterations": len(self.iterations),
            "rejection_reason": self.rejection_reason,
        }


class EMTrainer:
    """Hard-EM trainer for the SkillBank Agent.

    Orchestrates the full EM loop:
      1. Enrich trajectories with predicates (Stage 0)
      2. For each iteration:
         a. Propose cuts (Stage 1)
         b. Decode segments (Stage 2)
         c. Learn contracts (Stage 3)
         d. Update bank (Stage 4: refine, materialize, merge, split)
      3. SkillEval gating on holdout
      4. Commit or rollback

    Supports an optional ``segmentation_store`` so that per-trajectory
    segmentations are persisted and can be updated across training steps.

    Args:
        bank_store: VersionedBankStore for transactional updates
        config: EMConfig with all hyperparameters
        diff_logger: DiffLogger for recording bank changes
        segmentation_store: optional store for persistent segment tracking
    """

    def __init__(
        self,
        bank_store: VersionedBankStore,
        config: Optional[EMConfig] = None,
        diff_logger: Optional[DiffLogger] = None,
        segmentation_store: Any = None,
    ):
        self.store = bank_store
        self.cfg = config or EMConfig()
        self.diff_logger = diff_logger or DiffLogger()
        self._index = KeywordIndex()
        self._seg_store = segmentation_store

    def run(self, trajectories: List[TrajectoryForEM]) -> EMRunResult:
        """Execute the full EM pipeline on a batch of trajectories.

        Args:
            trajectories: list of TrajectoryForEM objects (from ingest_rollouts)

        Returns:
            EMRunResult with accept/reject, bank version, and diagnostics.
        """
        if not trajectories:
            return EMRunResult(accepted=False, rejection_reason="no trajectories")

        train_trajs, holdout_trajs = split_holdout(
            trajectories, self.cfg.holdout_fraction
        )

        for traj in trajectories:
            enrich_trajectory_predicates(
                traj,
                predicate_vocabulary=self.cfg.predicate_vocabulary,
                threshold=self.cfg.p_thresh,
                smoothing_window=self.cfg.smoothing_window,
            )

        old_bank = copy.deepcopy(self.store.current_bank)
        candidate = self.store.fork()

        result = EMRunResult(bank_version=self.store.version)
        prev_metrics: Dict[str, float] = {}

        for iteration in range(1, self.cfg.max_iterations + 1):
            iter_result = self._run_iteration(
                train_trajs, candidate, iteration
            )
            result.iterations.append(iter_result)

            if iter_result.new_rate <= self.cfg.convergence_new_rate:
                logger.info("EM converged at iteration %d (new_rate=%.3f)",
                            iteration, iter_result.new_rate)
                break

        last_decode = self._decode_all(train_trajs, candidate)
        self._last_decode = last_decode
        last_contracts = learn_contracts(
            last_decode, candidate, self.cfg.contracts
        )

        eval_result = evaluate_bank_update(
            decode_results=last_decode,
            contract_result=last_contracts,
            bank=candidate,
            config=self.cfg.skilleval,
            prev_metrics=prev_metrics,
        )
        result.skilleval = eval_result

        if eval_result.accepted:
            apply_contracts_to_bank(last_contracts, candidate)
            diff = compute_bank_diff(
                old_bank, candidate,
                old_version=self.store.version,
                new_version=self.store.version + 1,
            )
            result.diff_report = diff
            self.diff_logger.log_diff(diff)

            new_version = self.store.commit(rebuild_indices=True)
            result.bank_version = new_version
            result.accepted = True
            self._index.build(self.store.current_bank)

            self._update_segmentation_store(last_decode, new_version)

            logger.info("EM accepted → bank v%d (%d skills)",
                        new_version, len(getattr(self.store.current_bank, "skill_ids", [])))
        else:
            self.store.rollback()
            result.accepted = False
            result.rejection_reason = "; ".join(eval_result.rejection_reasons)
            logger.warning("EM rejected: %s", result.rejection_reason)

        return result

    def _update_segmentation_store(
        self, decode_results: List[DecodeResult], bank_version: int,
    ) -> None:
        """Persist decoded segmentations so they can be updated later."""
        if self._seg_store is None:
            return
        for dr in decode_results:
            seg_dicts = [s.to_dict() for s in dr.segments]
            self._seg_store.update(
                traj_id=dr.traj_id,
                segments=seg_dicts,
                bank_version=bank_version,
            )

    @property
    def last_decode_results(self) -> Optional[List[DecodeResult]]:
        return getattr(self, "_last_decode", None)

    def _run_iteration(
        self,
        trajectories: List[TrajectoryForEM],
        bank: Any,
        iteration: int,
    ) -> EMIterationResult:
        """Run one EM iteration: propose → decode → contract → update."""
        cuts_per_traj: Dict[str, List[CutCandidate]] = {}
        for traj in trajectories:
            cuts = propose_cuts(traj, config=self.cfg.propose_cuts)
            cuts_per_traj[traj.traj_id] = cuts

        decode_results = decode_batch(
            trajectories, cuts_per_traj, bank, config=self.cfg.decode
        )

        contract_result = learn_contracts(decode_results, bank, config=self.cfg.contracts)

        apply_contracts_to_bank(contract_result, bank, only_verified=True)

        update_result = run_update(
            decode_results, contract_result.contracts, bank, config=self.cfg.update
        )

        total_segs = sum(len(dr.segments) for dr in decode_results)
        total_new = sum(dr.n_new for dr in decode_results)
        all_margins = [
            seg.margin for dr in decode_results for seg in dr.segments
        ]
        mean_margin = sum(all_margins) / len(all_margins) if all_margins else 0.0
        pass_rates = list(contract_result.per_skill_pass_rates.values())
        mean_pr = sum(pass_rates) / len(pass_rates) if pass_rates else 1.0

        return EMIterationResult(
            iteration=iteration,
            n_segments=total_segs,
            n_new=total_new,
            new_rate=total_new / max(total_segs, 1),
            mean_margin=mean_margin,
            mean_pass_rate=mean_pr,
            update_events=(
                update_result.n_refine + update_result.n_materialize
                + update_result.n_merge + update_result.n_split
            ),
            skilleval_accepted=True,
        )

    def _decode_all(
        self,
        trajectories: List[TrajectoryForEM],
        bank: Any,
    ) -> List[DecodeResult]:
        """Decode all trajectories with the current bank."""
        cuts_per_traj: Dict[str, List[CutCandidate]] = {}
        for traj in trajectories:
            cuts = propose_cuts(traj, config=self.cfg.propose_cuts)
            cuts_per_traj[traj.traj_id] = cuts
        return decode_batch(trajectories, cuts_per_traj, bank, config=self.cfg.decode)
