"""
GRPO orchestrator: top-level entry point for the two-phase training loop.

Phase 1 (rollout): enables GRPO wrappers on all targeted LLM calls,
runs the EM pipeline, and collects samples in the buffer.

Phase 2 (training): runs ``GRPOLoRATrainer.train_step()`` on the
accumulated buffer, then clears it.

Usage::

    from skill_agents.grpo.orchestrator import GRPOOrchestrator

    orch = GRPOOrchestrator(llm=llm, config=grpo_config)
    orch.enable_wrappers(...)
    run_em_pipeline(...)
    stats = orch.train_step()
    orch.disable_wrappers()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from skill_agents.grpo.buffer import GRPOBuffer
from skill_agents.grpo.config import GRPOConfig
from skill_agents.grpo.trainer import GRPOLoRATrainer
from skill_agents.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


class GRPOOrchestrator:
    """Coordinates GRPO wrapping and training across all stages.

    Parameters
    ----------
    llm : MultiLoraSkillBankLLM
    config : GRPOConfig
    """

    def __init__(
        self,
        llm: Any,
        config: Optional[GRPOConfig] = None,
        io_log_dir: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._config = config or GRPOConfig()
        self._buffer = GRPOBuffer()
        self._trainer = GRPOLoRATrainer(llm, self._config, io_log_dir=io_log_dir)
        self._wrappers_enabled = False

    @property
    def buffer(self) -> GRPOBuffer:
        return self._buffer

    def enable_wrappers(
        self,
        *,
        # Stage 3 CONTRACT
        contract_consensus_add: Optional[Set[str]] = None,
        contract_consensus_del: Optional[Set[str]] = None,
        contract_holdout_instances: Optional[list] = None,
        contract_verify_config: Optional[Any] = None,
        # Stage 4 CURATOR
        curator_compute_quality_fn: Optional[Callable] = None,
        curator_execute_fn: Optional[Callable] = None,
        # Stage 2 SEGMENT
        segment_scorer_factory: Optional[Callable] = None,
        segment_decode_fn: Optional[Callable] = None,
    ) -> None:
        """Enable GRPO wrappers on all configured stages."""
        if self._wrappers_enabled:
            logger.warning("Wrappers already enabled")
            return

        # Stage 3 — CONTRACT
        if self._config.is_enabled(SkillFunction.CONTRACT):
            from skill_agents.stage3_mvp.llm_contract import enable_contract_grpo
            cfg = self._config.for_stage(SkillFunction.CONTRACT)
            enable_contract_grpo(
                buffer=self._buffer,
                group_size=cfg.group_size,
                temperature=cfg.temperature,
                consensus_add=contract_consensus_add,
                consensus_del=contract_consensus_del,
                holdout_instances=contract_holdout_instances,
                verify_config=contract_verify_config,
            )

        # Stage 4 — CURATOR
        if self._config.is_enabled(SkillFunction.CURATOR):
            from skill_agents.bank_maintenance.llm_curator import enable_curator_grpo
            cfg = self._config.for_stage(SkillFunction.CURATOR)
            enable_curator_grpo(
                buffer=self._buffer,
                group_size=cfg.group_size,
                temperature=cfg.temperature,
                compute_quality_fn=curator_compute_quality_fn,
                execute_fn=curator_execute_fn,
            )

        # Stage 2 — SEGMENT
        if self._config.is_enabled(SkillFunction.SEGMENT):
            from skill_agents.infer_segmentation.llm_teacher import enable_segment_grpo
            cfg = self._config.for_stage(SkillFunction.SEGMENT)
            enable_segment_grpo(
                buffer=self._buffer,
                group_size=cfg.group_size,
                temperature=cfg.temperature,
                scorer_factory=segment_scorer_factory,
                decode_fn=segment_decode_fn,
            )

        self._wrappers_enabled = True
        logger.info("GRPO wrappers enabled: %s", self._buffer)

    def disable_wrappers(self) -> None:
        """Disable all GRPO wrappers, restoring original functions."""
        from skill_agents.stage3_mvp.llm_contract import disable_contract_grpo
        from skill_agents.bank_maintenance.llm_curator import disable_curator_grpo
        from skill_agents.infer_segmentation.llm_teacher import disable_segment_grpo

        disable_contract_grpo()
        disable_curator_grpo()
        disable_segment_grpo()

        self._wrappers_enabled = False
        logger.info("GRPO wrappers disabled")

    def train_step(self) -> Dict[str, Any]:
        """Phase 2: train all adapters from the buffer, then clear it.

        Returns per-adapter training stats.
        """
        logger.info("GRPO train_step: buffer has %s", self._buffer)
        stats = self._trainer.train_step(self._buffer)
        self._buffer.clear()
        return stats

    def status(self) -> Dict[str, Any]:
        return {
            "wrappers_enabled": self._wrappers_enabled,
            "buffer": str(self._buffer),
            "buffer_size": self._buffer.size(),
            "adapters_with_data": [
                a.value for a in self._buffer.adapters_with_data()
            ],
        }
