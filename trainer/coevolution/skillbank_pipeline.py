"""Async skill bank pipeline wrapper for the co-evolution loop.

Wraps the synchronous ``SkillBankAgent`` pipeline (Stage 1+2 segmentation,
Stage 3 contract learning, Stage 4 bank maintenance) to run concurrently
with rollout collection.  Uses ``asyncio.Queue`` to receive completed
episodes and processes them in micro-batches through the pipeline stages.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from trainer.coevolution.episode_runner import EpisodeResult

logger = logging.getLogger(__name__)


def _candy_crush_ablation_modes(game_name: str) -> Dict[str, str]:
    """Return contract/segmentation ablation modes for Candy Crush only."""
    if game_name != "candy_crush":
        return {
            "effect_contract_mode": "consensus_verified",
            "segmentation_mode": "learned_decode",
        }

    effect_mode = os.environ.get(
        "COSPLAY_CC_EFFECT_CONTRACT_MODE",
        "consensus_verified",
    )
    segmentation_mode = os.environ.get(
        "COSPLAY_CC_SEGMENTATION_MODE",
        "learned_decode",
    )

    effect_aliases = {
        "consensus_verified": "consensus_verified",
        "full": "consensus_verified",
        "none": "none",
        "off": "none",
        "raw_delta": "raw_delta",
    }
    segmentation_aliases = {
        "learned_decode": "learned_decode",
        "full": "learned_decode",
        "heuristic_only": "heuristic_only",
    }

    if effect_mode not in effect_aliases:
        raise ValueError(f"Unknown COSPLAY_CC_EFFECT_CONTRACT_MODE={effect_mode!r}")
    if segmentation_mode not in segmentation_aliases:
        raise ValueError(f"Unknown COSPLAY_CC_SEGMENTATION_MODE={segmentation_mode!r}")

    return {
        "effect_contract_mode": effect_aliases[effect_mode],
        "segmentation_mode": segmentation_aliases[segmentation_mode],
    }


def _manager_ablation_modes(games: Iterable[str]) -> Dict[str, str]:
    """Resolve module-level wrapper modes for a manager's active games."""
    game_set = set(games)
    if "candy_crush" in game_set:
        return _candy_crush_ablation_modes("candy_crush")
    return _candy_crush_ablation_modes("generic")


def _strip_effect_contracts_for_ablation(agent: Any) -> int:
    """Remove predicate contracts while preserving skills/protocols."""
    bank = getattr(agent, "bank", None)
    if bank is None:
        return 0
    removed = 0
    for sid in list(getattr(bank, "skill_ids", [])):
        skill = bank.get_skill(sid) if hasattr(bank, "get_skill") else None
        if skill is not None and getattr(skill, "contract", None) is not None:
            skill.contract = None
            removed += 1
    reports = getattr(bank, "_reports", None)
    if isinstance(reports, dict):
        reports.clear()
    return removed


@dataclass
class SkillBankUpdateResult:
    accepted: bool = False
    bank_version: int = 0
    n_skills: int = 0
    n_new_skills: int = 0
    n_episodes_processed: int = 0
    wall_time_s: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    grpo_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class AsyncSkillBankPipeline:
    """Manages the skill bank update lifecycle across a co-evolution step.

    Receives completed episodes (via ``ingest_episode()`` or
    ``process_batch_async()``), processes them through the SkillBankAgent
    pipeline, and produces an updated bank.
    """

    def __init__(
        self,
        bank_dir: str = "runs/skillbank",
        model_name: str = "Qwen/Qwen3.5-9B",
        executor: Optional[ThreadPoolExecutor] = None,
        report_dir: Optional[str] = None,
        game_name: str = "generic",
    ):
        self.bank_dir = bank_dir
        self.model_name = model_name
        self.game_name = game_name
        self._executor = executor
        self.report_dir = report_dir or str(Path(bank_dir) / "reports")
        self._agent: Any = None
        self._query_engine: Any = None
        self._pending_episodes: List[Any] = []
        self._grpo_data: Dict[str, List[Dict[str, Any]]] = {
            "segment": [],
            "contract": [],
            "curator": [],
        }
        self._update_result: Optional[SkillBankUpdateResult] = None

    def _ensure_agent(self) -> Any:
        """Lazily create the SkillBankAgent."""
        if self._agent is not None:
            return self._agent

        from skill_agents.pipeline import SkillBankAgent, PipelineConfig

        bank_path = str(Path(self.bank_dir) / "skill_bank.jsonl")
        _teacher_max_tokens_env = os.environ.get("SKILLBANK_LLM_TEACHER_MAX_TOKENS")
        ablation_modes = _candy_crush_ablation_modes(self.game_name)
        config = PipelineConfig(
            bank_path=bank_path,
            env_name="llm",
            game_name=self.game_name,
            llm_model=self.model_name,
            extractor_model=self.model_name,
            segmentation_method="dp",
            segmentation_mode=ablation_modes["segmentation_mode"],
            effect_contract_mode=ablation_modes["effect_contract_mode"],
            preference_iterations=1,
            new_skill_penalty=2.0,
            eff_freq=0.5,
            min_instances_per_skill=1,
            start_end_window=3,
            new_pool_min_cluster_size=1,
            new_pool_min_consistency=0.3,
            new_pool_min_distinctiveness=0.15,
            min_new_cluster_size=1,
            report_dir=self.report_dir,
            max_concurrent_llm_calls=24,
            llm_teacher_max_workers=4,
            **({"llm_teacher_max_tokens": int(_teacher_max_tokens_env)}
               if _teacher_max_tokens_env is not None else {}),
        )
        self._agent = SkillBankAgent(config=config)

        if Path(bank_path).exists():
            try:
                self._agent.load()
                if ablation_modes["effect_contract_mode"] == "none":
                    removed = _strip_effect_contracts_for_ablation(self._agent)
                    if removed:
                        logger.info(
                            "Candy Crush no-effect-contract ablation: "
                            "stripped %d loaded contract(s)",
                            removed,
                        )
                n = len(self._agent.skill_ids)
                logger.info("Loaded existing skill bank: %d skills", n)
            except Exception as exc:
                logger.warning("Failed to load skill bank: %s", exc)

        return self._agent

    def load_bank(self, bank: Any) -> None:
        """Inject a pre-loaded bank into the pipeline agent."""
        agent = self._ensure_agent()
        if hasattr(agent, "bank") and bank is not None:
            agent.bank = bank

    def reload_bank_from_disk(self) -> Dict[str, Any]:
        """Refresh ``self._agent.bank`` from the on-disk
        ``<bank_dir>/skill_bank.jsonl`` *and* clear every cached
        ``SkillQueryEngine`` so the next ``get_bank()`` call returns an
        index that observes the new on-disk state.

        Required after the Phase B′ Crafter+Promotion writeback (see
        ``skill_bank.legacy_writeback``) — without this call the live
        actor would still see the pre-writeback bank because
        ``SkillBankMVP._skills`` is held in memory and
        ``SkillQueryEngine._build_index()`` snapshots ``skill_ids`` at
        construction time (it never re-indexes).

        Returns
        -------
        dict
            ``{"reloaded": bool, "n_before": int, "n_after": int,
               "bank_path": str, "agent_initialised": bool,
               "query_engine_invalidated": bool}``
            — useful for the orchestrator's step_log.
        """
        bank_path = str(Path(self.bank_dir) / "skill_bank.jsonl")
        n_before = 0
        agent_initialised = self._agent is not None
        if not agent_initialised:
            return {
                "reloaded": False,
                "n_before": 0, "n_after": 0,
                "bank_path": bank_path,
                "agent_initialised": False,
                "query_engine_invalidated": False,
                "skipped_reason": "agent not yet initialised — disk file is the source of truth",
            }

        try:
            n_before = len(self._agent.bank)
        except Exception:                                            # noqa: BLE001
            n_before = 0

        # Path may not exist on a brand-new run — that's fine, treat as
        # "no skills" rather than raising. The actor will see an empty
        # bank, which is the correct cold-start behaviour.
        path_obj = Path(bank_path)
        if not path_obj.is_file():
            return {
                "reloaded": False,
                "n_before": n_before, "n_after": n_before,
                "bank_path": bank_path,
                "agent_initialised": True,
                "query_engine_invalidated": False,
                "skipped_reason": "bank file missing on disk",
            }

        try:
            self._agent.load()
        except Exception as exc:                                     # noqa: BLE001
            logger.warning(
                "AsyncSkillBankPipeline.reload_bank_from_disk: "
                "agent.load() failed for %s: %s",
                bank_path, exc,
            )
            return {
                "reloaded": False,
                "n_before": n_before, "n_after": n_before,
                "bank_path": bank_path,
                "agent_initialised": True,
                "query_engine_invalidated": False,
                "skipped_reason": f"agent.load() failed: {exc}",
            }

        # Invalidate BOTH query-engine caches:
        #   * pipeline-level: returned by AsyncSkillBankPipeline.get_bank()
        #     — this is what the actor's rollout loop reads from.
        #   * agent-level: used internally by the segmentation pipeline
        #     (see SkillBankAgent._get_query_engine, line ~225); calling
        #     the existing _invalidate_query_engine() is the documented way.
        self._query_engine = None
        invalidated_agent_engine = False
        try:
            self._agent._invalidate_query_engine()
            invalidated_agent_engine = True
        except Exception as exc:                                     # noqa: BLE001
            logger.debug(
                "agent._invalidate_query_engine() failed (non-fatal): %s", exc,
            )

        try:
            n_after = len(self._agent.bank)
        except Exception:                                            # noqa: BLE001
            n_after = n_before

        return {
            "reloaded": True,
            "n_before": n_before, "n_after": n_after,
            "bank_path": bank_path,
            "agent_initialised": True,
            "query_engine_invalidated": invalidated_agent_engine,
        }

    def _convert_episode_result(self, result: EpisodeResult) -> Any:
        """Convert ``EpisodeResult`` to the ``Episode`` format for the pipeline.

        When role/side/stage metadata is present (unified_role_rollouts mode),
        it is forwarded into each ``Experience.interface`` dict and into the
        episode ``metadata`` so the skill bank can segment skills along
        those dimensions.
        """
        from data_structure.experience import Experience, Episode

        has_role_meta = bool(result.role)

        experiences = []
        for exp_dict in result.experiences:
            exp = Experience(
                state=exp_dict.get("state", ""),
                action=exp_dict.get("action", ""),
                reward=exp_dict.get("reward", 0.0),
                next_state=exp_dict.get("next_state", ""),
                done=exp_dict.get("done", False),
                intentions=exp_dict.get("intention"),
            )
            exp.idx = exp_dict.get("step", 0)
            exp.summary_state = exp_dict.get("summary_state", "")
            exp.action_type = "primitive"
            iface = {"env_name": "gamingagent", "game_name": result.game}
            if has_role_meta:
                iface["role"] = exp_dict.get("role", result.role)
                iface["side"] = exp_dict.get("side", result.side)
                iface["stage"] = exp_dict.get("stage", "")
            exp.interface = iface
            experiences.append(exp)

        meta: Dict[str, Any] = {
            "done": result.terminated or result.truncated,
            "steps": result.steps,
            "total_reward": result.total_reward,
        }
        if has_role_meta:
            meta["role"] = result.role
            meta["side"] = result.side
            meta["role_index"] = result.role_index

        episode = Episode(
            experiences=experiences,
            task=f"Play {result.game}",
            env_name="gamingagent",
            game_name=result.game,
            episode_id=result.episode_id,
            metadata=meta,
        )
        episode.set_outcome()
        return episode

    async def ingest_episode(self, result: EpisodeResult) -> None:
        """Convert and queue a completed episode for processing."""
        if result.steps == 0:
            return
        episode = self._convert_episode_result(result)
        self._pending_episodes.append(episode)

    _MAX_CONCURRENT_SEGMENTATIONS = int(
        os.environ.get("SKILLBANK_MAX_CONCURRENT_SEGMENTATIONS", "8")
    )
    # Per-episode segmentation timeout — *dynamic* (v11, 2026-05-04 PM).
    #
    # Original v4 fix used a static 180 s ceiling (lowered from 600 s)
    # with the assumption that ``SKILLBANK_MAX_SKILL_NAMES`` (in
    # ``skill_agents.pipeline.segment_episode``) bounds per-call LLM
    # cost.  That holds for short episodes (TF3 / AlteredBeast: 22-step
    # mean → ~70 s segmentation) but breaks down for long-episode
    # genres: v11 Phase-3 Columns has 130-step mean episodes and the
    # static 180 s ceiling caused **0/8 episodes segmented** for 5 of
    # the last 8 steps, starving the bank of evidence-driven skills
    # and silently degrading the whole skill loop.
    #
    # New policy: ``timeout = base + per_step * n_steps`` (clamped to
    # a configurable absolute ceiling to preserve the v4 zombie-thread
    # safety net).  The defaults below sit just above measured cost on
    # Columns (~3 s/step amortised) with ~50% headroom.  Both knobs
    # are env-tunable so an operator can either lift the ceiling for
    # very long episodes (e.g. WebShop) or tighten it for fast
    # iteration on short games.
    _SEGMENT_TIMEOUT_BASE_S = int(
        os.environ.get("SKILLBANK_SEGMENT_TIMEOUT_BASE_S", "60")
    )
    _SEGMENT_TIMEOUT_PER_STEP_S = float(
        os.environ.get("SKILLBANK_SEGMENT_TIMEOUT_PER_STEP_S", "5.0")
    )
    _SEGMENT_TIMEOUT_MAX_S = int(
        os.environ.get(
            "SKILLBANK_SEGMENT_TIMEOUT_MAX_S",
            os.environ.get("SKILLBANK_SEGMENT_TIMEOUT_S", "900"),
        )
    )

    # Legacy single-knob alias kept for any caller that introspects the
    # class attribute directly (no in-tree callers do, but external
    # eval scripts might).  Reflects the *ceiling*, not the dynamic
    # per-episode budget.
    _SEGMENT_TIMEOUT_S = _SEGMENT_TIMEOUT_MAX_S

    @classmethod
    def _segment_timeout_for(cls, ep: Any) -> float:
        """Compute the per-episode segmentation timeout (seconds).

        Scales linearly with the number of recorded experiences in the
        episode, capped at ``_SEGMENT_TIMEOUT_MAX_S``.  Falls back to
        the base budget if ``ep`` doesn't expose ``experiences``.
        """
        try:
            n_steps = len(getattr(ep, "experiences", ()) or ())
        except Exception:  # noqa: BLE001
            n_steps = 0
        budget = cls._SEGMENT_TIMEOUT_BASE_S + cls._SEGMENT_TIMEOUT_PER_STEP_S * n_steps
        return float(min(cls._SEGMENT_TIMEOUT_MAX_S, max(cls._SEGMENT_TIMEOUT_BASE_S, budget)))

    async def process_batch_async(
        self,
        results: List[EpisodeResult],
    ) -> None:
        """Process a micro-batch of completed episodes through Stages 1+2.

        Segments episodes concurrently via the thread executor (each
        segmentation involves LLM calls, so parallelism overlaps the
        network I/O).  A semaphore limits concurrent segmentations to
        avoid saturating vLLM with hundreds of simultaneous requests.

        Timeout discipline (added 2026-05-04, v4 post-mortem):

        * Each episode's segmentation runs as a future on the shared
          thread executor.  We track that future explicitly so a
          timeout can call ``.cancel()`` on it AND wait briefly for
          the underlying thread to release its LLM connections,
          rather than leaving a zombie that still hammers vLLM after
          the asyncio side has moved on.
        * Episodes whose segmentation timed out (or raised) are
          *dropped* from the pending pool.  Previously they were
          retained, which meant downstream stages (contract learning,
          bank maintenance) saw partially-segmented episodes and
          mis-counted skill provenance.  Dropping is the safe failure
          mode: the next step's rollouts will produce fresh episodes
          to feed the bank.
        """
        episodes = []
        for r in results:
            if r.steps > 0:
                episodes.append(self._convert_episode_result(r))

        if not episodes:
            return

        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()
        executor = self._executor
        t0 = time.monotonic()
        sem = asyncio.Semaphore(self._MAX_CONCURRENT_SEGMENTATIONS)

        def _segment_one(ep):
            try:
                result, sub_eps = agent.segment_episode(ep, env_name="llm")
                n_segs = len(result.segments) if hasattr(result, "segments") else 0
                logger.debug(
                    "Segmented %s: %d steps → %d segments",
                    ep.episode_id, len(ep.experiences), n_segs,
                )
                return True
            except Exception as exc:
                logger.warning("Segmentation failed for %s: %s", ep.episode_id, exc)
                return False

        # Track per-episode futures so timeouts can cancel them.  Map
        # holds (future, ep) so we can identify which episodes survived.
        in_flight: Dict[Any, Any] = {}

        async def _segment_with_sem(ep):
            async with sem:
                fut = loop.run_in_executor(executor, _segment_one, ep)
                in_flight[id(ep)] = fut
                ep_timeout = self._segment_timeout_for(ep)
                try:
                    res = await asyncio.wait_for(
                        asyncio.shield(fut),
                        timeout=ep_timeout,
                    )
                    return res
                except asyncio.TimeoutError:
                    n_steps = len(getattr(ep, "experiences", ()) or ())
                    logger.error(
                        "Segmentation timed out after %.0fs for %s "
                        "(n_steps=%d, cancelling future, dropping episode)",
                        ep_timeout,
                        getattr(ep, "episode_id", "?"),
                        n_steps,
                    )
                    # Best-effort cancel; the executor may not actually
                    # interrupt the running thread (Python doesn't
                    # support cooperative thread cancellation), but at
                    # minimum this marks the future as cancelled so
                    # asyncio stops waiting on it and the thread will
                    # not block the next step's submissions.
                    fut.cancel()
                    return False
                finally:
                    in_flight.pop(id(ep), None)

        results_ok = await asyncio.gather(
            *[_segment_with_sem(ep) for ep in episodes],
            return_exceptions=True,
        )

        n_ok = 0
        kept_episodes: List[Any] = []
        for ep, r in zip(episodes, results_ok):
            if r is True:
                n_ok += 1
                kept_episodes.append(ep)
            # else: timeout or raise — drop from pending so downstream
            # stages don't see partially-segmented data.
        elapsed = time.monotonic() - t0
        n_dropped = len(episodes) - n_ok
        if n_dropped > 0:
            logger.warning(
                "Segmented %d/%d episodes in %.1fs (%d dropped — "
                "timeout or raise, see preceding error/warning lines)",
                n_ok, len(episodes), elapsed, n_dropped,
            )
        else:
            logger.info(
                "Segmented %d/%d episodes in %.1fs",
                n_ok, len(episodes), elapsed,
            )

        self._pending_episodes.extend(kept_episodes)

    async def finalize_update(self) -> SkillBankUpdateResult:
        """Run the full skill-bank update pipeline.

        Execution order (changed from earlier versions):
          1. Proto-skill materialization — turn __NEW__ clusters into real
             skills so that contract learning and bank maintenance have
             non-empty skill vocabularies from the very first step.
          2. Contract learning (Stage 3) — learn effect summaries; now runs
             on materialized skill labels instead of seeing only __NEW__.
          3. Bank maintenance (Stage 4) — split / merge / refine existing
             skills, with LLM curator filtering.

        Returns the update result with bank metrics.
        """
        agent = self._ensure_agent()
        loop = asyncio.get_running_loop()
        executor = self._executor
        t0 = time.monotonic()
        stage_times: Dict[str, float] = {}

        n_episodes = len(self._pending_episodes)
        n_skills_before = len(agent.skill_ids)

        # ── 1. Proto-skill materialization (FIRST) ───────────────────
        t_mat = time.monotonic()

        def _materialize():
            try:
                n_formed = agent.form_proto_skills()
                n_verified = agent.verify_proto_skills()
                n_promoted = agent.promote_proto_skills()
                n_materialized = agent.materialize_new_skills()
                return {
                    "formed": n_formed, "verified": n_verified,
                    "promoted": n_promoted, "materialized": n_materialized,
                }
            except Exception as exc:
                logger.warning("Proto-skill processing failed: %s", exc)
                return {}

        mat_result = await loop.run_in_executor(executor, _materialize)
        stage_times["materialization"] = time.monotonic() - t_mat

        n_after_materialize = len(agent.skill_ids)
        if n_after_materialize > n_skills_before:
            logger.info(
                "Materialized %d new skills (%d→%d) — "
                "relabelling __NEW__ segments before contract learning",
                n_after_materialize - n_skills_before,
                n_skills_before, n_after_materialize,
            )

        # ── 2. Contract learning (Stage 3) ───────────────────────────
        t_s3 = time.monotonic()

        def _run_contracts():
            if agent._all_segments:
                try:
                    return agent.run_contract_learning()
                except Exception as exc:
                    logger.warning("Contract learning failed: %s", exc)
            return None

        s3_result = await loop.run_in_executor(executor, _run_contracts)
        stage_times["contract_learning"] = time.monotonic() - t_s3

        # ── 3. Bank maintenance (Stage 4) ────────────────────────────
        t_s4 = time.monotonic()

        def _run_maintenance():
            if agent._all_segments and len(agent.skill_ids) > 0:
                try:
                    return agent.run_bank_maintenance()
                except Exception as exc:
                    logger.warning("Bank maintenance failed: %s", exc)
            return None

        s4_result = await loop.run_in_executor(executor, _run_maintenance)
        stage_times["bank_maintenance"] = time.monotonic() - t_s4

        # ── 4. Skill enrichment (protocols, hints, durations) ─────
        t_enrich = time.monotonic()

        def _enrich():
            try:
                from trainer.coevolution.skill_enrichment import (
                    enrich_bank_after_update,
                )
                return enrich_bank_after_update(
                    agent, episodes=self._pending_episodes,
                )
            except Exception as exc:
                logger.warning("Skill enrichment failed: %s", exc)
                return {}

        enrich_result = await loop.run_in_executor(executor, _enrich)
        stage_times["enrichment"] = time.monotonic() - t_enrich

        # ── 5. LLM protocol synthesis (progressive) ──────────────────
        t_proto = time.monotonic()

        def _synthesize_protocols():
            import os
            try:
                n_updated = agent.update_protocols()
                n_refined = 0
                iteration = getattr(agent, "_iteration_count", 0)
                refine_every = int(os.environ.get("PROTOCOL_REFINE_EVERY", "3"))
                if refine_every > 0 and iteration % refine_every == 0:
                    n_refined = agent.refine_low_pass_protocols()
                if n_updated or n_refined:
                    logger.info(
                        "Protocol synthesis: %d synthesized, %d refined",
                        n_updated, n_refined,
                    )
                return {"synthesized": n_updated, "refined": n_refined}
            except Exception as exc:
                logger.warning("Protocol synthesis failed: %s", exc)
                return {}

        proto_result = await loop.run_in_executor(executor, _synthesize_protocols)
        stage_times["protocol_synthesis"] = time.monotonic() - t_proto

        # ── Save bank ────────────────────────────────────────────────
        def _save_bank():
            try:
                agent.save()
            except Exception as exc:
                logger.warning("Bank save failed: %s", exc)

        await loop.run_in_executor(executor, _save_bank)

        self._query_engine = None

        n_skills_after = len(agent.skill_ids)
        elapsed = time.monotonic() - t0

        self._update_result = SkillBankUpdateResult(
            accepted=True,
            bank_version=getattr(agent, "_iteration_count", 0),
            n_skills=n_skills_after,
            n_new_skills=max(0, n_skills_after - n_skills_before),
            n_episodes_processed=n_episodes,
            wall_time_s=elapsed,
            stage_times=stage_times,
            grpo_data=self._grpo_data,
        )

        logger.info(
            "Skill bank update: %d→%d skills (+%d), %d episodes, %.1fs",
            n_skills_before, n_skills_after,
            self._update_result.n_new_skills, n_episodes, elapsed,
        )

        return self._update_result

    def get_raw_bank(self) -> Any:
        """Return the raw ``SkillBankMVP`` (has ``.skill_ids``, etc.)."""
        if self._agent is not None:
            return self._agent.bank
        return None

    def get_bank(self) -> Any:
        """Return a query-engine-wrapped skill bank for decision agents.

        Wrapping in ``SkillQueryEngine`` provides the ``.select()`` method
        needed by ``get_top_k_skill_candidates`` for multi-candidate skill
        selection.  Without this, only a single fallback candidate is
        returned and the skill_selection adapter never fires.
        """
        if self._agent is None:
            return None
        bank = self._agent.bank
        if bank is None or len(bank) == 0:
            return bank
        if self._query_engine is not None:
            return self._query_engine
        try:
            from skill_agents.query import SkillQueryEngine
            self._query_engine = SkillQueryEngine(bank)
            return self._query_engine
        except Exception as exc:
            logger.warning(
                "SkillQueryEngine init failed for bank with %d skills: %s — "
                "skill_selection GRPO will not fire (only single fallback candidate)",
                len(bank), exc,
            )
            return bank

    def get_agent(self) -> Any:
        """Return the SkillBankAgent instance."""
        return self._agent

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._grpo_data

    def reset_for_step(self) -> None:
        """Clear per-step state (pending episodes, GRPO data).

        ``_new_pool_mgr`` is intentionally preserved across steps so
        that ``__NEW__`` segment candidates accumulate until they meet
        the clustering/promotion thresholds.  Only the per-step working
        lists (``_all_segments``, ``_new_pool``) are cleared.
        ``_observations_by_traj`` is also kept so cross-step
        materialization can access trajectory data from earlier steps.
        """
        self._pending_episodes.clear()
        self._grpo_data = {"segment": [], "contract": [], "curator": []}
        self._update_result = None
        if self._agent is not None:
            self._agent._all_segments = []
            self._agent._new_pool = []
            # NOTE: _new_pool_mgr and _observations_by_traj are NOT
            # cleared — they accumulate across steps on purpose.


class PerGameSkillBankManager:
    """Maintains a separate ``AsyncSkillBankPipeline`` per game.

    Each game gets its own ``skill_bank.jsonl`` under
    ``<bank_dir>/<game>/skill_bank.jsonl``, so skills learned in Tetris
    stay separate from Diplomacy, etc.

    When ``unified_role_rollouts=True``, Avalon and Diplomacy are
    further split into per-side / per-power sub-banks:

    - ``avalon/good/skill_bank.jsonl`` — Merlin, Percival, Servant
    - ``avalon/evil/skill_bank.jsonl`` — Minion, Assassin, Morgana, …
    - ``diplomacy/FRANCE/skill_bank.jsonl`` — one per power

    Other games keep a single bank.  The manager resolves which bank an
    episode belongs to via :func:`resolve_bank_key`.
    """

    def __init__(
        self,
        games: List[str],
        bank_dir: str = "runs/skillbank",
        model_name: str = "Qwen/Qwen3.5-9B",
        executor: Optional[ThreadPoolExecutor] = None,
        grpo_group_size: int = 4,
        seed_bank_dir: Optional[str] = None,
        process_executor: Optional[ProcessPoolExecutor] = None,
        unified_role_rollouts: bool = False,
    ):
        from trainer.coevolution.config import bank_keys_for_game, resolve_bank_key

        self._process_executor = process_executor
        self._unified_role_rollouts = unified_role_rollouts
        self._resolve_bank_key = resolve_bank_key
        self._pipelines: Dict[str, AsyncSkillBankPipeline] = {}

        for game in games:
            if unified_role_rollouts:
                keys = bank_keys_for_game(game)
            else:
                keys = [game]

            for key in keys:
                sub_dir = str(Path(bank_dir) / key)
                Path(sub_dir).mkdir(parents=True, exist_ok=True)
                self._pipelines[key] = AsyncSkillBankPipeline(
                    bank_dir=sub_dir,
                    model_name=model_name,
                    executor=executor,
                    report_dir=str(Path(sub_dir) / "reports"),
                    game_name=game,
                )

        self._bank_dir = bank_dir
        self._games = list(games)
        self._grpo_group_size = grpo_group_size
        self._grpo_buffer: Optional[Any] = None
        self._collected_grpo: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        logger.info(
            "PerGameSkillBankManager: %d bank(s) under %s "
            "(unified_role=%s, process_pool=%s)",
            len(self._pipelines), bank_dir,
            unified_role_rollouts, process_executor is not None,
        )

        if seed_bank_dir:
            self._seed_from_coldstart(seed_bank_dir)

    # ── Bank seeding ─────────────────────────────────────────────────

    def _seed_from_coldstart(self, seed_dir: str) -> None:
        """Copy skills from a cold-start bank into empty per-game banks.

        Only seeds a bank when it currently contains zero skills, so an
        in-progress run that already has its own skills is never
        overwritten.

        For composite keys (e.g. ``"avalon/good"``), first tries the
        matching sub-path in the seed dir, then falls back to the parent
        game key (e.g. ``seed_dir/avalon/skill_bank.jsonl``) so that a
        single legacy seed bank can bootstrap all sub-banks.

        Works at the file level (via ``SkillBankMVP``) rather than
        requiring the lazy ``SkillBankAgent`` to be initialised.
        """
        from skill_agents.skill_bank.bank import SkillBankMVP

        seed_path = Path(seed_dir)
        if not seed_path.is_dir():
            logger.warning("seed_bank_dir %s does not exist — skipping seed", seed_dir)
            return

        for key, pipe in self._pipelines.items():
            dest_file = Path(pipe.bank_dir) / "skill_bank.jsonl"

            if dest_file.exists() and dest_file.stat().st_size > 0:
                logger.info(
                    "Seed skip %s: bank file already exists at %s", key, dest_file,
                )
                continue

            candidate = seed_path / key / "skill_bank.jsonl"
            if not candidate.exists() and "/" in key:
                parent_game = key.split("/", 1)[0]
                candidate = seed_path / parent_game / "skill_bank.jsonl"
            if not candidate.exists():
                logger.info("Seed skip %s: no seed file found", key)
                continue

            bank = SkillBankMVP(str(dest_file))
            bank.load(str(candidate))
            n = len(bank)
            if n > 0:
                bank.save()
                logger.info(
                    "Seeded %s bank with %d skills from %s", key, n, candidate,
                )
            else:
                logger.info("Seed file %s was empty — nothing to load", candidate)

    # ── GRPO wrapper management ─────────────────────────────────────

    def _enable_grpo_wrappers(self) -> None:
        """Activate GRPO wrappers on skill-bank LLM calls (module-level)."""
        from skill_agents.grpo.buffer import GRPOBuffer
        from skill_agents.stage3_mvp.llm_contract import enable_contract_grpo
        from skill_agents.bank_maintenance.llm_curator import enable_curator_grpo
        from skill_agents.infer_segmentation.llm_teacher import enable_segment_grpo
        from skill_agents.infer_segmentation.episode_adapter import (
            grpo_scorer_factory,
            grpo_decode_fn,
        )

        self._grpo_buffer = GRPOBuffer()
        gs = self._grpo_group_size
        modes = _manager_ablation_modes(getattr(self, "_games", []))

        if modes["segmentation_mode"] != "heuristic_only":
            enable_segment_grpo(
                buffer=self._grpo_buffer, group_size=gs, temperature=1.0,
                scorer_factory=grpo_scorer_factory,
                decode_fn=grpo_decode_fn,
            )
        if modes["effect_contract_mode"] == "consensus_verified":
            enable_contract_grpo(buffer=self._grpo_buffer, group_size=gs, temperature=0.8)
        enable_curator_grpo(buffer=self._grpo_buffer, group_size=gs, temperature=0.8)
        logger.info("Contract/Curator reward context is dynamic — set before each LLM call")
        logger.info("Skill-bank GRPO wrappers enabled (G=%d)", gs)

    def _disable_grpo_wrappers(self) -> None:
        """Deactivate GRPO wrappers and restore original functions."""
        from skill_agents.stage3_mvp.llm_contract import disable_contract_grpo
        from skill_agents.bank_maintenance.llm_curator import disable_curator_grpo
        from skill_agents.infer_segmentation.llm_teacher import disable_segment_grpo

        disable_segment_grpo()
        disable_contract_grpo()
        disable_curator_grpo()
        logger.info("Skill-bank GRPO wrappers disabled")

    def _collect_grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Drain the shared GRPO buffer into the per-adapter dict format.

        Preserves ``metadata`` (including skill_id, game context) from
        each sample for downstream logging and per-game diagnostics.
        """
        from skill_agents.lora.skill_function import SkillFunction

        collected: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        if self._grpo_buffer is None:
            return collected

        adapter_map = {
            SkillFunction.SEGMENT: "segment",
            SkillFunction.CONTRACT: "contract",
            SkillFunction.CURATOR: "curator",
        }
        for sf, key in adapter_map.items():
            for sample in self._grpo_buffer.samples_for(sf):
                if sample.prompt and sample.completions:
                    collected[key].append({
                        "prompt": sample.prompt,
                        "completions": sample.completions,
                        "rewards": sample.rewards,
                        "metadata": sample.metadata,
                    })

        n_total = sum(len(v) for v in collected.values())
        if n_total:
            logger.info(
                "Collected %d GRPO samples: segment=%d, contract=%d, curator=%d",
                n_total, len(collected["segment"]),
                len(collected["contract"]), len(collected["curator"]),
            )
        return collected

    def pipeline_for(self, key: str) -> Optional[AsyncSkillBankPipeline]:
        return self._pipelines.get(key)

    def get_bank(self, key: str) -> Any:
        """Return the bank for *key*.

        *key* is a game name (legacy mode) or a composite key like
        ``"avalon/good"`` or ``"diplomacy/FRANCE"`` (unified-role mode).
        """
        pipe = self._pipelines.get(key)
        return pipe.get_bank() if pipe else None

    def get_banks(self) -> Dict[str, Any]:
        """Return ``{key: bank}`` for all pipelines that have a loaded bank.

        Keys are game names in legacy mode, or composite keys like
        ``"avalon/good"`` in unified-role mode.
        """
        return {
            key: pipe.get_bank()
            for key, pipe in self._pipelines.items()
            if pipe.get_bank() is not None
        }

    def get_agents(self) -> Dict[str, Any]:
        return {
            key: pipe.get_agent()
            for key, pipe in self._pipelines.items()
        }

    def ensure_agents_initialized(self) -> Dict[str, Any]:
        """Eagerly trigger lazy ``_ensure_agent`` on every pipeline.

        Returns ``{key: agent}`` where every ``agent`` is a fully
        constructed :class:`SkillBankAgent`. Used by the orchestrator's
        resume path so :func:`load_checkpoint` actually sees concrete
        agents instead of the lazy ``None`` placeholders that
        :meth:`get_agents` returns on a fresh trainer process — without
        this, ``load_checkpoint``'s ``if agent is None: continue`` would
        silently skip restoring the per-game skill bank, forcing the
        next outer step into spurious cold-start mode (the failure
        mode reproduced in run ``Qwen3.5-9B_20260504_144712`` where
        the trainer crashed mid-step-11 and the new process resumed
        with ``bank=0 (empty)``).
        """
        agents: Dict[str, Any] = {}
        for key, pipe in self._pipelines.items():
            try:
                agents[key] = pipe._ensure_agent()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ensure_agents_initialized: pipeline %s _ensure_agent "
                    "failed (%s) — bank restore for this key will be skipped",
                    key, exc,
                )
                agents[key] = None
        return agents

    def reload_banks_from_disk(
        self,
        keys: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Reload the in-memory skill bank for one or more pipelines from
        disk, invalidating their query-engine caches.

        Required after the Phase B′ Crafter+Promotion writeback so the
        actor's next-step rollout observes the freshly-promoted skills.

        Parameters
        ----------
        keys
            Iterable of pipeline keys to reload.  Defaults to *every*
            pipeline.  Unknown keys are ignored with a debug log line.

        Returns
        -------
        dict
            ``{key: AsyncSkillBankPipeline.reload_bank_from_disk()}``
            — one entry per key actually reloaded.
        """
        target_keys = list(keys) if keys is not None else list(self._pipelines.keys())
        out: Dict[str, Dict[str, Any]] = {}
        for key in target_keys:
            pipe = self._pipelines.get(key)
            if pipe is None:
                logger.debug(
                    "PerGameSkillBankManager.reload_banks_from_disk: "
                    "unknown key %s — skipping", key,
                )
                continue
            try:
                out[key] = pipe.reload_bank_from_disk()
            except Exception as exc:                                  # noqa: BLE001
                logger.warning(
                    "reload_banks_from_disk: pipeline %s raised: %s",
                    key, exc,
                )
                out[key] = {
                    "reloaded": False,
                    "skipped_reason": f"reload raised: {exc}",
                }
        return out

    def bank_paths(self, *, simple_only: bool = True) -> Dict[str, "Path"]:
        """Return ``{key: <bank_dir>/<key>/skill_bank.jsonl}`` for every
        per-game pipeline.

        Used by the per-step Crafter/Promotion hooks
        (``trainer/coevolution/_crafter_hook.py``,
        ``trainer/coevolution/_promotion_hook.py``) to resolve the
        on-disk per-game ``skill_bank.jsonl`` they read/writeback.

        Parameters
        ----------
        simple_only
            When ``True`` (default), composite keys like ``"avalon/good"``
            from ``unified_role_rollouts=True`` are filtered out — those
            don't round-trip through the offline-mirror's
            ``<corpus>/<source>/`` layout cleanly. The keystone-Phase-1
            integration only targets simple-keyed games (the 13 retro
            ``Temporal_*-v0`` games + tetris / 2048 / candy_crush /
            super_mario), which all satisfy this. Pass ``simple_only=False``
            once the offline mirror gains a unified-role corpus split.
        """
        from pathlib import Path

        out: Dict[str, "Path"] = {}
        for key, pipe in self._pipelines.items():
            if simple_only and "/" in key:
                continue
            out[key] = Path(pipe.bank_dir) / "skill_bank.jsonl"
        return out

    def reset_for_step(self) -> None:
        for pipe in self._pipelines.values():
            pipe.reset_for_step()
        self._grpo_buffer = None
        self._collected_grpo = {"segment": [], "contract": [], "curator": []}
        try:
            self._disable_grpo_wrappers()
        except Exception:
            pass
        try:
            self._enable_grpo_wrappers()
        except Exception as exc:
            logger.warning("Failed to enable GRPO wrappers: %s", exc)

    def _key_for_result(self, result: EpisodeResult) -> str:
        """Resolve the bank key for an ``EpisodeResult``."""
        if self._unified_role_rollouts and (result.role or result.side):
            return self._resolve_bank_key(
                result.game, result.role, result.side,
            )
        return result.game

    async def process_batch_async(
        self, results: List[EpisodeResult],
    ) -> None:
        """Route episodes to the correct per-game (or per-side/power) pipeline."""
        by_key: Dict[str, List[EpisodeResult]] = {}
        for r in results:
            key = self._key_for_result(r)
            by_key.setdefault(key, []).append(r)

        tasks = []
        for key, key_results in by_key.items():
            pipe = self._pipelines.get(key)
            if pipe is None:
                pipe = self._pipelines.get(key_results[0].game)
            if pipe is None:
                logger.warning(
                    "No skill bank pipeline for key '%s', skipping %d episodes",
                    key, len(key_results),
                )
                continue
            tasks.append(pipe.process_batch_async(key_results))

        if tasks:
            await asyncio.gather(*tasks)

    async def finalize_all(self) -> Dict[str, SkillBankUpdateResult]:
        """Finalize all per-game banks and return per-game results.

        When a ``ProcessPoolExecutor`` was provided, per-game finalization
        runs in separate processes for true parallelism on CPU-bound
        stages.  Otherwise falls back to asyncio tasks (concurrent but
        GIL-bound).
        """
        results: Dict[str, SkillBankUpdateResult] = {}

        async def _finalize_one(game: str, pipe: AsyncSkillBankPipeline):
            try:
                results[game] = await pipe.finalize_update()
            except Exception as exc:
                logger.error("Skill bank finalize failed for %s: %s", game, exc)

        tasks = [
            _finalize_one(game, pipe)
            for game, pipe in self._pipelines.items()
        ]
        await asyncio.gather(*tasks)

        try:
            self._disable_grpo_wrappers()
        except Exception as exc:
            logger.warning("Failed to disable GRPO wrappers: %s", exc)

        self._collected_grpo = self._collect_grpo_data()

        return results

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return GRPO training data collected by the wrappers."""
        return self._collected_grpo

    def total_skills(self) -> int:
        total = 0
        for pipe in self._pipelines.values():
            bank = pipe.get_raw_bank()
            if bank and hasattr(bank, "skill_ids"):
                total += len(list(bank.skill_ids))
        return total

    def skill_counts(self) -> Dict[str, int]:
        """Return ``{game: n_skills}``."""
        counts = {}
        for game, pipe in self._pipelines.items():
            bank = pipe.get_raw_bank()
            if bank and hasattr(bank, "skill_ids"):
                counts[game] = len(list(bank.skill_ids))
            else:
                counts[game] = 0
        return counts


# ---------------------------------------------------------------------------
# Shared-bank manager (cross-game lifelong-learning mode)
# ---------------------------------------------------------------------------

class SharedSkillBankManager:
    """One bank file shared across all curriculum games.

    Drop-in alternative to :class:`PerGameSkillBankManager` for the
    cross-game / lifelong-learning experiments described in
    ``training_notes/coevo-3phase-cross-game-ood-transfer-plan.md``.

    Design summary
    --------------
    * All games write to a single ``<bank_dir>/skill_bank.jsonl`` (no
      per-game sub-directory). Storage / atomic-save semantics carry
      over from :class:`skill_agents.skill_bank.bank.SkillBankMVP.save`.
    * Every newly-mined skill is stamped with ``feasible_tasks=[<source_game>]``
      so the harness :class:`harness.eligibility.EligibilityFilter` only
      admits it on its source game *unless* the cross-game translator
      (``skill_agents.skill_bank.translate_for_target``) emits a
      derived record with ``feasible_tasks=[<target_game>]``. This is
      the load-bearing invariant that prevents the §22 "100 % cross-
      contamination" pathology measured in
      ``labeling_supplement/_phase0_cross_eligibility_probe.py``.
    * The external interface (``bank_paths``, ``get_banks``,
      ``get_agents``, ``process_batch_async``, ``finalize_all``,
      ``reload_banks_from_disk``, ``reset_for_step``, ``total_skills``,
      ``skill_counts``, ``grpo_data``) matches
      :class:`PerGameSkillBankManager` 1:1 so
      :func:`trainer.coevolution.orchestrator.run_coevolution_loop`
      can branch on ``config.bank_mode`` without further changes.

    Per-game pipelines are *not* instantiated. Internally there is a
    single :class:`AsyncSkillBankPipeline` whose ``game_name`` is
    re-pointed to the active game whenever ``process_batch_async``
    receives episodes from a new game (LLM prompts that branch on
    ``game_name`` thus see the right context, identical to per-game
    mode within a single phase).

    Backward-compat: ``unified_role_rollouts=True`` is *not* supported
    in shared mode (Avalon / Diplomacy per-side splits are tied to the
    per-game layout). The constructor raises ``ValueError`` if both
    are set so we fail fast at orchestrator startup rather than mid-
    training.
    """

    def __init__(
        self,
        games: List[str],
        bank_dir: str = "runs/skillbank",
        model_name: str = "Qwen/Qwen3.5-9B",
        executor: Optional[ThreadPoolExecutor] = None,
        grpo_group_size: int = 4,
        seed_bank_dir: Optional[str] = None,
        process_executor: Optional[ProcessPoolExecutor] = None,
        unified_role_rollouts: bool = False,
    ):
        if unified_role_rollouts:
            raise ValueError(
                "SharedSkillBankManager does not support unified_role_rollouts=True. "
                "Use PerGameSkillBankManager (config.bank_mode='per_game') for "
                "per-side / per-power Avalon / Diplomacy banks."
            )

        self._games = list(games)
        self._bank_dir = bank_dir
        self._process_executor = process_executor
        self._grpo_group_size = grpo_group_size

        Path(bank_dir).mkdir(parents=True, exist_ok=True)
        self._shared_pipeline = AsyncSkillBankPipeline(
            bank_dir=bank_dir,
            model_name=model_name,
            executor=executor,
            report_dir=str(Path(bank_dir) / "reports"),
            game_name=(games[0] if games else "shared"),
        )
        self._grpo_buffer: Optional[Any] = None
        self._collected_grpo: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        # Track which game we last finalized so ``finalize_all`` can
        # stamp the right ``feasible_tasks`` on newly-minted skills.
        self._last_processed_game: Optional[str] = None

        logger.info(
            "SharedSkillBankManager: 1 shared bank under %s for %d game(s) "
            "(process_pool=%s)",
            bank_dir, len(self._games), process_executor is not None,
        )

        if seed_bank_dir:
            self._seed_from_coldstart(seed_bank_dir)

    # ── Bank seeding ─────────────────────────────────────────────────

    def _seed_from_coldstart(self, seed_dir: str) -> None:
        """Seed the shared bank from a per-game cold-start directory.

        Concatenates every ``<seed_dir>/<game>/skill_bank.jsonl`` we
        find for the configured games into the shared bank, stamping
        ``feasible_tasks=[<source_game>]`` on each entry so the harness
        eligibility filter routes them correctly. Existing skills
        already in the shared bank are preserved (no overwrites; we
        only seed when the destination is empty, mirroring
        :func:`PerGameSkillBankManager._seed_from_coldstart`).
        """
        from skill_agents.skill_bank.bank import SkillBankMVP

        seed_path = Path(seed_dir)
        if not seed_path.is_dir():
            logger.warning("seed_bank_dir %s does not exist — skipping seed", seed_dir)
            return

        dest_file = Path(self._shared_pipeline.bank_dir) / "skill_bank.jsonl"
        if dest_file.exists() and dest_file.stat().st_size > 0:
            logger.info(
                "SharedSkillBankManager seed skip: shared bank already exists at %s",
                dest_file,
            )
            return

        merged = SkillBankMVP(str(dest_file))
        n_loaded_total = 0
        for game in self._games:
            candidate = seed_path / game / "skill_bank.jsonl"
            if not candidate.exists():
                continue
            tmp = SkillBankMVP(str(candidate))
            tmp.load(str(candidate))
            for sid in tmp.skill_ids:
                skill = tmp.get_skill(sid)
                if skill is None:
                    continue
                # Stamp the source game on the seeded skill so the
                # harness's task-axis veto (F2′) admits it only on
                # ``state.task == game`` — the §22 invariant. Don't
                # widen ``feasible_tasks`` even when the skill already
                # carries an entry; the seed is *provenance*, the
                # translator is what registers a wider eligibility.
                if not skill.feasible_tasks:
                    skill.feasible_tasks = [game]
                merged.add_or_update_skill(skill)
                n_loaded_total += 1

        if n_loaded_total > 0:
            merged.save()
            logger.info(
                "SharedSkillBankManager seeded %d skills from %s "
                "(across %d game subdirs)",
                n_loaded_total, seed_dir, len(self._games),
            )
        else:
            logger.info("SharedSkillBankManager: no seed files found under %s", seed_dir)

    # ── External-interface methods (mirror PerGameSkillBankManager) ──

    def pipeline_for(self, key: str) -> Optional[AsyncSkillBankPipeline]:
        """Return the shared pipeline regardless of *key*."""
        return self._shared_pipeline

    def get_bank(self, key: str) -> Any:
        """Return the shared bank regardless of *key*."""
        return self._shared_pipeline.get_bank()

    def get_banks(self) -> Dict[str, Any]:
        """Return ``{game: shared_bank}`` — every game key maps to the
        single shared bank instance, so existing callers iterating per
        game still work without modification."""
        bank = self._shared_pipeline.get_bank()
        if bank is None:
            return {}
        return {game: bank for game in self._games}

    def get_agents(self) -> Dict[str, Any]:
        """Return ``{game: shared_agent}`` — same shared agent for every
        game. The agent's ``game_name`` is updated dynamically by
        :meth:`process_batch_async` so per-game prompt branches still
        receive the right context."""
        agent = self._shared_pipeline.get_agent()
        return {game: agent for game in self._games}

    def ensure_agents_initialized(self) -> Dict[str, Any]:
        """Eagerly trigger lazy ``_ensure_agent`` on the shared pipeline.

        Mirrors :meth:`PerGameSkillBankManager.ensure_agents_initialized`
        — returns ``{game: shared_agent}`` with the agent fully
        instantiated so :func:`load_checkpoint` can actually restore
        per-step bank snapshots on a freshly-respawned trainer process.
        """
        try:
            agent = self._shared_pipeline._ensure_agent()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ensure_agents_initialized (shared): _ensure_agent failed: %s — "
                "bank restore will be skipped",
                exc,
            )
            agent = None
        return {game: agent for game in self._games}

    def reload_banks_from_disk(
        self,
        keys: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Reload the shared bank from disk. *keys* are accepted for
        signature compatibility but only one reload occurs regardless;
        we return the same result keyed under every requested game so
        downstream loggers see one entry per game."""
        target_keys = list(keys) if keys is not None else list(self._games)
        try:
            result = self._shared_pipeline.reload_bank_from_disk()
        except Exception as exc:                                       # noqa: BLE001
            logger.warning(
                "SharedSkillBankManager.reload_banks_from_disk: %s", exc,
            )
            result = {"reloaded": False, "skipped_reason": f"reload raised: {exc}"}
        return {key: dict(result) for key in target_keys}

    def bank_paths(self, *, simple_only: bool = True) -> Dict[str, "Path"]:
        """Return ``{game: shared_bank_path}`` for every configured game.

        The shared mode has only one on-disk file but the per-step
        hooks (``_crafter_hook``, ``_promotion_hook``, ``_dashboard_hook``,
        ``_transfer_hook``) iterate the dict to read their per-game
        view. Pointing every key at the same file makes the hooks
        observe the union — which is exactly what shared mode wants.

        Composite (``"avalon/good"``) keys are never produced in shared
        mode (we forbid ``unified_role_rollouts=True`` in __init__),
        so ``simple_only`` is a no-op here but accepted for signature
        compatibility with :class:`PerGameSkillBankManager`.
        """
        from pathlib import Path

        shared = Path(self._shared_pipeline.bank_dir) / "skill_bank.jsonl"
        return {game: shared for game in self._games}

    # ── GRPO wrapper management (identical to PerGameSkillBankManager) ──

    def _enable_grpo_wrappers(self) -> None:
        from skill_agents.grpo.buffer import GRPOBuffer
        from skill_agents.stage3_mvp.llm_contract import enable_contract_grpo
        from skill_agents.bank_maintenance.llm_curator import enable_curator_grpo
        from skill_agents.infer_segmentation.llm_teacher import enable_segment_grpo
        from skill_agents.infer_segmentation.episode_adapter import (
            grpo_scorer_factory,
            grpo_decode_fn,
        )

        self._grpo_buffer = GRPOBuffer()
        gs = self._grpo_group_size
        modes = _manager_ablation_modes(getattr(self, "_games", []))
        if modes["segmentation_mode"] != "heuristic_only":
            enable_segment_grpo(
                buffer=self._grpo_buffer, group_size=gs, temperature=1.0,
                scorer_factory=grpo_scorer_factory,
                decode_fn=grpo_decode_fn,
            )
        if modes["effect_contract_mode"] == "consensus_verified":
            enable_contract_grpo(buffer=self._grpo_buffer, group_size=gs, temperature=0.8)
        enable_curator_grpo(buffer=self._grpo_buffer, group_size=gs, temperature=0.8)
        logger.info("Skill-bank GRPO wrappers enabled (G=%d, shared bank)", gs)

    def _disable_grpo_wrappers(self) -> None:
        from skill_agents.stage3_mvp.llm_contract import disable_contract_grpo
        from skill_agents.bank_maintenance.llm_curator import disable_curator_grpo
        from skill_agents.infer_segmentation.llm_teacher import disable_segment_grpo

        disable_segment_grpo()
        disable_contract_grpo()
        disable_curator_grpo()
        logger.info("Skill-bank GRPO wrappers disabled (shared bank)")

    def _collect_grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        from skill_agents.lora.skill_function import SkillFunction

        collected: Dict[str, List[Dict[str, Any]]] = {
            "segment": [], "contract": [], "curator": [],
        }
        if self._grpo_buffer is None:
            return collected
        adapter_map = {
            SkillFunction.SEGMENT: "segment",
            SkillFunction.CONTRACT: "contract",
            SkillFunction.CURATOR: "curator",
        }
        for sf, key in adapter_map.items():
            for sample in self._grpo_buffer.samples_for(sf):
                if sample.prompt and sample.completions:
                    collected[key].append({
                        "prompt": sample.prompt,
                        "completions": sample.completions,
                        "rewards": sample.rewards,
                        "metadata": sample.metadata,
                    })
        n_total = sum(len(v) for v in collected.values())
        if n_total:
            logger.info(
                "Collected %d GRPO samples (shared): segment=%d, contract=%d, curator=%d",
                n_total, len(collected["segment"]),
                len(collected["contract"]), len(collected["curator"]),
            )
        return collected

    def reset_for_step(self) -> None:
        self._shared_pipeline.reset_for_step()
        self._grpo_buffer = None
        self._collected_grpo = {"segment": [], "contract": [], "curator": []}
        try:
            self._disable_grpo_wrappers()
        except Exception:
            pass
        try:
            self._enable_grpo_wrappers()
        except Exception as exc:
            logger.warning("Failed to enable GRPO wrappers (shared): %s", exc)

    async def process_batch_async(
        self, results: List[EpisodeResult],
    ) -> None:
        """Feed every episode to the shared pipeline.

        Re-points the agent's ``game_name`` to the batch's dominant
        game so per-game prompt branches receive the right context.
        Almost always the dominant game is the *only* game (curriculum
        runs one game per phase), but we resolve by majority vote so
        a hybrid phase (e.g. ``--mixed`` curriculum) still works.
        """
        if not results:
            return

        # Majority-game vote for dynamic ``game_name`` re-pointing.
        from collections import Counter
        counts = Counter(r.game for r in results)
        dominant_game, _ = counts.most_common(1)[0]
        self._last_processed_game = dominant_game

        # Re-point the agent + pipeline so contract / curator / segment
        # prompts that branch on ``game_name`` see the right context.
        self._shared_pipeline.game_name = dominant_game
        agent = self._shared_pipeline.get_agent()
        if agent is not None and hasattr(agent, "game_name"):
            try:
                object.__setattr__(agent, "game_name", dominant_game)
            except Exception:                                          # noqa: BLE001
                pass

        await self._shared_pipeline.process_batch_async(results)

    async def finalize_all(self) -> Dict[str, "SkillBankUpdateResult"]:
        """Finalize the shared pipeline and stamp ``feasible_tasks``
        on every freshly-minted skill.

        Returned shape mirrors :meth:`PerGameSkillBankManager.finalize_all`
        so the orchestrator's per-game logging keeps working — we
        replicate the single result under every configured game key.
        """
        try:
            shared_result = await self._shared_pipeline.finalize_update()
        except Exception as exc:
            logger.error("SharedSkillBankManager.finalize_all: %s", exc)
            shared_result = SkillBankUpdateResult(accepted=False)

        # Stamp ``feasible_tasks=[current_game]`` on every skill that
        # doesn't already carry one. This is the §22 invariant: every
        # skill must be admitted for *some* concrete task — empty
        # ``feasible_tasks`` is back-compat task-agnostic and would
        # silently re-introduce 100 % cross-contamination.
        if self._last_processed_game is not None:
            bank = self._shared_pipeline.get_raw_bank()
            if bank is not None and hasattr(bank, "skill_ids"):
                stamped = 0
                for sid in list(bank.skill_ids):
                    skill = bank.get_skill(sid) if hasattr(bank, "get_skill") else None
                    if skill is None:
                        continue
                    if not getattr(skill, "feasible_tasks", None):
                        try:
                            skill.feasible_tasks = [self._last_processed_game]
                            stamped += 1
                        except Exception:                              # noqa: BLE001
                            continue
                if stamped > 0:
                    try:
                        bank.save()
                        logger.info(
                            "SharedSkillBankManager: stamped feasible_tasks=[%s] on %d skills",
                            self._last_processed_game, stamped,
                        )
                    except Exception as exc:                           # noqa: BLE001
                        logger.warning(
                            "SharedSkillBankManager: feasible_tasks stamp save failed: %s",
                            exc,
                        )

        try:
            self._disable_grpo_wrappers()
        except Exception as exc:
            logger.warning("Failed to disable GRPO wrappers (shared): %s", exc)
        self._collected_grpo = self._collect_grpo_data()

        return {game: shared_result for game in self._games}

    @property
    def grpo_data(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._collected_grpo

    def total_skills(self) -> int:
        bank = self._shared_pipeline.get_raw_bank()
        if bank and hasattr(bank, "skill_ids"):
            return len(list(bank.skill_ids))
        return 0

    def skill_counts(self) -> Dict[str, int]:
        """Return per-game skill counts derived from ``feasible_tasks``.

        Unlike :meth:`PerGameSkillBankManager.skill_counts` (which
        reports physical bank sizes), the shared-bank version reports
        the *eligibility-relevant* count — how many skills the harness
        would admit on each game. A skill stamped
        ``feasible_tasks=["candy_crush", "Columns"]`` (after the
        translator widens it) counts toward both games.
        """
        bank = self._shared_pipeline.get_raw_bank()
        counts = {game: 0 for game in self._games}
        if not bank or not hasattr(bank, "skill_ids"):
            return counts
        for sid in bank.skill_ids:
            skill = bank.get_skill(sid) if hasattr(bank, "get_skill") else None
            if skill is None:
                continue
            tasks = list(getattr(skill, "feasible_tasks", None) or [])
            if not tasks:
                # Task-agnostic: visible everywhere. Only legacy /
                # pre-shared-mode skills should be in this branch.
                for game in self._games:
                    counts[game] += 1
                continue
            for game in tasks:
                if game in counts:
                    counts[game] += 1
        return counts
