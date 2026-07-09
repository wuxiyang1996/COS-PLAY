"""Rollout collector with LPT scheduling and async concurrency control.

Collects episodes across all games concurrently, using
Longest-Processing-Time (LPT) ordering to start the slowest games first
and a semaphore to cap GPU-memory pressure from concurrent LLM requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from trainer.coevolution.config import (
    CoEvolutionConfig,
    GAME_DURATION_ORDER,
    GAME_MAX_STEPS,
    AVALON_ROLES,
    AVALON_SIDES,
    DIPLOMACY_POWERS,
    resolve_bank_key,
)
from trainer.coevolution.episode_runner import EpisodeResult, run_episode_async
from trainer.coevolution.vllm_client import AsyncVLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wave synchronizer — keeps episode LLM requests batched on the GPU
# ---------------------------------------------------------------------------

class WaveSynchronizer:
    """Aligns episode LLM-request waves for better GPU batching.

    vLLM throughput drops 10-20x when batch size falls to 1 (the GPU
    becomes memory-bandwidth bound instead of compute-bound).  This
    synchronizer prevents that by holding each episode at a barrier
    before its LLM calls until all active episodes arrive — or a short
    timeout expires — so that the requests reach vLLM together.
    """

    __slots__ = ("_n_active", "_timeout", "_count", "_event", "_lock")

    def __init__(self, n_participants: int, timeout_s: float = 0.10):
        self._n_active = n_participants
        self._timeout = timeout_s
        self._count = 0
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def arrive(self) -> None:
        """Block until all active episodes arrive or *timeout_s* elapses."""
        async with self._lock:
            self._count += 1
            if self._count >= self._n_active:
                self._event.set()
                self._count = 0
                self._event = asyncio.Event()
                return
            my_event = self._event

        try:
            await asyncio.wait_for(my_event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                if self._event is my_event:
                    my_event.set()
                    self._count = 0
                    self._event = asyncio.Event()

    def depart(self) -> None:
        """Decrease the active participant count (episode finished)."""
        self._n_active = max(1, self._n_active - 1)


@dataclass
class EpisodeSpec:
    """One episode to run: game name + settings."""
    game: str
    max_steps: int
    episode_idx: int
    estimated_duration_s: float = 0.0
    eval_only: bool = False
    # Multi-role fields (populated only in unified_role_rollouts mode)
    assigned_role: Optional[str] = None
    assigned_role_index: Optional[int] = None


def build_lpt_schedule(
    games: List[str],
    episodes_per_game: int,
    *,
    episodes_per_game_overrides: Optional[Dict[str, int]] = None,
    unified_role_rollouts: bool = False,
) -> List[EpisodeSpec]:
    """Build an LPT-ordered list of episode specs.

    Strategy: sort games by descending max_steps (proxy for duration), then
    interleave episodes round-robin so long-game episodes are spread across
    the entire collection window (enabling cross-system overlap with the
    skill bank pipeline).

    When *unified_role_rollouts* is ``True``, Avalon and Diplomacy episodes
    cycle through roles deterministically so each rollout covers a
    different role / power.
    """
    overrides = episodes_per_game_overrides or {}
    sorted_games = sorted(
        games,
        key=lambda g: GAME_MAX_STEPS.get(g, 200),
        reverse=True,
    )

    PER_STEP_S = 1.0
    buckets: Dict[str, List[EpisodeSpec]] = {}
    max_eps = 0

    for g in sorted_games:
        ms = GAME_MAX_STEPS.get(g, 200)
        est = ms * PER_STEP_S
        n_eps = overrides.get(g, episodes_per_game) if unified_role_rollouts else episodes_per_game

        specs: List[EpisodeSpec] = []
        for i in range(n_eps):
            spec = EpisodeSpec(
                game=g, max_steps=ms, episode_idx=i,
                estimated_duration_s=est,
            )
            if unified_role_rollouts:
                if g == "avalon":
                    spec.assigned_role_index = i % len(AVALON_ROLES)
                    spec.assigned_role = AVALON_ROLES[spec.assigned_role_index]
                elif g == "diplomacy":
                    spec.assigned_role_index = i % len(DIPLOMACY_POWERS)
                    spec.assigned_role = DIPLOMACY_POWERS[spec.assigned_role_index]
            specs.append(spec)

        buckets[g] = specs
        max_eps = max(max_eps, n_eps)

    schedule: List[EpisodeSpec] = []
    for ep_idx in range(max_eps):
        for g in sorted_games:
            if ep_idx < len(buckets[g]):
                schedule.append(buckets[g][ep_idx])

    return schedule


async def collect_rollouts(
    config: CoEvolutionConfig,
    vllm_client: AsyncVLLMClient,
    skill_bank: Any = None,
    skill_banks: Optional[Dict[str, Any]] = None,
    *,
    on_episode_done: Optional[Callable[[EpisodeResult], None]] = None,
    thread_executor: Optional[ThreadPoolExecutor] = None,
) -> List[EpisodeResult]:
    """Collect rollouts for all games with LPT scheduling and concurrency cap.

    Parameters
    ----------
    skill_bank : object | None
        Legacy single shared bank (used if *skill_banks* is not provided).
    skill_banks : dict | None
        Per-game banks: ``{game_name: bank_object}``.  Takes priority
        over *skill_bank* when both are provided.
    on_episode_done : callable | None
        Called (synchronously) each time an episode finishes.  Used by the
        orchestrator to feed completed trajectories into the skill bank
        pipeline concurrently (cross-system overlap, Strategy E).
    """
    _unified = getattr(config, "unified_role_rollouts", False)
    _overrides = getattr(config, "episodes_per_game_overrides", {})
    schedule = build_lpt_schedule(
        config.games,
        config.episodes_per_game,
        episodes_per_game_overrides=_overrides,
        unified_role_rollouts=_unified,
    )

    eval_games = getattr(config, "eval_games", [])
    eval_eps = getattr(config, "eval_episodes_per_game", 3)
    if eval_games:
        eval_schedule = build_lpt_schedule(eval_games, eval_eps)
        for spec in eval_schedule:
            spec.eval_only = True
        schedule.extend(eval_schedule)

    semaphore = asyncio.Semaphore(config.max_concurrent_episodes)

    sync_timeout = getattr(config, "rollout_sync_timeout_s", 0.10)
    step_sync: Optional[WaveSynchronizer] = None
    if sync_timeout > 0 and len(schedule) > 1:
        step_sync = WaveSynchronizer(len(schedule), timeout_s=sync_timeout)
        logger.info(
            "Rollout wave sync enabled: %d participants, %.0fms timeout",
            len(schedule), sync_timeout * 1000,
        )

    results: List[EpisodeResult] = []
    results_lock = asyncio.Lock()

    def _bank_for(spec: EpisodeSpec) -> Any:
        if skill_banks:
            if _unified and spec.assigned_role:
                key = resolve_bank_key(
                    spec.game,
                    spec.assigned_role or "",
                    AVALON_SIDES.get(spec.assigned_role, spec.assigned_role or ""),
                )
                bank = skill_banks.get(key)
                if bank is not None:
                    return bank
            return skill_banks.get(spec.game)
        return skill_bank

    max_retries = getattr(config, "max_episode_retries", 2)

    async def _run_one(spec: EpisodeSpec) -> None:
        async with semaphore:
            game_bank = _bank_for(spec)
            result: Optional[EpisodeResult] = None
            for attempt in range(1, max_retries + 1):
                try:
                    _opp_model = getattr(config, "opponent_model", None)
                    _opp_base = getattr(config, "opponent_api_base", None)
                    result = await run_episode_async(
                        game=spec.game,
                        max_steps=spec.max_steps,
                        vllm_client=vllm_client,
                        skill_bank=game_bank,
                        temperature=config.temperature,
                        executor=thread_executor,
                        stuck_window=config.stuck_window,
                        min_steps_before_stuck=config.min_steps_before_stuck_check,
                        vllm_base_urls=config.vllm_base_urls,
                        model_name=config.model_name,
                        assigned_role=spec.assigned_role,
                        assigned_role_index=spec.assigned_role_index,
                        step_sync=step_sync,
                        opponent_model=_opp_model if spec.game in ("avalon", "diplomacy") else None,
                        opponent_api_base=_opp_base if spec.game in ("avalon", "diplomacy") else None,
                    )
                    break
                except Exception as exc:
                    if attempt < max_retries:
                        logger.warning(
                            "Episode %s/%d attempt %d/%d failed: %s — retrying",
                            spec.game, spec.episode_idx, attempt, max_retries, exc,
                        )
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(
                            "Episode %s/%d failed after %d attempts: %s",
                            spec.game, spec.episode_idx, max_retries, exc,
                        )
                        result = EpisodeResult(
                            game=spec.game,
                            episode_id=f"{spec.game}_FAILED_{spec.episode_idx}",
                        )

            result.eval_only = spec.eval_only
            async with results_lock:
                results.append(result)

            if on_episode_done is not None:
                try:
                    on_episode_done(result)
                except Exception as exc:
                    logger.warning("on_episode_done callback failed: %s", exc)

    t0 = time.monotonic()
    logger.info(
        "Collecting rollouts: %d episodes (%d games, unified_role=%s), max_concurrent=%d",
        len(schedule), len(config.games), _unified,
        config.max_concurrent_episodes,
    )

    tasks = [asyncio.create_task(_run_one(spec)) for spec in schedule]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    n_ok = sum(1 for r in results if r.steps > 0)
    total_steps = sum(r.steps for r in results)
    total_reward = sum(r.total_reward for r in results)

    logger.info(
        "Rollout collection done: %d/%d ok, %d total steps, %.1f total reward, "
        "%.1fs wall time",
        n_ok, len(results), total_steps, total_reward, elapsed,
    )

    return results


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def compute_episode_metrics(results: List[EpisodeResult]) -> Dict[str, Any]:
    """Aggregate episode-level metrics for logging."""
    from collections import defaultdict

    per_game: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "episodes": 0, "total_reward": 0.0, "total_steps": 0,
        "rewards": [], "step_counts": [],
    })

    for r in results:
        if r.steps == 0:
            continue
        g = per_game[r.game]
        g["episodes"] += 1
        g["total_reward"] += r.total_reward
        g["total_steps"] += r.steps
        g["rewards"].append(r.total_reward)
        g["step_counts"].append(r.steps)

    summary: Dict[str, Any] = {"per_game": {}}
    for game, g in per_game.items():
        rewards = g["rewards"]
        summary["per_game"][game] = {
            "n_episodes": g["episodes"],
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "std_reward": _std(rewards),
            "mean_steps": sum(g["step_counts"]) / len(g["step_counts"]) if g["step_counts"] else 0,
            "total_reward": g["total_reward"],
        }

    all_rewards = [r.total_reward for r in results if r.steps > 0]
    summary["aggregate"] = {
        "n_episodes": len(all_rewards),
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "max_reward": max(all_rewards) if all_rewards else 0.0,
        "min_reward": min(all_rewards) if all_rewards else 0.0,
        "std_reward": _std(all_rewards),
        "total_steps": sum(r.steps for r in results),
        "wall_time_s": max(r.wall_time_s for r in results) if results else 0.0,
    }

    return summary
