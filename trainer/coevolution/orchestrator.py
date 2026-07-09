"""Co-evolution orchestrator — the main training loop.

Implements the three-phase co-evolution step:
  Phase A — Rollout collection (decision agent plays all games)
  Phase B — Skill bank update (segment, contracts, maintenance)
  Phase C — GRPO training (5 LoRA adapters across two agents)

Phases A and B overlap via cross-system batching: as short-game episodes
complete, their trajectories are immediately fed into the skill bank
pipeline while longer games continue running.

Phase C is **pipelined**: GRPO for step N runs on GPUs 4-7 concurrently
with rollout for step N+1 on GPUs 0-3 (vLLM inference).  Rollout N+1
uses adapter weights from step N-1 (one step behind), which is acceptable
for RL since GRPO uses importance sampling ratios to correct for
off-policy data.  This hides ~14 min of GRPO time behind ~22 min of
rollout, saving ~14 min per step.

Checkpoints every ``checkpoint_interval`` steps and logs all metrics
to Weights & Biases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainer.coevolution.checkpoint import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from trainer.coevolution.config import CoEvolutionConfig, prepare_adapters
from trainer.coevolution.episode_runner import EpisodeResult
from trainer.coevolution.grpo_training import GRPOStepResult, run_grpo_training
from trainer.coevolution.rollout_collector import (
    collect_rollouts,
    compute_episode_metrics,
)
from trainer.coevolution.skillbank_pipeline import (
    AsyncSkillBankPipeline,
    PerGameSkillBankManager,
    SkillBankUpdateResult,
)
from trainer.coevolution.vllm_client import AsyncVLLMClient

logger = logging.getLogger(__name__)


def _log_resumed_bank_sizes(sb_manager: Any, source: str, step: int) -> None:
    """Emit per-game bank sizes after a checkpoint restore.

    Helps detect silent bank-restore failures (the failure-mode where
    a respawning trainer's lazy ``SkillBankAgent`` was left as ``None``,
    causing :func:`load_checkpoint` to skip the bank load entirely and
    the next outer step to flip into spurious cold-start mode).  If
    every game reports zero skills here while the checkpoint snapshot
    on disk is non-empty, the resume path is broken.
    """
    try:
        counts = sb_manager.skill_counts()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Resume bank-size log failed (%s source=%s): %s", step, source, exc)
        return
    summary = ", ".join(f"{g}={n}" for g, n in counts.items())
    total = sum(counts.values())
    logger.info(
        "Resumed bank sizes (%s ckpt step %d): total=%d skills [%s]",
        source, step, total, summary or "empty",
    )


async def co_evolution_loop(config: CoEvolutionConfig) -> None:
    """Main co-evolution training loop.

    Step 0:  Rollouts (no bank) → Skill bank extraction → Bank_v1
    Step 1+: Rollouts (with bank) → Skill bank update → Bank_v{k+1}
             + GRPO updates for all 5 LoRAs

    Checkpoints saved every ``checkpoint_interval`` steps.
    All metrics logged to W&B in real time.
    """
    # ── Resolve all paths under run_dir ───────────────────────────
    config.resolve_paths()
    logger.info("Run directory: %s", config.run_dir)

    # ── Ensure LoRA adapters exist ──────────────────────────────────
    adapter_map = prepare_adapters(config)
    logger.info("Adapters ready: %s", list(adapter_map.keys()))

    # ── Initialize W&B ────────────────────────────────────────────
    wandb = None
    if config.wandb_enabled:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.to_dict(),
                resume="allow",
            )
            logger.info("W&B initialized: project=%s", config.wandb_project)
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)
            wandb = None

    # ── vLLM lifecycle manager (persistent on dedicated GPUs) ────
    vllm_manager = None
    if config.manage_vllm:
        from trainer.coevolution.vllm_server import VLLMServerManager

        vllm_log_dir = str(Path(config.run_dir) / "vllm_logs")
        vllm_manager = VLLMServerManager(
            model_name=config.model_name,
            adapter_dir=config.adapter_dir,
            gpu_ids=config.vllm_gpu_ids,
            base_port=config.vllm_base_port,
            gpu_util=config.vllm_gpu_util,
            log_dir=vllm_log_dir,
            speculative_model=config.speculative_model,
            num_speculative_tokens=config.num_speculative_tokens,
            speculative_method=config.speculative_method,
        )
        spec_info = ""
        if config.speculative_method == "mtp" and config.num_speculative_tokens > 0:
            spec_info = f"  |  spec_decode=MTP ({config.num_speculative_tokens} tokens)"
        elif config.speculative_model:
            spec_info = f"  |  spec_decode={config.speculative_model} ({config.num_speculative_tokens} tokens)"
        logger.info(
            "vLLM managed mode: %d × TP=1 instances on GPUs %s, "
            "ports %d–%d  |  GRPO on GPUs %s%s",
            len(config.vllm_gpu_ids), config.vllm_gpu_ids,
            config.vllm_base_port,
            config.vllm_base_port + len(config.vllm_gpu_ids) - 1,
            config.grpo_devices,
            spec_info,
        )

    # ── vLLM client (supports multiple backends) ──────────────
    vllm_client = AsyncVLLMClient(
        base_urls=config.vllm_base_urls,
        model=config.model_name,
        default_temperature=config.temperature,
        default_max_tokens=config.max_tokens,
    )

    if config.debug_io:
        vllm_client.enable_io_logging(config.debug_io_dir, step=0)
        logger.info("Debug I/O logging enabled: %s", config.debug_io_dir)

    # For externally-managed vLLM, verify reachability upfront
    if not vllm_manager:
        healthy = await vllm_client.health_check()
        if not healthy:
            logger.error("vLLM server not reachable at %s", config.vllm_base_url)
            raise ConnectionError(f"vLLM not reachable: {config.vllm_base_url}")
        logger.info("vLLM server healthy at %s", config.vllm_base_url)

    # Expose all vLLM URLs so API_func.ask_vllm round-robins across them
    os.environ["VLLM_BASE_URLS"] = ",".join(config.vllm_base_urls)

    # ── Executors ─────────────────────────────────────────────────
    thread_executor = ThreadPoolExecutor(
        max_workers=config.thread_workers, thread_name_prefix="coevo",
    )
    from concurrent.futures import ProcessPoolExecutor as _PPE
    process_executor = _PPE(max_workers=config.process_workers)

    # ── Skill bank pipelines (per-game default; opt-in shared) ─────
    all_games = list(config.games) + list(getattr(config, "eval_games", []))
    bank_mode = (getattr(config, "bank_mode", "per_game") or "per_game").lower()
    if bank_mode not in ("per_game", "shared"):
        raise ValueError(
            f"Unknown bank_mode={bank_mode!r}. Expected 'per_game' or 'shared'. "
            "See trainer/coevolution/config.py:CoEvolutionConfig.bank_mode."
        )
    if bank_mode == "shared":
        from trainer.coevolution.skillbank_pipeline import SharedSkillBankManager
        sb_manager = SharedSkillBankManager(
            games=all_games,
            bank_dir=config.bank_dir,
            model_name=config.model_name,
            executor=thread_executor,
            seed_bank_dir=getattr(config, "seed_bank_dir", None),
            process_executor=process_executor,
            unified_role_rollouts=getattr(config, "unified_role_rollouts", False),
        )
        logger.info("Skill bank mode: SHARED (one file across %d games)", len(all_games))
    else:
        sb_manager = PerGameSkillBankManager(
            games=all_games,
            bank_dir=config.bank_dir,
            model_name=config.model_name,
            executor=thread_executor,
            seed_bank_dir=getattr(config, "seed_bank_dir", None),
            process_executor=process_executor,
            unified_role_rollouts=getattr(config, "unified_role_rollouts", False),
        )
        logger.info("Skill bank mode: PER_GAME (legacy, %d separate banks)", len(all_games))

    # ── Determine start step ─────────────────────────────────────
    start_step = 0

    if config.start_mode == "from_scratch":
        logger.info("Starting from scratch — ignoring any existing checkpoints")

    elif config.start_mode == "resume":
        if config.resume_from_step is not None:
            start_step = config.resume_from_step
            try:
                # Eagerly initialise lazy SkillBankAgent instances *before*
                # load_checkpoint so per-game banks are actually restored
                # from the snapshot.  Without this, `get_agents()` returns
                # ``{game: None}`` and ``load_checkpoint``'s ``if agent is
                # None: continue`` silently no-ops the bank restore — see
                # run Qwen3.5-9B_20260504_144712 where a mid-step crash
                # left the trainer respawning with ``bank=0 (empty)``.
                sb_manager.ensure_agents_initialized()
                metadata = load_checkpoint(
                    config.checkpoint_dir, start_step,
                    adapter_dir=config.adapter_dir,
                    bank_agents=sb_manager.get_agents(),
                )
                _log_resumed_bank_sizes(sb_manager, "explicit step", start_step)
                logger.info("Resumed from checkpoint step %d", start_step)
                start_step += 1
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"--resume-from-step {start_step}: checkpoint not found "
                    f"in {config.checkpoint_dir}"
                )
        else:
            latest = find_latest_checkpoint(config.checkpoint_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"--resume requested but no checkpoint found "
                    f"in {config.checkpoint_dir}"
                )
            sb_manager.ensure_agents_initialized()
            metadata = load_checkpoint(
                config.checkpoint_dir, latest,
                adapter_dir=config.adapter_dir,
                bank_agents=sb_manager.get_agents(),
            )
            _log_resumed_bank_sizes(sb_manager, "latest", latest)
            start_step = latest + 1
            logger.info("Resumed from latest checkpoint step %d", latest)

    else:  # auto
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest is not None:
            try:
                sb_manager.ensure_agents_initialized()
                metadata = load_checkpoint(
                    config.checkpoint_dir, latest,
                    adapter_dir=config.adapter_dir,
                    bank_agents=sb_manager.get_agents(),
                )
                _log_resumed_bank_sizes(sb_manager, "auto", latest)
                start_step = latest + 1
                logger.info("Auto-resumed from checkpoint step %d", latest)
            except Exception:
                logger.warning("Auto-resume failed, starting from step 0")
                start_step = 0
        else:
            logger.info("No checkpoint found, starting from step 0")

    # ── Ensure output directories ─────────────────────────────────
    dirs_to_create = [
        config.bank_dir, config.adapter_dir,
        config.decision_adapter_dir, config.skillbank_adapter_dir,
        config.checkpoint_dir,
        config.log_dir, config.grpo_data_dir, config.rewards_dir,
        config.tensorboard_dir,
    ]
    if config.debug_io:
        dirs_to_create.append(config.debug_io_dir)
    for d in dirs_to_create:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── T2.4: single reward sink ──────────────────────────────────
    # One ``RewardLogger`` per orchestrator instance; threaded through
    # every rollout via ``collect_rollouts(..., reward_logger=...)`` so
    # eval and training read from one source. Disable by setting
    # ``CoEvolutionConfig.reward_log_path = None`` (or empty after
    # ``resolve_paths()`` post-process).
    # Teacher-anchored reward normalization (training_notes §4.5):
    # auto-derive per-game anchors from the SFT cold-start
    # ``rollout_summary.json`` files, layered over the static
    # 4-backbone-max fallback table. ``None`` ⇒ "no anchor"; the
    # downstream ``reward_normalized`` field is then ``None`` too,
    # which W&B / dashboards distinguish from a real ``0.0``.
    try:
        from common.reward_anchors import resolve_anchors as _resolve_anchors
        reward_anchors = _resolve_anchors()
        anchored_games = sorted(
            g for g, v in reward_anchors.items() if v is not None
        )
        logger.info(
            "Reward anchors (§4.5): %d games anchored (%s)",
            len(anchored_games), ", ".join(anchored_games),
        )
    except Exception as _anchor_exc:  # pragma: no cover  (defensive)
        logger.warning(
            "Reward anchor resolution failed: %s — normalization will be no-op.",
            _anchor_exc,
        )
        reward_anchors = None

    reward_logger = None
    if config.reward_log_path:
        try:
            from harness.reward_logger import RewardLogger as _RewardLogger
            reward_logger = _RewardLogger(
                log_path=config.reward_log_path,
                reward_anchors=reward_anchors,
            )
            logger.info("Reward sink (T2.4): %s", config.reward_log_path)
        except Exception as exc:
            logger.warning(
                "Reward sink init failed (T2.4): %s — proceeding without "
                "the single-sink mirror; GRPO records still attach reward "
                "inline. Eval will fall back to per-checkpoint scoreboards.",
                exc,
            )
            reward_logger = None

    # ── Initialize TensorBoard ────────────────────────────────────
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=config.tensorboard_dir)
        logger.info("TensorBoard initialized: %s", config.tensorboard_dir)
    except ImportError:
        logger.warning("torch.utils.tensorboard not available, TensorBoard disabled")
    except Exception as exc:
        logger.warning("TensorBoard init failed: %s", exc)

    # ── Persist full config snapshot ──────────────────────────────
    config_path = Path(config.log_dir) / "config.json"
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        logger.info("Config saved: %s", config_path)
    except Exception as exc:
        logger.warning("Config save failed: %s", exc)

    # ── Step history for logging ──────────────────────────────────
    step_log_path = Path(config.log_dir) / "step_log.jsonl"

    # Reviewer-facing JSONL instrumentation streams (block A patches):
    # harness rejections, validate_invocation, lifecycle transitions,
    # intention switches, per-component runtime breakdown.  Disable by
    # setting `config.reviewer_instrumentation_enabled = False`.
    try:
        from trainer.coevolution import _run_loggers as _rl_module  # noqa: WPS433
        if bool(getattr(config, "reviewer_instrumentation_enabled", True)):
            _rl_module.set_run_dir(Path(config.run_dir))
            logger.info(
                "Reviewer instrumentation enabled — writing to %s/{harness_log,"
                "lifecycle_log,intention_log,runtime_log}/",
                config.run_dir,
            )
        else:
            _rl_module.set_run_dir(None)
    except Exception as _rl_exc:  # noqa: BLE001
        logger.debug("Reviewer instrumentation setup skipped: %s", _rl_exc)

    logger.info(
        "Starting co-evolution loop: steps %d→%d, %d games × %d eps/game (default)",
        start_step, config.total_steps - 1,
        len(config.games), config.episodes_per_game,
    )
    _eps_overrides = getattr(config, "episodes_per_game_overrides", {}) or {}
    # Surface only the overrides that *differ* from the global default
    # AND that apply to a currently-active game (cuts noise from the
    # eval-only roster).
    _active_overrides = {
        g: n for g, n in _eps_overrides.items()
        if n != config.episodes_per_game and g in (
            list(config.games) + list(getattr(config, "eval_games", []))
        )
    }
    if _active_overrides:
        logger.info(
            "Per-game episode overrides active: %s",
            ", ".join(f"{g}={n}" for g, n in sorted(_active_overrides.items())),
        )

    # ── Start persistent vLLM instances (once) ─────────────────────
    if vllm_manager:
        logger.info(
            "Starting %d persistent vLLM instances...",
            vllm_manager.n_instances,
        )
        vllm_manager.start()
        healthy = await vllm_manager.wait_healthy()
        if not healthy:
            raise RuntimeError(
                "vLLM instances failed to start — check "
                f"{Path(config.run_dir) / 'vllm_logs'} for details"
            )
        vllm_manager.start_health_monitor()

    # ── Helper: finalize a completed step (deferred until GRPO done) ──

    async def _finalize_step(ctx, grpo_result, phase_c_time):
        """Log metrics, export GRPO data, and checkpoint a completed step.

        All mutable per-step data is pre-captured in *ctx* so this can
        run safely after the next step has begun.
        """
        _step = ctx['step']
        _mode = ctx['mode']
        _phase_ab_time = ctx['phase_ab_time']
        _phase_b_time = ctx['phase_b_time']
        _ep_metrics = ctx['episode_metrics']
        _sb_results = ctx['sb_update_results']
        _new_skills = ctx['total_new_skills']
        _vstats = ctx['vllm_stats']
        _sk_counts = ctx['skill_counts']
        _n_skills = sum(_sk_counts.values())
        # Layer D: cross-domain dashboard metrics. Empty dict on
        # disabled / skipped / fail — the wandb log call below is a
        # no-op for empty inputs.
        _dashboard_metrics: Dict[str, float] = ctx.get('dashboard_metrics') or {}

        if grpo_result:
            try:
                grpo_step_dir = Path(config.grpo_data_dir) / f"step_{_step:04d}"
                grpo_step_dir.mkdir(parents=True, exist_ok=True)
                for _aname, _recs in grpo_result.records.items():
                    _out = grpo_step_dir / f"{_aname}.jsonl"
                    with open(_out, "w", encoding="utf-8") as f:
                        for rec in _recs:
                            f.write(json.dumps(rec, default=str) + "\n")
                logger.debug("GRPO data exported: %s", grpo_step_dir)
            except Exception as exc:
                logger.warning("GRPO data export failed: %s", exc)

        step_elapsed = _phase_ab_time + _phase_b_time + phase_c_time
        per_game_rewards = {
            game: {
                "mean_reward": m["mean_reward"],
                "max_reward": m["max_reward"],
                "min_reward": m["min_reward"],
                "std_reward": m["std_reward"],
                "n_episodes": m["n_episodes"],
                "mean_steps": m["mean_steps"],
            }
            for game, m in _ep_metrics["per_game"].items()
        }

        step_summary = {
            "step": _step,
            "mode": _mode,
            "wall_time_s": step_elapsed,
            "phase_ab_time_s": _phase_ab_time,
            "phase_b_finalize_time_s": _phase_b_time,
            "phase_c_grpo_time_s": phase_c_time,
            "n_episodes": _ep_metrics["aggregate"]["n_episodes"],
            "total_steps_played": _ep_metrics["aggregate"]["total_steps"],
            "mean_reward": _ep_metrics["aggregate"]["mean_reward"],
            "reward_per_game": per_game_rewards,
            "n_skills": _n_skills,
            "n_new_skills": _new_skills,
            "skills_per_game": _sk_counts,
            "vllm_calls": _vstats["call_count"],
            "vllm_prompt_tokens": _vstats["total_prompt_tokens"],
            "vllm_completion_tokens": _vstats["total_completion_tokens"],
        }

        per_game_summary = ", ".join(
            f"{g}={m['mean_reward']:.1f}" for g, m in per_game_rewards.items()
        )
        logger.info(
            "Step %d complete: %.1fs | %d eps | mean_reward=%.2f | "
            "per_game=[%s] | %d skills (+%d) across %d games | %d vLLM calls",
            _step, step_elapsed,
            step_summary["n_episodes"], step_summary["mean_reward"],
            per_game_summary,
            _n_skills, _new_skills,
            sum(1 for c in _sk_counts.values() if c > 0),
            step_summary["vllm_calls"],
        )

        if wandb is not None:
            log_dict = {
                "step": _step,
                "wall_time_s": step_elapsed,
                "phase_ab_time_s": _phase_ab_time,
                "phase_b_finalize_time_s": _phase_b_time,
                "phase_c_grpo_time_s": phase_c_time,
                "mode": 0 if _mode == "cold-start" else 1,
                "n_skills": _n_skills,
                "n_new_skills": _new_skills,
                "vllm/calls": _vstats["call_count"],
                "vllm/prompt_tokens": _vstats["total_prompt_tokens"],
                "vllm/completion_tokens": _vstats["total_completion_tokens"],
            }
            # Per-game raw metrics (legacy ``reward/{game}/...`` namespace
            # is kept *unchanged* for backward-compat with existing W&B
            # dashboards/queries; new ``reward/raw/{game}/...`` mirrors
            # the legacy keys for readers that want the namespaced view).
            #
            # Per-game normalized metrics (training_notes §4.5):
            # ``reward/normalized/{game}/...``. ``None`` (no anchor) is
            # *omitted* from ``log_dict`` rather than logged as a NaN —
            # this preserves the "no anchor ≠ scored zero" semantic.
            try:
                from common.reward_anchors import normalize_reward as _norm
            except Exception:  # pragma: no cover
                _norm = None
            anchored_norms: List[float] = []
            for game, m in _ep_metrics["per_game"].items():
                # Legacy + ``reward/raw/`` namespace (identical values).
                for prefix in (f"reward/{game}", f"reward/raw/{game}"):
                    log_dict[f"{prefix}/mean"] = m["mean_reward"]
                    log_dict[f"{prefix}/max"] = m["max_reward"]
                    log_dict[f"{prefix}/min"] = m["min_reward"]
                    log_dict[f"{prefix}/std"] = m["std_reward"]
                log_dict[f"reward/{game}/n_episodes"] = m["n_episodes"]
                log_dict[f"reward/{game}/mean_steps"] = m["mean_steps"]

                # Normalized namespace — only emitted when an anchor exists.
                if _norm is not None and reward_anchors is not None:
                    n_mean = _norm(m["mean_reward"], game, anchors=reward_anchors)
                    n_max = _norm(m["max_reward"], game, anchors=reward_anchors)
                    n_min = _norm(m["min_reward"], game, anchors=reward_anchors)
                    if n_mean is not None:
                        log_dict[f"reward/normalized/{game}/mean"] = n_mean
                        anchored_norms.append(n_mean)
                    if n_max is not None:
                        log_dict[f"reward/normalized/{game}/max"] = n_max
                    if n_min is not None:
                        log_dict[f"reward/normalized/{game}/min"] = n_min
            log_dict["reward/mean"] = _ep_metrics["aggregate"]["mean_reward"]
            log_dict["reward/max"] = _ep_metrics["aggregate"]["max_reward"]
            log_dict["reward/min"] = _ep_metrics["aggregate"]["min_reward"]
            log_dict["reward/std"] = _ep_metrics["aggregate"]["std_reward"]
            log_dict["reward/total_steps"] = _ep_metrics["aggregate"]["total_steps"]
            # Cross-game normalized aggregate — the headline scalar that
            # is fair to compare across phases. Uses an unweighted mean
            # of per-game normalized means (each game contributes
            # equally regardless of episode count).
            if anchored_norms:
                log_dict["reward/normalized/mean"] = sum(anchored_norms) / len(anchored_norms)
                log_dict["reward/normalized/n_anchored_games"] = len(anchored_norms)
            for game, cnt in _sk_counts.items():
                log_dict[f"skillbank/{game}/n_skills"] = cnt
            for game, result in _sb_results.items():
                for stage, t in result.stage_times.items():
                    log_dict[f"skillbank/{game}/{stage}_time_s"] = t
            if grpo_result:
                for adapter, stats in grpo_result.decision_stats.items():
                    log_dict[f"grpo/decision/{adapter}/loss"] = stats.mean_loss
                    log_dict[f"grpo/decision/{adapter}/n_samples"] = stats.n_samples
                for adapter, stats in grpo_result.skillbank_stats.items():
                    log_dict[f"grpo/skillbank/{adapter}/loss"] = stats.mean_loss
                    log_dict[f"grpo/skillbank/{adapter}/n_samples"] = stats.n_samples
                log_dict["grpo/wall_time_s"] = grpo_result.wall_time_s
                for adapter, game_counts in grpo_result.per_game_counts.items():
                    for game, count in game_counts.items():
                        log_dict[f"grpo/{adapter}/{game}/n_samples"] = count
            # Layer D: fold the cross-domain dashboard metrics into
            # the same wandb step. Empty dict when disabled/skipped.
            for k, v in _dashboard_metrics.items():
                log_dict[k] = v
            try:
                wandb.log(log_dict, step=_step)
            except Exception as exc:
                logger.warning("W&B log failed: %s", exc)

        if tb_writer is not None:
            try:
                tb_writer.add_scalar("timing/wall_time_s", step_elapsed, _step)
                tb_writer.add_scalar("timing/phase_ab_s", _phase_ab_time, _step)
                tb_writer.add_scalar("timing/phase_b_finalize_s", _phase_b_time, _step)
                tb_writer.add_scalar("timing/phase_c_grpo_s", phase_c_time, _step)
                tb_writer.add_scalar("reward/mean", _ep_metrics["aggregate"]["mean_reward"], _step)
                tb_writer.add_scalar("reward/max", _ep_metrics["aggregate"]["max_reward"], _step)
                tb_writer.add_scalar("reward/min", _ep_metrics["aggregate"]["min_reward"], _step)
                tb_writer.add_scalar("reward/std", _ep_metrics["aggregate"]["std_reward"], _step)
                tb_writer.add_scalar("reward/total_steps", _ep_metrics["aggregate"]["total_steps"], _step)
                for game, m in _ep_metrics["per_game"].items():
                    tb_writer.add_scalar(f"reward/{game}/mean", m["mean_reward"], _step)
                    tb_writer.add_scalar(f"reward/{game}/max", m["max_reward"], _step)
                    tb_writer.add_scalar(f"reward/{game}/min", m["min_reward"], _step)
                    tb_writer.add_scalar(f"reward/{game}/std", m["std_reward"], _step)
                tb_writer.add_scalar("skillbank/n_skills", _n_skills, _step)
                tb_writer.add_scalar("skillbank/n_new_skills", _new_skills, _step)
                for game, cnt in _sk_counts.items():
                    tb_writer.add_scalar(f"skillbank/{game}/n_skills", cnt, _step)
                for game, result in _sb_results.items():
                    for stage, t in result.stage_times.items():
                        tb_writer.add_scalar(f"skillbank/{game}/{stage}_time_s", t, _step)
                tb_writer.add_scalar("vllm/calls", _vstats["call_count"], _step)
                tb_writer.add_scalar("vllm/prompt_tokens", _vstats["total_prompt_tokens"], _step)
                tb_writer.add_scalar("vllm/completion_tokens", _vstats["total_completion_tokens"], _step)
                if grpo_result:
                    for adapter, stats in grpo_result.decision_stats.items():
                        tb_writer.add_scalar(f"grpo/decision/{adapter}/loss", stats.mean_loss, _step)
                        tb_writer.add_scalar(f"grpo/decision/{adapter}/n_samples", stats.n_samples, _step)
                    for adapter, stats in grpo_result.skillbank_stats.items():
                        tb_writer.add_scalar(f"grpo/skillbank/{adapter}/loss", stats.mean_loss, _step)
                        tb_writer.add_scalar(f"grpo/skillbank/{adapter}/n_samples", stats.n_samples, _step)
                    tb_writer.add_scalar("grpo/wall_time_s", grpo_result.wall_time_s, _step)
                # Layer D: emit cross-domain dashboard scalars to TB
                # under the same `cross_domain/...` namespace as wandb.
                for k, v in _dashboard_metrics.items():
                    tb_writer.add_scalar(k, v, _step)
                tb_writer.flush()
            except Exception as exc:
                logger.warning("TensorBoard log failed: %s", exc)

        try:
            with open(step_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_summary, default=str) + "\n")
        except Exception:
            pass

        # Block A5: flush per-component runtime aggregates to JSONL.
        # Sources:
        #   * AsyncVLLMClient.snapshot_per_component() — actor traffic.
        #   * record_component_call() aggregator — non-actor 35B traffic
        #     (crafter LLM, promotion judge, harness validator, schema).
        try:
            from trainer.coevolution._run_loggers import (
                log_component_timing,
                flush_component_timings,
                flush_shaping_ratio,
            )
            try:
                actor_snap = vllm_client.snapshot_per_component(reset=True)
            except Exception:  # noqa: BLE001
                actor_snap = {}
            for component, bucket in actor_snap.items():
                log_component_timing(
                    step=_step,
                    component=str(component),
                    n_calls=int(bucket.get("n_calls", 0)),
                    total_ms=float(bucket.get("total_ms", 0.0)),
                    prompt_tokens=int(bucket.get("prompt_tokens", 0)),
                    completion_tokens=int(bucket.get("completion_tokens", 0)),
                )
            flush_component_timings(_step)
            try:
                flush_shaping_ratio(_step)
            except Exception as _shape_exc:  # noqa: BLE001
                logger.debug("shaping_ratio flush failed: %s", _shape_exc)
        except Exception as _rl_exc:  # noqa: BLE001
            logger.debug("component_timing flush failed: %s", _rl_exc)

        should_checkpoint = (
            (_step + 1) % config.checkpoint_interval == 0
            or _step == 0
            or _step == config.total_steps - 1
        )
        if should_checkpoint:
            try:
                ckpt_metadata = {
                    "n_skills": _n_skills,
                    "skills_per_game": _sk_counts,
                    "n_new_skills": _new_skills,
                    "mean_reward": step_summary["mean_reward"],
                    "reward_per_game": per_game_rewards,
                    "n_episodes": step_summary["n_episodes"],
                    "mode": _mode,
                }
                save_checkpoint(
                    config.checkpoint_dir, _step,
                    bank_agents=sb_manager.get_agents(),
                    adapter_dir=config.adapter_dir,
                    metadata=ckpt_metadata,
                )
                _keep = getattr(config, "checkpoint_keep_last", 0)
                if _keep > 0:
                    cleanup_old_checkpoints(config.checkpoint_dir, keep_last=_keep)
                logger.info("Checkpoint saved at step %d", _step)
            except Exception as exc:
                logger.error("Checkpoint save failed: %s", exc)

        nonlocal _best_reward, _best_step, _decline_count
        _cur_reward = step_summary["mean_reward"]
        if _cur_reward > _best_reward:
            _best_reward = _cur_reward
            _best_step = _step
            _decline_count = 0
            _BEST_CKPT_STEP = 99999
            try:
                save_checkpoint(
                    config.checkpoint_dir, _BEST_CKPT_STEP,
                    bank_agents=sb_manager.get_agents(),
                    adapter_dir=config.adapter_dir,
                    metadata={"best": True, "mean_reward": _cur_reward,
                              "original_step": _step},
                )
                logger.info(
                    "New best reward %.1f at step %d — saved best checkpoint",
                    _best_reward, _step,
                )
            except Exception:
                pass
        else:
            _decline_count += 1
            logger.info(
                "Reward %.1f < best %.1f (step %d), decline %d/%d",
                _cur_reward, _best_reward, _best_step,
                _decline_count, _DECLINE_PATIENCE,
            )
            _BEST_CKPT_STEP = 99999
            if _decline_count >= _DECLINE_PATIENCE and _best_step >= 0:
                try:
                    load_checkpoint(
                        config.checkpoint_dir, _BEST_CKPT_STEP,
                        adapter_dir=config.adapter_dir,
                        bank_agents=sb_manager.get_agents(),
                    )
                    logger.info(
                        "Rolled back adapters to best step %d (reward %.1f) "
                        "after %d consecutive declines",
                        _best_step, _best_reward, _decline_count,
                    )
                    _decline_count = 0
                    if vllm_manager:
                        try:
                            await vllm_manager.reload_adapters()
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("Best-checkpoint rollback failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Phase-start GameProfile cache (Path 1).  Populated on first entry
    # to a game and refreshed whenever the curriculum transitions to a
    # game we have not yet seen this run.  Entries are passed through
    # ``collect_rollouts → run_episode_async`` so the actor's
    # SYSTEM_PROMPT and SKILL_SELECTION_SYSTEM_PROMPT can prepend the
    # compact profile.  See ``trainer/coevolution/_game_schema.py``.
    # ──────────────────────────────────────────────────────────────────
    game_profiles: Dict[str, Any] = {}

    async def _refresh_game_schemas(games_to_check: List[str]) -> None:
        if not getattr(config, "game_schema_enabled", False):
            return
        new_games = [g for g in games_to_check if g not in game_profiles]
        if not new_games:
            return
        try:
            from trainer.coevolution._game_schema import ensure_game_schemas
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "game_schema import failed (%s); skipping GameProfile gen",
                exc,
            )
            return
        backbone_judge = (
            getattr(config, "game_schema_model", "")
            or os.environ.get("VLM_AGENT_BACKBONE_JUDGE_MODEL", "")
            or "Qwen/Qwen3.5-35B-A3B"
        )
        logger.info(
            "Phase-start GameProfile: generating for %d game(s) via %s "
            "(timeout=%.0fs, max_tokens=%d)",
            len(new_games), backbone_judge,
            config.game_schema_timeout_s, config.game_schema_max_tokens,
        )
        try:
            fresh = await ensure_game_schemas(
                games=new_games,
                run_dir=config.run_dir,
                model=backbone_judge,
                executor=thread_executor,
                max_tokens=config.game_schema_max_tokens,
                timeout_s=config.game_schema_timeout_s,
            )
            game_profiles.update(fresh)
            llm_count = sum(1 for p in fresh.values() if p.source == "llm")
            cache_count = sum(1 for p in fresh.values() if p.source == "cache")
            fb_count = len(fresh) - llm_count - cache_count
            logger.info(
                "Phase-start GameProfile: %d ready (llm=%d cache=%d "
                "fallback=%d)",
                len(fresh), llm_count, cache_count, fb_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Phase-start GameProfile gen failed (%s); proceeding "
                "without per-game profile injection",
                exc,
            )

    if getattr(config, "game_schema_enabled", False):
        await _refresh_game_schemas(list(config.games))

    # ==================================================================
    # MAIN LOOP — pipelined: GRPO(N) overlaps with rollout(N+1)
    # ==================================================================
    _pending_grpo = None  # (asyncio.Task, step_ctx) or None
    _best_reward = float("-inf")
    _best_step = -1
    _decline_count = 0
    _DECLINE_PATIENCE = getattr(config, "rollback_patience", 4)

    for step in range(start_step, config.total_steps):
        step_t0 = time.monotonic()
        is_cold_start = (step == 0)

        if hasattr(config, "active_games"):
            prev_games = list(config.games)
            config.games = config.active_games(step)
            if config.games != prev_games:
                logger.info(
                    "Curriculum transition at step %d: %s → %s",
                    step, prev_games, config.games,
                )
                await _refresh_game_schemas(list(config.games))

        try:
            from skill_agents.stage3_mvp.schemas import ProtoSkill
            ProtoSkill.set_relaxed(step <= 15)
        except Exception:
            pass

        skill_banks = sb_manager.get_banks()
        skill_counts = sb_manager.skill_counts()
        n_total_skills = sum(skill_counts.values())
        bank_available = n_total_skills > 0
        mode = "cold-start" if is_cold_start or not bank_available else "warm"

        per_game_str = ", ".join(
            f"{g}={c}" for g, c in skill_counts.items() if c > 0
        ) or "empty"
        logger.info(
            "═══ Step %d/%d [%s] ═══ bank=%d skills (%s) ═══",
            step, config.total_steps - 1, mode,
            n_total_skills, per_game_str,
        )

        sb_manager.reset_for_step()
        # T2.7 — refresh the curator warmup ramp once per outer step so
        # the GRPO wrapper scales ``curator_reward`` by the configured
        # weight × min(1, step / warmup_steps). Defaults
        # (curator_warmup_steps=0, curator_weight=1.0) are a no-op,
        # preserving pre-T2.7 behaviour.
        try:
            from skill_agents.bank_maintenance.llm_curator import (
                set_curator_warmup as _set_curator_warmup,
            )
            _set_curator_warmup(
                weight=config.curator_weight,
                warmup_steps=config.curator_warmup_steps,
                current_step=step,
            )
        except Exception as _curator_warmup_exc:  # noqa: BLE001
            logger.warning(
                "T2.7 curator warmup setter failed: %s", _curator_warmup_exc,
            )
        vllm_client.reset_stats()
        if config.debug_io:
            vllm_client.set_io_step(step)

        # ── Build per-game SkillHarnessHook (PLAN-HARNESS §5.2 + §3.4)
        # The hook reads the per-game ``skill_bank.jsonl`` once per step
        # and exposes :meth:`SkillHarnessHook.filter_candidates` /
        # :meth:`SkillHarnessHook.validate_choice` to the episode runner.
        # The aggregated rejection sink is drained by the Phase B′
        # Crafter hook into ``SkillRecord.false_binding_patterns``.
        # Off by default; controlled by ``config.harness_enabled``.
        harness_hooks: Dict[str, Any] = {}
        if getattr(config, "harness_enabled", False):
            try:
                from trainer.coevolution._harness_hook import SkillHarnessHook
                _hook_bank_paths = sb_manager.bank_paths(simple_only=True)
                _allow_shadow = bool(getattr(config, "harness_allow_shadow", True))
                # Path 4 — LLM Harness validator wiring.  All four
                # ``llm_harness_*`` knobs are passive when
                # ``llm_harness_validator_enabled=False`` (default).
                _llm_v_enabled = bool(getattr(
                    config, "llm_harness_validator_enabled", False,
                ))
                _llm_v_model = str(getattr(config, "llm_harness_model", "") or "")
                _llm_v_bootstrap = int(getattr(
                    config, "llm_harness_bootstrap_steps", 20,
                ))
                _llm_v_max_tokens = int(getattr(
                    config, "llm_harness_max_tokens", 256,
                ))
                _llm_v_temperature = float(getattr(
                    config, "llm_harness_temperature", 0.2,
                ))
                _llm_v_timeout = float(getattr(
                    config, "llm_harness_timeout_s", 30.0,
                ))
                _harness_mode = str(getattr(config, "harness_mode", "full") or "full")
                for _g, _bp in _hook_bank_paths.items():
                    try:
                        harness_hooks[_g] = SkillHarnessHook.for_game(
                            game=_g,
                            bank_path=Path(_bp),
                            allow_shadow=_allow_shadow,
                            llm_validator_enabled=_llm_v_enabled,
                            llm_validator_model=_llm_v_model,
                            trainer_step=step,
                            bootstrap_steps=_llm_v_bootstrap,
                            llm_validator_max_tokens=_llm_v_max_tokens,
                            llm_validator_temperature=_llm_v_temperature,
                            llm_validator_timeout_s=_llm_v_timeout,
                            game_profile=(game_profiles or {}).get(_g),
                            mode=_harness_mode,
                        )
                    except Exception as _hexc:                  # noqa: BLE001
                        logger.warning(
                            "harness_hook build failed for game=%s: %s",
                            _g, _hexc,
                        )
                logger.info(
                    "Phase A: harness enabled — %d hook(s) built (%s)",
                    len(harness_hooks),
                    ", ".join(
                        f"{g}={h.n_records()}" for g, h in harness_hooks.items()
                    ) or "empty",
                )
            except Exception as _hexc:                          # noqa: BLE001
                logger.warning(
                    "Phase A: harness setup failed at step=%d: %s — "
                    "falling back to harness-disabled rollout",
                    step, _hexc,
                )
                harness_hooks = {}

        # ── Phase A + B: Rollout collection with cross-system overlap ──
        phase_ab_t0 = time.monotonic()

        completed_queue: asyncio.Queue[EpisodeResult] = asyncio.Queue()

        def on_episode_done(result: EpisodeResult) -> None:
            completed_queue.put_nowait(result)

        async def skill_bank_consumer() -> int:
            """Consume completed episodes and feed to skill bank pipeline.

            Greedily drains the queue after each blocking wait so that
            episodes which completed during segmentation are batched together
            instead of being dequeued one-by-one with 5 s gaps.
            """
            n_processed = 0
            batch: List[EpisodeResult] = []
            sentinel_seen = False
            while not sentinel_seen:
                # Block until at least one item arrives (or timeout)
                try:
                    result = await asyncio.wait_for(
                        completed_queue.get(), timeout=5.0,
                    )
                    if result.game == "__SENTINEL__":
                        sentinel_seen = True
                    elif result.steps > 0:
                        batch.append(result)
                except asyncio.TimeoutError:
                    pass

                # Drain all remaining items without waiting
                while not completed_queue.empty():
                    try:
                        result = completed_queue.get_nowait()
                        if result.game == "__SENTINEL__":
                            sentinel_seen = True
                        elif result.steps > 0:
                            batch.append(result)
                    except asyncio.QueueEmpty:
                        break

                # Process when batch is big enough or we're done
                if len(batch) >= config.em_micro_batch_size or (sentinel_seen and batch):
                    logger.info(
                        "Skill bank consumer: processing %d episodes "
                        "(total so far: %d)",
                        len(batch), n_processed + len(batch),
                    )
                    await sb_manager.process_batch_async(batch)
                    n_processed += len(batch)
                    batch = []

            # Flush any remaining partial batch
            if batch:
                logger.info(
                    "Skill bank consumer: flushing final %d episodes "
                    "(total: %d)",
                    len(batch), n_processed + len(batch),
                )
                await sb_manager.process_batch_async(batch)
                n_processed += len(batch)

            logger.info("Skill bank consumer finished: %d episodes processed", n_processed)
            return n_processed

        rollout_task = asyncio.create_task(
            collect_rollouts(
                config, vllm_client,
                skill_banks=skill_banks if bank_available else None,
                on_episode_done=on_episode_done,
                thread_executor=thread_executor,
                harness_hooks=harness_hooks if harness_hooks else None,
                reward_logger=reward_logger,
                game_profiles=game_profiles if game_profiles else None,
            )
        )
        consumer_task = asyncio.create_task(skill_bank_consumer())

        # ── Await previous step's GRPO (concurrent with rollout above) ──
        _prev_grpo_result = None
        _prev_phase_c_time = 0.0
        if _pending_grpo is not None:
            _prev_task, _prev_ctx = _pending_grpo
            try:
                _prev_grpo_result = await _prev_task
            except Exception as exc:
                _exc_name = type(exc).__name__
                _is_oom = "OutOfMemory" in _exc_name or "CUDA out of memory" in str(exc)
                logger.error(
                    "GRPO training failed (step %d, %s%s): %s",
                    _prev_ctx['step'], _exc_name,
                    " — CUDA OOM, consider reducing batch_size" if _is_oom else "",
                    exc, exc_info=True,
                )
                try:
                    import gc as _gc
                    _gc.collect()
                    import torch as _torch
                    if _torch.cuda.is_available():
                        for _i in range(_torch.cuda.device_count()):
                            with _torch.cuda.device(_i):
                                _torch.cuda.empty_cache()
                except Exception:
                    pass
            _prev_phase_c_time = time.monotonic() - _prev_ctx['phase_c_t0']
            logger.info(
                "Phase C (GRPO, step %d): %.1fs",
                _prev_ctx['step'], _prev_phase_c_time,
            )

        rollout_results: List[EpisodeResult] = await rollout_task

        # Signal consumer to drain remaining
        completed_queue.put_nowait(
            EpisodeResult(game="__SENTINEL__", episode_id="__SENTINEL__", steps=0)
        )
        n_consumed = await consumer_task

        phase_ab_time = time.monotonic() - phase_ab_t0
        logger.info(
            "Phase A+B: %.1fs, %d episodes collected, %d consumed by skill bank",
            phase_ab_time, len(rollout_results), n_consumed,
        )

        # ── Finalize previous step (hot-reload + metrics + checkpoint) ──
        if _pending_grpo is not None:
            _, _prev_ctx = _pending_grpo
            if vllm_manager and _prev_grpo_result:
                try:
                    await vllm_manager.reload_adapters()
                except Exception as exc:
                    logger.warning("Adapter hot-reload failed: %s", exc)
            await _finalize_step(
                _prev_ctx, _prev_grpo_result, _prev_phase_c_time,
            )
            _pending_grpo = None

        # ── Export per-episode rewards ────────────────────────────────
        try:
            rewards_path = Path(config.rewards_dir) / f"step_{step:04d}.jsonl"
            with open(rewards_path, "w", encoding="utf-8") as f:
                for ep in rollout_results:
                    if ep.game == "__SENTINEL__":
                        continue
                    record = {
                        "game": ep.game,
                        "episode_id": ep.episode_id,
                        "steps": ep.steps,
                        "total_reward": ep.total_reward,
                        "terminated": ep.terminated,
                        "eval_only": ep.eval_only,
                        "wall_time_s": ep.wall_time_s,
                    }
                    f.write(json.dumps(record, default=str) + "\n")
            logger.debug("Rewards exported: %s", rewards_path)
        except Exception as exc:
            logger.warning("Rewards export failed: %s", exc)

        # ── Phase B: Finalize all per-game skill bank updates ────────
        phase_b_t0 = time.monotonic()
        sb_update_results: Dict[str, SkillBankUpdateResult] = {}
        try:
            sb_update_results = await sb_manager.finalize_all()
        except Exception as exc:
            logger.error("Skill bank finalize failed: %s", exc)

        total_new_skills = sum(
            r.n_new_skills for r in sb_update_results.values()
        )

        phase_b_time = time.monotonic() - phase_b_t0
        logger.info("Phase B finalize: %.1fs (%d game banks)", phase_b_time, len(sb_update_results))

        # ── Phase B′: Crafter + Promotion (one-way write-back) ───────
        # Off by default; controlled by `config.crafter_promotion_enabled`.
        # Spec: implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md §3
        # The hook runs *alongside* the legacy Stage-4 curator: legacy
        # mining still writes per-game `skill_bank.jsonl`, this hook just
        # appends promoted skills on top via `skill_bank.legacy_writeback`.
        crafter_report: Optional[Dict[str, Any]] = None
        promotion_report: Optional[Dict[str, Any]] = None
        # Block B2 — when ``crafter_enabled=False`` the entire crafter
        # phase is skipped.  Promotion/lifecycle still get a chance to
        # run, but with zero new proposals to evaluate.  This isolates
        # the crafter's contribution to the §5.5 ablation table.
        _crafter_enabled = bool(getattr(config, "crafter_enabled", True))
        if config.crafter_promotion_enabled and _crafter_enabled:
            phase_bp_t0 = time.monotonic()
            try:
                from trainer.coevolution._crafter_hook import run_crafter_step
                from trainer.coevolution._promotion_hook import run_promotion_step

                bank_paths = sb_manager.bank_paths(simple_only=True)
                # ``bank_was_available`` gates the NO_SKILL_BOUND F2 signal —
                # we only want it to fire when the actor *could* have bound
                # a skill but didn't.  Just gating on ``step > 0`` is wrong
                # whenever the run was launched with ``--seed-bank-dir`` (or
                # otherwise resumes from a pre-populated bank dir) because
                # the actor on step 0 has skills available.  Probe the
                # on-disk file directly: if *any* per-game bank has ≥1
                # byte, there were skills to bind.
                bank_was_available = any(
                    p.is_file() and p.stat().st_size > 0
                    for p in bank_paths.values()
                )
                crafter_step = run_crafter_step(
                    step=step,
                    run_dir=Path(config.run_dir),
                    rollout_results=rollout_results,
                    legacy_bank_paths=bank_paths,
                    bank_was_available=bank_was_available,
                    cycle_every_k_steps=config.crafter_cycle_every_k_steps,
                    outcome_failure_threshold=config.crafter_outcome_failure_threshold,
                    harness_hooks=harness_hooks if harness_hooks else None,
                    enable_protocol_patching=config.crafter_enable_protocol_patching,
                    # Path 2 — supplemental LLM Crafter (35B) wiring.
                    llm_crafter_enabled=getattr(
                        config, "llm_crafter_enabled", False,
                    ),
                    llm_crafter_model=getattr(
                        config, "llm_crafter_model", "",
                    ),
                    llm_crafter_k_max=getattr(
                        config, "llm_crafter_k_max", 5,
                    ),
                    llm_crafter_max_tokens=getattr(
                        config, "llm_crafter_max_tokens", 1024,
                    ),
                    llm_crafter_temperature=getattr(
                        config, "llm_crafter_temperature", 0.3,
                    ),
                    llm_crafter_timeout_s=getattr(
                        config, "llm_crafter_timeout_s", 60.0,
                    ),
                    # Stage 2 cross-domain knob — Stage 1 keeps False.
                    llm_crafter_enable_thinking=getattr(
                        config, "llm_crafter_enable_thinking", False,
                    ),
                    game_profiles=game_profiles if game_profiles else None,
                    # Hypothesizer fallthrough gate (post-v11 audit).
                    # Defaults to (3, 0.30) — i.e. fire only on
                    # recurring failures with no related bank skill —
                    # so the bank doesn't fill up with placeholder
                    # ``hypothesis__prop-...`` records the actor never
                    # selects.  Override via CoEvolutionConfig fields
                    # (added with the same getattr fallback in case an
                    # older config object is in use).
                    hypothesize_min_recurrences=getattr(
                        config, "crafter_hypothesize_min_recurrences", 3,
                    ),
                    hypothesize_related_skill_jaccard=getattr(
                        config, "crafter_hypothesize_related_skill_jaccard", 0.30,
                    ),
                )
                crafter_report = crafter_step.to_dict()

                if crafter_step.n_proposals > 0:
                    promotion_step = run_promotion_step(
                        step=step,
                        run_dir=Path(config.run_dir),
                        proposals_run_dir=crafter_step.run_dir,
                        legacy_bank_paths=bank_paths,
                        driver_timeout_s=config.crafter_promotion_timeout_s,
                        gate_mode=getattr(
                            config,
                            "crafter_promotion_gate_mode",
                            "offline-synthetic",
                        ),
                        # Stage 2 cross-domain knobs — only consulted
                        # when ``gate_mode == "offline-with-llm-judge"``;
                        # for any other mode the driver ignores them.
                        judge_enable_thinking=getattr(
                            config,
                            "crafter_promotion_judge_enable_thinking",
                            False,
                        ),
                        judge_max_tokens=getattr(
                            config,
                            "crafter_promotion_judge_max_tokens",
                            256,
                        ),
                        # Block B3 — w/o lifecycle gating ablation.
                        bypass_mode=getattr(
                            config, "promotion_bypass_mode", "gated",
                        ),
                    )
                    promotion_report = promotion_step.to_dict()

                    # ── Layer A: cross-domain transfer gate ──────────
                    # Re-evaluates each just-promoted skill's admit rate
                    # against the configured transfer targets. Skills
                    # failing crafter_transfer_admit_band[0] on every
                    # target are rolled back (dropped from the per-game
                    # skill_bank.jsonl) before the bank reload below.
                    # Off by default; conservative on any failure
                    # (subprocess crash / timeout / parse error ⇒ no
                    # demotions, full report still surfaced).
                    transfer_gate_report: Optional[Dict[str, Any]] = None
                    if (
                        config.crafter_transfer_gate_enabled
                        and config.crafter_transfer_targets
                        and not promotion_step.skipped
                        and promotion_step.driver_returncode == 0
                    ):
                        try:
                            from trainer.coevolution._transfer_hook import (
                                run_transfer_gate_step,
                            )
                            transfer_step = run_transfer_gate_step(
                                step=step,
                                run_dir=Path(config.run_dir),
                                promotion_writeback_per_game=(
                                    promotion_step.writeback_per_game or {}
                                ),
                                legacy_bank_paths=bank_paths,
                                transfer_targets=tuple(
                                    config.crafter_transfer_targets,
                                ),
                                transfer_admit_band=tuple(
                                    config.crafter_transfer_admit_band,
                                ),
                                transfer_min_targets_in_band=(
                                    config.crafter_transfer_min_targets_in_band
                                ),
                                transfer_max_skills_per_cell=(
                                    config.crafter_transfer_max_skills_per_cell
                                ),
                                transfer_driver_timeout_s=(
                                    config.crafter_transfer_timeout_s
                                ),
                            )
                            transfer_gate_report = transfer_step.to_dict()
                            if transfer_step.n_demote > 0:
                                logger.info(
                                    "Phase B′ transfer gate: %d/%d skills "
                                    "demoted (cross-domain admit floor)",
                                    transfer_step.n_demote,
                                    transfer_step.n_skills_in,
                                )
                            elif transfer_step.skipped:
                                logger.info(
                                    "Phase B′ transfer gate: skipped (%s)",
                                    transfer_step.skipped_reason,
                                )
                            if promotion_report is not None:
                                promotion_report["transfer_gate"] = (
                                    transfer_gate_report
                                )
                        except Exception as exc:  # noqa: BLE001
                            logger.error(
                                "Phase B′ transfer gate failed at step=%d: %s",
                                step, exc, exc_info=True,
                            )

                    # Critical: hot-reload the in-memory bank for any game
                    # whose on-disk skill_bank.jsonl was just mutated by
                    # legacy_writeback OR by the transfer gate's demotion
                    # rollback. Without this, AsyncSkillBankPipeline
                    # keeps serving the cached SkillBankMVP._skills from
                    # before the writeback (and its cached SkillQueryEngine
                    # never re-indexes), so the actor's *next* rollout
                    # would not observe the promoted skills. See
                    # implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md
                    # §"actor read path" for the trace.
                    keys_to_reload = [
                        game for game, wb in (
                            promotion_step.writeback_per_game or {}
                        ).items()
                        if int(wb.get("n_inserted", 0)) > 0
                        or int(wb.get("n_updated", 0)) > 0
                    ]
                    reload_report: Dict[str, Any] = {}
                    if keys_to_reload:
                        reload_report = sb_manager.reload_banks_from_disk(
                            keys_to_reload,
                        )
                    if promotion_report is not None:
                        promotion_report["bank_reload_per_game"] = reload_report
                    # Surface driver subprocess failures loudly — without
                    # this branch a non-zero returncode looks identical to
                    # a clean "no eligible promotions" outcome in the log.
                    if promotion_step.skipped or promotion_step.driver_returncode != 0:
                        logger.warning(
                            "Phase B′ Promotion driver did not complete "
                            "cleanly at step=%d: returncode=%d, skipped=%s, "
                            "reason=%r — see %s/_step_summary.json",
                            step,
                            promotion_step.driver_returncode,
                            promotion_step.skipped,
                            promotion_step.skipped_reason,
                            promotion_step.promotion_run_dir,
                        )
                    else:
                        logger.info(
                            "Phase B′ Crafter+Promotion: %d proposals → "
                            "PROMOTE=%d REJECT=%d (writeback +%d skills, "
                            "reloaded %d bank(s))",
                            crafter_step.n_proposals,
                            promotion_step.n_promote, promotion_step.n_reject,
                            promotion_step.n_writeback_inserted,
                            sum(1 for r in reload_report.values()
                                if r.get("reloaded")),
                        )
                else:
                    logger.info(
                        "Phase B′ Crafter: %d episodes reflected, no proposals → "
                        "promotion skipped",
                        crafter_step.n_episodes_reflected,
                    )
            except Exception as exc:                          # noqa: BLE001
                logger.error(
                    "Phase B′ Crafter+Promotion failed at step=%d: %s",
                    step, exc, exc_info=True,
                )
            phase_bp_time = time.monotonic() - phase_bp_t0
        else:
            phase_bp_time = 0.0

        # ── Layer D: cross-domain dashboard (periodic) ──────────────
        # Off by default; cadence-controlled via
        # `crafter_dashboard_every_k_steps`. Snapshots the trainer's
        # banks and runs the full Stage-6 N×N transfer matrix to
        # surface G1-G5 acceptance gates + per-cluster admit rates
        # as wandb / TB scalars. Decoupled from Layer A so the
        # dashboard cadence can be much lower than the per-step
        # transfer gate (a single sweep is ~1-30 minutes).
        dashboard_report: Optional[Dict[str, Any]] = None
        dashboard_metrics: Dict[str, float] = {}
        if config.crafter_dashboard_enabled:
            try:
                from trainer.coevolution._dashboard_hook import (
                    run_dashboard_step,
                    should_run_dashboard,
                )
                if should_run_dashboard(
                    step=step,
                    every_k_steps=config.crafter_dashboard_every_k_steps,
                    enabled=config.crafter_dashboard_enabled,
                ):
                    dashboard_step = run_dashboard_step(
                        step=step,
                        run_dir=Path(config.run_dir),
                        legacy_bank_paths=sb_manager.bank_paths(simple_only=True),
                        dashboard_targets=tuple(
                            config.crafter_dashboard_targets,
                        ),
                        dashboard_max_skills_per_cell=(
                            config.crafter_dashboard_max_skills_per_cell
                        ),
                        dashboard_driver_timeout_s=(
                            config.crafter_dashboard_timeout_s
                        ),
                    )
                    dashboard_report = dashboard_step.to_dict()
                    dashboard_metrics = dashboard_step.to_metrics()
                    if dashboard_step.skipped:
                        logger.info(
                            "Cross-domain dashboard at step=%d: skipped (%s)",
                            step, dashboard_step.skipped_reason,
                        )
                    else:
                        logger.info(
                            "Cross-domain dashboard at step=%d: %d cells, "
                            "mean admit=%.1f%%, gates=%s, %.1fs",
                            step,
                            dashboard_step.n_cells_evaluated,
                            dashboard_step.mean_admit_rate * 100,
                            dashboard_step.gate_verdicts,
                            dashboard_step.wall_time_s,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Cross-domain dashboard failed at step=%d: %s",
                    step, exc, exc_info=True,
                )

        # ── Snapshot step context for deferred finalization ─────────
        step_ctx = {
            'step': step,
            'mode': mode,
            'phase_ab_time': phase_ab_time,
            'phase_b_time': phase_b_time,
            'phase_bp_time': phase_bp_time,
            'crafter_report': crafter_report,
            'promotion_report': promotion_report,
            'dashboard_report': dashboard_report,
            'dashboard_metrics': dashboard_metrics,
            'episode_metrics': compute_episode_metrics(rollout_results),
            'sb_update_results': sb_update_results,
            'total_new_skills': total_new_skills,
            'vllm_stats': vllm_client.stats(),
            'skill_counts': sb_manager.skill_counts(),
        }

        # ── Phase C: Launch GRPO in background (overlaps with next rollout) ──
        if config.grpo_enabled:
            step_ctx['phase_c_t0'] = time.monotonic()
            _pending_grpo = (
                asyncio.create_task(
                    run_grpo_training(
                        rollout_results,
                        sb_manager.grpo_data,
                        config,
                        step=step,
                        executor=thread_executor,
                    )
                ),
                step_ctx,
            )
            logger.info("Phase C: GRPO launched in background for step %d", step)
        else:
            await _finalize_step(step_ctx, None, 0.0)

    # ── Finalize last step ────────────────────────────────────────
    if _pending_grpo is not None:
        _last_task, _last_ctx = _pending_grpo
        _last_result = None
        _last_phase_c_time = 0.0
        try:
            _last_result = await _last_task
        except Exception as exc:
            logger.error("Final GRPO training failed: %s", exc, exc_info=True)
            try:
                import gc as _gc
                _gc.collect()
                import torch as _torch
                if _torch.cuda.is_available():
                    for _i in range(_torch.cuda.device_count()):
                        with _torch.cuda.device(_i):
                            _torch.cuda.empty_cache()
            except Exception:
                pass
        _last_phase_c_time = time.monotonic() - _last_ctx['phase_c_t0']
        logger.info(
            "Phase C (GRPO, step %d): %.1fs",
            _last_ctx['step'], _last_phase_c_time,
        )
        if vllm_manager and _last_result:
            try:
                await vllm_manager.reload_adapters()
            except Exception as exc:
                logger.warning("Adapter hot-reload failed: %s", exc)
        await _finalize_step(_last_ctx, _last_result, _last_phase_c_time)
        _pending_grpo = None

    # ── Cleanup ───────────────────────────────────────────────────
    if vllm_manager:
        vllm_manager.stop()

    thread_executor.shutdown(wait=False)

    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception:
            pass

    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    logger.info(
        "Co-evolution loop complete: %d steps",
        config.total_steps - start_step,
    )
