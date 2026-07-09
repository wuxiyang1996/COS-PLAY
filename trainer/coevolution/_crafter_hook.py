"""Per-step Crafter hook for the trainer's co-evolution loop.

Splices into ``trainer/coevolution/orchestrator.py::co_evolution_loop``
between ``sb_manager.finalize_all()`` and the ``_pending_grpo`` setup.
For every (game, episode) tuple in the just-finished step:

  1. Hydrate an ephemeral ``SkillRepository`` from the per-game
     ``<bank_dir>/<game>/skill_bank.jsonl`` that the trainer's actor
     reads (the legacy 4-stage pipeline produces this file).
  2. Synthesize ``FailureTrace`` records from the ``EpisodeResult.experiences``
     dicts using a *narrow* heuristic — see "F2 — failure synthesis" below.
  3. Wrap them in an ``EpisodeReflection`` and call
     :meth:`crafter.service.SkillCrafterService.reflect_on_episode`.
  4. Optionally run the per-batch :meth:`SkillCrafterService.cycle` once
     per K steps over the accumulated failures.
  5. Translate the resulting live ``BankMutationProposal`` objects into
     the *offline-mirror JSONL row* schema and write
     ``<step_dir>/<corpus>/<source>/proposals.jsonl`` so the next stage
     (the Promotion hook, which subprocess-invokes
     ``decide_promotion_gpt54.py``) can read them via the documented
     ``--proposals-run`` CLI surface.

Strict trainer-mode contract
----------------------------
* No live env. No live LLM call. Pure JSON-in / JSON-out + the
  shipped ``crafter.service.SkillCrafterService`` (Phase-1 rule-based path).
* No import of ``skill_agents`` (preserves D8 Option A).
* No mutation of the input legacy bank — that's the Promotion hook's job
  via :func:`skill_bank.legacy_writeback.writeback_promotion`.

F2 — failure synthesis (narrower than the offline mirror's 4-signal set)
------------------------------------------------------------------------
The trainer's :class:`trainer.coevolution.episode_runner.EpisodeResult`
carries a per-step ``experiences: List[Dict[str, Any]]`` whose keys are
limited to ``{step, state, action, reward, raw_env_reward, next_state,
done, intention, summary_state, skill_id}`` plus an optional
``board_stats``. It does *not* yet carry ``skill_query.empty``,
``skills.applicability``, or ``skills.missing_effects`` — those would
require a live Harness emitting ``SkillEpisode`` records, which is out
of scope per the implementation note ("F2: ``EpisodeReflection.skill_episodes
= []`` because no Harness emits them").

So this module synthesizes *only* the two signals the trainer's
experiences dict actually supports:

* **OUTCOME_FAILURE** — episode-level: ``total_reward <= threshold``.
  Maps to ``FailureTrace(failure_class="INVARIANT_VIOLATION")``.
* **NO_SKILL_BOUND** — per-step: the bank was non-empty going into
  this episode but the actor's skill_selection LoRA returned no
  ``skill_id``. Maps to ``FailureTrace(failure_class="MISSING_ADAPTER",
  abort_reason="no_skill_bound")``. This is a loose proxy for the
  offline mirror's ``EMPTY_QUERY`` signal.

Cross-refs
----------
* `implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md` §3
  (D8 Option A, F2 limitation, alongside-Stage-4 posture).
* `crafter-harness-orchestrator-roles.md` §2.1 (Crafter input contract),
  §6.3 ("No driver imports another driver's code").
* `labeling_supplement/decide_promotion_gpt54.py::_OfflineProposal` /
  ``_translate_proposal`` — the consumer schema this hook writes to.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from common.enums import SkillSourceType, SkillStatus, SkillType
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    ComposeProposal,
    GeneralizeProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from skill_bank.lifecycle import SkillLifecycleManager
from skill_bank.repository import SkillRepository
from skill_bank.stores import SkillStore, StoreName

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — picked to match the offline mirror's defaults so that running
# on the same data produces a comparable proposal stream.
# ---------------------------------------------------------------------------

DEFAULT_OUTCOME_FAILURE_THRESHOLD: float = 0.0
DEFAULT_MAX_FAILURES_PER_EPISODE: int = 8
DEFAULT_HOT_PATTERN_THRESHOLD: int = 3
DEFAULT_COOLDOWN_PASSES: int = 5
# Hypothesizer fallthrough gate (post-v11 audit).  Both gates also live
# on ``crafter.service.SkillCrafterService`` so anyone instantiating the
# service directly inherits the same defaults; we plumb them through
# ``run_crafter_step`` so the orchestrator (and the launch-script CLI)
# can tune them without poking at the inner constructor.
DEFAULT_HYPOTHESIZE_MIN_RECURRENCES: int = 3
DEFAULT_HYPOTHESIZE_RELATED_JACCARD: float = 0.30

# Path 2 — supplemental LLM Crafter (35B-A3B teacher).  When enabled,
# at most ``DEFAULT_LLM_CRAFTER_K_MAX`` failure traces *per game per
# step* get one 35B proposal call each, in parallel.  Proposals are
# concatenated *after* the deterministic ones so the JSONL row order
# mirrors the rule-based-then-LLM precedence the gate stack expects.
DEFAULT_LLM_CRAFTER_K_MAX: int = 5
DEFAULT_LLM_CRAFTER_MAX_TOKENS: int = 1024
DEFAULT_LLM_CRAFTER_TEMPERATURE: float = 0.3
DEFAULT_LLM_CRAFTER_TIMEOUT_S: float = 60.0

# All five domains every BankMutationProposal must declare — PLAN-SKILL-CRAFTER
# §0.1 / §2.5. Mirrored from labeling_supplement/decide_skill_crafting_gpt54.py
# verbatim so a writeback round-trip is consistent.
ALL_FIVE_DOMAINS: Tuple[str, ...] = (
    "gymv", "browser", "osworld", "video", "visual_reasoning",
)

# Trainer game-name → offline-mirror corpus bucket.  This is the same split
# decide_skill_crafting_gpt54.py walks under ``--proposals-run``.
_GYMV_PREFIX_RE = re.compile(r"^Temporal_.*-v0$")


def corpus_for_game(game: str) -> str:
    """Map a trainer game name to the offline-mirror corpus name.

    * ``Temporal_*-v0`` → ``gym_v`` (the 13 retro envs).
    * Everything else (tetris, twenty_forty_eight, candy_crush,
      super_mario, …) → ``env_wrappers``.
    """
    return "gym_v" if _GYMV_PREFIX_RE.match(game) else "env_wrappers"


# ---------------------------------------------------------------------------
# Hook surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrafterStepReport:
    """What the Crafter hook produced for one trainer step."""

    step: int
    run_dir: Path
    n_episodes_reflected: int
    n_failure_traces: int
    n_proposals: int
    proposals_per_game: Dict[str, int] = field(default_factory=dict)
    proposals_jsonl_paths: Dict[str, Path] = field(default_factory=dict)
    n_patches_coalesced: int = 0
    n_patches_skipped_cooldown: int = 0
    cycle_ran: bool = False                # True iff per-batch cycle() fired
    n_cycle_proposals: int = 0
    # Path 2 — LLM Crafter rollups (zero when --llm-crafter-enabled off).
    n_llm_proposals: int = 0
    n_llm_calls_attempted: int = 0
    n_llm_calls_succeeded: int = 0
    n_llm_calls_failed: int = 0
    llm_proposals_per_kind: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "run_dir": str(self.run_dir),
            "n_episodes_reflected": self.n_episodes_reflected,
            "n_failure_traces": self.n_failure_traces,
            "n_proposals": self.n_proposals,
            "proposals_per_game": dict(self.proposals_per_game),
            "proposals_jsonl_paths": {
                k: str(v) for k, v in self.proposals_jsonl_paths.items()
            },
            "n_patches_coalesced": self.n_patches_coalesced,
            "n_patches_skipped_cooldown": self.n_patches_skipped_cooldown,
            "cycle_ran": self.cycle_ran,
            "n_cycle_proposals": self.n_cycle_proposals,
            "n_llm_proposals": self.n_llm_proposals,
            "n_llm_calls_attempted": self.n_llm_calls_attempted,
            "n_llm_calls_succeeded": self.n_llm_calls_succeeded,
            "n_llm_calls_failed": self.n_llm_calls_failed,
            "llm_proposals_per_kind": dict(self.llm_proposals_per_kind),
            "wall_time_s": self.wall_time_s,
        }


# Trainer's EpisodeResult lives in a sibling module; we accept any object
# with the required attributes (so this hook is unit-testable without
# importing episode_runner's heavy deps).
class _EpisodeResultLike:                                    # pragma: no cover
    game: str
    episode_id: str
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    experiences: List[Dict[str, Any]]


def run_crafter_step(
    *,
    step: int,
    run_dir: Path,
    rollout_results: Sequence[Any],                # _EpisodeResultLike per game
    legacy_bank_paths: Mapping[str, Path],         # game → skill_bank.jsonl
    bank_was_available: bool,
    cycle_every_k_steps: int = 0,                  # 0 = never run cycle()
    outcome_failure_threshold: float = DEFAULT_OUTCOME_FAILURE_THRESHOLD,
    max_failures_per_episode: int = DEFAULT_MAX_FAILURES_PER_EPISODE,
    hot_pattern_threshold: int = DEFAULT_HOT_PATTERN_THRESHOLD,
    cooldown_passes: int = DEFAULT_COOLDOWN_PASSES,
    teacher_model: Optional[str] = None,
    harness_hooks: Optional[Mapping[str, Any]] = None,
    enable_protocol_patching: bool = False,
    # Path 2 — supplemental LLM Crafter (35B). Off by default; enable
    # via ``llm_crafter_enabled=True`` (and a non-empty ``llm_crafter_model``
    # if you want to override the default ``BACKBONE_JUDGE_MODEL`` route).
    # When enabled, after the deterministic Crafter has emitted its
    # rule-based proposals, up to ``llm_crafter_k_max`` *additional*
    # proposals are minted per game by asking the 35B teacher to
    # respond to one ``FailureTrace`` per call.  See
    # ``trainer/coevolution/_llm_crafter.py``.
    llm_crafter_enabled: bool = False,
    llm_crafter_model: str = "",
    llm_crafter_k_max: int = DEFAULT_LLM_CRAFTER_K_MAX,
    llm_crafter_max_tokens: int = DEFAULT_LLM_CRAFTER_MAX_TOKENS,
    llm_crafter_temperature: float = DEFAULT_LLM_CRAFTER_TEMPERATURE,
    llm_crafter_timeout_s: float = DEFAULT_LLM_CRAFTER_TIMEOUT_S,
    # Stage 2 (cross-domain adaptation) opt-in: when ``True`` we
    # forward ``enable_thinking=True`` into ``API_func.ask_vllm`` so
    # Qwen3-A3B emits its ``<think>`` chain-of-thought.  Caller must
    # also bump ``llm_crafter_max_tokens`` to ≥ 4096 and
    # ``llm_crafter_timeout_s`` to ≥ 180 because the ``<think>`` block
    # routinely consumes 1-3K tokens and adds 5-10× wall time.
    # Stage 1 in-domain training keeps it ``False`` (current behaviour).
    llm_crafter_enable_thinking: bool = False,
    game_profiles: Optional[Mapping[str, Any]] = None,
    # Hypothesizer fallthrough gate (post-v11 audit).  The dispatcher
    # only mints a HypothesisProposal when (a) the same failure pattern
    # has recurred at least N times in the current FailureMemory window
    # AND (b) no existing bank skill plausibly covers the failure
    # context (token Jaccard < threshold against active+candidate
    # skills' name/description). Passing ``hypothesize_min_recurrences=1``
    # AND ``hypothesize_related_skill_jaccard=0.0`` reproduces the v11
    # behaviour (gates open) — used by integration tests that exercise
    # the dispatch routing in isolation.
    hypothesize_min_recurrences: int = DEFAULT_HYPOTHESIZE_MIN_RECURRENCES,
    hypothesize_related_skill_jaccard: float = DEFAULT_HYPOTHESIZE_RELATED_JACCARD,
) -> CrafterStepReport:
    """Run the per-step Crafter pass for one trainer step.

    Parameters
    ----------
    step
        Current trainer step index (0-based). Used for output dir naming.
    run_dir
        Trainer's run root (e.g. ``CoEvolutionConfig.run_dir``). Output
        JSONL goes under ``<run_dir>/crafter_proposals_out/step_<step>/<corpus>/<source>/``.
    rollout_results
        The list of ``EpisodeResult`` records the orchestrator collected
        in Phase A+B. Sentinel entries (``game == "__SENTINEL__"``) are
        skipped silently.
    legacy_bank_paths
        Map from trainer game name to the per-game ``skill_bank.jsonl``
        the trainer's actor reads. The hook hydrates an ephemeral
        ``SkillRepository`` from these so the live ``SkillCrafterService``
        has skills to repair / patch.
    bank_was_available
        ``True`` iff the actor was running with skill_selection enabled
        for this step. ``False`` short-circuits the NO_SKILL_BOUND
        signal — on cold-start step 0 the actor *always* runs without a
        bank and we shouldn't synthesize a "missing adapter" trace for
        it.
    cycle_every_k_steps
        If >0, run ``cycle()`` whenever ``(step + 1) % k == 0``. The
        result's proposals are concatenated with the per-episode ones in
        the JSONL output but tagged ``proposer="composer/generalizer"``
        so the downstream gate can tell them apart.
    """
    t0 = time.monotonic()

    # Lazy: keep the heavy crafter / orchestrator imports inside the
    # function so the trainer's import graph doesn't pull them in until
    # the hook actually fires. This matters for ``--no-grpo`` smoke
    # tests that want to run a single co-evolution step without paying
    # for the full crafter dep tree on import.
    from crafter.service import SkillCrafterService
    from orchestrator.artifact_store import ArtifactStore

    # Group rollouts by game.
    by_game: Dict[str, List[Any]] = {}
    for ep in rollout_results:
        game = getattr(ep, "game", "")
        if not game or game == "__SENTINEL__":
            continue
        if getattr(ep, "steps", 0) <= 0:
            continue
        by_game.setdefault(game, []).append(ep)

    out_root = Path(run_dir) / "crafter_proposals_out" / f"step_{step:04d}"
    out_root.mkdir(parents=True, exist_ok=True)

    # We co-locate the artifact store in the same step dir so per-proposal
    # JSONs from `crafter.put_proposal` are auditable, but the contract
    # the Promotion hook reads is the JSONL summary written below — same
    # split as the offline mirror.
    artifact_root = out_root / "_artifacts"
    artifact_root.mkdir(exist_ok=True)
    artifact_store = ArtifactStore(str(artifact_root))

    n_proposals_total = 0
    n_failures_total = 0
    n_episodes_total = 0
    proposals_per_game: Dict[str, int] = {}
    proposals_jsonl_paths: Dict[str, Path] = {}
    n_coalesced_total = 0
    n_cooldown_total = 0
    n_cycle_proposals_total = 0
    cycle_ran = (
        cycle_every_k_steps > 0 and ((step + 1) % cycle_every_k_steps == 0)
    )
    # Path 2 — LLM Crafter rollups (initialised even when disabled so the
    # final ``CrafterStepReport.to_dict`` always carries the keys).
    n_llm_proposals_total = 0
    n_llm_calls_attempted_total = 0
    n_llm_calls_succeeded_total = 0
    n_llm_calls_failed_total = 0
    n_llm_timeouts_total = 0
    n_llm_parse_failures_total = 0
    llm_proposals_per_kind_total: Dict[str, int] = {}
    llm_sample_errors_total: List[str] = []

    for game, episodes in by_game.items():
        bank_path = legacy_bank_paths.get(game)
        if bank_path is None:
            logger.warning(
                "crafter_hook: no legacy bank path for game=%s, skipping", game,
            )
            continue
        bank_path = Path(bank_path)

        # Each game gets its own ephemeral SkillRepository, rooted at a
        # temp dir we tear down after writing the JSONL — same pattern as
        # the offline mirror at
        # ``labeling_supplement/reflect_per_episode_gpt54.py:707-718``.
        with tempfile.TemporaryDirectory(prefix=f"crafter-step{step}-{game}-") as tmpdir:
            tmp_root = Path(tmpdir)
            repo = SkillRepository(
                draft_store=SkillStore(StoreName.DRAFT, str(tmp_root / "draft")),
                candidate_store=SkillStore(StoreName.CANDIDATE, str(tmp_root / "candidate")),
                active_store=SkillStore(StoreName.ACTIVE, str(tmp_root / "active")),
                archive_store=SkillStore(StoreName.ARCHIVE, str(tmp_root / "archive")),
            )
            lifecycle = SkillLifecycleManager(repo)

            n_seeded = _seed_repo_from_legacy_jsonl(
                lifecycle=lifecycle, bank_path=bank_path, default_domain="gymv",
            )
            if n_seeded == 0:
                logger.debug(
                    "crafter_hook: %s bank has zero seedable entries; "
                    "skipping reflection (cold-start)", game,
                )

            # ── Drain the harness rejection sink (Day-9c) ────────────
            # Before the Crafter reflects, fold the per-step harness
            # rejections into ``SkillRecord.false_binding_patterns`` so
            # the Repairer's patch-or-retire path sees them on the
            # *same* skill records it owns (PLAN-SKILL-BANK §4.3b /
            # harness/README §22). The hook drained nothing in the
            # cold-start case (sink is empty) — pure no-op then.
            game_hook = (harness_hooks or {}).get(game)
            if game_hook is not None:
                try:
                    flush_report = game_hook.flush_to_lifecycle(
                        lifecycle, min_count=1, reset=True,
                    )
                    if flush_report.n_patterns_written:
                        logger.debug(
                            "crafter_hook: %s harness sink → %d patterns on "
                            "%d skill record(s); %d unknown skill_id(s) skipped",
                            game,
                            flush_report.n_patterns_written,
                            flush_report.n_skills_touched,
                            len(flush_report.skipped_unknown_skill_ids),
                        )
                except Exception as exc:                                # noqa: BLE001
                    logger.warning(
                        "crafter_hook: harness sink flush failed for "
                        "step=%d game=%s: %s",
                        step, game, exc,
                    )

            service = SkillCrafterService(
                lifecycle=lifecycle,
                artifact_store=artifact_store,
                teacher_model=teacher_model,
                hot_pattern_threshold=hot_pattern_threshold,
                cooldown_passes=cooldown_passes,
                enable_protocol_patching=enable_protocol_patching,
                hypothesize_min_recurrences=hypothesize_min_recurrences,
                hypothesize_related_skill_jaccard=hypothesize_related_skill_jaccard,
            )

            game_proposals: List[BankMutationProposal] = []
            game_failures: List[FailureTrace] = []
            domain_for_proposal = "gymv"
            # Track Path 2 LLM-Crafter proposal IDs so the JSONL writer
            # can tag them with proposer="llm_crafter" without
            # mistaking deterministic proposals (which inherit a
            # ``teacher_model`` from ``SkillCrafterService``) for
            # LLM-driven ones.
            llm_proposal_ids: set = set()

            for ep in episodes:
                n_episodes_total += 1
                failures = _synthesize_failures(
                    episode=ep,
                    domain=domain_for_proposal,
                    outcome_failure_threshold=outcome_failure_threshold,
                    max_failures=max_failures_per_episode,
                    bank_was_available=bank_was_available,
                )
                game_failures.extend(failures)
                n_failures_total += len(failures)

                if not failures:
                    # No signal → reflect_on_episode short-circuits to a
                    # no-op (`EpisodeReflection.has_signal` is False).
                    continue

                reflection = EpisodeReflection(
                    episode_id=getattr(ep, "episode_id", "") or f"step{step}-{game}-anon",
                    domain=domain_for_proposal,
                    failure_traces=failures,
                    skill_episodes=[],                                  # F2: no Harness
                    new_candidate_skill_ids=[],                         # no Bank Agent in trainer
                    bank_agent_actions={},
                    outcome_summary={
                        "total_reward": float(getattr(ep, "total_reward", 0.0)),
                        "steps": int(getattr(ep, "steps", 0)),
                        "terminated": bool(getattr(ep, "terminated", False)),
                        "truncated": bool(getattr(ep, "truncated", False)),
                        "trainer_step": step,
                        "game": game,
                    },
                )
                try:
                    result = service.reflect_on_episode(reflection)
                except Exception as exc:                                # noqa: BLE001
                    logger.exception(
                        "crafter_hook: reflect_on_episode failed for "
                        "step=%d game=%s episode=%s: %s",
                        step, game, reflection.episode_id, exc,
                    )
                    continue
                game_proposals.extend(result.proposals)
                n_coalesced_total += getattr(result, "n_patches_coalesced", 0) or 0
                n_cooldown_total += getattr(result, "n_patches_skipped_cooldown", 0) or 0

            if cycle_ran and game_failures:
                try:
                    cyc = service.cycle(new_failures=[])  # failures already ingested
                except Exception as exc:                                # noqa: BLE001
                    logger.exception(
                        "crafter_hook: cycle() failed for step=%d game=%s: %s",
                        step, game, exc,
                    )
                else:
                    game_proposals.extend(cyc.proposals)
                    n_cycle_proposals_total += len(cyc.proposals)

            # ── Path 2 — supplemental LLM Crafter ──────────────────────
            # One 35B call per failure trace, capped at ``k_max``.
            # Proposals are appended *after* the deterministic /
            # cycle proposals so the JSONL row order (and the
            # downstream Promotion gate's iteration order) puts
            # rule-based proposals first and LLM proposals last —
            # matching the precedence the offline mirror uses.
            if llm_crafter_enabled and game_failures:
                try:
                    from trainer.coevolution._llm_crafter import (
                        run_llm_crafter,
                    )
                    profile = (game_profiles or {}).get(game)
                    # Empty `llm_crafter_model` → defer to the env-exported
                    # ``VLM_AGENT_BACKBONE_JUDGE_MODEL`` (set by the launch
                    # script in tandem with VLLM_BASE_URL_MAP).  Final
                    # fallback: the canonical 35B-A3B slug.  This keeps
                    # the Path-2 model in lockstep with whatever the
                    # judge endpoint at port :8004 is actually serving.
                    _resolved_crafter_model = (
                        (llm_crafter_model or "").strip()
                        or os.environ.get(
                            "VLM_AGENT_BACKBONE_JUDGE_MODEL", "",
                        ).strip()
                        or "Qwen/Qwen3.5-35B-A3B"
                    )
                    # Render the current bank as a compact list for the
                    # 35B prompt's ``existing_skills`` block. The LLM
                    # crafter uses this to prefer ``patch`` over
                    # ``hypothesize`` whenever a related skill exists
                    # (see _llm_crafter._build_prompt — added 2026-05-04
                    # as part of the v11 hypothesis-pollution fix).
                    existing_skills_for_prompt = _read_bank_summary_for_prompt(
                        bank_path
                    )
                    llm_proposals, llm_report = run_llm_crafter(
                        failures=game_failures,
                        game=game,
                        model=_resolved_crafter_model,
                        game_profile=profile,
                        k_max=llm_crafter_k_max,
                        max_tokens=llm_crafter_max_tokens,
                        temperature=llm_crafter_temperature,
                        timeout_s=llm_crafter_timeout_s,
                        enable_thinking=llm_crafter_enable_thinking,
                        existing_skills=existing_skills_for_prompt,
                    )
                    game_proposals.extend(llm_proposals)
                    for p in llm_proposals:
                        pid = getattr(p, "proposal_id", None)
                        if pid:
                            llm_proposal_ids.add(pid)
                    n_llm_proposals_total += len(llm_proposals)
                    n_llm_calls_attempted_total += llm_report.n_calls_attempted
                    n_llm_calls_succeeded_total += llm_report.n_calls_succeeded
                    n_llm_calls_failed_total += llm_report.n_calls_failed
                    n_llm_timeouts_total += llm_report.n_timeouts
                    n_llm_parse_failures_total += llm_report.n_parse_failures
                    for k, v in llm_report.proposals_per_kind.items():
                        llm_proposals_per_kind_total[k] = (
                            llm_proposals_per_kind_total.get(k, 0) + v
                        )
                    for err in llm_report.sample_errors:
                        if err not in llm_sample_errors_total:
                            llm_sample_errors_total.append(err)
                            if len(llm_sample_errors_total) >= 10:
                                break
                    if llm_proposals or llm_report.n_calls_attempted:
                        logger.info(
                            "crafter_hook[llm]: step=%d game=%s "
                            "n_calls=%d/%d → %d proposals "
                            "(timeouts=%d parse_fail=%d call_err=%d) "
                            "in %.2fs",
                            step, game,
                            llm_report.n_calls_succeeded,
                            llm_report.n_calls_attempted,
                            len(llm_proposals),
                            llm_report.n_timeouts,
                            llm_report.n_parse_failures,
                            llm_report.n_calls_failed
                            - llm_report.n_timeouts
                            - llm_report.n_parse_failures,
                            llm_report.wall_time_s,
                        )
                        if llm_report.sample_errors:
                            logger.warning(
                                "crafter_hook[llm]: step=%d game=%s "
                                "sample_errors=%r",
                                step, game,
                                llm_report.sample_errors[:3],
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "crafter_hook[llm]: run_llm_crafter raised for "
                        "step=%d game=%s: %s; "
                        "deterministic proposals proceed unchanged",
                        step, game, exc,
                    )

            # ── Quality gate (post-v11 fix) ──────────────────────────
            # Defense-in-depth filter: even with the upstream fixes
            # (service.py recurrence + relatedness gates,
            # _llm_crafter.py last-resort prompt, k_max=2), occasional
            # boilerplate hypotheses can still slip through if the 35B
            # disregards prompt instructions or the deterministic
            # Hypothesizer's diagnoser yields a degenerate template.
            # We drop them at the JSONL boundary so the Promotion gate
            # never even sees them — they don't waste a 35B judge call,
            # don't pollute the audit stream's PROMOTE/REJECT counts,
            # and don't (ever) reach the bank.
            #
            # Override via ``CRAFTER_ALLOW_BOILERPLATE_HYPOTHESIS=1``.
            # Rejected proposals are summarised in
            # ``_artifacts/audit.jsonl`` so the diagnostic dashboard
            # can show "35B wanted to add a hypothesis but it failed
            # the quality bar" — useful for tuning the prompt.
            game_proposals, n_dropped_boilerplate = _filter_boilerplate_hypotheses(
                game_proposals,
                artifact_store=artifact_store,
                step=step,
                game=game,
            )
            if n_dropped_boilerplate:
                logger.info(
                    "crafter_hook: step=%d game=%s dropped %d boilerplate "
                    "hypothesis proposal(s) at quality gate",
                    step, game, n_dropped_boilerplate,
                )

            # Write per-game JSONL in the offline-mirror schema.
            jsonl_path = _write_proposals_jsonl(
                step_root=out_root,
                game=game,
                proposals=game_proposals,
                domain=domain_for_proposal,
                llm_proposal_ids=llm_proposal_ids,
            )
            proposals_per_game[game] = len(game_proposals)
            n_proposals_total += len(game_proposals)
            proposals_jsonl_paths[game] = jsonl_path

    elapsed = time.monotonic() - t0

    # Per-step summary file mirrors the offline mirror's _run_summary.json
    # so dashboards can consume both interchangeably.
    summary = {
        "step": step,
        "n_games": len(by_game),
        "n_episodes_reflected": n_episodes_total,
        "n_failure_traces": n_failures_total,
        "n_proposals": n_proposals_total,
        "proposals_per_game": proposals_per_game,
        "n_patches_coalesced": n_coalesced_total,
        "n_patches_skipped_cooldown": n_cooldown_total,
        "cycle_ran": cycle_ran,
        "n_cycle_proposals": n_cycle_proposals_total,
        "n_llm_proposals": n_llm_proposals_total,
        "n_llm_calls_attempted": n_llm_calls_attempted_total,
        "n_llm_calls_succeeded": n_llm_calls_succeeded_total,
        "n_llm_calls_failed": n_llm_calls_failed_total,
        "n_llm_timeouts": n_llm_timeouts_total,
        "n_llm_parse_failures": n_llm_parse_failures_total,
        "llm_sample_errors": llm_sample_errors_total[:10],
        "llm_proposals_per_kind": llm_proposals_per_kind_total,
        "wall_time_s": elapsed,
        "params": {
            "outcome_failure_threshold": outcome_failure_threshold,
            "max_failures_per_episode": max_failures_per_episode,
            "hot_pattern_threshold": hot_pattern_threshold,
            "cooldown_passes": cooldown_passes,
            "cycle_every_k_steps": cycle_every_k_steps,
            "teacher_model": teacher_model,
            "enable_protocol_patching": enable_protocol_patching,
            "llm_crafter_enabled": llm_crafter_enabled,
            "llm_crafter_model": llm_crafter_model,
            "llm_crafter_k_max": llm_crafter_k_max,
            "llm_crafter_max_tokens": llm_crafter_max_tokens,
            "llm_crafter_temperature": llm_crafter_temperature,
            "llm_crafter_timeout_s": llm_crafter_timeout_s,
            "llm_crafter_enable_thinking": bool(llm_crafter_enable_thinking),
        },
    }
    try:
        (out_root / "_step_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("crafter_hook: could not write _step_summary.json: %s", exc)

    return CrafterStepReport(
        step=step,
        run_dir=out_root,
        n_episodes_reflected=n_episodes_total,
        n_failure_traces=n_failures_total,
        n_proposals=n_proposals_total,
        proposals_per_game=proposals_per_game,
        proposals_jsonl_paths=proposals_jsonl_paths,
        n_patches_coalesced=n_coalesced_total,
        n_patches_skipped_cooldown=n_cooldown_total,
        cycle_ran=cycle_ran,
        n_cycle_proposals=n_cycle_proposals_total,
        n_llm_proposals=n_llm_proposals_total,
        n_llm_calls_attempted=n_llm_calls_attempted_total,
        n_llm_calls_succeeded=n_llm_calls_succeeded_total,
        n_llm_calls_failed=n_llm_calls_failed_total,
        llm_proposals_per_kind=llm_proposals_per_kind_total,
        wall_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# F2 — failure synthesis (narrow trainer-mode subset)
# ---------------------------------------------------------------------------


def _synthesize_failures(
    *,
    episode: Any,
    domain: str,
    outcome_failure_threshold: float,
    max_failures: int,
    bank_was_available: bool,
) -> List[FailureTrace]:
    """Synthesize ``FailureTrace``s from a trainer ``EpisodeResult``.

    Two signals (the only two the trainer's experience dict supports —
    see module docstring "F2"):

    1. **OUTCOME_FAILURE** — episode-level. ``total_reward <= threshold``.
    2. **NO_SKILL_BOUND** — per-step. ``skill_id`` was missing on a
       step where the bank was non-empty going into the episode.
    """
    out: List[FailureTrace] = []
    episode_id = getattr(episode, "episode_id", "") or "anon"
    total_reward = float(getattr(episode, "total_reward", 0.0))
    n_steps = int(getattr(episode, "steps", 0) or 0)
    experiences = list(getattr(episode, "experiences", []) or [])
    truncated = bool(getattr(episode, "truncated", False))

    # ── 1. OUTCOME_FAILURE ────────────────────────────────────────────
    if total_reward <= outcome_failure_threshold:
        # Pick a representative skill id: the most-frequently-bound one
        # this episode (or empty if no skill ever bound).
        bound_counts: Dict[str, int] = {}
        for exp in experiences:
            sid = exp.get("skill_id")
            if isinstance(sid, str) and sid:
                bound_counts[sid] = bound_counts.get(sid, 0) + 1
        rep_skill = (
            max(bound_counts.items(), key=lambda kv: kv[1])[0]
            if bound_counts
            else ""
        )
        out.append(FailureTrace(
            skill_id=rep_skill,
            skill_episode_id=f"{episode_id}#outcome",
            domain=domain,
            failed_step_index=n_steps - 1 if n_steps else None,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=(
                f"episode_total_reward={total_reward:.3f}"
                f" <= threshold={outcome_failure_threshold:.3f}"
                + (" (truncated)" if truncated else "")
            ),
            extra={
                "synthesis_signal": "OUTCOME_FAILURE",
                "episode_id": episode_id,
                "n_steps": n_steps,
                "total_reward": total_reward,
                "truncated": truncated,
            },
        ))

    # ── 2. NO_SKILL_BOUND (only meaningful when the bank exists) ──────
    if bank_was_available:
        for i, exp in enumerate(experiences):
            sid = exp.get("skill_id")
            if sid:
                continue
            # Treat as failure only when there *was* a bank to consult.
            out.append(FailureTrace(
                skill_id="",
                skill_episode_id=f"{episode_id}#no_skill@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="MISSING_ADAPTER",
                abort_reason="no_skill_bound",
                extra={
                    "synthesis_signal": "NO_SKILL_BOUND",
                    "step_index": i,
                    "step_reward": float(exp.get("reward") or 0.0),
                },
            ))

    if len(out) > max_failures:
        out = out[:max_failures]
    return out


# ---------------------------------------------------------------------------
# Bank seeding — read legacy JSONL → SkillRepository as CANDIDATE
# ---------------------------------------------------------------------------


_ROLE_TO_SKILL_TYPE = {
    "GATHER": SkillType.GROUNDING,
    "VERIFY": SkillType.REASONING,
    "REASON": SkillType.REASONING,
    "COMMIT": SkillType.ACTION,
}


def _safe_skill_id(skill_id: str) -> str:
    """Filename-safe form for skill ids that contain ``/`` (the legacy
    cold-start convention is ``OPERATOR/SUBGOAL`` like ``COMMIT/ATTACK``,
    which collides with ``SkillStore``'s flat-filename layout)."""
    return (skill_id or "").replace("/", "__")


def _wrap_protocol_steps(raw_steps: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert NL-string ``protocol.steps`` from the legacy bank into the
    typed ``[{"action": ..., "payload": {}, "notes": ...}]`` shape that
    ``Repairer._rule_repair`` and friends expect.

    Mirrors the inverse of
    :func:`skill_bank.legacy_writeback._typed_protocol_to_nl_steps`.
    """
    out: List[Dict[str, Any]] = []
    for s in raw_steps or []:
        if isinstance(s, dict):
            out.append(dict(s))
        elif isinstance(s, str):
            out.append({"action": "EXEC", "payload": {}, "notes": s})
        else:
            out.append({"action": "EXEC", "payload": {}, "notes": str(s)})
    return out


def _record_from_bank_entry(
    entry: Mapping[str, Any], default_domain: str,
) -> Optional[SkillRecord]:
    """Hydrate one ``SkillRecord`` from one legacy ``skill_bank.jsonl``
    line. Returns ``None`` if the entry is malformed."""
    skill = entry.get("skill") or {}
    if not isinstance(skill, Mapping):
        return None
    raw_id = skill.get("skill_id")
    if not raw_id:
        return None
    role = (skill.get("evidence_role") or "COMMIT").upper()
    skill_type = _ROLE_TO_SKILL_TYPE.get(role, SkillType.MIXED)
    contract = skill.get("contract") or {}
    feasible = list(skill.get("applicable_domains") or []) or [default_domain]

    # Two on-disk shapes for ``protocol``:
    #   * legacy cold-start (pre-Day-2 lift) — a dict
    #     ``{"steps": [<NL prose strings>], "preconditions": [...], …}``
    #   * Day-2-lifted bank — a list of typed hops
    #     ``[{"action": "SLIDE", "payload": {…}, "notes": …}, …]``
    # The hop-list shape carries no companion preconditions/success_criteria
    # — those move into the lifted ``SkillContract`` upstream — so when we
    # see a list we just hand it to ``_wrap_protocol_steps`` and treat the
    # ancillary contract fields as empty.
    raw_protocol = skill.get("protocol")
    if isinstance(raw_protocol, list):
        protocol_steps = list(raw_protocol)
        protocol_blob: Mapping[str, Any] = {}
    elif isinstance(raw_protocol, Mapping):
        protocol_blob = raw_protocol
        protocol_steps = list(protocol_blob.get("steps") or [])
    else:
        protocol_blob = {}
        protocol_steps = []

    feasible_tasks = list(skill.get("feasible_tasks") or [])
    verified_tasks = list(skill.get("verified_tasks") or [])
    new_kwargs: Dict[str, Any] = dict(
        name=skill.get("name", str(raw_id)),
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=feasible,
        protocol=_wrap_protocol_steps(protocol_steps),
        contract=SkillContract(
            preconditions=list(protocol_blob.get("preconditions") or []),
            effects_add=list(contract.get("eff_add") or []),
            effects_del=list(contract.get("eff_del") or []),
            expected_evidence_roles=[role] if role else [],
            success_criteria=list(protocol_blob.get("success_criteria") or []),
            abort_criteria=list(protocol_blob.get("abort_criteria") or []),
        ),
    )
    # ``feasible_tasks`` / ``verified_tasks`` are Day-2 additive fields on
    # ``SkillRecord``; older callers (and ``SkillRecord.new`` signatures
    # that pre-date Day-2) won't accept the kwargs, so we splice in only
    # if the dataclass actually has them.
    rec_fields = {f.name for f in fields(SkillRecord)}
    if "feasible_tasks" in rec_fields and feasible_tasks:
        new_kwargs["feasible_tasks"] = feasible_tasks
    if "verified_tasks" in rec_fields and verified_tasks:
        new_kwargs["verified_tasks"] = verified_tasks
    rec = SkillRecord.new(**new_kwargs)
    # Force the bank-given skill_id (overrides the freshly-minted UUID)
    # so ``parent_skill_ids`` references resolve.
    object.__setattr__(rec, "skill_id", _safe_skill_id(str(raw_id)))
    return rec


# ---------------------------------------------------------------------------
# Quality gate — drop boilerplate hypothesis stubs at the JSONL boundary
# ---------------------------------------------------------------------------

# Words that are so generic they signal a placeholder protocol step
# rather than a real, game-specific instruction. Curated from the v11
# audit: every boilerplate ``hypothesis__prop-...`` skill in the TF3
# bank had its ``protocol.steps`` consist entirely of these tokens
# (e.g. "Evaluate best available action / Execute chosen action /
# Observe result"). Each token alone is benign; the heuristic fires
# only when EVERY step in the protocol is dominated by this set.
_BOILERPLATE_PROTOCOL_TOKENS = frozenset({
    "action", "actions", "available", "best", "chosen", "execute",
    "execution", "evaluate", "observe", "observed", "perform",
    "result", "results", "step", "steps", "verify", "verified",
})

# Same idea for preconditions: "Action opportunity present" /
# "No active state-changing events are pending" / "Have target" all
# tokenise into purely generic vocabulary.
_BOILERPLATE_PRECONDITION_TOKENS = frozenset({
    "action", "active", "available", "any", "appropriate", "events",
    "have", "no", "opportunity", "pending", "present", "ready",
    "state", "target",
})


def _is_boilerplate_protocol_step(step: Any) -> bool:
    """True iff a single protocol step looks like a generic placeholder.

    A step qualifies as boilerplate when its NL ``notes`` (or stringified
    form) tokenises into ≥ 80% members of
    :data:`_BOILERPLATE_PROTOCOL_TOKENS`. We require ≥ 2 tokens of signal
    so 1-word actions ("MOVE_LEFT") never accidentally match — short
    strings are *kept* by default.
    """
    if isinstance(step, dict):
        text = step.get("notes") or step.get("payload") or ""
    else:
        text = step
    text = str(text or "").strip()
    if not text:
        return True
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    if len(tokens) < 2:
        return False
    n_boiler = sum(1 for t in tokens if t in _BOILERPLATE_PROTOCOL_TOKENS)
    return (n_boiler / len(tokens)) >= 0.80


def _is_boilerplate_precondition(pred: Any) -> bool:
    """Same generic-token heuristic for preconditions."""
    text = str(pred or "").strip()
    if not text:
        return True
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    if len(tokens) < 2:
        return False
    n_boiler = sum(1 for t in tokens if t in _BOILERPLATE_PRECONDITION_TOKENS)
    return (n_boiler / len(tokens)) >= 0.80


def _classify_hypothesis_for_quality_gate(
    proposal: HypothesisProposal,
) -> Optional[str]:
    """Decide whether a HypothesisProposal looks like a placeholder stub.

    Returns:
      * ``None`` — proposal looks legitimate, keep it.
      * Otherwise, a short reason code suitable for audit logs (e.g.
        ``"name_is_placeholder"``, ``"empty_protocol"``,
        ``"all_steps_boilerplate"``, ``"empty_contract"``).

    The heuristics deliberately err on the side of *keeping*
    proposals when the signal is ambiguous (e.g. a 1-word step or a
    short name): the upstream gates already filter on recurrence and
    relatedness, so this layer's job is only to catch the
    pathological case where the LLM / Hypothesizer emitted an
    obvious template.
    """
    # 1. Name is the auto-generated placeholder — "hyp-XXXXXX" or
    #    "hypothesis__prop-...".  These are precisely the names the
    #    Hypothesizer / LLM crafter fall back to when they failed to
    #    supply a concrete game-specific name.
    name = (getattr(proposal, "name", "") or "").strip()
    if not name or name.startswith(("hyp-", "hypothesis__")):
        return "name_is_placeholder"

    # 2. Protocol must have ≥ 2 steps; otherwise the proposal carries
    #    no actionable instruction (the contract alone is not enough).
    proto = getattr(proposal, "novel_protocol", None) or []
    if len(proto) < 2:
        return "protocol_too_short"

    # 3. Every step must NOT be boilerplate. We flag the proposal only
    #    when *all* steps are generic — a proposal with one boilerplate
    #    "verify result" step among real ones is still useful.
    if all(_is_boilerplate_protocol_step(s) for s in proto):
        return "all_steps_boilerplate"

    # 4. Contract must carry SOMETHING — at least one precondition
    #    OR one effect literal. Proposals with an entirely empty
    #    contract are pure speculation; the bank already has retire
    #    + patch paths that handle "we don't know yet" cases.
    contract = getattr(proposal, "contract", None)
    has_signal = False
    if contract is not None:
        pre = list(getattr(contract, "preconditions", []) or [])
        eff_a = list(getattr(contract, "effects_add", []) or [])
        eff_d = list(getattr(contract, "effects_del", []) or [])
        # ≥ 1 non-boilerplate precondition OR any effect literal.
        non_boiler_pre = [p for p in pre if not _is_boilerplate_precondition(p)]
        if non_boiler_pre or eff_a or eff_d:
            has_signal = True
    if not has_signal:
        return "empty_contract"

    return None


def _filter_boilerplate_hypotheses(
    proposals: List[BankMutationProposal],
    *,
    artifact_store: Any,
    step: int,
    game: str,
) -> Tuple[List[BankMutationProposal], int]:
    """Drop ``HypothesisProposal``s whose protocol/contract is generic
    boilerplate. Returns ``(kept_proposals, n_dropped)``.

    Each rejected proposal is summarised in the artifact store's
    audit log so the dashboard can show ``35B wanted to add X but it
    failed the quality bar`` alongside the standard PROMOTE/REJECT
    breakdown.

    Override via ``CRAFTER_ALLOW_BOILERPLATE_HYPOTHESIS=1`` — when
    that env var is set we keep everything (audit log still records
    the reason code so the diagnostic is preserved even when the
    gate is off).
    """
    if os.environ.get("CRAFTER_ALLOW_BOILERPLATE_HYPOTHESIS", "0") in ("1", "true", "yes"):
        return list(proposals), 0
    kept: List[BankMutationProposal] = []
    n_dropped = 0
    for p in proposals:
        if isinstance(p, HypothesisProposal):
            reason = _classify_hypothesis_for_quality_gate(p)
            if reason is not None:
                n_dropped += 1
                try:
                    artifact_store.append_audit({
                        "kind": "hypothesis_dropped_quality_gate",
                        "step": step,
                        "game": game,
                        "proposal_id": getattr(p, "proposal_id", ""),
                        "name": getattr(p, "name", ""),
                        "reason": reason,
                        "n_protocol_steps": len(getattr(p, "novel_protocol", []) or []),
                    })
                except Exception:  # noqa: BLE001
                    # Audit-log failures never gate the trainer.
                    pass
                continue
        kept.append(p)
    return kept, n_dropped


def _read_bank_summary_for_prompt(
    bank_path: Path,
    *,
    max_skills: int = 12,
) -> List[Dict[str, Any]]:
    """Compact ``[{skill_id, name, strategic_description}]`` list for the
    LLM Crafter's ``existing_skills`` prompt block.

    Reads the legacy JSONL directly (cheaper than building a full
    SkillRepository+BankView for read-only summary rendering — the
    Path-2 LLM crafter is on the per-step hot path, so we avoid
    redundant lifecycle ingestion).

    Returns an empty list on cold-start (file missing / empty) or
    parse failure; the prompt's ``existing_skills`` block is then just
    empty, which the rewritten policy (``_build_prompt``) handles
    correctly — clause 4(a) "existing_skills is empty" satisfies the
    cold-start gate.

    Skills are returned in *reverse-on-disk* order (newest first),
    truncated to ``max_skills`` — the latest additions are most likely
    to match a recent failure context. We deliberately exclude entries
    whose name starts with ``hypothesis__`` or ``hyp-``: those are the
    placeholder hypothesis stubs from prior runs, and rendering them
    in the prompt would just teach the 35B that "hypothesize is fine"
    via in-context biasing.
    """
    if not bank_path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with bank_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        skill = entry.get("skill") or entry
        if not isinstance(skill, Mapping):
            continue
        sid = str(skill.get("skill_id") or "").strip()
        if not sid:
            continue
        name = str(skill.get("name") or sid)
        if name.startswith(("hypothesis__", "hyp-")):
            continue  # don't bias the prompt with prior placeholder stubs
        desc = (
            skill.get("strategic_description")
            or skill.get("description")
            or ""
        )
        out.append({
            "skill_id": sid,
            "name": name,
            "strategic_description": str(desc or ""),
        })
        if len(out) >= max_skills:
            break
    return out


def _seed_repo_from_legacy_jsonl(
    *,
    lifecycle: SkillLifecycleManager,
    bank_path: Path,
    default_domain: str,
) -> int:
    """Seed a fresh ``SkillRepository`` as ``CANDIDATE`` from a legacy
    ``skill_bank.jsonl``. Returns the count of successfully-seeded skills.

    Malformed lines are skipped with a debug log line; this matches the
    offline mirror's tolerance because the legacy bank is a *running*
    artefact and we'd rather seed N-1 skills than refuse to fire.
    """
    if not bank_path.is_file():
        return 0
    n_seeded = 0
    with bank_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.debug(
                    "crafter_hook: skipping malformed bank line %d in %s: %s",
                    line_no, bank_path, exc,
                )
                continue
            rec = _record_from_bank_entry(entry, default_domain)
            if rec is None:
                continue
            try:
                lifecycle.ingest_draft(rec)
                lifecycle.transition(
                    rec.skill_id,
                    to_status=SkillStatus.CANDIDATE,
                    rationale="trainer-crafter-hook seed",
                )
                n_seeded += 1
            except Exception as exc:                                # noqa: BLE001
                logger.debug(
                    "crafter_hook: skip seed %s: %s",
                    rec.skill_id, exc,
                )
    return n_seeded


# ---------------------------------------------------------------------------
# Live BankMutationProposal → offline-mirror JSONL row
# ---------------------------------------------------------------------------


def _to_offline_row(
    proposal: BankMutationProposal,
    *,
    domain: str,
    llm_proposal_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Project one live ``BankMutationProposal`` into the JSONL row
    schema that ``decide_promotion_gpt54.py::_OfflineProposal.from_json``
    accepts.

    The offline schema is intentionally flatter than the live dataclass —
    we only emit the fields ``_translate_proposal`` re-reads, plus
    ``adapter_plan`` (mirrored across all five domains so the live
    ``ALL_FIVE_DOMAINS`` check at the gate's Stage-0 passes).
    """
    # PLAN-SKILL-CRAFTER §0.1 / §2.5 — every proposal in the offline
    # mirror schema MUST enumerate all five target domains so the gate
    # stack can verify general feasibility. The live Repairer narrows
    # ``target_domains`` to the source domain only (it's working against
    # a single game's skill bank), but for the on-disk JSONL contract we
    # always expand to all five — the gate decides per-domain whether to
    # honor the proposal.
    base = {
        "proposal_id": proposal.proposal_id,
        "rationale": proposal.rationale,
        "proposer": _proposer_for(proposal, llm_proposal_ids=llm_proposal_ids),
        "target_domains": list(ALL_FIVE_DOMAINS),
        "adapter_plan": _default_adapter_plan(domain),
    }

    if isinstance(proposal, PatchProposal):
        base["proposal_kind"] = "patch"
        base["target_skill_id"] = proposal.base_skill_id
        base["patch_kind"] = proposal.recovery_strategy or "protocol_patch"
        base["evidence_role"] = _evidence_role_from_contract(proposal.patched_contract)
        base["seed_failure_ids"] = list(proposal.seed_failure_ids)
    elif isinstance(proposal, RetireProposal):
        base["proposal_kind"] = "retire"
        base["target_skill_id"] = proposal.target_skill_id
        base["retire_reason"] = proposal.reason or "evidence-starved"
        base["evidence_role"] = ""
    elif isinstance(proposal, ComposeProposal):
        base["proposal_kind"] = "compose"
        base["components"] = list(proposal.component_skill_ids)
        base["compose_op"] = "sequence"
        base["evidence_role"] = _evidence_role_from_contract(proposal.contract)
    elif isinstance(proposal, GeneralizeProposal):
        base["proposal_kind"] = "transfer"
        base["source_skill_id"] = proposal.base_skill_id
        base["source_domain"] = proposal.source_domain or "gymv"
        base["new_adapter_per_target"] = {proposal.target_domain: True} if proposal.target_domain else {}
        base["slot_remap_per_target"] = {proposal.target_domain: dict(proposal.slot_remap)} if proposal.target_domain else {}
        base["evidence_role"] = _evidence_role_from_contract(proposal.contract)
    elif isinstance(proposal, HypothesisProposal):
        base["proposal_kind"] = "hypothesize"
        base["new_skill_name"] = proposal.name
        base["evidence_role"] = _evidence_role_from_contract(proposal.contract)
    else:
        base["proposal_kind"] = type(proposal).__name__.lower()

    return base


def _proposer_for(
    p: BankMutationProposal,
    *,
    llm_proposal_ids: Optional[set] = None,
) -> str:
    """Map proposal class → offline-mirror ``proposer`` enum.

    Path 2 LLM-Crafter proposals are identified by membership in the
    explicit ``llm_proposal_ids`` set the hook collects when minting
    them.  We can't rely on the ``teacher_model`` attribute alone:
    :class:`crafter.service.SkillCrafterService` defaults it to
    ``BACKBONE_TEACHER_MODEL`` for *all* deterministic proposals it
    emits, so the field is non-empty even on the rule-based path.
    The set-membership check is the only unambiguous signal.
    """
    if (
        llm_proposal_ids is not None
        and getattr(p, "proposal_id", None) in llm_proposal_ids
    ):
        return "llm_crafter"
    if isinstance(p, (PatchProposal, RetireProposal)):
        return "reflector"
    if isinstance(p, ComposeProposal):
        return "composer"
    if isinstance(p, GeneralizeProposal):
        return "generalizer"
    if isinstance(p, HypothesisProposal):
        return "hypothesizer"
    return "reflector"


def _evidence_role_from_contract(contract: Any) -> str:
    if contract is None:
        return ""
    roles = getattr(contract, "expected_evidence_roles", None)
    if not roles:
        return ""
    return str(roles[0]).upper()


def _default_adapter_plan(source_domain: str) -> Dict[str, Dict[str, Any]]:
    """Default adapter plan: ``reuse`` on the source domain, marked as
    ``synthesize_from_slot_ontology`` on the four transfer targets.

    Mirrors the shape `decide_skill_crafting_gpt54.py` writes verbatim
    so a downstream reader doesn't need to special-case trainer output.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for d in ALL_FIVE_DOMAINS:
        if d == source_domain:
            out[d] = {
                "needs_72b_synthesis": False,
                "source_domain": source_domain,
                "strategy": "reuse",
            }
        else:
            out[d] = {
                "needs_72b_synthesis": True,
                "source_domain": source_domain,
                "strategy": "synthesize_from_slot_ontology",
            }
    return out


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def _write_proposals_jsonl(
    *,
    step_root: Path,
    game: str,
    proposals: Sequence[BankMutationProposal],
    domain: str,
    llm_proposal_ids: Optional[set] = None,
) -> Path:
    """Write per-game ``proposals.jsonl`` under ``<step_root>/<corpus>/<source>/``.

    Always creates the file (even when ``proposals`` is empty), so the
    Promotion hook's ``--proposals-run`` walk doesn't silently skip a
    game with zero proposals — empty file = "we looked, found nothing".
    """
    corpus = corpus_for_game(game)
    pair_dir = step_root / corpus / game
    pair_dir.mkdir(parents=True, exist_ok=True)
    out_path = pair_dir / "proposals.jsonl"
    # Write atomically — the Promotion hook may iterate concurrently.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=str(pair_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            for p in proposals:
                row = _to_offline_row(
                    p, domain=domain, llm_proposal_ids=llm_proposal_ids,
                )
                tmp.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
                tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return out_path


__all__ = [
    "ALL_FIVE_DOMAINS",
    "CrafterStepReport",
    "DEFAULT_COOLDOWN_PASSES",
    "DEFAULT_HOT_PATTERN_THRESHOLD",
    "DEFAULT_MAX_FAILURES_PER_EPISODE",
    "DEFAULT_OUTCOME_FAILURE_THRESHOLD",
    "corpus_for_game",
    "run_crafter_step",
]
