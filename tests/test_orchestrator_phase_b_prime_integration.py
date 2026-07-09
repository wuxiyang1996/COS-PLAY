"""End-to-end integration test for the orchestrator's Phase B′ splice block.

Drives the **exact code path** in
``trainer/coevolution/orchestrator.py::co_evolution_loop`` between
``finalize_all()`` and the step-context snapshot, but with rollouts
synthesised in-process so the test runs without vLLM / GPUs.

What is real (production code, not stubs):

* ``trainer.coevolution.skillbank_pipeline.PerGameSkillBankManager`` —
  including ``_seed_from_coldstart``, ``AsyncSkillBankPipeline``,
  ``SkillBankAgent`` auto-init, ``SkillBankMVP.load()``, and
  ``SkillQueryEngine``.
* ``trainer.coevolution._crafter_hook.run_crafter_step`` — including the
  live ``crafter.service.SkillCrafterService.reflect_on_episode`` (Phase-1
  rule-based path), ``SkillRepository`` hydration, F2 failure
  synthesis, and proposal projection to the offline-mirror JSONL row
  schema.
* ``trainer.coevolution._promotion_hook.run_promotion_step`` — including
  the *real* ``labeling_supplement/decide_promotion_gpt54.py`` driver
  invoked via subprocess in ``offline-synthetic`` gate mode, the
  resulting ``_run_summary.json`` parse, and
  ``skill_bank.legacy_writeback.writeback_promotion``.
* The reload loop (``sb_manager.reload_banks_from_disk(keys)``) and the
  post-reload ``pipe.get_bank()`` call that the actor's *next* rollout
  would issue.

What is stubbed:

* The trainer's Phase A rollout collection: instead of running real
  Airstriker episodes through vLLM, we synthesise an
  ``EpisodeResult``-shaped object with realistic per-step
  ``experiences`` dicts whose ``skill_id`` field references one of the
  cold-start skills (so the Crafter has something to bind a Patch
  proposal to).

What this test does **not** validate (intentional gap, documented):

* That the actor's vLLM-driven action_taking / skill_selection prompts
  on step N+1 actually receive the promoted skills in their RAG
  payload. ``test_writeback_real_loader.py::
  test_real_pipeline_reload_picks_up_writeback`` already verifies that
  ``pipe.get_bank()`` exposes them via ``SkillQueryEngine`` after
  reload — this test extends that by driving the same chain through
  the orchestrator's actual splice contract (same call signatures,
  same ordering, same error handling) — but the prompt-construction
  layer that sits *between* ``get_bank()`` and the LLM is not exercised
  here. To close that final gap, run::

      python scripts/run_coevolution.py --no-grpo --no-wandb \\
          --crafter-promotion-enabled \\
          --games Temporal_Airstriker-v0 \\
          --episodes-per-game 1 --total-steps 2 --max-concurrent 1 \\
          --vllm-gpus 0

  and grep ``coevolution.log`` for ``"Phase B′ Crafter+Promotion"``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _coldstart_bank_dir() -> Path:
    p = (
        _repo_root() / "labeling" / "skill_bank_out" / "run_20260430_030637"
        / "gym_v"
    )
    return p


def _has_fixtures() -> bool:
    bank = _coldstart_bank_dir() / "Temporal_Airstriker-v0" / "skill_bank.jsonl"
    driver = _repo_root() / "labeling_supplement" / "decide_promotion_gpt54.py"
    return bank.is_file() and driver.is_file()


# Mirror trainer/coevolution/episode_runner.py::EpisodeResult exactly —
# including all the fields the orchestrator splice reads. Synthesizing
# one is cheaper than running real Airstriker rollouts and gives us
# precise control over which skill_id each step bound to and what the
# total_reward came out to (so we can deterministically trigger
# OUTCOME_FAILURE).
@dataclass
class _SyntheticEpisode:
    game: str
    episode_id: str
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    skill_switches: int = 0
    grpo_records: list = field(default_factory=list)
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    eval_only: bool = False
    role: str = ""
    side: str = ""
    role_index: int = -1


def _make_failed_episode(
    game: str = "Temporal_Airstriker-v0",
    episode_id: str = "ep-0001",
    bound_skill_id: str = "COMMIT/ATTACK",
    n_steps: int = 4,
) -> _SyntheticEpisode:
    """Synthesise a failed Airstriker episode with steps bound to a
    real cold-start skill_id.  ``total_reward < 0`` deterministically
    triggers OUTCOME_FAILURE in the Crafter's F2 synthesis."""
    experiences: List[Dict[str, Any]] = []
    for i in range(n_steps):
        experiences.append({
            "step": i,
            "state": f"frame-{i}: enemy formation visible, ammo=12",
            "action": "FIRE" if i % 2 == 0 else "RIGHT",
            "reward": -0.1,                   # accumulating negative reward
            "raw_env_reward": -0.1,
            "next_state": f"frame-{i+1}: enemy formation closer",
            "done": False,
            "intention": "Attack the formation before it reaches us",
            "summary_state": f"step={i} ammo=12 enemies=3",
            "skill_id": bound_skill_id,
            "board_stats": {},
        })
    # Final step: died.
    experiences[-1]["done"] = True
    experiences[-1]["reward"] = -1.0
    experiences[-1]["raw_env_reward"] = -1.0

    return _SyntheticEpisode(
        game=game,
        episode_id=episode_id,
        steps=n_steps,
        total_reward=sum(e["reward"] for e in experiences),
        terminated=True,
        truncated=False,
        experiences=experiences,
        wall_time_s=2.0,
    )


# ---------------------------------------------------------------------------
# Splice block — exact copy of the orchestrator's Phase B′ logic so that
# any drift in orchestrator.py is caught by THIS test instead of in
# production. Refactor: when the splice changes, update both.
# ---------------------------------------------------------------------------


def _exec_phase_b_prime_splice(
    *,
    step: int,
    config_run_dir: Path,
    sb_manager,
    rollout_results: List[_SyntheticEpisode],
    crafter_promotion_enabled: bool = True,
    crafter_cycle_every_k_steps: int = 0,
    crafter_outcome_failure_threshold: float = 0.0,
    crafter_promotion_timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """Run the orchestrator's Phase B′ splice block in isolation.

    The body is literally lifted from
    ``trainer/coevolution/orchestrator.py:744-826``. If you edit this
    function, also edit the orchestrator (or vice versa).
    """
    crafter_report: Optional[Dict[str, Any]] = None
    promotion_report: Optional[Dict[str, Any]] = None
    reload_report: Dict[str, Any] = {}
    error: Optional[str] = None

    if crafter_promotion_enabled:
        try:
            from trainer.coevolution._crafter_hook import run_crafter_step
            from trainer.coevolution._promotion_hook import run_promotion_step

            bank_paths = sb_manager.bank_paths(simple_only=True)
            # Mirror orchestrator.py: probe disk to handle --seed-bank-dir
            # at step 0.  Plain ``step > 0`` would silently disable
            # NO_SKILL_BOUND on a seeded cold-start run.
            bank_was_available = any(
                p.is_file() and p.stat().st_size > 0
                for p in bank_paths.values()
            )
            crafter_step = run_crafter_step(
                step=step,
                run_dir=config_run_dir,
                rollout_results=rollout_results,
                legacy_bank_paths=bank_paths,
                bank_was_available=bank_was_available,
                cycle_every_k_steps=crafter_cycle_every_k_steps,
                outcome_failure_threshold=crafter_outcome_failure_threshold,
                # Lane-(b) opt-in: integration test exercises the
                # closed-loop PatchProposal → Promotion path that was
                # the original wire-up. Lane-(a) default (T1.3a) routes
                # to Hypothesizer instead — covered by
                # test_run_crafter_step_lane_a_default_routes_to_hypothesizer.
                enable_protocol_patching=True,
                # Disable the post-v11 hypothesizer-fallthrough gates so
                # this integration test exercises the closed-loop splice
                # in isolation. Production callers keep the gate
                # defaults (3 / 0.30) — see DEFAULT_HYPOTHESIZE_*.
                hypothesize_min_recurrences=1,
                hypothesize_related_skill_jaccard=0.0,
            )
            crafter_report = crafter_step.to_dict()

            if crafter_step.n_proposals > 0:
                promotion_step = run_promotion_step(
                    step=step,
                    run_dir=config_run_dir,
                    proposals_run_dir=crafter_step.run_dir,
                    legacy_bank_paths=bank_paths,
                    driver_timeout_s=crafter_promotion_timeout_s,
                )
                promotion_report = promotion_step.to_dict()

                keys_to_reload = [
                    game for game, wb in (
                        promotion_step.writeback_per_game or {}
                    ).items()
                    if int(wb.get("n_inserted", 0)) > 0
                    or int(wb.get("n_updated", 0)) > 0
                ]
                if keys_to_reload:
                    reload_report = sb_manager.reload_banks_from_disk(
                        keys_to_reload,
                    )
                if promotion_report is not None:
                    promotion_report["bank_reload_per_game"] = reload_report
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)

    return {
        "crafter_report": crafter_report,
        "promotion_report": promotion_report,
        "reload_report": reload_report,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_fixtures(), reason="Phase-0 fixtures missing")
def test_phase_b_prime_splice_grows_bank_and_reloads(tmp_path: Path):
    """Drive ONE iteration of the orchestrator's Phase B′ block:

      sb_manager.bank_paths()
       → run_crafter_step  (real SkillCrafterService, real SkillRepository)
       → run_promotion_step (real subprocess to decide_promotion_gpt54.py)
       → legacy_writeback   (real SkillBankMVP-loadable JSONL)
       → sb_manager.reload_banks_from_disk(keys_to_reload)

    Assertions cover every contract the orchestrator depends on:

    1. crafter_report is dict-typed and reports ≥1 proposal.
    2. promotion_report is dict-typed, has matching proposals_in count,
       and writeback_per_game has a non-empty entry for our game.
    3. The on-disk skill_bank.jsonl has new lines after the run.
    4. The post-promotion bank loads cleanly through real SkillBankMVP.
    5. reload_banks_from_disk returned ``reloaded=True`` for our game.
    6. ``pipe.get_bank()`` (the call the actor makes on the *next* step)
       returns a fresh ``SkillQueryEngine`` whose
       ``_skill_id_order`` includes the promoted skill_ids.
    """
    from trainer.coevolution.skillbank_pipeline import PerGameSkillBankManager

    game = "Temporal_Airstriker-v0"
    bank_root = tmp_path / "skillbank"
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()

    # ── Build a real PerGameSkillBankManager and seed from cold-start ──
    seed_dir = _coldstart_bank_dir()
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        sb_manager = PerGameSkillBankManager(
            games=[game],
            bank_dir=str(bank_root),
            executor=executor,
            seed_bank_dir=str(seed_dir),
            unified_role_rollouts=False,
        )

        # Force agent init + load bank, so reload_banks_from_disk has
        # something to mutate. Mirror the orchestrator: get_bank() is
        # what the actor calls before each rollout.
        pipe = sb_manager._pipelines[game]
        agent = pipe._ensure_agent()
        n_seeded = len(agent.bank)
        assert n_seeded >= 5, (
            f"cold-start seed should provide ≥5 skills, got {n_seeded}"
        )
        ids_before = set(agent.bank.skill_ids)

        pre_engine = pipe.get_bank()
        assert pre_engine is not None
        pre_indexed = set(pre_engine._skill_id_order)
        assert pre_indexed == ids_before, (
            "actor would not have seen all seeded skills "
            "(SkillQueryEngine indexed a subset)"
        )

        # ── Synthesise rollout results: one losing episode per skill_id ─
        rollout_results: List[_SyntheticEpisode] = [
            _make_failed_episode(
                game=game,
                episode_id=f"ep-{sid.replace('/','_')}",
                bound_skill_id=sid,
            )
            for sid in sorted(ids_before)[:3]   # 3 different bound skills
        ]
        for r in rollout_results:
            assert r.total_reward < 0, (
                "synthetic episodes must trigger OUTCOME_FAILURE"
            )

        # ── Drive the splice ───────────────────────────────────────────
        result = _exec_phase_b_prime_splice(
            step=1,                              # >0 so bank_was_available=True
            config_run_dir=run_dir,
            sb_manager=sb_manager,
            rollout_results=rollout_results,
            crafter_promotion_timeout_s=180.0,
        )

        assert result["error"] is None, (
            f"splice raised: {result['error']}"
        )

        # ── 1. Crafter produced ≥1 proposal ────────────────────────────
        crafter = result["crafter_report"]
        assert isinstance(crafter, dict)
        assert crafter["n_episodes_reflected"] == len(rollout_results)
        assert crafter["n_failure_traces"] >= len(rollout_results), (
            f"expected ≥{len(rollout_results)} OUTCOME_FAILUREs, "
            f"got {crafter['n_failure_traces']}"
        )
        assert crafter["n_proposals"] >= 1, (
            f"Crafter should have produced ≥1 proposal "
            f"(reflected {crafter['n_episodes_reflected']} episodes, "
            f"synthesised {crafter['n_failure_traces']} traces)"
        )
        n_proposals = crafter["n_proposals"]

        # ── 2. Promotion ran end-to-end and produced a writeback report ─
        promotion = result["promotion_report"]
        assert isinstance(promotion, dict), (
            "n_proposals>0 should have triggered the Promotion hook"
        )
        assert promotion["n_proposals_in"] == n_proposals, (
            "Promotion driver saw a different proposal count than "
            "Crafter wrote — JSONL roundtrip is broken"
        )
        # Subprocess succeeded.
        assert promotion["driver_returncode"] == 0, (
            f"decide_promotion_gpt54.py exited {promotion['driver_returncode']}"
        )
        wb_per_game = promotion.get("writeback_per_game") or {}
        assert game in wb_per_game, (
            f"writeback report missing entry for {game!r}: "
            f"{list(wb_per_game)}"
        )
        wb = wb_per_game[game]
        n_inserted = int(wb.get("n_inserted", 0))
        n_updated = int(wb.get("n_updated", 0))
        # Synthetic-mode gate caps Stage 1-4 at LIMITED_PASS, so we
        # expect ≥1 PROVISIONAL skill to be promoted and written back.
        assert (n_inserted + n_updated) >= 1, (
            f"writeback for {game} produced no insertions/updates: {wb}"
        )

        # ── 3. On-disk bank grew ───────────────────────────────────────
        bank_path = bank_root / game / "skill_bank.jsonl"
        with open(bank_path, encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        assert n_lines == n_seeded + n_inserted, (
            f"on-disk bank has {n_lines} lines, expected "
            f"{n_seeded} (seed) + {n_inserted} (writeback)"
        )

        # ── 4. Bank still loads cleanly through the REAL SkillBankMVP ──
        from skill_agents.skill_bank.bank import SkillBankMVP
        verify = SkillBankMVP(str(bank_path))
        verify.load()                          # MUST NOT RAISE
        assert len(verify) == n_seeded + n_inserted

        # ── 5. Manager reloaded the bank ───────────────────────────────
        reload_rep = result["reload_report"]
        assert game in reload_rep, (
            f"manager did not reload {game}: {list(reload_rep)}"
        )
        rr = reload_rep[game]
        assert rr.get("reloaded") is True, rr
        assert rr["n_after"] == n_seeded + n_inserted
        assert rr["query_engine_invalidated"] is True

        # ── 6. The actor's next get_bank() sees the promoted skills ────
        post_engine = pipe.get_bank()
        assert post_engine is not pre_engine, (
            "actor would still hold the stale SkillQueryEngine cache"
        )
        post_indexed = set(post_engine._skill_id_order)
        new_visible = post_indexed - pre_indexed
        assert len(new_visible) >= n_inserted, (
            f"only {len(new_visible)} new skill_ids visible to actor's "
            f"next get_bank(); writeback inserted {n_inserted}"
        )
        # Spot-check: the bank's full id set matches the engine's index.
        assert post_indexed == set(verify.skill_ids), (
            "SkillQueryEngine silently dropped some on-disk skills"
        )

    finally:
        executor.shutdown(wait=True)


@pytest.mark.skipif(not _has_fixtures(), reason="Phase-0 fixtures missing")
def test_phase_b_prime_splice_disabled_is_noop(tmp_path: Path):
    """When ``crafter_promotion_enabled=False`` the splice must be a
    pure no-op: no on-disk file changes, no reload, no errors."""
    from trainer.coevolution.skillbank_pipeline import PerGameSkillBankManager

    game = "Temporal_Airstriker-v0"
    bank_root = tmp_path / "skillbank"
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        sb_manager = PerGameSkillBankManager(
            games=[game],
            bank_dir=str(bank_root),
            executor=executor,
            seed_bank_dir=str(_coldstart_bank_dir()),
            unified_role_rollouts=False,
        )
        bank_path = bank_root / game / "skill_bank.jsonl"
        size_before = bank_path.stat().st_size

        result = _exec_phase_b_prime_splice(
            step=1,
            config_run_dir=run_dir,
            sb_manager=sb_manager,
            rollout_results=[_make_failed_episode()],
            crafter_promotion_enabled=False,
        )

        assert result["error"] is None
        assert result["crafter_report"] is None
        assert result["promotion_report"] is None
        assert result["reload_report"] == {}

        # On-disk file untouched.
        assert bank_path.stat().st_size == size_before
    finally:
        executor.shutdown(wait=True)


@pytest.mark.skipif(not _has_fixtures(), reason="Phase-0 fixtures missing")
def test_phase_b_prime_step_zero_with_seeded_bank_synthesises_failures(
    tmp_path: Path,
):
    """Regression test for the ``bank_was_available=(step > 0)`` bug.

    When a run is launched with ``--seed-bank-dir`` (or otherwise resumes
    from a pre-populated bank) the actor on step 0 has skills available,
    so NO_SKILL_BOUND F2 synthesis SHOULD fire on episodes where the
    actor failed to bind one.  The original splice gated on ``step > 0``
    which silently dropped that signal on the very first step of every
    seeded run.

    This test pins the fix: even at ``step=0``, if the on-disk bank file
    has non-zero size, the Crafter must observe ``bank_was_available=True``
    and emit ≥1 failure trace from a NO_SKILL_BOUND-style episode."""
    from trainer.coevolution.skillbank_pipeline import PerGameSkillBankManager

    game = "Temporal_Airstriker-v0"
    bank_root = tmp_path / "skillbank"
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        sb_manager = PerGameSkillBankManager(
            games=[game],
            bank_dir=str(bank_root),
            executor=executor,
            seed_bank_dir=str(_coldstart_bank_dir()),
        )
        # Sanity: seed wrote a real, non-empty file before any agent init.
        bank_path = bank_root / game / "skill_bank.jsonl"
        assert bank_path.is_file()
        assert bank_path.stat().st_size > 0

        # Successful episode (positive reward) where the actor failed to
        # bind a skill on every step. Critical: reward must stay > 0 so
        # OUTCOME_FAILURE does NOT fire — otherwise this test would pass
        # for the wrong reason and not actually catch the
        # ``bank_was_available=(step > 0)`` regression.
        ep = _make_failed_episode(bound_skill_id="COMMIT/ATTACK")
        for e in ep.experiences:
            e["skill_id"] = None                  # actor couldn't bind
            e["reward"] = 0.5                     # positive: no OUTCOME_FAILURE
            e["raw_env_reward"] = 0.5
        ep.total_reward = 2.0                     # well above threshold (0.0)
        ep.terminated = False
        ep.truncated = False

        result = _exec_phase_b_prime_splice(
            step=0,                               # the regressed case
            config_run_dir=run_dir,
            sb_manager=sb_manager,
            rollout_results=[ep],
        )
        assert result["error"] is None
        crafter = result["crafter_report"]
        assert isinstance(crafter, dict)
        # The ONLY signal that can fire here is NO_SKILL_BOUND, gated on
        # bank_was_available. With the buggy ``step > 0`` heuristic this
        # would have been 0 traces. Post-fix, the disk probe sees the
        # seeded bank and at least one trace synthesises.
        assert crafter["n_failure_traces"] >= 1, (
            f"NO_SKILL_BOUND F2 signal regressed: bank was seeded but "
            f"no failure traces synthesised at step=0 ({crafter})"
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.skipif(not _has_fixtures(), reason="Phase-0 fixtures missing")
def test_phase_b_prime_step_zero_without_bank_skips_no_skill_bound(
    tmp_path: Path,
):
    """Complement to the regression above: a TRUE cold-start (no
    ``--seed-bank-dir``, empty bank) at step 0 must still NOT fire
    NO_SKILL_BOUND traces — the actor literally couldn't bind because
    the bank was empty, so this is not a failure of the actor."""
    from trainer.coevolution.skillbank_pipeline import PerGameSkillBankManager

    game = "Temporal_Airstriker-v0"
    bank_root = tmp_path / "skillbank"
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        sb_manager = PerGameSkillBankManager(
            games=[game],
            bank_dir=str(bank_root),
            executor=executor,
            seed_bank_dir=None,                  # no seed
        )
        bank_path = bank_root / game / "skill_bank.jsonl"
        # File may or may not exist; if it does, it must be empty.
        assert (not bank_path.is_file()) or bank_path.stat().st_size == 0

        ep = _make_failed_episode()
        for e in ep.experiences:
            e["skill_id"] = None                  # actor couldn't bind
            e["reward"] = 0.5                     # positive: no OUTCOME_FAILURE
            e["raw_env_reward"] = 0.5
        ep.total_reward = 1.0                     # > threshold (0.0)
        ep.terminated = False
        ep.truncated = False

        result = _exec_phase_b_prime_splice(
            step=0,
            config_run_dir=run_dir,
            sb_manager=sb_manager,
            rollout_results=[ep],
        )
        assert result["error"] is None
        crafter = result["crafter_report"]
        assert isinstance(crafter, dict)
        # OUTCOME_FAILURE didn't fire (reward > threshold) and
        # NO_SKILL_BOUND didn't fire either (bank had nothing to bind to).
        # Net: 0 failure traces.
        assert crafter["n_failure_traces"] == 0, (
            f"cold-start with empty bank should not synthesise "
            f"NO_SKILL_BOUND, got {crafter['n_failure_traces']} traces"
        )
    finally:
        executor.shutdown(wait=True)


@pytest.mark.skipif(not _has_fixtures(), reason="Phase-0 fixtures missing")
def test_phase_b_prime_splice_no_failures_skips_promotion(tmp_path: Path):
    """When all episodes are *successful* (no F2 signals fire), the
    Crafter produces zero proposals → Promotion hook MUST be skipped
    (no subprocess invocation, no writeback, no reload)."""
    from trainer.coevolution.skillbank_pipeline import PerGameSkillBankManager

    game = "Temporal_Airstriker-v0"
    bank_root = tmp_path / "skillbank"
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        sb_manager = PerGameSkillBankManager(
            games=[game],
            bank_dir=str(bank_root),
            executor=executor,
            seed_bank_dir=str(_coldstart_bank_dir()),
        )

        # Successful episode (positive reward, no F2 signals).
        ep = _make_failed_episode(bound_skill_id="COMMIT/ATTACK", n_steps=2)
        for e in ep.experiences:
            e["reward"] = 1.0
            e["raw_env_reward"] = 1.0
            e["done"] = False
        ep.experiences[-1]["done"] = True
        ep.experiences[-1]["reward"] = 5.0
        ep.experiences[-1]["raw_env_reward"] = 5.0
        ep.total_reward = sum(e["reward"] for e in ep.experiences)
        ep.terminated = True
        assert ep.total_reward > 0, "fixture should NOT trigger OUTCOME_FAILURE"

        bank_path = bank_root / game / "skill_bank.jsonl"
        size_before = bank_path.stat().st_size

        result = _exec_phase_b_prime_splice(
            step=1,
            config_run_dir=run_dir,
            sb_manager=sb_manager,
            rollout_results=[ep],
        )

        assert result["error"] is None
        crafter = result["crafter_report"]
        assert isinstance(crafter, dict)
        # Crafter ran but emitted zero proposals.
        assert crafter["n_episodes_reflected"] == 1
        assert crafter["n_proposals"] == 0
        # Promotion hook MUST have been skipped.
        assert result["promotion_report"] is None, (
            "Promotion hook ran on zero proposals — wasted subprocess"
        )
        assert result["reload_report"] == {}
        assert bank_path.stat().st_size == size_before
    finally:
        executor.shutdown(wait=True)
