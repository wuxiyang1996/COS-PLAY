#!/usr/bin/env python
"""
Harness I/O dump driver — invokes the LIVE
`harness.SkillHarness` (online surface) and `orchestrator.GateService`
(offline GateRunner surface) against the cold-start corpus and dumps
every typed input/output to disk.

The point of this driver is *validation*: it asks the question
"given the data we have today, do the harness and gate APIs even
connect end-to-end, and where do they degrade?" The dump artefacts
become the integration-test substrate the harness fixes from
`harness/README.md` §§9–14 will be ratified against.

Mirror script of `reflect_per_episode_gpt54.py` (which dumps the
Crafter's behaviour); read that file alongside this one to understand
the per-(corpus, source) plumbing pattern.

Surfaces
--------
``--surface online``  — replay every step of every episode through:
    1. ``harness.select_eligible_skills(...)``
    2. ``harness.validate_invocation(...)``  (degraded stub today;
       see §9.1 of harness/README.md — the dump records the gap and
       continues)
    3. ``harness.run_skill(...)``            (only with --run-skill;
       expensive, behind a flag)
   Compare each step's eligible-set against the actor's actual
   ``skill_query.selected_skill_id`` to surface "harness vetoed what
   the actor picked" anomalies.

``--surface offline`` — for every typed `BankMutationProposal` in
   ``labeling_supplement/{crafter_proposals_out,episode_reflections_out}``:
    1. Load the target `SkillRecord` from the per-source bank.
    2. Synthesise replay seeds, shadow log, and (baseline, post)
       scalars from the cold-start corpus.
    3. Call ``GateService.evaluate(proposal=..., skill=..., ...)``
       and dump per-stage `StageVerdict`s + the assembled
       `SkillEvaluationRecord`.

``--surface both`` (default) does both.

Per-(corpus, source) flow
-------------------------
For each ``(corpus, source)`` pair:
  1. Load the FROZEN per-source skill bank into a fresh temp
     `SkillRepository`. Promote every skill to ``PROVISIONAL`` (not
     ``ACTIVE``) so the eligibility filter — which gates on
     ``status ∈ {ACTIVE, SHADOW, PROVISIONAL}`` — actually returns
     them WITHOUT triggering the ``≥2 feasible_domains`` invariant
     that ACTIVE imposes (most cold-start skills declare a single
     ``feasible_domain=["gymv"]``). This is a dump-only convention
     and is logged in ``_source_summary.json.thresholds``.
  2. Build a fresh `AdapterRegistry` with all five adapters
     registered (gymv source + four transfer targets). The default
     deterministic stub executor is used; real env binding can be
     plugged in via ``--with-real-executors`` (TODO — currently a no-op
     placeholder).
  3. Build a `SkillHarness` around the registry. Build a
     `GateService` around the harness.
  4. Run the requested surfaces.
  5. Tear down the temp repo (outputs are persisted under
     ``--output-dir``).

Outputs
-------
``<out>/<corpus>/<source>/online/episode_<NNN>/harness_io.json``
    Per-episode dump of every step's eligibility / validate /
    run_skill I/O.

``<out>/<corpus>/<source>/online/_online_summary.json``
    Per-source histogram (n_eligible distribution, agreement-rate,
    veto-reason histogram).

``<out>/<corpus>/<source>/offline/proposals/<proposal_id>/evaluation.json``
    The signed `SkillEvaluationRecord` (final verdict + per-stage
    metrics) plus the typed proposal that was evaluated.

``<out>/<corpus>/<source>/offline/proposals/<proposal_id>/stages.jsonl``
    One line per `StageVerdict`, mirroring the per-stage payload
    structure expected by `PromotionOrchestrator`.

``<out>/<corpus>/<source>/offline/_offline_summary.json``
    Per-source histogram (per-stage pass/fail/limited_pass counts,
    n_proposals_with_missing_skill).

``<out>/_run_meta.json`` and ``<out>/_run_summary.json``
    Run-level rollup.

Usage
-----

    # Process every (corpus, source) pair, both surfaces, default sizes.
    python labeling_supplement/dump_harness_io_gpt54.py \\
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \\
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \\
        --crafter-proposals-run \\
            labeling_supplement/crafter_proposals_out/run_<ts> \\
        --reflections-run \\
            labeling_supplement/episode_reflections_out/run_<ts> \\
        --output-dir   labeling_supplement/harness_io_out/run_<ts>

    # Smoke: one source, two episodes, three proposals, online only.
    python labeling_supplement/dump_harness_io_gpt54.py \\
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \\
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \\
        --corpus       env_wrappers --source twenty_forty_eight \\
        --max-episodes 2 --max-steps 5 --max-proposals 3 \\
        --surface online -v

The companion bash dispatcher ``run_dump_harness_io.sh`` fans this out
one worker per ``(corpus, source)``.

Cross-refs
----------
* ``harness/README.md``  §§9–14   (the gaps this driver surfaces)
* ``implementation_notes/legacy/crafter-harness-orchestrator-roles.md``
                                  (component boundaries the driver mirrors)
* ``labeling_supplement/reflect_per_episode_gpt54.py``
                                  (pattern this script copies for plumbing)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup so the script runs from any cwd (mirror sibling drivers).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Project imports — these load the LIVE Harness + Gate code.
# ---------------------------------------------------------------------------
from common.enums import GateStage, GateVerdict, SkillStatus           # noqa: E402
from harness import SkillHarness                                       # noqa: E402
from harness.adapter_registry import AdapterRegistry                   # noqa: E402
from harness.adapters import (                                         # noqa: E402
    BrowserAdapter,
    GymvAdapter,
    OsworldAdapter,
    VideoAdapter,
    VisualReasoningAdapter,
)
from harness.gate_runner import EvalSuite, GateRunner, GateRunnerConfig  # noqa: E402
from orchestrator.gate_service import GateService                      # noqa: E402
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore  # noqa: E402
from skill_bank.stores import StoreName                                # noqa: E402

# Local helpers (dump-driver-only).
from labeling_supplement._harness_io_helpers import (                  # noqa: E402
    CORPORA,
    baseline_post_from_summary,
    diagnose_agreement,
    load_bank_records,
    load_proposals,
    parse_online_step,
    safe_skill_id,
    seed_lifecycle,
    synthesize_replay_seeds,
    synthesize_shadow_log,
    synthesize_target_skill_for_proposal,
)

logger = logging.getLogger("labeling_supplement.dump_harness_io")


# ─────────────────────────────────────────────────────────────────────────
# Defaults (override via CLI)
# ─────────────────────────────────────────────────────────────────────────
DEFAULT_BANK_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_bank_out" / "run_20260430_030637"
)
DEFAULT_ACTIONS_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_actions_out" / "run_20260430_064325"
)
DEFAULT_OUTPUT_ROOT = (
    CODEBASE_ROOT / "labeling_supplement" / "harness_io_out"
)


# ─────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_registry() -> AdapterRegistry:
    """Construct an `AdapterRegistry` with all five domain adapters
    registered. Uses the default deterministic-stub executors; real env
    binding (e.g. `gymv_wrapper.adapter.set_executor`) is the
    responsibility of a future ``--with-real-executors`` plumbing path
    (PLAN-COMPONENTS-IMPLEMENTATION §4 Phase A.5).
    """
    reg = AdapterRegistry()
    reg.register(GymvAdapter())
    reg.register(BrowserAdapter())
    reg.register(OsworldAdapter())
    reg.register(VideoAdapter())
    reg.register(VisualReasoningAdapter())
    return reg


def _infer_domain(corpus: str, source: str) -> str:
    """Best-effort domain id for the bank lifecycle.

    The on-disk bank uses ``"gymv"`` for both gym_v envs and
    env_wrappers games (Phase-1 single-domain cold-start corpus).
    """
    return "gymv"


# ─────────────────────────────────────────────────────────────────────────
# Online surface
# ─────────────────────────────────────────────────────────────────────────


def _validate_invocation_real(
    *, harness: "SkillHarness", skill, state, bindings: Optional[Dict[str, Any]],
    eligible: Optional[Any] = None,
) -> Dict[str, Any]:
    """Day-7e: real call into `SkillHarness.validate_invocation` (§9.1
    closed). Returns the `ValidateInvocationResult.to_json()` payload
    plus the legacy ``{veto, veto_reason, source}`` keys so the dump
    schema is back-compat with previous runs.
    """
    res = harness.validate_invocation(
        skill, state, bindings=bindings or {}, eligible=eligible,
    )
    j = res.to_json()
    j.update({
        "veto": not res.ok,
        "veto_reason": (
            "; ".join(res.veto_reasons) if res.veto_reasons else None
        ),
        "source": "harness.validate_invocation",
        "status": "ok",
    })
    return j


def _process_online_episode(
    *,
    harness: SkillHarness,
    skills_by_id: Dict[str, Any],
    episode_path: Path,
    out_dir: Path,
    domain: str,
    max_steps: Optional[int],
    run_skill_enabled: bool,
) -> Dict[str, Any]:
    """Walk every step of one rollout episode through the online
    Harness API surface and dump the I/O.

    Returns a compact per-episode summary (full dump on disk).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(episode_path.read_text())
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("%s: episode load failed: %s", episode_path.name, exc)
        return {
            "episode": episode_path.name,
            "status": "load_failed",
            "error": str(exc),
            "n_steps_dumped": 0,
        }

    experiences = data.get("experiences") or data.get("steps") or []
    if max_steps is not None:
        experiences = experiences[:max_steps]

    step_records: List[Dict[str, Any]] = []
    n_eligible_hist: Counter[int] = Counter()
    agreement_hist: Counter[str] = Counter()
    candidate_miss_count = 0
    run_skill_outcomes: Counter[str] = Counter()
    n_validate_vetoed = 0

    for step in experiences:
        try:
            inputs = parse_online_step(step, fallback_domain=domain)
        except Exception as exc:                                   # noqa: BLE001
            step_records.append({
                "status": "parse_failed",
                "error": str(exc),
                "raw_step_keys": sorted(step.keys()) if isinstance(step, dict) else [],
            })
            continue

        # Resolve actor's retrieved candidates against the per-source bank.
        candidates: List[Any] = []
        miss: List[str] = []
        for sid in inputs.retrieved_skill_ids:
            rec = skills_by_id.get(sid)
            if rec is None:
                miss.append(sid)
                continue
            candidates.append(rec)
        candidate_miss_count += len(miss)

        # Online surface §1: select_eligible_skills.
        # NB: §9.2 of harness/README.md flags that the live API takes
        # only `(candidates, state, skill_type_hint)` today — the
        # spec'd `intention / active_skill / local_reasoning_trace`
        # parameters are not yet plumbed. We still record those raw
        # inputs in the dump so the §9.2 fix has a per-step sample to
        # plug into.
        try:
            eligible = harness.select_eligible_skills(
                candidates=candidates,
                state=inputs.state,
            )
            eligible_payload = [es.to_json() for es in eligible]
            eligible_ids = [es.skill.skill_id for es in eligible]
            sel_status = "ok"
            sel_error = None
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("select_eligible_skills failed at step %d: %s",
                         inputs.step_idx, exc)
            eligible = []
            eligible_payload = []
            eligible_ids = []
            sel_status = "error"
            sel_error = repr(exc)

        n_eligible_hist[len(eligible_ids)] += 1
        agreement = diagnose_agreement(eligible_ids, inputs.selected_skill_id)
        agreement_hist[agreement["agreement"]] += 1

        # Online surface §2: validate_invocation (Day-7e real call —
        # harness/README §9.1 closed). Picks up the matching
        # EligibleSkill so `shadow_only` propagates faithfully.
        validate_payload: Optional[Dict[str, Any]] = None
        if inputs.selected_skill_id and inputs.selected_skill_id in skills_by_id:
            matching_eligible = next(
                (es for es in eligible
                 if es.skill.skill_id == inputs.selected_skill_id),
                None,
            )
            try:
                validate_payload = _validate_invocation_real(
                    harness=harness,
                    skill=skills_by_id[inputs.selected_skill_id],
                    state=inputs.state,
                    bindings=None,
                    eligible=matching_eligible,
                )
            except Exception as exc:                                    # noqa: BLE001
                logger.debug(
                    "validate_invocation failed at step %d: %s",
                    inputs.step_idx, exc,
                )
                validate_payload = {
                    "veto": True,
                    "veto_reason": f"call_raised: {exc!r}",
                    "source": "harness.validate_invocation",
                    "status": "error",
                }
            if validate_payload.get("veto"):
                n_validate_vetoed += 1

        # Online surface §3: run_skill (opt-in via --run-skill).
        run_skill_payload: Optional[Dict[str, Any]] = None
        if run_skill_enabled and inputs.selected_skill_id and inputs.selected_skill_id in skills_by_id:
            try:
                ep = harness.run_skill(
                    skill=skills_by_id[inputs.selected_skill_id],
                    state=inputs.state,
                    parent_run_id=str(episode_path),
                )
                outcome_kind = (
                    "success" if (ep.outcome and ep.outcome.success)
                    else ("contract_unsatisfied"
                          if (ep.outcome and not ep.outcome.contract_satisfied)
                          else "no_outcome")
                )
                run_skill_outcomes[outcome_kind] += 1
                run_skill_payload = {
                    "status": "ok",
                    "episode_id": ep.episode_id,
                    "outcome": ep.outcome.to_json() if ep.outcome else None,
                    "n_steps": len(ep.steps),
                    "transfer_label": ep.transfer_label,
                }
            except Exception as exc:                               # noqa: BLE001
                run_skill_outcomes["error"] += 1
                run_skill_payload = {"status": "error", "error": repr(exc)}

        step_records.append({
            "step_idx": inputs.step_idx,
            "input": {
                "state": inputs.state.to_json(),
                "intention": inputs.intention,
                "retrieved_skill_ids": list(inputs.retrieved_skill_ids),
                "candidate_misses": list(miss),
                "selected_skill_id": inputs.selected_skill_id,
                "actor_action": inputs.actor_action,
                "reward": inputs.reward,
                "raw_step_keys": inputs.raw_step_keys,
            },
            "select_eligible_skills": {
                "status": sel_status,
                "error": sel_error,
                "eligible_skill_ids": eligible_ids,
                "eligible_payload": eligible_payload,
            },
            "validate_invocation": validate_payload,
            "run_skill": run_skill_payload,
            "agreement_with_actor": agreement,
        })

    (out_dir / "harness_io.json").write_text(
        json.dumps(
            {
                "episode": episode_path.name,
                "domain": domain,
                "n_experiences_total": len(data.get("experiences") or []),
                "n_steps_dumped": len(step_records),
                "steps": step_records,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return {
        "episode": episode_path.name,
        "status": "ok",
        "n_steps_dumped": len(step_records),
        "n_eligible_histogram": {str(k): v for k, v in sorted(n_eligible_hist.items())},
        "agreement_histogram": dict(agreement_hist),
        "candidate_miss_count": candidate_miss_count,
        "n_validate_vetoed": n_validate_vetoed,
        "run_skill_outcomes": dict(run_skill_outcomes),
    }


def _process_online_surface(
    *,
    harness: SkillHarness,
    skills_by_id: Dict[str, Any],
    actions_dir: Path,
    out_dir: Path,
    domain: str,
    max_episodes: Optional[int],
    max_steps: Optional[int],
    run_skill_enabled: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = sorted(actions_dir.glob("episode_*.json"))
    if max_episodes is not None:
        eps = eps[:max_episodes]
    if not eps:
        return {
            "status": "no_episodes",
            "actions_dir": str(actions_dir),
            "n_episodes": 0,
        }

    rows: List[Dict[str, Any]] = []
    grand_eligible: Counter[int] = Counter()
    grand_agreement: Counter[str] = Counter()
    grand_run_skill: Counter[str] = Counter()
    total_misses = 0
    total_vetoed = 0

    for ep_path in eps:
        ep_out = out_dir / ep_path.stem
        row = _process_online_episode(
            harness=harness,
            skills_by_id=skills_by_id,
            episode_path=ep_path,
            out_dir=ep_out,
            domain=domain,
            max_steps=max_steps,
            run_skill_enabled=run_skill_enabled,
        )
        rows.append(row)
        if row.get("status") == "ok":
            for k, v in (row.get("n_eligible_histogram") or {}).items():
                grand_eligible[int(k)] += v
            for k, v in (row.get("agreement_histogram") or {}).items():
                grand_agreement[k] += v
            for k, v in (row.get("run_skill_outcomes") or {}).items():
                grand_run_skill[k] += v
            total_misses += row.get("candidate_miss_count", 0)
            total_vetoed += row.get("n_validate_vetoed", 0)

    summary = {
        "status": "ok",
        "actions_dir": str(actions_dir),
        "n_episodes": len(eps),
        "n_steps_dumped": sum(r.get("n_steps_dumped", 0) for r in rows),
        "n_eligible_histogram": {str(k): v for k, v in sorted(grand_eligible.items())},
        "agreement_histogram": dict(grand_agreement),
        "run_skill_outcomes": dict(grand_run_skill),
        "candidate_miss_count": total_misses,
        "n_validate_vetoed": total_vetoed,
        "validate_invocation_status": "real_harness_validate_invocation",
        "select_eligible_intention_threading": "not_plumbed_pending_harness_§9.2",
        "per_episode": rows,
    }
    (out_dir / "_online_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────
# Offline surface (GateRunner-equivalent)
# ─────────────────────────────────────────────────────────────────────────


def _process_offline_proposal(
    *,
    gate: GateService,
    proposal,
    skills_by_id: Dict[str, Any],
    sub_episodes_json: Path,
    actions_dir: Path,
    skill_actions_summary: Path,
    out_dir: Path,
    domain: str,
    max_replay_seeds: int,
    max_shadow_episodes: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the *source* skill the proposal references (may be empty
    # for net-new proposals like Compose).
    source_id = (
        getattr(proposal, "base_skill_id", "")
        or getattr(proposal, "target_skill_id", "")
        or (proposal.parent_skill_ids[0] if proposal.parent_skill_ids else "")
    )
    source_skill = skills_by_id.get(source_id) if source_id else None
    proposal_kind = type(proposal).__name__

    # Synthesise the *target* skill for non-Patch / non-Retire proposals.
    # This mirrors what the live PromotionOrchestrator does before the
    # gate sees a Generalize/Compose/Hypothesis proposal — without it,
    # Stage 0 produces false-positive `source_type mismatch` and
    # `feasible_domains < 2` failures (the source skill was never claimed
    # to be transferable; the *new* target skill is).
    try:
        skill, synth_debug = synthesize_target_skill_for_proposal(
            proposal,
            source_skill=source_skill,
            default_domain=domain,
        )
    except Exception as exc:                                       # noqa: BLE001
        logger.debug(
            "target-skill synthesis failed for %s: %s",
            proposal.proposal_id, exc,
        )
        skill = None
        synth_debug = {
            "synthesised": False,
            "kind": type(proposal).__name__,
            "reason": "synthesis_raised",
            "error": repr(exc),
        }
    target_id = skill.skill_id if skill is not None else (source_id or "")

    # ── Skill missing — record gap & emit a "no_skill" verdict shell.
    if skill is None:
        record = {
            "proposal_id": proposal.proposal_id,
            "proposal_kind": proposal_kind,
            "target_skill_id": target_id,
            "status": "skill_not_in_bank",
            "evaluation": None,
            "stages": [],
            "inputs": {
                "n_replay_seeds": 0,
                "n_shadow_episodes": 0,
                "baseline_score": None,
                "post_score": None,
                "few_shot_demos": {},
            },
            "target_skill_synthesis": synth_debug,
        }
        (out_dir / "evaluation.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True)
        )
        return {
            "proposal_id": proposal.proposal_id,
            "proposal_kind": proposal_kind,
            "target_skill_id": target_id,
            "status": "skill_not_in_bank",
            "stage_verdicts": {},
            "final_verdict": None,
        }

    # ── Build stage inputs ────────────────────────────────────────────
    # NB: For Generalize/Compose/Hypothesis we synthesised a target skill,
    # so replay-seeds (which key on the *source* skill_id in
    # sub_episodes.json) must use the source. Keep `seed_skill_id`
    # explicit so the dump can flag the substitution.
    seed_skill_id = source_skill.skill_id if source_skill is not None else skill.skill_id
    replay_seeds = synthesize_replay_seeds(
        sub_episodes_json,
        skill_id=seed_skill_id,
        domain=domain,
        n_max=max_replay_seeds,
    )
    shadow_log = synthesize_shadow_log(
        actions_dir,
        max_episodes=max_shadow_episodes,
        domain=domain,
    )
    baseline, post = baseline_post_from_summary(skill_actions_summary)

    # Few-shot demos: §12.2 of harness/README.md flags that synthesising
    # cross-domain demos requires real per-domain rollouts that don't
    # exist yet in the cold-start corpus. We pass an empty mapping;
    # `_run_transfer` degrades to the "few_shot_skipped:no_targets" /
    # "few_shot_skipped:no_demos" branch and the dump records the gap.
    few_shot_demos: Dict[str, Any] = {}

    # ── Run the gate ─────────────────────────────────────────────────
    eval_status = "ok"
    eval_error: Optional[str] = None
    evaluation = None
    try:
        evaluation = gate.evaluate(
            proposal=proposal,
            skill=skill,
            replay_seeds=replay_seeds,
            shadow_log=shadow_log,
            baseline_score=baseline,
            post_score=post,
            few_shot_demos=few_shot_demos,
        )
    except Exception as exc:                                       # noqa: BLE001
        eval_status = "gate_error"
        eval_error = repr(exc)
        logger.debug(
            "gate.evaluate failed for %s: %s\n%s",
            proposal.proposal_id, exc, traceback.format_exc(),
        )

    # ── Serialize ────────────────────────────────────────────────────
    # Tag synthetic / vacuous metrics so a downstream consumer can tell
    # the difference between a real Stage-N pass and a placeholder pass.
    metric_warnings: List[str] = []
    if replay_seeds:
        # ReplayValidator dispatches each seed through the registered
        # adapter in `dry_run=True` mode. Today every concrete adapter
        # uses a deterministic stub executor that always returns
        # ok=True, so `replay.pass_rate` is structurally 1.0 regardless
        # of whether the proposal actually re-binds correctly. Until a
        # real adapter executor is wired in (see harness/README.md
        # work-order item 13), Stage 1 metrics are vacuous.
        metric_warnings.append(
            "stage_1_replay_metrics_vacuous: ReplayValidator dispatches "
            "to a deterministic-stub adapter executor — pass_rate=1.0 "
            "is structural, not a real signal"
        )
    if baseline is not None and post is not None and abs(post - baseline) < 1e-12:
        metric_warnings.append(
            "stage_4_non_regression_metrics_vacuous: baseline=post by "
            "construction (mean_confidence_per_episode used as both pre "
            "and post) — delta=0 is a placeholder until a frozen eval "
            "suite exists (harness/README.md §11/§12)"
        )
    if synth_debug.get("synthesised"):
        metric_warnings.append(
            f"target_skill_synthesised_for_{proposal_kind}: dump driver "
            f"materialised a draft target SkillRecord from the proposal "
            f"so Stage 0 sees the proposed shape (not the source skill)"
        )

    record: Dict[str, Any] = {
        "proposal_id": proposal.proposal_id,
        "proposal_kind": proposal_kind,
        "target_skill_id": skill.skill_id,
        "skill_status": skill.status.value,
        "skill_feasible_domains": list(skill.feasible_domains),
        "skill_source_type": skill.source_type.value,
        "source_skill_id": (source_skill.skill_id if source_skill else None),
        "target_skill_synthesis": synth_debug,
        "status": eval_status,
        "error": eval_error,
        "inputs": {
            "n_replay_seeds": len(replay_seeds),
            "replay_seed_lookup_skill_id": seed_skill_id,
            "n_shadow_episodes_logged": (
                len(list(shadow_log.filter(skill_id=seed_skill_id)))
                if shadow_log is not None else 0
            ),
            "baseline_score": baseline,
            "post_score": post,
            "few_shot_demos_provided": {
                k: len(v) for k, v in few_shot_demos.items()
            },
            "few_shot_demos_status": "empty_pending_corpus_extension",
        },
        "evaluation": evaluation.to_json() if evaluation is not None else None,
        "metric_warnings": metric_warnings,
        "reproducibility_anchors_status": (
            "incomplete_pending_harness_§11_+_§12.1 — "
            "SkillEvaluationRecord missing bank_snapshot_id, "
            "eval_suite_id, adapter_versions, ontology_version, "
            "status_before/after, rejected_domains, rollback_target"
        ),
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True)
    )

    stage_verdicts: Dict[str, str] = {}
    if evaluation is not None and evaluation.verdict is not None:
        with (out_dir / "stages.jsonl").open("w") as f:
            for s in evaluation.verdict.stages:
                f.write(json.dumps(s.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
                stage_verdicts[s.stage.value] = s.verdict.value
    return {
        "proposal_id": proposal.proposal_id,
        "proposal_kind": proposal_kind,
        "target_skill_id": skill.skill_id,
        "status": eval_status,
        "stage_verdicts": stage_verdicts,
        "final_verdict": (
            evaluation.verdict.final_verdict.value
            if (evaluation is not None and evaluation.verdict is not None)
            else None
        ),
        "n_replay_seeds": len(replay_seeds),
    }


def _process_offline_surface(
    *,
    gate: GateService,
    skills_by_id: Dict[str, Any],
    proposals: List[Any],
    sub_episodes_json: Path,
    actions_dir: Path,
    skill_actions_summary: Path,
    out_dir: Path,
    domain: str,
    max_proposals: Optional[int],
    max_replay_seeds: int,
    max_shadow_episodes: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir = out_dir / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)

    if max_proposals is not None:
        proposals = proposals[:max_proposals]

    rows: List[Dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    by_final: Counter[str] = Counter()
    per_stage: Dict[str, Counter[str]] = {}
    n_skill_missing = 0
    n_gate_error = 0

    for prop in proposals:
        # Use proposal_id (sanitised) for the on-disk dir name.
        safe_pid = (prop.proposal_id or "_").replace("/", "_")[:120]
        prop_out = proposals_dir / safe_pid
        row = _process_offline_proposal(
            gate=gate,
            proposal=prop,
            skills_by_id=skills_by_id,
            sub_episodes_json=sub_episodes_json,
            actions_dir=actions_dir,
            skill_actions_summary=skill_actions_summary,
            out_dir=prop_out,
            domain=domain,
            max_replay_seeds=max_replay_seeds,
            max_shadow_episodes=max_shadow_episodes,
        )
        rows.append(row)
        by_kind[row["proposal_kind"]] += 1
        if row["status"] == "skill_not_in_bank":
            n_skill_missing += 1
        elif row["status"] == "gate_error":
            n_gate_error += 1
        if row.get("final_verdict"):
            by_final[row["final_verdict"]] += 1
        for stage_name, verdict in (row.get("stage_verdicts") or {}).items():
            per_stage.setdefault(stage_name, Counter())[verdict] += 1

    summary = {
        "status": "ok",
        "n_proposals": len(rows),
        "n_skill_missing": n_skill_missing,
        "n_gate_error": n_gate_error,
        "by_proposal_kind": dict(by_kind),
        "by_final_verdict": dict(by_final),
        "per_stage_verdict_histogram": {
            k: dict(v) for k, v in per_stage.items()
        },
        "stage_signature_status": (
            "Stage 2 takes RewardLogger (dump uses synthetic log); "
            "Stage 4 takes scalars (dump uses mean_confidence proxy). "
            "Both are §12 fixes pending."
        ),
        "per_proposal": rows,
    }
    (out_dir / "_offline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────
# Per-source driver
# ─────────────────────────────────────────────────────────────────────────


def _process_source(
    *,
    corpus: str,
    source: str,
    bank_run: Path,
    actions_run: Path,
    crafter_proposals_run: Optional[Path],
    reflections_run: Optional[Path],
    output_root: Path,
    surfaces: Tuple[str, ...],
    max_episodes: Optional[int],
    max_steps: Optional[int],
    max_proposals: Optional[int],
    max_replay_seeds: int,
    max_shadow_episodes: int,
    run_skill_enabled: bool,
    promote_status: SkillStatus,
    gate_runner_enabled: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    src_actions = actions_run / corpus / source
    src_bank = bank_run / corpus / source
    bank_path = src_bank / "skill_bank.jsonl"
    sub_episodes_json = src_bank / "sub_episodes.json"
    actions_summary = src_actions / "_skill_actions_summary.json"

    out_src = output_root / corpus / source
    out_src.mkdir(parents=True, exist_ok=True)

    # Empty / missing data handling
    if not bank_path.exists() or not src_actions.exists():
        result = {
            "corpus": corpus,
            "source": source,
            "status": "missing_inputs",
            "bank_path_exists": bank_path.exists(),
            "actions_dir_exists": src_actions.exists(),
        }
        (out_src / "_source_summary.json").write_text(json.dumps(result, indent=2))
        return result

    domain = _infer_domain(corpus, source)

    # ── Temp lifecycle / harness / gate ─────────────────────────────
    temp_root = Path(tempfile.mkdtemp(prefix=f"harness_io_{corpus}_{source}_"))
    try:
        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(temp_root / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(temp_root / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(temp_root / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(temp_root / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)

        bank_records = load_bank_records(bank_path, default_domain=domain)
        n_seeded, n_seed_skipped = seed_lifecycle(
            lifecycle, bank_records, promote_to=promote_status,
        )
        # Re-fetch records (their status/store may have changed) keyed by id.
        skills_by_id: Dict[str, Any] = {}
        for rec in bank_records:
            cur = repo.get(rec.skill_id)
            if cur is not None:
                skills_by_id[rec.skill_id] = cur

        registry = _build_registry()
        harness = SkillHarness(registry=registry)
        # Day-7e: switch to GateRunner so the persisted
        # SkillEvaluationRecords carry the §11 reproducibility anchors
        # (bank_snapshot_id, eval_suite_id, adapter_versions,
        # ontology_version). Old behaviour (no anchors) is preserved
        # when --gate-runner is off; with the flag, a default
        # GateRunnerConfig is constructed from the dump driver's run
        # context (snapshot id derived from the bank path stem; eval
        # suite id derived from the actions dir stem).
        if gate_runner_enabled:
            gr_config = GateRunnerConfig(
                bank_snapshot_id=f"dump:{Path(bank_path).parent.name}",
                eval_suite_id=f"cold_start:{src_actions.parent.name}",
                adapter_versions={a.name: "v1" for a in registry.all()},
                ontology_version="cold_start_v1",
                seed=0,
                judge_model="dump_driver",
            )
            gate: Any = GateRunner(harness=harness, config=gr_config)
        else:
            gate = GateService(harness=harness)

        run_summary: Dict[str, Any] = {
            "corpus": corpus,
            "source": source,
            "domain": domain,
            "status": "ok",
            "bank_path": str(bank_path),
            "actions_dir": str(src_actions),
            "n_skills_in_bank": len(bank_records),
            "n_skills_seeded": n_seeded,
            "n_skills_skipped_in_seed": n_seed_skipped,
            "promote_status_used": promote_status.value,
            "surfaces_requested": list(surfaces),
            "online": None,
            "offline": None,
            "started_at": _utcnow_iso(),
        }

        # ── Online surface ─────────────────────────────────────────
        if "online" in surfaces:
            on_dir = out_src / "online"
            run_summary["online"] = _process_online_surface(
                harness=harness,
                skills_by_id=skills_by_id,
                actions_dir=src_actions,
                out_dir=on_dir,
                domain=domain,
                max_episodes=max_episodes,
                max_steps=max_steps,
                run_skill_enabled=run_skill_enabled,
            )

        # ── Offline surface ────────────────────────────────────────
        if "offline" in surfaces:
            off_dir = out_src / "offline"
            proposal_load = load_proposals(
                crafter_proposals_run,
                reflections_run,
                corpus,
                source,
                max_per_source=max_proposals,
            )
            run_summary["offline"] = {
                "n_proposals_loaded": len(proposal_load.proposals),
                "n_proposals_skipped": proposal_load.n_skipped,
                "by_proposal_kind_in": dict(proposal_load.by_kind),
                "by_source_file": dict(proposal_load.by_source_file),
                **_process_offline_surface(
                    gate=gate,
                    skills_by_id=skills_by_id,
                    proposals=proposal_load.proposals,
                    sub_episodes_json=sub_episodes_json,
                    actions_dir=src_actions,
                    skill_actions_summary=actions_summary,
                    out_dir=off_dir,
                    domain=domain,
                    max_proposals=max_proposals,
                    max_replay_seeds=max_replay_seeds,
                    max_shadow_episodes=max_shadow_episodes,
                ),
            }

        run_summary["finished_at"] = _utcnow_iso()
        run_summary["elapsed_sec"] = round(time.time() - t0, 3)
        (out_src / "_source_summary.json").write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=False, sort_keys=True)
        )
        return run_summary
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# Discovery (mirrors reflect_per_episode_gpt54._discover_pairs)
# ─────────────────────────────────────────────────────────────────────────


def _discover_pairs(
    bank_run: Path,
    actions_run: Path,
    corpus_filter: Optional[str],
    source_filter: Optional[str],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for corpus in CORPORA:
        if corpus_filter and corpus != corpus_filter:
            continue
        cdir_b = bank_run / corpus
        cdir_a = actions_run / corpus
        if not cdir_b.exists() or not cdir_a.exists():
            continue
        for src_dir in sorted(cdir_b.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("_"):
                continue
            if source_filter and src_dir.name != source_filter:
                continue
            if not (src_dir / "skill_bank.jsonl").exists():
                continue
            actions_dir = cdir_a / src_dir.name
            if not actions_dir.exists():
                continue
            if not list(actions_dir.glob("episode_*.json")):
                continue
            out.append((corpus, src_dir.name))
    return out


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bank-run", type=Path, default=DEFAULT_BANK_RUN,
                   help="Skill-bank snapshot directory.")
    p.add_argument("--actions-run", type=Path, default=DEFAULT_ACTIONS_RUN,
                   help="Skill-actions snapshot directory (per-episode JSONs).")
    p.add_argument("--crafter-proposals-run", type=Path, default=None,
                   help=("labeling_supplement/crafter_proposals_out/run_<ts> "
                         "directory (rule-based proposals)."))
    p.add_argument("--reflections-run", type=Path, default=None,
                   help=("labeling_supplement/episode_reflections_out/run_<ts> "
                         "directory (live-Crafter proposals)."))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output root; defaults to "
                        "<DEFAULT_OUTPUT_ROOT>/run_<UTC stamp>.")

    p.add_argument("--surface", choices=("online", "offline", "both"),
                   default="both",
                   help="Which harness surface(s) to dump.")

    p.add_argument("--corpus", type=str, default=None,
                   help="If set, only process this corpus (gym_v|env_wrappers).")
    p.add_argument("--source", type=str, default=None,
                   help="If set, only process this source (e.g. twenty_forty_eight).")

    p.add_argument("--max-episodes", type=int, default=None,
                   help="Per-source cap on rollout episodes (online surface).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Per-episode cap on inner steps (online surface).")
    p.add_argument("--max-proposals", type=int, default=None,
                   help="Per-source cap on proposals (offline surface).")
    p.add_argument("--max-replay-seeds", type=int, default=4,
                   help="Per-proposal cap on replay seed episodes (offline Stage 1).")
    p.add_argument("--max-shadow-episodes", type=int, default=5,
                   help="Per-proposal cap on rollout episodes "
                        "synthesised into the shadow log (offline Stage 2).")

    p.add_argument("--run-skill", action="store_true",
                   help="Enable harness.run_skill (online surface, expensive).")
    p.add_argument("--no-force-runnable", action="store_true",
                   help="Leave seeded skills at CANDIDATE (default: PROVISIONAL). "
                        "Useful to surface 'all CANDIDATE → 0 eligible' "
                        "as a diagnostic instead of bypassing it.")
    p.add_argument(
        "--gate-runner",
        action="store_true",
        help=(
            "Day-7e: switch the offline surface from "
            "`orchestrator.GateService` to `harness.GateRunner` so the "
            "persisted SkillEvaluationRecords carry §11 reproducibility "
            "anchors (bank_snapshot_id, eval_suite_id, adapter_versions, "
            "ontology_version). Default off — back-compat with prior "
            "dump runs."
        ),
    )

    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v INFO; -vv DEBUG.")
    return p


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else (
        logging.INFO if verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    bank_run: Path = args.bank_run.resolve()
    actions_run: Path = args.actions_run.resolve()
    crafter_proposals_run: Optional[Path] = (
        args.crafter_proposals_run.resolve() if args.crafter_proposals_run else None
    )
    reflections_run: Optional[Path] = (
        args.reflections_run.resolve() if args.reflections_run else None
    )

    if args.output_dir is not None:
        out_root: Path = args.output_dir.resolve()
    else:
        out_root = DEFAULT_OUTPUT_ROOT / f"run_{_utc_run_stamp()}"
    out_root.mkdir(parents=True, exist_ok=True)

    surfaces: Tuple[str, ...] = (
        ("online", "offline") if args.surface == "both" else (args.surface,)
    )
    promote_status = (
        SkillStatus.CANDIDATE if args.no_force_runnable else SkillStatus.PROVISIONAL
    )

    pairs = _discover_pairs(bank_run, actions_run, args.corpus, args.source)
    if not pairs:
        logger.error(
            "No (corpus, source) pairs found under bank_run=%s and actions_run=%s "
            "with corpus=%s source=%s",
            bank_run, actions_run, args.corpus, args.source,
        )
        return 2

    run_meta = {
        "started_at": _utcnow_iso(),
        "argv": list(sys.argv),
        "bank_run": str(bank_run),
        "actions_run": str(actions_run),
        "crafter_proposals_run": str(crafter_proposals_run) if crafter_proposals_run else None,
        "reflections_run": str(reflections_run) if reflections_run else None,
        "output_dir": str(out_root),
        "surfaces": list(surfaces),
        "promote_status": promote_status.value,
        "pairs": [{"corpus": c, "source": s} for (c, s) in pairs],
        "run_skill_enabled": bool(args.run_skill),
        "limits": {
            "max_episodes": args.max_episodes,
            "max_steps": args.max_steps,
            "max_proposals": args.max_proposals,
            "max_replay_seeds": args.max_replay_seeds,
            "max_shadow_episodes": args.max_shadow_episodes,
        },
    }
    (out_root / "_run_meta.json").write_text(json.dumps(run_meta, indent=2))

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for (corpus, source) in pairs:
        logger.info("[%s/%s] start", corpus, source)
        try:
            row = _process_source(
                corpus=corpus,
                source=source,
                bank_run=bank_run,
                actions_run=actions_run,
                crafter_proposals_run=crafter_proposals_run,
                reflections_run=reflections_run,
                output_root=out_root,
                surfaces=surfaces,
                max_episodes=args.max_episodes,
                max_steps=args.max_steps,
                max_proposals=args.max_proposals,
                max_replay_seeds=args.max_replay_seeds,
                max_shadow_episodes=args.max_shadow_episodes,
                run_skill_enabled=args.run_skill,
                promote_status=promote_status,
                gate_runner_enabled=args.gate_runner,
            )
        except Exception as exc:                                   # noqa: BLE001
            logger.error(
                "[%s/%s] CRASH: %s\n%s", corpus, source, exc,
                traceback.format_exc(),
            )
            row = {
                "corpus": corpus, "source": source,
                "status": "crashed", "error": repr(exc),
            }
        rows.append(row)
        logger.info("[%s/%s] done: %s", corpus, source, row.get("status"))

    elapsed = time.time() - t0
    summary = {
        "started_at": run_meta["started_at"],
        "finished_at": _utcnow_iso(),
        "elapsed_sec": round(elapsed, 3),
        "n_pairs": len(pairs),
        "n_pairs_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_pairs_failed": sum(1 for r in rows if r.get("status") not in ("ok",)),
        "per_source": rows,
        "harness_known_gaps": [
            "§9.1 — SkillHarness.validate_invocation: CLOSED in Day-8a; dump driver wired Day-7e.",
            "§9.2 — select_eligible_skills doesn't accept intention/active_skill/local_reasoning_trace (still pending — Day-10+, requires planner-context plumb)",
            "§9.3 — EligibleSkill: per-check booleans CLOSED in Day-8a; fit_score/risk_score still pending (Day-10+, LoRA scoring head)",
            "§10  — SkillEpisode: evidence_in/out, warrants, protocol_trace, contract_progress, reward_components, shadow, diagnostic_labels CLOSED in Day-8b",
            "§11  — SkillEvaluationRecord: anchors CLOSED in Day-8c; populated when --gate-runner is set on this driver",
            "§12  — GateService stage signatures: rollout_batch / eval_suite overloads CLOSED in Day-7a (use --gate-runner)",
            "§13  — GateService relocated under harness.GateRunner alias in Day-7a (use --gate-runner to opt in)",
        ],
    }
    (out_root / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    )
    logger.info(
        "harness I/O dump done: %s pairs in %.1fs → %s",
        len(pairs), elapsed, out_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
