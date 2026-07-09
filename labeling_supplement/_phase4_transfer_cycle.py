#!/usr/bin/env python
"""Phase-4 / Day-5 — Stage 3a `FewShotAdapter` cross-task transfer
cycle. The empirical companion to harness/README §22 Day-5b and the
write-up in `labeling_supplement/harness_io_out/_phase4_report.md`.

What this driver does:

    1. Load every lifted SkillRecord from
       `labeling/skill_bank_out/run_<ts>/env_wrappers/<source_game>/skill_bank.jsonl`
       via `record_from_bank_entry`.
    2. Build target-task `FewShotDemo`s by reading
       `labeling/skill_actions_out/run_<ts>/env_wrappers/<target_game>/episode_*.json`
       and parsing each step's `metadata.schema_canonical` block into
       a `StateSchema` (`harness.few_shot_demos_gymv`).
    3. Wire `GymvAdapter.set_executor(make_gymv_executor(target_env,
       schema_producer=make_gaming_env_producer(target_game)))` so the
       harness drives the *target* env's actual state.
    4. For each ACTION-typed lifted skill from the source game, call
       `FewShotAdapter.adapt(skill, target_domain="gymv",
        target_task=<target_game>, demos=…, success_fn=
        make_per_step_success_fn(skill))`.
    5. Aggregate `(skill_id × target_task)` verdicts; on PASS /
       LIMITED_PASS, append `target_task` to `verified_tasks` (held in
       memory by default; with ``--persist`` also write through a
       `SkillLifecycleManager.record_task_verification` so the change
       lands on disk — Day-9b).
    6. Re-run the cross-eligibility probe to demonstrate that skills
       with newly-broadened `verified_tasks` get admitted on the
       target task.

Usage::

    cd Multi-hop-Reasoning-VLM-Agent
    python labeling_supplement/_phase4_transfer_cycle.py \\
        --source twenty_forty_eight --target tetris --k 4 --seed 0

    # Day-9b: persist verified_tasks to disk via the lifecycle manager.
    python labeling_supplement/_phase4_transfer_cycle.py \\
        --source twenty_forty_eight --target tetris --persist \\
        --persist-bank-root /tmp/lift-test-bank

Defaults to ``2048 → tetris``. Pass ``--source tetris --target
twenty_forty_eight`` for the reverse direction. Same-task probes
(``--source X --target X``) are useful as a sanity-check baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.enums import SkillStatus, SkillType                          # noqa: E402
from data_structure.extensions.skill_record import SkillRecord           # noqa: E402
from harness import FewShotAdapter                                       # noqa: E402

# Reuse the Phase-2 driver's bank-loader and env-builder.
from labeling_supplement._phase2_real_env_skill_smoke import (           # noqa: E402
    DEFAULT_BANK_ROOT,
    DEFAULT_ACTIONS_ROOT,
    load_bank_records,
)
from labeling_supplement._phase4_target_dispatch import (                # noqa: E402
    TargetBuild,
    build_target,
    registered_target_domains,
)

logger = logging.getLogger("phase4_transfer_cycle")


@dataclass
class TransferVerdict:
    """One ``(source_skill, target_task)`` adaptation outcome."""

    skill_id: str
    skill_type: str
    source_task: str
    target_task: str
    target_domain: str
    n_demos: int
    n_demos_used: int
    n_success: int
    n_aborted: int
    pass_rate: float
    success: bool                       # by FewShotAdapter._pass_rate_min
    diagnostic_label: Optional[str]
    cost_ms: float
    cost_tokens: float
    verified_task_promoted: bool         # would lifecycle append target_task?
    feasible_tasks_before: List[str] = field(default_factory=list)
    verified_tasks_before: List[str] = field(default_factory=list)
    verified_tasks_after: List[str] = field(default_factory=list)


def _eligibility_admit_set(
    records: List[SkillRecord],
    *,
    domain: str,
    task: str,
) -> Dict[str, bool]:
    """Return ``{skill_id: admitted_on_(domain,task)}`` using the
    F2′ task-axis filter. Mirrors the Phase-0 cross-eligibility probe's
    logic in a single-shot dictionary form."""
    admit: Dict[str, bool] = {}
    for r in records:
        # F2 (domain): skill is admitted if its feasible_domains
        # covers the current domain (or is empty).
        feasible_domains = list(getattr(r, "feasible_domains", []) or [])
        domain_ok = (not feasible_domains) or (domain in feasible_domains)
        # F2′ (task): skill is admitted if its feasible_tasks covers
        # the current task (or is empty), OR if it has been verified
        # on the target task explicitly.
        feasible_tasks = list(getattr(r, "feasible_tasks", []) or [])
        verified_tasks = list(getattr(r, "verified_tasks", []) or [])
        task_ok = (
            not feasible_tasks
            or task in feasible_tasks
            or task in verified_tasks
        )
        admit[r.skill_id] = bool(domain_ok and task_ok)
    return admit


def _run_transfer(
    *,
    source_game: str,
    target_game: str,
    source_records: List[SkillRecord],
    target_build: TargetBuild,
    pass_rate_min: float = 0.5,
    k: int = 4,
    bindings_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[List[TransferVerdict], List[SkillRecord]]:
    """Drive the FewShotAdapter through every source skill on the
    pre-built target cell. Returns ``(verdicts, mutated_records)``
    — the records carry an updated ``verified_tasks`` (in-memory only)
    so the eligibility-probe diff is visible end-to-end.

    The ``target_build`` carries the per-domain adapter, harness,
    demos, and ``success_fn_factory``. See
    ``_phase4_target_dispatch.build_target`` for how it's constructed.
    """

    target_domain = target_build.target_domain
    harness = target_build.harness
    demos = target_build.demos

    verdicts: List[TransferVerdict] = []
    mutated: List[SkillRecord] = []

    for skill in source_records:
        # The adapter's `source_domains` check expects a gymv lineage;
        # ensure it's set (the bank rows may carry empty
        # source_domains by default).
        if not skill.source_domains:
            object.__setattr__(skill, "source_domains", ("gymv",))
        # Skill must be PROVISIONAL+ for run_skill — bank records are
        # DRAFT by default; promote on the in-memory copy.
        object.__setattr__(skill, "status", SkillStatus.PROVISIONAL)

        # Per-domain success_fn instance. For gymv this scores per-hop
        # effect predicates; for VR / video it scores answer match
        # against the demo's `expected.gold_answer`; for osworld /
        # browser it mirrors the gymv per-hop effect-predicate path
        # against the producer-emitted facts.
        success_fn = target_build.success_fn_factory(
            pass_rate_threshold=0.5,
        )

        # Build a per-skill FewShotAdapter with this success_fn.
        few_shot = FewShotAdapter(
            harness=harness,
            success_fn=success_fn,
            target_domain_pass_rate_min=pass_rate_min,
        )

        # Pre-fill bindings the executor's payload-value rescue can
        # use even when demo bindings don't resolve into the target
        # env's action_names.
        target_demos = list(demos)
        if bindings_overrides:
            for d in target_demos:
                for k_b, v_b in bindings_overrides.items():
                    d.bindings.setdefault(k_b, v_b)

        feas_before = list(skill.feasible_tasks or [])
        ver_before = list(skill.verified_tasks or [])

        try:
            r = few_shot.adapt(
                skill=skill,
                target_domain=target_domain,
                target_task=target_game,
                demos=target_demos,
                k=k,
            )
        except Exception as exc:                                       # noqa: BLE001
            logger.warning("skill=%s adapt raised: %r", skill.skill_id, exc)
            verdicts.append(TransferVerdict(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type.value,
                source_task=source_game,
                target_task=target_game,
                target_domain=target_domain,
                n_demos=len(target_demos),
                n_demos_used=0,
                n_success=0,
                n_aborted=0,
                pass_rate=0.0,
                success=False,
                diagnostic_label=f"adapt_raised: {exc.__class__.__name__}",
                cost_ms=0.0,
                cost_tokens=0.0,
                verified_task_promoted=False,
                feasible_tasks_before=feas_before,
                verified_tasks_before=ver_before,
                verified_tasks_after=ver_before,
            ))
            continue

        promoted = False
        ver_after = list(ver_before)
        if r.success and target_game not in ver_after:
            ver_after.append(target_game)
            object.__setattr__(skill, "verified_tasks", ver_after)
            promoted = True
            mutated.append(skill)

        verdicts.append(TransferVerdict(
            skill_id=skill.skill_id,
            skill_type=skill.skill_type.value,
            source_task=source_game,
            target_task=target_game,
            target_domain=target_domain,
            n_demos=len(target_demos),
            n_demos_used=r.k_used,
            n_success=r.n_success,
            n_aborted=r.aborted,
            pass_rate=r.pass_rate,
            success=r.success,
            diagnostic_label=r.diagnostic_label,
            cost_ms=r.cost_ms,
            cost_tokens=r.cost_tokens,
            verified_task_promoted=promoted,
            feasible_tasks_before=feas_before,
            verified_tasks_before=ver_before,
            verified_tasks_after=ver_after,
        ))
        logger.info(
            "skill=%s success=%s pass_rate=%.2f (%d/%d) diag=%r promoted=%s",
            skill.skill_id, r.success, r.pass_rate,
            r.n_success, r.k_used, r.diagnostic_label, promoted,
        )

    return verdicts, mutated


def _seed_lifecycle_for_persistence(
    bank_root: Path,
    records: List[SkillRecord],
) -> Tuple[Any, Dict[str, Any]]:
    """Day-9b helper: build a `SkillLifecycleManager` against
    `bank_root`, seed `records` as DRAFT (idempotent), and promote
    each to PROVISIONAL so `record_task_verification` runs against a
    runnable status.

    Returns ``(lifecycle, seeded_by_id)`` where ``seeded_by_id`` maps
    each input skill_id to the on-disk `SkillRecord` reference (which
    may be the same object if it was already in the bank).
    """
    from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore  # noqa: E402
    from skill_bank.stores import StoreName                                    # noqa: E402

    bank_root.mkdir(parents=True, exist_ok=True)
    repo = SkillRepository(
        draft_store=SkillStore(StoreName.DRAFT, str(bank_root / "draft")),
        candidate_store=SkillStore(StoreName.CANDIDATE, str(bank_root / "candidate")),
        active_store=SkillStore(StoreName.ACTIVE, str(bank_root / "active")),
        archive_store=SkillStore(StoreName.ARCHIVE, str(bank_root / "archive")),
    )
    lifecycle = SkillLifecycleManager(repo)
    seeded_by_id: Dict[str, Any] = {}
    for r in records:
        existing = repo.get(r.skill_id)
        if existing is None:
            object.__setattr__(r, "status", SkillStatus.DRAFT)
            lifecycle.ingest_draft(r)
            lifecycle.transition(
                r.skill_id,
                to_status=SkillStatus.CANDIDATE,
                rationale="phase4_transfer_cycle:persist-bootstrap",
            )
            lifecycle.transition(
                r.skill_id,
                to_status=SkillStatus.PROVISIONAL,
                rationale="phase4_transfer_cycle:persist-bootstrap",
            )
            existing = repo.get(r.skill_id)
        seeded_by_id[r.skill_id] = existing
    return lifecycle, seeded_by_id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="twenty_forty_eight",
                   help="Source game (skill-bank task)")
    p.add_argument("--target", default="tetris",
                   help=("Target task / target_task. For target_domain=gymv "
                         "this is a game name (tetris, twenty_forty_eight, "
                         "...); for visual_reasoning a sub-corpus "
                         "(visual_toolbench, tir_bench); for video "
                         "(video_holmes, siv_bench); for osworld a domain "
                         "(vlc, vs_code, gimp, ...); for browser a "
                         "task-id-prefix (assistantbench, miniwob, ...)."))
    p.add_argument(
        "--target-domain",
        default="gymv",
        choices=registered_target_domains(),
        help=("Transfer-target domain. 'gymv' (default) is the "
              "within-game baseline; the four cross-domain stages "
              "land progressively per "
              "implementation_notes/legacy/phase5-cross-domain-measurement.md."),
    )
    p.add_argument("--bank-root", default=str(DEFAULT_BANK_ROOT))
    p.add_argument("--actions-root", default=str(DEFAULT_ACTIONS_ROOT))
    p.add_argument(
        "--cold-start-root",
        default=None,
        help=("Root of cold-start corpus for the cross-domain target. "
              "Defaults to Cold-start-out-<target-domain>/ for "
              "visual_reasoning / video / osworld / browser. Ignored "
              "for target_domain=gymv (uses --actions-root instead)."),
    )
    p.add_argument("--max-skills", type=int, default=10)
    p.add_argument("--k", type=int, default=4,
                   help="Max demos per skill (FewShotAdapter k_shot)")
    p.add_argument("--max-demos-per-episode", type=int, default=2)
    p.add_argument("--max-episodes", type=int, default=3)
    p.add_argument("--pass-rate-min", type=float, default=0.5,
                   help="Pass-rate threshold for verified_tasks promotion")
    p.add_argument(
        "--bindings",
        action="append",
        default=[],
        help=(
            "Pre-fill `${slot}` placeholders, applied as fallbacks "
            "when target-game demos don't carry low-level action "
            "tokens (tetris cold-start uses high-level placement "
            "strings like 'S-flat col4 (+1hole, h=6)' which don't "
            "resolve to env action_names). Example: "
            "`--bindings direction=left --bindings target=left`."
        ),
    )
    p.add_argument("--out-dir",
                   default=str(REPO_ROOT / "labeling_supplement"
                               / "harness_io_out"))
    p.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Day-9b: on PASS, also call "
            "`SkillLifecycleManager.record_task_verification` so the "
            "verified_tasks change is persisted to disk (rather than "
            "just held in-memory). Requires --persist-bank-root pointing "
            "at a writable per-source bank (e.g. a copy of "
            "labeling/skill_bank_out/run_<ts>/env_wrappers/<source>/)."
        ),
    )
    p.add_argument(
        "--persist-bank-root",
        default=None,
        help=(
            "Writable per-source bank root. The driver expects "
            "<root>/draft, <root>/candidate, <root>/active, <root>/archive "
            "(SkillRepository's standard layout). On first run with "
            "--persist, the source skills are seeded as DRAFT and "
            "promoted to PROVISIONAL so subsequent runs find them."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.persist and not args.persist_bank_root:
        raise SystemExit(
            "--persist requires --persist-bank-root <path> "
            "(a writable per-source SkillRepository root)."
        )

    bindings_overrides: Dict[str, str] = {}
    for kv in args.bindings:
        if "=" in kv:
            k_b, v_b = kv.split("=", 1)
            bindings_overrides[k_b.strip()] = v_b.strip()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bank_path = Path(args.bank_root) / args.source / "skill_bank.jsonl"
    if not bank_path.exists():
        raise SystemExit(f"bank_jsonl missing: {bank_path}")
    actions_root = Path(args.actions_root)
    if args.target_domain == "gymv" and not actions_root.exists():
        raise SystemExit(f"actions_root missing: {actions_root}")

    source_records = load_bank_records(bank_path, default_domain="gymv")
    source_records.sort(key=lambda r: (r.skill_type != SkillType.ACTION, r.name))
    source_records = source_records[: args.max_skills]
    logger.info("loaded %d source-skill records from %s",
                len(source_records), bank_path)

    # Eligibility BEFORE: probe whether each source skill admits on
    # the target task. Should be False unless the bank already records
    # cross-task feasibility (it doesn't, by design). For cross-domain
    # the eligibility probe still keys on 'gymv' (the source domain
    # the F2 / F2' filters use) — the target_task axis is what
    # _phase4_transfer_cycle measures the delta on.
    admit_before = _eligibility_admit_set(
        source_records, domain="gymv", task=args.target,
    )
    n_admit_before = sum(1 for v in admit_before.values() if v)
    logger.info(
        "eligibility BEFORE: %d/%d skills admit on (%s, %s)",
        n_admit_before, len(source_records), "gymv", args.target,
    )

    # Per-target-domain dispatch. The gymv builder uses the existing
    # Day-5b path (env_wrappers cold-start episodes); cross-domain
    # builders (Stages 1-4) load demos from Cold-start-out-<domain>/.
    target_build = build_target(args.target_domain, args)

    started_at = time.time()
    verdicts, mutated = _run_transfer(
        source_game=args.source,
        target_game=args.target,
        source_records=source_records,
        target_build=target_build,
        pass_rate_min=args.pass_rate_min,
        k=args.k,
        bindings_overrides=bindings_overrides or None,
    )
    elapsed_s = time.time() - started_at

    # Day-9b: --persist writes the in-memory verified_tasks promotions
    # through a `SkillLifecycleManager` so they land on disk. Only
    # promoted skills are touched; idempotent against already-verified
    # tasks. Persistence failures are logged per-skill but do NOT abort
    # the rest of the run — empirical evidence (the verdict JSON) is
    # the headline output.
    n_persisted = 0
    persist_errors: List[Dict[str, Any]] = []
    if args.persist:
        bank_root = Path(args.persist_bank_root)
        try:
            lifecycle, seeded_by_id = _seed_lifecycle_for_persistence(
                bank_root, source_records,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "persist setup failed (bank_root=%s): %r", bank_root, exc,
            )
            persist_errors.append({
                "skill_id": None,
                "stage": "lifecycle_seed",
                "error": repr(exc),
            })
        else:
            promoted_skills = [s for s in mutated if s.skill_id in seeded_by_id]
            for skill in promoted_skills:
                v = next(
                    (vd for vd in verdicts
                     if vd.skill_id == skill.skill_id and vd.verified_task_promoted),
                    None,
                )
                if v is None:
                    continue
                metrics = {
                    args.target: {
                        "pass_rate": float(v.pass_rate),
                        "k_used": float(v.n_demos_used),
                    },
                }
                try:
                    lifecycle.record_task_verification(
                        skill.skill_id,
                        verified_tasks=[args.target],
                        evaluation_id=f"phase4-{args.source}-to-{args.target}",
                        per_task_metrics=metrics,
                        rationale=(
                            f"phase4_transfer_cycle: {args.source} → "
                            f"{args.target} pass_rate={v.pass_rate:.2f} "
                            f"k_used={v.n_demos_used}"
                        ),
                    )
                    n_persisted += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "persist failed for skill=%s: %r",
                        skill.skill_id, exc,
                    )
                    persist_errors.append({
                        "skill_id": skill.skill_id,
                        "stage": "record_task_verification",
                        "error": repr(exc),
                    })
        logger.info(
            "Day-9b persistence: %d skill(s) had verified_tasks=[%r] "
            "appended on disk (errors=%d, bank_root=%s)",
            n_persisted, args.target, len(persist_errors), args.persist_bank_root,
        )

    # Eligibility AFTER: rerun with the in-memory updated records.
    admit_after = _eligibility_admit_set(
        source_records, domain="gymv", task=args.target,
    )
    n_admit_after = sum(1 for v in admit_after.values() if v)
    n_promoted = sum(1 for v in verdicts if v.verified_task_promoted)

    # Persist verdicts.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        out_dir
        / f"_phase4_transfer_{args.source}_to_{args.target_domain}__{args.target}_{ts}.json"
    )
    out_path.write_text(json.dumps({
        "source_game": args.source,
        "target_domain": args.target_domain,
        "target_game": args.target,
        "bank_path": str(bank_path),
        "actions_root": str(actions_root),
        "cold_start_root": args.cold_start_root,
        "k": args.k,
        "pass_rate_min": args.pass_rate_min,
        "max_skills": args.max_skills,
        "max_demos_per_episode": args.max_demos_per_episode,
        "max_episodes": args.max_episodes,
        "n_skills": len(verdicts),
        "n_passed_target_pass_rate": sum(1 for v in verdicts if v.success),
        "n_verified_tasks_promoted": n_promoted,
        "n_verified_tasks_persisted": n_persisted,
        "persist_enabled": bool(args.persist),
        "persist_bank_root": args.persist_bank_root,
        "persist_errors": persist_errors,
        "eligibility_before_admit_count": n_admit_before,
        "eligibility_after_admit_count": n_admit_after,
        "eligibility_admit_delta": n_admit_after - n_admit_before,
        "elapsed_s": round(elapsed_s, 2),
        "verdicts": [asdict(v) for v in verdicts],
        "timestamp": ts,
    }, indent=2))
    logger.info("wrote %s", out_path)

    # Compact summary.
    print()
    print(
        f"=== Phase-4 transfer: {args.source} -> "
        f"{args.target_domain}:{args.target} (k={args.k}) ==="
    )
    print(f"{'skill_id':<24} {'type':<10} {'demos':>5} {'pass':>4} "
          f"{'rate':>5} {'ok':>3} {'promoted':>9} {'diag':<35}")
    for v in verdicts:
        print(
            f"{v.skill_id[:24]:<24} {v.skill_type:<10} "
            f"{v.n_demos_used:>5} {v.n_success:>4} {v.pass_rate:>5.2f} "
            f"{('Y' if v.success else 'n'):>3} "
            f"{('YES' if v.verified_task_promoted else '-'):>9} "
            f"{(v.diagnostic_label or '')[:35]:<35}"
        )
    print()
    print(
        f"eligibility on ({args.target_domain}, {args.target}): "
        f"BEFORE={n_admit_before}/{len(verdicts)}  "
        f"AFTER={n_admit_after}/{len(verdicts)}  "
        f"(delta = {n_admit_after - n_admit_before:+d})"
    )
    print(f"verified_tasks promotions: {n_promoted}")
    if args.persist:
        print(
            f"verified_tasks persisted to disk: {n_persisted} "
            f"(bank_root={args.persist_bank_root}, "
            f"errors={len(persist_errors)})"
        )
    print(f"full verdict json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
