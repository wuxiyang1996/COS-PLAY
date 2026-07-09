"""Per-step Cross-domain transfer gate for the trainer's co-evolution loop.

Layer A of the design in
``implementation_notes/coevolution-cross-domain-integration.md``:
re-evaluates the cross-domain admit rate of every skill the
:mod:`trainer.coevolution._promotion_hook` just promoted, and rolls
back promotions that fail a configurable admit-rate band on the
configured target corpora.

Splices into ``trainer/coevolution/orchestrator.py::co_evolution_loop``
*after* :func:`trainer.coevolution._promotion_hook.run_promotion_step`
finishes its writeback. Two-stage contract:

  1. Walk the just-completed promotion's ``writeback_per_game`` to
     collect the freshly-inserted skill_ids per game. Materialise a
     synthetic ``--game-bank-root`` whose per-game ``skill_bank.jsonl``
     contains *only* those skills (a strict subset of the trainer's
     full bank). This isolates the gate's evaluation to skills that
     just promoted, without re-running the whole bank.
  2. Subprocess-invoke
     ``labeling_supplement/_phase4_transfer_matrix.py`` against that
     filtered bank with ``--target-corpora`` set to the configured
     transfer targets. Output goes to
     ``<run_dir>/transfer_gate_out/step_<step>/``.
  3. Read the matrix subprocess's ``per_skill.jsonl`` artefact, group
     verdicts by ``skill_id``, and score each skill against the
     configured admit-rate band:

     * ``admit_rate < band[0]`` on **every** target ⇒ ``DEMOTE``
       (rolled back).
     * ``admit_rate ∈ band`` on ``>= K`` targets   ⇒ ``KEEP`` (the
       common case).
     * Missing data (no cell ran for ``skill_id × target``) is
       treated as ``KEEP`` — we don't punish a skill for a target
       the matrix didn't have demos to evaluate against.

  4. For each ``DEMOTE`` verdict, rewrite the trainer's per-game
     ``skill_bank.jsonl`` *in place*, dropping the demoted skill_id
     entries. This is a precise inverse of the
     ``legacy_writeback.writeback_promotion`` step that just inserted
     them — we keep the file's other entries (skills that pre-dated
     this step + skills that passed the gate) untouched.

Strict trainer-mode contract
----------------------------
* The matrix driver is invoked **as a subprocess** — there is no
  in-process Python coupling between the trainer and
  ``_phase4_transfer_matrix.py``, matching the "no driver imports
  another driver's code" rule from
  ``crafter-harness-orchestrator-roles.md`` §6.3.
* The synthetic ``--game-bank-root`` is built from per-game JSONL
  *copies* (not symlinks): the gate filters the bank to a strict
  subset, which symlinks can't represent.
* On any internal failure (subprocess returncode ≠ 0, parse error,
  timeout) the gate **does not demote**. The promoted skills
  remain in the bank and the report carries the failure reason. This
  is the conservative direction: a buggy gate must never throw away
  work the upstream promotion driver already approved.

Cross-refs
----------
* ``implementation_notes/coevolution-cross-domain-integration.md`` §4
  (Layer A — design rationale + acceptance criteria).
* ``labeling_supplement/_phase4_transfer_matrix.py`` — the driver
  this hook wraps; its ``per_skill.jsonl`` schema is the contract.
* ``trainer/coevolution/_promotion_hook.py`` — the hook this one
  layers on top of.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Per the design memo §4.4 / §11.5.4 of the cross-domain measurement
# plan: a skill needs ≥15% admit on at least one target to keep its
# promotion, and ≥60% to bypass extra checks. The lower bound is the
# only one currently enforced by the gate — the upper bound is
# informational and surfaced in the report for the dashboard layer to
# pick up.
DEFAULT_TRANSFER_ADMIT_BAND: Tuple[float, float] = (0.15, 0.60)

# Stage-6 driver default (see _phase4_transfer_matrix.py L675).
DEFAULT_TRANSFER_MAX_SKILLS_PER_CELL: int = 5

# How many target corpora must hit the band's lower bound for a skill
# to KEEP its promotion. K=1 is the most permissive: a single
# transferable target is enough.
DEFAULT_TRANSFER_MIN_TARGETS_IN_BAND: int = 1

# Stage-6 driver wall-clock budget. The matrix subprocess loads a
# couple of episodes per cell; per-skill bank-runs (single skill, 2-4
# targets) typically finish in well under five minutes, but we leave
# a generous ceiling because conda-env helpers (browser) cold-boot
# adds ~30s and the gate is not on the rollout hot path.
DEFAULT_TRANSFER_DRIVER_TIMEOUT_S: float = 1800.0


# ---------------------------------------------------------------------------
# Per-skill verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferSkillVerdict:
    """One skill's cross-domain verdict from the gate."""

    skill_id: str
    game: str
    decision: str                                 # "KEEP" / "DEMOTE" / "INSUFFICIENT_DATA"
    admit_rate_per_target: Dict[str, float] = field(default_factory=dict)
    n_targets_in_band: int = 0
    failure_class: Optional[str] = None           # populated when decision == "DEMOTE"
                                                   # ("CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "game": self.game,
            "decision": self.decision,
            "admit_rate_per_target": dict(self.admit_rate_per_target),
            "n_targets_in_band": self.n_targets_in_band,
            "failure_class": self.failure_class,
        }


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferGateReport:
    """What the transfer gate produced for one trainer step."""

    step: int
    run_dir: Path
    transfer_run_dir: Path
    n_skills_in: int
    n_keep: int
    n_demote: int
    n_insufficient_data: int
    verdicts: List[TransferSkillVerdict] = field(default_factory=list)
    demotions_per_game: Dict[str, List[str]] = field(default_factory=dict)
    driver_returncode: int = 0
    driver_wall_time_s: float = 0.0
    wall_time_s: float = 0.0
    skipped: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "run_dir": str(self.run_dir),
            "transfer_run_dir": str(self.transfer_run_dir),
            "n_skills_in": self.n_skills_in,
            "n_keep": self.n_keep,
            "n_demote": self.n_demote,
            "n_insufficient_data": self.n_insufficient_data,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "demotions_per_game": {
                g: list(ids) for g, ids in self.demotions_per_game.items()
            },
            "driver_returncode": self.driver_returncode,
            "driver_wall_time_s": self.driver_wall_time_s,
            "wall_time_s": self.wall_time_s,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
        }


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------


def run_transfer_gate_step(
    *,
    step: int,
    run_dir: Path,
    promotion_writeback_per_game: Mapping[str, Mapping[str, Any]],
    legacy_bank_paths: Mapping[str, Path],
    transfer_targets: Sequence[str],
    transfer_admit_band: Tuple[float, float] = DEFAULT_TRANSFER_ADMIT_BAND,
    transfer_min_targets_in_band: int = DEFAULT_TRANSFER_MIN_TARGETS_IN_BAND,
    transfer_max_skills_per_cell: int = DEFAULT_TRANSFER_MAX_SKILLS_PER_CELL,
    transfer_driver_timeout_s: float = DEFAULT_TRANSFER_DRIVER_TIMEOUT_S,
    driver_executable: Optional[Sequence[str]] = None,
    extra_driver_args: Optional[Sequence[str]] = None,
    apply_demotions: bool = True,
) -> TransferGateReport:
    """Run the per-step cross-domain transfer gate.

    Parameters
    ----------
    step
        Trainer step index.
    run_dir
        Trainer's run root.  Output goes to
        ``<run_dir>/transfer_gate_out/step_<step>/``.
    promotion_writeback_per_game
        Per-game writeback report from
        :class:`trainer.coevolution._promotion_hook.PromotionStepReport`.
        We read ``inserted_skill_ids`` (and, optionally,
        ``updated_skill_ids``) from each entry to decide which skills
        to evaluate.
    legacy_bank_paths
        Map from game name to per-game ``skill_bank.jsonl``. Used to
        materialise the synthetic single-skill bank-run AND, when
        ``apply_demotions=True``, as the in-place mutation target for
        rolled-back promotions.
    transfer_targets
        Target corpora the matrix driver evaluates against
        (e.g. ``("video", "visual_reasoning", "browser")``). At least
        one entry is required.
    transfer_admit_band
        ``(lower, upper)`` admit-rate band. Skills failing
        ``admit_rate < lower`` on every target are DEMOTED. Upper
        bound is informational (Layer D dashboard surfaces it).
    transfer_min_targets_in_band
        Minimum number of targets a skill must clear ``band[0]`` on
        to KEEP. Defaults to 1 (most permissive).
    transfer_max_skills_per_cell
        Forwarded to the matrix driver's ``--max-skills`` flag.
    transfer_driver_timeout_s
        Hard wall-clock limit on the subprocess.
    driver_executable
        Override for testing.  Defaults to
        ``[sys.executable, <repo>/labeling_supplement/_phase4_transfer_matrix.py]``.
    extra_driver_args
        Additional CLI flags forwarded to the subprocess.
    apply_demotions
        When ``True`` (default), DEMOTE verdicts mutate the per-game
        ``skill_bank.jsonl`` in place. When ``False``, the report
        carries the verdicts but the bank is untouched (dry-run).

    Returns
    -------
    TransferGateReport
        Per-skill verdicts + demotion plan + subprocess metadata.
    """
    t0 = time.monotonic()

    transfer_run_dir = (
        Path(run_dir) / "transfer_gate_out" / f"step_{step:04d}"
    )
    transfer_run_dir.mkdir(parents=True, exist_ok=True)

    just_promoted = _collect_just_promoted_skill_ids(promotion_writeback_per_game)
    n_skills_in = sum(len(ids) for ids in just_promoted.values())

    if n_skills_in == 0:
        elapsed = time.monotonic() - t0
        return TransferGateReport(
            step=step,
            run_dir=Path(run_dir),
            transfer_run_dir=transfer_run_dir,
            n_skills_in=0,
            n_keep=0, n_demote=0, n_insufficient_data=0,
            verdicts=[], demotions_per_game={},
            driver_returncode=0,
            driver_wall_time_s=0.0,
            wall_time_s=elapsed,
            skipped=True,
            skipped_reason="no skills promoted in this step",
        )

    if not transfer_targets:
        elapsed = time.monotonic() - t0
        return TransferGateReport(
            step=step,
            run_dir=Path(run_dir),
            transfer_run_dir=transfer_run_dir,
            n_skills_in=n_skills_in,
            n_keep=n_skills_in, n_demote=0, n_insufficient_data=0,
            verdicts=[
                TransferSkillVerdict(
                    skill_id=sid, game=game, decision="KEEP",
                    n_targets_in_band=0,
                )
                for game, ids in just_promoted.items() for sid in ids
            ],
            demotions_per_game={},
            driver_returncode=0,
            driver_wall_time_s=0.0,
            wall_time_s=elapsed,
            skipped=True,
            skipped_reason="no transfer_targets configured",
        )

    # Build the synthetic single-skill bank-run.
    synthetic_bank_root = transfer_run_dir / "synthetic_bank_root"
    _materialise_filtered_bank_run(
        bank_root=synthetic_bank_root,
        legacy_bank_paths=legacy_bank_paths,
        eligible_skill_ids_per_game=just_promoted,
    )

    cmd = list(_resolve_driver_executable(driver_executable))
    cmd += [
        "--game-bank-root", str(synthetic_bank_root.resolve()),
        "--target-corpora", *list(transfer_targets),
        "--source-corpora", *sorted(just_promoted.keys()),
        "--max-skills", str(transfer_max_skills_per_cell),
        "--out-dir", str(transfer_run_dir.resolve()),
    ]
    if extra_driver_args:
        cmd += list(extra_driver_args)

    driver_t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=transfer_driver_timeout_s,
            cwd=str(_codebase_root()),
            check=False,
            env=_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "transfer_hook: driver timed out after %.1fs at step=%d",
            transfer_driver_timeout_s, step,
        )
        return TransferGateReport(
            step=step,
            run_dir=Path(run_dir),
            transfer_run_dir=transfer_run_dir,
            n_skills_in=n_skills_in,
            n_keep=n_skills_in, n_demote=0, n_insufficient_data=0,
            verdicts=[], demotions_per_game={},
            driver_returncode=124,                   # GNU timeout convention
            driver_wall_time_s=transfer_driver_timeout_s,
            wall_time_s=time.monotonic() - t0,
            skipped=True,
            skipped_reason=f"driver timeout: {exc}",
        )
    driver_wall = time.monotonic() - driver_t0

    if proc.returncode != 0:
        logger.error(
            "transfer_hook: driver returned %d at step=%d\n"
            "  stdout tail: %s\n  stderr tail: %s",
            proc.returncode, step,
            (proc.stdout or "")[-800:], (proc.stderr or "")[-800:],
        )
        return TransferGateReport(
            step=step,
            run_dir=Path(run_dir),
            transfer_run_dir=transfer_run_dir,
            n_skills_in=n_skills_in,
            n_keep=n_skills_in, n_demote=0, n_insufficient_data=0,
            verdicts=[], demotions_per_game={},
            driver_returncode=proc.returncode,
            driver_wall_time_s=driver_wall,
            wall_time_s=time.monotonic() - t0,
            skipped=True,
            skipped_reason=f"driver returncode={proc.returncode}",
        )

    # Parse per_skill.jsonl into {skill_id: {target: admit_rate}}.
    per_skill = _parse_per_skill_results(transfer_run_dir)

    # Score each just-promoted skill against the band.
    verdicts: List[TransferSkillVerdict] = []
    demotions_per_game: Dict[str, List[str]] = {}
    n_keep = 0
    n_demote = 0
    n_insufficient_data = 0
    band_lo = float(transfer_admit_band[0])
    for game, sids in just_promoted.items():
        for sid in sids:
            target_admits = per_skill.get(sid, {})
            v = _decide_for_skill(
                skill_id=sid,
                game=game,
                target_admits=target_admits,
                band_lo=band_lo,
                min_in_band=int(transfer_min_targets_in_band),
                requested_targets=tuple(transfer_targets),
            )
            verdicts.append(v)
            if v.decision == "KEEP":
                n_keep += 1
            elif v.decision == "DEMOTE":
                n_demote += 1
                demotions_per_game.setdefault(game, []).append(sid)
            else:
                n_insufficient_data += 1

    # Apply demotions (rollback the promotion's bank insertion).
    if apply_demotions and demotions_per_game:
        for game, sids in demotions_per_game.items():
            bank_path = legacy_bank_paths.get(game)
            if bank_path is None:
                continue
            try:
                n_dropped = _drop_skill_ids_from_bank(Path(bank_path), set(sids))
                logger.info(
                    "transfer_hook: demoted %d skills from %s "
                    "(cross-domain admit floor)",
                    n_dropped, bank_path,
                )
            except OSError as exc:
                logger.error(
                    "transfer_hook: failed to demote skills from %s: %s",
                    bank_path, exc,
                )

    # Write per-step summary.
    elapsed = time.monotonic() - t0
    summary = {
        "step": step,
        "transfer_run_dir": str(transfer_run_dir),
        "n_skills_in": n_skills_in,
        "n_keep": n_keep,
        "n_demote": n_demote,
        "n_insufficient_data": n_insufficient_data,
        "verdicts": [v.to_dict() for v in verdicts],
        "demotions_per_game": {
            g: list(ids) for g, ids in demotions_per_game.items()
        },
        "params": {
            "transfer_targets": list(transfer_targets),
            "transfer_admit_band": list(transfer_admit_band),
            "transfer_min_targets_in_band": int(transfer_min_targets_in_band),
            "transfer_max_skills_per_cell": int(transfer_max_skills_per_cell),
            "apply_demotions": bool(apply_demotions),
        },
        "driver_wall_time_s": driver_wall,
        "wall_time_s": elapsed,
    }
    try:
        (transfer_run_dir / "_step_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(
            "transfer_hook: could not write _step_summary.json: %s", exc,
        )

    return TransferGateReport(
        step=step,
        run_dir=Path(run_dir),
        transfer_run_dir=transfer_run_dir,
        n_skills_in=n_skills_in,
        n_keep=n_keep,
        n_demote=n_demote,
        n_insufficient_data=n_insufficient_data,
        verdicts=verdicts,
        demotions_per_game=demotions_per_game,
        driver_returncode=0,
        driver_wall_time_s=driver_wall,
        wall_time_s=elapsed,
        skipped=False,
        skipped_reason="",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _codebase_root() -> Path:
    """Repo root — three levels up from this file
    (``trainer/coevolution/_transfer_hook.py``)."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve_driver_executable(
    override: Optional[Sequence[str]],
) -> Sequence[str]:
    if override is not None:
        return list(override)
    driver_path = (
        _codebase_root()
        / "labeling_supplement" / "_phase4_transfer_matrix.py"
    )
    return [sys.executable, str(driver_path)]


def _subprocess_env() -> Dict[str, str]:
    """Pin ``PYTHONPATH`` to the codebase root."""
    env = dict(os.environ)
    repo = str(_codebase_root())
    existing = env.get("PYTHONPATH", "")
    if repo not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{repo}{os.pathsep}{existing}" if existing else repo
        )
    return env


def _collect_just_promoted_skill_ids(
    writeback_per_game: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """Lift the ``inserted_skill_ids`` (and ``updated_skill_ids``) from a
    :class:`PromotionStepReport.writeback_per_game` mapping into a flat
    ``{game: [skill_id, ...]}`` dict.

    Skills that were merely *updated* (status transitions on existing
    bank entries) are evaluated alongside freshly-inserted ones — a
    cross-domain regression on an updated skill is just as
    transfer-gate-relevant as on a new one. Tests are free to stub
    this map directly.
    """
    out: Dict[str, List[str]] = {}
    for game, wb in (writeback_per_game or {}).items():
        if not isinstance(wb, Mapping):
            continue
        ids: List[str] = []
        for key in ("inserted_skill_ids", "updated_skill_ids"):
            v = wb.get(key, [])
            if isinstance(v, (list, tuple)):
                ids.extend(str(x) for x in v if x)
        if ids:
            out[str(game)] = ids
    return out


def _materialise_filtered_bank_run(
    *,
    bank_root: Path,
    legacy_bank_paths: Mapping[str, Path],
    eligible_skill_ids_per_game: Mapping[str, Iterable[str]],
) -> None:
    """Build a synthetic ``--game-bank-root`` containing only the
    just-promoted skills.

    Layout is ``<bank_root>/env_wrappers/<game>/skill_bank.jsonl`` and
    ``<bank_root>/gym_v/<game>/skill_bank.jsonl`` because the matrix
    driver's ``_resolve_default_target_corpora`` walks both. We only
    populate the per-game rows we have writeback for; missing files are
    tolerated downstream.

    Unlike :func:`_promotion_hook._materialise_synthetic_bank_run` we
    *can't* symlink — the matrix driver's source-bank loader reads the
    full file, and we need a strict subset. Per-skill JSONL filtering
    is cheap (one file open + line scan).
    """
    for game, ids in eligible_skill_ids_per_game.items():
        bank_path = legacy_bank_paths.get(game)
        if bank_path is None:
            continue
        bank_path = Path(bank_path)
        if not bank_path.is_file():
            logger.debug(
                "transfer_hook: skipping synthetic bank-run entry for %s — "
                "file missing: %s", game, bank_path,
            )
            continue
        keep: set = {str(s) for s in ids}
        if not keep:
            continue
        kept_lines: List[str] = []
        try:
            with bank_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    sid = ((entry.get("skill") or {}).get("skill_id") or "")
                    if sid in keep:
                        kept_lines.append(raw)
        except OSError as exc:
            logger.debug(
                "transfer_hook: failed to read %s: %s", bank_path, exc,
            )
            continue

        # Mirror the matrix driver's default game-bank-root layout
        # (``env_wrappers/<game>/skill_bank.jsonl``); the driver also
        # walks ``gym_v/`` for retro games but defaults skip those, so
        # we keep the simpler ``env_wrappers`` shape.
        out_dir = bank_root / "env_wrappers" / game
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "skill_bank.jsonl").write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )


def _parse_per_skill_results(
    transfer_run_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Read ``<transfer_run_dir>/per_skill.jsonl`` and group verdicts
    by ``skill_id`` × ``target_corpus``.

    The matrix driver writes one JSONL row per (cell, skill) with at
    least ``skill_id``, ``target_corpus``, ``success`` (bool), and
    ``pass_rate`` (float). We collapse multiple-cell rows for the same
    skill_id × target into the *mean* pass_rate to keep the band
    semantic intuitive (one cell per (source, target) is the common
    case anyway).

    Returns ``{skill_id: {target_corpus: pass_rate}}``. Missing or
    malformed file ⇒ empty dict (the caller treats this as
    INSUFFICIENT_DATA for every skill).
    """
    path = transfer_run_dir / "per_skill.jsonl"
    if not path.is_file():
        logger.debug(
            "transfer_hook: per_skill.jsonl missing at %s", path,
        )
        return {}
    by_skill: Dict[str, Dict[str, List[float]]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                sid = str(row.get("skill_id") or "")
                tgt = str(row.get("target_corpus") or "")
                if not sid or not tgt:
                    continue
                # Prefer pass_rate (continuous) over success (binary)
                # so the band's lower bound has actual signal at
                # admit_rate=0.20 etc.
                rate = row.get("pass_rate")
                if rate is None:
                    rate = 1.0 if row.get("success") else 0.0
                try:
                    rate_f = float(rate)
                except (TypeError, ValueError):
                    continue
                by_skill.setdefault(sid, {}).setdefault(tgt, []).append(rate_f)
    except OSError as exc:
        logger.debug(
            "transfer_hook: failed to read %s: %s", path, exc,
        )
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for sid, by_tgt in by_skill.items():
        out[sid] = {
            tgt: (sum(rates) / len(rates)) if rates else 0.0
            for tgt, rates in by_tgt.items()
        }
    return out


def _decide_for_skill(
    *,
    skill_id: str,
    game: str,
    target_admits: Mapping[str, float],
    band_lo: float,
    min_in_band: int,
    requested_targets: Sequence[str],
) -> TransferSkillVerdict:
    """Apply the band to one skill's per-target admit rates.

    A skill KEEPs when at least ``min_in_band`` targets cleared
    ``band_lo``. It DEMOTEs when at least one requested target ran AND
    every target that ran fell below ``band_lo``. INSUFFICIENT_DATA
    fires when no requested target produced a rate (matrix driver had
    no demos to evaluate against).
    """
    n_targets_in_band = sum(
        1 for r in target_admits.values() if r >= band_lo
    )
    # Restrict scoring to the targets the caller actually asked about.
    # Extra targets the matrix surfaced (e.g. cross-cluster cells) are
    # informational only.
    relevant: Dict[str, float] = {
        t: target_admits[t]
        for t in requested_targets
        if t in target_admits
    }
    if not relevant:
        return TransferSkillVerdict(
            skill_id=skill_id, game=game,
            decision="INSUFFICIENT_DATA",
            admit_rate_per_target=dict(target_admits),
            n_targets_in_band=n_targets_in_band,
        )

    relevant_in_band = sum(1 for r in relevant.values() if r >= band_lo)

    if relevant_in_band >= max(1, int(min_in_band)):
        return TransferSkillVerdict(
            skill_id=skill_id, game=game,
            decision="KEEP",
            admit_rate_per_target=dict(target_admits),
            n_targets_in_band=relevant_in_band,
        )

    # Floor violation: every requested target that ran fell below
    # band[0]. Tag with the canonical failure class so the
    # configs/failure_routing.yaml `cross_domain_taxonomy` block can
    # route the skill to the generalizer / retirer modes.
    return TransferSkillVerdict(
        skill_id=skill_id, game=game,
        decision="DEMOTE",
        admit_rate_per_target=dict(target_admits),
        n_targets_in_band=relevant_in_band,
        failure_class="CROSS_DOMAIN_ADMIT_FLOOR_VIOLATION",
    )


def _drop_skill_ids_from_bank(
    bank_path: Path,
    skill_ids_to_drop: Iterable[str],
) -> int:
    """Mutate ``bank_path`` in place, removing JSONL entries whose
    ``skill.skill_id`` is in ``skill_ids_to_drop``.

    Returns the number of entries dropped. Other entries (skills that
    pre-dated this step + skills that passed the gate) are preserved
    byte-for-byte. Malformed / non-skill rows are also preserved (we
    only filter on the strict envelope shape; anything else passes
    through untouched, matching :func:`_load_legacy_jsonl`'s
    tolerance).

    Atomicity: the mutation goes via ``<path>.tmp`` + ``os.replace``
    so a crash mid-write can't leave the trainer's bank in a
    half-rewritten state.
    """
    drop_set = {str(s) for s in skill_ids_to_drop}
    if not drop_set:
        return 0
    if not bank_path.is_file():
        return 0
    kept_lines: List[str] = []
    n_dropped = 0
    with bank_path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.rstrip("\n")
            if not stripped.strip():
                kept_lines.append(stripped)
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(stripped)
                continue
            skill_blob = entry.get("skill") if isinstance(entry, dict) else None
            sid = ""
            if isinstance(skill_blob, dict):
                sid = str(skill_blob.get("skill_id") or "")
            if sid and sid in drop_set:
                n_dropped += 1
                continue
            kept_lines.append(stripped)

    if n_dropped == 0:
        return 0
    tmp_path = bank_path.with_suffix(bank_path.suffix + ".tmp")
    tmp_path.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )
    os.replace(tmp_path, bank_path)
    return n_dropped


__all__ = [
    "DEFAULT_TRANSFER_ADMIT_BAND",
    "DEFAULT_TRANSFER_DRIVER_TIMEOUT_S",
    "DEFAULT_TRANSFER_MAX_SKILLS_PER_CELL",
    "DEFAULT_TRANSFER_MIN_TARGETS_IN_BAND",
    "TransferGateReport",
    "TransferSkillVerdict",
    "run_transfer_gate_step",
]
