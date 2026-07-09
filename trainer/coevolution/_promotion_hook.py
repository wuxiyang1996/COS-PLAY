"""Per-step Promotion hook for the trainer's co-evolution loop.

Splices into ``trainer/coevolution/orchestrator.py::co_evolution_loop``
*right after* the Crafter hook (see ``_crafter_hook.py``).  Two-stage
contract:

  1. Subprocess-invoke ``labeling_supplement/decide_promotion_gpt54.py``
     with ``--gate-mode offline-synthetic`` against the proposals JSONL
     the Crafter hook just produced and a *synthetic*
     ``--bank-run`` directory laid out in the offline-mirror's
     ``<corpus>/<source>/skill_bank.jsonl`` shape.  Output goes to
     ``<run_dir>/promotion_decisions_out/step_<step>/``.
  2. For each ``(corpus, source)`` pair, call
     :func:`skill_bank.legacy_writeback.writeback_promotion` to project
     the eligible promoted skills (PROVISIONAL / ACTIVE / SHADOW per F3)
     back into the trainer's per-game ``skill_bank.jsonl`` so the next
     step's actor sees them.

Strict trainer-mode contract
----------------------------
* The driver is invoked *as a subprocess* — there is no in-process
  Python coupling between the trainer and ``decide_promotion_gpt54.py``,
  matching the "no driver imports another driver's code" rule from
  ``crafter-harness-orchestrator-roles.md`` §6.3.
* The synthetic ``--bank-run`` is built from **symlinks** into the
  trainer's per-game ``skill_bank.jsonl`` files.  We never mutate those
  files (the writeback step does — but via the explicit
  ``legacy_writeback.writeback_promotion`` projector, not via
  ``decide_promotion_gpt54.py``'s output).

Cross-refs
----------
* `harness/README.md` §16.3 / §17 — the keystone problem this hook
  resolves (``bank.runnable()`` becomes non-empty after this fires).
* `implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md`
  §3 D8 (Option A — one-way writeback) and F3 (gate-mode = synthetic
  caps at PROVISIONAL).
* ``labeling_supplement/decide_promotion_gpt54.py`` — the driver this
  hook wraps; CLI flags pinned at lines ~1901-1951 of that file.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence

from skill_bank.legacy_writeback import (
    DEFAULT_ELIGIBLE_STATUSES,
    WritebackReport,
    find_latest_snapshot,
    writeback_promotion,
)
from trainer.coevolution._crafter_hook import corpus_for_game

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — match `decide_promotion_gpt54.py` flag defaults so we don't
# silently drift from the offline-mirror's behaviour.
# ---------------------------------------------------------------------------

DEFAULT_GATE_MODE: str = "offline-synthetic"
DEFAULT_DRIVER_TIMEOUT_S: float = 300.0


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionStepReport:
    """What the Promotion hook produced for one trainer step."""

    step: int
    run_dir: Path
    promotion_run_dir: Path
    n_proposals_in: int
    n_promote: int
    n_reject: int
    n_defer: int
    n_rollback: int
    writeback_per_game: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    driver_returncode: int = 0
    driver_wall_time_s: float = 0.0
    wall_time_s: float = 0.0
    skipped: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "run_dir": str(self.run_dir),
            "promotion_run_dir": str(self.promotion_run_dir),
            "n_proposals_in": self.n_proposals_in,
            "n_promote": self.n_promote,
            "n_reject": self.n_reject,
            "n_defer": self.n_defer,
            "n_rollback": self.n_rollback,
            "writeback_per_game": dict(self.writeback_per_game),
            "driver_returncode": self.driver_returncode,
            "driver_wall_time_s": self.driver_wall_time_s,
            "wall_time_s": self.wall_time_s,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
        }

    @property
    def n_writeback_inserted(self) -> int:
        return sum(int(v.get("n_inserted", 0)) for v in self.writeback_per_game.values())

    @property
    def n_writeback_updated(self) -> int:
        return sum(int(v.get("n_updated", 0)) for v in self.writeback_per_game.values())


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------


def run_promotion_step(
    *,
    step: int,
    run_dir: Path,
    proposals_run_dir: Path,
    legacy_bank_paths: Mapping[str, Path],
    eligible_statuses: Iterable[str] = DEFAULT_ELIGIBLE_STATUSES,
    gate_mode: str = DEFAULT_GATE_MODE,
    driver_executable: Optional[Sequence[str]] = None,   # default = [sys.executable, decide_promotion_gpt54.py]
    driver_timeout_s: float = DEFAULT_DRIVER_TIMEOUT_S,
    extra_driver_args: Optional[Sequence[str]] = None,
    # Stage 2 (cross-domain) opt-in.  Only consulted when
    # ``gate_mode == "offline-with-llm-judge"`` because no other
    # gate mode invokes the LLM judge.
    judge_enable_thinking: bool = False,
    judge_max_tokens: int = 256,
    # Block B3 — promotion gate ablation.  ``"gated"`` (default)
    # routes through the driver subprocess as historical.
    # ``"permissive"`` bypasses the driver entirely and auto-promotes
    # every proposal — used by the §5.5 "w/o lifecycle gating"
    # ablation to measure the gate's contribution in isolation.
    bypass_mode: str = "gated",
) -> PromotionStepReport:
    """Run the per-step Promotion pass for one trainer step.

    Parameters
    ----------
    step
        Current trainer step index (0-based).
    run_dir
        Trainer's run root.  Output goes to
        ``<run_dir>/promotion_decisions_out/step_<step>/``.
    proposals_run_dir
        Directory the Crafter hook wrote to (its ``run_dir`` field —
        the per-step root containing ``<corpus>/<source>/proposals.jsonl``
        files).
    legacy_bank_paths
        Map from trainer game name to per-game ``skill_bank.jsonl``.
        Used both as the synthetic ``--bank-run`` source (via symlinks)
        and as the writeback target.
    eligible_statuses
        Forwarded to :func:`writeback_promotion`.  Default
        ``{"active", "provisional", "shadow"}`` per F3.
    gate_mode
        Forwarded as ``--gate-mode``.  ``offline-synthetic`` (default)
        runs Stage 0 inline + ``LIMITED_PASS`` synthetic verdicts for
        Stages 1–4 ⇒ skills cap at PROVISIONAL.  Switch to
        ``--gate-mode external --gate-verdicts-run <harness_run>`` only
        when the Harness lands.
    driver_executable
        Override for testing.  Defaults to ``[sys.executable, <repo>/labeling_supplement/decide_promotion_gpt54.py]``.
    driver_timeout_s
        Hard wall-clock limit on the subprocess.  Phase-0 baseline is
        ~0.3 s/source; 300 s is generous even for a 13-game sweep.
    extra_driver_args
        Additional CLI flags forwarded to the subprocess (e.g.
        ``["--corpus", "gym_v"]`` to restrict the sweep).
    """
    t0 = time.monotonic()

    promotion_run_dir = (
        Path(run_dir) / "promotion_decisions_out" / f"step_{step:04d}"
    )
    promotion_run_dir.mkdir(parents=True, exist_ok=True)

    # Counts tally is computed both from disk (canonical) and from the
    # writeback projector. We start with empty defaults so an early-skip
    # path still returns a valid report.
    n_proposals_in = _count_proposals(Path(proposals_run_dir))
    if n_proposals_in == 0:
        elapsed = time.monotonic() - t0
        return PromotionStepReport(
            step=step,
            run_dir=Path(run_dir),
            promotion_run_dir=promotion_run_dir,
            n_proposals_in=0,
            n_promote=0, n_reject=0, n_defer=0, n_rollback=0,
            writeback_per_game={},
            driver_returncode=0,
            driver_wall_time_s=0.0,
            wall_time_s=elapsed,
            skipped=True,
            skipped_reason="no proposals on disk",
        )

    # Block B3: permissive bypass — when ``bypass_mode == "permissive"``
    # we rewrite ``gate_mode`` to ``"permissive"`` so the driver
    # subprocess auto-PASSes every stage (DRAFT → ACTIVE).  Writeback
    # still runs so the bank reflects the promoted skills; the only
    # thing skipped is the judge / verdict computation.  Keeps the
    # subprocess-based artifact layout intact so analysis scripts work
    # without a special case.
    if bypass_mode == "permissive":
        gate_mode = "permissive"

    # Build the synthetic --bank-run directory that decide_promotion expects.
    with tempfile.TemporaryDirectory(prefix=f"promotion-step{step}-bankrun-") as bank_run_str:
        bank_run = Path(bank_run_str)
        _materialise_synthetic_bank_run(
            bank_run=bank_run,
            legacy_bank_paths=legacy_bank_paths,
        )

        cmd = list(_resolve_driver_executable(driver_executable))
        cmd += [
            "--proposals-run", str(Path(proposals_run_dir).resolve()),
            "--bank-run", str(bank_run.resolve()),
            "--no-actions",                              # F5: no rollback signal in trainer
            "--gate-mode", gate_mode,
            "--output-dir", str(promotion_run_dir.resolve()),
            # Stage 2 cross-domain knobs are only meaningful when the
            # judge actually fires (gate_mode=="offline-with-llm-judge");
            # for other modes the driver simply ignores them.
            "--judge-max-tokens", str(int(judge_max_tokens)),
        ]
        if judge_enable_thinking:
            cmd += ["--enable-thinking"]
        if extra_driver_args:
            cmd += list(extra_driver_args)

        driver_t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=driver_timeout_s,
                cwd=str(_codebase_root()),
                check=False,
                env=_subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            logger.error(
                "promotion_hook: driver timed out after %.1fs at step=%d",
                driver_timeout_s, step,
            )
            return PromotionStepReport(
                step=step,
                run_dir=Path(run_dir),
                promotion_run_dir=promotion_run_dir,
                n_proposals_in=n_proposals_in,
                n_promote=0, n_reject=0, n_defer=0, n_rollback=0,
                writeback_per_game={},
                driver_returncode=124,                   # GNU timeout convention
                driver_wall_time_s=driver_timeout_s,
                wall_time_s=time.monotonic() - t0,
                skipped=True,
                skipped_reason=f"driver timeout: {exc}",
            )
        driver_wall = time.monotonic() - driver_t0

        if proc.returncode != 0:
            logger.error(
                "promotion_hook: driver returned %d at step=%d\n"
                "  stdout tail: %s\n  stderr tail: %s",
                proc.returncode, step,
                (proc.stdout or "")[-800:], (proc.stderr or "")[-800:],
            )
            return PromotionStepReport(
                step=step,
                run_dir=Path(run_dir),
                promotion_run_dir=promotion_run_dir,
                n_proposals_in=n_proposals_in,
                n_promote=0, n_reject=0, n_defer=0, n_rollback=0,
                writeback_per_game={},
                driver_returncode=proc.returncode,
                driver_wall_time_s=driver_wall,
                wall_time_s=time.monotonic() - t0,
                skipped=True,
                skipped_reason=f"driver returncode={proc.returncode}",
            )

    # Read decision counts from the driver's _run_summary.json (the
    # authoritative summary it writes per the docstring at lines 154-155).
    n_promote, n_reject, n_defer, n_rollback = _read_decision_counts(promotion_run_dir)

    # Project promoted skills back into the trainer's per-game banks.
    writeback_per_game: Dict[str, Dict[str, Any]] = {}
    for game, bank_path in legacy_bank_paths.items():
        corpus = corpus_for_game(game)
        pair_dir = promotion_run_dir / corpus / game
        if not pair_dir.is_dir():
            continue
        snap = find_latest_snapshot(pair_dir)
        if snap is None:
            # No snapshot ⇒ all proposals for this game were rejected, or
            # there were no proposals to begin with.  Don't fail; record
            # the no-op so dashboards can see it.
            writeback_per_game[game] = {
                "snapshot_path": None,
                "n_inserted": 0, "n_updated": 0,
                "n_skipped_status": 0, "n_skipped_invalid": 0,
            }
            continue
        try:
            wb: WritebackReport = writeback_promotion(
                snapshot_path=snap,
                legacy_bank_path=Path(bank_path),
                eligible_statuses=eligible_statuses,
            )
        except Exception as exc:                               # noqa: BLE001
            logger.exception(
                "promotion_hook: writeback failed for game=%s: %s", game, exc,
            )
            writeback_per_game[game] = {
                "snapshot_path": str(snap),
                "error": str(exc),
                "n_inserted": 0, "n_updated": 0,
            }
            continue
        writeback_per_game[game] = {
            "snapshot_path": str(snap),
            "n_total_in_snapshot": wb.n_total_in_snapshot,
            "n_eligible": wb.n_eligible,
            "n_inserted": wb.n_inserted,
            "n_updated": wb.n_updated,
            "n_skipped_status": wb.n_skipped_status,
            "n_skipped_invalid": wb.n_skipped_invalid,
            "inserted_skill_ids": wb.inserted_skill_ids,
            "updated_skill_ids": wb.updated_skill_ids,
        }

    elapsed = time.monotonic() - t0

    # Per-step summary file for trainer step_log + dashboards.
    summary = {
        "step": step,
        "promotion_run_dir": str(promotion_run_dir),
        "proposals_run_dir": str(proposals_run_dir),
        "n_proposals_in": n_proposals_in,
        "n_promote": n_promote,
        "n_reject": n_reject,
        "n_defer": n_defer,
        "n_rollback": n_rollback,
        "writeback_per_game": writeback_per_game,
        "driver_wall_time_s": driver_wall,
        "wall_time_s": elapsed,
        "params": {
            "gate_mode": gate_mode,
            "eligible_statuses": sorted(set(s.lower() for s in eligible_statuses)),
        },
    }
    try:
        (promotion_run_dir / "_step_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(
            "promotion_hook: could not write _step_summary.json: %s", exc,
        )

    return PromotionStepReport(
        step=step,
        run_dir=Path(run_dir),
        promotion_run_dir=promotion_run_dir,
        n_proposals_in=n_proposals_in,
        n_promote=n_promote, n_reject=n_reject,
        n_defer=n_defer, n_rollback=n_rollback,
        writeback_per_game=writeback_per_game,
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
    (``trainer/coevolution/_promotion_hook.py``)."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve_driver_executable(
    override: Optional[Sequence[str]],
) -> Sequence[str]:
    if override is not None:
        return list(override)
    driver_path = (
        _codebase_root()
        / "labeling_supplement" / "decide_promotion_gpt54.py"
    )
    return [sys.executable, str(driver_path)]


def _subprocess_env() -> Dict[str, str]:
    """Pin ``PYTHONPATH`` to the codebase root so the driver can import
    ``orchestrator``, ``skill_bank``, etc.  Inherits everything else from
    the parent's environment."""
    env = dict(os.environ)
    repo = str(_codebase_root())
    existing = env.get("PYTHONPATH", "")
    if repo not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{repo}{os.pathsep}{existing}" if existing else repo
        )
    return env


def _count_proposals(proposals_run_dir: Path) -> int:
    """Total non-blank lines across all
    ``<proposals_run_dir>/<corpus>/<source>/proposals.jsonl`` files."""
    if not proposals_run_dir.is_dir():
        return 0
    n = 0
    for jsonl in proposals_run_dir.glob("*/*/proposals.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8") as f:
                n += sum(1 for line in f if line.strip())
        except OSError:
            continue
    return n


def _materialise_synthetic_bank_run(
    *,
    bank_run: Path,
    legacy_bank_paths: Mapping[str, Path],
) -> None:
    """Build a synthetic ``--bank-run`` directory in the offline-mirror's
    ``<corpus>/<source>/skill_bank.jsonl`` shape.  Uses **symlinks** so
    the trainer's source-of-truth bank files remain untouched (read-only
    semantics matter — the writeback at the *end* of the hook is the
    only legitimate mutation path).

    On systems that don't support symlinks (rare on POSIX, never on
    sane CI), falls back to copying the file.
    """
    for game, bank_path in legacy_bank_paths.items():
        bank_path = Path(bank_path)
        if not bank_path.is_file():
            logger.debug(
                "promotion_hook: skipping synthetic bank-run entry for "
                "%s — file missing: %s",
                game, bank_path,
            )
            continue
        corpus = corpus_for_game(game)
        target_dir = bank_run / corpus / game
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "skill_bank.jsonl"
        try:
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            os.symlink(bank_path.resolve(), target_path)
        except OSError:
            # Fallback: copy contents.
            target_path.write_text(
                bank_path.read_text(encoding="utf-8"), encoding="utf-8",
            )


def _read_decision_counts(promotion_run_dir: Path) -> tuple:
    """Read aggregate (PROMOTE, REJECT, DEFER, ROLLBACK) counts from the
    driver's ``_run_summary.json`` if present, else compute them by
    walking ``promotion_decisions.jsonl`` files."""
    summary_path = promotion_run_dir / "_run_summary.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            by_decision = (data.get("by_decision") or {})
            return (
                int(by_decision.get("PROMOTE", 0)),
                int(by_decision.get("REJECT", 0)),
                int(by_decision.get("DEFER", 0)),
                int(by_decision.get("ROLLBACK", 0)),
            )
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    # Fallback: walk every promotion_decisions.jsonl.
    counts = {"PROMOTE": 0, "REJECT": 0, "DEFER": 0, "ROLLBACK": 0}
    for jsonl in promotion_run_dir.glob("*/*/promotion_decisions.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    d = str(row.get("decision") or "").upper()
                    if d in counts:
                        counts[d] += 1
        except OSError:
            continue
    return counts["PROMOTE"], counts["REJECT"], counts["DEFER"], counts["ROLLBACK"]


__all__ = [
    "DEFAULT_DRIVER_TIMEOUT_S",
    "DEFAULT_GATE_MODE",
    "PromotionStepReport",
    "run_promotion_step",
]
