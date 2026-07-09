"""Periodic cross-domain dashboard hook for the trainer's co-evolution loop.

Layer D of the design in
``implementation_notes/coevolution-cross-domain-integration.md``: every
``crafter_dashboard_every_k_steps`` trainer steps, snapshot the
trainer's per-game ``skill_bank.jsonl`` files into a synthetic
``--game-bank-root``, subprocess-invoke
``labeling_supplement/_phase4_transfer_matrix.py`` against the full
configured target set, parse the resulting ``cells.json``, compute the
G1-G6 acceptance-gate verdicts (memo section 11.5.6), and emit a
structured metrics dict suitable for the trainer's wandb /
TensorBoard sink.

Splices into ``trainer/coevolution/orchestrator.py::co_evolution_loop``
at the end-of-step block. Off by default; cadence is decoupled from
the per-step transfer gate (Layer A) so dashboards can run at low
frequency (every 100 steps) while the per-step gate runs at higher
frequency (every K steps with K << 100) without doubling the
subprocess load.

Strict trainer-mode contract
----------------------------
* Subprocess-only coupling with the matrix driver, same rule as
  Layer A. No Python imports across the boundary.
* The hook **never** mutates the trainer's banks. Snapshots are
  copies — they cannot leak back. This is in contrast to Layer A's
  in-place demotion, which is intentional (Layer A is a gate; Layer
  D is a measurement).
* On any internal failure the hook returns a ``skipped`` report
  with the reason in-band. The orchestrator surfaces the failure
  in its step log but never breaks the rollout.

The dashboard is a *measurement* layer: it surfaces the live
trainer's cross-domain transfer health to the same wandb session
that's tracking GRPO loss / promotion counts. It does not influence
training.

Cross-refs
----------
* ``implementation_notes/coevolution-cross-domain-integration.md`` §5
  (Layer D — design rationale + acceptance criteria).
* ``labeling_supplement/_phase4_transfer_matrix.py`` — produces
  ``cells.json`` we parse here.
* ``labeling_supplement/_phase4_transfer_report.py::_section_acceptance_gates``
  — the canonical implementation of the G1-G6 verdicts; we re-implement
  inline rather than import (driver-isolation rule).
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DASHBOARD_DRIVER_TIMEOUT_S: float = 3600.0
DEFAULT_DASHBOARD_MAX_SKILLS_PER_CELL: int = 5

# G1-G6 thresholds match _phase4_transfer_report._section_acceptance_gates.
# Hardcoded here (rather than re-imported) to keep the trainer's
# subprocess boundary clean — see "no driver imports another driver's
# code" rule in crafter-harness-orchestrator-roles.md §6.3.
G1_DIAGONAL_FLOOR: float = 0.80
G2_WITHIN_CLUSTER_FLOOR: float = 0.30
G3_GAME_IMAGE_BAND: Tuple[float, float] = (0.15, 0.35)
G4_GAME_VIDEO_BAND: Tuple[float, float] = (0.15, 0.30)
G5_QA_TO_GAME_FLOOR: float = 0.05         # soft-FAIL above this


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DashboardReport:
    """One trainer-step dashboard pass."""

    step: int
    run_dir: Path
    dashboard_run_dir: Path
    n_cells_evaluated: int = 0
    n_cells_errored: int = 0
    mean_admit_rate: float = 0.0
    mean_diagonal_admit_rate: float = 0.0
    mean_off_diagonal_admit_rate: float = 0.0
    gate_verdicts: Dict[str, str] = field(default_factory=dict)   # {"G1": "PASS", ...}
    per_cluster_admit_rate: Dict[str, float] = field(default_factory=dict)
    driver_returncode: int = 0
    driver_wall_time_s: float = 0.0
    wall_time_s: float = 0.0
    skipped: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "run_dir": str(self.run_dir),
            "dashboard_run_dir": str(self.dashboard_run_dir),
            "n_cells_evaluated": self.n_cells_evaluated,
            "n_cells_errored": self.n_cells_errored,
            "mean_admit_rate": self.mean_admit_rate,
            "mean_diagonal_admit_rate": self.mean_diagonal_admit_rate,
            "mean_off_diagonal_admit_rate": self.mean_off_diagonal_admit_rate,
            "gate_verdicts": dict(self.gate_verdicts),
            "per_cluster_admit_rate": dict(self.per_cluster_admit_rate),
            "driver_returncode": self.driver_returncode,
            "driver_wall_time_s": self.driver_wall_time_s,
            "wall_time_s": self.wall_time_s,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
        }

    def to_metrics(self, *, prefix: str = "cross_domain") -> Dict[str, float]:
        """Flatten into a flat ``{key: value}`` numeric dict for the
        trainer's wandb / TensorBoard sink. String gate verdicts get
        encoded as ``1.0=PASS, 0.0=FAIL, -1.0=N-A, 0.5=soft-FAIL`` so
        they show up as scalars in the dashboard.

        Skipped reports return an empty dict — the caller will detect
        the skip via ``skipped is True`` and surface the reason in the
        step log instead.
        """
        if self.skipped:
            return {}
        out: Dict[str, float] = {
            f"{prefix}/n_cells_evaluated": float(self.n_cells_evaluated),
            f"{prefix}/n_cells_errored": float(self.n_cells_errored),
            f"{prefix}/mean_admit_rate": float(self.mean_admit_rate),
            f"{prefix}/mean_diagonal_admit_rate": float(
                self.mean_diagonal_admit_rate,
            ),
            f"{prefix}/mean_off_diagonal_admit_rate": float(
                self.mean_off_diagonal_admit_rate,
            ),
            f"{prefix}/driver_wall_time_s": float(self.driver_wall_time_s),
        }
        for gate, verdict in self.gate_verdicts.items():
            out[f"{prefix}/gates/{gate}"] = _verdict_to_scalar(verdict)
        for cluster, rate in self.per_cluster_admit_rate.items():
            out[f"{prefix}/per_cluster/{cluster}"] = float(rate)
        return out


def _verdict_to_scalar(verdict: str) -> float:
    """Map G1-G6 verdict strings to wandb-friendly scalars."""
    v = (verdict or "").upper()
    if v == "PASS":
        return 1.0
    if v == "FAIL":
        return 0.0
    if v == "SOFT-FAIL":
        return 0.5
    return -1.0                                         # N-A / unknown


# ---------------------------------------------------------------------------
# Cadence helper
# ---------------------------------------------------------------------------


def should_run_dashboard(
    *,
    step: int,
    every_k_steps: int,
    enabled: bool,
) -> bool:
    """Returns ``True`` iff the dashboard should fire at ``step``.

    Off when:
      * ``enabled=False``,
      * ``every_k_steps <= 0`` (disabled-via-cadence),
      * ``step % every_k_steps != 0``.

    Step 0 fires when enabled — gives the dashboard a baseline before
    any training has happened, useful for sanity-checking the bank
    layout.
    """
    if not enabled:
        return False
    if int(every_k_steps) <= 0:
        return False
    return (int(step) % int(every_k_steps)) == 0


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------


def run_dashboard_step(
    *,
    step: int,
    run_dir: Path,
    legacy_bank_paths: Mapping[str, Path],
    dashboard_targets: Sequence[str],
    dashboard_sources: Optional[Sequence[str]] = None,
    dashboard_max_skills_per_cell: int = DEFAULT_DASHBOARD_MAX_SKILLS_PER_CELL,
    dashboard_driver_timeout_s: float = DEFAULT_DASHBOARD_DRIVER_TIMEOUT_S,
    driver_executable: Optional[Sequence[str]] = None,
    extra_driver_args: Optional[Sequence[str]] = None,
) -> DashboardReport:
    """Run the periodic cross-domain dashboard pass.

    Parameters
    ----------
    step
        Trainer step index.
    run_dir
        Trainer's run root.  Output goes to
        ``<run_dir>/cross_domain_dashboard_out/step_<step>/``.
    legacy_bank_paths
        Map from game name to per-game ``skill_bank.jsonl``. Snapshots
        are *copies* placed under ``<dashboard_run_dir>/snapshot_bank_root/``;
        the trainer's banks are never mutated.
    dashboard_targets
        Target corpora the matrix driver evaluates against. Empty
        list ⇒ skipped (no dashboard).
    dashboard_sources
        Source corpora to iterate. Defaults to the keys of
        ``legacy_bank_paths`` so the dashboard only spans the
        trainer's own games.
    dashboard_max_skills_per_cell
        Forwarded to the matrix driver's ``--max-skills`` flag.
    dashboard_driver_timeout_s
        Hard wall-clock limit on the subprocess. Default 1h is
        generous for the full N×N sweep with conda-env helper boots.
    driver_executable
        Override for testing.
    extra_driver_args
        Additional CLI flags forwarded to the subprocess.

    Returns
    -------
    DashboardReport
        Structured metrics + gate verdicts + subprocess metadata.
        Use :meth:`DashboardReport.to_metrics` to flatten for wandb.
    """
    t0 = time.monotonic()

    dashboard_run_dir = (
        Path(run_dir) / "cross_domain_dashboard_out" / f"step_{step:04d}"
    )
    dashboard_run_dir.mkdir(parents=True, exist_ok=True)

    if not legacy_bank_paths:
        return _skipped(
            step=step, run_dir=run_dir, dashboard_run_dir=dashboard_run_dir,
            reason="no legacy_bank_paths provided", t0=t0,
        )
    if not dashboard_targets:
        return _skipped(
            step=step, run_dir=run_dir, dashboard_run_dir=dashboard_run_dir,
            reason="no dashboard_targets configured", t0=t0,
        )

    # Snapshot the trainer's banks. We copy (not symlink) so the
    # matrix driver gets a stable view even if the trainer's writeback
    # races with the subprocess.
    snapshot_root = dashboard_run_dir / "snapshot_bank_root"
    n_snapshotted = _snapshot_bank_root(
        snapshot_root=snapshot_root,
        legacy_bank_paths=legacy_bank_paths,
    )
    if n_snapshotted == 0:
        return _skipped(
            step=step, run_dir=run_dir, dashboard_run_dir=dashboard_run_dir,
            reason="no per-game banks available to snapshot", t0=t0,
        )

    sources = (
        list(dashboard_sources)
        if dashboard_sources is not None
        else sorted(legacy_bank_paths.keys())
    )

    cmd = list(_resolve_driver_executable(driver_executable))
    cmd += [
        "--game-bank-root", str(snapshot_root.resolve()),
        "--source-corpora", *sources,
        "--target-corpora", *list(dashboard_targets),
        "--max-skills", str(dashboard_max_skills_per_cell),
        "--out-dir", str(dashboard_run_dir.resolve()),
    ]
    if extra_driver_args:
        cmd += list(extra_driver_args)

    driver_t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=dashboard_driver_timeout_s,
            cwd=str(_codebase_root()),
            check=False,
            env=_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "dashboard_hook: driver timed out after %.1fs at step=%d",
            dashboard_driver_timeout_s, step,
        )
        return DashboardReport(
            step=step,
            run_dir=Path(run_dir),
            dashboard_run_dir=dashboard_run_dir,
            driver_returncode=124,
            driver_wall_time_s=dashboard_driver_timeout_s,
            wall_time_s=time.monotonic() - t0,
            skipped=True,
            skipped_reason=f"driver timeout: {exc}",
        )
    driver_wall = time.monotonic() - driver_t0

    if proc.returncode != 0:
        logger.error(
            "dashboard_hook: driver returned %d at step=%d\n"
            "  stdout tail: %s\n  stderr tail: %s",
            proc.returncode, step,
            (proc.stdout or "")[-800:], (proc.stderr or "")[-800:],
        )
        return DashboardReport(
            step=step,
            run_dir=Path(run_dir),
            dashboard_run_dir=dashboard_run_dir,
            driver_returncode=proc.returncode,
            driver_wall_time_s=driver_wall,
            wall_time_s=time.monotonic() - t0,
            skipped=True,
            skipped_reason=f"driver returncode={proc.returncode}",
        )

    # Parse cells.json + compute summary statistics.
    cells_data = _read_cells_json(dashboard_run_dir)
    if cells_data is None:
        return DashboardReport(
            step=step,
            run_dir=Path(run_dir),
            dashboard_run_dir=dashboard_run_dir,
            driver_returncode=0,
            driver_wall_time_s=driver_wall,
            wall_time_s=time.monotonic() - t0,
            skipped=True,
            skipped_reason="cells.json missing or malformed",
        )
    cells = list(cells_data.get("cells") or [])
    measurable = [c for c in cells if not c.get("error")]
    n_errored = len(cells) - len(measurable)

    mean_all = _mean_admit(measurable)
    diag_cells = [
        c for c in measurable
        if c.get("source_corpus") == c.get("target_corpus")
    ]
    off_diag_cells = [
        c for c in measurable
        if c.get("source_corpus") != c.get("target_corpus")
    ]
    mean_diag = _mean_admit(diag_cells)
    mean_off = _mean_admit(off_diag_cells)
    per_cluster = _per_target_cluster_means(measurable)
    gates = _compute_gate_verdicts(measurable)

    # Persist a per-step summary alongside the cells.json.
    summary = {
        "step": step,
        "dashboard_run_dir": str(dashboard_run_dir),
        "n_cells_evaluated": len(measurable),
        "n_cells_errored": n_errored,
        "mean_admit_rate": mean_all,
        "mean_diagonal_admit_rate": mean_diag,
        "mean_off_diagonal_admit_rate": mean_off,
        "per_cluster_admit_rate": per_cluster,
        "gate_verdicts": gates,
        "driver_wall_time_s": driver_wall,
        "wall_time_s": time.monotonic() - t0,
    }
    try:
        (dashboard_run_dir / "_step_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(
            "dashboard_hook: could not write _step_summary.json: %s", exc,
        )

    return DashboardReport(
        step=step,
        run_dir=Path(run_dir),
        dashboard_run_dir=dashboard_run_dir,
        n_cells_evaluated=len(measurable),
        n_cells_errored=n_errored,
        mean_admit_rate=mean_all,
        mean_diagonal_admit_rate=mean_diag,
        mean_off_diagonal_admit_rate=mean_off,
        gate_verdicts=gates,
        per_cluster_admit_rate=per_cluster,
        driver_returncode=0,
        driver_wall_time_s=driver_wall,
        wall_time_s=time.monotonic() - t0,
        skipped=False,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _codebase_root() -> Path:
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
    env = dict(os.environ)
    repo = str(_codebase_root())
    existing = env.get("PYTHONPATH", "")
    if repo not in existing.split(os.pathsep):
        env["PYTHONPATH"] = (
            f"{repo}{os.pathsep}{existing}" if existing else repo
        )
    return env


def _skipped(
    *,
    step: int,
    run_dir: Path,
    dashboard_run_dir: Path,
    reason: str,
    t0: float,
) -> DashboardReport:
    return DashboardReport(
        step=step,
        run_dir=Path(run_dir),
        dashboard_run_dir=dashboard_run_dir,
        wall_time_s=time.monotonic() - t0,
        skipped=True,
        skipped_reason=reason,
    )


def _snapshot_bank_root(
    *,
    snapshot_root: Path,
    legacy_bank_paths: Mapping[str, Path],
) -> int:
    """Copy the trainer's per-game ``skill_bank.jsonl`` files into a
    synthetic ``--game-bank-root`` layout. Returns the number of
    snapshotted files. Missing files are tolerated (skipped).

    Layout: ``<snapshot_root>/env_wrappers/<game>/skill_bank.jsonl``,
    matching :func:`_phase4_transfer_matrix._resolve_default_target_corpora`'s
    walk pattern.
    """
    n = 0
    for game, bank_path in legacy_bank_paths.items():
        bank_path = Path(bank_path)
        if not bank_path.is_file():
            continue
        out_dir = snapshot_root / "env_wrappers" / game
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "skill_bank.jsonl"
        try:
            target.write_text(
                bank_path.read_text(encoding="utf-8"), encoding="utf-8",
            )
            n += 1
        except OSError as exc:
            logger.debug(
                "dashboard_hook: failed to snapshot %s: %s", bank_path, exc,
            )
    return n


def _read_cells_json(dashboard_run_dir: Path) -> Optional[Dict[str, Any]]:
    path = dashboard_run_dir / "cells.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            "dashboard_hook: failed to read %s: %s", path, exc,
        )
        return None


def _mean_admit(cells: Sequence[Mapping[str, Any]]) -> float:
    if not cells:
        return 0.0
    rates = [float(c.get("admit_rate") or 0.0) for c in cells]
    return sum(rates) / len(rates)


def _per_target_cluster_means(
    cells: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """Group measurable cells by ``target_cluster`` and compute the
    per-cluster mean admit rate. Cells without a ``target_cluster``
    field are bucketed under ``"unknown"``."""
    by_cluster: Dict[str, List[float]] = {}
    for c in cells:
        cluster = str(c.get("target_cluster") or "unknown")
        rate = float(c.get("admit_rate") or 0.0)
        by_cluster.setdefault(cluster, []).append(rate)
    return {
        cluster: (sum(rates) / len(rates)) if rates else 0.0
        for cluster, rates in by_cluster.items()
    }


def _compute_gate_verdicts(
    cells: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    """Compute G1-G5 verdicts (memo §11.5.6).

    Re-implements the logic in
    :func:`labeling_supplement._phase4_transfer_report._section_acceptance_gates`
    inline to keep the trainer's subprocess boundary clean. G6
    (upper-bound conformance) is omitted: it requires a separate
    ``upper_bounds.csv`` artefact that lives in the offline Stage-0
    pipeline and is not always available during a live trainer run.
    """
    verdicts: Dict[str, str] = {}

    if not cells:
        for g in ("G1", "G2", "G3", "G4", "G5"):
            verdicts[g] = "N-A"
        return verdicts

    # G1: diagonal cells (source == target) >= 80%.
    diag = [
        c for c in cells
        if c.get("source_corpus") == c.get("target_corpus")
    ]
    if not diag:
        verdicts["G1"] = "N-A"
    else:
        verdicts["G1"] = (
            "PASS"
            if all(float(c.get("admit_rate") or 0.0) >= G1_DIAGONAL_FLOOR
                   for c in diag)
            else "FAIL"
        )

    # G2: within-cluster off-diagonal (same cluster, different corpus) >= 30%.
    within_off = [
        c for c in cells
        if (c.get("source_cluster") == c.get("target_cluster"))
        and (c.get("source_corpus") != c.get("target_corpus"))
    ]
    if not within_off:
        verdicts["G2"] = "N-A"
    else:
        verdicts["G2"] = (
            "PASS"
            if all(float(c.get("admit_rate") or 0.0) >= G2_WITHIN_CLUSTER_FLOOR
                   for c in within_off)
            else "FAIL"
        )

    # G3: cross-cluster game <-> image-VR in [15%, 35%].
    g3_cells = [
        c for c in cells
        if (c.get("source_cluster") == "game"
            and c.get("target_cluster") == "image")
        or (c.get("source_cluster") == "image"
            and c.get("target_cluster") == "game")
    ]
    if not g3_cells:
        verdicts["G3"] = "N-A"
    else:
        lo, hi = G3_GAME_IMAGE_BAND
        verdicts["G3"] = (
            "PASS"
            if all(lo <= float(c.get("admit_rate") or 0.0) <= hi
                   for c in g3_cells)
            else "FAIL"
        )

    # G4: cross-cluster game <-> video-VR in [15%, 30%].
    g4_cells = [
        c for c in cells
        if (c.get("source_cluster") == "game"
            and c.get("target_cluster") == "video")
        or (c.get("source_cluster") == "video"
            and c.get("target_cluster") == "game")
    ]
    if not g4_cells:
        verdicts["G4"] = "N-A"
    else:
        lo, hi = G4_GAME_VIDEO_BAND
        verdicts["G4"] = (
            "PASS"
            if all(lo <= float(c.get("admit_rate") or 0.0) <= hi
                   for c in g4_cells)
            else "FAIL"
        )

    # G5: QA-source -> game-target near-zero (<5%); soft-FAIL otherwise.
    g5_cells = [
        c for c in cells
        if c.get("source_cluster") in ("image", "video")
        and c.get("target_cluster") == "game"
    ]
    if not g5_cells:
        verdicts["G5"] = "N-A"
    else:
        max_rate = max(
            float(c.get("admit_rate") or 0.0) for c in g5_cells
        )
        verdicts["G5"] = "PASS" if max_rate < G5_QA_TO_GAME_FLOOR else "soft-FAIL"

    return verdicts


__all__ = [
    "DEFAULT_DASHBOARD_DRIVER_TIMEOUT_S",
    "DEFAULT_DASHBOARD_MAX_SKILLS_PER_CELL",
    "DashboardReport",
    "run_dashboard_step",
    "should_run_dashboard",
]
