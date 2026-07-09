"""Cross-domain dashboard sidecar — Layer D, run aside the training thread.

Architecture
------------
This is the "Pattern B" deployment of the Layer D cross-domain transfer
dashboard from
``implementation_notes/coevolution-cross-domain-integration.md`` §5:

* Architecturally: a separate Python process that polls the trainer's
  on-disk per-game ``skill_bank.jsonl`` files and never imports any
  trainer state. Run it on a different GPU (or different node — the
  only coupling is the run-dir filesystem) so the matrix subprocess
  never contends with vLLM / GRPO for compute.

* Implementation-wise (the "Pattern C" implementation of Pattern B):
  reuse ``trainer.coevolution._dashboard_hook.run_dashboard_step`` as a
  library call rather than re-implementing snapshotting + matrix
  subprocess + G1-G5 verdict computation by hand. This keeps the
  sidecar's verdicts bit-for-bit identical to what the in-trainer hook
  would emit if you flipped ``crafter_dashboard_enabled=True`` instead.

The sidecar is a measurement-only layer:
* It only reads ``<run_dir>/skillbank/<game>/skill_bank.jsonl``.
* It writes its own outputs into
  ``<run_dir>/cross_domain_dashboard_out/step_<NNNN>/`` (the same path
  the in-trainer hook uses; the two paths never collide because the
  in-trainer hook runs at trainer ``step`` indices while this sidecar
  uses its own monotonic ``poll_idx`` namespace).
* It logs G1-G5 verdicts + per-cluster admit rates to a separate wandb
  run tagged ``dashboard-sidecar``.

Race-safety vs. the trainer
----------------------------
``SkillBankMVP.save()`` was patched (alongside this sidecar) to do
``tempfile + os.fsync + os.replace`` so concurrent readers see either
the prior complete file or the new complete file — never a half-written
truncate-and-write transient. The sidecar additionally hashes all
banks each tick and skips re-running the matrix subprocess unless the
content has changed, both to dedupe and to avoid wasted GPU.

Usage
-----
::

    # 1) GPU-isolated sidecar on the same node as the trainer
    CUDA_VISIBLE_DEVICES=8 \\
    python scripts/dashboard_sidecar.py \\
        --run-dir runs/Qwen3.5-9B_20260503_120000 \\
        --targets video visual_reasoning \\
        --cadence-s 900 \\
        --wandb-project game-ai-coevolution

    # 2) Post-mortem one-shot after training finishes
    python scripts/dashboard_sidecar.py \\
        --run-dir runs/Qwen3.5-9B_20260503_120000 \\
        --once \\
        --targets video visual_reasoning osworld

Cross-refs
----------
* ``trainer/coevolution/_dashboard_hook.py`` — the hook this sidecar
  wraps as a library; G1-G5 verdict semantics live there.
* ``labeling_supplement/_phase4_transfer_matrix.py`` — the subprocess
  the hook spawns; consumes ``--game-bank-root`` + ``--target-corpora``.
* ``implementation_notes/coevolution-cross-domain-integration.md`` §5
  — Layer D design rationale.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.coevolution._dashboard_hook import (  # noqa: E402
    run_dashboard_step,
)

logger = logging.getLogger("dashboard_sidecar")


DEFAULT_TARGETS: tuple = ("video", "visual_reasoning")


def _discover_bank_paths(run_dir: Path, bank_subdir: str) -> Dict[str, Path]:
    """Walk ``<run_dir>/<bank_subdir>/<game>/skill_bank.jsonl`` and
    return ``{game: path}`` for every file present.

    Excludes composite keys (``avalon/good`` etc.) — Layer D's
    in-trainer caller passes ``simple_only=True`` and we mirror that.
    """
    out: Dict[str, Path] = {}
    bank_root = run_dir / bank_subdir
    if not bank_root.is_dir():
        return out
    for child in sorted(bank_root.iterdir()):
        if not child.is_dir():
            continue
        bank_file = child / "skill_bank.jsonl"
        if bank_file.is_file():
            out[child.name] = bank_file
    return out


def _bank_signature(bank_paths: Dict[str, Path]) -> str:
    """sha256 over the concatenated bank-file contents.

    Used as a cheap "did anything change since last poll?" key. Reads
    every file fully — fine because banks are <1 MB each and we only
    poll every minutes.
    """
    h = hashlib.sha256()
    for game in sorted(bank_paths):
        path = bank_paths[game]
        h.update(game.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


def _maybe_init_wandb(
    *,
    project: Optional[str],
    name: Optional[str],
    run_dir: Path,
    targets: Sequence[str],
):
    if not project:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb not installed — sidecar will skip wandb logging")
        return None
    return wandb.init(
        project=project,
        name=name or f"sidecar_{run_dir.name}",
        tags=["dashboard-sidecar"],
        config={
            "run_dir": str(run_dir),
            "targets": list(targets),
            "kind": "cross_domain_dashboard_sidecar",
        },
        resume="allow",
        reinit=True,
    )


def run_once(
    *,
    run_dir: Path,
    bank_paths: Dict[str, Path],
    targets: Sequence[str],
    max_skills_per_cell: int,
    timeout_s: float,
    poll_idx: int,
    wandb_run,
) -> Optional[dict]:
    """Single dashboard pass. Returns the metrics dict or ``None`` on skip."""
    if not bank_paths:
        logger.info("[poll %d] no per-game banks under run-dir yet — skipping",
                    poll_idx)
        return None

    logger.info(
        "[poll %d] running matrix on %d banks (%s) → %d targets (%s)",
        poll_idx, len(bank_paths), sorted(bank_paths),
        len(targets), list(targets),
    )

    report = run_dashboard_step(
        step=poll_idx,
        run_dir=run_dir,
        legacy_bank_paths=bank_paths,
        dashboard_targets=tuple(targets),
        dashboard_max_skills_per_cell=max_skills_per_cell,
        dashboard_driver_timeout_s=timeout_s,
    )

    if report.skipped:
        logger.info(
            "[poll %d] dashboard skipped: %s",
            poll_idx, report.skipped_reason,
        )
        return None

    logger.info(
        "[poll %d] cells=%d mean_admit=%.1f%% diag=%.1f%% off=%.1f%% "
        "gates=%s wall=%.1fs",
        poll_idx,
        report.n_cells_evaluated,
        report.mean_admit_rate * 100.0,
        report.mean_diagonal_admit_rate * 100.0,
        report.mean_off_diagonal_admit_rate * 100.0,
        report.gate_verdicts,
        report.wall_time_s,
    )

    metrics = report.to_metrics(prefix="cross_domain")
    if wandb_run is not None and metrics:
        wandb_run.log(metrics, step=poll_idx)

    return metrics


_STOP = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _STOP
        logger.info("Caught signal %d — stopping after current poll", signum)
        _STOP = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Trainer run-dir; banks live at "
                        "<run-dir>/<bank-subdir>/<game>/skill_bank.jsonl.")
    p.add_argument("--bank-subdir", type=str, default="skillbank",
                   help="Per-game bank subdirectory under --run-dir "
                        "(default: 'skillbank', matching CoEvolutionConfig "
                        "default).")
    p.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS),
                   help="Target corpora the matrix evaluates against. "
                        "Default: video visual_reasoning (VLM-only, no "
                        "docker/playwright). Add 'osworld' / 'browser' only "
                        "if the corresponding helper envs are reachable from "
                        "this process.")
    p.add_argument("--max-skills", type=int, default=5,
                   help="Forwarded to _phase4_transfer_matrix.py --max-skills "
                        "(default: 5).")
    p.add_argument("--cadence-s", type=float, default=900.0,
                   help="Seconds between polls in daemon mode "
                        "(default: 900 = 15 min).")
    p.add_argument("--timeout-s", type=float, default=3600.0,
                   help="Hard wall-clock cap on each matrix subprocess "
                        "(default: 3600 = 1h, mirroring the in-trainer "
                        "default).")
    p.add_argument("--once", action="store_true",
                   help="Run one matrix pass and exit (post-mortem mode).")
    p.add_argument("--max-polls", type=int, default=0,
                   help="Stop after N polls in daemon mode (0 = unlimited).")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="If set, log G1-G5 verdicts + admit rates to a wandb "
                        "run with tag 'dashboard-sidecar'.")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="Override the auto-generated wandb run name "
                        "('sidecar_<run-dir-basename>').")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    _install_signal_handlers()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        logger.error("run-dir does not exist: %s", run_dir)
        return 2

    wandb_run = _maybe_init_wandb(
        project=args.wandb_project,
        name=args.wandb_run_name,
        run_dir=run_dir,
        targets=args.targets,
    )

    poll_idx = 0
    last_signature: Optional[str] = None

    try:
        while not _STOP:
            bank_paths = _discover_bank_paths(run_dir, args.bank_subdir)
            signature = _bank_signature(bank_paths)
            if signature == last_signature and bank_paths:
                logger.debug("[poll %d] bank signature unchanged — skipping",
                             poll_idx)
            else:
                last_signature = signature
                run_once(
                    run_dir=run_dir,
                    bank_paths=bank_paths,
                    targets=args.targets,
                    max_skills_per_cell=args.max_skills,
                    timeout_s=args.timeout_s,
                    poll_idx=poll_idx,
                    wandb_run=wandb_run,
                )

            poll_idx += 1
            if args.once:
                break
            if args.max_polls and poll_idx >= args.max_polls:
                break

            slept = 0.0
            while slept < args.cadence_s and not _STOP:
                time.sleep(min(1.0, args.cadence_s - slept))
                slept += 1.0
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:  # noqa: BLE001
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
