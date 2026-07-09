"""Cross-domain transfer matrix runner (block D1).

For each phase snapshot under ``<run-dir>/phase_snapshots/`` (created by
``scripts/run_phase1_curriculum.sh``) we evaluate against every
non-training target domain and record the resulting primary metric.
The output is a 2-d matrix keyed by ``(snapshot, target_domain)`` that
the paper's transfer-matrix figure reads directly.

Note: this driver assumes a vLLM endpoint where the snapshot's LoRA has
been pre-loaded.  We expose ``--snapshot-loader`` as a hook (a shell
command template) that swaps adapters in between snapshots; if unset,
we just call the eval drivers and assume the user has hot-swapped the
adapter externally.

Example::

    python -m scripts.skillbridge_eval.run_transfer_matrix \\
        --run-dir runs/skillbridge_v12 \\
        --vllm-base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3.5-9B \\
        --domains visual_reasoning video gymv \\
        --snapshot-loader 'curl -X POST http://localhost:8000/v1/load_lora -d ' \\
        --output runs/skillbridge_v12/eval/transfer_matrix.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DRIVER_MODULE = {
    "browsergym":       "scripts.skillbridge_eval.eval_browsergym",
    "osworld":          "scripts.skillbridge_eval.eval_osworld",
    "visual_reasoning": "scripts.skillbridge_eval.eval_visual_reasoning",
    "video":            "scripts.skillbridge_eval.eval_video",
    "gymv":             "scripts.skillbridge_eval.eval_gymv",
}


_RESULT_PREFIX = {
    "browsergym":       "browsergym_result_",
    "osworld":          "osworld_result_",
    "visual_reasoning": "visual_reasoning_result_",
    "video":            "video_result_",
    "gymv":             "gymv_result_",
}


_PRIMARY_PATH = {
    "browsergym":       ("overall", "success_rate_macro"),
    "osworld":          ("overall", "success_rate_macro"),
    "visual_reasoning": ("overall", "accuracy_micro"),
    "video":            ("overall", "accuracy_micro"),
    "gymv":             ("overall", "mean_reward_macro"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--snapshots-dir", type=Path, default=None,
        help="Override <run-dir>/phase_snapshots/.",
    )
    p.add_argument(
        "--snapshots", nargs="+", default=None,
        help="Explicit list of snapshot names "
             "(e.g. phase_01_browsergym phase_02_osworld). "
             "Default: all directories under snapshots-dir sorted by name.",
    )
    p.add_argument(
        "--domains", nargs="+", default=list(_DRIVER_MODULE),
        help=f"Target domains to evaluate against. "
             f"Default: {list(_DRIVER_MODULE)}.",
    )
    p.add_argument(
        "--snapshot-loader", type=str, default=None,
        help="Shell command template for hot-swapping the snapshot's "
             "LoRA into vLLM.  ``{snapshot}`` is substituted with the "
             "snapshot dir; ``{model}`` with --model.  If unset, the "
             "driver assumes the user manages adapters externally.",
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument(
        "--vllm-base-url", type=str,
        default="http://localhost:8000/v1",
    )
    p.add_argument("--episodes-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument(
        "--vr-num-cases", type=int, default=200,
        help="num_test_cases passed to visual_reasoning / video runs.",
    )
    p.add_argument(
        "--gymv-games", nargs="+", default=None,
        help="Held-out GymV games.  Empty -> use the eval driver default.",
    )
    p.add_argument(
        "--include-self", action="store_true",
        help="By default the matrix skips the snapshot's own training "
             "domain (training-domain reward is reported elsewhere). "
             "Pass --include-self to evaluate every cell.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _resolve_snapshots(args: argparse.Namespace) -> List[Path]:
    snap_root = args.snapshots_dir or (args.run_dir / "phase_snapshots")
    if args.snapshots:
        return [snap_root / name for name in args.snapshots]
    if not snap_root.exists():
        return []
    return sorted([p for p in snap_root.iterdir() if p.is_dir()])


def _snapshot_meta(snap: Path) -> Dict[str, Any]:
    meta_path = snap / "phase_meta.json"
    if not meta_path.exists():
        return {"snapshot": snap.name}
    try:
        return {"snapshot": snap.name, **json.loads(meta_path.read_text())}
    except Exception:  # noqa: BLE001
        return {"snapshot": snap.name}


def _run_loader(template: Optional[str], snapshot: Path, model: str) -> int:
    if not template:
        return 0
    cmd = template.format(snapshot=str(snapshot), model=model)
    logger.info("transfer-matrix loader: $ %s", cmd)
    return subprocess.run(shlex.split(cmd)).returncode


def _run_domain(
    *,
    domain: str,
    snapshot: Path,
    args: argparse.Namespace,
) -> Optional[Path]:
    module = _DRIVER_MODULE[domain]
    eval_dir = args.run_dir / "eval" / "transfer" / snapshot.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", module,
        "--run-dir", str(args.run_dir),
        "--model", args.model,
        "--vllm-base-url", args.vllm_base_url,
        "--label", f"transfer:{snapshot.name}",
    ]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"{_RESULT_PREFIX[domain]}{ts}.json"
    cmd += ["--output", str(out_path)]
    if domain in ("browsergym", "osworld"):
        cmd += [
            "--episodes-per-task", str(args.episodes_per_task),
            "--max-steps", str(args.max_steps),
        ]
    if domain in ("visual_reasoning", "video"):
        cmd += ["--num-test-cases", str(args.vr_num_cases)]
    if domain == "gymv":
        cmd += [
            "--episodes-per-game", str(args.episodes_per_task),
            "--max-steps", str(args.max_steps),
        ]
        if args.gymv_games:
            cmd += ["--games", *args.gymv_games]

    env = os.environ.copy()
    env["VLLM_BASE_URL"] = args.vllm_base_url

    logger.info("transfer-matrix [%s] @ %s: $ %s", snapshot.name, domain,
                " ".join(cmd))
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        logger.warning(
            "transfer-matrix domain=%s snapshot=%s exit=%d",
            domain, snapshot.name, rc,
        )
    return out_path if out_path.exists() else None


def _read_primary(domain: str, path: Path) -> Optional[float]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not parse %s: %s", path, exc)
        return None
    bucket, key = _PRIMARY_PATH[domain]
    val = (data.get(bucket) or {}).get(key)
    return float(val) if isinstance(val, (int, float)) else None


def _matches_self_domain(meta: Dict[str, Any], domain: str) -> bool:
    g = (meta.get("game") or "").lower()
    if not g:
        return False
    if domain == "browsergym":
        return "browsergym" in g
    if domain == "osworld":
        return "osworld" in g
    if domain == "visual_reasoning":
        return any(t in g for t in ("vtb", "tir_bench", "visual_toolbench"))
    if domain == "video":
        return any(t in g for t in ("video_holmes", "siv_bench", "video"))
    if domain == "gymv":
        return any(t in g for t in (
            "crafter", "procgen", "babyai", "minigrid", "ataris",
        ))
    return False


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.run_dir.exists():
        logger.error("run-dir %s missing", args.run_dir)
        return 1

    snapshots = _resolve_snapshots(args)
    if not snapshots:
        logger.error("no snapshots resolved under %s",
                     args.snapshots_dir or args.run_dir / "phase_snapshots")
        return 1

    matrix: Dict[str, Dict[str, Any]] = {}
    for snap in snapshots:
        meta = _snapshot_meta(snap)
        if _run_loader(args.snapshot_loader, snap, args.model) != 0:
            logger.warning("snapshot loader failed for %s — continuing anyway",
                           snap.name)
        cell: Dict[str, Any] = {"meta": meta, "domains": {}}
        for domain in args.domains:
            if domain not in _DRIVER_MODULE:
                logger.warning("unknown domain %s — skipping", domain)
                continue
            if not args.include_self and _matches_self_domain(meta, domain):
                cell["domains"][domain] = {"skipped": "self_domain"}
                continue
            result_path = _run_domain(
                domain=domain, snapshot=snap, args=args,
            )
            if not result_path:
                cell["domains"][domain] = {"failed": True}
                continue
            primary = _read_primary(domain, result_path)
            cell["domains"][domain] = {
                "result_path": str(result_path),
                "primary": primary,
            }
        matrix[snap.name] = cell

    out_path = args.output or (args.run_dir / "eval" / "transfer_matrix.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "run_dir": str(args.run_dir),
                "domains": list(args.domains),
                "matrix": matrix,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("wrote transfer matrix %s", out_path)

    print("\n=== transfer matrix ===")
    header = "snapshot".ljust(40) + " | " + " | ".join(
        d.ljust(12) for d in args.domains
    )
    print(header)
    print("-" * len(header))
    for snap_name, cell in matrix.items():
        cells = []
        for d in args.domains:
            v = cell["domains"].get(d, {})
            if "skipped" in v:
                cells.append("self".ljust(12))
            elif "failed" in v:
                cells.append("FAIL".ljust(12))
            else:
                p = v.get("primary")
                cells.append((f"{p:.4f}" if p is not None else "—").ljust(12))
        print(snap_name.ljust(40) + " | " + " | ".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
