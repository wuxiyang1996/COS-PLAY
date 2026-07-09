"""build_osworld_subset50.py — generate a deterministic 50-task OSWorld
subset for the 4-model teacher baseline.

Why a separate script (and not just a one-liner):
-------------------------------------------------
The 4-provider OSWorld baseline is a *paired* comparison — every
model is scored on the *identical* 50 tasks so the per-task delta is
meaningful. The plan at
``implementation_notes/osworld-4model-baseline-plan.md`` §3 explicitly
requires:

  * Per-domain selection (5 tasks × 10 domains = 50).
  * Deterministic ordering — sort task ids lexicographically and take
    the first 5 — so re-running the script today, tomorrow, or on
    another machine produces byte-identical output.
  * A pinned subset *file* (not just a seed) committed to the repo so
    upstream changes to ``test_nogdrive.json`` cannot silently shift
    the comparison set.
  * A manifest line capturing the source-catalog hash + sampling rule
    + script git-sha for reviewer reproducibility.

The output catalog drops into the same shape as ``test_nogdrive.json``
(``{domain: [task_id, …]}``) so existing OSWorld driver code reads it
without changes.

Usage:
    python scripts/build_osworld_subset50.py
    python scripts/build_osworld_subset50.py --tasks_per_domain 10  # 100-task variant
    python scripts/build_osworld_subset50.py --dry_run               # print, don't write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


# Canonical paths — keep in sync with the plan §7 output layout.
DEFAULT_SOURCE = Path("/workspace/OSWorld/evaluation_examples/test_nogdrive.json")
DEFAULT_SUBSET = Path("/workspace/OSWorld/evaluation_examples/test_nogdrive_subset50_v1.json")
DEFAULT_MANIFEST = Path("/workspace/Multi-hop-Reasoning-VLM-Agent/runs/osworld_baseline_50/manifest.json")


def _load_catalog(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"[FATAL] catalog at {path} is not a top-level dict")
    out: Dict[str, List[str]] = {}
    for dom, tasks in data.items():
        if not isinstance(tasks, list):
            raise SystemExit(
                f"[FATAL] domain {dom!r} value is {type(tasks).__name__}, "
                f"expected list"
            )
        out[str(dom)] = [str(t) for t in tasks]
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha_for(path: Path) -> str:
    """Best-effort git sha for the given file (HEAD commit). Returns ''
    on any failure — manifest is informational, not load-bearing."""
    try:
        repo = path.parent
        sha = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return sha
    except Exception:
        return ""


def build_subset(
    catalog: Dict[str, List[str]],
    *,
    tasks_per_domain: int,
) -> Dict[str, List[str]]:
    """Take ``tasks_per_domain`` lex-sorted task ids per domain.

    If a domain has fewer tasks than ``tasks_per_domain`` we silently
    take all of them and emit a warning to stderr — the alternative
    (raise) would block reasonable subset sizes for sparse domains
    like ``thunderbird`` (15 tasks) at K=20.
    """
    out: Dict[str, List[str]] = {}
    for dom in sorted(catalog.keys()):
        tasks = sorted(catalog[dom])  # lex order is deterministic
        chosen = tasks[:tasks_per_domain]
        if len(chosen) < tasks_per_domain:
            print(
                f"[WARN] domain {dom!r} has only {len(chosen)} tasks "
                f"(requested {tasks_per_domain}); taking all of them.",
                file=sys.stderr,
            )
        out[dom] = chosen
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_catalog", "--source-catalog",
                   type=Path, default=DEFAULT_SOURCE,
                   help=f"input catalog file (default: {DEFAULT_SOURCE})")
    p.add_argument("--subset_out", "--subset-out",
                   type=Path, default=DEFAULT_SUBSET,
                   help=f"output subset catalog file (default: {DEFAULT_SUBSET})")
    p.add_argument("--manifest_out", "--manifest-out",
                   type=Path, default=DEFAULT_MANIFEST,
                   help=f"manifest JSON file (default: {DEFAULT_MANIFEST})")
    p.add_argument("--tasks_per_domain", "--tasks-per-domain",
                   type=int, default=5,
                   help="how many tasks per domain (default: 5 → 50 total over 10 domains)")
    p.add_argument("--dry_run", "--dry-run", action="store_true",
                   help="print the chosen task ids but do NOT write any files")
    args = p.parse_args()

    src = args.source_catalog.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"[FATAL] source catalog not found: {src}")

    catalog = _load_catalog(src)
    subset = build_subset(catalog, tasks_per_domain=args.tasks_per_domain)
    total = sum(len(v) for v in subset.values())

    print("=" * 64)
    print(f"  source:           {src}")
    print(f"  source SHA-256:   {_file_sha256(src)[:16]}...")
    print(f"  source size:      {sum(len(v) for v in catalog.values())} tasks "
          f"across {len(catalog)} domains")
    print(f"  subset size:      {total} tasks "
          f"({args.tasks_per_domain}/domain × {len(subset)} domains)")
    print("=" * 64)
    for dom in sorted(subset.keys()):
        print(f"  {dom:25s} {len(subset[dom]):>2d}  "
              f"({', '.join(t[:8] for t in subset[dom])})")
    print("=" * 64)

    if args.dry_run:
        print("  DRY RUN — no files written.")
        return 0

    # Write subset
    args.subset_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.subset_out, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  wrote subset:     {args.subset_out}")

    # Write manifest
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_catalog": str(src),
        "source_sha256": _file_sha256(src),
        "source_total_tasks": sum(len(v) for v in catalog.values()),
        "source_domains": sorted(catalog.keys()),
        "subset_catalog": str(args.subset_out.resolve()),
        "subset_sha256": _file_sha256(args.subset_out),
        "subset_total_tasks": total,
        "tasks_per_domain": args.tasks_per_domain,
        "sampling_rule": (
            "Per domain, sort tasks lexicographically by task_id and "
            "take the first N (= tasks_per_domain). Deterministic + "
            "reviewer-reproducible. Pinned to the source_sha256 above; "
            "if the source catalog ever changes upstream, regenerating "
            "the subset will produce a new file with a new "
            "subset_sha256 and the comparison MUST be redone end to end."
        ),
        "script_path": str(Path(__file__).resolve()),
        "script_git_sha": _git_sha_for(Path(__file__).resolve()),
        "tasks_per_domain_breakdown": {
            dom: len(subset[dom]) for dom in sorted(subset.keys())
        },
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  wrote manifest:   {args.manifest_out}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
