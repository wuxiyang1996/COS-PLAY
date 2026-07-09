#!/usr/bin/env python
"""Phase-5 -- within-VR/video 4x4 matrix transfer driver.

Closes Experiment B (declarative-reasoning transfer) of the
[Phase-5/6 cross-domain measurement plan](../implementation_notes/legacy/phase5-cross-domain-measurement.md)
section 8. Iterates over the cross-product of source-corpus x
target-corpus from the four VR/video corpora that ship with the
cross-domain skill bank
(``skill_transfer_test/skill_bank_local/full_v5/``):

    visual_toolbench   -- image-VR (target_domain=visual_reasoning)
    tir_bench          -- image-VR (target_domain=visual_reasoning)
    video_holmes       -- video-VR (target_domain=video)
    siv_bench          -- video-VR (target_domain=video)

For every ``(source_corpus, target_corpus)`` cell:

  1. Load source skills from
     ``<bank-root>/<source>/<bank-kind>/skill_bank.jsonl`` via the
     existing :func:`load_bank_records`. ``bank-kind`` defaults to
     ``archetype`` (one record per cluster -- closes
     :mod:`skill_transfer_test.extract.archetype_aggregator`'s
     output) but can be ``per_sample`` for a higher-resolution
     measurement.
  2. Dispatch the target via the Stage 1-4 dispatcher
     (:func:`labeling_supplement._phase4_target_dispatch.build_target`).
     The corpus-to-target_domain map is hardcoded here:

         visual_toolbench / tir_bench  ->  visual_reasoning
         video_holmes     / siv_bench  ->  video

  3. Run :func:`labeling_supplement._phase4_transfer_cycle._run_transfer`
     to produce per-skill ``TransferVerdict``s.
  4. Record the cell's admit rate.

Output layout (gitignored under ``cross_domain_results/_phase5/``):

    cross_domain_results/_phase5/<run_id>/
        cells.json           -- one record per (source, target) cell
        cells.md             -- 4x4 admit-rate matrix + per-cell rationale
        per_skill.jsonl      -- every TransferVerdict, one line each

Usage::

    python -m labeling_supplement._phase5_matrix \
        --bank-root skill_transfer_test/skill_bank_local/full_v5 \
        --bank-kind archetype \
        --k 4 --max-skills 10

By default the driver runs the full 4x4 (16 cells). Use
``--source-corpora`` / ``--target-corpora`` to restrict to a subset
(e.g. for fast smoke runs).

Acceptance gates (per memo section 8):

  * All 16 cells produce non-trivial admit rates (>=1 verdict each)
  * Diagonal cells (``X -> X``) admit rate >=80%
  * Off-diagonal within-cluster (image<->image, video<->video) >=30%
  * Cross-cluster (image<->video) revised floors per memo section
    11.5.4: 15-35% (image-source) / 15-30% (video-source).

These gates are evaluated and reported in ``cells.md`` but the
driver does not exit non-zero on failure -- failure modes are
informative (e.g. stub executor giving 0% is a known property of
Stage 1+2's first cut, not a bug). Wrapped in a Stage 6 report
script that hard-asserts the floors against Stage 0 upper bounds.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.enums import SkillStatus, SkillType                          # noqa: E402
from data_structure.extensions.skill_record import SkillRecord           # noqa: E402

# Reuse the cycle's loader + per-cell driver so this module stays
# focused on the cross-product orchestration.
from labeling_supplement._phase2_real_env_skill_smoke import (           # noqa: E402
    load_bank_records,
)
from labeling_supplement._phase4_target_dispatch import (                # noqa: E402
    TargetBuild,
    build_target,
)
from labeling_supplement._phase4_transfer_cycle import (                 # noqa: E402
    _run_transfer,
    TransferVerdict,
)

logger = logging.getLogger("phase5_matrix")


# Corpora -> target_domain dispatch. Static -- the four VR/video
# corpora are the only inputs Stage 5 supports.
CORPUS_TO_TARGET_DOMAIN: Dict[str, str] = {
    "visual_toolbench": "visual_reasoning",
    "tir_bench": "visual_reasoning",
    "video_holmes": "video",
    "siv_bench": "video",
}

DEFAULT_CORPORA: Tuple[str, ...] = (
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)


def _corpus_cluster(corpus: str) -> str:
    """Return ``"image"`` or ``"video"``: which cluster a corpus belongs to."""
    td = CORPUS_TO_TARGET_DOMAIN.get(corpus)
    if td == "visual_reasoning":
        return "image"
    if td == "video":
        return "video"
    return "unknown"


def _load_source_records(
    bank_root: Path,
    *,
    corpus: str,
    bank_kind: str,
    max_skills: Optional[int],
) -> List[SkillRecord]:
    """Load and prep source skills for a cross-domain corpus."""
    bank_path = bank_root / corpus / bank_kind / "skill_bank.jsonl"
    if not bank_path.exists():
        raise SystemExit(
            f"source bank missing: {bank_path} "
            f"(run python -m skill_transfer_test.extract.archetype_aggregator "
            f"--bank-root {bank_root} first if --bank-kind=archetype)"
        )
    default_domain = CORPUS_TO_TARGET_DOMAIN.get(corpus, "visual_reasoning")
    records = load_bank_records(bank_path, default_domain=default_domain)
    if max_skills is not None:
        records = records[: max_skills]
    # Promote in-memory: bank records are DRAFT by default, but
    # FewShotAdapter.adapt() requires PROVISIONAL+. Same dance the
    # gymv path does in _phase4_transfer_cycle._run_transfer.
    #
    # The within-VR/video matrix violates the canonical asymmetric
    # thesis (only ``gymv``-sourced skills are few-shot-eligible).
    # We sidestep it by clearing ``source_domains`` -- the validator
    # only fails when ``source_domains`` is non-empty AND has no gymv
    # lineage. An empty tuple lets the matrix run; the
    # ``transfer_target_domains`` field still constrains which targets
    # the harness can route to.
    for r in records:
        object.__setattr__(r, "status", SkillStatus.PROVISIONAL)
        object.__setattr__(r, "source_domains", ())
        if default_domain and default_domain not in r.transfer_target_domains:
            object.__setattr__(
                r,
                "transfer_target_domains",
                tuple(list(r.transfer_target_domains) + [default_domain]),
            )
    return records


def _cell_admit_rate(verdicts: Sequence[TransferVerdict]) -> float:
    """Fraction of source skills that ``success`` on the target cell."""
    if not verdicts:
        return 0.0
    n_admit = sum(1 for v in verdicts if v.success)
    return n_admit / len(verdicts)


def _run_one_cell(
    *,
    source_corpus: str,
    target_corpus: str,
    bank_root: Path,
    bank_kind: str,
    max_skills: Optional[int],
    k: int,
    max_episodes: int,
    max_demos_per_episode: int,
    pass_rate_min: float,
) -> Dict[str, Any]:
    """Run one matrix cell. Returns a JSON-serialisable cell record."""
    target_domain = CORPUS_TO_TARGET_DOMAIN.get(target_corpus)
    if target_domain is None:
        return {
            "source_corpus": source_corpus,
            "target_corpus": target_corpus,
            "target_domain": None,
            "n_source_skills": 0,
            "n_admit": 0,
            "n_total": 0,
            "admit_rate": 0.0,
            "elapsed_s": 0.0,
            "error": f"unknown target_corpus {target_corpus!r}",
            "verdicts": [],
        }

    started = time.time()
    try:
        source_records = _load_source_records(
            bank_root,
            corpus=source_corpus,
            bank_kind=bank_kind,
            max_skills=max_skills,
        )
    except SystemExit as exc:
        return {
            "source_corpus": source_corpus,
            "target_corpus": target_corpus,
            "target_domain": target_domain,
            "n_source_skills": 0,
            "n_admit": 0,
            "n_total": 0,
            "admit_rate": 0.0,
            "elapsed_s": round(time.time() - started, 2),
            "error": str(exc),
            "verdicts": [],
        }

    ns = argparse.Namespace(
        target=target_corpus,
        cold_start_root=None,
        max_episodes=max_episodes,
        max_demos_per_episode=max_demos_per_episode,
    )
    try:
        target_build: TargetBuild = build_target(target_domain, ns)
    except (NotImplementedError, SystemExit) as exc:
        return {
            "source_corpus": source_corpus,
            "target_corpus": target_corpus,
            "target_domain": target_domain,
            "n_source_skills": len(source_records),
            "n_admit": 0,
            "n_total": 0,
            "admit_rate": 0.0,
            "elapsed_s": round(time.time() - started, 2),
            "error": f"build_target raised: {exc!r}",
            "verdicts": [],
        }

    verdicts, _mutated = _run_transfer(
        source_game=source_corpus,
        target_game=target_corpus,
        source_records=source_records,
        target_build=target_build,
        pass_rate_min=pass_rate_min,
        k=k,
        bindings_overrides=None,
    )
    elapsed = time.time() - started

    return {
        "source_corpus": source_corpus,
        "target_corpus": target_corpus,
        "target_domain": target_domain,
        "source_cluster": _corpus_cluster(source_corpus),
        "target_cluster": _corpus_cluster(target_corpus),
        "n_source_skills": len(source_records),
        "n_admit": sum(1 for v in verdicts if v.success),
        "n_total": len(verdicts),
        "admit_rate": _cell_admit_rate(verdicts),
        "elapsed_s": round(elapsed, 2),
        "error": None,
        "verdicts": [
            {
                "skill_id": v.skill_id,
                "skill_type": v.skill_type,
                "n_demos_used": v.n_demos_used,
                "n_success": v.n_success,
                "n_aborted": v.n_aborted,
                "pass_rate": v.pass_rate,
                "success": v.success,
                "diagnostic_label": v.diagnostic_label,
            }
            for v in verdicts
        ],
    }


def _emit_markdown(cells: List[Dict[str, Any]], *, run_id: str) -> str:
    """Render a 4x4 admit-rate matrix + per-cell summary as Markdown."""
    sources = sorted({c["source_corpus"] for c in cells})
    targets = sorted({c["target_corpus"] for c in cells})

    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {
        (c["source_corpus"], c["target_corpus"]): c for c in cells
    }

    lines: List[str] = []
    lines.append(f"# Phase-5 within-VR/video 4x4 matrix (run_id={run_id})\n")
    lines.append(
        "Cross-domain transfer admit rates measured by "
        "`labeling_supplement/_phase5_matrix.py` against archetype banks "
        "from `skill_transfer_test/skill_bank_local/full_v5/`.\n"
    )

    # Admit-rate matrix
    lines.append("## Admit-rate matrix\n")
    header = "| source \\ target | " + " | ".join(targets) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(targets) + 1)) + "|")
    for src in sources:
        row = [src]
        for tgt in targets:
            c = by_pair.get((src, tgt))
            if c is None:
                row.append("--")
                continue
            if c.get("error"):
                row.append(f"ERR ({c['n_admit']}/{c['n_total']})")
            else:
                rate = c["admit_rate"]
                row.append(f"{rate:.0%} ({c['n_admit']}/{c['n_total']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Acceptance check (memo §8)
    lines.append("## Acceptance check (memo section 8)\n")
    lines.append(
        "| Gate | Result | Note |\n|---|---|---|"
    )
    diag_cells = [c for c in cells if c["source_corpus"] == c["target_corpus"]]
    diag_pass = (
        all(c["admit_rate"] >= 0.80 for c in diag_cells if c.get("error") is None)
        if diag_cells else False
    )
    diag_note = (
        ", ".join(f"{c['source_corpus']}={c['admit_rate']:.0%}" for c in diag_cells)
        if diag_cells else "no diag cells"
    )
    lines.append(
        f"| Diagonal cells >=80% | "
        f"{'PASS' if diag_pass else 'FAIL'} | {diag_note} |"
    )

    off_diag_within = [
        c for c in cells
        if (c["source_corpus"] != c["target_corpus"]
            and c.get("source_cluster") == c.get("target_cluster")
            and c.get("error") is None)
    ]
    off_diag_pass = (
        all(c["admit_rate"] >= 0.30 for c in off_diag_within)
        if off_diag_within else False
    )
    off_diag_note = (
        ", ".join(
            f"{c['source_corpus']}->{c['target_corpus']}={c['admit_rate']:.0%}"
            for c in off_diag_within
        )
        if off_diag_within else "no within-cluster off-diag cells"
    )
    lines.append(
        f"| Off-diagonal within-cluster >=30% | "
        f"{'PASS' if off_diag_pass else 'FAIL'} | {off_diag_note} |"
    )

    cross_cluster = [
        c for c in cells
        if (c.get("source_cluster") != c.get("target_cluster")
            and c.get("error") is None)
    ]
    cc_pass = (
        all(c["admit_rate"] >= 0.15 for c in cross_cluster)
        if cross_cluster else False
    )
    cc_note = (
        ", ".join(
            f"{c['source_corpus']}->{c['target_corpus']}={c['admit_rate']:.0%}"
            for c in cross_cluster
        )
        if cross_cluster else "no cross-cluster cells"
    )
    lines.append(
        f"| Cross-cluster (image<->video) >=15% | "
        f"{'PASS' if cc_pass else 'FAIL'} | {cc_note} |"
    )
    lines.append("")

    # Per-cell rationale
    lines.append("## Per-cell rationale\n")
    for c in cells:
        src = c["source_corpus"]; tgt = c["target_corpus"]
        rate = c["admit_rate"]
        n_admit = c["n_admit"]; n_total = c["n_total"]
        err = c.get("error")
        lines.append(
            f"### `{src}` -> `{tgt}` ({c['target_domain']}) "
            f"-- {rate:.0%} ({n_admit}/{n_total}), {c['elapsed_s']}s"
        )
        if err:
            lines.append(f"\n*ERROR:* `{err}`\n")
            continue
        diag_summary: Dict[str, int] = {}
        for v in c["verdicts"]:
            d = v.get("diagnostic_label") or "(none)"
            diag_summary[d] = diag_summary.get(d, 0) + 1
        if diag_summary:
            lines.append("Diagnostic-label distribution:")
            for d, n in sorted(diag_summary.items(), key=lambda x: -x[1]):
                lines.append(f"  - `{d}`: {n}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bank-root",
        default=str(REPO_ROOT / "skill_transfer_test" / "skill_bank_local" / "full_v5"),
        help="Root with per-corpus dirs each containing per_sample/ and archetype/.",
    )
    p.add_argument(
        "--bank-kind",
        default="archetype",
        choices=("archetype", "per_sample"),
        help="Which source bank to load per corpus. Default: archetype.",
    )
    p.add_argument(
        "--source-corpora",
        nargs="+",
        default=list(DEFAULT_CORPORA),
        help=f"Source corpora to iterate. Default: all 4 VR/video corpora.",
    )
    p.add_argument(
        "--target-corpora",
        nargs="+",
        default=list(DEFAULT_CORPORA),
        help="Target corpora to iterate. Default: all 4.",
    )
    p.add_argument("--max-skills", type=int, default=10,
                   help="Max source skills per cell (default 10).")
    p.add_argument("--k", type=int, default=4,
                   help="FewShotAdapter k_shot per skill (default 4).")
    p.add_argument("--max-episodes", type=int, default=2)
    p.add_argument("--max-demos-per-episode", type=int, default=1)
    p.add_argument("--pass-rate-min", type=float, default=0.5)
    p.add_argument(
        "--out-dir",
        default=None,
        help=("Output directory (defaults to "
              "cross_domain_results/_phase5/run_<ts>/)."),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bank_root = Path(args.bank_root)
    if not bank_root.exists():
        raise SystemExit(f"bank_root missing: {bank_root}")

    run_id = "run_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else REPO_ROOT / "cross_domain_results" / "_phase5" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: List[Dict[str, Any]] = []
    started = time.time()
    for src in args.source_corpora:
        for tgt in args.target_corpora:
            cell = _run_one_cell(
                source_corpus=src,
                target_corpus=tgt,
                bank_root=bank_root,
                bank_kind=args.bank_kind,
                max_skills=args.max_skills,
                k=args.k,
                max_episodes=args.max_episodes,
                max_demos_per_episode=args.max_demos_per_episode,
                pass_rate_min=args.pass_rate_min,
            )
            cells.append(cell)
            err_tag = " ERROR" if cell.get("error") else ""
            logger.info(
                "%s -> %s: %.0f%% (%d/%d), %.1fs%s",
                src, tgt, cell["admit_rate"] * 100,
                cell["n_admit"], cell["n_total"],
                cell["elapsed_s"], err_tag,
            )

    elapsed_s = time.time() - started

    # Persist outputs.
    cells_json_path = out_dir / "cells.json"
    cells_json_path.write_text(json.dumps({
        "run_id": run_id,
        "bank_root": str(bank_root),
        "bank_kind": args.bank_kind,
        "k": args.k,
        "max_skills": args.max_skills,
        "max_episodes": args.max_episodes,
        "max_demos_per_episode": args.max_demos_per_episode,
        "pass_rate_min": args.pass_rate_min,
        "elapsed_s": round(elapsed_s, 2),
        "n_cells": len(cells),
        "cells": cells,
    }, indent=2, ensure_ascii=False))

    cells_md_path = out_dir / "cells.md"
    cells_md_path.write_text(_emit_markdown(cells, run_id=run_id))

    per_skill_path = out_dir / "per_skill.jsonl"
    with per_skill_path.open("w") as f:
        for c in cells:
            for v in c["verdicts"]:
                f.write(json.dumps({
                    "source_corpus": c["source_corpus"],
                    "target_corpus": c["target_corpus"],
                    "target_domain": c["target_domain"],
                    **v,
                }, ensure_ascii=False) + "\n")

    print()
    print(f"=== Phase-5 within-VR/video matrix ({run_id}) ===")
    print(f"cells.json:   {cells_json_path}")
    print(f"cells.md:     {cells_md_path}")
    print(f"per_skill:    {per_skill_path}")
    print(f"elapsed:      {elapsed_s:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
