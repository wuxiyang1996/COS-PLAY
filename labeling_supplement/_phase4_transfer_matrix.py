#!/usr/bin/env python
"""Phase-5/6 -- Stage 6 full NxN cross-domain transfer matrix driver.

Closes the final cell of the Phase-5/6 cross-domain measurement plan
(implementation_notes/legacy/phase5-cross-domain-measurement.md, section 9).
Stages 0-5 are already shipped:

    Stage 0  -- pre-flight static audits (upper_bounds.csv).
    Stage 1  -- image-VR live measurement (visual_toolbench, tir_bench).
    Stage 2  -- video-VR live measurement (video_holmes, siv_bench).
    Stage 3  -- osworld live measurement.
    Stage 4  -- browsergym live measurement.
    Stage 5  -- within-VR/video 4x4 driver (`_phase5_matrix.py`).

Stage 6 is the unified driver: it generalises Stage 5's within-VR/video
matrix into the full cross-product over heterogeneous source banks
(game banks + cross-domain banks) and target corpora (games + VR/video
+ desktop + browser). For every (source_corpus, target_corpus) cell it:

  1. Loads source skills from the appropriate bank file (env_wrappers
     game / gym_v aggregated / cross-domain `<bank-kind>` bank). Game
     banks keep ``source_domains=('gymv',)``; cross-domain banks clear
     ``source_domains=()`` so the downstream
     ``_phase4_transfer_cycle._run_transfer`` path can re-populate it
     to ``('gymv',)`` -- the asymmetric-thesis bypass that keeps
     `FewShotAdapter._validate` happy when adapting cross-domain
     records onto a game (or any) target.
  2. Dispatches the target via the Stage 1-4 dispatcher
     (``labeling_supplement._phase4_target_dispatch.build_target``).
     Cells that raise (e.g. osworld with no usable sub-task name) are
     captured as `error` cells with ``admit_rate=0.0``.
  3. Runs ``_phase4_transfer_cycle._run_transfer`` to produce per-skill
     ``TransferVerdict``s and records the cell's admit rate.

I/O contract:

    Inputs
    ------
    --game-bank-root          (default labeling/skill_bank_out/run_20260429_235830)
        Root containing env_wrappers/<game>/skill_bank.jsonl AND
        gym_v/<Temporal_*>/episode_snapshots/episode_*/skill_bank.jsonl.
    --cross-domain-bank-root  (default skill_transfer_test/skill_bank_local/full_v5)
        Root with per-corpus dirs each containing per_sample/ and
        archetype/.
    --bank-kind {archetype,per_sample}
        Which cross-domain bank flavour to load. Default: archetype.
    --source-corpora ...
        Default: 4 env_wrappers games + 4 cross-domain VR/video corpora
        (8 sources). With --include-gym-v also pulls in the 13
        Temporal_*-v0 retro games.
    --target-corpora ...
        Default: 4 env_wrappers games + 4 cross-domain VR/video corpora
        (8 targets). osworld / browsergym are accepted explicitly but
        not added to the default set since their per-sub-task dispatch
        requires a name we don't synthesise here.

    Outputs
    -------
    cross_domain_results/_final/<run_id>/cells.json
        Per-cell record (source_corpus, target_corpus, target_domain,
        admit_rate, n_admit, n_total, error, verdicts list). Schema
        matches the Stage 6 report's input contract.
    cross_domain_results/_final/<run_id>/cells.md
        Markdown admit-rate matrix (rows=sources, cols=targets) plus
        per-cell diagnostic-label distribution.
    cross_domain_results/_final/<run_id>/per_skill.jsonl
        One JSON line per (source_corpus, target_corpus, skill) verdict.

    The unified Stage-6 _report.md is emitted by the sibling
    `_phase4_transfer_report.py` driver, which consumes cells.json +
    Stage 0's upper_bounds.csv.

Usage::

    python -m labeling_supplement._phase4_transfer_matrix \\
        --bank-kind archetype \\
        --max-skills 5 --k 2 \\
        --max-episodes 1 --max-demos-per-episode 1

    # smoke run (4x4)
    python -m labeling_supplement._phase4_transfer_matrix \\
        --source-corpora tetris twenty_forty_eight visual_toolbench tir_bench \\
        --target-corpora tetris twenty_forty_eight visual_toolbench tir_bench \\
        --max-skills 3 --k 2 --max-episodes 1 --max-demos-per-episode 1

This driver does NOT exit non-zero on individual cell failures (the
per-cell error field captures them). The companion
``_phase4_transfer_report.py`` evaluates the acceptance gates against
Stage 0's upper bounds and emits the final report.
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

from common.enums import SkillStatus                                    # noqa: E402
from data_structure.extensions.skill_record import SkillRecord          # noqa: E402

from labeling_supplement._harness_io_helpers import (                   # noqa: E402
    load_bank_records,
)
from labeling_supplement._phase2_real_env_skill_smoke import (          # noqa: E402
    DEFAULT_ACTIONS_ROOT,
)
from labeling_supplement._phase4_target_dispatch import (               # noqa: E402
    TargetBuild,
    build_target,
)
from labeling_supplement._phase4_transfer_cycle import (                # noqa: E402
    _run_transfer,
    TransferVerdict,
)

logger = logging.getLogger("phase4_transfer_matrix")


# ---------------------------------------------------------------------------
# Corpus registries
# ---------------------------------------------------------------------------

# env_wrappers games shipped under
# labeling/skill_bank_out/<run>/env_wrappers/<game>/skill_bank.jsonl.
ENV_WRAPPER_GAMES: Tuple[str, ...] = (
    "tetris",
    "twenty_forty_eight",
    "super_mario",
    "candy_crush",
)

# gym_v retro games. Bank lines live under
# labeling/skill_bank_out/<run>/gym_v/<game>/episode_snapshots/episode_*/skill_bank.jsonl
# and are aggregated on load.
GYM_V_GAMES: Tuple[str, ...] = (
    "Temporal_Airstriker-v0",
    "Temporal_AlteredBeast-v0",
    "Temporal_CastleOfIllusion-v0",
    "Temporal_CastlevaniaBloodlines-v0",
    "Temporal_Columns-v0",
    "Temporal_DynamiteHeaddy-v0",
    "Temporal_GoldenAxe-v0",
    "Temporal_KidChameleon-v0",
    "Temporal_MortalKombatII-v0",
    "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
)

# Cross-domain VR/video corpora (Stage 1 + Stage 2). Banks live under
# skill_transfer_test/skill_bank_local/full_v5/<corpus>/<bank-kind>/skill_bank.jsonl.
CROSS_DOMAIN_VR_VIDEO: Tuple[str, ...] = (
    "visual_toolbench",
    "tir_bench",
    "video_holmes",
    "siv_bench",
)

# Optional cross-domain targets (Stage 3 + Stage 4). NOT in default
# target list because their dispatch keys on a sub-task name we don't
# auto-synthesise; left as accepted args for explicit smoke runs.
EXTRA_CROSS_DOMAIN_TARGETS: Tuple[str, ...] = (
    "osworld",
    "browsergym",
)

# (corpus -> target_domain) for `build_target(target_domain, ns)`. The
# game-target rows treat ns.target as the game name; the cross-domain
# rows treat it as the sub-corpus / sub-task name.
CORPUS_TO_TARGET_DOMAIN: Dict[str, str] = {
    **{g: "gymv" for g in ENV_WRAPPER_GAMES},
    **{g: "gymv" for g in GYM_V_GAMES},
    "visual_toolbench": "visual_reasoning",
    "tir_bench": "visual_reasoning",
    "video_holmes": "video",
    "siv_bench": "video",
    "osworld": "osworld",
    "browsergym": "browser",
}

# Cluster taxonomy for the report's experiment partitioning.
#   game  : env_wrappers + gym_v
#   image : visual_toolbench + tir_bench
#   video : video_holmes + siv_bench
#   osworld / browser : their own clusters (one corpus each)
CORPUS_TO_CLUSTER: Dict[str, str] = {
    **{g: "game" for g in ENV_WRAPPER_GAMES},
    **{g: "game" for g in GYM_V_GAMES},
    "visual_toolbench": "image",
    "tir_bench": "image",
    "video_holmes": "video",
    "siv_bench": "video",
    "osworld": "osworld",
    "browsergym": "browser",
}


# ---------------------------------------------------------------------------
# Source-bank discovery
# ---------------------------------------------------------------------------


def _resolve_source_bank_paths(
    corpus: str,
    *,
    game_bank_root: Path,
    xd_bank_root: Path,
    bank_kind: str,
) -> Tuple[List[Path], Tuple[str, ...], str]:
    """Return ``(bank_paths, source_domains, source_cluster)`` for ``corpus``.

    ``source_domains`` is ``('gymv',)`` for game banks (kept) and ``()``
    for cross-domain banks (cleared per the Stage 5 asymmetric-thesis
    bypass; see ``_phase5_matrix._load_source_records`` for the same
    trick). ``source_cluster`` matches ``CORPUS_TO_CLUSTER``.
    """
    if corpus in ENV_WRAPPER_GAMES:
        path = game_bank_root / "env_wrappers" / corpus / "skill_bank.jsonl"
        return ([path], ("gymv",), "game")

    if corpus in GYM_V_GAMES:
        ep_root = game_bank_root / "gym_v" / corpus / "episode_snapshots"
        paths = sorted(ep_root.glob("episode_*/skill_bank.jsonl"))
        return (paths, ("gymv",), "game")

    if corpus in CROSS_DOMAIN_VR_VIDEO or corpus in EXTRA_CROSS_DOMAIN_TARGETS:
        path = xd_bank_root / corpus / bank_kind / "skill_bank.jsonl"
        return ([path], (), CORPUS_TO_CLUSTER.get(corpus, "unknown"))

    return ([], (), "unknown")


def _load_source_records(
    corpus: str,
    *,
    game_bank_root: Path,
    xd_bank_root: Path,
    bank_kind: str,
    max_skills: Optional[int],
) -> Tuple[List[SkillRecord], List[Path], Tuple[str, ...], str]:
    """Load source skills + flag the bank's source_domains lineage.

    Returns ``(records, paths_used, source_domains, source_cluster)``.
    Empty / missing banks return an empty ``records`` list and the
    caller is expected to record a missing-source error.
    """
    paths, source_domains, cluster = _resolve_source_bank_paths(
        corpus,
        game_bank_root=game_bank_root,
        xd_bank_root=xd_bank_root,
        bank_kind=bank_kind,
    )
    paths_used = [p for p in paths if p.exists() and p.stat().st_size > 0]

    default_domain = CORPUS_TO_TARGET_DOMAIN.get(corpus, "gymv")
    records: List[SkillRecord] = []
    for p in paths_used:
        records.extend(load_bank_records(p, default_domain=default_domain))

    # Apply Stage 5's bypass + status promotion. The bank rows are
    # DRAFT by default; FewShotAdapter.adapt() requires PROVISIONAL+.
    for r in records:
        object.__setattr__(r, "status", SkillStatus.PROVISIONAL)
        if source_domains:
            object.__setattr__(r, "source_domains", source_domains)
        else:
            # Cross-domain bank: clear source_domains so the matrix
            # can run cross-domain -> game-target cells without
            # FewShotAdapter raising. _run_transfer's
            # `if not skill.source_domains` shim repopulates with
            # ('gymv',), which IS in SOURCE_DOMAINS, so the validator
            # passes.
            object.__setattr__(r, "source_domains", ())
        if default_domain and default_domain not in r.transfer_target_domains:
            object.__setattr__(
                r,
                "transfer_target_domains",
                tuple(list(r.transfer_target_domains) + [default_domain]),
            )

    if max_skills is not None:
        records = records[: max_skills]
    return records, paths_used, source_domains, cluster


# ---------------------------------------------------------------------------
# Default source / target resolution
# ---------------------------------------------------------------------------


def _bank_path_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _resolve_default_source_corpora(
    *,
    game_bank_root: Path,
    xd_bank_root: Path,
    bank_kind: str,
    include_gym_v: bool,
) -> List[str]:
    out: List[str] = []
    for g in ENV_WRAPPER_GAMES:
        path = game_bank_root / "env_wrappers" / g / "skill_bank.jsonl"
        if _bank_path_nonempty(path):
            out.append(g)
    if include_gym_v:
        for g in GYM_V_GAMES:
            ep_root = game_bank_root / "gym_v" / g / "episode_snapshots"
            if any(_bank_path_nonempty(p)
                   for p in ep_root.glob("episode_*/skill_bank.jsonl")):
                out.append(g)
    for c in CROSS_DOMAIN_VR_VIDEO:
        path = xd_bank_root / c / bank_kind / "skill_bank.jsonl"
        if _bank_path_nonempty(path):
            out.append(c)
    return out


def _resolve_default_target_corpora(
    *,
    game_bank_root: Path,
) -> List[str]:
    """Default targets: 4 env_wrappers + 4 cross-domain VR/video.

    osworld + browsergym are intentionally excluded from the default
    set: their per-sub-task dispatch requires a name we don't auto
    synthesise here. Pass them explicitly via --target-corpora to
    exercise (cells will record an error if dispatch fails).
    """
    out: List[str] = []
    for g in ENV_WRAPPER_GAMES:
        # env_wrappers cold-start episodes are loaded by the gymv
        # builder via DEFAULT_ACTIONS_ROOT; presence of the source
        # bank is a reasonable proxy for "this game is buildable".
        path = game_bank_root / "env_wrappers" / g / "skill_bank.jsonl"
        if _bank_path_nonempty(path):
            out.append(g)
    out.extend(CROSS_DOMAIN_VR_VIDEO)
    return out


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------


def _cell_admit_rate(verdicts: Sequence[TransferVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.success) / len(verdicts)


def _empty_cell(
    *,
    source_corpus: str,
    source_cluster: str,
    source_bank_path: str,
    target_corpus: str,
    target_cluster: str,
    target_domain: Optional[str],
    elapsed_s: float,
    error: str,
    n_source_skills: int = 0,
) -> Dict[str, Any]:
    return {
        "source_corpus": source_corpus,
        "source_cluster": source_cluster,
        "source_bank_path": source_bank_path,
        "target_corpus": target_corpus,
        "target_cluster": target_cluster,
        "target_domain": target_domain,
        "n_source_skills": n_source_skills,
        "n_admit": 0,
        "n_total": 0,
        "admit_rate": 0.0,
        "elapsed_s": round(elapsed_s, 2),
        "error": error,
        "verdicts": [],
    }


def _run_one_cell(
    *,
    source_corpus: str,
    target_corpus: str,
    game_bank_root: Path,
    xd_bank_root: Path,
    bank_kind: str,
    actions_root: Path,
    max_skills: Optional[int],
    k: int,
    max_episodes: int,
    max_demos_per_episode: int,
    pass_rate_min: float,
) -> Dict[str, Any]:
    """Run one (source, target) cell and return a JSON-serialisable record."""
    source_cluster = CORPUS_TO_CLUSTER.get(source_corpus, "unknown")
    target_cluster = CORPUS_TO_CLUSTER.get(target_corpus, "unknown")
    target_domain = CORPUS_TO_TARGET_DOMAIN.get(target_corpus)

    started = time.time()

    if target_domain is None:
        return _empty_cell(
            source_corpus=source_corpus,
            source_cluster=source_cluster,
            source_bank_path="",
            target_corpus=target_corpus,
            target_cluster=target_cluster,
            target_domain=None,
            elapsed_s=time.time() - started,
            error=f"unknown target_corpus {target_corpus!r}",
        )

    try:
        records, paths_used, _src_doms, src_cluster_resolved = _load_source_records(
            source_corpus,
            game_bank_root=game_bank_root,
            xd_bank_root=xd_bank_root,
            bank_kind=bank_kind,
            max_skills=max_skills,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_cell(
            source_corpus=source_corpus,
            source_cluster=source_cluster,
            source_bank_path="",
            target_corpus=target_corpus,
            target_cluster=target_cluster,
            target_domain=target_domain,
            elapsed_s=time.time() - started,
            error=f"source load raised: {exc!r}",
        )

    source_bank_path = "; ".join(str(p) for p in paths_used)
    if src_cluster_resolved != "unknown":
        source_cluster = src_cluster_resolved

    if not records:
        return _empty_cell(
            source_corpus=source_corpus,
            source_cluster=source_cluster,
            source_bank_path=source_bank_path or "(missing)",
            target_corpus=target_corpus,
            target_cluster=target_cluster,
            target_domain=target_domain,
            elapsed_s=time.time() - started,
            error=f"no source skills loaded for {source_corpus!r}",
        )

    ns = argparse.Namespace(
        target=target_corpus,
        cold_start_root=None,
        actions_root=str(actions_root),
        max_episodes=max_episodes,
        max_demos_per_episode=max_demos_per_episode,
    )

    try:
        target_build: TargetBuild = build_target(target_domain, ns)
    except (NotImplementedError, SystemExit) as exc:
        return _empty_cell(
            source_corpus=source_corpus,
            source_cluster=source_cluster,
            source_bank_path=source_bank_path,
            target_corpus=target_corpus,
            target_cluster=target_cluster,
            target_domain=target_domain,
            elapsed_s=time.time() - started,
            error=f"build_target raised: {exc!r}",
            n_source_skills=len(records),
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_cell(
            source_corpus=source_corpus,
            source_cluster=source_cluster,
            source_bank_path=source_bank_path,
            target_corpus=target_corpus,
            target_cluster=target_cluster,
            target_domain=target_domain,
            elapsed_s=time.time() - started,
            error=f"build_target raised: {exc!r}",
            n_source_skills=len(records),
        )

    try:
        verdicts, _mut = _run_transfer(
            source_game=source_corpus,
            target_game=target_corpus,
            source_records=records,
            target_build=target_build,
            pass_rate_min=pass_rate_min,
            k=k,
            bindings_overrides=None,
        )
    except Exception as exc:  # noqa: BLE001
        return _empty_cell(
            source_corpus=source_corpus,
            source_cluster=source_cluster,
            source_bank_path=source_bank_path,
            target_corpus=target_corpus,
            target_cluster=target_cluster,
            target_domain=target_domain,
            elapsed_s=time.time() - started,
            error=f"_run_transfer raised: {exc!r}",
            n_source_skills=len(records),
        )

    elapsed_s = time.time() - started
    return {
        "source_corpus": source_corpus,
        "source_cluster": source_cluster,
        "source_bank_path": source_bank_path,
        "target_corpus": target_corpus,
        "target_cluster": target_cluster,
        "target_domain": target_domain,
        "n_source_skills": len(records),
        "n_admit": sum(1 for v in verdicts if v.success),
        "n_total": len(verdicts),
        "admit_rate": _cell_admit_rate(verdicts),
        "elapsed_s": round(elapsed_s, 2),
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


# ---------------------------------------------------------------------------
# Output emitters
# ---------------------------------------------------------------------------


def _emit_admit_matrix_md(
    cells: List[Dict[str, Any]],
    *,
    run_id: str,
    sources: Sequence[str],
    targets: Sequence[str],
) -> str:
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {
        (c["source_corpus"], c["target_corpus"]): c for c in cells
    }
    lines: List[str] = []
    lines.append(f"# Phase-5/6 Stage-6 transfer matrix (run_id={run_id})\n")
    lines.append(
        "Cross-domain transfer admit rates measured by "
        "`labeling_supplement/_phase4_transfer_matrix.py`. Each cell "
        "shows `<rate>% (admit/total)`. Empty / errored cells render "
        "as `ERR ...`. The companion `_phase4_transfer_report.py` "
        "consumes `cells.json` and writes the unified `_report.md` "
        "(Experiment-A/B/C tables, Stage 0 upper-bound comparison, "
        "acceptance gates).\n"
    )
    lines.append("## Admit-rate matrix\n")
    header = "| source \\\\ target | " + " | ".join(targets) + " |"
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

    lines.append("## Per-cell diagnostic distribution\n")
    for c in cells:
        src = c["source_corpus"]
        tgt = c["target_corpus"]
        rate = c["admit_rate"]
        n_admit = c["n_admit"]
        n_total = c["n_total"]
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--game-bank-root",
        default=str(REPO_ROOT / "labeling" / "skill_bank_out" / "run_20260429_235830"),
        help=("Root with env_wrappers/<game>/skill_bank.jsonl AND "
              "gym_v/<game>/episode_snapshots/episode_*/skill_bank.jsonl."),
    )
    p.add_argument(
        "--cross-domain-bank-root",
        default=str(REPO_ROOT / "skill_transfer_test" / "skill_bank_local" / "full_v5"),
        help=("Root with per-corpus dirs each containing per_sample/ "
              "and archetype/."),
    )
    p.add_argument(
        "--bank-kind",
        default="archetype",
        choices=("archetype", "per_sample"),
        help="Cross-domain bank flavour to load. Default: archetype.",
    )
    p.add_argument(
        "--actions-root",
        default=str(DEFAULT_ACTIONS_ROOT),
        help=("Root for env_wrappers cold-start episodes (gymv target "
              "demos). Default: labeling/skill_actions_out/run_20260430_064325."),
    )
    p.add_argument(
        "--source-corpora",
        nargs="+",
        default=None,
        help=("Source corpora to iterate. Default: 4 env_wrappers games "
              "+ 4 cross-domain VR/video corpora (8 sources). With "
              "--include-gym-v also adds the 13 Temporal_*-v0 retro "
              "games."),
    )
    p.add_argument(
        "--target-corpora",
        nargs="+",
        default=None,
        help=("Target corpora to iterate. Default: same 4 env_wrappers + "
              "4 cross-domain VR/video corpora."),
    )
    p.add_argument(
        "--include-gym-v",
        action="store_true",
        help=("Pull the 13 Temporal_*-v0 gym_v retro games into the "
              "default source list. Off by default (~13x more cells)."),
    )
    p.add_argument("--max-skills", type=int, default=5,
                   help="Max source skills per cell (default 5).")
    p.add_argument("--k", type=int, default=2,
                   help="FewShotAdapter k_shot per skill (default 2).")
    p.add_argument("--max-episodes", type=int, default=1)
    p.add_argument("--max-demos-per-episode", type=int, default=1)
    p.add_argument("--pass-rate-min", type=float, default=0.5)
    p.add_argument(
        "--out-dir",
        default=None,
        help=("Output directory (default: "
              "cross_domain_results/_final/run_<ts>/)."),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    game_bank_root = Path(args.game_bank_root)
    xd_bank_root = Path(args.cross_domain_bank_root)
    actions_root = Path(args.actions_root)

    if not game_bank_root.exists():
        logger.warning("game-bank-root missing: %s", game_bank_root)
    if not xd_bank_root.exists():
        logger.warning("cross-domain-bank-root missing: %s", xd_bank_root)

    # Source / target resolution.
    if args.source_corpora is None:
        source_corpora = _resolve_default_source_corpora(
            game_bank_root=game_bank_root,
            xd_bank_root=xd_bank_root,
            bank_kind=args.bank_kind,
            include_gym_v=args.include_gym_v,
        )
    else:
        source_corpora = list(args.source_corpora)
    if args.target_corpora is None:
        target_corpora = _resolve_default_target_corpora(
            game_bank_root=game_bank_root,
        )
    else:
        target_corpora = list(args.target_corpora)

    logger.info("source corpora (%d): %s", len(source_corpora), source_corpora)
    logger.info("target corpora (%d): %s", len(target_corpora), target_corpora)

    run_id = "run_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else REPO_ROOT / "cross_domain_results" / "_final" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: List[Dict[str, Any]] = []
    started = time.time()
    for src in source_corpora:
        for tgt in target_corpora:
            cell = _run_one_cell(
                source_corpus=src,
                target_corpus=tgt,
                game_bank_root=game_bank_root,
                xd_bank_root=xd_bank_root,
                bank_kind=args.bank_kind,
                actions_root=actions_root,
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

    cells_json_path = out_dir / "cells.json"
    cells_json_path.write_text(json.dumps({
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "game_bank_root": str(game_bank_root),
            "cross_domain_bank_root": str(xd_bank_root),
            "actions_root": str(actions_root),
            "bank_kind": args.bank_kind,
            "k": args.k,
            "max_skills": args.max_skills,
            "max_episodes": args.max_episodes,
            "max_demos_per_episode": args.max_demos_per_episode,
            "pass_rate_min": args.pass_rate_min,
            "include_gym_v": bool(args.include_gym_v),
        },
        "source_corpora": source_corpora,
        "target_corpora": target_corpora,
        "n_cells": len(cells),
        "elapsed_s": round(elapsed_s, 2),
        "cells": cells,
    }, indent=2, ensure_ascii=False))

    cells_md_path = out_dir / "cells.md"
    cells_md_path.write_text(_emit_admit_matrix_md(
        cells, run_id=run_id, sources=source_corpora, targets=target_corpora,
    ))

    per_skill_path = out_dir / "per_skill.jsonl"
    with per_skill_path.open("w") as f:
        for c in cells:
            for v in c["verdicts"]:
                f.write(json.dumps({
                    "source_corpus": c["source_corpus"],
                    "source_cluster": c["source_cluster"],
                    "target_corpus": c["target_corpus"],
                    "target_cluster": c["target_cluster"],
                    "target_domain": c["target_domain"],
                    **v,
                }, ensure_ascii=False) + "\n")

    print()
    print(f"=== Phase-5/6 Stage-6 transfer matrix ({run_id}) ===")
    print(f"cells.json:   {cells_json_path}")
    print(f"cells.md:     {cells_md_path}")
    print(f"per_skill:    {per_skill_path}")
    print(f"n_cells:      {len(cells)}")
    print(f"elapsed:      {elapsed_s:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
