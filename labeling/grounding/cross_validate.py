#!/usr/bin/env python
"""Cross-validate heuristic vs vision-LLM schemas in collected triples.

Reads all ``triples.jsonl`` files under a collected output directory
(e.g. ``labeling/output/grounding/gymv``) and computes per-section
agreement between the deterministic heuristic schema and the vision-LLM
schema (``gpt-5.5`` by default; set ``VLM_LABEL_MODEL`` to override).

Why this matters
----------------
The Phase-1 SFT plan (PLAN-VISUAL-GROUNDING-MILESTONES §5.1) treats the
heuristic schema as the gold for **structural fidelity** (it knows the
exact entity list from ``obs.text`` / AXTree) and the vision-LLM schema
as the gold for **visual richness** (it sees the actual pixels).  Phase-1
training requires that the two agree on entities + targets within a
tolerance — disagreements above the threshold mark a frame as a
"hard case" routed to human review or to the inner-MDP tool loop for
Path-B repair (PLAN-V-G-MILESTONES §3 routing policy).

Usage::

    python -m labeling.grounding.cross_validate \\
        --input_root labeling/output/grounding/gymv \\
        --output labeling/output/grounding/gymv/cross_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("labeling.grounding.cross_validate")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_wrapper.eval.metrics import (  # noqa: E402
    compute_field_accuracy,
    compute_format_compliance,
    compute_target_accuracy,
)
from vlm_wrapper.schema import semantic_validate  # noqa: E402


_SECTIONS = (
    "entities", "attributes", "relations",
    "state_flags", "targets", "actions",
)


def _iter_triples_files(input_root: Path) -> Iterator[Path]:
    """Yield every ``triples.jsonl`` under ``input_root`` recursively."""
    for path in input_root.rglob("triples.jsonl"):
        yield path


def _load_triples(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line in %s: %s", path, exc)


def _domain_for(triple: dict[str, Any]) -> str:
    """Infer the schema domain from a triple's identifying fields."""
    if "task_id" in triple and "url" in triple:
        return "browser"
    if "env_id" in triple:
        return "gymv"
    return "auto"


def _accumulate_mean(running: dict[str, list[float]], k: str, v: float) -> None:
    running.setdefault(k, []).append(v)


def cross_validate(
    input_root: Path,
    *,
    output: Path | None = None,
    hard_case_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run cross-validation over every ``triples.jsonl`` under ``input_root``.

    Returns a report dict whose top-level keys are:

    * ``n_triples`` / ``n_with_vision``                   — counts
    * ``format_compliance``                               — heuristic / vision
    * ``semantic_valid_rate``                             — heuristic / vision
    * ``mean_field_f1``                                   — per section
    * ``target_agreement`` / ``blocker_agreement``        — float
    * ``hard_cases``                                      — list of (frame, score)
                                                            triples below
                                                            ``hard_case_threshold``
    * ``per_env``                                         — per-env breakdown

    Hard cases are written to ``input_root / "hard_cases.jsonl"`` so the
    Phase-1 SFT pipeline can route them through Path-B (tool repair) or
    Path-C (offline teacher escalation).
    """
    section_scores: dict[str, list[float]] = {}
    target_agreements: list[float] = []
    blocker_agreements: list[float] = []
    fmt_heuristic = {"ok": 0, "n": 0}
    fmt_vision = {"ok": 0, "n": 0}
    sem_heuristic = {"ok": 0, "n": 0}
    sem_vision = {"ok": 0, "n": 0}
    per_env: dict[str, dict[str, Any]] = {}
    hard_cases: list[dict[str, Any]] = []
    n_triples = 0
    n_with_vision = 0

    for triples_path in _iter_triples_files(input_root):
        for triple in _load_triples(triples_path):
            n_triples += 1
            heuristic = triple.get("heuristic_schema") or ""
            vision = triple.get("vision_schema")
            domain = _domain_for(triple)

            env_key = (
                triple.get("env_id") or triple.get("task_id") or "unknown"
            )
            env_stats = per_env.setdefault(env_key, {
                "n": 0, "n_with_vision": 0,
                "field_scores": {},
                "target_agree": 0, "target_total": 0,
            })
            env_stats["n"] += 1

            ok_h, _ = compute_format_compliance(heuristic)
            fmt_heuristic["n"] += 1
            fmt_heuristic["ok"] += int(ok_h)
            sem_h = semantic_validate(heuristic, domain=domain)
            sem_heuristic["n"] += 1
            sem_heuristic["ok"] += int(sem_h.valid)

            if vision:
                n_with_vision += 1
                env_stats["n_with_vision"] += 1
                ok_v, _ = compute_format_compliance(vision)
                fmt_vision["n"] += 1
                fmt_vision["ok"] += int(ok_v)
                sem_v = semantic_validate(vision, domain=domain)
                sem_vision["n"] += 1
                sem_vision["ok"] += int(sem_v.valid)

                # Score vision against heuristic (heuristic is the
                # structural gold).
                field_acc = compute_field_accuracy(vision, heuristic)
                worst = 1.0
                for sec, score in field_acc.items():
                    _accumulate_mean(section_scores, sec, score)
                    sec_scores = env_stats["field_scores"].setdefault(sec, [])
                    sec_scores.append(score)
                    if score < worst:
                        worst = score

                tgt_ok, blk_ok = compute_target_accuracy(vision, heuristic)
                if tgt_ok is not None:
                    target_agreements.append(float(tgt_ok))
                    env_stats["target_total"] += 1
                    env_stats["target_agree"] += int(tgt_ok)
                if blk_ok is not None:
                    blocker_agreements.append(float(blk_ok))

                if worst < hard_case_threshold or (tgt_ok is False):
                    hard_cases.append({
                        "frame_path": triple.get("frame_path"),
                        "env": env_key,
                        "step": triple.get("step"),
                        "min_section_f1": worst,
                        "field_accuracy": field_acc,
                        "target_match": tgt_ok,
                        "blocker_match": blk_ok,
                    })

    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    report: dict[str, Any] = {
        "input_root": str(input_root),
        "n_triples": n_triples,
        "n_with_vision": n_with_vision,
        "format_compliance": {
            "heuristic": (
                fmt_heuristic["ok"] / fmt_heuristic["n"]
                if fmt_heuristic["n"] else None
            ),
            "vision": (
                fmt_vision["ok"] / fmt_vision["n"]
                if fmt_vision["n"] else None
            ),
        },
        "semantic_valid_rate": {
            "heuristic": (
                sem_heuristic["ok"] / sem_heuristic["n"]
                if sem_heuristic["n"] else None
            ),
            "vision": (
                sem_vision["ok"] / sem_vision["n"]
                if sem_vision["n"] else None
            ),
        },
        "mean_field_f1": {
            sec: round(sum(scores) / len(scores), 4)
            for sec, scores in section_scores.items()
        },
        "target_agreement": _mean(target_agreements),
        "blocker_agreement": _mean(blocker_agreements),
        "n_hard_cases": len(hard_cases),
        "hard_case_threshold": hard_case_threshold,
        "per_env": {
            env: {
                "n": stats["n"],
                "n_with_vision": stats["n_with_vision"],
                "mean_field_f1": {
                    sec: round(sum(s) / len(s), 4)
                    for sec, s in stats["field_scores"].items()
                },
                "target_agreement": (
                    stats["target_agree"] / stats["target_total"]
                    if stats["target_total"] else None
                ),
            }
            for env, stats in per_env.items()
        },
    }

    # Always persist hard cases — the Phase-1 trainer uses these for
    # Path-B routing during data construction.
    hc_path = input_root / "hard_cases.jsonl"
    with hc_path.open("w", encoding="utf-8") as f:
        for case in hard_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    logger.info("Wrote %d hard cases to %s", len(hard_cases), hc_path)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Wrote cross-validation report to %s", output)

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-validate heuristic vs vision schemas",
    )
    p.add_argument(
        "--input_root", required=True,
        help="Root of a collected grounding output, e.g. "
             "labeling/output/grounding/gymv",
    )
    p.add_argument(
        "--output", default=None,
        help="Where to write the JSON report.  Defaults to "
             "<input_root>/cross_validation.json",
    )
    p.add_argument(
        "--hard_case_threshold", type=float, default=0.5,
        help="Minimum per-section F1 below which a triple is flagged as "
             "a hard case.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"input_root {input_root} does not exist")
    output = Path(args.output) if args.output else (
        input_root / "cross_validation.json"
    )
    report = cross_validate(
        input_root,
        output=output,
        hard_case_threshold=args.hard_case_threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
