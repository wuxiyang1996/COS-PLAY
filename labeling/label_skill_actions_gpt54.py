#!/usr/bin/env python
"""
Skill-conditioned action labeling — REPLAY mode.

For every step in every episode of the canonical SFT corpus, query the
per-game skill bank with the (already dual-axis labeled) intention plus
the step's ``summary_state`` and attach the selected skill guidance to
the record.  The original rollout action is preserved verbatim — this is
"replay-only", not regeneration.

Why no harness?
---------------
This labeler is **pure offline read-only replay**:
  * we never call ``env.step()``;
  * the action vocabulary is already grounded inside the rollout
    (every step records ``available_actions``);
  * the skill bank is consumed read-only — no draft promotion happens
    inside this loop.

So neither the per-task ``decision_agents/core/harness*.py`` contract
nor the (still un-implemented) skill-promotion gate are wired in.

Inputs
------
1. **Dual-axis labeled intentions** under
   ``labeling/intentions_out/<run>/{gym_v,env_wrappers}/<game>/episode_NNN.json``.
   Each step already has ``intentions`` (composite ``[OP/SG] note``),
   ``intention_tag`` (operator), ``intention_subgoal`` and
   ``intention_note`` plus the full original rollout fields
   (``state``, ``action``, ``summary_state``, ``available_actions``, ...).
2. **Per-game skill banks** under
   ``labeling/skill_bank_out/<run>/{gym_v,env_wrappers}/<game>/skill_bank.jsonl``
   produced by ``run_skill_discovery.sh``.

Output
------
``labeling/skill_actions_out/<run>/{gym_v,env_wrappers}/<game>/episode_NNN.json``
where every step record is the input verbatim plus two new fields:

  - ``skills`` : structured skill guidance (id, name, relevance,
    applicability, confidence, protocol, execution_hint,
    expected_effects, status, applicable_domains, ...).
  - ``skill_query`` : module-level I/O snapshot
    (query text, top-k candidates with scores, selected id, method).

Per-game ``_skill_actions_summary.json`` and run-level
``_run_summary.json`` capture coverage / diversity / confidence stats.

Usage
-----

    # Single (corpus, game) — used by the parallel dispatcher.
    python labeling/label_skill_actions_gpt54.py \\
        --intentions-run labeling/intentions_out/run_dualaxis_20260429_224917 \\
        --bank-run       labeling/skill_bank_out/run_20260430_030637 \\
        --corpus gym_v --game Temporal_Airstriker-v0 \\
        --output-dir     labeling/skill_actions_out/run_<ts>

    # All (corpus, game) pairs serially (slow; prefer the dispatcher).
    python labeling/label_skill_actions_gpt54.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup so the script runs from any cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for p in [str(WORKSPACE_ROOT), str(CODEBASE_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from skill_agents.skill_bank.bank import SkillBankMVP
except ImportError as exc:  # pragma: no cover
    SkillBankMVP = None  # type: ignore
    print(f"[label_skill_actions] WARN: SkillBankMVP unavailable ({exc}); skill_query disabled.")

try:
    from skill_agents.query import SkillQueryEngine
except ImportError as exc:  # pragma: no cover
    SkillQueryEngine = None  # type: ignore
    print(f"[label_skill_actions] WARN: SkillQueryEngine unavailable ({exc}).")


# ---------------------------------------------------------------------------
# Defaults — point at the latest dual-axis run + the latest skill bank run
# ---------------------------------------------------------------------------
DEFAULT_INTENTIONS_RUN = (
    CODEBASE_ROOT / "labeling" / "intentions_out" / "run_dualaxis_20260429_224917"
)
DEFAULT_BANK_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_bank_out" / "run_20260430_030637"
)
DEFAULT_OUTPUT_ROOT = CODEBASE_ROOT / "labeling" / "skill_actions_out"

CORPORA = ("gym_v", "env_wrappers")

# How many skills to retrieve per step (we keep all in skill_query.candidates,
# pick top-1 for skills.* guidance).
DEFAULT_TOP_K = 5


# ═══════════════════════════════════════════════════════════════════════
# Utility — parse summary_state string into a {predicate: value} dict
# ═══════════════════════════════════════════════════════════════════════

def parse_summary_state(summary_state: str) -> Dict[str, float]:
    """Parse ``key=value | key=value | ...`` into a flat predicate map.

    Numeric values become ``float(v)``; non-numeric truthy strings map
    to ``1.0``; explicit ``"false"`` / ``"none"`` map to ``0.0``.  Used
    only as a soft signal for ``SkillQueryEngine.select`` applicability
    scoring — passing an empty dict simply degrades to relevance-only
    ranking, which is fine when summary_state is sparse.
    """
    out: Dict[str, float] = {}
    if not summary_state:
        return out
    for seg in summary_state.split("|"):
        seg = seg.strip()
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        k = k.strip().lower().replace(" ", "_")
        v = v.strip()
        if not k:
            continue
        try:
            num = float(v.split(",")[0])
            out[k] = num
            continue
        except (ValueError, IndexError):
            pass
        low = v.lower()
        if low in ("false", "none", "null", ""):
            out[k] = 0.0
        elif low in ("true",):
            out[k] = 1.0
        else:
            out[f"{k}_{low.replace(' ', '_')}"] = 1.0
    return out


# ═══════════════════════════════════════════════════════════════════════
# Bank loading
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LoadedBank:
    """A loaded skill bank + its query engine for one (corpus, game)."""

    bank: Any
    engine: Any
    n_skills: int
    bank_path: Path


def resolve_bank_path(bank_run: Path, corpus: str, game: str) -> Optional[Path]:
    """Locate ``skill_bank.jsonl`` (preferred) or ``bank.jsonl`` for a game."""
    base = bank_run / corpus / game
    for fname in ("skill_bank.jsonl", "bank.jsonl"):
        p = base / fname
        if p.exists():
            return p
    return None


def load_bank(bank_run: Path, corpus: str, game: str) -> Optional[LoadedBank]:
    """Load the per-game bank + query engine; ``None`` when missing."""
    if SkillBankMVP is None:
        return None
    bank_path = resolve_bank_path(bank_run, corpus, game)
    if bank_path is None:
        print(f"[label_skill_actions] {corpus}/{game}: NO bank.jsonl under {bank_run}")
        return None

    try:
        bank = SkillBankMVP(path=str(bank_path))
        bank.load()
    except Exception as exc:
        print(f"[label_skill_actions] {corpus}/{game}: bank load FAILED: {exc}")
        return None

    n = len(bank)
    if n == 0:
        print(f"[label_skill_actions] {corpus}/{game}: bank loaded but empty")
        return None

    engine = None
    if SkillQueryEngine is not None:
        try:
            engine = SkillQueryEngine(bank)
        except Exception as exc:
            print(f"[label_skill_actions] {corpus}/{game}: SkillQueryEngine init FAILED: {exc}")
            engine = None

    return LoadedBank(bank=bank, engine=engine, n_skills=n, bank_path=bank_path)


# ═══════════════════════════════════════════════════════════════════════
# Per-step skill query
# ═══════════════════════════════════════════════════════════════════════

def _build_query_text(step: Dict[str, Any]) -> str:
    """Build the natural-language query string handed to the engine.

    Combines the dual-axis tagged intention with a compact slice of the
    step's ``summary_state`` so retrieval picks up both the strategic
    signal (operator/subgoal) and the entity-level tokens that match
    skill effect predicates.
    """
    intention = (step.get("intentions") or "").strip()
    summary = (step.get("summary_state") or "").strip()

    parts: List[str] = []
    if intention:
        parts.append(intention)
    if summary:
        clipped = summary if len(summary) <= 300 else summary[:300] + " ..."
        parts.append(clipped)
    return " | ".join(parts) if parts else "EXECUTE"


def _result_to_top1(result: Any) -> Dict[str, Any]:
    """Normalise a ``SkillSelectionResult`` (or duck-type) into a dict.

    Slim the output down to fields that downstream SFT consumers care
    about so episode JSONs don't bloat with the full contract dump.
    """
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        d = result.to_dict()
    elif isinstance(result, dict):
        d = dict(result)
    else:  # pragma: no cover
        d = {}

    keep = (
        "skill_id",
        "skill_name",
        "why_selected",
        "relevance",
        "applicability_score",
        "confidence",
        "execution_hint",
        "termination_hint",
        "preconditions",
        "expected_effects",
        "failure_modes",
        "matched_effects",
        "missing_effects",
        "n_instances",
        "pass_rate",
    )
    out: Dict[str, Any] = {k: d.get(k) for k in keep if k in d}
    out["applicability"] = d.get("applicability_score", d.get("applicability"))

    sid = out.get("skill_id")
    if sid is not None:
        record = None
        for attr in ("get_skill_record", "_skill_records", "skills"):
            container = getattr(result, attr, None) if not isinstance(result, dict) else None
            if container is not None:
                break
        meta_keys = ("applicable_domains", "verified_domains", "status",
                     "source_type", "evidence_role")
        for k in meta_keys:
            if k not in out and k in d:
                out[k] = d[k]
    return out


def _slim_candidate(result: Any) -> Dict[str, Any]:
    """Compact representation of one ranked candidate for the I/O log."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
    elif isinstance(result, dict):
        d = result
    else:  # pragma: no cover
        d = {}
    return {
        "skill_id": d.get("skill_id"),
        "relevance": d.get("relevance"),
        "applicability": d.get("applicability_score", d.get("applicability")),
        "confidence": d.get("confidence"),
    }


def query_step(
    step: Dict[str, Any],
    loaded: LoadedBank,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one bank query for *step*; return ``(skills, skill_query)``.

    Both return dicts are JSON-serialisable.  When the engine returns
    nothing we still emit ``skill_query`` with ``"empty": true`` so
    downstream readers can distinguish "queried, got nothing" from
    "never tried".
    """
    query_text = _build_query_text(step)
    state_predicates = parse_summary_state(step.get("summary_state") or "")

    skills_field: Dict[str, Any] = {}
    skill_query_field: Dict[str, Any] = {
        "query_text": query_text,
        "top_k_requested": top_k,
        "candidates": [],
        "selected_skill_id": None,
        "selection_method": "skill_query_engine_select",
        "empty": True,
    }

    if loaded.engine is None:
        skill_query_field["selection_method"] = "skill_bank_disabled"
        return skills_field, skill_query_field

    try:
        ranked = loaded.engine.select(
            query=query_text,
            current_state=state_predicates,
            top_k=top_k,
        )
    except Exception as exc:
        skill_query_field["error"] = f"{type(exc).__name__}: {exc}"
        skill_query_field["selection_method"] = "skill_query_engine_select_failed"
        return skills_field, skill_query_field

    if not ranked:
        return skills_field, skill_query_field

    skill_query_field["candidates"] = [_slim_candidate(r) for r in ranked]
    skill_query_field["selected_skill_id"] = getattr(
        ranked[0], "skill_id", None
    ) or (ranked[0].get("skill_id") if isinstance(ranked[0], dict) else None)
    skill_query_field["empty"] = False

    skills_field = _result_to_top1(ranked[0])
    return skills_field, skill_query_field


# ═══════════════════════════════════════════════════════════════════════
# Episode-level processing
# ═══════════════════════════════════════════════════════════════════════

def label_episode(
    episode_path: Path,
    loaded: LoadedBank,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Read *episode_path*, attach skill labels per step, return (data, stats)."""
    with episode_path.open("r") as f:
        data = json.load(f)

    exps = data.get("experiences") or data.get("steps") or []

    stats = {
        "n_steps": len(exps),
        "n_with_skill": 0,
        "n_failed": 0,
    }
    selected_counter: Counter[str] = Counter()
    confidences: List[float] = []

    for exp in exps:
        skills_field, skill_query_field = query_step(exp, loaded, top_k=top_k)
        exp["skills"] = skills_field if skills_field else None
        exp["skill_query"] = skill_query_field

        if skills_field and skills_field.get("skill_id"):
            stats["n_with_skill"] += 1
            selected_counter[skills_field["skill_id"]] += 1
            conf = skills_field.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
        if "error" in skill_query_field:
            stats["n_failed"] += 1

    data["skill_actions_label_meta"] = {
        "bank_path": str(loaded.bank_path),
        "n_skills_in_bank": loaded.n_skills,
        "top_k": top_k,
        "n_steps": stats["n_steps"],
        "n_with_skill": stats["n_with_skill"],
        "n_failed": stats["n_failed"],
        "coverage": (stats["n_with_skill"] / stats["n_steps"]) if stats["n_steps"] else 0.0,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "selection_histogram": dict(selected_counter),
        "labeled_at": datetime.utcnow().isoformat() + "Z",
    }
    return data, dict(stats)


def process_corpus_game(
    intentions_run: Path,
    bank_run: Path,
    output_root: Path,
    corpus: str,
    game: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    limit_episodes: Optional[int] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Drive labeling for one ``(corpus, game)`` pair."""
    in_dir = intentions_run / corpus / game
    if not in_dir.is_dir():
        msg = f"intentions dir missing: {in_dir}"
        if not quiet:
            print(f"[label_skill_actions] {corpus}/{game}: SKIP — {msg}")
        return {
            "corpus": corpus,
            "game": game,
            "status": "skip_no_intentions",
            "message": msg,
        }

    loaded = load_bank(bank_run, corpus, game)
    if loaded is None:
        return {
            "corpus": corpus,
            "game": game,
            "status": "skip_no_bank",
            "message": f"no bank under {bank_run}/{corpus}/{game}",
        }

    out_dir = output_root / corpus / game
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(in_dir.glob("episode_*.json"))
    if limit_episodes is not None:
        episode_files = episode_files[:limit_episodes]
    if not episode_files:
        return {
            "corpus": corpus,
            "game": game,
            "status": "skip_no_episodes",
            "message": f"no episode_*.json under {in_dir}",
        }

    t0 = time.time()
    agg = {
        "n_episodes": 0,
        "n_steps": 0,
        "n_with_skill": 0,
        "n_failed": 0,
    }
    selected_counter: Counter[str] = Counter()
    confidences: List[float] = []
    per_episode: List[Dict[str, Any]] = []

    for ep_path in episode_files:
        try:
            data, stats = label_episode(ep_path, loaded, top_k=top_k)
        except Exception as exc:
            print(f"[label_skill_actions] {corpus}/{game}/{ep_path.name}: ERROR {exc}")
            traceback.print_exc()
            continue

        out_path = out_dir / ep_path.name
        with out_path.open("w") as f:
            json.dump(data, f, indent=2)

        agg["n_episodes"] += 1
        agg["n_steps"] += stats["n_steps"]
        agg["n_with_skill"] += stats["n_with_skill"]
        agg["n_failed"] += stats["n_failed"]
        meta = data.get("skill_actions_label_meta", {})
        for sid, c in (meta.get("selection_histogram") or {}).items():
            selected_counter[sid] += c
        if meta.get("mean_confidence") is not None:
            confidences.append(meta["mean_confidence"])

        per_episode.append({
            "episode": ep_path.name,
            "n_steps": stats["n_steps"],
            "coverage": (stats["n_with_skill"] / stats["n_steps"]) if stats["n_steps"] else 0.0,
        })

    elapsed = time.time() - t0
    summary = {
        "corpus": corpus,
        "game": game,
        "status": "ok",
        "bank_path": str(loaded.bank_path),
        "n_skills_in_bank": loaded.n_skills,
        "n_episodes": agg["n_episodes"],
        "n_steps": agg["n_steps"],
        "n_with_skill": agg["n_with_skill"],
        "n_failed": agg["n_failed"],
        "coverage": (agg["n_with_skill"] / agg["n_steps"]) if agg["n_steps"] else 0.0,
        "mean_confidence_per_episode": (
            sum(confidences) / len(confidences)
        ) if confidences else None,
        "distinct_skills_selected": len(selected_counter),
        "selection_histogram": dict(selected_counter),
        "per_episode": per_episode,
        "elapsed_sec": round(elapsed, 2),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    with (out_dir / "_skill_actions_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if not quiet:
        print(
            f"[label_skill_actions] {corpus}/{game}: "
            f"{agg['n_episodes']} eps, {agg['n_steps']} steps, "
            f"coverage={summary['coverage']:.2%}, "
            f"{summary['distinct_skills_selected']} distinct skills, "
            f"{elapsed:.1f}s"
        )
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Multi-game driver
# ═══════════════════════════════════════════════════════════════════════

def discover_games(intentions_run: Path) -> List[Tuple[str, str]]:
    """Walk ``intentions_run`` to find every (corpus, game) pair with episodes."""
    pairs: List[Tuple[str, str]] = []
    for corpus in CORPORA:
        cdir = intentions_run / corpus
        if not cdir.is_dir():
            continue
        for game_dir in sorted(cdir.iterdir()):
            if not game_dir.is_dir():
                continue
            if any(game_dir.glob("episode_*.json")):
                pairs.append((corpus, game_dir.name))
    return pairs


def run_all(
    intentions_run: Path,
    bank_run: Path,
    output_root: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    limit_episodes: Optional[int] = None,
    only_corpora: Optional[List[str]] = None,
    only_games: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Process every ``(corpus, game)`` discovered under *intentions_run*."""
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = discover_games(intentions_run)
    if only_corpora:
        pairs = [p for p in pairs if p[0] in only_corpora]
    if only_games:
        pairs = [p for p in pairs if p[1] in only_games]

    print(
        f"[label_skill_actions] discovered {len(pairs)} (corpus, game) pairs "
        f"under {intentions_run}"
    )

    results: List[Dict[str, Any]] = []
    for corpus, game in pairs:
        try:
            res = process_corpus_game(
                intentions_run=intentions_run,
                bank_run=bank_run,
                output_root=output_root,
                corpus=corpus,
                game=game,
                top_k=top_k,
                limit_episodes=limit_episodes,
            )
        except Exception as exc:
            print(f"[label_skill_actions] {corpus}/{game}: FATAL {exc}")
            traceback.print_exc()
            res = {
                "corpus": corpus,
                "game": game,
                "status": "fatal",
                "message": f"{type(exc).__name__}: {exc}",
            }
        results.append(res)

    summary = aggregate_run_summary(
        results=results,
        intentions_run=intentions_run,
        bank_run=bank_run,
        output_root=output_root,
        top_k=top_k,
    )
    return summary


def aggregate_run_summary(
    *,
    results: List[Dict[str, Any]],
    intentions_run: Path,
    bank_run: Path,
    output_root: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Roll per-game summaries into one ``_run_summary.json``."""
    n_episodes = sum(r.get("n_episodes", 0) for r in results)
    n_steps = sum(r.get("n_steps", 0) for r in results)
    n_with_skill = sum(r.get("n_with_skill", 0) for r in results)
    n_failed = sum(r.get("n_failed", 0) for r in results)
    by_corpus: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "games": 0, "episodes": 0, "steps": 0, "with_skill": 0,
        "distinct_skills": set(),
    })
    for r in results:
        if r.get("status") != "ok":
            continue
        b = by_corpus[r["corpus"]]
        b["games"] += 1
        b["episodes"] += r.get("n_episodes", 0)
        b["steps"] += r.get("n_steps", 0)
        b["with_skill"] += r.get("n_with_skill", 0)
        for sid in (r.get("selection_histogram") or {}).keys():
            b["distinct_skills"].add(sid)
    by_corpus_serializable = {
        c: {
            "games": v["games"],
            "episodes": v["episodes"],
            "steps": v["steps"],
            "with_skill": v["with_skill"],
            "coverage": (v["with_skill"] / v["steps"]) if v["steps"] else 0.0,
            "distinct_skills": sorted(v["distinct_skills"]),
            "n_distinct_skills": len(v["distinct_skills"]),
        }
        for c, v in by_corpus.items()
    }

    summary = {
        "intentions_run": str(intentions_run),
        "bank_run": str(bank_run),
        "output_root": str(output_root),
        "top_k": top_k,
        "n_pairs": len(results),
        "n_pairs_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "n_with_skill": n_with_skill,
        "n_failed": n_failed,
        "coverage": (n_with_skill / n_steps) if n_steps else 0.0,
        "by_corpus": by_corpus_serializable,
        "per_pair": results,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    out = output_root / "_run_summary.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[label_skill_actions] run summary -> {out}")
    print(
        f"[label_skill_actions] TOTALS: "
        f"{summary['n_pairs_ok']}/{summary['n_pairs']} pairs ok, "
        f"{n_episodes} eps, {n_steps} steps, coverage={summary['coverage']:.2%}"
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Skill-conditioned action labeling (replay-only).",
    )
    p.add_argument(
        "--intentions-run",
        type=Path,
        default=DEFAULT_INTENTIONS_RUN,
        help=f"Dual-axis intentions run dir (default: {DEFAULT_INTENTIONS_RUN}).",
    )
    p.add_argument(
        "--bank-run",
        type=Path,
        default=DEFAULT_BANK_RUN,
        help=f"Skill bank run dir (default: {DEFAULT_BANK_RUN}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir; defaults to "
            f"{DEFAULT_OUTPUT_ROOT}/run_<utc-timestamp>."
        ),
    )
    p.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Top-k retrieval per step (default: {DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--limit-episodes", type=int, default=None,
        help="Process only the first N episodes per game (smoke test).",
    )
    p.add_argument(
        "--corpus", choices=CORPORA, default=None,
        help="Restrict to one corpus.",
    )
    p.add_argument(
        "--game", default=None,
        help="Restrict to one game (within --corpus).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Process every (corpus, game) pair under --intentions-run.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    intentions_run: Path = args.intentions_run
    bank_run: Path = args.bank_run
    if not intentions_run.is_dir():
        print(f"[label_skill_actions] intentions-run missing: {intentions_run}")
        return 2
    if not bank_run.is_dir():
        print(f"[label_skill_actions] bank-run missing: {bank_run}")
        return 2

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / f"run_{ts}"
    output_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "intentions_run": str(intentions_run),
        "bank_run": str(bank_run),
        "output_root": str(output_root),
        "top_k": args.top_k,
        "limit_episodes": args.limit_episodes,
        "corpus_filter": args.corpus,
        "game_filter": args.game,
        "all": args.all,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
    }
    with (output_root / "_run_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    if args.all or (args.corpus is None and args.game is None):
        run_all(
            intentions_run=intentions_run,
            bank_run=bank_run,
            output_root=output_root,
            top_k=args.top_k,
            limit_episodes=args.limit_episodes,
        )
        return 0

    if args.corpus is None or args.game is None:
        print("[label_skill_actions] --corpus AND --game required when not --all")
        return 2

    res = process_corpus_game(
        intentions_run=intentions_run,
        bank_run=bank_run,
        output_root=output_root,
        corpus=args.corpus,
        game=args.game,
        top_k=args.top_k,
        limit_episodes=args.limit_episodes,
    )
    aggregate_run_summary(
        results=[res],
        intentions_run=intentions_run,
        bank_run=bank_run,
        output_root=output_root,
        top_k=args.top_k,
    )
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
