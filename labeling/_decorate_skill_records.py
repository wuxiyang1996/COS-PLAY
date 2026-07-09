#!/usr/bin/env python3
"""Decorate extractor outputs with ``SkillRecord``-shape future-proofing fields.

The two skill-bank extractors (``extract_skillbank_gpt54.py`` for env_wrappers
and ``extract_skillbank_gymv_gpt54.py`` for gym-v) produce per-env outputs:

  <env_dir>/skill_bank.jsonl     — one row per skill (Skill.to_dict())
  <env_dir>/skill_catalog.json   — {"skills": [...], "n_skills": ...}

Today these rows do *not* yet carry the canonical ``SkillRecord`` fields
required by the unified gate (PLAN-UNIFIED-SKILL-GATE §3.1). This script
walks the extractor output tree after the run finishes and adds the five
fields needed for the future ``lifecycle.ingest_draft`` import shim — no
gate evaluation is performed, the skills only get *shaped* so they can
be ingested later without a destructive migration.

Fields added (all idempotent):

  source_type         : "mined_from_trace"           (SkillSourceType.MINED)
  applicable_domains  : ["gymv"]                     (game-foundry source)
  verified_domains    : []                           (filled by gate Stage 3 later)
  evidence_role       : <GATHER|VERIFY|REASON|COMMIT>  (from skill.tags / segment op)
  status              : "draft"                      (SkillStatus.DRAFT)
  feasible_tasks      : ["<source_name>"]            (intra-domain task axis,
                                                      harness/README §22)
  verified_tasks      : []                           (filled by gate Stage 3a
                                                      transfer cycle later)

Plus a top-level ``corpus`` ("gym_v" | "env_wrappers") for traceability.

The script also adds a sibling ``_lifecycle_meta.json`` to each env folder
describing the provenance (``cold_start_run``, ``intentions_run``,
``model``, ``skills_extracted``, ``decorator_version``).

Idempotency: if ``source_type`` is already present on a row, that row is
left untouched. Re-running the decorator is safe.

Usage:

    python labeling/_decorate_skill_records.py \\
        --root labeling/skill_bank_out/run_<timestamp> \\
        --intentions_run run_dualaxis_<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# When invoked as a script (`python labeling/_decorate_skill_records.py`)
# the parent of `labeling/` (the repo root) is not on sys.path. Add it so
# `labeling._protocol_lift` resolves regardless of how the file is run.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from labeling._protocol_lift import (  # noqa: E402  (after sys.path edit)
    GameSchemaIndex,
    LiftStats,
    build_schema_index_for_game,
    lift_protocol_to_typed_hops,
)

# ---------------------------------------------------------------------------
# Constants — match `common/enums.py` exactly so the decorated output is
# load-able by `lifecycle.ingest_draft` without translation.
# ---------------------------------------------------------------------------
DECORATOR_VERSION = "skillrecord_shape_v2"  # v2 adds feasible_tasks / verified_tasks (§22)

SOURCE_TYPE_MINED = "mined_from_trace"
STATUS_DRAFT = "draft"
APPLICABLE_DOMAIN_GAME = ["gymv"]   # both gym_v and env_wrappers are game-foundry

# Map dual-axis INTENT_OPERATOR → EVIDENCE_ROLE.
#
#   EVIDENCE_ROLES = (GATHER, VERIFY, REASON, COMMIT)
#
# RECOVER is a corrective action (still a COMMIT-style write to the env);
# COMPARE involves cross-step reasoning. INSPECT/TRACK both gather signal.
_OPERATOR_TO_EVIDENCE_ROLE: Dict[str, str] = {
    "INSPECT": "GATHER",
    "TRACK":   "GATHER",
    "COMPARE": "REASON",
    "COMMIT":  "COMMIT",
    "VERIFY":  "VERIFY",
    "RECOVER": "COMMIT",
}
_DEFAULT_EVIDENCE_ROLE = "COMMIT"


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _evidence_role_from_skill(skill: Dict[str, Any]) -> str:
    """Pull the operator out of skill.tags / expected_tag_pattern and
    map it to the four-value EVIDENCE_ROLES taxonomy."""
    candidates: List[str] = []
    for key in ("tags", "expected_tag_pattern"):
        v = skill.get(key)
        if isinstance(v, list):
            candidates.extend(str(x).upper() for x in v if x)
        elif isinstance(v, str) and v:
            candidates.append(v.upper())

    skill_id = str(skill.get("skill_id") or "").upper()
    if skill_id:
        # gym-v skills have ids like "mid:NAVIGATE" or "skill_<game>_<tag>_<idx>"
        for op in _OPERATOR_TO_EVIDENCE_ROLE:
            if op in skill_id:
                candidates.append(op)
                break

    for c in candidates:
        if c in _OPERATOR_TO_EVIDENCE_ROLE:
            return _OPERATOR_TO_EVIDENCE_ROLE[c]
    return _DEFAULT_EVIDENCE_ROLE


def _decorate_skill_dict(
    entry: Dict[str, Any],
    *,
    corpus: str,
    source_name: str,
    schema_index: Optional[GameSchemaIndex] = None,
    lift_stats: Optional[LiftStats] = None,
    force_relift: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """Add SkillRecord-shape fields to one skill dict. Returns (entry, mutated).

    Idempotency is keyed on the v1 fields (source_type / applicable_domains /
    status). v2 adds `feasible_tasks` / `verified_tasks` (harness/README §22)
    and back-fills them onto already-v1-decorated rows when they're missing —
    so re-running the v2 decorator over a v1 bank lifts it forward without
    re-touching anything else. Re-running the v2 decorator over a v2 bank is
    a no-op for the v1/v2 fields.

    The protocol lift (harness/README.md §21) is run when `schema_index` is
    provided. Idempotency on the lift is keyed on the hop shape:
    `protocol: List[Dict]` where every hop's `op` is in the §4.1 taxonomy ∪
    {"EXEC"}. Already-lifted skills are passed through unchanged.
    """
    skill = entry.get("skill") if isinstance(entry.get("skill"), dict) else entry
    if not isinstance(skill, dict):
        return entry, False

    fully_v1_decorated = (
        "source_type" in skill
        and "applicable_domains" in skill
        and "status" in skill
    )
    has_task_axis = "feasible_tasks" in skill and "verified_tasks" in skill

    mutated = False

    if not fully_v1_decorated:
        # First-time v1+v2 decoration on a fresh bank.
        skill["source_type"] = skill.get("source_type") or SOURCE_TYPE_MINED
        skill["applicable_domains"] = skill.get("applicable_domains") or list(
            APPLICABLE_DOMAIN_GAME
        )
        skill.setdefault("verified_domains", [])
        skill["evidence_role"] = skill.get("evidence_role") or _evidence_role_from_skill(skill)
        skill["status"] = skill.get("status") or STATUS_DRAFT
        mutated = True

    # v2: task axis. Defaults derive from the directory name (`source_name`),
    # which equals the cold-start game / env. `verified_tasks` always starts
    # empty — the gate Stage 3a transfer cycle is what populates it.
    if not has_task_axis or not skill.get("feasible_tasks"):
        skill["feasible_tasks"] = (
            list(skill.get("feasible_tasks") or []) or
            ([source_name] if source_name else [])
        )
        skill.setdefault("verified_tasks", [])
        mutated = True

    # v2 protocol lift: replace prose `steps` (or `_wrap_protocol_steps` shape-
    # lift output) with typed hops, and roll up `effects_add` / `effects_del`
    # onto the contract. Skipped silently if no schema_index is available
    # (caller passed `--skip-protocol-lift` or no `--actions_root`).
    if schema_index is not None:
        # Day-4: `--force_relift` restores `protocol` from the
        # preserved `protocol_raw` *before* the lift fires, so a
        # trigger-set or verb-table update sweeps the bank cleanly.
        # Old `eff_add` / `eff_del` are dropped on the same pass so the
        # contract roll-up reflects the v3 mining without union'ing in
        # stale predicates from the v1 pass.
        if force_relift and isinstance(skill.get("protocol_raw"), dict):
            skill["protocol"] = skill["protocol_raw"]
            contract = skill.get("contract")
            if isinstance(contract, dict):
                contract["eff_add"] = []
                contract["eff_del"] = []
        typed, contract_add, contract_del = lift_protocol_to_typed_hops(
            skill, schema_index=schema_index, stats=lift_stats,
        )
        if typed is not None:
            # Preserve the original prose dict under `protocol_raw` so the
            # diff is recoverable and downstream callers that prefer the
            # raw shape (e.g. `cold_start_labeling/build_skill_bank_gymv`)
            # can still find it.
            if isinstance(skill.get("protocol"), dict):
                skill["protocol_raw"] = skill["protocol"]
            skill["protocol"] = typed
            contract = skill.get("contract")
            if not isinstance(contract, dict):
                contract = {}
                skill["contract"] = contract
            # Don't clobber existing populated contract effects; merge.
            existing_add = list(contract.get("eff_add") or [])
            existing_del = list(contract.get("eff_del") or [])
            contract["eff_add"] = sorted(set(existing_add) | set(contract_add))
            contract["eff_del"] = sorted(set(existing_del) | set(contract_del))
            mutated = True

    if mutated:
        skill.setdefault("provenance", {})
        if isinstance(skill["provenance"], dict):
            skill["provenance"].setdefault("corpus", corpus)
            skill["provenance"].setdefault("source_name", source_name)
            skill["provenance"]["decorator_version"] = DECORATOR_VERSION

    if "skill" in entry and isinstance(entry["skill"], dict):
        entry["skill"] = skill
    else:
        entry = skill
    return entry, mutated


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def _decorate_skill_bank_jsonl(
    path: Path,
    *,
    corpus: str,
    source_name: str,
    schema_index: Optional[GameSchemaIndex] = None,
    lift_stats: Optional[LiftStats] = None,
    force_relift: bool = False,
) -> Tuple[int, int]:
    """Rewrite a skill_bank.jsonl in place, adding fields. Returns (n_rows, n_decorated)."""
    if not path.exists():
        return 0, 0
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    n_decorated = 0
    for row in rows:
        _, mutated = _decorate_skill_dict(
            row,
            corpus=corpus,
            source_name=source_name,
            schema_index=schema_index,
            lift_stats=lift_stats,
            force_relift=force_relift,
        )
        if mutated:
            n_decorated += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)
    return len(rows), n_decorated


def _decorate_skill_catalog_json(
    path: Path,
    *,
    corpus: str,
    source_name: str,
    schema_index: Optional[GameSchemaIndex] = None,
    lift_stats: Optional[LiftStats] = None,
    force_relift: bool = False,
) -> Tuple[int, int]:
    """Rewrite a skill_catalog.json in place. Returns (n_skills, n_decorated)."""
    if not path.exists():
        return 0, 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return 0, 0
    if not isinstance(doc, dict):
        return 0, 0

    skills = doc.get("skills") or []
    if not isinstance(skills, list):
        return 0, 0

    n_decorated = 0
    new_skills: List[Dict[str, Any]] = []
    for s in skills:
        if not isinstance(s, dict):
            new_skills.append(s)
            continue
        decorated, mutated = _decorate_skill_dict(
            s,
            corpus=corpus,
            source_name=source_name,
            schema_index=schema_index,
            lift_stats=lift_stats,
            force_relift=force_relift,
        )
        if mutated:
            n_decorated += 1
        new_skills.append(decorated)

    doc["skills"] = new_skills
    doc.setdefault("corpus", corpus)
    doc.setdefault("source_name", source_name)
    doc["decorator_version"] = DECORATOR_VERSION  # always bump on touch

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)
    return len(skills), n_decorated


def _write_lifecycle_meta(
    env_dir: Path,
    *,
    corpus: str,
    source_name: str,
    n_skills: int,
    intentions_run: Optional[str] = None,
    cold_start_run: Optional[str] = None,
    model: Optional[str] = None,
    lift_stats: Optional[LiftStats] = None,
    schema_index_size: Optional[int] = None,
) -> None:
    meta: Dict[str, Any] = {
        "decorator_version": DECORATOR_VERSION,
        "corpus": corpus,
        "source_name": source_name,
        "n_skills": n_skills,
        "applicable_domains": list(APPLICABLE_DOMAIN_GAME),
        "default_status": STATUS_DRAFT,
        "default_source_type": SOURCE_TYPE_MINED,
        "default_feasible_tasks": [source_name] if source_name else [],
        "default_verified_tasks": [],
        "intentions_run": intentions_run,
        "cold_start_run": cold_start_run,
        "model": model,
        "notes": (
            "These rows are SkillRecord-shape but un-gated. "
            "Use `skill_bank.lifecycle.ingest_draft` plus `gate_service.evaluate` "
            "with stages={STATIC, REPLAY} to import this bank into draft_store/. "
            "See PLAN-UNIFIED-SKILL-GATE §6 for the canonical import path. "
            "`feasible_tasks` / `verified_tasks` (v2 / harness/README §22) gate "
            "the EligibilityFilter F2′ task-axis veto: a skill is admitted on a "
            "step iff `state.task` segment ∈ feasible_tasks (or the list is "
            "empty, which means task-agnostic)."
        ),
    }
    if lift_stats is not None:
        meta["protocol_lift"] = lift_stats.to_json()
        meta["schema_index_entity_count"] = schema_index_size
    out = env_dir / "_lifecycle_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------

def decorate_corpus_tree(
    root: Path,
    *,
    corpus_name: str,
    intentions_run: Optional[str],
    cold_start_run: Optional[str],
    model: Optional[str],
    actions_root: Optional[Path] = None,
    force_relift: bool = False,
) -> Dict[str, Any]:
    """Walk root/<source_name>/ and decorate each env's outputs.

    ``root`` is e.g. ``labeling/skill_bank_out/run_<ts>/gym_v`` or
    ``.../env_wrappers``. Each immediate subfolder is one env / game.

    When ``actions_root`` is given (e.g. ``labeling/skill_actions_out/run_<ts>``),
    a per-game `GameSchemaIndex` is built from up to 3 cold-start episodes
    and the protocol lift runs in addition to the v1+v2 field decoration.
    Without it, the lift is skipped (callers can re-run later with the
    flag — the v1/v2 fields stay idempotent).
    """
    summary: Dict[str, Any] = {
        "corpus": corpus_name,
        "root": str(root),
        "envs": [],
        "totals": {
            "envs": 0,
            "skills_decorated": 0,
            "skills_total": 0,
            "lift_n_hops": 0,
            "lift_n_fallback_exec": 0,
        },
    }
    if not root.exists():
        return summary

    for env_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        bank_path = env_dir / "skill_bank.jsonl"
        cat_path = env_dir / "skill_catalog.json"

        if actions_root is not None:
            schema_index = build_schema_index_for_game(
                actions_root, corpus=corpus_name, game=env_dir.name,
            )
            lift_stats = LiftStats()
        else:
            schema_index = None
            lift_stats = None

        n_rows, n_dec_rows = _decorate_skill_bank_jsonl(
            bank_path,
            corpus=corpus_name,
            source_name=env_dir.name,
            schema_index=schema_index,
            lift_stats=lift_stats,
            force_relift=force_relift,
        )
        # The catalog re-runs the lift on the same skill bodies — share the
        # stats counter so we don't double-count.
        cat_lift_stats = LiftStats() if lift_stats is not None else None
        n_cat, n_dec_cat = _decorate_skill_catalog_json(
            cat_path,
            corpus=corpus_name,
            source_name=env_dir.name,
            schema_index=schema_index,
            lift_stats=cat_lift_stats,
            force_relift=force_relift,
        )
        n_skills = max(n_rows, n_cat)
        schema_size = (
            len(schema_index.entity_labels) if schema_index is not None else None
        )
        _write_lifecycle_meta(
            env_dir,
            corpus=corpus_name,
            source_name=env_dir.name,
            n_skills=n_skills,
            intentions_run=intentions_run,
            cold_start_run=cold_start_run,
            model=model,
            lift_stats=lift_stats,
            schema_index_size=schema_size,
        )
        env_summary: Dict[str, Any] = {
            "source_name": env_dir.name,
            "skill_bank_rows": n_rows,
            "skill_bank_rows_decorated": n_dec_rows,
            "skill_catalog_rows": n_cat,
            "skill_catalog_rows_decorated": n_dec_cat,
        }
        if lift_stats is not None:
            env_summary["lift"] = lift_stats.to_json()
            summary["totals"]["lift_n_hops"] += lift_stats.n_hops
            summary["totals"]["lift_n_fallback_exec"] += lift_stats.n_fallback_exec
        summary["envs"].append(env_summary)
        summary["totals"]["envs"] += 1
        summary["totals"]["skills_total"] += n_skills
        summary["totals"]["skills_decorated"] += max(n_dec_rows, n_dec_cat)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Decorate skill-discovery outputs with SkillRecord-shape fields.",
    )
    p.add_argument(
        "--root", required=True, type=str,
        help="Root containing gym_v/ and/or env_wrappers/ subfolders.",
    )
    p.add_argument(
        "--actions_root", type=str, default=None,
        help=(
            "Optional path to the matching `labeling/skill_actions_out/run_<ts>` "
            "tree. When given, the per-game schema_canonical vocabulary is "
            "mined and the protocol lift (harness/README §21) runs alongside "
            "the v1/v2 field decoration. Without it, the lift is skipped — "
            "v1/v2 fields still land. Re-running with the flag later is safe "
            "(the lift is idempotent on already-lifted rows)."
        ),
    )
    p.add_argument(
        "--skip_protocol_lift", action="store_true",
        help="Force-skip the protocol lift even if --actions_root is given.",
    )
    p.add_argument(
        "--force_relift", action="store_true",
        help=(
            "Re-run the protocol lift even on rows whose `protocol` is "
            "already a list of typed hops. Restores the original prose "
            "from `protocol_raw` (kept by an earlier lift run) before "
            "re-lifting — so trigger-set updates and verb-table "
            "extensions sweep the bank without manual surgery. No-op on "
            "rows that have no `protocol_raw`."
        ),
    )
    p.add_argument("--intentions_run", type=str, default=None)
    p.add_argument("--cold_start_run", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        return 2

    actions_root: Optional[Path] = None
    if args.actions_root and not args.skip_protocol_lift:
        actions_root = Path(args.actions_root).resolve()
        if not actions_root.exists():
            print(
                f"[WARN] --actions_root {actions_root} not found; "
                "protocol lift will be skipped.",
                file=sys.stderr,
            )
            actions_root = None

    print("=" * 62)
    print(f"  decorate_skill_records  ({DECORATOR_VERSION})")
    print("=" * 62)
    print(f"  Root            : {root}")
    print(f"  Actions root    : {actions_root or '(none — protocol lift skipped)'}")
    print(f"  Intentions run  : {args.intentions_run}")
    print(f"  Cold-start run  : {args.cold_start_run}")
    print(f"  Model           : {args.model}")
    print()

    overall: Dict[str, Any] = {"corpora": []}
    for corpus in ("gym_v", "env_wrappers"):
        corpus_root = root / corpus
        if not corpus_root.exists():
            continue
        print(f"  -- {corpus} --")
        s = decorate_corpus_tree(
            corpus_root,
            corpus_name=corpus,
            intentions_run=args.intentions_run,
            cold_start_run=args.cold_start_run,
            model=args.model,
            actions_root=actions_root,
            force_relift=args.force_relift,
        )
        for env in s["envs"]:
            line = (
                f"    {env['source_name']:34s} "
                f"jsonl={env['skill_bank_rows']:>3d} "
                f"(decorated={env['skill_bank_rows_decorated']:>3d}) "
                f"catalog={env['skill_catalog_rows']:>3d} "
                f"(decorated={env['skill_catalog_rows_decorated']:>3d})"
            )
            lift = env.get("lift")
            if lift:
                line += (
                    f"  hops={lift['n_hops']:>3d} "
                    f"exec={lift['n_fallback_exec']:>2d}/{lift['n_hops']:>2d} "
                    f"({100 * lift['fallback_exec_pct']:.1f}%)"
                )
            print(line)
        totals = s["totals"]
        if totals["lift_n_hops"]:
            pct = 100 * totals["lift_n_fallback_exec"] / totals["lift_n_hops"]
            lift_summary = (
                f"  lift_hops={totals['lift_n_hops']}  "
                f"fallback_exec={totals['lift_n_fallback_exec']} ({pct:.1f}%)"
            )
        else:
            lift_summary = ""
        print(
            f"    -> total envs={totals['envs']}  "
            f"skills={totals['skills_total']}  "
            f"decorated={totals['skills_decorated']}{lift_summary}"
        )
        print()
        overall["corpora"].append(s)

    out_summary = root / "_decorator_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Wrote summary: {out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
