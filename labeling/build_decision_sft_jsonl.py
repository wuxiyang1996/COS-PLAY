#!/usr/bin/env python
"""
Convert ``labeling/skill_actions_out/<run>/<corpus>/<game>/episode_*.json``
into the per-game ``{skill_selection,action_taking}.jsonl`` files that
``trainer/SFT/data_loader.py`` already knows how to upgrade.

The trainer's :func:`_align_action_taking_to_coevolution` is *designed*
to take simpler legacy rows and inject the missing co-evolution pieces
(SUBGOAL line, urgency, full skill protocol block) at training time, so
this converter stays intentionally minimal:

* For ``action_taking`` we emit the old-format prompt
  (``SYSTEM_PROMPT + Game state + numbered actions + closing``) plus
  the metadata fields the loader's alignment helpers consult
  (``intention``, ``active_skill``, ``valid_actions``, ``image``).
  ``_align_action_taking_to_coevolution`` then folds in
  ``SUBGOAL: …`` + protocol-shaped Active Skill block.

* For ``skill_selection`` we emit
  ``SKILL_SELECTION_SYSTEM_PROMPT + Game state + Available strategies
  (top-k candidates from skill_query) + closing``.
  The loader's ``_enrich_skill_selection_prompt`` rewrites the bare-id
  list into rich descriptions when a bank is supplied.

Dual-axis projection
--------------------
Our intentions are ``[OPERATOR/SUBGOAL] note`` (e.g.
``[COMMIT/ATTACK] strike the orc``) but the trainer's
``SUBGOAL_TAGS`` are still single-axis.  We project per-row::

    intention_field = f"[{intention_subgoal}] {intention_note}"

so the trainer sees the legacy ``[TAG] phrase`` shape.  The original
composite tag stays in ``extras.intention_full`` for any future
trainer-side dual-axis upgrade.

Layout
------
Output goes to ``labeling/decision_sft_jsonl/run_<ts>/<game>/{skill_selection,action_taking}.jsonl``.
Per-corpus / per-game stats land in ``_run_summary.json``.

Game names from the two corpora do not collide
(``Temporal_*-v0`` vs ``tetris/super_mario/...``) so a single flat
``<game>/`` directory is safe — the trainer reads
``<root>/<game>/<adapter>.jsonl`` directly.

CLI::

    # Default: read latest skill_actions_out run, write timestamped output.
    python labeling/build_decision_sft_jsonl.py

    # Pin a specific input run + output dir.
    python labeling/build_decision_sft_jsonl.py \\
        --skill-actions-run labeling/skill_actions_out/run_20260430_064325 \\
        --output-dir         labeling/decision_sft_jsonl/run_my

    # Smoke (1 episode per game).
    python labeling/build_decision_sft_jsonl.py --limit-episodes 1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = CODEBASE_ROOT / "labeling" / "skill_actions_out"
DEFAULT_OUTPUT_ROOT = CODEBASE_ROOT / "labeling" / "decision_sft_jsonl"

CORPORA = ("gym_v", "env_wrappers")

# Trainer's single-axis subgoal vocabulary (kept in sync with
# ``trainer/SFT/data_loader.py:SUBGOAL_TAGS``).  Any composite-tag
# subgoal not in this list is mapped to ``EXECUTE`` so the legacy
# alignment path still validates.
SUBGOAL_TAGS = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)
_SUBGOAL_TAG_SET = frozenset(SUBGOAL_TAGS)


# ---------------------------------------------------------------------------
# Static prompt templates (mirror trainer/coevolution/episode_runner.py
# verbatim — the *legacy/simple* shape the loader's alignment helpers
# upgrade at training time)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n"
)

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are an expert game strategist. "
    "Given the current game state and candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits>\n"
    "SKILL: <number>\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_intention(step: Dict[str, Any]) -> str:
    """Project the dual-axis intention to a single-tag ``[SUBGOAL] note``.

    Falls back gracefully when only the legacy single-tag form is present.
    """
    sg = (step.get("intention_subgoal") or "").strip().upper()
    note = (step.get("intention_note") or "").strip()
    if sg and sg in _SUBGOAL_TAG_SET and note:
        return f"[{sg}] {note}"
    if sg and note:
        return f"[EXECUTE] {note}"

    raw = (step.get("intentions") or "").strip()
    if raw.startswith("["):
        return raw
    return "[EXECUTE] act in the game"


def _format_numbered_actions(actions: List[str]) -> str:
    """``1. A\\n2. B\\n…``"""
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))


def _action_index_1based(action: str, valid_actions: List[str]) -> int:
    """Return 1-based index of *action* in *valid_actions*, or 1 on miss."""
    for i, a in enumerate(valid_actions):
        if a == action:
            return i + 1
    # Loose match (case-insensitive, strip)
    target = (action or "").strip().lower()
    for i, a in enumerate(valid_actions):
        if (a or "").strip().lower() == target:
            return i + 1
    return 1


def _resolve_image(step: Dict[str, Any], episode_root: Path) -> Optional[Dict[str, Any]]:
    """Best-effort frame path resolution from the step's metadata."""
    meta = step.get("metadata") or {}
    fp = meta.get("frame_path") or step.get("frame_path")
    if not fp:
        return None
    p = Path(fp)
    if not p.is_absolute():
        # gymv frames live at ``<corpus_root>/<env>/frames/ep_NNN/<step>.png``
        # but step's metadata typically stores a relative path under that.
        guess = episode_root / fp
        if guess.exists():
            p = guess
    if not p.exists():
        return None
    return {"path": str(p), "mime_type": "image/png"}


def _extract_schema_text(step: Dict[str, Any]) -> str:
    """Pull the rich ``<state>…</state>`` schema text for the actor prompt.

    The cold-start rollouts store the gpt-5.4 / gpt-5.5 schema in
    ``step.metadata.schema``.  ``summary_state`` is mostly ``None`` in the
    SFT corpus (the labeling script never repopulated it after schema
    generation), so the schema field is the canonical source.  We fall
    back to the short ``state``/``raw_state`` line for the rare step
    that has no schema (schema_error rows from cold-start retries).
    """
    meta = step.get("metadata") or {}
    schema = meta.get("schema")
    if isinstance(schema, str) and schema.strip():
        return schema.strip()
    summary = step.get("summary_state")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    state = step.get("state") or step.get("raw_state") or ""
    return str(state).strip()


# ---------------------------------------------------------------------------
# Per-step row builders
# ---------------------------------------------------------------------------

def build_action_taking_row(
    step: Dict[str, Any],
    *,
    game: str,
    episode_id: str,
    step_idx: int,
    image: Optional[Dict[str, Any]] = None,
    corpus: str = "",
) -> Optional[Dict[str, Any]]:
    """Emit one ``action_taking`` JSONL row for a step.

    Returns ``None`` when the step lacks the minimum required fields.
    """
    schema_text = _extract_schema_text(step)
    valid_actions = step.get("available_actions") or (
        (step.get("metadata") or {}).get("valid_actions") or []
    )
    action = step.get("action")
    if not schema_text or not valid_actions or not action:
        return None

    intention = _project_intention(step)

    user = (
        f"Game state:\n\n{schema_text}\n\n"
        f"Available actions (pick ONE by number):\n"
        f"{_format_numbered_actions(valid_actions)}\n\n"
        f"Choose the best action. Output REASONING then ACTION number."
    )
    prompt = SYSTEM_PROMPT + "\n" + user

    action_num = _action_index_1based(action, valid_actions)
    note = (step.get("intention_note") or "").strip() or "Expert play."
    completion = f"REASONING: {note[:200]}\nACTION: {action_num}"

    skills = step.get("skills") or {}
    active_skill = skills.get("skill_id") or ""

    extras: Dict[str, Any] = {
        "game": game,
        "corpus": corpus,
        "episode_id": episode_id,
        "step_idx": step_idx,
        "valid_actions": list(valid_actions),
        "intention_full": (step.get("intentions") or "").strip(),
        "intention_operator": step.get("intention_tag") or "",
        "intention_subgoal": step.get("intention_subgoal") or "",
        "reward": step.get("reward"),
    }
    if skills:
        extras["skill_execution_hint"] = skills.get("execution_hint") or ""
        extras["skill_pass_rate"] = skills.get("pass_rate")
        extras["skill_n_instances"] = skills.get("n_instances")

    row: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "intention": intention,
        "active_skill": active_skill,
    }
    if image is not None:
        row["image"] = image
    row.update({k: v for k, v in extras.items() if v is not None})
    return row


def build_skill_selection_row(
    step: Dict[str, Any],
    *,
    game: str,
    episode_id: str,
    step_idx: int,
    image: Optional[Dict[str, Any]] = None,
    corpus: str = "",
) -> Optional[Dict[str, Any]]:
    """Emit one ``skill_selection`` JSONL row for a step.

    Built from the top-k candidates already on disk in
    ``step['skill_query']['candidates']``.  The "correct" choice is the
    candidate whose ``skill_id`` matches ``skills.skill_id`` (the top-1
    pick the bank engine itself made at labeling time).

    Returns ``None`` when fewer than 2 candidates are present (a
    single-candidate decision is degenerate; not useful for SFT).
    """
    sq = step.get("skill_query") or {}
    candidates = sq.get("candidates") or []
    if len(candidates) < 2:
        return None

    skill_ids = [c.get("skill_id") for c in candidates if c.get("skill_id")]
    if len(skill_ids) < 2:
        return None

    # Selected skill — prefer the materialised skills.skill_id, fall back
    # to skill_query.selected_skill_id, fall back to first candidate.
    selected = (step.get("skills") or {}).get("skill_id") or sq.get("selected_skill_id") or skill_ids[0]
    if selected not in skill_ids:
        return None
    sel_idx = skill_ids.index(selected) + 1

    schema_text = _extract_schema_text(step)
    if not schema_text:
        return None

    numbered_skills = "\n".join(
        f"  {i + 1}. {sid}" for i, sid in enumerate(skill_ids)
    )

    user = (
        f"Game state:\n{schema_text}\n\n"
        f"Available strategies (pick ONE by number):\n{numbered_skills}\n\n"
        f"Choose the best strategy. Output REASONING then SKILL number."
    )
    prompt = SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user

    why = (step.get("skills") or {}).get("why_selected") or "Best fit for current state."
    completion = f"REASONING: {why[:200]}\nSKILL: {sel_idx}"

    extras: Dict[str, Any] = {
        "game": game,
        "corpus": corpus,
        "episode_id": episode_id,
        "step_idx": step_idx,
        "candidates": skill_ids,
        "selected_skill_id": selected,
    }
    intention = _project_intention(step)
    row: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "intention": intention,
        "active_skill": selected,
    }
    if image is not None:
        row["image"] = image
    row.update({k: v for k, v in extras.items() if v is not None})
    return row


# ---------------------------------------------------------------------------
# Episode / game / run drivers
# ---------------------------------------------------------------------------

def process_episode(
    ep_path: Path,
    *,
    game: str,
    corpus: str,
    out_action: Path,
    out_skill: Path,
) -> Dict[str, int]:
    """Stream rows from one episode file to the per-adapter JSONLs."""
    with ep_path.open("r") as f:
        ep = json.load(f)
    episode_id = str(ep.get("episode_id") or ep_path.stem)
    exps = ep.get("experiences") or ep.get("steps") or []

    n_action, n_skill, n_skipped = 0, 0, 0
    image_root = ep_path.parent  # frames live next to the episode JSON
    with out_action.open("a") as fa, out_skill.open("a") as fs:
        for i, step in enumerate(exps):
            image = _resolve_image(step, image_root)
            ar = build_action_taking_row(
                step, game=game, episode_id=episode_id,
                step_idx=i, image=image, corpus=corpus,
            )
            if ar:
                fa.write(json.dumps(ar) + "\n")
                n_action += 1
            else:
                n_skipped += 1
            sr = build_skill_selection_row(
                step, game=game, episode_id=episode_id,
                step_idx=i, image=image, corpus=corpus,
            )
            if sr:
                fs.write(json.dumps(sr) + "\n")
                n_skill += 1
    return {
        "n_steps": len(exps),
        "n_action_taking": n_action,
        "n_skill_selection": n_skill,
        "n_skipped_action": n_skipped,
    }


def process_game(
    game_dir: Path,
    *,
    corpus: str,
    output_root: Path,
    limit_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert every episode in one ``<corpus>/<game>/`` dir."""
    game = game_dir.name
    out_dir = output_root / game
    out_dir.mkdir(parents=True, exist_ok=True)
    out_action = out_dir / "action_taking.jsonl"
    out_skill = out_dir / "skill_selection.jsonl"

    # Truncate (idempotent re-runs).
    out_action.write_text("")
    out_skill.write_text("")

    episode_files = sorted(game_dir.glob("episode_*.json"))
    if limit_episodes is not None:
        episode_files = episode_files[:limit_episodes]

    agg = {
        "corpus": corpus,
        "game": game,
        "n_episodes": 0,
        "n_steps": 0,
        "n_action_taking": 0,
        "n_skill_selection": 0,
    }
    for ep_path in episode_files:
        try:
            stats = process_episode(
                ep_path, game=game, corpus=corpus,
                out_action=out_action, out_skill=out_skill,
            )
        except Exception as exc:
            print(f"[build_decision_sft] {corpus}/{game}/{ep_path.name}: ERROR {exc}")
            continue
        agg["n_episodes"] += 1
        agg["n_steps"] += stats["n_steps"]
        agg["n_action_taking"] += stats["n_action_taking"]
        agg["n_skill_selection"] += stats["n_skill_selection"]

    print(
        f"[build_decision_sft] {corpus}/{game}: "
        f"eps={agg['n_episodes']} steps={agg['n_steps']} "
        f"action={agg['n_action_taking']} skill={agg['n_skill_selection']}"
    )
    return agg


def discover_games(input_root: Path) -> List[Tuple[str, Path]]:
    """Walk ``input_root/<corpus>/<game>/`` and return ``[(corpus, game_dir), …]``."""
    pairs: List[Tuple[str, Path]] = []
    for corpus in CORPORA:
        cdir = input_root / corpus
        if not cdir.is_dir():
            continue
        for gd in sorted(cdir.iterdir()):
            if gd.is_dir() and any(gd.glob("episode_*.json")):
                pairs.append((corpus, gd))
    return pairs


def latest_run(parent: Path) -> Optional[Path]:
    """Return the most-recently-modified ``run_*`` directory under *parent*."""
    if not parent.is_dir():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument(
        "--skill-actions-run", type=Path, default=None,
        help=(
            "Input run dir under labeling/skill_actions_out (default: latest). "
            "Layout: <root>/{gym_v,env_wrappers}/<game>/episode_*.json"
        ),
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output dir; defaults to "
            "labeling/decision_sft_jsonl/run_<utc-timestamp>."
        ),
    )
    ap.add_argument(
        "--limit-episodes", type=int, default=None,
        help="Process only the first N episodes per game (smoke).",
    )
    ap.add_argument(
        "--corpus", choices=CORPORA, default=None,
        help="Restrict to one corpus.",
    )
    ap.add_argument(
        "--games", nargs="+", default=None,
        help="Restrict to a subset of game names.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    in_run: Optional[Path] = args.skill_actions_run
    if in_run is None:
        in_run = latest_run(DEFAULT_INPUT_ROOT)
    if in_run is None or not in_run.is_dir():
        print(
            f"[build_decision_sft] no skill_actions_out run found "
            f"(input={in_run})"
        )
        return 2

    if args.output_dir is not None:
        out_root = args.output_dir
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = DEFAULT_OUTPUT_ROOT / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_games(in_run)
    if args.corpus:
        pairs = [(c, gd) for (c, gd) in pairs if c == args.corpus]
    if args.games:
        keep = set(args.games)
        pairs = [(c, gd) for (c, gd) in pairs if gd.name in keep]
    if not pairs:
        print(f"[build_decision_sft] no (corpus, game) pairs under {in_run}")
        return 2

    print(
        f"[build_decision_sft] input run : {in_run}\n"
        f"[build_decision_sft] output    : {out_root}\n"
        f"[build_decision_sft] pairs     : {len(pairs)}\n"
    )

    summaries: List[Dict[str, Any]] = []
    for corpus, gd in pairs:
        s = process_game(
            gd, corpus=corpus, output_root=out_root,
            limit_episodes=args.limit_episodes,
        )
        summaries.append(s)

    totals = {
        "input_run": str(in_run),
        "output_root": str(out_root),
        "n_pairs": len(summaries),
        "n_episodes": sum(s["n_episodes"] for s in summaries),
        "n_steps": sum(s["n_steps"] for s in summaries),
        "n_action_taking_rows": sum(s["n_action_taking"] for s in summaries),
        "n_skill_selection_rows": sum(s["n_skill_selection"] for s in summaries),
        "by_corpus": {},
        "per_game": summaries,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    by_corpus: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"games": 0, "episodes": 0, "steps": 0,
                 "action_taking": 0, "skill_selection": 0}
    )
    for s in summaries:
        b = by_corpus[s["corpus"]]
        b["games"] += 1
        b["episodes"] += s["n_episodes"]
        b["steps"] += s["n_steps"]
        b["action_taking"] += s["n_action_taking"]
        b["skill_selection"] += s["n_skill_selection"]
    totals["by_corpus"] = dict(by_corpus)

    summary_path = out_root / "_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(totals, f, indent=2)
    print(
        f"\n[build_decision_sft] TOTALS: "
        f"games={totals['n_pairs']} eps={totals['n_episodes']} "
        f"steps={totals['n_steps']} "
        f"action_rows={totals['n_action_taking_rows']} "
        f"skill_rows={totals['n_skill_selection_rows']}"
    )
    print(f"[build_decision_sft] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
