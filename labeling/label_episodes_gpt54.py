#!/usr/bin/env python
"""
Label cold-start episode trajectories using GPT-5.4.

Reads existing episode JSON files and generates concise per-step annotations:

  - summary_state : compact key=value facts (deterministic, no LLM)
        e.g. game=tetris | phase=midgame | step=50/86 | stack_h=14 | holes=32
  - summary       : summary_state + LLM strategic note (delta-aware)
        e.g. ...holes=32 | note=Sharp hole spike; avoid covering center wells.
  - intentions    : [TAG] subgoal phrase via LLM (delta + urgency aware)
        e.g. [CLEAR] Place S flat to reduce holes and set up line clears
  - skills        : renamed from sub_tasks (null — populated downstream)

The labeling pipeline uses game-aware deterministic extractors from
``decision_agents.agent_helper`` (build_rag_summary, extract_game_facts)
and adds lightweight LLM enrichment (~25 tokens for note, ~40 for intention).

Key mechanisms for step-to-step differentiation:
  - State delta: ``_compute_state_delta`` feeds the LLM a compact diff
    (e.g. ``holes:10->32``) so notes and tags reflect what changed.
  - Urgency detection: ``_detect_urgency`` triggers on absolute-value
    thresholds (e.g. Tetris holes>25) so tags shift to [SURVIVE]/[CLEAR]
    even when per-step deltas are small.

Output structure (labeling/output/gpt54/<game_name>/):
  - episode_NNN.json        Labeled episode (original + new fields)
  - labeling_summary.json   Run statistics

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # Label all episodes for all games in the default cold-start output
    python labeling/label_episodes_gpt54.py

    # Label specific game(s)
    python labeling/label_episodes_gpt54.py --games tetris candy_crush

    # Label a single file
    python labeling/label_episodes_gpt54.py --input_file cold_start/output/gpt54/tetris/episode_000.json

    # Dry run (preview first episode without saving)
    python labeling/label_episodes_gpt54.py --dry_run --games tetris --max_episodes 1

    # Process exactly one rollout per game (quick test across all games)
    python labeling/label_episodes_gpt54.py --one_per_game -v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------
from decision_agents.agent_helper import (
    compact_text_observation,
    get_state_summary,
    infer_intention,
    strip_think_tags,
    build_rag_summary,
    extract_game_facts,
    HARD_SUMMARY_CHAR_LIMIT,
    SUBGOAL_TAGS,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

try:
    import openai
    from api_keys import openai_api_key, open_router_api_key
except (ImportError, AttributeError):
    openai = None
    openai_api_key = None
    open_router_api_key = None

try:
    from API_func import OPENROUTER_BASE
except ImportError:
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"

_OUTPUT_ROOT = CODEBASE_ROOT / "cold_start" / "output"
DEFAULT_INPUT_DIRS: List[Path] = [
    _OUTPUT_ROOT / "gpt54",
    _OUTPUT_ROOT / "gpt54_evolver",
    _OUTPUT_ROOT / "gpt54_orak",
    _OUTPUT_ROOT / "gpt54_sokoban",
]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "gpt54"

SUMMARY_CHAR_BUDGET = 200
SUMMARY_PROSE_WORD_BUDGET = 30
INTENTION_WORD_BUDGET = 15

_SUBGOAL_TAG_SET = frozenset(SUBGOAL_TAGS)

_TAG_ALIASES: Dict[str, str] = {
    "PLACE": "SETUP", "DROP": "EXECUTE", "MOVE": "NAVIGATE",
    "SWAP": "EXECUTE", "PUSH": "NAVIGATE", "JUMP": "NAVIGATE",
    "MATCH": "CLEAR", "PLAN": "SETUP", "ARRANGE": "SETUP",
    "ROTATE": "SETUP", "ORGANIZE": "OPTIMIZE", "SCORE": "EXECUTE",
    "PROTECT": "DEFEND", "GRAB": "COLLECT", "FLEE": "SURVIVE",
    "RUN": "NAVIGATE", "CREATE": "BUILD", "FIND": "EXPLORE",
    "FIX": "OPTIMIZE", "ALIGN": "POSITION", "TARGET": "ATTACK",
    "SECURE": "DEFEND", "EXPAND": "ATTACK", "RETREAT": "DEFEND",
}

_TAG_RE = re.compile(r"\[(\w+)\]\s*")


def _normalize_intention(raw: str) -> str:
    """Ensure intention has a valid ``[TAG] phrase`` format."""
    raw = raw.split("\n")[0].strip().strip('"').strip("'")
    if not raw.startswith("["):
        return f"[EXECUTE] {raw}"
    m = _TAG_RE.match(raw)
    if not m:
        return f"[EXECUTE] {raw}"
    tag = m.group(1).upper()
    rest = raw[m.end():].strip()
    if tag not in _SUBGOAL_TAG_SET:
        tag = _TAG_ALIASES.get(tag, "EXECUTE")
    return f"[{tag}] {rest}" if rest else f"[{tag}]"


def _compute_state_delta(prev_ss: str, curr_ss: str) -> str:
    """Return a compact string of key changes between two ``summary_state`` values.

    Example return: ``stack_h:0->8, holes:0->10, piece:L->S``
    Only value-bearing fields that actually changed are included; meta keys
    like ``game``, ``step``, ``phase`` are skipped.
    """
    if not prev_ss or not curr_ss:
        return ""

    def _parse(ss: str) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for seg in ss.split(" | "):
            if "=" in seg:
                k, v = seg.split("=", 1)
                d[k.strip()] = v.strip()
        return d

    skip = {"game", "step", "phase"}
    p, c = _parse(prev_ss), _parse(curr_ss)
    changes: List[str] = []
    for k, v in c.items():
        if k in skip:
            continue
        pv = p.get(k)
        if pv is not None and pv != v:
            changes.append(f"{k}:{pv}->{v}")
    return ", ".join(changes[:5])


def _detect_urgency(summary_state: str, game_name: str) -> str:
    """Return a short urgency warning when absolute state values are critical.

    Complements ``_compute_state_delta`` (relative change) by catching
    situations where per-step deltas are small but the cumulative state is
    dangerous (e.g. Tetris with 40 holes and stack_h=16).
    """

    def _val(key: str) -> Optional[float]:
        for seg in summary_state.split(" | "):
            seg = seg.strip()
            if seg.startswith(f"{key}="):
                try:
                    return float(seg.split("=", 1)[1].split(",")[0])
                except (ValueError, IndexError):
                    return None
        return None

    gn = game_name.lower()
    warnings: List[str] = []

    if gn == "tetris":
        h = _val("holes")
        sh = _val("stack_h")
        if h is not None and h > 25:
            warnings.append("severe holes—prioritise CLEAR or SURVIVE")
        if sh is not None and sh > 14:
            warnings.append("stack near ceiling—SURVIVE")
    elif gn in ("2048", "twenty_forty_eight"):
        e = _val("empty")
        if e is not None and e < 3:
            warnings.append("board nearly full—must MERGE now")
    elif "candy" in gn:
        m = _val("moves")
        if m is not None and m < 5:
            warnings.append("very few moves left—maximise every action")
    elif "mario" in gn:
        t = _val("time")
        if t is not None and t < 50:
            warnings.append("time running out—NAVIGATE quickly")

    return "; ".join(warnings)


# ---------------------------------------------------------------------------
# GPT-5.4 chat helper (function-calling style for structured output)
# ---------------------------------------------------------------------------

def _ask_gpt54(
    prompt: str,
    *,
    system: str = "",
    model: str = MODEL_GPT54,
    temperature: float = 0.2,
    max_tokens: int = 150,
) -> Optional[str]:
    """Send a prompt to GPT-5.4 via ask_model and return the cleaned reply."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    if ask_model is not None:
        result = ask_model(full_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        if result and not result.startswith("Error"):
            return strip_think_tags(result).strip()

    return None


# ---------------------------------------------------------------------------
# Labeling functions — wrappers around decision_agents helpers + GPT-5.4
# ---------------------------------------------------------------------------

def generate_summary_state(
    state: str,
    game_name: str = "",
    step_idx: int = -1,
    total_steps: int = -1,
    reward: float = 0.0,
) -> str:
    """Produce a compact ``key=value`` state summary for RAG retrieval.

    Fully deterministic (no LLM call).  Uses game-aware fact extraction
    so the output matches the online agent's structured format and embeds
    well for cosine-similarity retrieval.
    """
    return build_rag_summary(
        state,
        game_name,
        step_idx=step_idx,
        total_steps=total_steps,
        reward=reward,
    )


def generate_intention(
    state: str,
    action: str,
    original_reasoning: Optional[str] = None,
    game_name: str = "",
    summary_state: str = "",
    prev_intention: str = "",
    prev_summary_state: str = "",
    model: str = MODEL_GPT54,
) -> str:
    """Produce a ``[TAG] subgoal phrase`` grounded in extracted state facts.

    Format: ``[TAG] short phrase`` (max ~15 words).

    ``summary_state`` anchors the prompt in concrete facts.
    ``prev_summary_state`` feeds a delta so the LLM sees what changed and can
    decide whether the subgoal should shift.  ``prev_intention`` provides the
    baseline — the tag **should** change when the delta is large enough to
    indicate a genuine strategic shift.
    """
    tags_str = "|".join(SUBGOAL_TAGS)
    facts_line = f"Facts: {summary_state}\n" if summary_state else ""
    delta = _compute_state_delta(prev_summary_state, summary_state)
    delta_line = f"Changed: {delta}\n" if delta else ""
    urgency = _detect_urgency(summary_state, game_name)
    urgency_line = f"URGENCY: {urgency}\n" if urgency else ""
    prev_line = f"Previous subgoal: {prev_intention}\n" if prev_intention else ""
    shift_hint = (
        "IMPORTANT: If the situation changed significantly or urgency is high, pick a NEW tag that matches the new priority.\n"
        if delta or urgency else ""
    )

    if original_reasoning and len(original_reasoning) > 20:
        prompt = (
            f"Condense to [TAG] subgoal phrase (max {INTENTION_WORD_BUDGET} words).\n"
            f"Tags: {tags_str}\n"
            f"{facts_line}"
            f"{delta_line}"
            f"{urgency_line}"
            f"{prev_line}"
            f"{shift_hint}"
            f"Reasoning: {original_reasoning[:300]}\n"
            f"Reply ONLY: [TAG] phrase\n"
            f"Examples:\n"
            f"  [CLEAR] Place S flat to reduce holes and set up line clears\n"
            f"  [SURVIVE] Place Z vertically to avoid overhangs and prevent topping out\n"
            f"  [MERGE] Merge left for space while keeping the 64 anchored\n"
            f"Subgoal:"
        )
    else:
        game_label = game_name.replace("_", " ") if game_name else "game"
        compact = compact_text_observation(state, max_chars=200)
        state_text = compact if compact else state[:800]
        prompt = (
            f"{game_label}. Action: {action}\n"
            f"State: {state_text}\n"
            f"{facts_line}"
            f"{delta_line}"
            f"{urgency_line}"
            f"{prev_line}"
            f"{shift_hint}"
            f"What subgoal? Reply ONLY: [TAG] phrase "
            f"(max {INTENTION_WORD_BUDGET} words)\n"
            f"Tags: {tags_str}\n"
            f"Examples:\n"
            f"  [SETUP] Consolidate tiles toward one side for an early merge\n"
            f"  [ATTACK] Swap for an immediate match and better cascade potential\n"
            f"  [SURVIVE] Merge vertical 2s now to create space and survive\n"
            f"Subgoal:"
        )

    result = _ask_gpt54(prompt, model=model, max_tokens=40)
    if result:
        return _normalize_intention(result)[:150]

    fallback = infer_intention(state, game=game_name, model=model)
    if fallback:
        return _normalize_intention(f"[EXECUTE] {fallback}")[:150]
    return f"[EXECUTE] {action}"


def generate_summary_prose(
    state: str,
    game_name: str = "",
    summary_state: str = "",
    prev_summary_state: str = "",
    model: str = MODEL_GPT54,
) -> str:
    """Produce ``summary = summary_state + | note=<strategic assessment>``.

    The deterministic ``summary_state`` facts are always present so key info
    is never lost across steps.  The LLM only adds a short strategic note
    (threat / opportunity, ~10 words) — very cheap output budget.

    ``prev_summary_state`` feeds a delta line (e.g. ``holes:10->32``) so the
    note focuses on what actually changed rather than repeating generic advice.
    """
    if not summary_state:
        summary_state = generate_summary_state(state, game_name)

    compact = compact_text_observation(state, max_chars=200)
    state_text = compact if compact else state[:1000]
    game_label = game_name.replace("_", " ") if game_name else "game"

    delta = _compute_state_delta(prev_summary_state, summary_state)
    delta_line = f"Changed since last step: {delta}\n" if delta else ""

    prompt = (
        f"{game_label}: {state_text}\n"
        f"{delta_line}"
        f"Key strategic note about the current threat or opportunity "
        f"(max 10 words, be specific to what changed).\n"
        f"Examples: \"Sharp hole spike; avoid covering center wells.\" / "
        f"\"No empty cells; next move must create one.\"\n"
        f"Note:"
    )

    note = _ask_gpt54(prompt, model=model, max_tokens=25)
    if note:
        note = note.split("\n")[0].strip().strip('"').strip("'")
        note = note[:80]
        return f"{summary_state} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT]
    return summary_state[:HARD_SUMMARY_CHAR_LIMIT]


# ---------------------------------------------------------------------------
# Per-experience labeling
# ---------------------------------------------------------------------------

def label_experience(
    exp: Dict[str, Any],
    game_name: str = "",
    model: str = MODEL_GPT54,
    delay: float = 0.0,
    step_idx: int = -1,
    total_steps: int = -1,
    prev_intention: str = "",
    prev_summary_state: str = "",
) -> Dict[str, Any]:
    """Label a single experience dict in-place and return it.

    Produces three complementary fields whose key facts are identical:

    * ``summary_state`` — compact ``key=value`` string (deterministic, no LLM)
      optimised for RAG embedding retrieval.
    * ``summary`` — ``summary_state | note=<strategic assessment>``; same
      facts plus a short LLM-generated note grounded in **what changed** since
      the previous step so every note is unique and specific.
    * ``intentions`` — ``[TAG] subgoal phrase`` via LLM, grounded in the
      ``summary_state`` facts and a state-delta from the previous step.
      The tag is encouraged to shift when the delta indicates a genuine
      strategic change (e.g. rising holes → ``[SURVIVE]``).
    """
    state = exp.get("state", "")
    action = exp.get("action", "")
    reward = exp.get("reward", 0.0)
    original_reasoning = exp.get("intentions")
    idx = exp.get("idx", step_idx)

    # --- summary_state (structured key=value, deterministic, 0 LLM calls) ---
    summary_state = generate_summary_state(
        state,
        game_name=game_name,
        step_idx=idx if isinstance(idx, int) else step_idx,
        total_steps=total_steps,
        reward=reward if isinstance(reward, (int, float)) else 0.0,
    )
    exp["summary_state"] = summary_state

    # --- summary (summary_state + delta-aware strategic note, 1 LLM call) ---
    summary = generate_summary_prose(
        state, game_name=game_name,
        summary_state=summary_state,
        prev_summary_state=prev_summary_state,
        model=model,
    )
    exp["summary"] = summary

    if delay > 0:
        time.sleep(delay)

    # --- intention ([TAG] subgoal, delta-aware + prev step, 1 LLM call) ---
    intention = generate_intention(
        state, action,
        original_reasoning=original_reasoning,
        game_name=game_name,
        summary_state=summary_state,
        prev_intention=prev_intention,
        prev_summary_state=prev_summary_state,
        model=model,
    )
    exp["intentions"] = intention

    # --- skills (renamed from sub_tasks, null for now) ---
    exp.pop("sub_tasks", None)
    exp["skills"] = None

    if delay > 0:
        time.sleep(delay)

    return exp


# ---------------------------------------------------------------------------
# Per-episode labeling
# ---------------------------------------------------------------------------

def label_episode(
    episode_data: Dict[str, Any],
    model: str = MODEL_GPT54,
    delay: float = 0.1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Label all experiences in an episode dict in-place."""
    game_name = episode_data.get("game_name", "")
    experiences = episode_data.get("experiences", [])
    n = len(experiences)

    prev_intention = ""
    prev_summary_state = ""
    for i, exp in enumerate(experiences):
        try:
            label_experience(
                exp, game_name=game_name, model=model, delay=delay,
                step_idx=i, total_steps=n,
                prev_intention=prev_intention,
                prev_summary_state=prev_summary_state,
            )
            prev_intention = exp.get("intentions", "")
            prev_summary_state = exp.get("summary_state", "")
            if verbose:
                ss = (exp["summary_state"][:60] + "...") if len(exp.get("summary_state", "")) > 60 else exp.get("summary_state", "")
                it = (exp["intentions"][:60] + "...") if len(exp.get("intentions", "")) > 60 else exp.get("intentions", "")
                print(f"    step {exp.get('idx', i):>3d}/{n}: summary_state={ss}")
                print(f"           intention={it}")
        except Exception as exc:
            print(f"    [WARN] step {exp.get('idx', i)}: labeling failed ({exc})")
            exp.setdefault("summary_state", None)
            exp.setdefault("summary", None)
            exp.setdefault("intentions", exp.get("intentions"))
            exp.pop("sub_tasks", None)
            exp.setdefault("skills", None)

    return episode_data


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_episode_files(
    input_dirs: List[Path],
    games: Optional[List[str]] = None,
) -> List[Path]:
    """Find all episode_*.json files under one or more *input_dirs*.

    Scans every sub-directory that looks like a game folder.  When *games* is
    given, only matching folder names are included.
    """
    files: List[Path] = []
    seen_games: set = set()

    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for gd in sorted(input_dir.iterdir()):
            if not gd.is_dir():
                continue
            game_name = gd.name
            if games is not None and game_name not in games:
                continue
            for fp in sorted(gd.glob("episode_*.json")):
                if fp.name == "episode_buffer.json":
                    continue
                files.append(fp)
            if any(gd.glob("episode_*.json")):
                seen_games.add(game_name)
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label cold-start episodes with GPT-5.4 (summary_state, intention, skills)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, nargs="*", default=None,
                        help="Input director(ies) with game sub-folders (default: all gpt54* output dirs)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Label a single episode JSON file instead of scanning a directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory for labeled episodes (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Only label these games (default: all found)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model for labeling (default: {MODEL_GPT54})")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to label per game (default: all)")
    parser.add_argument("--one_per_game", action="store_true",
                        help="Process only the first episode for each game (quick test run)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay in seconds between API calls (default: 0.1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite already-labeled episodes in output dir")
    parser.add_argument("--in_place", action="store_true",
                        help="Write labels back to the original input files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview labeling on first episode without saving")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-step labeling details")

    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dir] if args.input_dir else DEFAULT_INPUT_DIRS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    if args.one_per_game:
        args.max_episodes = 1

    # ---- Validate API key ----
    has_key = bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or (open_router_api_key and str(open_router_api_key).strip())
    )
    if not has_key:
        print("[WARNING] No API key detected. LLM calls will fail.")
        print("  Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

    # ---- Collect files ----
    if args.input_file:
        files = [Path(args.input_file)]
        if not files[0].exists():
            print(f"[ERROR] File not found: {args.input_file}")
            sys.exit(1)
    else:
        files = find_episode_files(input_dirs, games=args.games)

    if not files:
        dirs_str = ", ".join(str(d) for d in input_dirs)
        print(f"[ERROR] No episode files found under: {dirs_str}")
        sys.exit(1)

    # Group by game
    game_files: Dict[str, List[Path]] = {}
    for fp in files:
        game = fp.parent.name
        game_files.setdefault(game, []).append(fp)

    if not args.dry_run and not args.in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  GPT-5.4 Episode Labeling")
    print("=" * 78)
    print(f"  Model:       {args.model}")
    if args.input_file:
        print(f"  Input:       {args.input_file}")
    else:
        for d in input_dirs:
            print(f"  Input:       {d}")
    print(f"  Output:      {'in-place' if args.in_place else output_dir}")
    print(f"  Games:       {', '.join(sorted(game_files.keys()))}")
    print(f"  Episodes:    {sum(len(v) for v in game_files.values())} total")
    per_game = args.max_episodes if args.max_episodes else "all"
    print(f"  Per game:    {per_game} episode(s)")
    print(f"  Delay:       {args.delay}s between calls")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 78)

    overall_t0 = time.time()
    all_stats: List[Dict[str, Any]] = []

    for game, gfiles in sorted(game_files.items()):
        episode_files = gfiles[:args.max_episodes] if args.max_episodes else gfiles
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game} ({len(episode_files)} episodes)")
        print(f"{'━' * 78}")

        game_out_dir = output_dir / game
        if not args.dry_run and not args.in_place:
            game_out_dir.mkdir(parents=True, exist_ok=True)

        game_t0 = time.time()
        game_labeled = 0

        for fp in episode_files:
            out_path = game_out_dir / fp.name if not args.in_place else fp

            if not args.overwrite and not args.in_place and out_path.exists():
                print(f"  [SKIP] {fp.name} (already labeled)")
                continue

            print(f"\n  Labeling {fp.name} ...")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    episode_data = json.load(f)
            except Exception as exc:
                print(f"    [ERROR] Failed to load: {exc}")
                continue

            n_steps = len(episode_data.get("experiences", []))
            t0 = time.time()

            label_episode(
                episode_data,
                model=args.model,
                delay=args.delay,
                verbose=args.verbose,
            )

            elapsed = time.time() - t0
            print(f"    Labeled {n_steps} steps in {elapsed:.1f}s")

            if args.dry_run:
                exps = episode_data.get("experiences", [])
                preview = exps[0] if exps else {}
                print("\n    --- Preview (step 0) ---")
                print(f"    summary_state: {preview.get('summary_state')}")
                print(f"    summary:       {preview.get('summary')}")
                print(f"    intentions:    {preview.get('intentions')}")
                print(f"    skills:        {preview.get('skills')}")
                if len(exps) > 1:
                    print(f"\n    --- Preview (step 1) ---")
                    print(f"    summary_state: {exps[1].get('summary_state')}")
                    print(f"    intentions:    {exps[1].get('intentions')}")
                print("    --- end preview ---\n")
                break

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=2, ensure_ascii=False, default=str)
                print(f"    Saved → {out_path}")
                game_labeled += 1
            except Exception as exc:
                print(f"    [ERROR] Failed to save: {exc}")

        game_elapsed = time.time() - game_t0
        stat = {
            "game": game,
            "episodes_labeled": game_labeled,
            "elapsed_seconds": game_elapsed,
        }
        all_stats.append(stat)

        if not args.dry_run and not args.in_place and game_labeled > 0:
            summary_path = game_out_dir / "labeling_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({
                    "game": game,
                    "model": args.model,
                    "timestamp": datetime.now().isoformat(),
                    "episodes_labeled": game_labeled,
                    "elapsed_seconds": game_elapsed,
                }, f, indent=2, ensure_ascii=False)

    overall_elapsed = time.time() - overall_t0

    print(f"\n{'=' * 78}")
    print("  LABELING COMPLETE")
    print(f"{'=' * 78}")
    total_labeled = sum(s["episodes_labeled"] for s in all_stats)
    print(f"  Episodes labeled: {total_labeled}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    if not args.dry_run and not args.in_place:
        print(f"  Output:           {output_dir}")

        master = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total_elapsed_seconds": overall_elapsed,
            "per_game": all_stats,
        }
        master_path = output_dir / "labeling_batch_summary.json"
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        print(f"  Summary:          {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
