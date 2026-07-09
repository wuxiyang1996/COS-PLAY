#!/usr/bin/env python
"""
Label cold-start episode trajectories using GPT-5.4 with Skill Bank guidance.

Extends the base labeling pipeline (label_episodes_gpt54.py) by loading a
pre-built skill bank and running skill selection at every step.  The selected
skill's protocol, execution hints, and strategic description are:

  1. Injected into the intention-generation prompt so the LLM can propose
     subgoals aligned with known high-quality strategies.
  2. Stored in the ``skills`` field of each experience (as a structured dict,
     not ``null``).

Fields produced per step:

  - summary_state : compact key=value facts (deterministic, no LLM)
  - summary       : summary_state + LLM strategic note (delta-aware)
  - intentions    : [TAG] subgoal phrase via LLM (skill-aware)
  - skills        : structured skill guidance dict from the skill bank

Skill selection pipeline (mirrors decision_agents protocol):

  1. Retrieve top-k candidate skills via ``select_skill_from_bank``
     (TF-IDF keyword scoring / SkillQueryEngine.select).
  2. When k > 1, ask the LLM to pick the best candidate given state + intention.
  3. Format the chosen skill as structured guidance (protocol steps,
     preconditions, success/abort criteria, execution hint).
  4. Feed the guidance into the intention prompt so the [TAG] and phrase
     reflect the activated skill.

Output structure (labeling/output/gpt54_with_skills/<game_name>/):
  - episode_NNN.json        Labeled episode (original + new fields)
  - labeling_summary.json   Run statistics

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # Label all episodes (auto-detect skill banks per game)
    python labeling/label_episodes_with_skills.py

    # Specify skill bank directory
    python labeling/label_episodes_with_skills.py \\
        --bank skill_agents_grpo/extract_skillbank/output/gpt54_skillbank_grpo

    # Without skill bank (falls back to null skills, like base labeler)
    python labeling/label_episodes_with_skills.py --no-bank

    # Label specific games, verbose
    python labeling/label_episodes_with_skills.py --games tetris candy_crush -v

    # Quick test: one episode per game, dry run
    python labeling/label_episodes_with_skills.py --one_per_game --dry_run
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
    select_skill_from_bank,
    HARD_SUMMARY_CHAR_LIMIT,
    SUBGOAL_TAGS,
)

try:
    from decision_agents.agent_helper import _get_protocol_for_skill
except ImportError:
    _get_protocol_for_skill = None

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

from skill_agents.skill_bank.bank import SkillBankMVP

try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "gpt54_skill_labeled"

DEFAULT_BANK_DIRS: List[Path] = [
    CODEBASE_ROOT / "labeling" / "output" / "gpt54_skillbank",
    CODEBASE_ROOT / "skill_agents_grpo" / "extract_skillbank" / "output" / "gpt54_skillbank_grpo",
]

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
    """Return a compact string of key changes between two ``summary_state`` values."""
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
    """Return a short urgency warning when absolute state values are critical."""

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
# GPT-5.4 chat helper
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
# Skill bank loading
# ---------------------------------------------------------------------------

def load_skill_bank(
    bank_path: str,
    *,
    use_query_engine: bool = True,
) -> Tuple[Any, Any]:
    """Load a SkillBankMVP from a JSONL file or directory.

    Returns (bank, engine).  *engine* is a SkillQueryEngine when available,
    otherwise None.
    """
    bp = Path(bank_path)
    if bp.is_dir():
        candidates = ["bank.jsonl", "skill_bank.jsonl"]
        jsonl = None
        for c in candidates:
            if (bp / c).exists():
                jsonl = bp / c
                break
        if jsonl is None:
            jsonls = sorted(bp.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            jsonl = jsonls[0] if jsonls else None
        if jsonl is None:
            print(f"[load_skill_bank] WARNING: no .jsonl found in {bp}")
            return None, None
        bp = jsonl

    bank = SkillBankMVP(path=str(bp))
    bank.load()
    print(f"[load_skill_bank] Loaded {len(bank)} skills from {bp}")

    engine = None
    if use_query_engine and SkillQueryEngine is not None and len(bank) > 0:
        try:
            engine = SkillQueryEngine(bank)
            print(f"[load_skill_bank] SkillQueryEngine initialised")
        except Exception as exc:
            print(f"[load_skill_bank] SkillQueryEngine init failed: {exc}")
    return bank, engine


def discover_skill_banks(
    bank_dirs: List[Path],
    games: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Auto-discover per-game skill banks under one or more root directories.

    Returns ``{game_name: skill_bank_object_or_engine}``.
    """
    per_game: Dict[str, Any] = {}
    for root in bank_dirs:
        if not root.exists():
            continue
        for gd in sorted(root.iterdir()):
            if not gd.is_dir():
                continue
            game_name = gd.name
            if games is not None and game_name not in games:
                continue
            if game_name in per_game:
                continue
            bank, engine = load_skill_bank(str(gd), use_query_engine=True)
            if bank is not None and len(bank) > 0:
                per_game[game_name] = engine if engine is not None else bank
    return per_game


# ---------------------------------------------------------------------------
# Skill selection: top-k retrieval + LLM pick
# ---------------------------------------------------------------------------

def get_skill_guidance(
    skill_bank: Any,
    state_text: str,
    game_name: str = "",
    intention: str = "",
    structured_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Query the skill bank for the single best skill matching the state."""
    if skill_bank is None:
        return None
    key_parts = []
    if game_name:
        key_parts.append(game_name)
    if intention:
        key_parts.append(intention)
    key_parts.append(state_text[:1500])
    key = " ".join(key_parts)

    state_for_scoring = None
    if structured_state and isinstance(structured_state, dict):
        state_for_scoring = {
            k: (float(v) if isinstance(v, (int, float, bool)) else 1.0)
            for k, v in structured_state.items()
            if v is not None and str(v).strip()
        }

    try:
        result = select_skill_from_bank(
            skill_bank, key,
            current_state=state_for_scoring,
            top_k=1,
        )
        if result and result.get("skill_id"):
            _enrich_candidate(skill_bank, result)
            return result
    except Exception:
        pass
    return None


def get_top_k_skill_candidates(
    skill_bank: Any,
    state_text: str,
    game_name: str = "",
    intention: str = "",
    structured_state: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve *top_k* skill candidates from the skill bank."""
    if skill_bank is None:
        return []

    key_parts = []
    if game_name:
        key_parts.append(game_name)
    if intention:
        key_parts.append(intention)
    key_parts.append(state_text[:1500])
    key = " ".join(key_parts)

    state_for_scoring = None
    if structured_state and isinstance(structured_state, dict):
        state_for_scoring = {
            k: (float(v) if isinstance(v, (int, float, bool)) else 1.0)
            for k, v in structured_state.items()
            if v is not None and str(v).strip()
        }

    candidates: List[Dict[str, Any]] = []

    if hasattr(skill_bank, "select"):
        try:
            results = skill_bank.select(
                key,
                current_state=state_for_scoring,
                current_predicates=state_for_scoring,
                top_k=top_k,
            )
            for r in (results or []):
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                sid = d.get("skill_id")
                if not sid:
                    continue
                if _get_protocol_for_skill is not None:
                    protocol = _get_protocol_for_skill(skill_bank, sid)
                    d["protocol"] = protocol or d.get("protocol", {})
                _enrich_candidate(skill_bank, d)
                candidates.append(d)
        except Exception:
            pass

    if candidates:
        return candidates

    try:
        single = get_skill_guidance(
            skill_bank, state_text, game_name, intention, structured_state,
        )
        if single and single.get("skill_id"):
            return [single]
    except Exception:
        pass

    return []


def _enrich_candidate(skill_bank: Any, d: Dict[str, Any]) -> None:
    """Fill in missing skill_name / execution_hint from the bank."""
    if d.get("skill_name") and d.get("execution_hint"):
        return
    sid = d.get("skill_id")
    if not sid:
        return
    underlying = (
        getattr(skill_bank, "_bank", None)
        or getattr(skill_bank, "bank", None)
        or skill_bank
    )
    if hasattr(underlying, "get_skill"):
        skill_obj = underlying.get_skill(sid)
        if skill_obj:
            if not d.get("skill_name"):
                d["skill_name"] = skill_obj.name or sid
            if not d.get("execution_hint"):
                d["execution_hint"] = skill_obj.strategic_description or ""


# ---------------------------------------------------------------------------
# LLM-based skill selection among candidates
# ---------------------------------------------------------------------------

SKILL_SELECTION_SYSTEM = (
    "You are an expert game strategist. "
    "Given the current game state and candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits>\n"
    "SKILL: <number>\n"
)


def _format_candidates_for_selection(candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, c in enumerate(candidates, 1):
        name = c.get("skill_name") or c.get("skill_id", f"strategy_{i}")
        hint = c.get("execution_hint", "")
        protocol = c.get("protocol", {})
        steps = protocol.get("steps", []) if isinstance(protocol, dict) else []

        lines.append(f"  {i}. {name}")
        if hint:
            lines.append(f"     Strategy: {hint[:150]}")
        if steps:
            step_text = " -> ".join(steps[:4])
            if len(steps) > 4:
                step_text += " -> ..."
            lines.append(f"     Plan: {step_text}")

        confidence = c.get("confidence")
        if confidence is not None:
            lines.append(f"     Confidence: {confidence:.2f}")
    return "\n".join(lines)


def select_skill_via_llm(
    candidates: List[Dict[str, Any]],
    state_summary: str,
    intention: str,
    model: str = MODEL_GPT54,
    temperature: float = 0.3,
) -> Tuple[int, Optional[str]]:
    """LLM selects the best skill from *candidates*.

    Returns (chosen_index, reasoning).  Falls back to index 0 on failure.
    """
    if not candidates:
        return 0, None
    if len(candidates) == 1:
        return 0, "only one candidate"
    if ask_model is None:
        return 0, None

    candidates_text = _format_candidates_for_selection(candidates)
    user_content = (
        f"Game state:\n{state_summary[:3000]}\n\n"
        f"Current intention: {intention[:500]}\n\n"
        f"Available strategies (pick ONE by number):\n{candidates_text}\n\n"
        f"Choose the best strategy. Output REASONING then SKILL number."
    )
    prompt = SKILL_SELECTION_SYSTEM + "\n" + user_content

    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=256)
        if reply and not reply.startswith("Error"):
            return _parse_skill_selection(reply, len(candidates), candidates)
    except Exception:
        pass

    return 0, None


def _parse_skill_selection(
    reply: str,
    n_candidates: int,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, Optional[str]]:
    """Parse the LLM skill selection response. Returns (0-based index, reasoning)."""
    if not reply:
        return 0, None

    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    reasoning = None
    reasoning_m = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nSKILL|\Z)", cleaned, re.DOTALL | re.IGNORECASE,
    )
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    skill_m = re.search(r"SKILL\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if skill_m:
        idx = int(skill_m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning

    tail = cleaned[-100:]
    nums = re.findall(r"\b(\d+)\b", tail)
    for n_str in reversed(nums):
        idx = int(n_str) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning

    if candidates:
        cleaned_lower = cleaned.lower()
        for i, c in enumerate(candidates):
            name = (c.get("skill_name") or "").lower()
            if name and len(name) >= 4 and name in cleaned_lower:
                return i, reasoning

    return 0, reasoning


# ---------------------------------------------------------------------------
# Format skill guidance for prompt injection
# ---------------------------------------------------------------------------

def _skill_guidance_to_label(guidance: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert a skill guidance dict into a compact label for storage."""
    if guidance is None or not guidance.get("skill_id"):
        return None
    label: Dict[str, Any] = {
        "skill_id": guidance["skill_id"],
        "skill_name": guidance.get("skill_name", ""),
    }
    if guidance.get("execution_hint"):
        label["execution_hint"] = guidance["execution_hint"][:200]
    if guidance.get("why_selected"):
        label["why_selected"] = guidance["why_selected"][:200]
    protocol = guidance.get("protocol", {})
    if protocol and isinstance(protocol, dict):
        compact_proto: Dict[str, Any] = {}
        if protocol.get("steps"):
            compact_proto["steps"] = protocol["steps"][:7]
        if protocol.get("preconditions"):
            compact_proto["preconditions"] = protocol["preconditions"][:3]
        if protocol.get("success_criteria"):
            compact_proto["success_criteria"] = protocol["success_criteria"][:3]
        if compact_proto:
            label["protocol"] = compact_proto
    if guidance.get("confidence") is not None:
        label["confidence"] = guidance["confidence"]
    return label


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
    """Produce a compact ``key=value`` state summary (deterministic, no LLM)."""
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
    skill_bank: Any = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Label a single experience dict in-place and return it.

    Produces four fields:

    * ``summary_state`` — compact ``key=value`` string (deterministic)
    * ``summary`` — ``summary_state | note=<strategic note>``
    * ``intentions`` — ``[TAG] subgoal phrase`` (skill-aware)
    * ``skills`` — structured skill guidance dict (or None when no bank)
    """
    state = exp.get("state", "")
    action = exp.get("action", "")
    reward = exp.get("reward", 0.0)
    original_reasoning = exp.get("intentions")
    idx = exp.get("idx", step_idx)

    # --- summary_state (deterministic, 0 LLM calls) ---
    summary_state = generate_summary_state(
        state,
        game_name=game_name,
        step_idx=idx if isinstance(idx, int) else step_idx,
        total_steps=total_steps,
        reward=reward if isinstance(reward, (int, float)) else 0.0,
    )
    exp["summary_state"] = summary_state

    # --- skill selection (top-k candidates + LLM pick) ---
    guidance: Optional[Dict[str, Any]] = None
    if skill_bank is not None:
        facts = extract_game_facts(state, game_name)
        structured_state = {k: v for k, v in facts.items() if v}

        candidates = get_top_k_skill_candidates(
            skill_bank,
            summary_state or state[:1500],
            game_name=game_name,
            intention=prev_intention,
            structured_state=structured_state if structured_state else None,
            top_k=top_k,
        )

        if candidates:
            chosen_idx, skill_reasoning = select_skill_via_llm(
                candidates,
                state_summary=summary_state or state[:500],
                intention=prev_intention,
                model=model,
            )
            guidance = candidates[chosen_idx]
            if skill_reasoning:
                guidance["why_selected"] = skill_reasoning

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

    # --- intention ([TAG] subgoal, delta + prev step, 1 LLM call) ---
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

    # --- skills (structured label from selected skill) ---
    exp.pop("sub_tasks", None)
    exp["skills"] = _skill_guidance_to_label(guidance)

    # --- GRPO training metadata for skill-selection LoRA ---
    if skill_bank is not None and candidates:
        exp["skill_candidates"] = [c.get("skill_id") for c in candidates]
        exp["skill_chosen_idx"] = chosen_idx
        exp["skill_reasoning"] = skill_reasoning if skill_reasoning else None

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
    skill_bank: Any = None,
    top_k: int = 3,
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
                skill_bank=skill_bank,
                top_k=top_k,
            )
            prev_intention = exp.get("intentions", "")
            prev_summary_state = exp.get("summary_state", "")
            if verbose:
                ss = (exp["summary_state"][:60] + "...") if len(exp.get("summary_state", "")) > 60 else exp.get("summary_state", "")
                it = (exp["intentions"][:60] + "...") if len(exp.get("intentions", "")) > 60 else exp.get("intentions", "")
                sk = exp.get("skills")
                sk_name = sk.get("skill_name", sk.get("skill_id", "none")) if isinstance(sk, dict) else "none"
                print(f"    step {exp.get('idx', i):>3d}/{n}: summary_state={ss}")
                print(f"           intention={it}")
                print(f"           skill={sk_name}")
        except Exception as exc:
            print(f"    [WARN] step {exp.get('idx', i)}: labeling failed ({exc})")
            exp.setdefault("summary_state", None)
            exp.setdefault("summary", None)
            exp.setdefault("intentions", exp.get("intentions"))
            exp.pop("sub_tasks", None)
            exp.setdefault("skills", None)

    return episode_data


# ---------------------------------------------------------------------------
# GRPO cold-start data export for decision-agent LoRA training
# ---------------------------------------------------------------------------
#
# Two JSONL files per game:
#   action_taking.jsonl    — state + available actions → chosen action
#   skill_selection.jsonl  — state + intention + candidates → chosen skill
#
# Both follow the same schema so a single GRPO trainer can consume either:
#   { "type", "game", "episode", "step",
#     "prompt", "completion", "reward",
#     + type-specific metadata }
# ---------------------------------------------------------------------------

_ACTION_SYSTEM = (
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

_ACTION_SYSTEM_POKEMON = (
    "You are an expert Pokemon Red player. "
    "You receive a game state with a full map and must choose the best action.\n\n"
    "Rules:\n"
    "- Study the map and state carefully before choosing.\n"
    "- Use tool calls with coordinates read from the map.\n"
    "- NEVER repeat the same action more than 2 times in a row.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Available actions:\n"
    "  Buttons: up, down, left, right, a, b, start, select\n"
    "  Tools:\n"
    "    move_to(x, y) - Walk to walkable tile (O or G).\n"
    "    warp_with_warp_point(x, y) - Walk to WarpPoint and warp.\n"
    "    interact_with_object(name) - Interact with named sprite.\n"
    "    continue_dialog - Advance dialog.\n"
    "    select_move_in_battle - Use first move in battle.\n"
    "    switch_pkmn_in_battle - Switch Pokemon.\n"
    "    run_away - Flee from battle.\n"
    "    use_item_in_battle - Use item in battle.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <button or tool call, e.g. move_to(5, 3)>\n"
)

_SKILL_SYSTEM = (
    "You are an expert game strategist. "
    "Given the current game state and candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits>\n"
    "SKILL: <number>\n"
)


_POKEMON_RED_ACTIONS = [
    "up", "down", "left", "right", "a", "b", "start", "select",
    "move_to", "interact_with_object", "warp_with_warp_point",
    "continue_dialog", "select_move_in_battle",
    "switch_pkmn_in_battle", "run_away", "use_item_in_battle",
]


_OVERWORLD_DIR_MAP = {"north": "up", "south": "down", "west": "left", "east": "right"}


def _pokemon_action_to_tool_call(action: str) -> str:
    """Convert 'use_tool(tool_name, (args))' to 'tool_name(arg_vals)' format."""
    if not action.startswith("use_tool("):
        return action
    inside = action[len("use_tool("):-1] if action.endswith(")") else action[len("use_tool("):]
    parts = inside.split(",", 1)
    tool_name = parts[0].strip()
    if len(parts) < 2:
        return tool_name
    arg_str = parts[1].strip().strip("()")
    if not arg_str:
        return tool_name
    vals = []
    for part in arg_str.split(","):
        part = part.strip()
        if "=" in part:
            val = part.split("=", 1)[1].strip().strip("'\"")
            vals.append(val)
        else:
            vals.append(part.strip("'\""))
    return f"{tool_name}({', '.join(vals)})"


def _normalize_action(action: str, available_actions: list, game_name: str):
    """Return (available_actions, action_str_or_index) for the cold-start sample.

    For Pokemon Red, returns the full tool call string (e.g. 'move_to(5, 3)')
    instead of a numeric index, so SFT teaches the parametric format.

    ``overworld_map_transition(direction=...)`` is not in the Orak action
    space; it maps to the corresponding d-pad button (north→up, etc.).
    """
    if game_name == "pokemon_red":
        if action in _POKEMON_RED_ACTIONS:
            return _POKEMON_RED_ACTIONS, action

        if action.startswith("use_tool("):
            bare = action.split("(", 1)[1].split(",")[0].strip()
            if bare == "overworld_map_transition":
                for direction, button in _OVERWORLD_DIR_MAP.items():
                    if direction in action:
                        return _POKEMON_RED_ACTIONS, button
            tool_call = _pokemon_action_to_tool_call(action)
            return _POKEMON_RED_ACTIONS, tool_call

        if action in available_actions:
            return available_actions, action
        return _POKEMON_RED_ACTIONS, "a"

    if action in available_actions:
        return available_actions, available_actions.index(action) + 1

    return available_actions, 1


def export_grpo_coldstart_data(
    episode_data: Dict[str, Any],
    grpo_dir: Path,
) -> Dict[str, int]:
    """Export GRPO cold-start training data for both action-taking and skill-selection.

    Writes two append-mode JSONL files under *grpo_dir*:

    ``action_taking.jsonl``
        One row per step.  Contains the full action-selection prompt
        (state + available actions + active skill guidance) and the
        expert action chosen by GPT-5.4.  Reward is the step reward.

    ``skill_selection.jsonl``
        One row per step where >= 2 skill candidates were available.
        Contains the skill-selection prompt (state + intention +
        numbered candidate menu) and the expert skill choice.
        Reward is the step reward.

    Returns ``{"action": n_action_samples, "skill": n_skill_samples}``.
    """
    game_name = episode_data.get("game_name", "")
    episode_id = episode_data.get("episode_id", "")
    if not episode_id:
        episode_id = f"{game_name}_episode"
    experiences = episode_data.get("experiences", [])

    grpo_dir.mkdir(parents=True, exist_ok=True)
    action_path = grpo_dir / "action_taking.jsonl"
    skill_path = grpo_dir / "skill_selection.jsonl"

    n_action = 0
    n_skill = 0

    with open(action_path, "a", encoding="utf-8") as f_act, \
         open(skill_path, "a", encoding="utf-8") as f_sk:

        for exp in experiences:
            step_idx = exp.get("idx", -1)
            state = exp.get("state", "")
            action = exp.get("action", "")
            reward = exp.get("reward", 0.0)
            summary_state = exp.get("summary_state", "")
            intention = exp.get("intentions", "")
            skills_label = exp.get("skills")

            # ── Action-taking sample ──────────────────────────
            raw_available = exp.get("available_actions")
            if not raw_available:
                raw_available = [action]

            available_actions, action_num = _normalize_action(
                action, raw_available, game_name,
            )

            state_text = summary_state if summary_state else state[:4000]

            action_lines = "\n".join(
                f"  {i+1}. {a}" for i, a in enumerate(available_actions)
            )

            skill_block = ""
            if skills_label and isinstance(skills_label, dict) and skills_label.get("skill_id"):
                sk_parts = [f"\n--- Active Skill: {skills_label.get('skill_name', skills_label['skill_id'])} ---"]
                if skills_label.get("execution_hint"):
                    sk_parts.append(f"  Strategy: {skills_label['execution_hint'][:200]}")
                proto = skills_label.get("protocol", {})
                if proto and isinstance(proto, dict) and proto.get("steps"):
                    sk_parts.append(f"  Plan: {' -> '.join(proto['steps'][:5])}")
                sk_parts.append("--- end skill ---\n")
                skill_block = "\n".join(sk_parts)

            if game_name == "pokemon_red":
                action_prompt = (
                    _ACTION_SYSTEM_POKEMON + skill_block + "\n"
                    f"Game state:\n\n{state_text}\n\n"
                    f"Choose the best action. Output REASONING then ACTION (button or tool call)."
                )
            else:
                action_prompt = (
                    _ACTION_SYSTEM + skill_block + "\n"
                    f"Game state:\n\n{state_text}\n\n"
                    f"Available actions (pick ONE by number):\n{action_lines}\n\n"
                    f"Choose the best action. Output REASONING then ACTION number."
                )

            reasoning_text = exp.get("skill_reasoning") or ""
            if reasoning_text:
                reasoning_text = reasoning_text.split("\n")[0].strip()
                if len(reasoning_text) > 200:
                    reasoning_text = reasoning_text[:197] + "..."
            if not reasoning_text or len(reasoning_text) < 10:
                reasoning_text = "Expert play."
            action_completion = f"REASONING: {reasoning_text}\nACTION: {action_num}"

            action_sample = {
                "type": "action_taking",
                "game": game_name,
                "episode": episode_id,
                "step": step_idx,
                "prompt": action_prompt,
                "completion": action_completion,
                "chosen_action": action,
                "available_actions": available_actions,
                "reward": reward,
                "summary_state": summary_state,
                "intention": intention,
                "active_skill": skills_label.get("skill_id") if isinstance(skills_label, dict) else None,
            }
            f_act.write(json.dumps(action_sample, ensure_ascii=False) + "\n")
            n_action += 1

            # ── Skill-selection sample ────────────────────────
            candidates = exp.get("skill_candidates")
            if not candidates or len(candidates) < 2:
                continue

            chosen_idx = exp.get("skill_chosen_idx", 0)
            chosen_skill_id = candidates[chosen_idx] if chosen_idx < len(candidates) else candidates[0]
            skill_reasoning = exp.get("skill_reasoning")

            candidate_lines = "\n".join(
                f"  {ci+1}. {cid}" for ci, cid in enumerate(candidates)
            )

            skill_prompt = (
                _SKILL_SYSTEM + "\n"
                f"Game state:\n{summary_state}\n\n"
                f"Current intention: {intention}\n\n"
                f"Available strategies (pick ONE by number):\n{candidate_lines}\n\n"
                f"Choose the best strategy. Output REASONING then SKILL number."
            )

            if skill_reasoning:
                skill_completion = f"REASONING: {skill_reasoning}\nSKILL: {chosen_idx + 1}"
            else:
                skill_completion = f"SKILL: {chosen_idx + 1}"

            skill_sample = {
                "type": "skill_selection",
                "game": game_name,
                "episode": episode_id,
                "step": step_idx,
                "prompt": skill_prompt,
                "completion": skill_completion,
                "chosen_idx": chosen_idx,
                "skill_candidates": candidates,
                "chosen_skill_id": chosen_skill_id,
                "reward": reward,
                "summary_state": summary_state,
                "intention": intention,
            }
            f_sk.write(json.dumps(skill_sample, ensure_ascii=False) + "\n")
            n_skill += 1

    return {"action": n_action, "skill": n_skill}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_episode_files(
    input_dirs: List[Path],
    games: Optional[List[str]] = None,
) -> List[Path]:
    """Find all episode_*.json files under one or more *input_dirs*."""
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
        description="Label cold-start episodes with GPT-5.4 + skill bank guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, nargs="*", default=None,
                        help="Input director(ies) with game sub-folders (default: all gpt54* output dirs)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Label a single episode JSON file instead of scanning a directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Only label these games (default: all found)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model for labeling (default: {MODEL_GPT54})")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to label per game (default: all)")
    parser.add_argument("--one_per_game", action="store_true",
                        help="Process only the first episode for each game")
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

    # Skill bank arguments
    parser.add_argument("--bank", type=str, default=None,
                        help="Skill bank directory (or JSONL file). Auto-discovers per-game banks within.")
    parser.add_argument("--bank_dir", type=str, nargs="*", default=None,
                        help="Additional skill bank root director(ies) to scan for per-game banks")
    parser.add_argument("--no-bank", action="store_true",
                        help="Disable skill bank (skills field will be null)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of skill candidates to retrieve before LLM selection (default: 3)")
    parser.add_argument("--no-query-engine", action="store_true",
                        help="Disable SkillQueryEngine (use TF-IDF fallback only)")

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

    # ---- Collect episode files ----
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

    game_files: Dict[str, List[Path]] = {}
    for fp in files:
        game = fp.parent.name
        game_files.setdefault(game, []).append(fp)

    # ---- Load skill banks ----
    per_game_banks: Dict[str, Any] = {}
    bank_path_str = "none"

    if not args.no_bank:
        bank_search_dirs: List[Path] = []
        if args.bank:
            bp = Path(args.bank)
            if bp.is_file() or any((bp / c).exists() for c in ["bank.jsonl", "skill_bank.jsonl"]):
                bank, engine = load_skill_bank(str(bp), use_query_engine=not args.no_query_engine)
                if bank is not None:
                    for game in game_files:
                        per_game_banks[game] = engine if engine is not None else bank
                    bank_path_str = str(bp)
            else:
                bank_search_dirs.append(bp)
        if args.bank_dir:
            bank_search_dirs.extend(Path(d) for d in args.bank_dir)
        if not bank_search_dirs and not per_game_banks:
            bank_search_dirs = DEFAULT_BANK_DIRS

        if bank_search_dirs:
            discovered = discover_skill_banks(
                bank_search_dirs, games=args.games or list(game_files.keys()),
            )
            for gn, sb in discovered.items():
                if gn not in per_game_banks:
                    per_game_banks[gn] = sb
            if discovered:
                bank_path_str = ", ".join(str(d) for d in bank_search_dirs)

    if not args.dry_run and not args.in_place:
        output_dir.mkdir(parents=True, exist_ok=True)

    bank_summary = (
        f"{len(per_game_banks)} game(s): {', '.join(sorted(per_game_banks.keys()))}"
        if per_game_banks else "none"
    )

    print("=" * 78)
    print("  GPT-5.4 Episode Labeling with Skill Bank")
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
    print(f"  Skill bank:  {bank_summary}")
    print(f"  Bank source: {bank_path_str}")
    print(f"  Top-k:       {args.top_k}")
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

        game_skill_bank = per_game_banks.get(game)
        if game_skill_bank is not None:
            bank_size = len(game_skill_bank) if hasattr(game_skill_bank, "__len__") else "?"
            print(f"  Skill bank: {bank_size} skills loaded for {game}")
        else:
            print(f"  Skill bank: none for {game} (skills will be null)")

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
                skill_bank=game_skill_bank,
                top_k=args.top_k,
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
                sk = preview.get("skills")
                if isinstance(sk, dict):
                    print(f"    skills:        {sk.get('skill_name', sk.get('skill_id', 'none'))}")
                    if sk.get("execution_hint"):
                        print(f"                   hint: {sk['execution_hint'][:80]}")
                    proto = sk.get("protocol", {})
                    if proto and proto.get("steps"):
                        print(f"                   steps: {' -> '.join(proto['steps'][:3])}")
                else:
                    print(f"    skills:        {sk}")
                if len(exps) > 1:
                    print(f"\n    --- Preview (step 1) ---")
                    print(f"    summary_state: {exps[1].get('summary_state')}")
                    print(f"    intentions:    {exps[1].get('intentions')}")
                    sk1 = exps[1].get("skills")
                    if isinstance(sk1, dict):
                        print(f"    skills:        {sk1.get('skill_name', sk1.get('skill_id', 'none'))}")
                    else:
                        print(f"    skills:        {sk1}")
                print("    --- end preview ---\n")
                break

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=2, ensure_ascii=False, default=str)
                print(f"    Saved → {out_path}")
                game_labeled += 1

                # Export GRPO cold-start training data (action-taking + skill-selection)
                grpo_game_dir = output_dir / "grpo_coldstart" / game
                grpo_counts = export_grpo_coldstart_data(episode_data, grpo_game_dir)
                n_act = grpo_counts["action"]
                n_sk = grpo_counts["skill"]
                print(f"    GRPO data: {n_act} action + {n_sk} skill samples → {grpo_game_dir}")

            except Exception as exc:
                print(f"    [ERROR] Failed to save: {exc}")

        game_elapsed = time.time() - game_t0
        stat = {
            "game": game,
            "episodes_labeled": game_labeled,
            "elapsed_seconds": game_elapsed,
            "skill_bank_loaded": game_skill_bank is not None,
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
                    "skill_bank": bank_path_str,
                    "top_k": args.top_k,
                }, f, indent=2, ensure_ascii=False)

    overall_elapsed = time.time() - overall_t0

    print(f"\n{'=' * 78}")
    print("  LABELING COMPLETE")
    print(f"{'=' * 78}")
    total_labeled = sum(s["episodes_labeled"] for s in all_stats)
    print(f"  Episodes labeled: {total_labeled}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    games_with_skills = sum(1 for s in all_stats if s.get("skill_bank_loaded"))
    print(f"  Games w/ skills:  {games_with_skills}/{len(all_stats)}")
    if not args.dry_run and not args.in_place:
        print(f"  Output:           {output_dir}")

        master = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total_elapsed_seconds": overall_elapsed,
            "skill_bank": bank_path_str,
            "top_k": args.top_k,
            "per_game": all_stats,
        }
        master_path = output_dir / "labeling_batch_summary.json"
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        print(f"  Summary:          {master_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
