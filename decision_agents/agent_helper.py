# This file defines helper functions for the VLM decision agent: state summarization,
# intention inference, episodic memory store, and skill-bank formatting.
#
# The state-summary pipeline produces compact key=value summaries optimised for
# LLM/VLM context windows, retrieval, skill-bank indexing, and trajectory
# segmentation.  Summaries are *not* human-readable paragraphs — they are short
# structured abstractions of the current game state.

from __future__ import annotations

import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from API_func import ask_model
except ImportError:
    ask_model = None


# ---------------------------------------------------------------------------
# Summary budget constants
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
"""Fallback LLM model when none is explicitly provided by the caller.
Both GPT and Qwen inference paths should always pass an explicit model;
this constant only guards against accidental None."""

DEFAULT_SUMMARY_CHAR_BUDGET: int = 400
"""Default budget for state summaries (characters).  Prefer ~220-380 when
possible; treat 400 as the hard cap, not a target to fill."""

HARD_SUMMARY_CHAR_LIMIT: int = 400
"""Absolute upper bound.  No summary should exceed this."""


# ---------------------------------------------------------------------------
# Boilerplate patterns to strip from raw observations
# ---------------------------------------------------------------------------

_BOILERPLATE_RE = re.compile(
    r"(?i)"
    r"(choose one action[^\n]*"
    r"|valid actions?[^\n]*"
    r"|possible actions?[^\n]*"
    r"|examples?[:\s][^\n]*"
    r"|respond with[^\n]*"
    r"|reply with[^\n]*"
    r"|submit your orders[^\n]*"
    r"|output format[^\n]*"
    r"|order format[^\n]*"
    r"|--- order format ---[^\n]*"
    r"|  hold:[^\n]*"
    r"|  move:[^\n]*"
    r"|  support hold:[^\n]*"
    r"|  support move:[^\n]*"
    r"|  convoy:[^\n]*"
    r"|  retreat:[^\n]*"
    r"|  disband:[^\n]*"
    r"|  build:[^\n]*"
    r"|example:?\s*\[?\"[A-Z]\s[A-Z]{3}[^\n]*"
    r")"
)

# Keys to prioritise when compressing a structured state dict
_PRIORITY_KEYS = (
    "game", "phase", "subgoal", "objective", "self", "ally", "enemy",
    "critical", "resources", "orders", "inventory", "progress",
    "time_left", "affordance", "delta", "valid_actions",
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_str(x: Any) -> str:
    """Coerce *x* to a short string suitable for a summary slot."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return ""
        items = [_safe_str(i) for i in x[:6]]
        out = ",".join(items)
        if len(x) > 6:
            out += f"..+{len(x) - 6}"
        return out
    if isinstance(x, dict):
        parts = [f"{k}:{_safe_str(v)}" for k, v in list(x.items())[:4]]
        return "{" + ",".join(parts) + "}"
    return str(x).strip()


def _remove_boilerplate(obs: str) -> str:
    """Strip action-formatting instructions and boilerplate from *obs*."""
    cleaned = _BOILERPLATE_RE.sub("", obs)
    # collapse blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _truncate_keep_important(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars* keeping the most information-dense prefix.

    Prefers cutting at a sentence / clause boundary when possible.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # try to cut at last sentence-end
    for sep in (". ", "| ", "; ", ", ", " "):
        pos = cut.rfind(sep)
        if pos > max_chars // 2:
            return cut[:pos + len(sep)].rstrip()
    return cut.rstrip()


def _join_kv(parts: List[Tuple[str, str]], max_chars: int) -> str:
    """Join ``(key, value)`` pairs as ``key=value | key=value | ...``.

    Drops trailing pairs if the result would exceed *max_chars*.
    """
    segments: List[str] = []
    length = 0
    for key, val in parts:
        if not val:
            continue
        seg = f"{key}={val}"
        added = len(seg) + (3 if segments else 0)  # " | " separator
        if length + added > max_chars:
            break
        segments.append(seg)
        length += added
    return " | ".join(segments)


# ---------------------------------------------------------------------------
# Structured-state compressor
# ---------------------------------------------------------------------------

def compact_structured_state(
    structured_state: Dict[str, Any],
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
) -> str:
    """Compress a structured state dict into a compact ``key=value`` summary.

    *structured_state* should be a flat-ish dict produced by an env wrapper's
    ``build_structured_state_summary()``.  Keys listed in ``_PRIORITY_KEYS``
    are emitted first; remaining keys are appended if budget allows.

    Returns:
        A string of at most *max_chars* characters.
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)
    if not structured_state:
        return ""

    ordered: List[Tuple[str, str]] = []
    seen = set()
    for k in _PRIORITY_KEYS:
        if k in structured_state:
            ordered.append((k, _safe_str(structured_state[k])))
            seen.add(k)
    for k, v in structured_state.items():
        if k not in seen:
            ordered.append((k, _safe_str(v)))

    return _join_kv(ordered, max_chars)


# ---------------------------------------------------------------------------
# Text-observation compressor
# ---------------------------------------------------------------------------

def compact_text_observation(
    observation: str,
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
) -> str:
    """Deterministically compress a raw text observation into a short summary.

    Steps:
      1. Strip boilerplate / action-format instructions.
      2. Split into clauses.
      3. Keep the most informative clauses that fit within *max_chars*.

    Returns:
        A string of at most *max_chars* characters.  Never returns the raw
        observation verbatim (even if it is already short).
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)
    if not observation or not isinstance(observation, str):
        return ""

    text = _remove_boilerplate(observation)
    if not text:
        return ""

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Split into clauses (sentences, newlines, pipe-delimited)
    clauses = re.split(r"[.\n|]+", text)
    clauses = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 2]

    if not clauses:
        return _truncate_keep_important(text, max_chars)

    # Heuristic: drop lines that are purely decorative (=== headers, --- separators)
    clauses = [c for c in clauses if not re.match(r"^[-=]{3,}", c)]

    # Build output greedily, keeping as many clauses as fit
    parts: List[str] = []
    length = 0
    for c in clauses:
        needed = len(c) + (3 if parts else 0)  # " | " separator
        if length + needed > max_chars:
            break
        parts.append(c)
        length += needed

    if not parts:
        # Single long clause — truncate it
        return _truncate_keep_important(clauses[0], max_chars)

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Subgoal tags for structured intention labeling
# ---------------------------------------------------------------------------

SUBGOAL_TAGS = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)


# ---------------------------------------------------------------------------
# Game-aware fact extraction (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _try_parse_dict(s: str) -> Optional[Dict[str, Any]]:
    """Try to parse *s* as a Python dict literal or JSON object."""
    s = s.strip()
    if not s.startswith("{"):
        return None
    import json as _json
    try:
        return _json.loads(s)
    except Exception:
        pass
    import ast as _ast
    try:
        result = _ast.literal_eval(s)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _extract_tetris_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    lines = state.split("\n")
    board_lines: List[str] = []
    in_board = False
    for ln in lines:
        s = ln.strip()
        if len(s) == 10 and all(c in ".IOTSZJL" for c in s):
            in_board = True
            board_lines.append(s)
            if len(board_lines) >= 20:
                break
        elif in_board:
            break
    if board_lines:
        # Find boundary between active piece (floating) and settled stack.
        # Scan from bottom upward; the first fully-empty row marks the top
        # of the settled region (cleared rows can't exist in Tetris).
        settled_top = len(board_lines)
        for row_idx in range(len(board_lines) - 1, -1, -1):
            if all(c == "." for c in board_lines[row_idx]):
                settled_top = row_idx + 1
                break
        else:
            settled_top = 0
        settled = board_lines[settled_top:]
        facts["stack_h"] = str(len(settled))
        holes = 0
        for col in range(min(10, len(board_lines[0]))):
            found_block = False
            for row in settled:
                c = row[col] if col < len(row) else "."
                if c != ".":
                    found_block = True
                elif found_block:
                    holes += 1
        if holes:
            facts["holes"] = str(holes)
        active_chars = set()
        for row in board_lines[:settled_top]:
            active_chars.update(c for c in row if c != ".")
        if active_chars:
            facts["piece"] = "".join(sorted(active_chars))
    m = re.search(r"Next\s+Pieces?\s*:\s*([A-Z,]+)", state)
    if m:
        facts["next"] = m.group(1).strip()
    m = re.search(r"Lv\s*:\s*(\d+)", state)
    if m:
        facts["level"] = m.group(1)
    return facts


def _extract_candy_crush_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    m = re.search(r"Score\s*:\s*(\d+)", state)
    if m:
        facts["score"] = m.group(1)
    m = re.search(r"Moves\s+Left\s*:\s*(\d+)", state)
    if m:
        facts["moves"] = m.group(1)
    board_rows: List[List[str]] = []
    for line in state.split("\n"):
        m2 = re.match(r"\s*\d+\|\s*(.+)", line.strip())
        if m2:
            board_rows.append(m2.group(1).strip().split())
    if board_rows:
        nrows, ncols = len(board_rows), len(board_rows[0])
        facts["board"] = f"{nrows}x{ncols}"
        pairs = 0
        for r in range(nrows):
            for c in range(ncols - 1):
                if board_rows[r][c] == board_rows[r][c + 1]:
                    pairs += 1
        for c in range(ncols):
            for r in range(nrows - 1):
                if board_rows[r][c] == board_rows[r + 1][c]:
                    pairs += 1
        facts["pairs"] = str(pairs)
    return facts


def _extract_2048_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    d = _try_parse_dict(state)
    if d and "board" in d:
        board = d["board"]
        tiles = [v for row in board for v in row if v > 0]
        facts["highest"] = str(max(tiles)) if tiles else "0"
        facts["empty"] = str(sum(1 for row in board for v in row if v == 0))
        facts["tiles"] = ",".join(str(v) for v in sorted(tiles, reverse=True)[:5])
        merges = 0
        for r in range(len(board)):
            for c in range(len(board[r])):
                v = board[r][c]
                if v == 0:
                    continue
                if c + 1 < len(board[r]) and board[r][c + 1] == v:
                    merges += 1
                if r + 1 < len(board) and board[r + 1][c] == v:
                    merges += 1
        if merges:
            facts["merges"] = str(merges)
    else:
        m = re.search(r"highest.?tile.*?(\d+)", state, re.I)
        if m:
            facts["highest"] = m.group(1)
        m = re.search(r"(\d+)\s+empty", state, re.I)
        if m:
            facts["empty"] = m.group(1)
    return facts


def _extract_sokoban_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    boxes, docks = [], []
    worker_pos = None
    for line in state.split("\n"):
        m = re.match(
            r"\s*\d+\s*\|\s*(.+?)\s*\|\s*\((\d+),\s*(\d+)\)", line.strip()
        )
        if not m:
            continue
        item = m.group(1).strip().lower()
        pos = (int(m.group(2)), int(m.group(3)))
        if "worker" in item:
            worker_pos = pos
            if "dock" in item:
                docks.append(pos)
        elif "box" in item and "dark" not in item:
            boxes.append(pos)
            if "dock" in item:
                docks.append(pos)
        elif "dock" in item:
            docks.append(pos)
    if worker_pos:
        facts["worker"] = f"({worker_pos[0]},{worker_pos[1]})"
    if boxes:
        facts["boxes"] = str(len(boxes))
    if docks:
        on_dock = len(set(boxes) & set(docks))
        facts["solved"] = f"{on_dock}/{len(docks)}"
    return facts


def _extract_mario_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    m = re.search(r"Position\s+of\s+Mario\s*:\s*\((\d+),\s*(\d+)\)", state)
    if m:
        facts["mario"] = f"({m.group(1)},{m.group(2)})"
    for obj, key in [
        ("Question Blocks", "qblocks"),
        ("Monster Goomba", "goomba"),
        ("Monster Koopas", "koopas"),
        ("Item Mushrooms", "mushroom"),
        ("Pit", "pit"),
        ("Flag", "flag"),
        ("Warp Pipe", "pipe"),
    ]:
        m = re.search(rf"-\s*{re.escape(obj)}\s*:\s*(.+)", state)
        if m:
            val = m.group(1).strip()
            if "none" not in val.lower():
                positions = re.findall(r"\([\d,\s]+\)", val)
                if positions:
                    facts[key] = ",".join(
                        p.replace(" ", "") for p in positions[:4]
                    )
    return facts


def _extract_avalon_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    d = _try_parse_dict(state)
    text = str(list(d.values())[0]) if isinstance(d, dict) and d else state

    # Phase from the header (e.g. "=== Avalon Game — Team Voting ===")
    m = re.search(r"Avalon Game\s*[—–-]+\s*(.+?)\s*===", text)
    if m:
        facts["phase"] = m.group(1).strip().lower().replace(" ", "_")

    for pat, key in [
        (r"Current quest\s*:\s*(\d+)", "quest"),
        (r"Current round\s*:\s*(\d+)", "round"),
        (r"Your role\s*:\s*(\w+)", "role"),
        (r"Team size\s*(?:required)?\s*:\s*(\d+)", "team_size"),
    ]:
        m = re.search(pat, text)
        if m:
            facts[key] = m.group(1)

    # Side (Good/Evil) — critical for action selection
    m = re.search(r"Your role\s*:\s*\w+\s*\((\w+)\s+side\)", text)
    if m:
        facts["side"] = m.group(1).lower()

    # Quest scoreboard (e.g. "Good 2 - Evil 1")
    m = re.search(r"\(Good\s+(\d+)\s*-\s*Evil\s+(\d+)\)", text)
    if m:
        facts["score"] = f"G{m.group(1)}-E{m.group(2)}"

    # Proposed team for voting phases
    m = re.search(r"Proposed team\s*:\s*\[([^\]]+)\]", text)
    if m:
        facts["team"] = m.group(1).strip()

    # Whether the player is on the quest team
    if "You are on the quest team" in text:
        facts["on_team"] = "yes"
        m = re.search(r"on the quest team\s*:\s*\[([^\]]+)\]", text)
        if m:
            facts["team"] = m.group(1).strip()

    # Merlin's knowledge of evil players
    m = re.search(r"players\s*\[([^\]]+)\]\s*are Evil", text)
    if m:
        facts["evil_visible"] = m.group(1).strip()

    # Evil teammates visible to Assassin/Minion
    m = re.search(r"Evil teammates\s*\(visible to you\)\s*:\s*\[([^\]]+)\]", text)
    if m:
        facts["allies"] = m.group(1).strip()

    # Good-side quest vote hint
    if "As a Good player" in text:
        facts["hint"] = "good_must_pass"

    return facts


def _extract_diplomacy_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    d = _try_parse_dict(state)
    text = str(list(d.values())[0]) if isinstance(d, dict) and d else state
    for pat, key in [
        (r"Phase\s*:\s*(\S+)", "phase"),
        (r"You are\s*:\s*(\w+)", "power"),
        (r"\((\d+)\s+total\)", "centers"),
    ]:
        m = re.search(pat, text)
        if m:
            facts[key] = m.group(1)
    m = re.search(r"Your units\s*:\s*\[([^\]]+)\]", text)
    if m:
        facts["units"] = m.group(1).replace("'", "").strip()[:80]
    return facts


def _extract_generic_facts(state: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    for pat, key in [
        (r"(?:Score|score|points?)\s*[=:]\s*(\d+)", "score"),
        (r"(?:Level|Lv|level)\s*[=:]\s*(\d+)", "level"),
    ]:
        m = re.search(pat, state)
        if m:
            facts[key] = m.group(1)
    return facts


_GAME_EXTRACTORS: Dict[str, Any] = {
    "tetris": _extract_tetris_facts,
    "candy_crush": _extract_candy_crush_facts,
    "candy": _extract_candy_crush_facts,
    "twenty_forty_eight": _extract_2048_facts,
    "2048": _extract_2048_facts,
    "sokoban": _extract_sokoban_facts,
    "super_mario": _extract_mario_facts,
    "mario": _extract_mario_facts,
    "avalon": _extract_avalon_facts,
    "diplomacy": _extract_diplomacy_facts,
}


def extract_game_facts(state: str, game_name: str = "") -> Dict[str, str]:
    """Deterministically extract structured facts from raw game state text.

    Returns ``{fact_name: value_string}``.  No LLM is called.
    Supports: tetris, candy_crush, twenty_forty_eight, sokoban, super_mario,
    avalon, diplomacy.  Falls back to generic score/level extraction.
    """
    gn = game_name.lower().replace(" ", "_")
    fn = _GAME_EXTRACTORS.get(gn, _extract_generic_facts)
    try:
        return fn(state)
    except Exception:
        return _extract_generic_facts(state)


def estimate_game_phase(step_idx: int, total_steps: int) -> str:
    """Estimate game phase from step progress ratio."""
    if total_steps <= 0:
        return "unknown"
    ratio = step_idx / total_steps
    if ratio < 0.25:
        return "opening"
    if ratio < 0.65:
        return "midgame"
    return "endgame"


def build_rag_summary(
    state: str,
    game_name: str = "",
    *,
    step_idx: int = -1,
    total_steps: int = -1,
    reward: float = 0.0,
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
) -> str:
    """Build a compact ``key=value`` summary optimised for RAG embedding retrieval.

    Fully deterministic (no LLM).  Combines game-aware fact extraction with
    phase estimation and reward context.  Falls back to
    ``compact_text_observation`` when game-specific extraction is sparse.

    Example output::

        game=tetris | phase=opening | next=S,O,I,J | stack_h=2 | level=1
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)
    facts = extract_game_facts(state, game_name)

    parts: List[Tuple[str, str]] = []
    if game_name:
        parts.append(("game", game_name.replace("_", " ")))
    if step_idx >= 0 and total_steps > 0:
        if "phase" not in facts:
            parts.append(("phase", estimate_game_phase(step_idx, total_steps)))
        parts.append(("step", f"{step_idx}/{total_steps}"))
    for k, v in facts.items():
        if v:
            parts.append((k, v))
    if reward and reward != 0:
        parts.append(("reward", f"{reward:+g}"))

    result = _join_kv(parts, max_chars)

    if len(result) < 50 and state:
        budget = max_chars - len(result) - 3
        if budget > 20:
            compact = compact_text_observation(state, max_chars=budget)
            if compact:
                result = (result + " | " + compact) if result else compact

    noise_spec = os.environ.get("COSPLAY_SUMMARY_NOISE", "")
    if noise_spec:
        result = _apply_summary_noise(result, noise_spec)

    return result[:max_chars]


# ---------------------------------------------------------------------------
# R10: State-summary noise robustness hook (env-var gated, zero-impact when unset)
# ---------------------------------------------------------------------------

def _apply_summary_noise(summary: str, spec: str) -> str:
    """Perturb a ``key=value | key=value | ...`` summary string.

    Spec format (semicolon-separated; first matching token wins):
      - ``dropout:p=<float>``   — drop each kv pair with prob p
      - ``drop:<key>[,<key>...]`` — drop named kv pair(s) entirely
      - ``num_noise:<frac>``    — multiply numeric values by 1 ± U(0, frac)
      - ``shuffle``             — preserve content, scramble kv order

    Seed is taken from env var ``COSPLAY_SUMMARY_NOISE_SEED`` (default 42) so
    perturbations are reproducible across episodes within a cell.
    """
    if not summary or not spec:
        return summary
    try:
        seed_env = os.environ.get("COSPLAY_SUMMARY_NOISE_SEED", "42")
        rng = random.Random(int(seed_env) ^ (hash(summary) & 0xFFFFFFFF))
    except Exception:
        rng = random.Random(42)

    parts = [p.strip() for p in summary.split("|")]
    kvs: List[Tuple[str, str, str]] = []
    for raw in parts:
        if "=" in raw:
            k, v = raw.split("=", 1)
            kvs.append((k.strip(), v.strip(), raw))
        else:
            kvs.append(("", "", raw))

    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        if token.startswith("dropout:"):
            try:
                p = float(token.split("p=", 1)[1])
            except Exception:
                continue
            kvs = [(k, v, raw) for (k, v, raw) in kvs if not (k and rng.random() < p)]
        elif token.startswith("drop:"):
            keys = {k.strip() for k in token.split(":", 1)[1].split(",") if k.strip()}
            kvs = [(k, v, raw) for (k, v, raw) in kvs if k not in keys]
        elif token.startswith("num_noise:"):
            try:
                frac = float(token.split(":", 1)[1])
            except Exception:
                continue
            new_kvs: List[Tuple[str, str, str]] = []
            for (k, v, raw) in kvs:
                if not k or not v:
                    new_kvs.append((k, v, raw))
                    continue
                m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)", v)
                if m:
                    try:
                        num = float(m.group(1))
                        delta = rng.uniform(-frac, frac)
                        new_num = num * (1.0 + delta)
                        if "." not in v:
                            new_num = int(round(new_num))
                        new_v = str(new_num)
                        new_kvs.append((k, new_v, f"{k}={new_v}"))
                        continue
                    except Exception:
                        pass
                new_kvs.append((k, v, raw))
            kvs = new_kvs
        elif token == "shuffle":
            rng.shuffle(kvs)

    out_parts: List[str] = []
    for (k, v, raw) in kvs:
        if k and v:
            out_parts.append(f"{k}={v}")
        elif raw:
            out_parts.append(raw)
    return " | ".join(out_parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_state_summary(
    observation: str,
    structured_state: Optional[Dict[str, Any]] = None,
    *,
    max_chars: int = DEFAULT_SUMMARY_CHAR_BUDGET,
    use_llm_fallback: bool = False,
    llm_callable: Optional[Callable[..., str]] = None,
    # Legacy keyword args kept for backward compatibility
    game: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Produce a compact state summary for agent context / retrieval / memory.

    Priority order:
      1. ``structured_state`` → ``compact_structured_state()``
      2. ``observation``      → ``compact_text_observation()``
      3. LLM fallback (only if *use_llm_fallback* is True)

    The result is **never** the raw observation verbatim and always respects
    *max_chars* (capped at ``HARD_SUMMARY_CHAR_LIMIT``).

    Args:
        observation: Raw text observation from the environment.
        structured_state: Optional dict from wrapper's
            ``build_structured_state_summary()``.
        max_chars: Soft character budget (default 220).
        use_llm_fallback: If True and deterministic compression is very lossy,
            attempt an LLM-based summarisation.  Disabled by default.
        llm_callable: Custom LLM callable ``(prompt, **kw) -> str``.
            Falls back to ``ask_model`` if available.
        game: (legacy) Game hint — ignored by deterministic path.
        model: (legacy) Model name for LLM fallback.

    Returns:
        Compact summary string, always ≤ ``HARD_SUMMARY_CHAR_LIMIT`` chars.
    """
    max_chars = min(max_chars, HARD_SUMMARY_CHAR_LIMIT)

    # --- Path 1: structured state ---
    if structured_state:
        summary = compact_structured_state(structured_state, max_chars)
        if summary:
            return summary

    # --- Path 2: deterministic text compression ---
    summary = compact_text_observation(observation, max_chars)
    if summary:
        return summary

    # --- Path 3 (optional): LLM fallback ---
    if use_llm_fallback:
        _llm = llm_callable or ask_model
        if _llm is not None:
            obs_slice = (observation or "")[:3000]
            prompt = (
                "Compress this game state into a compact key=value summary. "
                f"Max {max_chars} characters. No prose. "
                "Format: key=value | key=value | ...\n\n" + obs_slice
            )
            try:
                result = _llm(prompt, model=model or DEFAULT_LLM_MODEL,
                              temperature=0.0, max_tokens=200)
                if result:
                    return _truncate_keep_important(result.strip(), max_chars)
            except Exception:
                pass

    return ""

# ---------------------------------------------------------------------------
# Think-tag stripping (for reasoning models like Qwen3, QwQ, etc.)
# ---------------------------------------------------------------------------

_THINK_COMPLETE_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from model output.

    Handles both complete and truncated (opened but never closed) blocks.
    Exported so other modules (e.g. eval scripts) can reuse it.
    """
    if not text or "<think>" not in text:
        return text
    result = _THINK_COMPLETE_RE.sub("", text)
    result = _THINK_OPEN_RE.sub("", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Intention inference
# ---------------------------------------------------------------------------

_INTENTION_TAG_RE = re.compile(r"\[(\w+)\]\s*")
_INTENTION_TAG_SET = frozenset(SUBGOAL_TAGS)
_INTENTION_TAG_ALIASES: Dict[str, str] = {
    "PLACE": "SETUP", "DROP": "EXECUTE", "MOVE": "NAVIGATE",
    "RETREAT": "DEFEND", "ADVANCE": "ATTACK", "GATHER": "COLLECT",
    "CRAFT": "BUILD", "PLAN": "SETUP", "SCORE": "ATTACK",
    "CONSOLIDATE": "POSITION", "FLEE": "SURVIVE",
}


def _normalize_intention_tag(raw: str) -> str:
    """Ensure intention has a valid ``[TAG] phrase`` format."""
    raw = raw.split("\n")[0].strip().strip('"').strip("'")
    if not raw.startswith("["):
        return f"[EXECUTE] {raw}"
    m = _INTENTION_TAG_RE.match(raw)
    if not m:
        return f"[EXECUTE] {raw}"
    tag = m.group(1).upper()
    rest = raw[m.end():].strip()
    if tag not in _INTENTION_TAG_SET:
        tag = _INTENTION_TAG_ALIASES.get(tag, "EXECUTE")
    return f"[{tag}] {rest}" if rest else f"[{tag}]"


def infer_intention(
    summary_or_observation: str,
    game: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Infer a concise ``[TAG] subgoal phrase`` from a state summary.

    Returns a tagged phrase (<=15 words) in the same format used by the
    labeling pipeline, suitable for skill-bank indexing, decision-making
    context, and trajectory annotation.

    Tags: SETUP | CLEAR | MERGE | ATTACK | DEFEND | NAVIGATE | POSITION |
          COLLECT | BUILD | SURVIVE | OPTIMIZE | EXPLORE | EXECUTE

    *context* may include: ``last_actions``, ``task``, ``progress_notes``,
    ``power_name``, ``phase``, ``sc_delta``.
    """
    if not summary_or_observation or not isinstance(summary_or_observation, str):
        return ""
    if ask_model is None:
        return ""

    ctx = context or {}
    tags_str = "|".join(SUBGOAL_TAGS)

    if game == "diplomacy":
        prompt = _build_diplomacy_intention_prompt(summary_or_observation, ctx)
    else:
        extra = ""
        if ctx.get("last_actions"):
            extra += "\nRecent actions: " + ", ".join(str(a) for a in ctx["last_actions"][-3:])
        if ctx.get("task"):
            extra += "\nTask: " + str(ctx["task"])
        prompt = (
            "State:\n" + summary_or_observation[:2000] + extra + "\n\n"
            f"Reply with ONLY a [TAG] subgoal phrase (max 12 words).\n"
            f"Tags: {tags_str}\n"
            "Examples:\n"
            "  [MERGE] Combine 4-tiles toward top-left corner\n"
            "  [CLEAR] Clear bottom rows to create space\n"
            "  [NAVIGATE] Push box onto goal tile\n"
            "  [SURVIVE] Avoid topping out by clearing lines now\n"
            "Subgoal:"
        )

    out = ask_model(prompt, model=model or DEFAULT_LLM_MODEL, temperature=0.2, max_tokens=200)
    if not out or out.startswith("Error"):
        return ""

    cleaned = strip_think_tags(out)
    if not cleaned:
        return ""

    cleaned = cleaned.split("\n")[0].strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"^[A-Z]{3,}:\s*", "", cleaned)

    words = cleaned.split()
    if len(words) > 15:
        cleaned = " ".join(words[:15])

    return _normalize_intention_tag(cleaned)[:150]


def _build_diplomacy_intention_prompt(
    summary: str,
    ctx: Dict[str, Any],
) -> str:
    """Build a Diplomacy-specific intention prompt anchored to one power."""
    power = ctx.get("power_name", "")
    phase = ctx.get("phase", "")
    sc_delta = ctx.get("sc_delta", "")

    parts = [f"You are {power} in Diplomacy." if power else "You are playing Diplomacy."]
    if phase:
        parts.append(f"Phase: {phase}.")
    parts.append(f"\nSituation:\n{summary[:1500]}")

    if sc_delta:
        parts.append(f"\nRecent SC changes: {sc_delta}")

    if ctx.get("last_actions"):
        recent = ctx["last_actions"][-3:]
        parts.append("\nRecent orders: " + " | ".join(str(a) for a in recent))

    tags_str = "|".join(SUBGOAL_TAGS)
    parts.append(
        f"\n\nReply with ONLY a [TAG] subgoal phrase (max 12 words) as {power or 'this power'}.\n"
        f"Tags: {tags_str}\n"
        "Focus on territorial objectives, not descriptions of the board.\n"
        "Examples:\n"
        "  [ATTACK] Capture SER and RUM via BUL support\n"
        "  [DEFEND] Hold BUD against Russian advance from GAL\n"
        "  [NAVIGATE] Convoy army to TUN via ION\n"
        "  [POSITION] Ally with France against Germany\n"
        "Subgoal:"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Episodic memory store (for query_memory)
# ---------------------------------------------------------------------------

class EpisodicMemoryStore:
    """Episodic memory with RAG-embedding retrieval (cosine similarity) and
    keyword-overlap fallback.

    When an ``embedder`` is provided (a ``TextEmbedderBase`` from
    ``rag.embedding``), every memory is embedded on ``add`` and queries are
    scored via cosine similarity.  The final score is a weighted mix of
    embedding similarity and keyword overlap so the system degrades
    gracefully if the embedding model is unavailable.

    When no embedder is provided, behaviour is identical to the original
    keyword-overlap-only store.
    """

    def __init__(
        self,
        max_entries: int = 500,
        embedder: Any = None,
        embedding_weight: float = 0.7,
    ) -> None:
        """
        Args:
            max_entries: Maximum number of memories to keep (FIFO eviction).
            embedder: Optional ``TextEmbedderBase`` (e.g. from
                ``rag.get_text_embedder()``).  Enables embedding retrieval.
            embedding_weight: Blend weight for embedding vs keyword score
                (0 = keyword only, 1 = embedding only).
        """
        self._entries: List[Dict[str, Any]] = []
        self._max_entries = max_entries
        self._embedder = embedder
        self._embedding_weight = embedding_weight
        self._memory_store: Any = None
        if embedder is not None:
            self._init_memory_store(embedder)

    def _init_memory_store(self, embedder: Any) -> None:
        try:
            from rag.retrieval import MemoryStore
            self._memory_store = MemoryStore(embedder=embedder, top_k=self._max_entries)
        except ImportError:
            self._memory_store = None

    def add(
        self,
        key: str,
        summary: str,
        action: Any = None,
        outcome: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add one memory entry and embed it if an embedder is available."""
        entry = {
            "key": key,
            "summary": summary,
            "action": action,
            "outcome": outcome,
            **(extra or {}),
        }
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        if self._memory_store is not None:
            text = (key + " " + summary).strip()
            if text:
                try:
                    self._memory_store.add_texts([text], payloads=[entry])
                except Exception:
                    pass

    def add_experience(self, state_summary: str, action: Any, next_state_summary: str, done: bool) -> None:
        """Convenience: add from a single experience."""
        key = state_summary[:200] if state_summary else ""
        self.add(key=key, summary=state_summary, action=action, outcome=next_state_summary, extra={"done": done})

    def query(self, query_key: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by embedding similarity + keyword overlap.

        If an embedder is available, scores are a weighted blend of cosine
        similarity and keyword overlap.  Otherwise falls back to keyword only.
        """
        if not query_key or not self._entries:
            return []

        keyword_scores = self._keyword_scores(query_key)

        if self._memory_store is not None and len(self._memory_store) > 0:
            try:
                ranked = self._memory_store.rank(query_key, k=len(self._entries))
                emb_scores = {idx: score for idx, score, _ in ranked}
            except Exception:
                emb_scores = {}
        else:
            emb_scores = {}

        w = self._embedding_weight if emb_scores else 0.0
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for i, entry in enumerate(self._entries):
            kw = keyword_scores[i]
            emb = emb_scores.get(i, 0.0)
            combined = w * emb + (1.0 - w) * kw
            scored.append((combined, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    _KW_SPLIT_RE = re.compile(r"[\s=|:,;/]+")

    def _keyword_scores(self, query_key: str) -> List[float]:
        q_lower = query_key.lower()
        q_words = set(w for w in self._KW_SPLIT_RE.split(q_lower) if len(w) >= 2)
        scores: List[float] = []
        for e in self._entries:
            text = (e.get("key", "") + " " + e.get("summary", "")).lower()
            t_words = set(w for w in self._KW_SPLIT_RE.split(text) if len(w) >= 2)
            overlap = len(q_words & t_words) / max(len(q_words), 1)
            scores.append(overlap)
        return scores

    @property
    def has_embedder(self) -> bool:
        return self._memory_store is not None

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Skill bank formatting for agent prompt
# ---------------------------------------------------------------------------

def skill_bank_to_text(skill_bank: Any) -> str:
    """Format skill bank as selectable options for the agent prompt.

    Each skill shows its name, what it does, when to use it, and confidence.
    Designed so the agent can pick a skill via select_skill by referencing
    the description.
    """
    if skill_bank is None:
        return "(no skill bank — act without skills)"

    bank = getattr(skill_bank, "bank", skill_bank)

    try:
        ids = list(bank.skill_ids)[:50]
    except AttributeError:
        return "(no skill bank)"
    if not ids:
        return "(empty skill bank — no learned skills yet)"

    has_get_skill = hasattr(bank, "get_skill")

    lines = [f"{len(ids)} skills available (use select_skill to activate one):"]
    for idx, sid in enumerate(ids[:15]):
        try:
            skill = bank.get_skill(sid) if has_get_skill else None
            if skill is not None and (skill.strategic_description or skill.protocol.steps):
                name = skill.name or sid
                desc = skill.strategic_description[:90] if skill.strategic_description else ""
                confidence = f"{skill.confidence:.0%}" if skill.n_instances > 0 else "?"
                dur = skill.protocol.expected_duration
                parts = [f"[{sid}] {name}"]
                if desc:
                    parts[0] += f" — {desc}"
                parts.append(f"    confidence={confidence}, ~{dur} steps")
                if skill.protocol.preconditions:
                    parts.append(f"    when: {'; '.join(skill.protocol.preconditions[:2])}")
                eh = getattr(skill, "execution_hint", None)
                if eh is not None and eh.termination_cues:
                    parts.append(f"    done: {'; '.join(eh.termination_cues[:2])}")
                elif skill.protocol.success_criteria:
                    parts.append(f"    done: {'; '.join(skill.protocol.success_criteria[:2])}")
                lines.append("\n".join(parts))
            else:
                c = bank.get_contract(sid)
                if c is not None:
                    add = getattr(c, "eff_add", set()) or set()
                    dele = getattr(c, "eff_del", set()) or set()
                    name = getattr(c, "name", None) or sid
                    effects = []
                    if add:
                        effects.append("+" + ", +".join(sorted(add)[:3]))
                    if dele:
                        effects.append("-" + ", -".join(sorted(dele)[:2]))
                    eff_str = "; ".join(effects) if effects else "?"
                    r = bank.get_report(sid) if hasattr(bank, "get_report") else None
                    pr = f" pass={r.overall_pass_rate:.0%}" if r else ""
                    lines.append(f"  [{sid}] {name}: effects=[{eff_str}]{pr}")
                else:
                    lines.append(f"  [{sid}]")
        except Exception:
            lines.append(f"  [{sid}]")
    return "\n".join(lines)


_KW_TOKENIZE_RE = re.compile(r"[\s=|:,;/\(\)\[\]\{\}\"']+")


def _tokenize_for_matching(text: str) -> List[str]:
    """Split text into lowercase tokens for keyword matching."""
    return [w for w in _KW_TOKENIZE_RE.split(text.lower()) if len(w) >= 2]


def _rank_skills_by_relevance(
    bank: Any,
    ids: List[str],
    query: str,
    has_get_skill: bool,
) -> Optional[str]:
    """Score each skill against *query* using TF-IDF-style keyword overlap.

    Builds a document per skill from its name, description, tags, and
    preconditions, then computes an IDF-weighted overlap score against the
    query tokens.  Returns the best skill_id, or ``ids[0]`` only when all
    scores are zero (should not happen for a well-populated bank).
    """
    import math

    query_tokens = _tokenize_for_matching(query)
    if not query_tokens:
        return ids[0] if ids else None

    skill_docs: List[List[str]] = []
    for sid in ids:
        parts = [sid]
        if has_get_skill:
            skill_obj = bank.get_skill(sid)
            if skill_obj:
                if skill_obj.name:
                    parts.append(skill_obj.name)
                if skill_obj.strategic_description:
                    parts.append(skill_obj.strategic_description)
                if getattr(skill_obj, "tags", None):
                    parts.extend(skill_obj.tags)
                proto = skill_obj.protocol
                if proto:
                    if proto.preconditions:
                        parts.extend(proto.preconditions[:3])
                    if proto.success_criteria:
                        parts.extend(proto.success_criteria[:2])
        skill_docs.append(_tokenize_for_matching(" ".join(parts)))

    n_docs = len(skill_docs)
    df: Dict[str, int] = {}
    for doc in skill_docs:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1

    scores: List[float] = []
    query_set = set(query_tokens)
    for doc_tokens in skill_docs:
        doc_set = set(doc_tokens)
        overlap = query_set & doc_set
        if not overlap:
            scores.append(0.0)
            continue
        score = sum(
            math.log(1 + n_docs / (df.get(tok, 1)))
            for tok in overlap
        )
        scores.append(score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_idx] > 0:
        return ids[best_idx]
    return ids[0] if ids else None


def select_skill_from_bank(
    skill_bank: Any,
    key: str,
    current_state: Optional[Dict[str, Any]] = None,
    memory: Optional[Any] = None,
    top_k: int = 1,
) -> Dict[str, Any]:
    """Select a skill from the bank with full structured guidance.

    This is the single protocol to query prior rollouts, experience, and
    skill plans.  Uses state-aware selection when current_state is provided.

    Returns a structured guidance dict with:
    - skill_id, skill_name, why_selected
    - protocol (steps, preconditions, success_criteria, abort_criteria)
    - execution_hint, termination_hint, failure_modes
    - expected_effects, applicability_score, confidence
    - micro_plan (fallback action sequence)
    - relevant_memory (from episodic memory, if available)
    """
    empty_result: Dict[str, Any] = {
        "skill_id": None,
        "skill_name": "",
        "why_selected": "no skills available",
        "protocol": {},
        "micro_plan": [],
        "execution_hint": "",
        "termination_hint": "",
        "failure_modes": [],
        "expected_effects": [],
        "relevant_memory": [],
    }

    if skill_bank is None:
        return empty_result

    # ── Try SkillQueryEngine.select() — the richest path (relevance + applicability) ──
    if hasattr(skill_bank, "select"):
        try:
            results = skill_bank.select(
                key,
                current_state=current_state,
                current_predicates=current_state,
                top_k=top_k,
            )
            if results:
                best = results[0]
                best_dict = best.to_dict() if hasattr(best, "to_dict") else dict(best)
                sid = best_dict.get("skill_id")
                if sid:
                    protocol = _get_protocol_for_skill(skill_bank, sid)
                    best_dict["protocol"] = protocol or best_dict.get("protocol", {})
                    best_dict.setdefault("relevant_memory", _query_memory_for_skill(memory, key))
                    return best_dict
        except Exception:
            pass

    # ── Try SkillQueryEngine.query_for_decision_agent() ──
    if hasattr(skill_bank, "query_for_decision_agent"):
        try:
            result = skill_bank.query_for_decision_agent(
                key,
                current_state=current_state,
                current_predicates=current_state,
                top_k=top_k,
            )
            sid = result.get("skill_id")
            if sid:
                protocol = _get_protocol_for_skill(skill_bank, sid)
                result["protocol"] = protocol or result.get("protocol", {})
                result.setdefault("relevant_memory", _query_memory_for_skill(memory, key))
                return result
        except Exception:
            pass

    # ── Try SkillBankAgent.select_skill() ──
    if hasattr(skill_bank, "select_skill"):
        try:
            results = skill_bank.select_skill(
                key,
                current_state=current_state,
                current_predicates=current_state,
                top_k=top_k,
            )
            if results:
                best = results[0]
                protocol = _get_protocol_for_skill(skill_bank, best.get("skill_id"))
                best["protocol"] = protocol or best.get("protocol", {})
                best.setdefault("relevant_memory", _query_memory_for_skill(memory, key))
                return best
        except Exception:
            pass

    # ── Fallback: plain SkillBankMVP — TF-IDF keyword scoring ──
    bank = getattr(skill_bank, "bank", skill_bank)
    try:
        ids = list(bank.skill_ids)
    except AttributeError:
        return empty_result

    if not ids:
        return empty_result

    has_get_skill = hasattr(bank, "get_skill")
    skill_id = _rank_skills_by_relevance(bank, ids, key, has_get_skill)

    if not skill_id:
        return empty_result

    protocol = _get_protocol_for_skill(bank, skill_id)
    skill_obj = bank.get_skill(skill_id) if has_get_skill else None
    skill_name = ""
    execution_hint = ""
    if skill_obj:
        skill_name = skill_obj.name or skill_id
        execution_hint = skill_obj.strategic_description or ""
        eh = getattr(skill_obj, "execution_hint", None)
        if eh is not None and eh.execution_description:
            execution_hint = eh.execution_description

    if protocol and protocol.get("steps"):
        steps = [{"action": s} for s in protocol["steps"][:7]]
    else:
        c = bank.get_contract(skill_id) if hasattr(bank, "get_contract") else None
        if c:
            add_set = getattr(c, "eff_add", set()) or set()
            steps = [{"action": None, "effect": lit} for lit in sorted(add_set)[:5]]
        else:
            steps = [{"action": "proceed"}]

    return {
        "skill_id": skill_id,
        "skill_name": skill_name,
        "why_selected": f"keyword-scored match for state",
        "protocol": protocol,
        "micro_plan": steps,
        "execution_hint": execution_hint,
        "termination_hint": "",
        "failure_modes": [],
        "expected_effects": [],
        "relevant_memory": _query_memory_for_skill(memory, key),
    }


def _query_memory_for_skill(memory: Optional[Any], key: str) -> List[Dict[str, Any]]:
    """Query episodic memory for experiences relevant to the skill key."""
    if memory is None or not key:
        return []
    try:
        return memory.query(key, k=2)
    except Exception:
        return []


def _get_protocol_for_skill(skill_bank: Any, skill_id: Optional[str]) -> Dict[str, Any]:
    """Extract protocol dict from a skill bank for a given skill_id."""
    if not skill_id:
        return {}
    bank = getattr(skill_bank, "bank", skill_bank)
    if hasattr(bank, "get_skill"):
        skill = bank.get_skill(skill_id)
        if skill is not None and skill.protocol.steps:
            return skill.protocol.to_dict()
    return {}


# Backward compatibility alias
query_skill_bank = select_skill_from_bank
