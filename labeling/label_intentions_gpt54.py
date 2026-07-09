#!/usr/bin/env python
"""Label cold-start episodes with an ``(intention_tag, intention_note)``
pair derived from the per-step ``(schema, action)`` view of each
``Experience``.

Why a dedicated step-level intention labeler
--------------------------------------------

Both cold-start corpora (``Cold-start-out`` for env_wrappers, and
``Cold-start-out-gymv`` for gym-v Temporal ROMs) record a free-form
English ``intentions`` string per step — written by the cold-start
actor as part of its action-selection reasoning. That string lacks
the bracketed ``[TAG]`` prefix the skill-bank segmenter expects, so
``parse_intention_tag`` returns ``UNKNOWN`` and the Stage-2
intention-fit term collapses to zero (see
``labeling/extract_skillbank_gymv_gpt54.py`` for the full diagnosis).

This driver fixes that **at the data layer**: it reads each cold-start
episode JSON, extracts the schema (``metadata.schema``) and the chosen
action for every step, asks gpt-5.4 to categorise the step into one
canonical operator, and writes a labelled copy of the episode where:

* ``Experience.intention_tag``  ← e.g. ``"COMMIT"``
* ``Experience.intention_note`` ← ≤15-word concrete reason
* ``Experience.intentions``     ← rewritten to ``"[TAG] note"``
  (so the existing skill-bank pipeline picks it up natively)
* ``Experience.raw_intentions`` ← preserved original natural-language

Vocabularies — dual-axis (current canonical scheme)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every step is labelled with **two orthogonal tags**:

* ``operator`` ∈ :data:`INTENT_OPERATORS`
  ``INSPECT / TRACK / COMPARE / COMMIT / VERIFY / RECOVER``
  The agent's *cognitive mode* — what it is doing with attention.
  Domain-agnostic, transferable across every game.  Future-aligned with
  the two-level MDP inner-hop alphabet
  (``plans/02-action-agent/PLAN-ACTION-AGENT.md §5.3``).

* ``subgoal`` ∈ :data:`UNIFIED_SUBGOALS`
  ``SETUP / NAVIGATE / POSITION / CLEAR / MERGE / COLLECT / BUILD /
  ATTACK / DEFEND / EVADE / OPTIMIZE / SURVIVE / EXPLORE / EXECUTE``
  The agent's *game-level achievement* — what concrete domain goal is
  being pursued this step.  Anchors skill discovery to game state.

The two are written into ``Experience.intentions`` as
``"[OPERATOR/SUBGOAL] note"`` (e.g.
``"[COMMIT/EVADE] sidestep left to avoid bullets"``).  Both
:func:`skill_agents.boundary_proposal.signal_extractors.parse_intention_tag`
and the new :func:`parse_intention_tags` helper accept this shape and
also fall back gracefully on legacy single-tag intentions.

Corpus auto-detection
~~~~~~~~~~~~~~~~~~~~~

The corpus is inferred from the input layout:

* ``Cold-start-out-gymv/<run>/Temporal_<Title>-v0/episode_*.json``
  → ``corpus="gym_v"`` → INTENT_OPERATORS vocab.
* ``Cold-start-out/<run>/<group>/<game>/episode_*.json``
  → ``corpus="env_wrappers"`` → SUBGOAL_TAGS vocab.

Pass ``--corpus {gym_v,env_wrappers,auto}`` to override.

Per-step fields written into each ``Experience`` dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Experience.intentions``        ← ``"[OPERATOR/SUBGOAL] note"``
  (drop-in for the existing skill-bank pipeline; ``parse_intention_tag``
  returns the operator, ``parse_intention_tags`` returns both)
* ``Experience.intention_tag``     ← operator (primary, cross-domain)
* ``Experience.intention_subgoal`` ← subgoal  (domain anchor)
* ``Experience.intention_note``    ← ≤15-word concrete reason
* ``Experience.raw_intentions``    ← preserved original natural-language

Layout produced
~~~~~~~~~~~~~~~

::

    labeling/intentions_out/<run>/
    ├── gym_v/
    │   └── Temporal_Airstriker-v0/
    │       ├── episode_000.json     (full original + new fields)
    │       └── ...
    ├── env_wrappers/
    │   └── tetris/
    │       ├── episode_000.json
    │       └── ...
    └── _intentions_summary.json     (per-env tag distribution + costs)

Usage
~~~~~

    python labeling/label_intentions_gpt54.py \\
        --gymv_input  Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_20260429_080127 \\
        --envw_input  Cold-start-out/sft_envw_e20_gpt5p4_20260429_080916 \\
        --output_dir  labeling/intentions_out/run_20260429 \\
        --workers 8

    # Single env quick test
    python labeling/label_intentions_gpt54.py \\
        --gymv_input Cold-start-out-gymv/.../Temporal_Airstriker-v0 \\
        --output_dir labeling/intentions_out/airstriker_test \\
        --max_episodes 1 --workers 4 --dry_run

The companion bash dispatcher ``run_label_intentions.sh`` fans this
out one worker per env/game across both corpora.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — mirror sibling drivers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# Bootstrap api_keys.py from the workspace root.
try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key  # type: ignore
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key  # type: ignore
except Exception:  # pragma: no cover - missing key file just falls through
    pass

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from decision_agents.agent_helper import (
    INTENT_OPERATORS,
    OPERATOR_TO_SUBGOAL,
    SUBGOAL_TAGS,
    SUBGOAL_TO_OPERATOR,
    UNIFIED_SUBGOALS,
    strip_think_tags,
)
from labeling.extract_skillbank_gymv_gpt54 import (
    _classify_intent_operator,
    _compact_summary_from_schema,
    _pick_schema,
    _strip_noisy_inline_kvs,
)

try:
    from API_func import ask_model  # type: ignore
except ImportError:  # pragma: no cover - exercised in unit tests with mocks
    ask_model = None

logger = logging.getLogger("labeling.label_intentions")

DEFAULT_MODEL = "gpt-5.4"
INTENT_NOTE_WORD_BUDGET = 15
SCHEMA_CHAR_BUDGET = 700
DELTA_MAX_CHANGES = 5
LLM_MAX_TOKENS = 120
LLM_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Corpus inference
# ---------------------------------------------------------------------------

def _infer_corpus_from_path(path: Path) -> str:
    """Return ``"gym_v"`` or ``"env_wrappers"`` based on a folder/file path.

    Looks for a ``Temporal_*`` segment anywhere in the resolved path
    (gym-v env folders all start with ``Temporal_``); everything else
    is treated as env_wrappers.
    """
    parts = [p for p in str(path.resolve()).split(os.sep) if p]
    for part in parts:
        if part.startswith("Temporal_") or "Cold-start-out-gymv" in part:
            return "gym_v"
    return "env_wrappers"


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Source:
    corpus: str          # "gym_v" | "env_wrappers"
    bucket: str          # env name (gym-v) or game name (env_wrappers)
    files: Tuple[Path, ...]
    base_dir: Path       # the top-level corpus root


def _find_gymv_sources(input_dir: Path) -> List[_Source]:
    out: List[_Source] = []
    if not input_dir.exists():
        return out
    for env_dir in sorted(input_dir.iterdir()):
        if not env_dir.is_dir() or not env_dir.name.startswith("Temporal_"):
            continue
        eps = tuple(sorted(
            f for f in env_dir.glob("episode_[0-9]*.json")
            if f.name != "episode_buffer.json"
        ))
        if eps:
            out.append(_Source(
                corpus="gym_v", bucket=env_dir.name,
                files=eps, base_dir=input_dir,
            ))
    return out


def _find_envw_sources(input_dir: Path) -> List[_Source]:
    """env_wrappers cold-start runs are nested ``<group>/<game>/episode_*.json``."""
    out: List[_Source] = []
    if not input_dir.exists():
        return out
    for group_dir in sorted(input_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        for game_dir in sorted(group_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            eps = tuple(sorted(
                f for f in game_dir.glob("episode_[0-9]*.json")
                if f.name != "episode_buffer.json"
            ))
            if eps:
                out.append(_Source(
                    corpus="env_wrappers", bucket=game_dir.name,
                    files=eps, base_dir=input_dir,
                ))
    return out


# ---------------------------------------------------------------------------
# Schema / delta helpers
# ---------------------------------------------------------------------------

def _select_schema_text(exp: Dict[str, Any]) -> str:
    """Prefer the VLM ``<state>`` block; fall back to the textual ``state``."""
    md = exp.get("metadata") or {}
    s = _pick_schema(md)
    if s:
        return s
    state = exp.get("state")
    if isinstance(state, str) and state.strip():
        return state
    return ""


def _kv_pairs_from_summary(summary: str) -> Dict[str, str]:
    """Parse a ``key=value | key=value`` summary string into a dict.

    Mirrors the loose parsing used by the env_wrappers labeler so we
    can compute compact deltas for both corpora identically.
    """
    out: Dict[str, str] = {}
    if not summary:
        return out
    for seg in summary.split("|"):
        seg = seg.strip()
        if not seg or "=" not in seg:
            continue
        k, _, v = seg.partition("=")
        k = k.strip().lower()
        v = v.strip()
        if k and v and len(k) <= 32:
            out[k] = v[:80]
    return out


def _compute_delta(prev_summary: str, curr_summary: str) -> str:
    """Return up to N changes between two summary strings (compact)."""
    if not prev_summary or not curr_summary:
        return ""
    prev, curr = _kv_pairs_from_summary(prev_summary), _kv_pairs_from_summary(curr_summary)
    skip = {"step", "domain", "task", "goal"}
    changes: List[str] = []
    for k, v in curr.items():
        if k in skip:
            continue
        pv = prev.get(k)
        if pv is not None and pv != v:
            changes.append(f"{k}:{pv}->{v}")
        if len(changes) >= DELTA_MAX_CHANGES:
            break
    return ", ".join(changes)


# ---------------------------------------------------------------------------
# Prompt construction — DUAL-AXIS (operator + subgoal)
# ---------------------------------------------------------------------------
#
# Calibration philosophy
# ~~~~~~~~~~~~~~~~~~~~~~
# Across a diverse mix of games we expect roughly:
#
# * operator: COMMIT 30–45 % | VERIFY 5–15 % | RECOVER 5–15 %
#             TRACK 5–15 %  | INSPECT 5–15 % | COMPARE 5–10 %
# * subgoal:  largely flat across the 14-tag set; long-tail tags such as
#             EXPLORE / BUILD will be rare for puzzle games but should
#             show up in exploration-heavy gym-v environments.
#
# The decision rules below explicitly bound VERIFY usage so the model
# does NOT collapse to "previous X failed → VERIFY" the way the
# single-axis labeller did (84 % VERIFY in the first run).
# ---------------------------------------------------------------------------

# Operator definitions — written in cognitive-mode terms so the model
# decides on attention rather than action shape.
_OPERATOR_DEFINITIONS: List[Tuple[str, str]] = [
    ("INSPECT",
     "PARSE the scene / orient yourself: title screen, menu, transition, "
     "first encounter, exploring controls. Use when a concrete plan does "
     "NOT yet exist."),
    ("TRACK",
     "WAIT / FOLLOW a change the agent does not control: NOOP, idle, "
     "opponent animating, falling piece descending, page loading. The "
     "agent's role is observer this step."),
    ("COMPARE",
     "WEIGH multiple candidate moves before committing: 'A vs B', 'either "
     "left or right', 'two valid columns', 'consider whether'."),
    ("COMMIT",
     "EXECUTE a *new* directional plan with goal-progressing intent. This "
     "is the default for any step where the agent decides on a concrete "
     "action to advance state — INCLUDING re-trying a different action "
     "after a previous one failed (that is a NEW commit, not a verify)."),
    ("VERIFY",
     "ONLY observe the outcome of an already-completed prior action with "
     "NO new directional intent: 'idle one frame to confirm menu cleared', "
     "'NOOP to let the score tick', 'press the same button again to "
     "confirm registration'. Reserve VERIFY for steps with no fresh "
     "directional decision — if the agent chooses a NEW direction "
     "(left vs right, A vs B), that is COMMIT or COMPARE, not VERIFY."),
    ("RECOVER",
     "REACT defensively to surprise / failure / threat: dodge incoming "
     "shot, sidestep enemy, retreat from low health, undo a mistake, "
     "panic-block."),
]

# Unified subgoal definitions — each line is a one-clause game-state
# anchor, not a button-pressed action.  These are domain-anchors that
# pair with the operator to form the dual-axis label.
_SUBGOAL_DEFINITIONS: List[Tuple[str, str]] = [
    ("SETUP",    "opening / menu / pre-game / get-ready / piece prep / configuring"),
    ("NAVIGATE", "locomotion toward a target / destination, no immediate threat"),
    ("POSITION", "alignment / staging / fine spatial adjustment for a future commit"),
    ("CLEAR",    "remove obstacles, lines, matches, blockers"),
    ("MERGE",    "combine same-value units into a larger unit (2048, fusion, combo)"),
    ("COLLECT",  "pick up an item / coin / power-up / orb"),
    ("BUILD",    "construct / extend / craft a longer-term structure"),
    ("ATTACK",   "offensive action toward an enemy or objective"),
    ("DEFEND",   "block / shield / guard / parry in place"),
    ("EVADE",    "dodge / sidestep / outrun an incoming threat"),
    ("OPTIMIZE", "improve current configuration without committing the goal move"),
    ("SURVIVE",  "avoid imminent failure: low HP, topping out, time critical"),
    ("EXPLORE",  "probe unknown state / new region / open up unseen options"),
    ("EXECUTE",  "primitive low-level action — catch-all when nothing else fits"),
]


def _build_vocab_block() -> str:
    op_lines = [
        f"- {tag:<8} — {desc}" for tag, desc in _OPERATOR_DEFINITIONS
    ]
    sg_lines = [
        f"- {tag:<8} — {desc}" for tag, desc in _SUBGOAL_DEFINITIONS
    ]
    return (
        "OPERATORS (cognitive mode — what is the agent doing with attention?):\n"
        + "\n".join(op_lines)
        + "\n\n"
        + "SUBGOALS (game-level achievement — what is being attempted?):\n"
        + "\n".join(sg_lines)
    )


def _truncate_schema(schema_text: str, *, char_budget: int = SCHEMA_CHAR_BUDGET) -> str:
    """Strip XML wrappers + trim to keep the prompt budget tight."""
    if not schema_text:
        return ""
    s = schema_text.strip()
    if s.startswith("<state>"):
        s = s[len("<state>"):]
    if s.endswith("</state>"):
        s = s[: -len("</state>")]
    s = s.strip()
    if len(s) <= char_budget:
        return s
    head = s[: char_budget - 80]
    return head.rstrip() + "\n... <truncated> ..."


_SYSTEM_PROMPT = (
    "You categorise ONE decision step in a game-playing trajectory along "
    "TWO independent axes:\n"
    "  - operator (cognitive mode, 6 choices)\n"
    "  - subgoal  (game-level achievement, 14 choices)\n"
    "The two axes are orthogonal: operator says HOW the agent is "
    "thinking, subgoal says WHAT goal is being pursued. The same primitive "
    "button press can map to very different (operator, subgoal) pairs "
    "depending on the surrounding context.\n"
    "Reply ONLY a JSON object with three keys: operator, subgoal, note."
)


# Few-shot examples — chosen to demonstrate the orthogonality of the two
# axes and to break the COMMIT-everywhere / VERIFY-everywhere collapse.
# Cover both gym-v action games and env_wrappers puzzle games in the
# same prompt so the model sees one unified vocabulary.
_FEWSHOT: List[Dict[str, str]] = [
    {  # gym-v opening menu
        "schema": "task=Temporal/Airstriker-v0 | step=0 | scene=get ready overlay",
        "action": "START",
        "delta": "",
        "reasoning": "Get ready screen at level 1, gameplay hasn't started.",
        "operator": "INSPECT",
        "subgoal":  "SETUP",
        "note": "Press start to leave the get-ready overlay.",
    },
    {  # NOOP wait — pure observer
        "schema": "task=Temporal/Columns-v0 | step=12 | falling column descending in left well",
        "action": "NOOP",
        "delta": "column_y:5->7",
        "reasoning": "Wait for the falling stack to land before next move.",
        "operator": "TRACK",
        "subgoal":  "POSITION",
        "note": "Wait one frame for the falling column to settle.",
    },
    {  # gym-v fresh attack — directional COMMIT, NOT verify
        "schema": "task=Temporal/Airstriker-v0 | step=24 | enemy in lane",
        "action": "LEFT",
        "delta": "",
        "reasoning": "Recent B had no effect, last RIGHT also did nothing — try LEFT.",
        "operator": "COMMIT",
        "subgoal":  "EXPLORE",
        "note": "Try LEFT after B and RIGHT failed — new directional probe.",
    },
    {  # NOOP confirm — true VERIFY
        "schema": "task=Temporal/Columns-v0 | step=8 | menu transition mid-fade",
        "action": "NOOP",
        "delta": "scene:menu->play",
        "reasoning": "Press start was issued last frame; idle to confirm transition.",
        "operator": "VERIFY",
        "subgoal":  "SETUP",
        "note": "Idle to confirm the menu transition completed.",
    },
    {  # incoming threat — RECOVER + EVADE
        "schema": "task=Temporal/Airstriker-v0 | step=44 | bullet cluster at center",
        "action": "RIGHT",
        "delta": "",
        "reasoning": "Two bullets descending in center lane; sidestep to right.",
        "operator": "RECOVER",
        "subgoal":  "EVADE",
        "note": "Sidestep right to dodge incoming center-lane bullets.",
    },
    {  # gym-v compare two valid options
        "schema": "task=Temporal/Columns-v0 | step=18 | two valid placement columns",
        "action": "RIGHT",
        "delta": "",
        "reasoning": "Either left well for color match or right well for height — right gives match.",
        "operator": "COMPARE",
        "subgoal":  "POSITION",
        "note": "Choose right well over left for immediate color match.",
    },
    {  # tetris piece prep
        "schema": "game=tetris | step=3 | stack_h=2 holes=0 piece=I",
        "action": "left col0",
        "delta": "",
        "reasoning": "Set up flat surface for line build.",
        "operator": "COMMIT",
        "subgoal":  "SETUP",
        "note": "Place I flat on left to keep surface even.",
    },
    {  # tetris immediate clear
        "schema": "game=tetris | step=20 | stack_h=10 holes=2 piece=L",
        "action": "L-rot1 col5 (line!)",
        "delta": "lines:0->1",
        "reasoning": "Drop L to fill the gap and clear a line.",
        "operator": "COMMIT",
        "subgoal":  "CLEAR",
        "note": "Drop L into gap — triggers a line clear.",
    },
    {  # 2048 merge
        "schema": "game=2048 | step=15 | empty=4 max=128",
        "action": "left",
        "delta": "max:128->256",
        "reasoning": "Merge 128 tiles to consolidate into 256.",
        "operator": "COMMIT",
        "subgoal":  "MERGE",
        "note": "Slide left to merge two 128 tiles into 256.",
    },
    {  # tetris emergency
        "schema": "game=tetris | step=72 | stack_h=15 holes=28 piece=S",
        "action": "S-rot1 col8",
        "delta": "stack_h:14->15",
        "reasoning": "Stack near ceiling, holes=28; cannot risk topping out.",
        "operator": "RECOVER",
        "subgoal":  "SURVIVE",
        "note": "Stack near top — place S vertically to avoid overhang.",
    },
    {  # mario navigate around enemy
        "schema": "game=super_mario | step=22 | mario=(5,11) goomba=(8,11)",
        "action": "right",
        "delta": "mario_x:5->6",
        "reasoning": "Goomba ahead; close distance to plan a stomp.",
        "operator": "COMMIT",
        "subgoal":  "NAVIGATE",
        "note": "Walk right to approach Goomba within stomp range.",
    },
    {  # candy_crush match
        "schema": "game=candy_crush | step=8 | board=8x8 pairs=4",
        "action": "swap (3,4)<->(4,4)",
        "delta": "score:0->90",
        "reasoning": "Form a horizontal triple to clear the row.",
        "operator": "COMMIT",
        "subgoal":  "CLEAR",
        "note": "Swap to form a horizontal triple of red candies.",
    },
]


def _format_fewshot_block(examples: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for i, ex in enumerate(examples):
        lines.append(f"Example {i+1}:")
        lines.append(f"  schema   : {ex['schema']}")
        lines.append(f"  action   : {ex['action']}")
        if ex.get("delta"):
            lines.append(f"  delta    : {ex['delta']}")
        if ex.get("reasoning"):
            lines.append(f"  reasoning: {ex['reasoning']}")
        lines.append(
            "  output   : "
            f'{{"operator":"{ex["operator"]}",'
            f'"subgoal":"{ex["subgoal"]}",'
            f'"note":"{ex["note"]}"}}'
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_user_prompt(
    *,
    corpus: str,                 # kept for back-compat; unused in the prompt
    schema_text: str,
    action: str,
    delta: str,
    raw_reasoning: str,
    prev_tag: str = "",          # operator hint from prior step
    prev_subgoal: str = "",      # subgoal  hint from prior step
    prev_note: str = "",
) -> str:
    raw_reasoning = (raw_reasoning or "").strip()
    if len(raw_reasoning) > 360:
        raw_reasoning = raw_reasoning[:360].rsplit(" ", 1)[0] + "..."

    parts: List[str] = [
        _build_vocab_block(),
        "",
        "DECISION RULES (apply in order; orthogonal — operator AND subgoal):",
        "  1. THREAT: schema shows incoming projectile / low health / topping out / time-critical:",
        "       operator = RECOVER",
        "       subgoal  = EVADE  (dodge/sidestep), DEFEND (block in place), or SURVIVE (existential).",
        "  2. RETRY-AFTER-FAIL: reasoning mentions 'X had no effect / didn't work / try Y',",
        "     AND the agent picks a NEW direction this step:",
        "       operator = COMMIT  (it is a fresh directional decision, NOT a verify),",
        "       subgoal  = EXPLORE if probing unknowns, else the matching domain subgoal.",
        "  3. PURE OBSERVE: action is NOOP / IDLE / repeats last button only to confirm:",
        "       operator = VERIFY  (confirming outcome) or TRACK  (waiting on uncontrolled change).",
        "       VERIFY is reserved for steps with NO new directional intent.",
        "  4. WEIGH-OPTIONS: reasoning explicitly compares two candidates ('A or B'):",
        "       operator = COMPARE",
        "       subgoal  = POSITION (spatial alternative) or whichever achievement applies.",
        "  5. OPENING/MENU: schema shows get-ready / menu / round-start / first 3 steps:",
        "       operator = INSPECT,  subgoal = SETUP.",
        "  6. EVERY OTHER directional action (default):",
        "       operator = COMMIT,",
        "       subgoal  = the most specific game achievement — CLEAR, MERGE, COLLECT,",
        "                  ATTACK, NAVIGATE, BUILD, OPTIMIZE, EXPLORE, EXECUTE.",
        "",
        "VERIFY-OVERUSE GUARD:",
        "  If you would label this step VERIFY, ask: \"is the agent picking a NEW",
        "  direction this step?\" If yes, the correct operator is COMMIT (or COMPARE).",
        "  VERIFY should appear in well under 20% of steps across a trajectory.",
        "",
        "FEW-SHOT EXAMPLES:",
        _format_fewshot_block(_FEWSHOT),
        "",
        "===== STEP TO LABEL =====",
        "Schema (current state):",
        "<state>",
        _truncate_schema(schema_text),
        "</state>",
        "",
        f"Action chosen: {action or '(none)'}",
    ]
    if delta:
        parts.append(f"State delta from prior step: {delta}")
    if raw_reasoning:
        parts.append(f"Actor reasoning: {raw_reasoning}")
    if prev_tag:
        ctx = f"[{prev_tag}/{prev_subgoal}]" if prev_subgoal else f"[{prev_tag}]"
        if prev_note:
            ctx += f" {prev_note[:60]}"
        parts.append(f"Prior step label: {ctx}")
    parts.extend([
        "",
        "Reply ONLY a JSON object with three keys:",
        '  {"operator":"<one of the 6 operators>",',
        '   "subgoal":"<one of the 14 subgoals>",',
        '   "note":"<=15 words, concrete, references state or action>"}',
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call + JSON parser
# ---------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object in ``text``."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        # strip ``` fences if the model wrapped its output
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: take the first balanced JSON object substring.
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


_OP_VALID = frozenset(INTENT_OPERATORS)
_SG_VALID = frozenset(UNIFIED_SUBGOALS)

# Operator synonyms.  Drift candidates → canonical operator.
_OP_SYN: Dict[str, str] = {
    "GROUND": "INSPECT", "RETRIEVE": "INSPECT", "OBSERVE": "INSPECT",
    "PARSE": "INSPECT", "STUDY": "INSPECT", "EXAMINE": "INSPECT",
    "WATCH": "TRACK", "WAIT": "TRACK", "FOLLOW": "TRACK", "IDLE": "TRACK",
    "CHECK": "VERIFY", "CONFIRM": "VERIFY", "VALIDATE": "VERIFY",
    "ASSERT": "VERIFY",
    "WEIGH": "COMPARE", "EVALUATE": "COMPARE", "CHOOSE": "COMPARE",
    "DECIDE": "COMPARE",
    "ACT": "COMMIT", "ADVANCE": "COMMIT", "ENGAGE": "COMMIT",
    "PROGRESS": "COMMIT", "EXECUTE": "COMMIT",
    "DODGE": "RECOVER", "AVOID": "RECOVER", "EVADE": "RECOVER",
    "BLOCK": "RECOVER", "RETREAT": "RECOVER", "REACT": "RECOVER",
    "UNDO": "RECOVER", "DEFEND": "RECOVER",
}

# Subgoal synonyms.  Drift candidates → canonical subgoal.
_SG_SYN: Dict[str, str] = {
    "PLACE": "SETUP", "ARRANGE": "SETUP", "ROTATE": "SETUP",
    "DROP": "EXECUTE", "PRESS": "EXECUTE", "INPUT": "EXECUTE",
    "MOVE": "NAVIGATE", "WALK": "NAVIGATE", "RUN": "NAVIGATE",
    "JUMP": "NAVIGATE", "CLIMB": "NAVIGATE",
    "MATCH": "CLEAR", "BREAK": "CLEAR", "REMOVE": "CLEAR",
    "COMBINE": "MERGE", "FUSE": "MERGE", "STACK": "MERGE",
    "GRAB": "COLLECT", "PICKUP": "COLLECT", "GATHER": "COLLECT",
    "PICK_UP": "COLLECT",
    "CRAFT": "BUILD", "CREATE": "BUILD", "CONSTRUCT": "BUILD",
    "TARGET": "ATTACK", "STRIKE": "ATTACK", "SHOOT": "ATTACK",
    "FIRE": "ATTACK", "PUNCH": "ATTACK", "KICK": "ATTACK",
    "GUARD": "DEFEND", "BLOCK": "DEFEND", "PROTECT": "DEFEND",
    "PARRY": "DEFEND",
    "DODGE": "EVADE", "SIDESTEP": "EVADE", "AVOID": "EVADE",
    "RETREAT": "EVADE", "FLEE": "EVADE",
    "CONSOLIDATE": "OPTIMIZE", "FIX": "OPTIMIZE", "REFINE": "OPTIMIZE",
    "ORGANIZE": "OPTIMIZE",
    "ALIGN": "POSITION", "ADJUST": "POSITION",
    "SCORE": "EXECUTE", "PROGRESS": "EXECUTE",
    "FIND": "EXPLORE", "PROBE": "EXPLORE", "SEARCH": "EXPLORE",
    "STAY_ALIVE": "SURVIVE", "RECOVER_HP": "SURVIVE",
    # Cross-axis fallback — if the model returns an OPERATOR where we
    # asked for a subgoal, lift it via OPERATOR_TO_SUBGOAL below.
}


def _normalize_operator(raw: str) -> str:
    """Map a free-form operator string to the canonical ``INTENT_OPERATORS``."""
    s = (raw or "").strip().upper().strip("[]")
    if not s:
        return "COMMIT"
    if s in _OP_VALID:
        return s
    if s in _OP_SYN:
        return _OP_SYN[s]
    # If the model returned a SUBGOAL where we asked for an operator,
    # lift it back to its associated operator.
    if s in _SG_VALID:
        return SUBGOAL_TO_OPERATOR.get(s, "COMMIT")
    return "COMMIT"


def _normalize_subgoal(raw: str, operator: str = "") -> str:
    """Map a free-form subgoal string to the canonical ``UNIFIED_SUBGOALS``."""
    s = (raw or "").strip().upper().strip("[]")
    if not s:
        return OPERATOR_TO_SUBGOAL.get(operator, "EXECUTE")
    if s in _SG_VALID:
        return s
    if s in _SG_SYN:
        return _SG_SYN[s]
    # Operator returned where subgoal was expected — lift via map.
    if s in _OP_VALID:
        return OPERATOR_TO_SUBGOAL.get(s, "EXECUTE")
    return OPERATOR_TO_SUBGOAL.get(operator, "EXECUTE")


def _normalize_dual_tag(
    raw_obj: Dict[str, Any],
) -> Tuple[str, str]:
    """Return the canonical ``(operator, subgoal)`` from a parsed JSON.

    Tolerates the old single-tag schema (``intention_tag``) and the new
    dual-axis schema (``operator`` + ``subgoal``).  Always returns a
    valid pair from the official vocabularies.
    """
    op_raw = (
        raw_obj.get("operator")
        or raw_obj.get("op")
        or raw_obj.get("intention_tag")  # legacy single-axis fallback
        or ""
    )
    sg_raw = (
        raw_obj.get("subgoal")
        or raw_obj.get("sg")
        or raw_obj.get("intention_subgoal")
        or ""
    )
    op = _normalize_operator(str(op_raw))
    # If the legacy single-tag was actually a subgoal name (e.g. CLEAR,
    # MERGE), pull it out of the op slot and into the subgoal slot.
    if str(op_raw).strip().upper() in _SG_VALID and not sg_raw:
        sg_raw = op_raw
    sg = _normalize_subgoal(str(sg_raw), operator=op)
    return op, sg


def _trim_note(raw_note: str) -> str:
    note = (raw_note or "").strip().strip("\"'`")
    if not note:
        return ""
    words = note.split()
    if len(words) > INTENT_NOTE_WORD_BUDGET:
        words = words[:INTENT_NOTE_WORD_BUDGET]
        # drop trailing partial punctuation
        joined = " ".join(words).rstrip(",.;:")
        return joined
    return note


@dataclass
class LabelOutcome:
    operator: str
    subgoal: str
    note: str
    source: str  # "llm" | "rule_classifier" | "fallback_default"


# ---------------------------------------------------------------------------
# Heuristic subgoal classifier (paired with the operator rule classifier
# already imported from ``extract_skillbank_gymv_gpt54``).  Used only when
# the LLM call fails.  Order matters — first match wins.
# ---------------------------------------------------------------------------

_SUBGOAL_HEURISTICS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:dodge|sidestep|evade|avoid|outrun|circle\s+away)\b", re.I), "EVADE"),
    (re.compile(r"\b(?:block|guard|shield|parry|defend)\b", re.I), "DEFEND"),
    (re.compile(r"\b(?:low\s+health|critical|topping\s+out|near\s+death|hp\s+low|game\s+over)\b", re.I), "SURVIVE"),
    (re.compile(r"\b(?:line\s+clear|clear\s+(?:row|line|match)|trigger\s+(?:clear|line))\b", re.I), "CLEAR"),
    (re.compile(r"\bmerge\b|\bcombine\s+\d+\s*[-]?\s*tile|\bfuse\b", re.I), "MERGE"),
    (re.compile(r"\b(?:pick\s*up|collect|grab|gather|coin|powerup|orb)\b", re.I), "COLLECT"),
    (re.compile(r"\b(?:build|craft|construct|extend\s+the)\b", re.I), "BUILD"),
    (re.compile(r"\b(?:attack|strike|hit|punch|kick|fire|shoot|combo)\b", re.I), "ATTACK"),
    (re.compile(r"\b(?:get\s+ready|press\s+start|round\s+1\b|menu|title|opening|stage\s+select)\b", re.I), "SETUP"),
    (re.compile(r"\b(?:explore|probe|investigate|test\s+(?:button|input)|try\s+a\s+different)\b", re.I), "EXPLORE"),
    (re.compile(r"\b(?:align|position|stage|set\s+up\s+for|adjacent\s+to)\b", re.I), "POSITION"),
    (re.compile(r"\b(?:walk|run|move\s+toward|navigate|approach)\b", re.I), "NAVIGATE"),
    (re.compile(r"\b(?:consolidate|optimize|refine|tidy|reduce\s+holes)\b", re.I), "OPTIMIZE"),
]


def _classify_subgoal_heuristic(
    *, intent_text: str, schema_text: str, operator: str
) -> str:
    """Best-effort subgoal classification from text alone.

    Falls back to the operator's default subgoal (via OPERATOR_TO_SUBGOAL)
    when no keyword matches.  Always returns a valid UNIFIED_SUBGOALS tag.
    """
    for regex, sg in _SUBGOAL_HEURISTICS:
        if regex.search(intent_text or "") or regex.search(schema_text or ""):
            return sg
    return OPERATOR_TO_SUBGOAL.get(operator, "EXECUTE")


def _label_step(
    *,
    corpus: str,
    schema_text: str,
    action: str,
    raw_reasoning: str,
    delta: str,
    step_idx: int,
    model: str,
    prev_tag: str = "",
    prev_subgoal: str = "",
    prev_note: str = "",
) -> LabelOutcome:
    """Run one LLM labelling call with a graceful fallback chain.

    Returns a ``LabelOutcome`` carrying both axes.  Always succeeds with
    a valid ``(operator, subgoal)`` pair — only ``source`` changes
    depending on which fallback fired.
    """

    # 1) LLM call (dual-axis schema).
    if ask_model is not None:
        prompt = _build_user_prompt(
            corpus=corpus,
            schema_text=schema_text,
            action=action,
            delta=delta,
            raw_reasoning=raw_reasoning,
            prev_tag=prev_tag,
            prev_subgoal=prev_subgoal,
            prev_note=prev_note,
        )
        try:
            raw = ask_model(
                f"{_SYSTEM_PROMPT}\n\n{prompt}",
                model=model,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM call raised: %s", exc)
            raw = None

        if raw and not raw.startswith("Error"):
            cleaned = strip_think_tags(raw or "").strip()
            obj = _extract_json_object(cleaned)
            if obj:
                op, sg = _normalize_dual_tag(obj)
                note = _trim_note(
                    str(
                        obj.get("note")
                        or obj.get("intention_note")
                        or ""
                    )
                )
                if note:
                    return LabelOutcome(
                        operator=op, subgoal=sg, note=note, source="llm",
                    )

    # 2) Rule-based fallback — operator from existing classifier, subgoal
    #    from heuristic keyword scan.  Always lands on a valid pair.
    op = _classify_intent_operator(
        intent_text=raw_reasoning,
        schema_text=schema_text,
        action=action,
        step_idx=step_idx,
    )
    sg = _classify_subgoal_heuristic(
        intent_text=raw_reasoning,
        schema_text=schema_text,
        operator=op,
    )
    note_seed = raw_reasoning.split(".")[0].strip() if raw_reasoning else f"action {action}"
    return LabelOutcome(
        operator=op,
        subgoal=sg,
        note=_trim_note(note_seed),
        source="rule_classifier",
    )


# ---------------------------------------------------------------------------
# Per-episode labelling
# ---------------------------------------------------------------------------

def _summary_state_for_delta(corpus: str, exp: Dict[str, Any], schema_text: str) -> str:
    """Return a compact summary string used for inter-step delta computation."""
    pre = exp.get("summary_state")
    if isinstance(pre, str) and pre.strip():
        return pre
    if corpus == "gym_v":
        return _compact_summary_from_schema(schema_text)
    # env_wrappers — strip the schema block if present, otherwise compact state text
    return _compact_summary_from_schema(schema_text) or ""


def label_episode(
    ep_data: Dict[str, Any],
    *,
    corpus: str,
    model: str,
    workers: int,
    log_prefix: str,
    bucket: str,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Label every step in one episode in parallel; return augmented dict + counters."""
    raw_exps = ep_data.get("experiences") or []
    n = len(raw_exps)
    counters: Dict[str, int] = {"llm": 0, "rule_classifier": 0, "fallback_default": 0}
    if n == 0:
        return ep_data, counters

    # Pre-extract schema texts and pre-compute deltas serially (cheap).
    schemas: List[str] = [_select_schema_text(exp) for exp in raw_exps]
    summaries: List[str] = [
        _summary_state_for_delta(corpus, raw_exps[i], schemas[i]) for i in range(n)
    ]

    # Pre-compute rule-classifier (operator, subgoal) for every step so
    # parallel LLM calls can still see a deterministic prior-step hint
    # for trajectory context.  Runs in microseconds.
    prior_op: List[str] = []
    prior_sg: List[str] = []
    for i, exp in enumerate(raw_exps):
        intent_txt = str(exp.get("intentions") or "")
        op = _classify_intent_operator(
            intent_text=intent_txt,
            schema_text=schemas[i],
            action=str(exp.get("action") or ""),
            step_idx=i,
        )
        sg = _classify_subgoal_heuristic(
            intent_text=intent_txt,
            schema_text=schemas[i],
            operator=op,
        )
        prior_op.append(op)
        prior_sg.append(sg)

    def _label_one(i: int) -> Tuple[int, LabelOutcome]:
        exp = raw_exps[i]
        schema_text = schemas[i]
        delta = _compute_delta(summaries[i - 1], summaries[i]) if i > 0 else ""
        action = str(exp.get("action") or "")
        raw_reasoning = str(exp.get("intentions") or exp.get("subgoal") or "")
        outcome = _label_step(
            corpus=corpus,
            schema_text=schema_text,
            action=action,
            raw_reasoning=raw_reasoning,
            delta=delta,
            step_idx=i,
            model=model,
            prev_tag=prior_op[i - 1] if i > 0 else "",
            prev_subgoal=prior_sg[i - 1] if i > 0 else "",
            prev_note="",
        )
        return i, outcome

    outcomes: List[Optional[LabelOutcome]] = [None] * n
    if workers <= 1 or n <= 1:
        for i in range(n):
            _, oc = _label_one(i)
            outcomes[i] = oc
            counters[oc.source] += 1
    else:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, n))) as ex:
            for fut in as_completed(ex.submit(_label_one, i) for i in range(n)):
                i, oc = fut.result()
                outcomes[i] = oc
                counters[oc.source] += 1

    # Write the labels back into the episode dict (mutating a copy).
    out_data = dict(ep_data)
    new_exps: List[Dict[str, Any]] = []
    for i, exp in enumerate(raw_exps):
        oc = outcomes[i] or LabelOutcome(
            operator="COMMIT",
            subgoal=OPERATOR_TO_SUBGOAL.get("COMMIT", "EXECUTE"),
            note=str(exp.get("action") or ""),
            source="fallback_default",
        )
        new_exp = dict(exp)
        new_exp["raw_intentions"] = exp.get("intentions") or ""
        new_exp["intention_tag"] = oc.operator
        new_exp["intention_subgoal"] = oc.subgoal
        new_exp["intention_note"] = oc.note
        # Composite ``[OPERATOR/SUBGOAL] note`` — drop-in for the
        # skill-bank pipeline; parse_intention_tag returns the operator,
        # parse_intention_tags returns both axes.
        head = f"[{oc.operator}/{oc.subgoal}]"
        new_exp["intentions"] = f"{head} {oc.note}".strip() if oc.note else head
        meta = dict(new_exp.get("metadata") or {})
        meta["intent_operator"] = oc.operator
        meta["intent_subgoal"]  = oc.subgoal
        meta["intent_label_source"] = oc.source
        new_exp["metadata"] = meta
        new_exps.append(new_exp)
    out_data["experiences"] = new_exps
    out_data.setdefault("intentions_label_meta", {})
    out_data["intentions_label_meta"].update({
        "corpus": corpus,
        "bucket": bucket,
        "model": model,
        "labeler": "labeling.label_intentions_gpt54",
        "label_scheme": "dual_axis_v1",
        "labelled_at": datetime.utcnow().isoformat() + "Z",
        "step_count": n,
        "source_counts": counters,
    })
    if log_prefix:
        logger.info(
            "%s episode -> %d steps  llm=%d  rule=%d  fallback=%d",
            log_prefix, n, counters["llm"], counters["rule_classifier"],
            counters["fallback_default"],
        )
    return out_data, counters


# ---------------------------------------------------------------------------
# Per-source driver
# ---------------------------------------------------------------------------

def _process_source(
    src: _Source,
    *,
    output_dir: Path,
    model: str,
    workers: int,
    max_episodes: Optional[int],
    resume: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    bucket_out = output_dir / src.corpus / src.bucket
    bucket_out.mkdir(parents=True, exist_ok=True)
    files = list(src.files)
    if max_episodes is not None:
        files = files[: max_episodes]

    aggregate_counters: Dict[str, int] = {
        "llm": 0, "rule_classifier": 0, "fallback_default": 0,
    }
    operator_counts: Dict[str, int] = {}
    subgoal_counts: Dict[str, int] = {}
    pair_counts: Dict[str, int] = {}
    per_episode: List[Dict[str, Any]] = []

    for fp in files:
        out_path = bucket_out / fp.name
        if resume and out_path.exists():
            logger.info("[%s] skip (resume): %s", src.bucket, fp.name)
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                ep_raw = json.load(f)
        except Exception as exc:
            logger.warning("[%s] failed to load %s: %s", src.bucket, fp.name, exc)
            continue
        log_prefix = f"[{src.corpus}/{src.bucket}/{fp.stem}]"
        labelled, counters = label_episode(
            ep_raw,
            corpus=src.corpus, model=model, workers=workers,
            log_prefix=log_prefix, bucket=src.bucket,
        )
        for k, v in counters.items():
            aggregate_counters[k] = aggregate_counters.get(k, 0) + v
        for exp in labelled.get("experiences") or []:
            op = exp.get("intention_tag") or "?"
            sg = exp.get("intention_subgoal") or "?"
            operator_counts[op] = operator_counts.get(op, 0) + 1
            subgoal_counts[sg]  = subgoal_counts.get(sg, 0) + 1
            pair = f"{op}/{sg}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        per_episode.append({
            "file": fp.name,
            "step_count": len(labelled.get("experiences") or []),
            "source_counts": counters,
        })
        if dry_run:
            logger.info("[%s] dry_run — not writing %s", src.bucket, out_path)
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(labelled, f, ensure_ascii=False, indent=2)
        logger.info("[%s] wrote %s", src.bucket, out_path.relative_to(output_dir))

    summary = {
        "corpus": src.corpus,
        "bucket": src.bucket,
        "input_dir": str(src.base_dir),
        "output_dir": str(bucket_out),
        "model": model,
        "label_scheme": "dual_axis_v1",
        "episodes_labelled": len(per_episode),
        "step_count_total": sum(p["step_count"] for p in per_episode),
        "source_counts": aggregate_counters,
        "operator_distribution": dict(sorted(operator_counts.items(), key=lambda kv: -kv[1])),
        "subgoal_distribution":  dict(sorted(subgoal_counts.items(),  key=lambda kv: -kv[1])),
        "pair_distribution":     dict(sorted(pair_counts.items(),     key=lambda kv: -kv[1])),
        "per_episode": per_episode,
    }
    if not dry_run:
        with open(bucket_out / "_intentions_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label cold-start episodes with (intention_tag, intention_note) per step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--gymv_input", type=str, default="",
        help="Path to a gym-v cold-start run root (Cold-start-out-gymv/<run>) or a single Temporal_*-v0 dir.",
    )
    p.add_argument(
        "--envw_input", type=str, default="",
        help="Path to an env_wrappers cold-start run root (Cold-start-out/<run>) or a single <group>/<game> dir.",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Destination root for labelled episodes.",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--corpus", type=str, default="auto", choices=["auto", "gym_v", "env_wrappers"])
    p.add_argument("--envs", nargs="*", default=None,
                   help="Restrict to these gym-v env names (e.g. Temporal_Airstriker-v0).")
    p.add_argument("--games", nargs="*", default=None,
                   help="Restrict to these env_wrappers game names (e.g. tetris super_mario).")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent LLM calls within an episode.")
    p.add_argument("--resume", action="store_true",
                   help="Skip episode files that already exist in --output_dir.")
    p.add_argument("--dry_run", action="store_true",
                   help="Process episodes but do not write outputs.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _filter_sources(
    sources: List[_Source],
    envs: Optional[List[str]],
    games: Optional[List[str]],
) -> List[_Source]:
    out: List[_Source] = []
    for s in sources:
        if s.corpus == "gym_v" and envs is not None and s.bucket not in envs:
            continue
        if s.corpus == "env_wrappers" and games is not None and s.bucket not in games:
            continue
        out.append(s)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    if not args.gymv_input and not args.envw_input:
        logger.error("Provide at least one of --gymv_input or --envw_input.")
        return 2

    sources: List[_Source] = []

    if args.gymv_input:
        gymv_root = Path(args.gymv_input).resolve()
        if gymv_root.is_dir() and gymv_root.name.startswith("Temporal_"):
            # User pointed at a single env folder.
            eps = tuple(sorted(gymv_root.glob("episode_[0-9]*.json")))
            if eps:
                sources.append(_Source(
                    corpus="gym_v", bucket=gymv_root.name,
                    files=eps, base_dir=gymv_root.parent,
                ))
        else:
            sources.extend(_find_gymv_sources(gymv_root))

    if args.envw_input:
        envw_root = Path(args.envw_input).resolve()
        if (
            envw_root.is_dir()
            and any(envw_root.glob("episode_[0-9]*.json"))
        ):
            sources.append(_Source(
                corpus="env_wrappers", bucket=envw_root.name,
                files=tuple(sorted(envw_root.glob("episode_[0-9]*.json"))),
                base_dir=envw_root.parent.parent if envw_root.parent.parent.exists() else envw_root.parent,
            ))
        else:
            sources.extend(_find_envw_sources(envw_root))

    sources = _filter_sources(sources, args.envs, args.games)
    if not sources:
        logger.error("No matching episode sources found.")
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovered %d source bucket(s):", len(sources))
    for s in sources:
        logger.info("  %s/%s  (%d episodes)", s.corpus, s.bucket, len(s.files))

    t0 = time.time()
    all_summaries: List[Dict[str, Any]] = []
    failures = 0
    for s in sources:
        try:
            summary = _process_source(
                s,
                output_dir=output_dir,
                model=args.model,
                workers=max(1, args.workers),
                max_episodes=args.max_episodes,
                resume=args.resume,
                dry_run=args.dry_run,
            )
            all_summaries.append(summary)
        except Exception as exc:
            logger.error("Bucket %s/%s failed: %s\n%s",
                         s.corpus, s.bucket, exc, traceback.format_exc())
            failures += 1

    elapsed = time.time() - t0
    run_summary = {
        "labeller": "labeling.label_intentions_gpt54",
        "model": args.model,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 2),
        "buckets": all_summaries,
        "failures": failures,
    }
    if not args.dry_run:
        with open(output_dir / "_intentions_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

    logger.info("=== run summary ===")
    logger.info("  elapsed: %.1fs   buckets: %d   failures: %d",
                elapsed, len(all_summaries), failures)
    for s in all_summaries:
        op_top = ",".join(
            f"{k}:{v}" for k, v in list(s["operator_distribution"].items())[:4]
        )
        sg_top = ",".join(
            f"{k}:{v}" for k, v in list(s["subgoal_distribution"].items())[:4]
        )
        logger.info(
            "  %s/%s  steps=%d  llm=%d  rule=%d  op_top=%s  sg_top=%s",
            s["corpus"], s["bucket"], s["step_count_total"],
            s["source_counts"].get("llm", 0),
            s["source_counts"].get("rule_classifier", 0),
            op_top, sg_top,
        )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
