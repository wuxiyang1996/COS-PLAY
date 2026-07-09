"""Async episode runner for the co-evolution loop.

Mirrors ``scripts/qwen3_decision_agent.run_episode()`` but replaces every
synchronous LLM call with an ``await`` on the shared :class:`AsyncVLLMClient`,
and runs ``env.step()`` in an executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Headless mode for retro/pyglet/SDL — must be set before any game env import
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

from trainer.coevolution.vllm_client import AsyncVLLMClient

logger = logging.getLogger(__name__)


def _critical_actions_for(game: str, valid_actions: List[str]) -> List[str]:
    """Return the subset of :data:`GAME_CRITICAL_ACTIONS` that are
    actually exposed for *game* this step.  Imports lazily so callers
    that don't pass a real game string (eval / smoke tests) avoid
    pulling the whole config module into the hot path on every call.
    """
    try:
        from trainer.coevolution.config import GAME_CRITICAL_ACTIONS
    except Exception:                                        # pragma: no cover
        return []
    desired = GAME_CRITICAL_ACTIONS.get(game) or []
    if not desired:
        return []
    valid_set = set(valid_actions)
    return [a for a in desired if a in valid_set]


# ---------------------------------------------------------------------------
# Lazy imports — these pull in heavyweight packages that live in the project
# ---------------------------------------------------------------------------

_IMPORTS_CACHE: Dict[str, Any] = {}

# Games that use Orak env (env_wrappers.orak_nl_wrapper.make_orak_env)
ORAK_GAMES_SET = {"super_mario"}
# Orak games that MUST use SubprocessEnv (nes_py / NumPy 2.x incompatibility)
ORAK_SUBPROCESS_GAMES = {"super_mario"}
# Games that use GamingAgent make_gaming_env
GAMINGAGENT_GAMES = {
    "twenty_forty_eight", "candy_crush", "tetris",
}
# Games that use Gym-V Temporal/* (stable-retro Genesis) via
# env_wrappers.gymv_temporal_nl_wrapper.make_gymv_temporal_env. The set of
# wired slugs lives in that module's GYMV_TEMPORAL_GAMES dict (8 games for
# the default benchmark scope — the 4 Phase-1 source games plus the 4
# Phase-2 holdouts from
# training_notes/coevo-3phase-cross-game-ood-transfer-plan.md §4.1 / §7.1).
# We import the dict lazily inside _lazy_imports() so module-import time
# stays cheap when gym_v / stable-retro / ROMs aren't installed.
GYMV_TEMPORAL_GAMES_SET: set = set()  # populated by _lazy_imports()
# Gym-V games MUST use SubprocessEnv: stable-retro's Genesis emulator is
# a process-singleton ("Cannot create multiple emulator instances per
# process"), so any concurrent in-process episodes after the first one
# crash and return ``EpisodeResult(steps=0)`` — i.e. ``1/8 ok`` on the
# rollout collector log line.  Subprocess isolation gives each
# concurrent episode its own emulator and restores 8/8 GRPO rollouts.
# Set to the same membership as GYMV_TEMPORAL_GAMES_SET (populated lazily
# below) — keep them in sync.
GYMV_SUBPROCESS_GAMES: set = set()  # populated by _lazy_imports() (mirror)


def _lazy_imports():
    """Return project modules, imported once and cached."""
    global _IMPORTS_CACHE, GYMV_TEMPORAL_GAMES_SET, GYMV_SUBPROCESS_GAMES
    if not _IMPORTS_CACHE:
        from env_wrappers.game_configs import GAME_CONFIGS
        from env_wrappers.gym_like import make_gaming_env
        from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
        # Orak env (Super Mario, etc.)
        try:
            from env_wrappers.orak_nl_wrapper import make_orak_env
        except ImportError:
            make_orak_env = None

        from env_wrappers.subprocess_env import SubprocessEnv

        # Gym-V Temporal/* (stable-retro Genesis) — Phase-1 source +
        # Phase-2 holdout games. Import is best-effort: if gym_v /
        # stable-retro aren't installed, the slug set stays empty and
        # any --games gymv_<slug> request falls through to the
        # GAME_CONFIGS-not-found error instead of crashing the whole
        # runner at import time.
        try:
            from env_wrappers.gymv_temporal_nl_wrapper import (
                GYMV_TEMPORAL_GAMES,
                make_gymv_temporal_env,
            )
            GYMV_TEMPORAL_GAMES_SET.update(GYMV_TEMPORAL_GAMES.keys())
            # Mirror the slug set into the subprocess gate; we always
            # subprocess-isolate stable-retro envs (process-singleton
            # constraint, see GYMV_SUBPROCESS_GAMES docstring).
            GYMV_SUBPROCESS_GAMES.update(GYMV_TEMPORAL_GAMES.keys())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "env_wrappers.gymv_temporal_nl_wrapper unavailable (%s); "
                "Gym-V Temporal/* games will not be runnable in this process.",
                exc,
            )
            GYMV_TEMPORAL_GAMES = {}
            make_gymv_temporal_env = None

        from decision_agents.agent_helper import (
            build_rag_summary,
            compact_text_observation,
            extract_game_facts,
            infer_intention,
            strip_think_tags,
            HARD_SUMMARY_CHAR_LIMIT,
            SUBGOAL_TAGS,
        )
        try:
            from decision_agents.agent_helper import _get_protocol_for_skill
        except ImportError:
            _get_protocol_for_skill = None

        _IMPORTS_CACHE = {
            "GAME_CONFIGS": GAME_CONFIGS,
            "make_gaming_env": make_gaming_env,
            "make_orak_env": make_orak_env,
            "make_gymv_temporal_env": make_gymv_temporal_env,
            "GYMV_TEMPORAL_GAMES": GYMV_TEMPORAL_GAMES,
            "SubprocessEnv": SubprocessEnv,
            "GamingAgentNLWrapper": GamingAgentNLWrapper,
            "build_rag_summary": build_rag_summary,
            "compact_text_observation": compact_text_observation,
            "extract_game_facts": extract_game_facts,
            "infer_intention": infer_intention,
            "strip_think_tags": strip_think_tags,
            "HARD_SUMMARY_CHAR_LIMIT": HARD_SUMMARY_CHAR_LIMIT,
            "SUBGOAL_TAGS": SUBGOAL_TAGS,
            "_get_protocol_for_skill": _get_protocol_for_skill,
        }
    return _IMPORTS_CACHE

INTENTION_WORD_BUDGET = 15
MAX_REPEAT_ACTIONS = 2

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row — try something different.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "ACTION: <number>\n"
)

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are an expert game strategist. "
    "Given the current game state and a set of candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits the current state>\n"
    "SKILL: <number>\n"
)

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

_TAG_KEYWORD_MAP: Dict[str, str] = {
    "surviv": "SURVIVE", "clear": "CLEAR", "merge": "MERGE",
    "setup": "SETUP", "position": "POSITION", "navigat": "NAVIGATE",
    "optimiz": "OPTIMIZE", "defend": "DEFEND", "attack": "ATTACK",
    "build": "BUILD", "explor": "EXPLORE", "collect": "COLLECT",
}


def _infer_tag_from_text(text: str) -> str:
    """Scan for keyword stems and return the best-matching tag."""
    low = text.lower()
    for stem, tag in _TAG_KEYWORD_MAP.items():
        if stem in low:
            return tag
    return "SETUP"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GRPORecord:
    adapter: str  # "action_taking" or "skill_selection"
    game: str
    episode_id: str
    step: int
    prompt: str = ""
    completion: str = ""
    reward: float = 0.0
    episode_length: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    game: str
    episode_id: str
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    skill_switches: int = 0
    grpo_records: List[GRPORecord] = field(default_factory=list)
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    eval_only: bool = False

    # Multi-role metadata (populated when unified_role_rollouts=True)
    role: str = ""          # e.g. "Merlin", "FRANCE"
    side: str = ""          # e.g. "good", "evil", or power name
    role_index: int = -1    # player index (Avalon) or power ordinal


# ---------------------------------------------------------------------------
# Helpers (lightweight, no LLM calls)
# ---------------------------------------------------------------------------

def _generate_summary_state(
    state: str, game_name: str, step_idx: int, total_steps: int, reward: float,
) -> str:
    imp = _lazy_imports()
    return imp["build_rag_summary"](
        state, game_name, step_idx=step_idx, total_steps=total_steps, reward=reward,
    )


def _compute_state_delta(prev_ss: str, curr_ss: str) -> str:
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
    changes = [f"{k}:{p[k]}->{v}" for k, v in c.items()
               if k not in skip and k in p and p[k] != v]
    return ", ".join(changes[:5])


def _detect_urgency(summary_state: str, game_name: str) -> str:
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


def _build_rich_state_observation(
    info: Dict[str, Any],
    summary_state: str,
) -> str:
    """Build a rich spatial observation for games that provide grid data.
    Falls back to *summary_state* when grid is absent."""
    parts: List[str] = []
    grid_str = info.get("grid_string")
    if grid_str:
        parts.append(f"Board:\n{grid_str}")
    element_summary = info.get("element_summary")
    if element_summary:
        parts.append(f"Elements:\n{element_summary}")
    spatial = info.get("spatial_analysis")
    if spatial:
        parts.append(f"Analysis:\n{spatial}")
    deadlock_info = info.get("deadlock_info")
    if deadlock_info:
        parts.append(f"*** WARNING: {deadlock_info} — consider restart ***")
    if summary_state:
        parts.append(f"Status: {summary_state}")
    if parts:
        return "\n\n".join(parts)
    return summary_state or ""


def _normalize_intention(raw: str) -> str:
    imp = _lazy_imports()
    _SUBGOAL_TAG_SET = frozenset(imp["SUBGOAL_TAGS"])
    raw = raw.split("\n")[0].strip().strip('"').strip("'")
    if not raw.startswith("["):
        tag = _infer_tag_from_text(raw)
        return f"[{tag}] {raw}"
    m = _TAG_RE.match(raw)
    if not m:
        tag = _infer_tag_from_text(raw)
        return f"[{tag}] {raw}"
    tag = m.group(1).upper()
    rest = raw[m.end():].strip()
    if tag not in _SUBGOAL_TAG_SET:
        tag = _TAG_ALIASES.get(tag, _infer_tag_from_text(rest))
    return f"[{tag}] {rest}" if rest else f"[{tag}]"


async def _generate_intention(
    vllm_client: AsyncVLLMClient,
    state_text: str,
    game_name: str,
    summary_state: str,
    prev_intention: str,
    prev_summary_state: str,
    delta: str,
    urgency: str,
    skill_guidance: Optional[Dict[str, Any]],
    last_action: str,
    tag_history: Optional[List[str]] = None,
) -> str:
    """Generate a ``[TAG] subgoal`` via the **base model** (no LoRA).

    Ported from ``qwen3_decision_agent.generate_skill_aware_intention()``.
    Uses higher temperature (0.7) so the base model's SFT-trained tag
    diversity is preserved — unlike the action_taking LoRA which has
    collapsed to a single tag.
    """
    imp = _lazy_imports()
    SUBGOAL_TAGS = imp["SUBGOAL_TAGS"]
    tags_str = "|".join(SUBGOAL_TAGS)
    facts_line = f"Facts: {summary_state}\n" if summary_state else ""
    delta_line = f"Changed: {delta}\n" if delta else ""
    urgency_line = f"URGENCY: {urgency}\n" if urgency else ""
    prev_line = f"Previous subgoal: {prev_intention}\n" if prev_intention else ""
    shift_hint = (
        "IMPORTANT: If the situation changed significantly or urgency is high, "
        "pick a NEW tag that matches the new priority.\n"
        if delta or urgency else ""
    )

    diversity_hint = ""
    if tag_history and len(tag_history) >= 5:
        from collections import Counter
        window = tag_history[-10:]
        counts = Counter(window)
        top_tag, top_count = counts.most_common(1)[0]
        if top_count / len(window) > 0.5:
            others = [t for t in SUBGOAL_TAGS if t != top_tag][:4]
            diversity_hint = (
                f"DIVERSITY: You used [{top_tag}] {top_count}/{len(window)} "
                f"recent steps. Try a DIFFERENT tag like "
                f"{', '.join(others)}.\n"
            )

    skill_context = ""
    if skill_guidance and skill_guidance.get("skill_id"):
        sk_name = skill_guidance.get("skill_name", skill_guidance["skill_id"])
        sk_hint = skill_guidance.get("execution_hint", "")
        skill_context = f"Active skill: {sk_name}"
        if sk_hint:
            skill_context += f" — {sk_hint[:100]}"
        skill_context += "\n"

    game_label = game_name.replace("_", " ") if game_name else "game"

    examples = (
        "Examples:\n"
        "  tetris, stack_h=14, holes=8 → [SURVIVE] reduce stack height before game over\n"
        "  tetris, holes=2, stack_h=6 → [SETUP] position piece for future line clear\n"
        "  tetris, full row forming → [CLEAR] complete the line to score points\n"
        "  2048, empty=3, max=256 → [MERGE] combine tiles to free board space\n"
        "  2048, large tile in corner → [POSITION] keep max tile anchored in corner\n"
        "  2048, board nearly full → [SURVIVE] avoid game over by creating space\n"
        "  candy_crush, moves=4, target=500 → [CLEAR] maximize cascade combos now\n"
        "  candy_crush, special candy available → [EXECUTE] activate combo for big score\n"
        "  candy_crush, board cluttered → [OPTIMIZE] clear blockers to open matches\n"
    )

    prompt = (
        f"{game_label}. Action: {last_action}\n"
        f"State: {state_text}\n"
        f"{facts_line}"
        f"{delta_line}"
        f"{urgency_line}"
        f"{skill_context}"
        f"{prev_line}"
        f"{shift_hint}"
        f"{diversity_hint}"
        f"{examples}\n"
        f"What subgoal? Reply ONLY: [TAG] phrase "
        f"(max {INTENTION_WORD_BUDGET} words)\n"
        f"Tags: {tags_str}\n"
        f"Subgoal:"
    )

    try:
        result = await vllm_client.generate_chat(
            [{"role": "user", "content": prompt}],
            adapter="base", temperature=0.7, max_tokens=96,
        )
        text = result.text.strip() if result.text else ""
        if text:
            imp2 = _lazy_imports()
            text = imp2["strip_think_tags"](text).strip()
            first_line = text.split("\n")[0].strip()
            if first_line:
                return _normalize_intention(first_line)[:150]
    except Exception as exc:
        logger.debug("Intention generation failed: %s", exc)

    if prev_intention and prev_intention != "[EXECUTE] play":
        return prev_intention
    fallback_tag = _infer_tag_from_text(urgency or summary_state or "")
    return f"[{fallback_tag}] {game_label}"


def _format_numbered_actions(action_names: List[str]) -> str:
    return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))


def _build_recent_context(recent_actions: List[str], recent_rewards: List[float]) -> str:
    if not recent_actions:
        return ""
    lines = ["Recent actions and rewards:"]
    for a, r in zip(recent_actions[-5:], recent_rewards[-5:]):
        lines.append(f"  {a} -> reward {r:.1f}")
    total = sum(recent_rewards[-5:])
    if total == 0 and len(recent_actions) >= 3:
        lines.append("WARNING: Recent actions got 0 reward. Try a DIFFERENT action!")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_skill_guidance_for_prompt(
    guidance: Optional[Dict[str, Any]],
    protocol_step_idx: int = 0,
    progress_summary: str = "",
) -> str:
    if guidance is None or not guidance.get("skill_id"):
        return ""
    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:200]}")
    if progress_summary:
        parts.append(f"  Progress: {progress_summary}")
    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7], 1):
            marker = ">>" if (i - 1) == protocol_step_idx else "  "
            parts.append(f"  {marker} {i}. {step}")
    preconditions = protocol.get("preconditions", []) if isinstance(protocol, dict) else []
    if preconditions:
        parts.append(f"  Preconditions: {'; '.join(preconditions[:3])}")
    success = protocol.get("success_criteria", []) if isinstance(protocol, dict) else []
    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")
    abort = protocol.get("abort_criteria", []) if isinstance(protocol, dict) else []
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")
    parts.append("--- end skill ---\n")
    return "\n".join(parts)


def _format_candidates_for_selection(candidates: List[Dict[str, Any]]) -> str:
    """Trainer-side mirror of
    :func:`scripts.qwen3_decision_agent._format_candidates_for_selection`.

    Renders ``_harness_adaptation_score`` (Refinement B) and the
    ``_harness_deboost`` recent-veto rate (Refinement A) when those
    fields are present on the candidate dict. Both fields are best-
    effort and omitted silently when the harness path didn't decorate
    the candidate. See ``harness/README.md`` §22.5 for the design.
    """
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
        adapt = c.get("_harness_adaptation_score")
        if isinstance(adapt, (int, float)):
            lines.append(f"     Adaptation: {float(adapt):.2f}")
        deboost = c.get("_harness_deboost")
        if isinstance(deboost, (int, float)) and float(deboost) < 0.95:
            lines.append(f"     Recent veto rate: {1.0 - float(deboost):.2f}")
    return "\n".join(lines)


def _parse_skill_selection(reply: str, n_candidates: int, candidates: Optional[List[Dict[str, Any]]] = None) -> Tuple[int, Optional[str]]:
    imp = _lazy_imports()
    strip_think_tags = imp["strip_think_tags"]

    if not reply:
        return 0, None
    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply
    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nSKILL|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
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


def _parse_action_response(
    reply: str, valid_actions: List[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse action response (may contain optional SUBGOAL/REASONING lines).

    Returns (action, reasoning, intention).
    """
    imp = _lazy_imports()
    strip_think_tags = imp["strip_think_tags"]

    if not reply:
        return (valid_actions[0] if valid_actions else "stay"), None, None
    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    intention = None
    subgoal_m = re.search(
        r"SUBGOAL\s*:\s*(.+?)(?=\nREASONING|\nACTION|\Z)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if subgoal_m:
        raw_sg = subgoal_m.group(1).strip().split("\n")[0].strip()
        intention = _normalize_intention(raw_sg)[:150] if raw_sg else None

    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_m:
        raw = action_m.group(1).strip()
        matched = _fuzzy_match_action(raw, valid_actions)
        if matched:
            return matched, reasoning, intention

    fallback = valid_actions[0] if valid_actions else "stay"
    return _ActionFallback(fallback), reasoning, intention


class _ActionFallback(str):
    """Marker subclass so callers can detect fuzzy-match failures."""
    pass


def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    if not raw or not valid_actions:
        return None
    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}
    if raw_lower in lower_map:
        return lower_map[raw_lower]

    raw_compact = re.sub(r"\s+", "", raw_lower)
    compact_map = {re.sub(r"\s+", "", a.lower()): a for a in valid_actions}
    if raw_compact in compact_map:
        return compact_map[raw_compact]
    num_m = re.match(r"^(\d+)\.?\s*$", raw_lower)
    if num_m:
        idx = int(num_m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]
    num_m2 = re.search(r"(?:^|\s)(\d+)\s*[.:\-]", raw_lower)
    if num_m2:
        idx = int(num_m2.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]
    for canon_lower, canon in lower_map.items():
        if len(canon_lower) < 3 and len(raw_lower) > 5:
            continue
        if canon_lower in raw_lower or raw_lower in canon_lower:
            return canon
    return None


_CRITICAL_ACTION_DRY_SPELL = 8
"""Number of consecutive zero-reward decisions that haven't selected a
known-critical action before the anti-repetition shim force-substitutes
the critical action.  Tuned for gymv shooters where the natural episode
length is ~20 frames (frame_skip=8) and 8 consecutive non-firing decisions
≈ ~50% of an average episode without any scoring attempt."""


def _apply_anti_repetition(
    action: str, valid_actions: List[str],
    recent_actions: List[str], recent_rewards: List[float],
    game: str = "",
) -> str:
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]

    critical = _critical_actions_for(game, valid_actions)

    # Stuck on a single non-scoring action — break the loop, preferring
    # a critical action over a random alternative.
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in valid_actions if a != action]
        if not alternatives:
            return action
        critical_alt = next((c for c in critical if c != action), None)
        if critical_alt is not None:
            return critical_alt
        return random.choice(alternatives)

    # Critical-action dry spell: the policy is exploring varied actions
    # but the env reward is zero AND no critical action has been picked
    # in the recent window.  Force-substitute the first critical action
    # so the learner at least observes the scoring action distribution.
    # Only fires for shooter-class games (those with critical actions).
    if (
        critical
        and len(recent_actions) >= _CRITICAL_ACTION_DRY_SPELL
        and len(recent_rewards) >= _CRITICAL_ACTION_DRY_SPELL
    ):
        window_actions = recent_actions[-_CRITICAL_ACTION_DRY_SPELL:]
        window_rewards = recent_rewards[-_CRITICAL_ACTION_DRY_SPELL:]
        no_critical_used = not any(a in critical for a in window_actions)
        no_reward = sum(r for r in window_rewards if r is not None) <= 0
        # Only override if the proposed action itself is non-critical;
        # never replace a critical action with a critical action.
        if no_critical_used and no_reward and action not in critical:
            return critical[0]

    return action


# ---------------------------------------------------------------------------
# Skill tracker (same logic as qwen3_decision_agent._SkillTracker)
# ---------------------------------------------------------------------------

class _SkillTracker:
    """Protocol-aware skill lifecycle tracker (co-evolution variant).

    Uses predicate-based criteria when available, with keyword fallback.
    Step advancement is condition-based when ``step_checks`` are present.
    """

    def __init__(self):
        self.active_skill_id: Optional[str] = None
        self.active_skill_name: str = ""
        self.steps_on_skill: int = 0
        self.reward_on_skill: float = 0.0
        self.max_skill_duration: int = 10
        self.skill_switches: int = 0
        self._protocol: Optional[Dict[str, Any]] = None
        self._protocol_step_idx: int = 0
        self._success_criteria: List[str] = []
        self._abort_criteria: List[str] = []
        self._predicate_success: List[str] = []
        self._predicate_abort: List[str] = []
        self._prev_reward_on_skill: float = 0.0
        self._prev_steps_on_skill: int = 0
        self._just_switched: bool = False
        self._step_checks: List[str] = []
        self._reselect_reason: str = ""
        self._intrinsic_bonus: float = 0.0

    @property
    def protocol_step_idx(self) -> int:
        return self._protocol_step_idx

    @property
    def total_protocol_steps(self) -> int:
        if self._protocol and isinstance(self._protocol, dict):
            return len(self._protocol.get("steps", []))
        return 0

    def _check_criteria(self, state_text: str, is_abort: bool) -> Optional[str]:
        from decision_agents.protocol_utils import (
            parse_summary_state, check_any_predicate, keyword_match,
        )
        preds = self._predicate_abort if is_abort else self._predicate_success
        texts = self._abort_criteria if is_abort else self._success_criteria
        label = "abort" if is_abort else "success"

        if preds:
            state_dict = parse_summary_state(state_text)
            if check_any_predicate(preds, state_dict):
                return f"{label}:predicate"

        for crit in texts:
            if keyword_match(crit, state_text):
                return f"{label}:{crit[:40]}"

        return None

    def should_reselect(self, guidance: Optional[Dict[str, Any]], state_text: str = "") -> bool:
        self._reselect_reason = ""
        if guidance is None or not guidance.get("skill_id"):
            self._reselect_reason = "no_skill"
            return True
        new_id = guidance["skill_id"]
        if new_id != self.active_skill_id:
            return False
        if self.steps_on_skill >= self.max_skill_duration:
            self._reselect_reason = "duration_exceeded"
            return True
        if self.steps_on_skill >= 4 and self.reward_on_skill <= 0:
            self._reselect_reason = "zero_reward_stall"
            return True
        if state_text:
            abort_reason = self._check_criteria(state_text, is_abort=True)
            if abort_reason:
                self._reselect_reason = abort_reason
                return True
            if self.steps_on_skill >= 2:
                success_reason = self._check_criteria(state_text, is_abort=False)
                if success_reason:
                    self._reselect_reason = success_reason
                    return True
        return False

    def update(self, skill_id: Optional[str], skill_name: str, reward: float,
               state_text: str = ""):
        self._intrinsic_bonus = 0.0
        if skill_id != self.active_skill_id:
            self._prev_reward_on_skill = self.reward_on_skill
            self._prev_steps_on_skill = self.steps_on_skill
            self._just_switched = self.active_skill_id is not None and self.steps_on_skill > 0
            self.active_skill_id = skill_id
            self.active_skill_name = skill_name
            self.steps_on_skill = 1
            self.reward_on_skill = reward
            self.skill_switches += 1
            self._protocol_step_idx = 0
        else:
            self._just_switched = False
            self.steps_on_skill += 1
            self.reward_on_skill += reward
            prev_step_idx = self._protocol_step_idx
            n_steps = self.total_protocol_steps
            if n_steps > 0:
                from decision_agents.protocol_utils import (
                    compute_step_advancement, parse_summary_state,
                )
                state_dict = parse_summary_state(state_text)
                self._protocol_step_idx = compute_step_advancement(
                    self._protocol_step_idx, self._step_checks, state_dict, n_steps,
                )
                if self._protocol_step_idx > prev_step_idx:
                    self._intrinsic_bonus += 0.1

            if state_text and self.active_skill_id:
                success = self._check_criteria(state_text, is_abort=False)
                abort = self._check_criteria(state_text, is_abort=True)
                if success:
                    self._intrinsic_bonus += 0.3
                if abort:
                    self._intrinsic_bonus -= 0.1

    def get_progress_summary(self, state_text: str = "") -> str:
        if not self._protocol or not isinstance(self._protocol, dict):
            return ""
        steps = self._protocol.get("steps", [])
        if not steps:
            return ""
        from decision_agents.protocol_utils import (
            build_progress_summary, parse_summary_state,
        )
        state_dict = parse_summary_state(state_text)
        return build_progress_summary(
            steps, self._step_checks, self._protocol_step_idx, state_dict,
        )

    def set_protocol(self, protocol: Optional[Dict[str, Any]]):
        self._protocol = protocol
        self._protocol_step_idx = 0
        self._success_criteria = []
        self._abort_criteria = []
        self._predicate_success = []
        self._predicate_abort = []
        self._step_checks = []
        if protocol and isinstance(protocol, dict):
            dur = protocol.get("expected_duration", 0)
            if isinstance(dur, (int, float)) and dur > 0:
                self.max_skill_duration = max(int(dur) + 3, 5)
            else:
                self.max_skill_duration = 10
            self._success_criteria = protocol.get("success_criteria", []) or []
            self._abort_criteria = protocol.get("abort_criteria", []) or []
            self._predicate_success = protocol.get("predicate_success", []) or []
            self._predicate_abort = protocol.get("predicate_abort", []) or []
            self._step_checks = protocol.get("step_checks", []) or []
        else:
            self.max_skill_duration = 10


# ---------------------------------------------------------------------------
# Async episode runner
# ---------------------------------------------------------------------------

async def run_episode_async(
    game: str,
    max_steps: int,
    vllm_client: AsyncVLLMClient,
    *,
    skill_bank: Any = None,
    temperature: float = 0.3,
    executor: Optional[ThreadPoolExecutor] = None,
    stuck_window: int = 15,
    min_steps_before_stuck: int = 20,
    vllm_base_urls: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    assigned_role: Optional[str] = None,
    assigned_role_index: Optional[int] = None,
    step_sync: Any = None,
    opponent_model: Optional[str] = None,
    opponent_api_base: Optional[str] = None,
    harness_hook: Any = None,
    reward_logger: Any = None,
    game_profile: Any = None,
    # Block B4 — intention trigger ablation:
    #   * "every-step"  (default): historical behaviour, intention LLM
    #     fires every inner step.
    #   * "sharp-shift": fires only when state delta or urgency
    #     indicates a meaningful shift; otherwise the prev intention
    #     is reused verbatim.
    #   * "disabled":   fires only at step 0; subsequent steps reuse
    #     the bootstrapped intention.
    intention_trigger: str = "every-step",
    # Block B5 — actor-side bank cap.  ``0`` (default) = no cap.
    # When >0, the SkillQueryEngine.select() restricts the candidate
    # pool to the top-K skills by relevance.
    actor_bank_cap_k: int = 0,
) -> EpisodeResult:
    """Run one game episode asynchronously.

    All LLM calls go through *vllm_client* (``await``).
    ``env.step()`` runs in *executor* to avoid blocking the event loop.

    Parameters
    ----------
    skill_bank : object | None
        ``None`` triggers cold-start mode (no skill selection).
    vllm_base_urls : list[str] | None
        Base URLs for vLLM instances (used for LLM opponent policies).
    model_name : str | None
        Model name for LLM opponent policy requests.
    assigned_role : str | None
        Reserved for multi-role games (unused for current game set).
    assigned_role_index : int | None
        Reserved for multi-role games (unused for current game set).
    opponent_model : str | None
        External API model for opponents (e.g. ``"gpt-5-mini"``).
        When set, non-controlled players use this model via API
        instead of vLLM self-play.
    opponent_api_base : str | None
        Base URL for the opponent API (default: OpenRouter).
    game_profile : ``trainer.coevolution._game_schema.GameProfile`` | None
        Per-phase static GameProfile (Path 1).  When supplied, the
        compact rendering (goal / win_signal / hazards / key_actions /
        failure_modes) is prepended to ``SYSTEM_PROMPT`` and
        ``SKILL_SELECTION_SYSTEM_PROMPT`` for every step of this
        episode.  Adds ~150 tokens to each actor / skill-selection
        prompt; no per-step LLM cost.  ``None`` (default) preserves
        the legacy hard-coded prompt.
    """
    imp = _lazy_imports()
    GAME_CONFIGS = imp["GAME_CONFIGS"]
    make_gaming_env = imp["make_gaming_env"]
    make_orak_env = imp["make_orak_env"]
    GamingAgentNLWrapper = imp["GamingAgentNLWrapper"]
    HARD_SUMMARY_CHAR_LIMIT = imp["HARD_SUMMARY_CHAR_LIMIT"]
    extract_game_facts = imp["extract_game_facts"]
    compact_text_observation = imp["compact_text_observation"]
    strip_think_tags = imp["strip_think_tags"]

    # Path 1 wiring.  Imported at function entry so a refactor of
    # ``_state_to_markup`` / ``_game_schema`` cannot wedge episode
    # collection at module-import time, and so the ``GameProfile``
    # rendering is computed once per episode (not per step) for the
    # actor-prompt prefix.
    from trainer.coevolution._state_to_markup import state_to_markup
    if game_profile is not None:
        try:
            from trainer.coevolution._game_schema import render_for_actor_prompt
            _profile_prefix = render_for_actor_prompt(game_profile) + "\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GameProfile render failed for game=%s: %s — "
                "proceeding without profile prefix",
                game, exc,
            )
            _profile_prefix = ""
    else:
        _profile_prefix = ""

    loop = asyncio.get_running_loop()
    t0 = time.monotonic()

    game_cfg = GAME_CONFIGS.get(game)
    episode_id = f"{game}_{uuid.uuid4().hex[:8]}"
    exe = executor

    if game in ORAK_GAMES_SET:
        SubprocessEnv = imp["SubprocessEnv"]
        use_subprocess = (
            make_orak_env is None or game in ORAK_SUBPROCESS_GAMES
        )
        if use_subprocess:
            logger.info(
                "Using SubprocessEnv for %s (orak_import=%s, forced=%s)",
                game, make_orak_env is not None, game in ORAK_SUBPROCESS_GAMES,
            )
            if exe:
                env = await loop.run_in_executor(
                    exe, SubprocessEnv, game, max_steps,
                )
            else:
                env = SubprocessEnv(game=game, max_steps=max_steps)
        elif exe:
            env = await loop.run_in_executor(
                exe, make_orak_env, game, max_steps,
            )
        else:
            env = make_orak_env(game, max_steps=max_steps)

    elif game in GYMV_TEMPORAL_GAMES_SET:
        # Gym-V Temporal/* (stable-retro Genesis).  Always go through
        # SubprocessEnv: stable-retro's emulator is a process singleton
        # ("Cannot create multiple emulator instances per process"), so
        # in-process concurrent episodes drop 7/8 rollouts — see
        # GYMV_SUBPROCESS_GAMES docstring above.  The worker calls
        # ``make_gymv_temporal_env`` inside the child (frame_skip=8,
        # max_steps from caller) so the obs_nl / info contract is
        # preserved 1:1 with the in-process path.
        SubprocessEnv = imp["SubprocessEnv"]
        if game not in GYMV_SUBPROCESS_GAMES:
            raise RuntimeError(
                f"Gym-V Temporal/* env requested ({game!r}) but gym_v / "
                "stable-retro / Mega Drive ROMs are not installed; run "
                "install/install_gymv.sh + install/gymv_temporal_patch/"
                "apply_patch.sh and retry."
            )
        logger.info(
            "Using SubprocessEnv[gymv] for %s (emulator-singleton isolation)",
            game,
        )
        if exe:
            env = await loop.run_in_executor(
                exe,
                lambda: SubprocessEnv(
                    game=game, max_steps=max_steps, env_kind="gymv",
                ),
            )
        else:
            env = SubprocessEnv(game=game, max_steps=max_steps, env_kind="gymv")

    else:
        if exe:
            base_env = await loop.run_in_executor(
                exe, make_gaming_env, game, max_steps,
            )
        else:
            base_env = make_gaming_env(game=game, max_steps=max_steps)

        if game == "tetris":
            from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
            env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))
        else:
            env = GamingAgentNLWrapper(base_env)

    # ── Resolve role / side metadata for multi-role games ───────
    _ep_role = ""
    _ep_side = ""
    _ep_role_idx = -1

    if exe:
        obs_nl, info = await loop.run_in_executor(exe, env.reset)
    else:
        obs_nl, info = env.reset()

    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")
    # Path 1: emit the unified <state> markup so it's available to any
    # downstream consumer (skill_selection, Crafter, Harness validator,
    # eval scorers).  Same call site is used at every env.step below so
    # train and eval see byte-identical schema for the same observation.
    try:
        info["state_markup"] = state_to_markup(
            obs_nl=obs_nl, info=info, game=game, step=0,
        )
    except Exception as _markup_exc:  # noqa: BLE001
        logger.debug("state_to_markup failed at reset: %s", _markup_exc)
    current_info = info

    bank_available = skill_bank is not None and (
        hasattr(skill_bank, "__len__") and len(skill_bank) > 0
        or hasattr(skill_bank, "skill_ids") and len(list(skill_bank.skill_ids)) > 0
    )

    grpo_records: List[GRPORecord] = []
    experiences: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    current_intention = ""
    prev_summary_state = ""
    prev_intention = ""

    recent_actions: List[str] = []
    recent_rewards: List[float] = []
    tag_history: List[str] = []
    skill_tracker = _SkillTracker()
    last_guidance: Optional[Dict[str, Any]] = None
    last_candidates: List[Dict[str, Any]] = []
    last_chosen_idx = 0
    last_skill_reasoning: Optional[str] = None

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        # ── 1. summary_state (deterministic, 0 LLM calls) ────────
        summary_state = _generate_summary_state(
            obs_nl, game_name=game,
            step_idx=step_count, total_steps=max_steps,
            reward=total_reward,
        )

        compact = compact_text_observation(obs_nl, max_chars=200)
        state_text = compact if compact else obs_nl[:1000]
        game_label = game.replace("_", " ")
        delta = _compute_state_delta(prev_summary_state, summary_state)
        delta_line = f"Changed since last step: {delta}\n" if delta else ""

        # Pre-compute urgency (needed by both intention and action prompts)
        urgency = _detect_urgency(summary_state, game)

        # ── 2+3+4. summary_prose, skill_selection, intention (PARALLEL)
        summary_prompt = (
            f"{game_label}: {state_text}\n"
            f"{delta_line}"
            f"Key strategic note about the current threat or opportunity "
            f"(max 10 words, be specific to what changed).\nNote:"
        )
        summary_coro = vllm_client.generate_chat(
            [{"role": "user", "content": summary_prompt}],
            adapter="base", temperature=0.2, max_tokens=64,
        )

        # Block B4 — intention trigger ablation.  When the trigger
        # fires, we generate a fresh intention via the LLM; when it
        # doesn't, we synthesise a coroutine that returns
        # ``prev_intention`` so the downstream ``asyncio.gather`` shape
        # stays unchanged.
        def _should_regen_intention() -> bool:
            if intention_trigger == "every-step":
                return True
            if intention_trigger == "disabled":
                return step_count == 0 or not prev_intention
            # "sharp-shift": fire on bootstrap, on detected urgency, or
            # on a non-trivial state delta.  Heuristic chosen so the
            # ablation matches the §4.1 definition without a separate
            # 35B classifier.
            if not prev_intention:
                return True
            if urgency:
                return True
            if delta and len(delta) >= 8:
                return True
            return False

        if _should_regen_intention():
            intention_coro = _generate_intention(
                vllm_client,
                state_text=state_text,
                game_name=game,
                summary_state=summary_state,
                prev_intention=prev_intention,
                prev_summary_state=prev_summary_state,
                delta=delta,
                urgency=urgency,
                skill_guidance=last_guidance,
                last_action=recent_actions[-1] if recent_actions else "start",
                tag_history=tag_history,
            )
        else:
            async def _passthrough_intention(_keep: str = prev_intention) -> str:
                return _keep
            intention_coro = _passthrough_intention()

        need_reselect = skill_tracker.should_reselect(
            last_guidance, state_text=summary_state or obs_nl,
        )
        skill_select_prompt: Optional[str] = None
        skill_coro = None

        # T2.15: pre-bind `harness_filter_diag` at the per-step scope.
        # Previously initialized only inside the `if bank_available …:`
        # block below, but read unconditionally at the experience-dict
        # assembly site (~L1384) — when the skill bank was empty (cold-
        # start step 0) or sticky-guidance kept us out of the inner
        # block, we hit `UnboundLocalError: harness_filter_diag`, which
        # collapsed every rollout in the wave (8/8 episodes ERR with 0
        # GRPO records).  Mirror the existing `harness_validate_diag`
        # outer init below.
        harness_filter_diag: Optional[Dict[str, Any]] = None

        if bank_available and (need_reselect or last_guidance is None):
            facts = extract_game_facts(obs_nl, game)
            step_structured = {k: v for k, v in facts.items() if v}

            from scripts.qwen3_decision_agent import get_top_k_skill_candidates
            candidates = get_top_k_skill_candidates(
                skill_bank,
                summary_state or obs_nl,
                game_name=game,
                intention=current_intention,
                structured_state=step_structured if step_structured else structured_state,
                top_k=3,
                bank_cap_k=int(actor_bank_cap_k or 0),
            )

            # Pre-LLM harness eligibility filter (PLAN-HARNESS §5.2). When
            # `harness_hook` is supplied, candidates the harness vetoes
            # (status / domain / task / adapter / can_handle) are dropped
            # *before* the skill_selection LLM sees them, and their veto
            # reason is observed by the hook's RejectedSkillSink for the
            # Crafter to consume in Phase B′. See
            # `trainer/coevolution/_harness_hook.py` for the full
            # contract.
            if harness_hook is not None:
                try:
                    _hstate = harness_hook.state_for_step(
                        game=game,
                        summary_state=summary_state,
                        intention=current_intention,
                        inner_step=step_count,
                        outer_step=step_count,
                    )
                    candidates, harness_filter_diag = harness_hook.filter_candidates(
                        list(candidates), _hstate,
                        episode_id=episode_id,
                    )
                except Exception as _hexc:                       # noqa: BLE001
                    logger.debug(
                        "harness_hook.filter_candidates failed at step=%d: %s — "
                        "falling back to unfiltered candidates",
                        step_count, _hexc,
                    )
                    harness_filter_diag = {"harness_error": repr(_hexc)}

            if candidates and len(candidates) >= 2:
                candidates_text = _format_candidates_for_selection(candidates)
                user_content = (
                    f"Game state:\n{(summary_state or obs_nl)[:3000]}\n\n"
                    f"Current intention: {current_intention[:500]}\n\n"
                    f"Available strategies (pick ONE by number):\n{candidates_text}\n\n"
                    f"Choose the best strategy. Output REASONING then SKILL number."
                )
                skill_select_prompt = (
                    _profile_prefix
                    + SKILL_SELECTION_SYSTEM_PROMPT
                    + "\n" + user_content
                )
                skill_coro = vllm_client.generate_chat(
                    [{"role": "user", "content": skill_select_prompt}],
                    adapter="skill_selection",
                    temperature=temperature, max_tokens=128,
                    stop=["\n\nAvailable", "\n\nGame state", "\n\n---"],
                )

        # Sync with other episodes so LLM requests hit vLLM together
        # (batch-size-1 throughput is 10-20x worse than batched).
        if step_sync is not None:
            await step_sync.arrive()

        # Fire all LLM calls concurrently
        if skill_coro is not None:
            summary_result, assigned_subgoal, sk_result = await asyncio.gather(
                summary_coro, intention_coro, skill_coro,
            )
        else:
            summary_result, assigned_subgoal = await asyncio.gather(
                summary_coro, intention_coro,
            )
            sk_result = None

        # Process summary result
        note = strip_think_tags(summary_result.text).strip() if summary_result.text else ""
        note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
        current_summary = f"{summary_state} | note={note}" if note else summary_state
        current_summary = current_summary[:HARD_SUMMARY_CHAR_LIMIT]

        # Post-LLM harness validate_invocation (PLAN-UNIFIED §3.4).
        # Wraps the LLM's chosen skill in a structured second-pass veto.
        # When vetoed we walk to the next candidate; when no eligible
        # candidate survives, we drop guidance and run unguided.
        harness_validate_diag: Optional[Dict[str, Any]] = None

        def _harness_validate(_idx: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
            """Validate `candidates[_idx]` via the harness.

            Returns ``(ok, diag)``. ``ok=True`` when no harness hook is
            configured or when validation admits.
            """
            if harness_hook is None:
                return True, None
            try:
                _sid = (candidates[_idx] or {}).get("skill_id")
                _hstate2 = harness_hook.state_for_step(
                    game=game,
                    summary_state=summary_state,
                    intention=current_intention,
                    inner_step=step_count,
                    outer_step=step_count,
                )
                # Path 4 — pass ``episode_id`` for per-episode LLM
                # validator caching (no-op when validator is off).
                # ``inner_step`` is plumbed for block A2 logging only.
                _ok, _d = harness_hook.validate_choice(
                    _sid, _hstate2,
                    episode_id=episode_id,
                    inner_step=step_count,
                )
                return bool(_ok), _d
            except Exception as _vexc:                          # noqa: BLE001
                logger.debug(
                    "harness_hook.validate_choice failed at step=%d "
                    "idx=%d: %s — admitting",
                    step_count, _idx, _vexc,
                )
                return True, {"status": "harness_error", "error": repr(_vexc)}

        # Process skill selection result
        if bank_available and (need_reselect or last_guidance is None):
            if sk_result is not None and candidates and len(candidates) >= 2:
                chosen_idx, skill_reasoning = _parse_skill_selection(
                    sk_result.text, len(candidates), candidates,
                )
                # Walk through candidates starting at the LLM's pick; if
                # the harness vetoes it, fall through to the next one.
                _scan_order = [chosen_idx] + [
                    i for i in range(len(candidates)) if i != chosen_idx
                ]
                _picked: Optional[int] = None
                _last_v: Optional[Dict[str, Any]] = None
                for _i in _scan_order:
                    _ok, _d = _harness_validate(_i)
                    if _ok:
                        _picked = _i
                        _last_v = _d
                        if _i != chosen_idx:
                            skill_reasoning = (
                                f"{skill_reasoning or 'LLM-selected'} "
                                f"(harness re-routed from idx={chosen_idx} "
                                f"→ idx={_i})"
                            )
                        break
                    _last_v = _d
                harness_validate_diag = _last_v
                if _picked is None:
                    guidance = None
                    last_candidates = candidates
                    last_chosen_idx = chosen_idx
                    last_skill_reasoning = "harness vetoed all candidates"
                else:
                    chosen_idx = _picked
                    guidance = candidates[chosen_idx]
                    if skill_reasoning:
                        guidance["why_selected"] = skill_reasoning
                    last_candidates = candidates
                    last_chosen_idx = chosen_idx
                    last_skill_reasoning = skill_reasoning
                    skill_tracker.set_protocol(guidance.get("protocol"))
                    _chosen_sid = guidance.get("skill_id")
                    if _chosen_sid and hasattr(skill_bank, "selection_tracker"):
                        skill_bank.selection_tracker.increment(_chosen_sid)
            elif candidates:
                _picked2: Optional[int] = None
                _last_v2: Optional[Dict[str, Any]] = None
                for _i in range(len(candidates)):
                    _ok, _d = _harness_validate(_i)
                    if _ok:
                        _picked2 = _i
                        _last_v2 = _d
                        break
                    _last_v2 = _d
                harness_validate_diag = _last_v2
                if _picked2 is None:
                    guidance = None
                    last_candidates = candidates
                    last_chosen_idx = 0
                    last_skill_reasoning = "harness vetoed all candidates"
                else:
                    guidance = candidates[_picked2]
                    last_candidates = candidates
                    last_chosen_idx = _picked2
                    last_skill_reasoning = (
                        "only one candidate" if _picked2 == 0
                        else f"harness re-routed to idx={_picked2}"
                    )
                    skill_tracker.set_protocol(guidance.get("protocol"))
                    _chosen_sid = guidance.get("skill_id")
                    if _chosen_sid and hasattr(skill_bank, "selection_tracker"):
                        skill_bank.selection_tracker.increment(_chosen_sid)
            else:
                guidance = None
                last_candidates = []
                last_chosen_idx = 0
                last_skill_reasoning = None

            last_guidance = guidance
        elif not bank_available:
            guidance = None
            last_guidance = None
        else:
            guidance = last_guidance

        # ── 5. Action selection (action_taking LoRA) ────────────
        # The intention tag comes from the base model (step 4 above).
        # We inject it as "Assigned subgoal" so the LoRA follows it.
        urgency_line = f"URGENCY: {urgency}\n" if urgency else ""
        skill_context = ""
        if guidance and guidance.get("skill_id"):
            sk_name = guidance.get("skill_name", guidance["skill_id"])
            sk_hint = guidance.get("execution_hint", "")
            skill_context = f"Active skill: {sk_name}"
            if sk_hint:
                skill_context += f" — {sk_hint[:100]}"
            skill_context += "\n"

        recent_context = _build_recent_context(recent_actions, recent_rewards)
        _use_raw_obs = getattr(env, "_is_macro_action", False)
        _use_rich_obs = getattr(env, "_has_rich_observation", False)
        if _use_raw_obs:
            summary_for_action = obs_nl[:4000]
        elif _use_rich_obs and current_info:
            summary_for_action = _build_rich_state_observation(
                current_info, summary_state,
            )
        else:
            summary_for_action = (
                current_summary if current_summary else obs_nl[:4000]
            )
        skill_text = _format_skill_guidance_for_prompt(
            guidance, skill_tracker.protocol_step_idx,
            progress_summary=skill_tracker.get_progress_summary(summary_state),
        )

        imp_tags = imp["SUBGOAL_TAGS"]
        tags_str = "|".join(imp_tags)
        # Surface critical actions (e.g. B = fire on shooter games) as
        # a single-line in-context prior so the LLM doesn't have to
        # rediscover the action vocabulary via GRPO alone.  Schema
        # source: trainer.coevolution.config.GAME_CRITICAL_ACTIONS.
        _critical_hint = ""
        _critical_for_game = _critical_actions_for(game, step_actions)
        if _critical_for_game:
            _critical_hint = (
                f"Critical actions for this game (use frequently when scoring): "
                f"{', '.join(_critical_for_game)}.\n"
            )
        action_user = (
            f"Game state:\n\n{summary_for_action}\n\n"
            f"Subgoal: {assigned_subgoal}\n"
            f"{urgency_line}{skill_context}{recent_context}{_critical_hint}"
            f"Available actions (pick ONE by number):\n{_format_numbered_actions(step_actions)}\n\n"
            f"Output ACTION number."
        )
        action_prompt = (
            _profile_prefix + SYSTEM_PROMPT + skill_text + "\n" + action_user
        )

        if step_sync is not None:
            await step_sync.arrive()

        action_result = await vllm_client.generate_chat(
            [{"role": "user", "content": action_prompt}],
            adapter="action_taking",
            temperature=temperature, max_tokens=128,
            stop=["\n\nAvailable", "\n\nGame state", "\n\n---", "\n\n"],
        )
        action, reasoning, parsed_intention = _parse_action_response(
            action_result.text, step_actions,
        )
        current_intention = parsed_intention or assigned_subgoal or prev_intention or f"[SETUP] {game}"
        action = _apply_anti_repetition(action, step_actions, recent_actions, recent_rewards, game=game)

        # Block A4: stream the per-step intention update.  ``switched``
        # = textual inequality;  ``sharp_shift`` = tag-prefix change OR
        # high urgency (a working definition of §4.1's "sharp shift" we
        # can sharpen post-hoc).  Drives intention-trigger ablation +
        # the §4.1 method-section sharp-shift threshold definition.
        try:
            from trainer.coevolution._run_loggers import (  # noqa: WPS433
                log_intention_switch,
            )
            _prev_tag_m = _TAG_RE.match(prev_intention) if prev_intention else None
            _curr_tag_m = _TAG_RE.match(current_intention) if current_intention else None
            _prev_tag = _prev_tag_m.group(1).upper() if _prev_tag_m else ""
            _curr_tag = _curr_tag_m.group(1).upper() if _curr_tag_m else ""
            _switched = bool(prev_intention) and (current_intention != prev_intention)
            _sharp = (
                bool(_prev_tag)
                and bool(_curr_tag)
                and _prev_tag != _curr_tag
            ) or bool(urgency)
            # Trainer outer step is best-effort: read from the harness
            # hook when wired (orchestrator sets it via for_game).  When
            # there is no hook (e.g. cold-start mode), emit step=-1 and
            # post-hoc joiners can correlate via timestamp.
            _outer = -1
            if harness_hook is not None and hasattr(harness_hook, "_trainer_step"):
                try:
                    _outer = int(getattr(harness_hook, "_trainer_step", -1))
                except Exception:
                    _outer = -1
            log_intention_switch(
                step=_outer,
                episode_id=episode_id,
                game=game,
                inner_step=step_count,
                prev_intention=prev_intention,
                new_intention=current_intention,
                switched=_switched,
                sharp_shift=_sharp,
                summary_state_delta=delta or "",
                urgency=urgency or "",
            )
        except Exception:  # noqa: BLE001
            pass

        # ── 6. env.step() (in executor) ─────────────────────────
        try:
            if exe:
                next_obs_nl, reward, terminated, truncated, next_info = await loop.run_in_executor(
                    exe, env.step, action,
                )
            else:
                next_obs_nl, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            logger.warning("env.step failed at step %d: %s", step_count, e)
            break

        done = terminated or truncated
        raw_env_reward = next_info.get("raw_env_reward", float(reward))
        total_reward += reward
        next_action_names = next_info.get("action_names", action_names)
        next_structured_state = next_info.get("structured_state")
        try:
            next_info["state_markup"] = state_to_markup(
                obs_nl=next_obs_nl, info=next_info, game=game,
                step=step_count + 1,
            )
        except Exception as _markup_exc:  # noqa: BLE001
            logger.debug(
                "state_to_markup failed at step %d: %s",
                step_count + 1, _markup_exc,
            )

        recent_actions.append(str(action))
        recent_rewards.append(float(reward))

        skill_id = guidance.get("skill_id") if guidance else None
        skill_name_val = guidance.get("skill_name", "") if guidance else ""
        skill_tracker.update(skill_id, skill_name_val, float(reward),
                             state_text=summary_state)

        # ── 7. Record GRPO I/O ───────────────────────────────────
        if action_prompt:
            _format_failed = isinstance(action, _ActionFallback)
            try:
                action_num = step_actions.index(action) + 1
            except ValueError:
                action_num = 1
            subgoal_line = f"SUBGOAL: {current_intention}\n" if current_intention else ""
            if _format_failed:
                action_completion = action_result.text.strip()[:150]
                _action_reward = 0.0
            else:
                action_completion = f"{subgoal_line}REASONING: {reasoning or 'Expert play.'}\nACTION: {action_num}"
                _action_reward = float(reward) + skill_tracker._intrinsic_bonus + 1.0
                # Track shaping composition so the orchestrator's per-step
                # flush surfaces the (intrinsic + const) / raw_env ratio
                # that GRPO actually sees.  See run-loggers
                # ``record_shaping_signal`` / ``flush_shaping_ratio`` for
                # the warning threshold and the post-mortem rationale
                # (TF3 phase-1 collapse: 83% zero-reward episodes meant
                # the +1.0 survival constant dominated GRPO advantages).
                try:
                    from trainer.coevolution._run_loggers import (  # noqa: WPS433
                        record_shaping_signal,
                    )
                    record_shaping_signal(
                        game=game,
                        raw_env=float(raw_env_reward),
                        intrinsic=float(skill_tracker._intrinsic_bonus),
                        constant_offset=1.0,
                    )
                except Exception:                                # pragma: no cover
                    pass
            _action_meta = {
                "chosen_action": str(action),
                "available_actions": list(step_actions),
                "summary_state": summary_state,
                "intention": current_intention,
                "assigned_intention": assigned_subgoal,
                "intention_source": "base_model",
                "active_skill": skill_id,
                "intrinsic_bonus": skill_tracker._intrinsic_bonus,
                "raw_env_reward": raw_env_reward,
                "placement_metrics": next_info.get("placement_metrics"),
                "board_stats": next_info.get("board_stats"),
            }
            grpo_records.append(GRPORecord(
                adapter="action_taking", game=game, episode_id=episode_id, step=step_count,
                prompt=action_prompt, completion=action_completion, reward=_action_reward,
                metadata=_action_meta,
            ))
            # T2.4 single-sink: mirror the same scalar + metadata into
            # ``RewardLogger`` so eval and training read from one source.
            # Logger errors are non-fatal — we never let a logging hiccup
            # break a rollout. Metadata is kept *minimal* (chosen_action +
            # raw_env_reward) to avoid bloating the JSONL; full per-step
            # metadata stays on the in-memory ``GRPORecord``.
            if reward_logger is not None:
                try:
                    reward_logger.log_grpo_record(
                        episode_id=episode_id,
                        adapter="action_taking",
                        step=step_count,
                        reward=_action_reward,
                        game=game,
                        metadata={
                            "chosen_action": _action_meta["chosen_action"],
                            "raw_env_reward": _action_meta["raw_env_reward"],
                            "active_skill": _action_meta["active_skill"],
                            "intrinsic_bonus": _action_meta["intrinsic_bonus"],
                        },
                    )
                except Exception:  # pragma: no cover  (defensive)
                    logger.exception("reward_logger.log_grpo_record(action_taking) failed")

        if skill_select_prompt and last_candidates and len(last_candidates) >= 2:
            sk_completion = (
                f"REASONING: {last_skill_reasoning}\nSKILL: {last_chosen_idx + 1}"
                if last_skill_reasoning
                else f"SKILL: {last_chosen_idx + 1}"
            )
            # Protocol-aware delayed reward at skill-switch time.
            # Uses the multi-signal reward from rewards.py instead of
            # a simple clamped env reward, incorporating efficiency,
            # success/abort criteria, and RAG confidence.
            if skill_tracker._just_switched and skill_tracker._prev_steps_on_skill > 0:
                from skill_agents.grpo.rewards import skill_selection_reward
                _reason = skill_tracker._reselect_reason
                sk_reward = skill_selection_reward(
                    reward_on_skill=skill_tracker._prev_reward_on_skill,
                    steps_on_skill=skill_tracker._prev_steps_on_skill,
                    max_skill_duration=skill_tracker.max_skill_duration,
                    success_met=_reason.startswith("success:") if _reason else False,
                    abort_triggered=_reason.startswith("abort:") if _reason else False,
                    confidence=0.5,
                )
            else:
                sk_reward = min(1.0, max(0.0, float(reward)))
            _sk_meta = {
                "chosen_idx": last_chosen_idx,
                "skill_candidates": [c.get("skill_id") for c in last_candidates],
                "chosen_skill_id": (
                    last_candidates[last_chosen_idx].get("skill_id")
                    if last_chosen_idx < len(last_candidates) else None
                ),
                "summary_state": summary_state,
                "intention": current_intention,
                "reselect_reason": skill_tracker._reselect_reason,
            }
            grpo_records.append(GRPORecord(
                adapter="skill_selection", game=game, episode_id=episode_id, step=step_count,
                prompt=skill_select_prompt, completion=sk_completion, reward=sk_reward,
                metadata=_sk_meta,
            ))
            # T2.4 single-sink mirror — see the action_taking branch.
            if reward_logger is not None:
                try:
                    reward_logger.log_grpo_record(
                        episode_id=episode_id,
                        adapter="skill_selection",
                        step=step_count,
                        reward=sk_reward,
                        game=game,
                        metadata={
                            "chosen_skill_id": _sk_meta["chosen_skill_id"],
                            "reselect_reason": _sk_meta["reselect_reason"],
                            "n_candidates": len(_sk_meta["skill_candidates"]),
                        },
                    )
                except Exception:  # pragma: no cover  (defensive)
                    logger.exception("reward_logger.log_grpo_record(skill_selection) failed")

        _exp_dict: Dict[str, Any] = {
            "step": step_count,
            "state": obs_nl,
            "action": str(action),
            "reward": float(reward),
            "raw_env_reward": raw_env_reward,
            "next_state": next_obs_nl,
            "done": done,
            "intention": current_intention,
            "summary_state": summary_state,
            "skill_id": skill_id,
        }
        if next_info.get("board_stats"):
            _exp_dict["board_stats"] = next_info["board_stats"]
        if _ep_role:
            _exp_dict["role"] = _ep_role
            _exp_dict["side"] = _ep_side
        # Harness diagnostics — surfaces eligibility filter / validate_invocation
        # output for the Crafter hook (Phase B′) to drain into
        # `SkillRecord.false_binding_patterns` via `RejectedSkillSink`.
        # Empty dict when the harness wasn't enabled this step.
        if harness_filter_diag is not None or harness_validate_diag is not None:
            _exp_dict["harness"] = {
                "filter": harness_filter_diag,
                "validate": harness_validate_diag,
            }
        experiences.append(_exp_dict)

        prev_summary_state = summary_state
        prev_intention = current_intention
        m = _TAG_RE.match(current_intention) if current_intention else None
        if m:
            tag_history.append(m.group(1).upper())
        obs_nl = next_obs_nl
        current_info = next_info
        action_names = next_action_names
        structured_state = next_structured_state
        step_count += 1

        if done:
            break

        # Early termination: stuck detection
        if (step_count >= min_steps_before_stuck
                and len(recent_rewards) >= stuck_window
                and sum(recent_rewards[-stuck_window:]) <= 0):
            logger.debug("Episode %s stuck at step %d, terminating early", episode_id, step_count)
            break

    if step_sync is not None:
        step_sync.depart()

    try:
        env.close()
    except Exception:
        pass

    for rec in grpo_records:
        rec.episode_length = max(step_count, 1)

    wall_time = time.monotonic() - t0
    return EpisodeResult(
        game=game,
        episode_id=episode_id,
        steps=step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        skill_switches=skill_tracker.skill_switches,
        grpo_records=grpo_records,
        experiences=experiences,
        wall_time_s=wall_time,
        role=_ep_role,
        side=_ep_side,
        role_index=_ep_role_idx,
    )
