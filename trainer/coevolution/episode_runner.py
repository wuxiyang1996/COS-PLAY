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


# ---------------------------------------------------------------------------
# Lazy imports — these pull in heavyweight packages that live in the project
# ---------------------------------------------------------------------------

_IMPORTS_CACHE: Dict[str, Any] = {}

# Games that use Orak env (evaluate_orak.orak_nl_wrapper.make_orak_env)
ORAK_GAMES_SET = {"super_mario", "pokemon_red"}
# Orak games that MUST use SubprocessEnv (nes_py / NumPy 2.x incompatibility)
ORAK_SUBPROCESS_GAMES = {"super_mario"}
# Games that use AgentEvolver wrappers (env_wrappers)
EVOLVER_GAMES_SET = {"diplomacy", "avalon"}
# Games that use GamingAgent make_gaming_env
GAMINGAGENT_GAMES = {
    "twenty_forty_eight", "sokoban", "candy_crush", "tetris",
}


def _lazy_imports():
    """Return project modules, imported once and cached."""
    global _IMPORTS_CACHE
    if not _IMPORTS_CACHE:
        from evaluate_gamingagent.game_configs import GAME_CONFIGS
        from evaluate_gamingagent.gym_like import make_gaming_env
        from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
        try:
            from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
        except ImportError:
            SokobanNLWrapper = None

        # Orak env (Super Mario, etc.)
        try:
            from evaluate_orak.orak_nl_wrapper import make_orak_env
        except ImportError:
            make_orak_env = None

        from env_wrappers.subprocess_env import SubprocessEnv

        # Evolver wrappers (Diplomacy, Avalon)
        try:
            from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper
        except ImportError:
            DiplomacyNLWrapper = None
        try:
            from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper
        except ImportError:
            AvalonNLWrapper = None

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
            "SubprocessEnv": SubprocessEnv,
            "GamingAgentNLWrapper": GamingAgentNLWrapper,
            "SokobanNLWrapper": SokobanNLWrapper,
            "DiplomacyNLWrapper": DiplomacyNLWrapper,
            "AvalonNLWrapper": AvalonNLWrapper,
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
# Adapters for multi-agent / complex-action games
# ---------------------------------------------------------------------------

class _AvalonAdapter:
    """Wraps AvalonNLWrapper (single-agent) to provide discrete action_names."""

    def __init__(self, env):
        self._env = env
        self._last_info: dict = {}

    def reset(self):
        obs, info = self._env.reset()
        info["action_names"] = self._build_actions(info)
        return obs, info

    def step(self, action_str: str):
        real = self._convert(action_str, self._last_info)
        obs, reward, term, trunc, info = self._env.step(real)
        info["action_names"] = self._build_actions(info)
        self._last_info = info
        return obs, reward, term, trunc, info

    def _build_actions(self, info):
        self._last_info = info
        phase = info.get("phase", -1)
        if phase == 1:
            return ["approve", "reject"]
        if phase == 2:
            return ["pass", "fail"]
        if phase == 0:
            team_size = info.get("team_size", 2)
            n = self._env.num_players
            from itertools import combinations
            combos = list(combinations(range(n), team_size))
            return [",".join(str(p) for p in c) for c in combos[:15]]
        if phase == 3:
            return [str(i) for i in range(self._env.num_players)]
        return ["wait"]

    def _convert(self, action_str: str, info):
        phase = info.get("phase", -1)
        if phase == 0:
            try:
                return [int(x) for x in action_str.split(",")]
            except ValueError:
                ts = info.get("team_size", 2)
                return list(range(ts))
        if phase == 3:
            try:
                return int(action_str)
            except ValueError:
                return 0
        return action_str

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def done(self):
        return self._env.done


class _DiplomacyAdapter:
    """Wraps DiplomacyNLWrapper (single-agent) to provide discrete action_names.

    Presents each unit's possible orders as flat choices.  The LLM picks one
    order; unmentioned units use random valid orders.
    """

    def __init__(self, env):
        self._env = env
        self._last_info = {}

    def reset(self):
        obs, info = self._env.reset()
        info["action_names"] = self._build_actions(info)
        self._last_info = info
        return obs, info

    def step(self, action_str: str):
        orders = self._make_orders(action_str)
        obs, reward, term, trunc, info = self._env.step(orders)
        info["action_names"] = self._build_actions(info)
        self._last_info = info
        return obs, reward, term, trunc, info

    def _build_actions(self, info):
        cp = self._env._controlled_power
        possible = info.get("possible_orders", {}).get(cp, {})
        flat: List[str] = []
        for loc, orders in possible.items():
            flat.extend(orders[:8])
        if not flat:
            return ["hold"]
        return flat[:20]

    def _make_orders(self, action_str: str) -> list:
        cp = self._env._controlled_power
        possible = self._last_info.get("possible_orders", {}).get(cp, {})
        orders: List[str] = []
        used = False
        for loc, loc_orders in possible.items():
            if not used and action_str in loc_orders:
                orders.append(action_str)
                used = True
            else:
                if loc_orders:
                    orders.append(random.choice(loc_orders))
        return orders

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def done(self):
        return self._env.done


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class _PokemonExploreTracker:
    """Tracks tile/map exploration for Pokemon Red reward shaping.

    Produces per-step bonuses so that GRPO sees meaningful reward
    variance instead of the flat +1.0 that starves the training signal.

    Reward components:
      +NEW_TILE_BONUS   first visit to a tile
      +NEW_MAP_BONUS    first visit to a map
      -REVISIT_PENALTY  revisiting a tile seen in the last REVISIT_WINDOW steps
      -STAGNATION_PENALTY  no new tile discovered for STAGNATION_WINDOW steps
      -BOUNCE_PENALTY   returning to a map visited in the last BOUNCE_WINDOW steps
    """

    NEW_TILE_BONUS = 0.5
    NEW_MAP_BONUS = 3.0
    REVISIT_PENALTY = -0.2
    STAGNATION_WINDOW = 5
    STAGNATION_PENALTY = -0.3
    REVISIT_WINDOW = 4
    BOUNCE_PENALTY = -0.4
    BOUNCE_WINDOW = 6

    def __init__(self) -> None:
        self.visited_tiles: set = set()
        self.visited_maps: set = set()
        self.recent_tiles: List[Any] = []
        self.recent_maps: List[str] = []
        self.steps_since_new_tile: int = 0

    def update(self, info: Dict[str, Any]) -> float:
        """Return exploration bonus (or penalty) for this step."""
        map_name = info.get("pokemon_map_name", "")
        px = info.get("pokemon_player_x")
        py = info.get("pokemon_player_y")
        if not map_name or px is None or py is None:
            self.steps_since_new_tile += 1
            if self.steps_since_new_tile >= self.STAGNATION_WINDOW:
                return self.STAGNATION_PENALTY
            return 0.0

        bonus = 0.0
        tile = (map_name, int(px), int(py))

        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            bonus += self.NEW_TILE_BONUS
            self.steps_since_new_tile = 0
        else:
            self.steps_since_new_tile += 1
            if tile in self.recent_tiles[-self.REVISIT_WINDOW:]:
                bonus += self.REVISIT_PENALTY
            if self.steps_since_new_tile >= self.STAGNATION_WINDOW:
                bonus += self.STAGNATION_PENALTY

        if (self.recent_maps
                and map_name != self.recent_maps[-1]
                and map_name in self.recent_maps[-self.BOUNCE_WINDOW:]):
            bonus += self.BOUNCE_PENALTY

        if map_name not in self.visited_maps:
            self.visited_maps.add(map_name)
            bonus += self.NEW_MAP_BONUS

        self.recent_tiles.append(tile)
        if len(self.recent_tiles) > 50:
            self.recent_tiles = self.recent_tiles[-50:]

        self.recent_maps.append(map_name)
        if len(self.recent_maps) > 50:
            self.recent_maps = self.recent_maps[-50:]

        return bonus


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
# Stage / side inference for multi-role games
# ---------------------------------------------------------------------------

def _detect_avalon_stage(step: int, max_steps: int, info: dict) -> str:
    """Return Avalon game stage based on quest progress and phase."""
    phase = info.get("phase", -1)
    if phase == 3:
        return "assassination"
    quest = info.get("quest", info.get("current_quest", 0))
    if quest <= 1:
        return "early_quests"
    if quest <= 3:
        return "mid_quests"
    return "late_quests"


def _detect_diplomacy_stage(step: int, max_steps: int, info: dict) -> str:
    """Return Diplomacy game stage based on phase progression."""
    phase_name = info.get("phase_name", "")
    if phase_name:
        year_match = re.search(r"(\d{4})", phase_name)
        if year_match:
            year = int(year_match.group(1))
            if year <= 1902:
                return "opening"
            if year <= 1907:
                return "midgame"
            return "endgame"
    ratio = step / max(max_steps, 1)
    if ratio < 0.25:
        return "opening"
    if ratio < 0.65:
        return "midgame"
    return "endgame"


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
    """Build a rich spatial observation for games that provide grid data
    (e.g. sokoban).  Falls back to *summary_state* when grid is absent."""
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
        "  sokoban, box misaligned → [POSITION] push box toward its target square\n"
        "  sokoban, path blocked by wall → [NAVIGATE] find route around obstacle\n"
        "  avalon, suspicious player → [DEFEND] block suspected spy from mission\n"
        "  avalon, team forming → [ATTACK] push to lead the next mission\n"
        "  diplomacy, ally requesting support → [BUILD] strengthen alliance for next turn\n"
        "  diplomacy, unexplored border → [EXPLORE] scout neighbor's intentions\n"
        "  pokemon, item on ground → [COLLECT] pick up item before moving on\n"
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


_POKEMON_SYSTEM_PROMPT = """\
You are an expert AI player for Pokemon Red (Game Boy).
Your goal is to progress: explore, talk to NPCs, battle, earn badges.

# GAME STATES
- **Field**: Move, interact, use menu. You have a map with tile data.
- **Dialog**: Advance with continue_dialog tool, or press b to exit.
  If choices appear, use d-pad + a to select (raw buttons, NOT continue_dialog).
- **Battle**: Use battle tools (select_move_in_battle, run_away, etc.).

# AVAILABLE TOOLS
## Field State Tools
- move_to(x_dest, y_dest): A* pathfind to walkable tile ('O' or 'G').
  Usage: use_tool(move_to, (x_dest=X, y_dest=Y))
  CRITICAL: Dest MUST be walkable ('O','G'). NOT '?','X','WarpPoint','TalkTo', etc.
- warp_with_warp_point(x_dest, y_dest): Move to 'WarpPoint' tile and warp.
  Usage: use_tool(warp_with_warp_point, (x_dest=X, y_dest=Y))
  Use this for stairs, doors, cave entrances marked 'WarpPoint' on the map.
- overworld_map_transition(direction): Cross 'overworld'-type map boundary.
  Usage: use_tool(overworld_map_transition, (direction="north"))
  direction: 'north'|'south'|'west'|'east'. Only for overworld-type maps.
- interact_with_object(object_name): Pathfind to + face + interact with named object.
  Usage: use_tool(interact_with_object, (object_name="SPRITE_OAK_1"))
  Object must appear in [Notable Objects] list.

## Dialog State Tools
- continue_dialog(): Advance dialog until choices appear or dialog ends.
  Usage: use_tool(continue_dialog, ())
  ONLY use when in Dialog state with NO selection/choice box visible.

## Battle State Tools
- select_move_in_battle(): Use first available attack move.
  Usage: use_tool(select_move_in_battle, ())
- switch_pkmn_in_battle(): Switch active Pokemon.
  Usage: use_tool(switch_pkmn_in_battle, ())
- use_item_in_battle(): Use item.
  Usage: use_tool(use_item_in_battle, ())
- run_away(): Flee from wild battle.
  Usage: use_tool(run_away, ())

## Raw buttons (only when no tool applies)
a | b | start | select | up | down | left | right

# MAP READING (CRITICAL — read EVERY turn)
- [Full Map] shows tile grid: 'O'=walkable, 'G'=grass, 'X'=wall, '?'=unexplored.
  'WarpPoint' = door/stairs (use warp tool). Named objects = interact.
- ALWAYS read [Full Map] to find the ACTUAL coordinates of WarpPoints.
  DO NOT guess or reuse coordinates from a different floor/map.
- '?' tiles are unexplored. You MUST explore them by using move_to on a
  nearby 'O' or 'G' tile. This reveals hidden doors, exits, and items.
- If the map has many '?' tiles, PRIORITIZE exploring before warping.
  The exit from a building is often behind unexplored tiles.
- To leave a room: find a WarpPoint in [Notable Objects], use warp tool on it.
  If no exit WarpPoint is visible, explore '?' areas to reveal it.

# CRITICAL NAVIGATION RULES
- NEVER call warp_with_warp_point on coordinates that show '?' on the map.
- NEVER reuse coordinates from a previous map — each map has different layout.
- If you keep returning to the same room, you are using the wrong WarpPoint.
  Look for a DIFFERENT WarpPoint or explore '?' tiles to find the exit.
- To leave Red's House: go downstairs (warp), then walk to the exit door
  (it may be hidden in unexplored '?' tiles at the bottom of 1F).

# IMPORTANT RULES
- In Field: prefer tools (move_to, warp, interact) over raw buttons.
- In Dialog WITHOUT choices: use continue_dialog.
- In Dialog WITH choices: use raw buttons (d-pad to select, a to confirm).
- In Battle: use battle tools.

Output format (strict):
REASONING: <1-2 sentences about tool/action choice>
ACTION: <use_tool(...) or raw button>
"""

_POKEMON_ACTION_SECTION = """\
Output your action as a tool call or raw button.
Examples:
  ACTION: use_tool(move_to, (x_dest=5, y_dest=3))
  ACTION: use_tool(warp_with_warp_point, (x_dest=7, y_dest=7))
  ACTION: use_tool(overworld_map_transition, (direction="south"))
  ACTION: use_tool(continue_dialog, ())
  ACTION: use_tool(select_move_in_battle, ())
  ACTION: up"""


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


_POKEMON_TOOL_NAMES = frozenset({
    "move_to", "interact_with_object", "warp_with_warp_point",
    "continue_dialog", "select_move_in_battle",
    "switch_pkmn_in_battle", "run_away", "use_item_in_battle",
    "overworld_map_transition",
})


def _unwrap_use_tool(raw: str) -> str:
    """Convert use_tool(tool_name, (kwargs)) -> tool_name(args) for env.step()."""
    m = re.match(
        r'^use_tool\s*\(\s*(\w+)\s*,\s*\(([^)]*)\)\s*\)',
        raw.strip(), re.IGNORECASE,
    )
    if not m:
        return raw
    tool_name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return tool_name
    kvs = [kv.strip() for kv in args_str.split(",") if "=" in kv]
    vals = []
    for kv in kvs:
        _, v = kv.split("=", 1)
        vals.append(v.strip().strip('"').strip("'"))
    return f"{tool_name}({', '.join(vals)})"


def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    if not raw or not valid_actions:
        return None
    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}
    if raw_lower in lower_map:
        return lower_map[raw_lower]

    # use_tool(...) wrapper from SFT cold-start format
    if raw_lower.startswith("use_tool("):
        unwrapped = _unwrap_use_tool(raw)
        unwrapped_lower = unwrapped.lower().strip()
        if unwrapped_lower in lower_map:
            return lower_map[unwrapped_lower]
        tool_m2 = re.match(r'^(\w+)\s*\(', unwrapped_lower)
        if tool_m2 and tool_m2.group(1) in _POKEMON_TOOL_NAMES:
            return unwrapped.strip()
        if unwrapped_lower in _POKEMON_TOOL_NAMES:
            return unwrapped.strip()

    # "use tool: tool_name" or "use_tool: tool_name" (malformed variant)
    _use_tool_colon = re.match(r'^use[\s_]tool\s*:\s*(\w+)', raw_lower)
    if _use_tool_colon:
        _tname = _use_tool_colon.group(1)
        if _tname in _POKEMON_TOOL_NAMES:
            return _tname

    # Parametric tool call: move_to(5, 3), warp_with_warp_point(7, 7), etc.
    tool_m = re.match(r'^(\w+)\s*\(', raw_lower)
    if tool_m:
        tool_name = tool_m.group(1)
        if tool_name in _POKEMON_TOOL_NAMES:
            return raw.strip()

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


def _apply_anti_repetition(
    action: str, valid_actions: List[str],
    recent_actions: List[str], recent_rewards: List[float],
    game: str = "",
) -> str:
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in valid_actions if a != action]
        if alternatives:
            return random.choice(alternatives)

    if game == "pokemon_red" and len(recent_actions) >= 4:
        last6 = recent_actions[-6:]
        last6_rewards = recent_rewards[-6:]
        action_set = set(last6 + [action])
        if len(action_set) <= 2 and sum(last6_rewards) <= 0:
            explore_actions = ["move_to", "down", "left", "right", "up"]
            return random.choice(explore_actions)

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
        Explicit role/power to control (unified-role mode).  When
        *None* the legacy random-role selection is used.
    assigned_role_index : int | None
        Explicit player index (Avalon) or power ordinal (Diplomacy).
    opponent_model : str | None
        External API model for opponents (e.g. ``"gpt-5-mini"``).
        When set, non-controlled players use this model via API
        instead of vLLM self-play.
    opponent_api_base : str | None
        Base URL for the opponent API (default: OpenRouter).
    """
    imp = _lazy_imports()
    GAME_CONFIGS = imp["GAME_CONFIGS"]
    make_gaming_env = imp["make_gaming_env"]
    make_orak_env = imp["make_orak_env"]
    GamingAgentNLWrapper = imp["GamingAgentNLWrapper"]
    SokobanNLWrapper = imp["SokobanNLWrapper"]
    DiplomacyNLWrapper = imp["DiplomacyNLWrapper"]
    AvalonNLWrapper = imp["AvalonNLWrapper"]
    HARD_SUMMARY_CHAR_LIMIT = imp["HARD_SUMMARY_CHAR_LIMIT"]
    extract_game_facts = imp["extract_game_facts"]
    compact_text_observation = imp["compact_text_observation"]
    strip_think_tags = imp["strip_think_tags"]

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

    elif game == "diplomacy":
        if DiplomacyNLWrapper is None:
            raise ImportError("DiplomacyNLWrapper not available")
        _ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        if assigned_role is not None:
            power = assigned_role
        else:
            power = random.choice(_ALL_POWERS)
        _role_idx = _ALL_POWERS.index(power) if power in _ALL_POWERS else 0
        logger.info("Diplomacy: controlling %s this episode", power)
        env = _DiplomacyAdapter(DiplomacyNLWrapper(
            controlled_power=power, max_phases=20,
            vllm_base_urls=vllm_base_urls, model_name=model_name,
            skill_bank=skill_bank,
            opponent_model=opponent_model,
            opponent_api_base=opponent_api_base,
        ))

    elif game == "avalon":
        if AvalonNLWrapper is None:
            raise ImportError("AvalonNLWrapper not available")
        _AVALON_ROLE_NAMES = ["Merlin", "Servant", "Servant", "Minion", "Assassin"]
        _AVALON_SIDE_MAP = {
            "Merlin": "good", "Percival": "good", "Servant": "good",
            "Mordred": "evil", "Morgana": "evil", "Oberon": "evil",
            "Minion": "evil", "Assassin": "evil",
        }
        if assigned_role_index is not None:
            player = assigned_role_index
        else:
            player = random.randint(0, 4)
        _role_name = _AVALON_ROLE_NAMES[player] if player < len(_AVALON_ROLE_NAMES) else "Servant"
        _role_side = _AVALON_SIDE_MAP.get(_role_name, "good")
        logger.info("Avalon: controlling player %d (%s/%s) this episode", player, _role_name, _role_side)
        env = _AvalonAdapter(AvalonNLWrapper(
            num_players=5, controlled_player=player,
            vllm_base_urls=vllm_base_urls, model_name=model_name,
            skill_bank=skill_bank,
            opponent_model=opponent_model,
            opponent_api_base=opponent_api_base,
        ))

    else:
        if exe:
            base_env = await loop.run_in_executor(
                exe, make_gaming_env, game, max_steps,
            )
        else:
            base_env = make_gaming_env(game=game, max_steps=max_steps)

        if game == "tetris":
            try:
                from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
                env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))
            except ImportError:
                logger.warning("TetrisMacroActionWrapper unavailable, using primitive actions")
                env = GamingAgentNLWrapper(base_env)
        elif game == "sokoban" and SokobanNLWrapper is not None:
            env = SokobanNLWrapper(
                base_env, reflect_every=3,
                enable_restart=True, auto_detect_deadlock=True,
                distance_shaping=True,
            )
        else:
            env = GamingAgentNLWrapper(base_env)

    # ── Resolve role / side metadata for multi-role games ───────
    _ep_role = ""
    _ep_side = ""
    _ep_role_idx = -1
    if game == "diplomacy":
        _ep_role = power
        _ep_side = power          # each power is its own "side"
        _ep_role_idx = _role_idx
    elif game == "avalon":
        _ep_role = _role_name
        _ep_side = _role_side
        _ep_role_idx = player

    if exe:
        obs_nl, info = await loop.run_in_executor(exe, env.reset)
    else:
        obs_nl, info = env.reset()

    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")
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

    pokemon_explore: Optional[_PokemonExploreTracker] = None
    if game == "pokemon_red":
        pokemon_explore = _PokemonExploreTracker()
        pokemon_explore.update(info)

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

        # Intention generation — base model, no LoRA, higher temp
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

        need_reselect = skill_tracker.should_reselect(
            last_guidance, state_text=summary_state or obs_nl,
        )
        skill_select_prompt: Optional[str] = None
        skill_coro = None

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
            )

            if candidates and len(candidates) >= 2:
                candidates_text = _format_candidates_for_selection(candidates)
                user_content = (
                    f"Game state:\n{(summary_state or obs_nl)[:3000]}\n\n"
                    f"Current intention: {current_intention[:500]}\n\n"
                    f"Available strategies (pick ONE by number):\n{candidates_text}\n\n"
                    f"Choose the best strategy. Output REASONING then SKILL number."
                )
                skill_select_prompt = SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user_content
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

        # Process skill selection result
        if bank_available and (need_reselect or last_guidance is None):
            if sk_result is not None and candidates and len(candidates) >= 2:
                chosen_idx, skill_reasoning = _parse_skill_selection(
                    sk_result.text, len(candidates), candidates,
                )
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
                guidance = candidates[0]
                last_candidates = candidates
                last_chosen_idx = 0
                last_skill_reasoning = "only one candidate"
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
        if game == "pokemon_red":
            summary_for_action = obs_nl[:4000]
        elif _use_raw_obs:
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
        if game == "pokemon_red":
            action_user = (
                f"## Current Game State\n{summary_for_action}\n\n"
                f"Subgoal: {assigned_subgoal}\n"
                f"{urgency_line}{skill_context}{recent_context}\n"
                f"{_POKEMON_ACTION_SECTION}"
            )
            action_prompt = _POKEMON_SYSTEM_PROMPT + "\n" + action_user
        else:
            action_user = (
                f"Game state:\n\n{summary_for_action}\n\n"
                f"Subgoal: {assigned_subgoal}\n"
                f"{urgency_line}{skill_context}{recent_context}"
                f"Available actions (pick ONE by number):\n{_format_numbered_actions(step_actions)}\n\n"
                f"Output ACTION number."
            )
            action_prompt = SYSTEM_PROMPT + skill_text + "\n" + action_user

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
            elif game == "pokemon_red":
                _poke_action_str = str(action)
                _ptm = re.match(r'^(\w+)\s*\((.+)\)$', _poke_action_str)
                if _ptm and _ptm.group(1) in _POKEMON_TOOL_NAMES:
                    _poke_action_str = f"use_tool({_ptm.group(1)}, ({_ptm.group(2)}))"
                elif _poke_action_str in _POKEMON_TOOL_NAMES:
                    _poke_action_str = f"use_tool({_poke_action_str}, ())"
                action_completion = f"REASONING: {reasoning or 'Expert play.'}\nACTION: {_poke_action_str}"
                _explore_bonus = pokemon_explore.update(next_info) if pokemon_explore else 0.0
                _action_reward = float(reward) + _explore_bonus + skill_tracker._intrinsic_bonus
            else:
                action_completion = f"{subgoal_line}REASONING: {reasoning or 'Expert play.'}\nACTION: {action_num}"
                _action_reward = float(reward) + skill_tracker._intrinsic_bonus + 1.0
            grpo_records.append(GRPORecord(
                adapter="action_taking", game=game, episode_id=episode_id, step=step_count,
                prompt=action_prompt, completion=action_completion, reward=_action_reward,
                metadata={
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
                },
            ))

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
                from skill_agents_grpo.grpo.rewards import skill_selection_reward
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
            grpo_records.append(GRPORecord(
                adapter="skill_selection", game=game, episode_id=episode_id, step=step_count,
                prompt=skill_select_prompt, completion=sk_completion, reward=sk_reward,
                metadata={
                    "chosen_idx": last_chosen_idx,
                    "skill_candidates": [c.get("skill_id") for c in last_candidates],
                    "chosen_skill_id": (
                        last_candidates[last_chosen_idx].get("skill_id")
                        if last_chosen_idx < len(last_candidates) else None
                    ),
                    "summary_state": summary_state,
                    "intention": current_intention,
                    "reselect_reason": skill_tracker._reselect_reason,
                },
            ))

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
            if game == "avalon":
                _exp_dict["stage"] = _detect_avalon_stage(
                    step_count, max_steps, next_info,
                )
            elif game == "diplomacy":
                _exp_dict["stage"] = _detect_diplomacy_stage(
                    step_count, max_steps, next_info,
                )
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
        # Skip for games with sparse rewards (reward only at game end).
        # Sokoban now uses distance-based reward shaping, so it produces
        # positive rewards on productive pushes and can use stuck detection.
        _STUCK_EXEMPT_GAMES = {"avalon", "diplomacy", "pokemon_red"}
        if (game not in _STUCK_EXEMPT_GAMES
                and step_count >= min_steps_before_stuck
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
