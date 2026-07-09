#!/usr/bin/env python
"""
Pokemon Red cold-start rollout generation using Orak's env + toolset.

Uses Orak's PokemonRedEnv (PyBoyRunner) directly -- not via MCP -- so
high-level tools (continue_dialog, select_move_in_battle, move_to, …) run
inline.  One tool call can execute dozens of button presses, making LLM
usage vastly more efficient than raw-button-per-call.

Natural termination conditions:
  1. Whiteout:  all party Pokemon fainted (HP == 0)
  2. No-progress: same location for N steps, no dialog/badge change
  3. Score completion: Orak evaluate() returns done
  4. Max steps: configurable hard cap (default 500)

Output structure (cold_start/output/gpt54/pokemon_red/), same as other envs:
  - episode_NNN.json        Individual episode (Episode.to_dict() + metadata)
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line, rollout_metadata)
  - rollout_summary.json    Per-game run stats (target_episodes, max_steps, labeled, etc.)

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$(pwd)/../Orak/src:$PYTHONPATH"

    python cold_start/generate_cold_start_pokemon_red.py --episodes 3 --verbose
    # Add --label to label trajectories (labeling is in labeling/ folder)
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent                        # Game-AI-Agent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"
ORAK_SRC = CODEBASE_ROOT.parent / "Orak" / "src"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT), str(ORAK_SRC)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Orak imports
# ---------------------------------------------------------------------------
from mcp_game_servers.pokemon_red.game.pokemon_red_env import PokemonRedEnv, PokemonRedObs
from mcp_game_servers.pokemon_red.game.pyboy_runner import PyBoyRunner
from mcp_game_servers.pokemon_red.game.utils.pokemon_tools import (
    PokemonToolset,
    process_state_tool,
    execute_action_response,
)
from mcp_game_servers.pokemon_red.game.utils.map_utils import (
    construct_init_map,
    refine_current_map,
    replace_map_on_screen_with_full_map,
    replace_filtered_screen_text,
)

# Cold-start data structures
from data_structure.experience import Experience, Episode, Episode_Buffer

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
GAME_NAME = "pokemon_red"
MAX_MEMORY_STEPS = 10
NO_PROGRESS_THRESHOLD = 80


# ===================================================================
# Lightweight agent wrapper (matches what PokemonToolset expects)
# ===================================================================

@dataclass
class AgentMemory:
    state_dict: Dict[str, Any] = field(default_factory=dict)
    map_memory_dict: Dict[str, Any] = field(default_factory=dict)
    dialog_buffer: List[str] = field(default_factory=list)


class AgentShell:
    """Minimal stand-in for the full Orak agent object.

    The non-MCP PokemonToolset (pokemon_tools.py) accesses:
      - self.agent.env          -> PokemonRedEnv
      - self.agent.memory.*     -> AgentMemory
    """

    def __init__(self, env: PokemonRedEnv):
        self.env = env
        self.memory = AgentMemory()


# ===================================================================
# Orak env helper: create and configure from rom_path
# ===================================================================

def _suppress_map_warnings():
    """Monkey-patch PyBoyRunner's map/asm loaders to suppress the expected
    '[WARN] Map module not found' and '[WARN] asm not found' prints that
    fire on every state read when processed_map data isn't available."""
    import mcp_game_servers.pokemon_red.game.pyboy_runner as _pbr
    import io

    _orig_load_map = _pbr.load_map_module
    def _quiet_load_map(map_name):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return _orig_load_map(map_name)
        finally:
            sys.stdout = old_stdout
    _pbr.load_map_module = _quiet_load_map

    _orig_parse = _pbr.parse_object_sprites
    def _quiet_parse(asm_path):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return _orig_parse(asm_path)
        finally:
            sys.stdout = old_stdout
    _pbr.parse_object_sprites = _quiet_parse


def _isolate_rom(rom_path: str) -> str:
    """Copy ROM to a temp dir so each PyBoy instance gets its own .ram/.sav files.

    Returns the path to the ROM copy.  The temp dir is registered for
    cleanup at process exit.
    """
    tmp_dir = tempfile.mkdtemp(prefix="pokemon_red_")
    rom_copy = os.path.join(tmp_dir, os.path.basename(rom_path))
    shutil.copy2(rom_path, rom_copy)
    atexit.register(shutil.rmtree, tmp_dir, True)
    return rom_copy


def make_orak_env(rom_path: str, log_path: str, task: str = "DefeatBrock") -> PokemonRedEnv:
    """Instantiate an Orak PokemonRedEnv without the MCP/omegaconf stack.

    Orak's PyBoyRunner uses relative paths (e.g. ./src/mcp_game_servers/...)
    so we chdir to the Orak root during setup, then restore the original cwd.

    The ROM is copied to a temp dir so parallel episodes each get their own
    .ram/.sav files and don't conflict.
    """
    import types

    _suppress_map_warnings()

    orak_root = str(ORAK_SRC.parent)
    old_cwd = os.getcwd()
    try:
        os.chdir(orak_root)

        rom_copy = _isolate_rom(os.path.abspath(rom_path))

        cfg = types.SimpleNamespace(
            log_path=log_path,
            task=task,
            rom_path=rom_copy,
            success_condition="get_boulder_badge",
            input_modality="text",
        )
        env = PokemonRedEnv.__new__(PokemonRedEnv)
        env.cfg = cfg
        env.configure()
        return env
    finally:
        os.chdir(old_cwd)


# ===================================================================
# Intro skip (cold-boot: title → NEW GAME → Oak → naming → play)
# ===================================================================

def skip_intro(runner: PyBoyRunner, max_presses: int = 350, verbose: bool = False, fast: bool = False):
    """Press buttons until we are truly past the intro.

    A "true" Field state means: get_battle_state() == "Field" AND we
    have a non-trivial map (map dimensions > 0 or party has a Pokemon).
    The simple "Field" check alone triggers on brief transition frames
    between the title screen and Oak's speech.
    """

    last_text = ""
    same = 0
    field_confirm = 0
    FIELD_CONFIRM_NEEDED = 3  # require Field N times in a row to be sure

    for i in range(max_presses):
        state = runner.get_battle_state()

        if state == "Field":
            mem = runner.pyboy.memory
            map_h = mem[0xD368]
            party_count = mem[0xD163]
            # Real field: map has dimensions OR player has Pokemon
            if map_h > 0 or party_count > 0:
                field_confirm += 1
                if field_confirm >= FIELD_CONFIRM_NEEDED:
                    if verbose:
                        print(f"  [intro] Field confirmed after {i} presses "
                              f"(map_h={map_h}, party={party_count})")
                    return
                # Wait a beat and re-check rather than pressing more buttons
                time.sleep(0.02 if fast else 0.2)
                continue
            else:
                field_confirm = 0
        else:
            field_confirm = 0

        dialog = runner.get_dialog()
        text_portion = dialog.split("[Selection Box Text]")[0] if dialog else ""

        if text_portion == last_text:
            same += 1
        else:
            same = 0
            last_text = text_portion

        if same >= 5:
            runner.send_input("down")
            time.sleep(0.02 if fast else 0.15)
            runner.send_input("a")
            time.sleep(0.02 if fast else 0.15)
            same = 0
        else:
            runner.send_input("a")
            time.sleep(0.02 if fast else 0.15)

    if verbose:
        print(f"  [intro] WARNING: hit max presses ({max_presses})")


# ===================================================================
# Prompts (adapted from Orak's text prompts, trimmed for cold-start)
# ===================================================================

SYSTEM_PROMPT = """\
You are an expert AI player for Pokemon Red (Game Boy).
Your goal is to progress: explore, talk to NPCs, battle, earn badges.

# GAME STATES
- **Field**: Move, interact, use menu. You have a map with tile data.
- **Dialog**: Advance with `continue_dialog` tool, or press b to exit.
  If choices (▶) appear, use d-pad + a to select (raw buttons, NOT continue_dialog).
- **Battle**: Use battle tools (select_move_in_battle, run_away, etc.).
- **Title**: Press a repeatedly.

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
  Object must appear in [Notable Objects] list. Handles its own dialog.

## Dialog State Tools
- continue_dialog(): Press a repeatedly until dialog ends or choices appear.
  Usage: use_tool(continue_dialog, ())
  ONLY use when in Dialog state with NO selection/choice box (▶) visible.

## Battle State Tools
- select_move_in_battle(move_name): Select attack move (UPPERCASE).
  Usage: use_tool(select_move_in_battle, (move_name="MOVE"))
- switch_pkmn_in_battle(pokemon_name): Switch active Pokemon.
  Usage: use_tool(switch_pkmn_in_battle, (pokemon_name="NAME"))
- use_item_in_battle(item_name, pokemon_name=None): Use item.
  Usage: use_tool(use_item_in_battle, (item_name="ITEM"))
- run_away(): Flee wild battle (not trainer).
  Usage: use_tool(run_away, ())

## Raw buttons (max 5 per turn, only when no tool applies)
a | b | start | select | up | down | left | right
Use raw buttons for: menu navigation, dialog choices (▶), title screen.

# MAP READING
- [Full Map] shows tile grid. 'O'=walkable, 'G'=grass, 'X'=wall, '?'=unexplored.
  'W'=WarpPoint (use warp tool), 'S'=SIGN/SPRITE (use interact tool).
- [Notable Objects] lists named objects with coordinates.
- Your position is shown in [Map Info].
- To explore: move_to a walkable tile near '?' areas to reveal them.
- To leave a room: use warp_with_warp_point on a WarpPoint tile.

# IMPORTANT RULES
- In Field: prefer tools (move_to, warp, interact) over raw buttons.
- In Dialog WITHOUT choices: use continue_dialog.
- In Dialog WITH choices (▶): use raw buttons (d-pad to select, a to confirm).
- In Battle: use battle tools.
- Cursor move and confirm must be SEPARATE turns for choices (e.g., 'down' then next turn 'a').

# RESPONSE FORMAT
### State_summary
<1-2 lines: situation, intent>

### Action_reasoning
<Brief reasoning about tool/action choice>

### Actions
use_tool(<tool_name>, (<args>))
OR
<button1> | <button2> | ...
"""

USER_PROMPT_TEMPLATE = """\
## Current Game State
{state_text}

## Recent History ({n_history} steps)
{history}

## Reflection
{reflection}

Choose the best tool or raw buttons for the current situation.\
"""

REFLECTION_PROMPT = """\
Analyze the last {n} steps of this Pokemon Red playthrough:

{trajectory}

Current state summary:
{state_summary}

Briefly reflect (under 80 words):
1. Progress made (new area, dialog, battle)?
2. Any stuck/oscillation patterns?
3. Immediate next goal?"""


# ===================================================================
# Memory and reflection
# ===================================================================

class RollingMemory:
    def __init__(self, max_steps: int = MAX_MEMORY_STEPS):
        self.max_steps = max_steps
        self.history: List[Dict[str, str]] = []
        self.last_reflection: str = "Game just started. Explore and talk to NPCs."

    def add(self, step: int, action: str, state_summary: str):
        self.history.append({"step": step, "action": action, "summary": state_summary})
        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]

    def format_history(self) -> str:
        if not self.history:
            return "(no previous actions)"
        return "\n".join(
            f"  Step {h['step']}: {h['action']} → {h['summary'][:80]}"
            for h in self.history
        )

    def format_for_reflection(self) -> str:
        if not self.history:
            return "(no actions yet)"
        return "\n".join(
            f"Step {h['step']}: action={h['action']}\n  State: {h['summary'][:120]}"
            for h in self.history
        )


# ===================================================================
# Termination checks
# ===================================================================

class ProgressTracker:
    def __init__(self, threshold: int = NO_PROGRESS_THRESHOLD):
        self.threshold = threshold
        self.last_location: Optional[str] = None
        self.last_badge_count: int = 0
        self.steps_at_location: int = 0
        self.had_dialog: bool = False

    def update(self, state_dict: Dict) -> bool:
        location = state_dict.get("map_info", {}).get("map_name")
        badge_text = state_dict.get("badge_list", "")
        badge_count = badge_text.count(",") + 1 if badge_text and badge_text != "N/A" else 0

        dialog_text = state_dict.get("filtered_screen_text", "N/A")
        has_dialog = dialog_text and dialog_text != "N/A"
        if has_dialog:
            self.had_dialog = True

        if badge_count > self.last_badge_count:
            self.last_badge_count = badge_count
            self.steps_at_location = 0
            self.had_dialog = False
            return False

        if location and location != self.last_location:
            self.last_location = location
            self.steps_at_location = 0
            self.had_dialog = False
            return False

        self.steps_at_location += 1
        return self.steps_at_location >= self.threshold and not self.had_dialog


def check_whiteout(state_dict: Dict) -> bool:
    party_text = state_dict.get("your_party", "")
    if not party_text or party_text == "N/A":
        return False
    hp_matches = re.findall(r"HP:\s*(\d+)/(\d+)", party_text)
    if not hp_matches:
        return False
    return all(int(cur) == 0 for cur, _ in hp_matches)


# ===================================================================
# LLM interaction
# ===================================================================

VALID_ACTIONS = {"a", "b", "start", "select", "up", "down", "left", "right"}


def _make_client(client_kw: Dict) -> "openai.OpenAI":
    """Create or return a cached OpenAI client (avoids re-creating HTTP sessions)."""
    key = tuple(sorted(client_kw.items()))
    if not hasattr(_make_client, "_cache"):
        _make_client._cache = {}
    if key not in _make_client._cache:
        _make_client._cache[key] = openai.OpenAI(**client_kw)
    return _make_client._cache[key]


def get_reflection(memory: RollingMemory, state_summary: str,
                   client_kw: Dict, model: str) -> str:
    if len(memory.history) < 4:
        return memory.last_reflection
    prompt = REFLECTION_PROMPT.format(
        n=len(memory.history),
        trajectory=memory.format_for_reflection(),
        state_summary=state_summary,
    )
    try:
        client = _make_client(client_kw)
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=150,
            messages=[
                {"role": "system", "content": "You are a Pokemon Red game analyst. Brief, actionable reflections."},
                {"role": "user", "content": prompt},
            ],
        )
        memory.last_reflection = resp.choices[0].message.content.strip()
    except Exception:
        pass
    return memory.last_reflection


def parse_llm_actions(text: str) -> str:
    """Extract the ### Actions block from LLM response."""
    m = re.search(r"###\s*Actions?\s*\n(.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        action_block = m.group(1).strip().split("\n")[0].strip()
        return action_block
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line.startswith("use_tool(") or any(a in line.lower().split() for a in VALID_ACTIONS):
            return line
    return "a"


def llm_decide(state_text: str, memory: RollingMemory,
               client_kw: Dict, model: str, temperature: float,
               do_reflect: bool) -> Tuple[str, str]:
    """Query LLM. Returns (action_string, full_response)."""
    if openai is None:
        return "a", ""

    state_summary_short = state_text[:300]
    reflection = memory.last_reflection
    if do_reflect and len(memory.history) >= 4:
        reflection = get_reflection(memory, state_summary_short, client_kw, model)

    user_content = USER_PROMPT_TEMPLATE.format(
        state_text=state_text[:3000],
        n_history=len(memory.history),
        history=memory.format_history(),
        reflection=reflection,
    )

    try:
        client = _make_client(client_kw)
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        full = resp.choices[0].message.content.strip()
        return parse_llm_actions(full), full
    except Exception as exc:
        print(f"    [WARN] LLM call failed ({exc}), fallback to 'a'")
        return "a", ""


# ===================================================================
# Action execution (mirrors Orak's action_tool_execution.py)
# ===================================================================

_NAV_TOOLS = {"move_to", "warp_with_warp_point", "interact_with_object",
              "overworld_map_transition"}


def _has_map_data(toolset: PokemonToolset) -> bool:
    """Check whether the current map has valid explored_map data."""
    try:
        map_name = toolset.agent.memory.state_dict["map_info"]["map_name"]
        explored = toolset.agent.memory.map_memory_dict.get(map_name, {}).get("explored_map")
        return explored is not None and len(explored) > 0 and len(explored[0]) > 0
    except (KeyError, TypeError):
        return False


def execute_action(action_str: str, toolset: PokemonToolset,
                   env: PokemonRedEnv, fast: bool = False) -> str:
    """Execute a parsed action string (tool or raw buttons).

    Returns a feedback string for recording.
    """
    action_str = action_str.split("\n")[0].strip()
    parts = [p.strip() for p in re.split(r'\s*\|\s*', action_str) if p.strip()]
    parts = parts[:5]

    results = []
    for part in parts:
        if part.startswith("use_tool("):
            # Guard: nav tools need valid map data
            tool_match = re.match(r"use_tool\((\w+)", part)
            tool_name = tool_match.group(1) if tool_match else ""
            if tool_name in _NAV_TOOLS and not _has_map_data(toolset):
                results.append(f"{part} -> (False, 'No map data available — use raw d-pad buttons instead')")
                continue
            result = execute_action_response(toolset, part)
            results.append(f"{part} -> {result}")
        elif part.lower() in VALID_ACTIONS:
            env.send_action_set([part.lower()])
            time.sleep(0.05 if fast else 0.3)
            results.append(part.lower())
        elif part.lower() == "quit":
            results.append("quit")
        else:
            results.append(f"(ignored: {part})")

    return " | ".join(results)


# ===================================================================
# Episode runner
# ===================================================================

def run_pokemon_episode(
    rom_path: str,
    model: str = MODEL_GPT54,
    max_steps: int = 200,
    temperature: float = 0.4,
    reflect_every: int = 5,
    verbose: bool = False,
    log_path: str = "/tmp/pokemon_cold_start",
    fast: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Pokemon Red episode using Orak env + toolset."""

    os.makedirs(log_path, exist_ok=True)

    # --- LLM client setup ---
    use_router = open_router_api_key and open_router_api_key.strip()
    client_kw: Dict[str, Any] = {}
    effective_model = model
    if use_router:
        client_kw = {"base_url": OPENROUTER_BASE, "api_key": open_router_api_key.strip()}
        effective_model = model if "/" in model else f"openai/{model}"
    elif openai_api_key:
        client_kw = {"api_key": openai_api_key}

    # --- Init Orak env ---
    env = make_orak_env(rom_path, log_path)
    agent = AgentShell(env)
    toolset = PokemonToolset(agent)

    # --- Skip intro ---
    if verbose:
        print("  [env] Skipping intro…")
    skip_intro(env.runner, verbose=verbose, fast=fast)

    # --- Init state ---
    state_text = env._receive_state()
    agent.memory.state_dict = env.parse_game_state(state_text)
    agent.memory.map_memory_dict = toolset.get_map_memory_dict(
        agent.memory.state_dict, agent.memory.map_memory_dict
    )

    memory = RollingMemory()
    progress = ProgressTracker()
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    termination_reason = "max_steps"

    task_desc = "Play Pokemon Red: explore, battle, earn badges."
    action_names = list(VALID_ACTIONS)

    # Sync env internal state for evaluate() to work properly
    state_text = env._receive_state()
    env.state_dict = env.parse_game_state(state_text)
    env.prev_state_dict = dict(env.state_dict)
    score_str = "0.0 (0/12)"
    prev_score_val = 0.0

    while step_count < max_steps:
        # --- Read state ---
        state_text = env._receive_state()
        state_dict = env.parse_game_state(state_text)
        agent.memory.state_dict = state_dict

        # Sync env internal state so evaluate() can compare prev vs current
        env.prev_state_dict = dict(env.state_dict)
        env.state_dict = state_dict
        env.state_text = state_text

        agent.memory.map_memory_dict = toolset.get_map_memory_dict(
            state_dict, agent.memory.map_memory_dict
        )

        # Process state with map memory for richer observation
        processed_text, state_dict, agent.memory.map_memory_dict, _, agent.memory.dialog_buffer = \
            process_state_tool(
                env, toolset, agent.memory.map_memory_dict,
                step_count, agent.memory.dialog_buffer, state_text
            )
        agent.memory.state_dict = state_dict

        # --- Natural termination: whiteout ---
        if check_whiteout(state_dict):
            if verbose:
                print(f"    [TERM] Whiteout at step {step_count}")
            termination_reason = "whiteout"
            break

        # --- Natural termination: no-progress ---
        if progress.update(state_dict):
            if verbose:
                print(f"    [TERM] No progress for {NO_PROGRESS_THRESHOLD} steps")
            termination_reason = "no_progress"
            break

        # --- Score check ---
        try:
            score_str, score_done = env.evaluate(PokemonRedObs(state_text=state_text))
        except (KeyError, TypeError):
            score_done = False
        if score_done:
            if verbose:
                print(f"    [TERM] Score completion: {score_str}")
            termination_reason = "score_complete"
            break

        # --- LLM decision ---
        do_reflect = step_count > 0 and step_count % reflect_every == 0
        action_str, llm_response = llm_decide(
            processed_text, memory, client_kw, effective_model,
            temperature, do_reflect,
        )

        # --- Execute action ---
        feedback = execute_action(action_str, toolset, env, fast=fast)
        step_count += 1

        # --- Post-step state ---
        next_state_text = env._receive_state()
        next_state_dict = env.parse_game_state(next_state_text)
        agent.memory.state_dict = next_state_dict

        # Sync env internal state again
        env.prev_state_dict = dict(env.state_dict)
        env.state_dict = next_state_dict
        env.state_text = next_state_text

        try:
            score_str, score_done = env.evaluate(PokemonRedObs(state_text=next_state_text))
        except (KeyError, TypeError):
            score_done = False
        reward = 0.0
        try:
            score_val = float(score_str.split("(")[0].strip())
            reward = (score_val - prev_score_val) / 100.0
            prev_score_val = score_val
        except Exception:
            pass
        total_reward += reward

        summary = (f"State={next_state_dict.get('state','?')} "
                   f"Map={next_state_dict.get('map_info',{}).get('map_name','?')} "
                   f"Score={score_str}")
        memory.add(step_count, action_str, summary)

        exp = Experience(
            state=processed_text,
            action=action_str,
            reward=float(reward),
            next_state=next_state_text,
            done=score_done,
            intentions=llm_response,
            tasks=task_desc,
        )
        exp.idx = step_count - 1
        exp.action_type = "tool" if "use_tool" in action_str else "primitive"
        exp.available_actions = action_names
        exp.interface = {"env_name": "orak", "game_name": GAME_NAME}
        exp.raw_state = str(state_text) if state_text else None
        exp.raw_next_state = str(next_state_text) if next_state_text else None
        experiences.append(exp)

        if verbose:
            loc = next_state_dict.get("map_info", {}).get("map_name", "?")
            act_short = action_str[:50]
            # Show full feedback so tool results (True/False, messages) aren't cut off
            fb = feedback if len(feedback) <= 200 else feedback[:197] + "..."
            print(f"  step {step_count}: {act_short} @ {loc}  score={score_str}  "
                  f"feedback={fb}")

        if score_done:
            termination_reason = "score_complete"
            break

    # --- Cleanup ---
    try:
        env.runner.running = False
        time.sleep(0.1 if fast else 0.5)
    except Exception:
        pass

    episode = Episode(
        experiences=experiences,
        task=task_desc,
        env_name="orak",
        game_name=GAME_NAME,
    )
    episode.set_outcome()

    terminated = termination_reason in ("score_complete", "whiteout")
    truncated = termination_reason in ("max_steps", "no_progress")
    stats = {
        "game": GAME_NAME,
        "steps": step_count,
        "total_reward": total_reward,
        "termination_reason": termination_reason,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "gpt54_orak_toolset",
        "final_location": state_dict.get("map_info", {}).get("map_name"),
        "final_score": score_str,
    }
    return episode, stats


# ===================================================================
# Batch rollout helpers
# ===================================================================

def count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_all_episodes(args, output_dir: Path) -> Dict:
    game_dir = output_dir / GAME_NAME
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {start_idx}/{args.episodes} episodes already done")
            return {"game": GAME_NAME, "skipped": True}
        if start_idx:
            print(f"  [RESUME] from episode {start_idx}")

    from cold_start.generate_cold_start import get_cold_start_max_steps, label_trajectory
    effective_max_steps = args.max_steps or get_cold_start_max_steps(GAME_NAME)

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [pokemon_red] Episode {ep_idx + 1}/{args.episodes}")
        ep_log = str(output_dir / "logs" / f"ep_{ep_idx:03d}")

        try:
            episode, stats = run_pokemon_episode(
                rom_path=args.rom_path,
                model=args.model,
                max_steps=effective_max_steps,
                temperature=args.temperature,
                reflect_every=args.reflect_every,
                verbose=args.verbose,
                log_path=ep_log,
                fast=args.fast,
            )

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}, "
                  f"End: {stats['termination_reason']}, "
                  f"Location: {stats.get('final_location', '?')}, "
                  f"Score: {stats.get('final_score', '?')}")

            if args.label and not args.no_label:
                episode = label_trajectory(episode, args.label_model)

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({"game": GAME_NAME, "episode_index": ep_idx,
                              "error": str(e), "steps": 0, "total_reward": 0.0})

    elapsed = time.time() - t0
    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary: Dict = {
        "game": GAME_NAME,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "gpt54_orak_toolset",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": effective_max_steps,
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    good = [s for s in all_stats if "error" not in s]
    if good:
        rewards = [s.get("total_reward", 0) for s in good]
        steps = [s.get("steps", 0) for s in good]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
        reasons = {}
        for s in good:
            r = s.get("termination_reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        summary["termination_reasons"] = reasons

    with open(game_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pokemon Red cold-start using Orak env + toolset",
    )
    parser.add_argument("--rom_path", type=str, default=None,
                        help="Path to Pokemon Red .gb/.gbc ROM")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--model", type=str, default=MODEL_GPT54)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--reflect_every", type=int, default=5)
    parser.add_argument("--label", action="store_true",
                        help="Label trajectories with LLM (default: off; use labeling/ for that)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip labeling (default: no labeling)")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Max speed: no emulator throttle, shorter sleeps, faster startup")

    args = parser.parse_args()

    if args.fast:
        os.environ["PYBOY_FRAME_TIME"] = "0"
        os.environ["POKEMON_STARTUP_DELAY"] = "1"

    # Resolve ROM path
    if not args.rom_path:
        candidates = [
            CODEBASE_ROOT.parent / "GamingAgent" / "gamingagent" / "configs" / "custom_06_pokemon_red" / "rom" / "pokemon.gb",
            CODEBASE_ROOT.parent / "ROMs" / "Pokemon - Red Version (USA, Europe).gb",
        ]
        for c in candidates:
            if c.exists():
                args.rom_path = str(c)
                break
        if not args.rom_path:
            print("[ERROR] ROM not found. Provide --rom_path or place ROM at:")
            for c in candidates:
                print(f"  {c}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54"
    output_dir.mkdir(parents=True, exist_ok=True)

    has_key = bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or (open_router_api_key and open_router_api_key.strip())
    )
    if not has_key:
        print("[WARNING] No API key. LLM calls will fail.")

    from cold_start.generate_cold_start import get_cold_start_max_steps
    effective_max = args.max_steps or get_cold_start_max_steps(GAME_NAME)

    print("=" * 72)
    print("  Pokemon Red Cold-Start (Orak Env + Toolset)")
    print("=" * 72)
    print(f"  ROM:             {args.rom_path}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Max steps:       {effective_max}")
    print(f"  Model:           {args.model}")
    print(f"  Temp:            {args.temperature}")
    print(f"  Reflect every:   {args.reflect_every}")
    print(f"  No-progress cap: {NO_PROGRESS_THRESHOLD}")
    print(f"  Fast mode:       {args.fast}")
    print(f"  Labeling:        {args.label and not args.no_label}")
    print(f"  Output:          {output_dir}")
    print("=" * 72)
    print()
    print("  High-level tools available:")
    print("    continue_dialog, select_move_in_battle, switch_pkmn_in_battle,")
    print("    run_away, use_item_in_battle, move_to, warp_with_warp_point,")
    print("    overworld_map_transition, interact_with_object")
    print()
    print("  Natural termination:")
    print("    1. Whiteout (all party HP=0)")
    print(f"    2. No progress ({NO_PROGRESS_THRESHOLD} steps same location)")
    print("    3. Score completion (Orak evaluate)")
    print(f"    4. Max steps ({effective_max})")
    print("=" * 72)

    summary = run_all_episodes(args, output_dir)

    print(f"\n{'=' * 72}")
    print("  COMPLETE")
    print(f"{'=' * 72}")
    if not summary.get("skipped"):
        print(f"  Episodes:  {summary.get('total_episodes', 0)}")
        print(f"  Elapsed:   {summary.get('elapsed_seconds', 0):.1f}s")
        if "mean_reward" in summary:
            print(f"  Avg reward: {summary['mean_reward']:.2f}")
            print(f"  Avg steps:  {summary['mean_steps']:.1f}")
        if "termination_reasons" in summary:
            print(f"  End reasons: {summary['termination_reasons']}")
    print(f"  Output:    {output_dir / GAME_NAME}  (same layout as gpt54/<game>/)")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
