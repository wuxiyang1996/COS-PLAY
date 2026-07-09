"""
Orak game environment NL wrapper.

Wraps Orak game environments (BaseEnv subclasses from krafton-ai/Orak) so that:
- Observations are natural language strings (via obs2text).
- step() accepts string actions (parsed via text2action).

Orak benchmark covers 12 games across 6 genres:
  Action:      street_fighter, super_mario
  Adventure:   pwaat (Ace Attorney), her_story
  RPG:         pokemon_red, darkest_dungeon
  Simulation:  minecraft, stardew_valley
  Strategy:    star_craft, star_craft_multi, slay_the_spire
  Puzzle:      baba_is_you, twenty_fourty_eight

Usage:
    from evaluate_orak.orak_nl_wrapper import OrakNLWrapper, make_orak_env

    env = make_orak_env("star_craft")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(...)
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


_SCRIPT_DIR = Path(__file__).resolve().parent
_CODEBASE_ROOT = _SCRIPT_DIR.parent

# Primary: cloned Orak repo
_ORAK_REPO = _CODEBASE_ROOT.parent / "Orak"
_ORAK_SRC = _ORAK_REPO / "src"
_ORAK_MCP_GAMES = _ORAK_SRC / "mcp_game_servers"
_ORAK_MCP_AGENTS = _ORAK_SRC / "mcp_agent_client"

for _p in [str(_ORAK_SRC), str(_ORAK_MCP_GAMES)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


@contextlib.contextmanager
def _orak_cwd():
    """Temporarily chdir to the Orak repo root for games that use relative paths."""
    prev = os.getcwd()
    try:
        os.chdir(str(_ORAK_REPO))
        yield
    finally:
        os.chdir(prev)


def _cfg_path(game: str) -> str:
    """Resolve config.yaml path from the Orak repo."""
    return str(_ORAK_MCP_AGENTS / "configs" / game / "config.yaml")


_pokemon_warnings_suppressed = False


def _suppress_pokemon_map_warnings() -> None:
    """Monkey-patch PyBoyRunner to silence '[WARN] asm not found' and
    '[WARN] Map module not found' prints that fire on every state read."""
    global _pokemon_warnings_suppressed
    if _pokemon_warnings_suppressed:
        return
    _pokemon_warnings_suppressed = True
    try:
        import io as _io
        import mcp_game_servers.pokemon_red.game.pyboy_runner as _pbr

        _orig_load_map = _pbr.load_map_module

        def _quiet_load_map(map_name: str):
            old_stdout = sys.stdout
            sys.stdout = _io.StringIO()
            try:
                return _orig_load_map(map_name)
            finally:
                sys.stdout = old_stdout

        _pbr.load_map_module = _quiet_load_map

        _orig_parse = _pbr.parse_object_sprites

        def _quiet_parse(asm_path: str):
            old_stdout = sys.stdout
            sys.stdout = _io.StringIO()
            try:
                return _orig_parse(asm_path)
            finally:
                sys.stdout = old_stdout

        _pbr.parse_object_sprites = _quiet_parse
    except Exception:
        pass


_POKEMON_TOOL_ACTIONS = frozenset({
    "move_to", "interact_with_object", "warp_with_warp_point",
    "continue_dialog", "select_move_in_battle",
    "switch_pkmn_in_battle", "run_away", "use_item_in_battle",
    "overworld_map_transition",
})


# ── Pokemon Red intro skip ────────────────────────────────────────────

def _skip_pokemon_red_intro(env: Any, max_presses: int = 350) -> None:
    """Press through title screen, NEW GAME, Oak intro, and naming.

    Brings the game from Title screen to Field state (Red's bedroom).
    Adapted from cold_start/generate_cold_start_pokemon_red.py.
    """
    runner = env.runner
    last_text = ""
    same = 0
    field_confirm = 0
    FIELD_CONFIRM_NEEDED = 3
    _pause = float(os.environ.get("PYBOY_FRAME_TIME", "0.01")) * 15
    _pause = max(_pause, 0.015)

    for i in range(max_presses):
        state = runner.get_battle_state()

        if state == "Field":
            mem = runner.pyboy.memory
            map_h = mem[0xD368]
            party_count = mem[0xD163]
            if map_h > 0 or party_count > 0:
                field_confirm += 1
                if field_confirm >= FIELD_CONFIRM_NEEDED:
                    logger.info(
                        "Pokemon Red intro skipped after %d presses "
                        "(map_h=%d, party=%d)",
                        i, map_h, party_count,
                    )
                    return
                _time.sleep(_pause)
                continue
            else:
                field_confirm = 0
        else:
            field_confirm = 0

        dialog = runner.get_dialog()
        text_portion = (
            dialog.split("[Selection Box Text]")[0] if dialog else ""
        )

        if text_portion == last_text:
            same += 1
        else:
            same = 0
            last_text = text_portion

        if same >= 5:
            runner.send_input("down")
            _time.sleep(_pause)
            runner.send_input("a")
            _time.sleep(_pause)
            same = 0
        else:
            runner.send_input("a")
            _time.sleep(_pause)

    logger.warning("Pokemon Red intro skip: hit max presses (%d)", max_presses)


# ── Pokemon Red PokemonToolset integration ─────────────────────────────

class _PokemonAgentMemory:
    """Minimal memory object matching what PokemonToolset expects."""
    __slots__ = ("state_dict", "map_memory_dict", "dialog_buffer")

    def __init__(self) -> None:
        self.state_dict: Dict[str, Any] = {}
        self.map_memory_dict: Dict[str, Any] = {}
        self.dialog_buffer: List[str] = []


class _PokemonAgentShell:
    """Minimal agent wrapper matching what PokemonToolset expects.

    PokemonToolset accesses ``self.agent.env`` and ``self.agent.memory``.
    """
    __slots__ = ("env", "memory")

    def __init__(self, env: Any) -> None:
        self.env = env
        self.memory = _PokemonAgentMemory()


ORAK_GAMES: Dict[str, Dict[str, Any]] = {
    # ── Puzzle ──────────────────────────────────────────────────────────
    "twenty_fourty_eight": {
        "config_yaml": _cfg_path("twenty_fourty_eight"),
        "action_names": ["up", "down", "left", "right"],
        "task": "Merge tiles to reach 2048. Score = min(score/20000*100, 100).",
        "genre": "puzzle",
    },
    "baba_is_you": {
        "config_yaml": _cfg_path("baba_is_you"),
        "action_names": ["idle", "left", "right", "up", "down"],
        "task": "Solve the Baba Is You puzzle by manipulating rules. 100=win, 40=WIN exists, 20=WALL broken, 0=fail.",
        "genre": "puzzle",
    },
    # ── Action ──────────────────────────────────────────────────────────
    "super_mario": {
        "config_yaml": _cfg_path("super_mario"),
        "action_names": [f"Jump Level: {i}" for i in range(7)],
        "task": "Advance Mario as far right as possible. Score = x_pos / 3161 * 100.",
        "genre": "action",
    },
    "street_fighter": {
        "config_yaml": _cfg_path("street_fighter"),
        "action_names": [
            "Move Closer", "Move Away", "Fireball", "Megapunch", "Hurricane",
            "Low Kick", "Medium Kick", "High Kick", "Jump Closer", "Jump Away",
            "Crouch", "Block", "Low Punch", "Medium Punch", "High Punch",
        ],
        "task": "Defeat the opponent in Street Fighter III. Score = stages cleared.",
        "genre": "action",
    },
    # ── Strategy ────────────────────────────────────────────────────────
    "star_craft": {
        "config_yaml": _cfg_path("star_craft"),
        "action_names": [
            "TRAIN PROBE", "TRAIN ZEALOT", "TRAIN STALKER", "BUILD PYLON",
            "BUILD GATEWAY", "BUILD ASSIMILATOR", "BUILD NEXUS",
            "BUILD CYBERNETICSCORE", "RESEARCH WARPGATERESEARCH",
            "RESEARCH CHARGE", "SCOUTING PROBE", "MULTI-ATTACK",
            "MULTI-RETREAT", "CHRONOBOOST NEXUS", "EMPTY ACTION",
        ],
        "task": "Win 1v1 as Protoss vs Zerg bot. Provide 5 sequential macro actions per step.",
        "genre": "strategy",
    },
    "star_craft_multi": {
        "config_yaml": _cfg_path("star_craft_multi"),
        "action_names": [],
        "task": "Win 1v1 StarCraft II (multi-player mode). Provide 5 actions per step.",
        "genre": "strategy",
    },
    "slay_the_spire": {
        "config_yaml": _cfg_path("slay_the_spire"),
        "action_names": ["PLAY", "END", "CHOOSE", "SKIP"],
        "task": "Climb the Spire, defeat enemies with card combos. Score = floor reached (max 50).",
        "genre": "strategy",
    },
    # ── RPG ─────────────────────────────────────────────────────────────
    "pokemon_red": {
        "config_yaml": _cfg_path("pokemon_red"),
        "action_names": [
            "up", "down", "left", "right", "a", "b", "start", "select",
            "move_to", "interact_with_object", "warp_with_warp_point",
            "continue_dialog", "select_move_in_battle",
            "switch_pkmn_in_battle", "run_away", "use_item_in_battle",
        ],
        "task": "Progress through Pokemon Red storyline milestones (0-12 flags).",
        "genre": "rpg",
    },
    "darkest_dungeon": {
        "config_yaml": _cfg_path("darkest_dungeon"),
        "action_names": ["attack", "heal", "swap", "idle", "skip"],
        "task": "Survive dungeon raids. Score = 0.4*combat + 0.3*survival + 0.3*(1-stress).",
        "genre": "rpg",
    },
    # ── Adventure ───────────────────────────────────────────────────────
    "pwaat": {
        "config_yaml": _cfg_path("pwaat"),
        "action_names": [
            "Ok", "Back", "Down", "Up", "Left", "Right",
            "Present evidence", "Press",
        ],
        "task": "Solve cases in Ace Attorney. Score = milestone rewards.",
        "genre": "adventure",
    },
    "her_story": {
        "config_yaml": _cfg_path("her_story"),
        "action_names": ["Search", "Play Video"],
        "task": "Uncover the story by searching keywords and watching videos. Score = videos viewed / 272.",
        "genre": "adventure",
    },
    # ── Simulation ──────────────────────────────────────────────────────
    "minecraft": {
        "config_yaml": _cfg_path("minecraft"),
        "action_names": [],
        "task": "Craft target items in Minecraft. Actions are JavaScript async functions.",
        "genre": "simulation",
    },
    "stardew_valley": {
        "config_yaml": _cfg_path("stardew_valley"),
        "action_names": [
            "till_soil", "plant_seeds", "water_seeds", "harvest_crops",
            "sell_item", "buy_item", "get_out_of_house", "go_house_and_sleep",
        ],
        "task": "Complete farming tasks in Stardew Valley (cleanup, cultivation, shopping, earn money).",
        "genre": "simulation",
    },
}


def _parse_mario_state(
    obs_text: str, score_val: float, step: int, max_steps: int,
) -> Dict[str, Any]:
    """Extract structured state from Mario obs_text for XML rendering."""
    import re as _re

    ss: Dict[str, Any] = {"game": "super_mario"}

    m = _re.search(r"Mario:\s*\((\d+),\s*(\d+)\)", obs_text)
    if m:
        ss["mario_pos"] = (int(m.group(1)), int(m.group(2)))

    def _parse_positions(label: str) -> list:
        pat = rf"{label}:\s*(.+?)(?:\n|$)"
        match = _re.search(pat, obs_text, _re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.lower() == "none":
            return []
        return _re.findall(r"\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", raw)

    for key, label in [
        ("bricks", "Bricks"),
        ("question_blocks", "Question Blocks"),
        ("goombas", "Monster Goomba"),
        ("koopas", "Monster Koopas"),
        ("pipes", "Warp Pipe"),
        ("mushrooms", "Item Mushrooms"),
        ("stairs", "Stair Blocks"),
        ("flag", "Flag"),
    ]:
        positions = _parse_positions(label)
        if positions:
            ss[key] = [(int(p[0]), int(p[1])) for p in positions]

    pit_m = _re.search(r"Pit:\s*start at (\d+|None),\s*end at (\d+|None)", obs_text)
    if pit_m and pit_m.group(1) != "None":
        ss["pit"] = (int(pit_m.group(1)), int(pit_m.group(2)))

    progress = min(1.0, score_val / 100.0) if score_val else 0.0
    ss["progress"] = progress
    ss["score"] = score_val
    ss["step"] = step
    ss["max_steps"] = max_steps

    if progress < 0.3:
        ss["phase"] = "early"
    elif progress < 0.7:
        ss["phase"] = "midgame"
    else:
        ss["phase"] = "endgame"

    return ss


def _obs_to_text(obs_obj: Any, env: Any) -> str:
    """Convert an Orak Obs dataclass to text via env.obs2text."""
    text = env.obs2text(obs_obj)
    if text is None:
        if hasattr(obs_obj, "to_text"):
            text = obs_obj.to_text()
        else:
            text = str(obs_obj)
    return text or ""


class OrakNLWrapper:
    """
    Wraps an Orak BaseEnv so observations are NL strings and step()
    accepts string actions. Presents the same interface as other
    Game-AI-Agent NL wrappers.
    """

    def __init__(
        self,
        env: Any,
        game_name: str,
        include_action_hint: bool = True,
        max_steps: int = 1000,
    ):
        self._env = env
        self._game_name = game_name
        self._include_action_hint = include_action_hint
        self._max_steps = max_steps
        self._step_count = 0
        self._last_reward: Optional[float] = None
        self._prev_score_val: float = 0.0

        game_info = ORAK_GAMES.get(game_name, {})
        self._action_names: List[str] = game_info.get("action_names", [])
        self._task: str = game_info.get("task", "")

        self._pokemon_toolset: Any = None
        self._pokemon_agent: Optional[_PokemonAgentShell] = None

    @property
    def env(self):
        return self._env

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    def _pokemon_state_info(self) -> Dict[str, Any]:
        """Extract structured position info from Pokemon Red state_dict."""
        try:
            sd = getattr(self._env, "state_dict", None) or {}
            mi = sd.get("map_info", {})
            return {
                "pokemon_map_name": mi.get("map_name", ""),
                "pokemon_player_x": mi.get("player_pos_x"),
                "pokemon_player_y": mi.get("player_pos_y"),
                "pokemon_game_state": sd.get("state", ""),
            }
        except Exception:
            return {}

    def _format_obs(self, obs_text: str) -> str:
        nl = obs_text
        if self._include_action_hint and self._action_names:
            if self._game_name in ("star_craft", "star_craft_multi"):
                nl += "\n\nProvide exactly 5 actions in the format:\n1: <ACTION>\n2: <ACTION>\n...\n5: <ACTION>"
                nl += f"\n\nValid actions: {', '.join(a for a in self._action_names if a != 'EMPTY ACTION')}, EMPTY ACTION"
            elif self._game_name == "slay_the_spire":
                nl += "\n\nChoose: PLAY <card_idx> [target_idx], END, CHOOSE <idx>, or SKIP."
            elif self._game_name == "minecraft":
                nl += "\n\nWrite a JavaScript async function with bot parameter."
            elif self._game_name == "stardew_valley":
                nl += f"\n\nReturn a Python list of skill calls. Available: {', '.join(self._action_names)}"
            else:
                nl += f"\n\nValid actions: {', '.join(self._action_names[:20])}. Choose one."
        return nl

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._step_count = 0
        self._last_reward = None
        self._prev_score_val = 0.0

        with _orak_cwd():
            # Orak BaseEnv uses initial_obs() rather than Gymnasium's reset()
            if hasattr(self._env, "initial_obs"):
                obs_obj = self._env.initial_obs()
            else:
                kw = {}
                if seed is not None:
                    kw["seed"] = seed
                if options is not None:
                    kw["options"] = options
                obs_obj = self._env.reset(**kw) if kw else self._env.reset()
                if isinstance(obs_obj, tuple):
                    obs_obj = obs_obj[0]

            obs_text = _obs_to_text(obs_obj, self._env)

        if self._game_name == "pokemon_red":
            obs_text = self._pokemon_enhance_obs(obs_text)

        # SC2: the bot's first on_step() hasn't run yet, so the initial
        # observation is empty.  Do one warmup step with EMPTY ACTIONs to
        # get the real first game state from the running SC2 process.
        if not obs_text.strip() and self._game_name in ("star_craft", "star_craft_multi"):
            n_actions = getattr(self._env, "num_actions", 5)
            empty_actions = "\n".join(f"{i}: EMPTY ACTION" for i in range(1, n_actions + 1))
            warmup_nl, _, _, _, warmup_info = self.step(empty_actions)
            self._step_count = 0
            return warmup_nl, warmup_info

        nl = self._format_obs(obs_text)

        game_info = {}
        if hasattr(self._env, "get_game_info"):
            game_info = self._env.get_game_info() or {}

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "task": self._task,
            **game_info,
        }
        if self._game_name == "super_mario":
            info["structured_state"] = _parse_mario_state(
                obs_text, 0.0, 0, self._max_steps,
            )
        if self._game_name == "pokemon_red":
            info.update(self._pokemon_state_info())
        return nl, info

    def step(
        self,
        action: Union[str, int],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        action_str = str(action).strip()

        if self._game_name == "pokemon_red":
            _use_tool_m = re.match(
                r'^use_tool\s*\(\s*(\w+)\s*,\s*\(([^)]*)\)\s*\)',
                action_str, re.IGNORECASE,
            )
            if _use_tool_m:
                _tool_name = _use_tool_m.group(1)
                _args = _use_tool_m.group(2).strip()
                if _args:
                    action_str = f"{_tool_name}({_args})"
                else:
                    action_str = _tool_name
            _lower = action_str.lower()
            if _lower in _POKEMON_TOOL_ACTIONS:
                return self._pokemon_tool_step(action_str)
            _tool_m = re.match(r'^(\w+)\s*\(', _lower)
            if _tool_m and _tool_m.group(1) in _POKEMON_TOOL_ACTIONS:
                return self._pokemon_tool_step(action_str)

        with _orak_cwd():
            action_obj = self._env.text2action(action_str)

            result = self._env.step(action_obj)
            obs_obj, reward_raw, terminated, truncated, step_info = (
                result[0], result[1], result[2], result[3], result[4]
            )

            score, done = self._env.evaluate(obs_obj)
            obs_text = _obs_to_text(obs_obj, self._env)

        if self._game_name == "pokemon_red":
            obs_text = self._pokemon_enhance_obs(obs_text)

        # Convert evaluate() score → numeric reward (delta-based).
        #
        # evaluate() return formats vary by game:
        #   SC2:         ("Victory"|"Defeat"|"Tie"|None, done)
        #   Pokemon Red: ("8.3 (1/12)", done)  — numeric prefix with milestone info
        #   Mario:       (float, done)
        #   Others:      (float|str|None, done)
        #
        # Pokemon Red scores are on a 0-100+ scale; divide by 100 to match
        _SC2_RESULT_MAP = {"Victory": 100.0, "Defeat": 0.0, "Tie": 50.0}
        score_val = 0.0
        if isinstance(score, str):
            if score in _SC2_RESULT_MAP:
                score_val = _SC2_RESULT_MAP[score]
            else:
                try:
                    score_val = float(score.split("(")[0].strip())
                except (ValueError, AttributeError):
                    score_val = 0.0
        elif score is not None:
            try:
                score_val = float(score)
            except (ValueError, TypeError):
                score_val = 0.0

        reward = score_val - self._prev_score_val
        self._prev_score_val = score_val

        self._step_count += 1
        self._last_reward = reward

        if self._step_count >= self._max_steps and not (terminated or truncated):
            truncated = True

        nl = self._format_obs(obs_text)

        game_info = {}
        if hasattr(self._env, "get_game_info"):
            game_info = self._env.get_game_info() or {}

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "step": self._step_count,
            "score": score,
            "task": self._task,
            **game_info,
        }

        if self._game_name == "super_mario":
            info["structured_state"] = _parse_mario_state(
                obs_text, score_val, self._step_count, self._max_steps,
            )

        if self._game_name == "pokemon_red":
            info.update(self._pokemon_state_info())
        return nl, reward, bool(terminated or done), bool(truncated), info

    # ── Pokemon Red tool action support ──────────────────────────────

    def _init_pokemon_toolset(self) -> None:
        """Lazy-init PokemonToolset for handling high-level actions."""
        from mcp_game_servers.pokemon_red.game.utils.pokemon_tools import (
            PokemonToolset,
        )
        self._pokemon_agent = _PokemonAgentShell(self._env)
        self._pokemon_toolset = PokemonToolset(self._pokemon_agent)

    def _sync_pokemon_state(self) -> None:
        """Sync PokemonToolset agent memory with env state."""
        agent = self._pokemon_agent
        env = self._env
        if not env.state_dict:
            state_text = env._receive_state()
            env.state_text = state_text
            env.state_dict = env.parse_game_state(state_text)
        agent.memory.state_dict = env.state_dict.copy()
        map_name = agent.memory.state_dict.get("map_info", {}).get("map_name")
        if map_name and agent.memory.state_dict.get("map_info", {}).get("x_max") is not None:
            try:
                agent.memory.map_memory_dict = (
                    self._pokemon_toolset.get_map_memory_dict(
                        agent.memory.state_dict,
                        agent.memory.map_memory_dict,
                    )
                )
            except Exception:
                pass

    def _pokemon_finalize_step(self) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """After a tool action, read new state, evaluate, return step tuple."""
        env = self._env

        with _orak_cwd():
            state_text = env._receive_state()
            env.state_text = state_text
            env.state_dict = env.parse_game_state(state_text)

            from mcp_game_servers.pokemon_red.game.pokemon_red_env import (
                PokemonRedObs,
            )
            obs = PokemonRedObs(state_text=state_text, image=None)
            obs_text = env.obs2text(obs)
            score, done = env.evaluate(obs)

        obs_text = self._pokemon_enhance_obs(obs_text)

        score_val = 0.0
        if isinstance(score, str):
            try:
                score_val = float(score.split("(")[0].strip())
            except (ValueError, AttributeError):
                score_val = 0.0
        elif score is not None:
            try:
                score_val = float(score)
            except (ValueError, TypeError):
                score_val = 0.0

        reward = score_val - self._prev_score_val
        self._prev_score_val = score_val

        self._step_count += 1
        self._last_reward = reward

        truncated = False
        if self._step_count >= self._max_steps and not done:
            truncated = True

        nl = self._format_obs(obs_text)
        game_info = {}
        if hasattr(env, "get_game_info"):
            game_info = env.get_game_info() or {}

        info: Dict[str, Any] = {
            "state_natural_language": nl,
            "action_names": self._action_names,
            "env_name": "orak",
            "game_name": self._game_name,
            "step": self._step_count,
            "score": score,
            "task": self._task,
            **game_info,
        }
        info.update(self._pokemon_state_info())
        return nl, reward, bool(done), bool(truncated), info

    @staticmethod
    def _parse_tool_call(action_str: str) -> Tuple[str, Dict[str, Any]]:
        """Parse 'tool_name(arg1, arg2)' into (tool_name, {kwargs})."""
        m = re.match(r'^(\w+)\s*\(([^)]*)\)', action_str)
        if not m:
            return action_str.lower().strip(), {}
        tool = m.group(1).lower().strip()
        args_str = m.group(2).strip()
        if not args_str:
            return tool, {}
        kwargs: Dict[str, Any] = {}
        if "=" in args_str:
            for part in args_str.split(","):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    try:
                        kwargs[k] = int(v)
                    except ValueError:
                        kwargs[k] = v
        else:
            vals = [v.strip().strip("'\"") for v in args_str.split(",")]
            if tool in ("move_to", "warp_with_warp_point") and len(vals) >= 2:
                try:
                    kwargs = {"x_dest": int(vals[0]), "y_dest": int(vals[1])}
                except ValueError:
                    pass
            elif tool == "interact_with_object" and vals:
                kwargs = {"object_name": vals[0]}
        return tool, kwargs

    def _pokemon_tool_step(
        self, action_str: str,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Execute a Pokemon Red tool action and return a step tuple."""
        if self._pokemon_toolset is None:
            self._init_pokemon_toolset()

        env = self._env
        env.prev_state_text = env.state_text
        env.prev_state_dict = (
            env.state_dict.copy() if isinstance(env.state_dict, dict)
            else env.state_dict
        )
        self._sync_pokemon_state()

        tool, kwargs = self._parse_tool_call(action_str)
        toolset = self._pokemon_toolset

        with _orak_cwd():
            try:
                if tool == "continue_dialog":
                    toolset.continue_dialog()
                elif tool == "run_away":
                    toolset.run_away()
                elif tool == "select_move_in_battle":
                    self._pokemon_auto_attack()
                elif tool == "switch_pkmn_in_battle":
                    self._pokemon_auto_switch()
                elif tool == "move_to":
                    if "x_dest" in kwargs and "y_dest" in kwargs:
                        toolset.move_to(**kwargs)
                    else:
                        self._pokemon_auto_move()
                elif tool == "interact_with_object":
                    if "object_name" in kwargs:
                        toolset.interact_with_object(**kwargs)
                    else:
                        self._pokemon_auto_interact()
                elif tool == "warp_with_warp_point":
                    if "x_dest" in kwargs and "y_dest" in kwargs:
                        result = toolset.warp_with_warp_point(**kwargs)
                        if isinstance(result, tuple) and not result[0]:
                            self._pokemon_explore_or_warp()
                    else:
                        self._pokemon_explore_or_warp()
                elif tool == "use_item_in_battle":
                    self._pokemon_auto_use_item()
                elif tool == "overworld_map_transition":
                    direction = kwargs.get("direction", "south")
                    toolset.overworld_map_transition(direction=direction)
            except Exception as exc:
                logger.debug("Pokemon tool %s(%s) failed: %s", tool, kwargs, exc)

        return self._pokemon_finalize_step()

    # -- Observation enhancement for Pokemon Red -------------------------

    def _pokemon_enhance_obs(self, obs_text: str) -> str:
        """Replace 9x9 viewport with full explored map + notable objects."""
        try:
            if self._pokemon_toolset is None:
                self._init_pokemon_toolset()
            self._sync_pokemon_state()
            agent = self._pokemon_agent
            map_name = agent.memory.state_dict.get("map_info", {}).get("map_name", "")
            mm = agent.memory.map_memory_dict.get(map_name, {})
            explored_map = mm.get("explored_map", [])
            if explored_map:
                from mcp_game_servers.pokemon_red.game.utils.map_utils import (
                    replace_map_on_screen_with_full_map,
                )
                with _orak_cwd():
                    obs_text = replace_map_on_screen_with_full_map(obs_text, explored_map)
        except Exception as exc:
            logger.debug("Pokemon map enhancement failed: %s", exc)
        return obs_text

    # -- Simplified tool implementations for parameterized actions ------

    def _pokemon_auto_attack(self) -> None:
        """FIGHT -> first available move."""
        env = self._env
        state = self._pokemon_agent.memory.state_dict
        if "Battle" not in state.get("state", ""):
            env.send_action_set(["a"])
            _time.sleep(0.5)
            return
        # Navigate to FIGHT (up-left) and press A
        env.send_action_set(["up", "left", "a"])
        _time.sleep(0.3)
        # Select first move (top-left) and press A
        env.send_action_set(["up", "left", "a"])
        _time.sleep(1.0)
        self._pokemon_mash_a_until_actionable()

    def _pokemon_auto_switch(self) -> None:
        """PKMN -> first pokemon -> SWITCH."""
        env = self._env
        state = self._pokemon_agent.memory.state_dict
        if "Battle" not in state.get("state", ""):
            env.send_action_set(["a"])
            _time.sleep(0.5)
            return
        env.send_action_set(["up", "right", "a"])
        _time.sleep(0.3)
        env.send_action_set(["a"])
        _time.sleep(0.3)
        env.send_action_set(["a"])
        _time.sleep(1.0)
        self._pokemon_mash_a_until_actionable()

    def _pokemon_auto_use_item(self) -> None:
        """ITEM -> first item -> USE."""
        env = self._env
        state = self._pokemon_agent.memory.state_dict
        if "Battle" not in state.get("state", ""):
            env.send_action_set(["a"])
            _time.sleep(0.5)
            return
        env.send_action_set(["down", "left", "a"])
        _time.sleep(0.3)
        env.send_action_set(["a"])
        _time.sleep(0.3)
        env.send_action_set(["a"])
        _time.sleep(1.0)
        self._pokemon_mash_a_until_actionable()

    def _pokemon_auto_move(self) -> None:
        """Without coordinates, try to move toward a warp point or explore."""
        env = self._env
        agent = self._pokemon_agent
        state = agent.memory.state_dict
        if state.get("state") != "Field":
            env.send_action_set(["a"])
            _time.sleep(0.5)
            return

        map_name = state.get("map_info", {}).get("map_name", "")
        mm = agent.memory.map_memory_dict.get(map_name, {})
        explored_map = mm.get("explored_map", [])

        if explored_map:
            px = state["map_info"].get("player_pos_x", 0)
            py = state["map_info"].get("player_pos_y", 0)
            try:
                success, path = self._pokemon_toolset._find_path_inner(
                    *self._pokemon_pick_target(explored_map, px, py)
                )
                if success and path:
                    cmds = re.split(r"[|/;, \t\n]+", path)[:8]
                    for c in cmds:
                        env.send_action_set([c])
                        _time.sleep(0.3)
                    return
            except Exception:
                pass

        direction = random.choice(["up", "down", "left", "right"])
        for _ in range(3):
            env.send_action_set([direction])
            _time.sleep(0.3)

    def _pokemon_pick_target(
        self, explored_map: list, px: int, py: int,
    ) -> Tuple[int, int]:
        """Pick a useful destination: nearest WarpPoint, or unexplored tile."""
        warp_points = []
        walkable = []
        for y, row in enumerate(explored_map):
            for x, cell in enumerate(row):
                if cell == "WarpPoint":
                    warp_points.append((x, y))
                elif cell in ("O", "G") and (x, y) != (px, py):
                    walkable.append((x, y))

        best = None
        best_dist = float("inf")
        for x, y in warp_points:
            d = abs(x - px) + abs(y - py)
            if 0 < d < best_dist:
                best = (x, y)
                best_dist = d
        if best:
            return best

        if walkable:
            random.shuffle(walkable)
            return walkable[0]
        return (px, py)

    def _pokemon_auto_interact(self) -> None:
        """Press A to interact with whatever is in front, continue dialog."""
        env = self._env
        state = self._pokemon_agent.memory.state_dict
        if state.get("state") != "Field":
            env.send_action_set(["a"])
            _time.sleep(0.5)
            return
        env.send_action_set(["a"])
        _time.sleep(0.5)
        text_obs = env._receive_state()
        sd = env.parse_game_state(text_obs)
        self._pokemon_agent.memory.state_dict = sd
        if sd.get("state") == "Dialog":
            try:
                self._pokemon_toolset.continue_dialog()
            except Exception:
                pass

    def _pokemon_explore_or_warp(self) -> None:
        """Try to explore toward '?' tiles first; fall back to farthest WarpPoint.

        This prevents the common pathology of bouncing between floors:
        if there are unexplored tiles, walk toward them (which may reveal
        the exit door).  Only warp as last resort, and prefer the warp
        point farthest from the player (most likely to be the exit).
        """
        env = self._env
        agent = self._pokemon_agent
        state = agent.memory.state_dict
        if state.get("state") != "Field":
            env.send_action_set(["a"])
            _time.sleep(0.5)
            return

        map_name = state.get("map_info", {}).get("map_name", "")
        mm = agent.memory.map_memory_dict.get(map_name, {})
        explored_map = mm.get("explored_map", [])
        if not explored_map:
            direction = random.choice(["up", "down", "left", "right"])
            for _ in range(5):
                env.send_action_set([direction])
                _time.sleep(0.3)
            return

        px = state["map_info"].get("player_pos_x", 0)
        py = state["map_info"].get("player_pos_y", 0)

        walkable = {"O", "G"}
        best_target = None
        best_dist = -1
        for y, row in enumerate(explored_map):
            for x, cell in enumerate(row):
                if cell not in walkable:
                    continue
                has_unknown_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < len(explored_map) and 0 <= nx < len(row):
                        if explored_map[ny][nx] == "?":
                            has_unknown_neighbor = True
                            break
                if has_unknown_neighbor:
                    dist = abs(x - px) + abs(y - py)
                    if dist > best_dist:
                        best_dist = dist
                        best_target = (x, y)

        if best_target:
            try:
                self._pokemon_toolset.move_to(
                    x_dest=best_target[0], y_dest=best_target[1],
                )
                return
            except Exception:
                pass

        warp_points = []
        for y, row in enumerate(explored_map):
            for x, cell in enumerate(row):
                if cell == "WarpPoint":
                    dist = abs(x - px) + abs(y - py)
                    warp_points.append((x, y, dist))
        warp_points.sort(key=lambda t: -t[2])
        for wx, wy, _ in warp_points:
            try:
                self._pokemon_toolset.warp_with_warp_point(wx, wy)
                return
            except Exception:
                continue

        direction = random.choice(["up", "down", "left", "right"])
        for _ in range(5):
            env.send_action_set([direction])
            _time.sleep(0.3)

    def _pokemon_auto_warp(self) -> None:
        """Find nearest WarpPoint and try to warp there via toolset."""
        self._pokemon_explore_or_warp()

    def _pokemon_mash_a_until_actionable(self, max_presses: int = 15) -> None:
        """Press A until state returns to Field or a selection box appears."""
        env = self._env
        for _ in range(max_presses):
            env.send_action_set(["a"])
            _time.sleep(0.5)
            text_obs = env._receive_state()
            sd = env.parse_game_state(text_obs)
            self._pokemon_agent.memory.state_dict = sd
            if sd.get("state") == "Field":
                break
            sel = sd.get("selection_box_text", "N/A")
            if sel != "N/A" and "FIGHT" in sel:
                break

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)


def make_orak_env(
    game_name: str,
    max_steps: int = 1000,
    config_override: Optional[str] = None,
) -> OrakNLWrapper:
    """Create a wrapped Orak game environment.

    Args:
        game_name: One of the keys in ORAK_GAMES (e.g. "star_craft", "baba_is_you").
        max_steps: Max steps before truncation.
        config_override: Optional path to a custom config YAML.

    Returns:
        OrakNLWrapper with standard reset()/step() Gymnasium interface.
    """
    if game_name not in ORAK_GAMES:
        raise ValueError(f"Unknown Orak game '{game_name}'. Available: {sorted(ORAK_GAMES.keys())}")

    config_path = config_override or ORAK_GAMES[game_name]["config_yaml"]

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Ensure the Orak repo is cloned at {_ORAK_REPO}"
        )

    import omegaconf
    from mcp_game_servers.utils.module_creator import EnvCreator

    cfg = omegaconf.OmegaConf.load(config_path)

    log_dir = str(_CODEBASE_ROOT / "orak_logs" / game_name)
    os.makedirs(log_dir, exist_ok=True)

    _NO_SCREENSHOT_GAMES = {"pokemon_red", "super_mario"}

    with omegaconf.open_dict(cfg):
        if "env" in cfg:
            cfg.env.log_path = log_dir
            if hasattr(cfg.env, "show_graphic"):
                cfg.env.show_graphic = False
            if game_name in _NO_SCREENSHOT_GAMES and hasattr(cfg.env, "save_screenshots"):
                cfg.env.save_screenshots = False
            # Resolve rom_path relative to Orak repo root so it works
            # regardless of the caller's cwd.
            if hasattr(cfg.env, "rom_path") and not os.path.isabs(cfg.env.rom_path):
                abs_rom = os.path.normpath(os.path.join(str(_ORAK_REPO), cfg.env.rom_path))
                cfg.env.rom_path = abs_rom
            # Isolate ROM to a temp dir so parallel PyBoy instances don't
            # conflict on .ram/.sav files (each derived from the ROM path).
            if hasattr(cfg.env, "rom_path") and game_name == "pokemon_red":
                _tmp = tempfile.mkdtemp(prefix="pokemon_red_orak_")
                _rom_copy = os.path.join(_tmp, os.path.basename(cfg.env.rom_path))
                _src_rom = cfg.env.rom_path
                if os.path.islink(_src_rom):
                    _src_rom = os.path.realpath(_src_rom)
                shutil.copy2(_src_rom, _rom_copy)
                atexit.register(shutil.rmtree, _tmp, True)
                cfg.env.rom_path = _rom_copy
        cfg.log_path = log_dir

    with _orak_cwd():
        env = EnvCreator(cfg).create()

    # Pokemon Red: suppress noisy PyBoyRunner warnings about missing
    # pokered disassembly .asm files, then skip the title/Oak intro.
    if game_name == "pokemon_red":
        _suppress_pokemon_map_warnings()
        logger.info("Pokemon Red: skipping intro (title → Field)…")
        with _orak_cwd():
            _skip_pokemon_red_intro(env)
            state_text = env._receive_state()
            env.state_text = state_text
            env.state_dict = env.parse_game_state(state_text)
        logger.info(
            "Pokemon Red: intro done, state=%s map=%s",
            env.state_dict.get("state"),
            env.state_dict.get("map_info", {}).get("map_name"),
        )

    wrapper = OrakNLWrapper(env, game_name=game_name, max_steps=max_steps)

    # Use the real action space from the env when available (e.g. SC2 has 72
    # Protoss actions vs the 15-action subset in ORAK_GAMES).
    if hasattr(env, "action_dict") and env.action_dict:
        wrapper._action_names = sorted(
            env.action_dict.keys(), key=lambda a: env.action_dict[a]
        )

    return wrapper
