#!/usr/bin/env python
"""
Cold-start data generation for Game-AI-Agent.

Generates unlabeled trajectories using the prompt decision agent (VLMDecisionAgent)
and/or the dummy language agent, then labels them with GPT-5-mini to produce
initial seed data for the skill database.

Usage (from Game-AI-Agent root):

    # Prefer OpenRouter (used by default): set open_router_api_key in api_keys.py or:
    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # Generate cold-start data for 2048
    python cold_start/generate_cold_start.py \
        --game twenty_forty_eight \
        --episodes 3 --max_steps 50 --model gpt-5-mini

    # Generate for sokoban with VLM decision agent
    python cold_start/generate_cold_start.py \
        --game sokoban \
        --agent_type vlm --episodes 2 --max_steps 100

    # Generate for all supported games
    python cold_start/generate_cold_start.py --all_games --episodes 2 --max_steps 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
# Imports from Game-AI-Agent
# ---------------------------------------------------------------------------
from API_func import ask_model
from data_structure.experience import Experience, Episode, Episode_Buffer
from decision_agents.agent import (
    VLMDecisionAgent,
    run_episode_vlm_agent,
    run_tool,
    TOOL_TAKE_ACTION,
    TOOL_REWARD,
)
from decision_agents.dummy_agent import (
    language_agent_action,
    detect_game,
    _default_action,
    GAME_GAMINGAGENT,
)
from decision_agents.reward_func import RewardConfig, RewardResult

# ---------------------------------------------------------------------------
# GamingAgent environment imports: LAZY to avoid loading retro/pyglet (X11) when
# only running custom games (2048, Sokoban, Candy Crush, Tetris) on headless servers.
# ---------------------------------------------------------------------------
import importlib

_ENV_IMPORT_MAP = {
    "twenty_forty_eight": ("gamingagent.envs.custom_01_2048.twentyFortyEightEnv", "TwentyFortyEightEnv"),
    "sokoban": ("gamingagent.envs.custom_02_sokoban.sokobanEnv", "SokobanEnv"),
    "candy_crush": ("gamingagent.envs.custom_03_candy_crush.candyCrushEnv", "CandyCrushEnv"),
    "tetris": ("gamingagent.envs.custom_04_tetris.tetrisEnv", "TetrisEnv"),
    "doom": ("gamingagent.envs.custom_05_doom.doomEnv", "DoomEnvWrapper"),
    "pokemon_red": ("gamingagent.envs.custom_06_pokemon_red.pokemonRedEnv", "PokemonRedEnv"),
    "super_mario_bros": ("gamingagent.envs.retro_01_super_mario_bros.superMarioBrosEnv", "SuperMarioBrosEnv"),
    "ace_attorney": ("gamingagent.envs.retro_02_ace_attorney.aceAttorneyEnv", "AceAttorneyEnv"),
    "nineteen_forty_two": ("gamingagent.envs.retro_03_1942.NineteenFortyTwo_env", "NineteenFortyTwoEnv"),
    "tic_tac_toe": ("gamingagent.envs.zoo_01_tictactoe.TicTacToeEnv", "SingleTicTacToeEnv"),
    "texas_holdem": ("gamingagent.envs.zoo_02_texasholdem.TexasHoldemEnv", "SingleTexasHoldemEnv"),
}


def _get_env_class(game_name: str):
    """Lazy-load the env class for a game (avoids importing retro/pyglet until needed)."""
    entry = _ENV_IMPORT_MAP.get(game_name)
    if not entry:
        return None
    mod_path, cls_name = entry
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Orak benchmark (krafton-ai/Orak) game environment imports
# ---------------------------------------------------------------------------
_ORAK_REPO = CODEBASE_ROOT.parent / "Orak"
_ORAK_SRC = _ORAK_REPO / "src"
_ORAK_MCP_GAMES = _ORAK_SRC / "mcp_game_servers"
_ORAK_MCP_AGENTS = _ORAK_SRC / "mcp_agent_client"
for _orak_p in [str(_ORAK_SRC), str(_ORAK_MCP_GAMES)]:
    if Path(_orak_p).exists() and _orak_p not in sys.path:
        sys.path.insert(0, _orak_p)


# ---------------------------------------------------------------------------
# Game registry: name -> (env_class, config_path, action_names)
# ---------------------------------------------------------------------------

def _config_path(game_dir: str) -> str:
    return str(GAMINGAGENT_ROOT / "gamingagent" / "envs" / game_dir / "game_env_config.json")


def _adapter_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """Standard GymEnvAdapter kwargs (2048, sokoban, tetris)."""
    return dict(
        render_mode=None,
        observation_mode_for_adapter="text",
        agent_cache_dir_for_adapter=cache_dir,
        game_specific_config_path_for_adapter=config_path,
    )


def _candy_crush_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """CandyCrushEnv — adapter pattern but no render_mode parameter."""
    return dict(
        observation_mode_for_adapter="text",
        agent_cache_dir_for_adapter=cache_dir,
        game_specific_config_path_for_adapter=config_path,
    )


def _retro_kwargs(game_name: str):
    """Factory for retro envs (SuperMarioBros, NineteenFortyTwo) that take
    (game_name, config_dir_path, observation_mode, base_log_dir)."""
    def _build(cache_dir: str, config_path: str) -> Dict[str, Any]:
        return dict(
            game_name=game_name,
            config_dir_path=str(Path(config_path).parent),
            observation_mode="text",
            base_log_dir=cache_dir,
        )
    return _build


def _doom_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """DoomEnvWrapper — retro-style args plus render_mode."""
    return dict(
        game_name="doom",
        config_dir_path=str(Path(config_path).parent),
        observation_mode="text",
        base_log_dir=cache_dir,
        render_mode=None,
        headless=True,
    )


def _pokemon_red_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """PokemonRedEnv — adapter pattern but also needs rom_path from config."""
    import json as _json
    rom_path = None
    try:
        with open(config_path) as _f:
            _cfg = _json.load(_f)
            rom_path = _cfg.get("env_init_kwargs", {}).get("rom_path")
    except Exception:
        pass
    if rom_path and not os.path.isabs(rom_path):
        rom_path = str(GAMINGAGENT_ROOT / rom_path)
    kwargs: Dict[str, Any] = dict(
        render_mode=None,
        observation_mode_for_adapter="text",
        agent_cache_dir_for_adapter=cache_dir,
        game_specific_config_path_for_adapter=config_path,
    )
    if rom_path:
        kwargs["rom_path"] = rom_path
    return kwargs


def _ace_attorney_kwargs(cache_dir: str, config_path: str) -> Dict[str, Any]:
    """AceAttorneyEnv — uses adapter_ prefixed args."""
    return dict(
        adapter_observation_mode="text",
        adapter_agent_cache_dir=cache_dir,
        adapter_config_path=config_path,
    )


# "lazy" = load env class on first use (avoids importing retro/pyglet on headless servers)
GAME_REGISTRY: Dict[str, Dict[str, Any]] = {
    "twenty_forty_eight": {
        "env_class": "lazy",
        "config_path": _config_path("custom_01_2048"),
        "action_names": ["up", "down", "left", "right"],
        "task": "Achieve the highest possible tile in 2048 by merging tiles strategically.",
        "init_kwargs": _adapter_kwargs,
    },
    "sokoban": {
        "env_class": "lazy",
        "config_path": _config_path("custom_02_sokoban"),
        "action_names": ["up", "down", "left", "right", "push up", "push down", "push left", "push right", "no_op"],
        "task": "Push all boxes onto goal positions in the Sokoban puzzle.",
        "init_kwargs": _adapter_kwargs,
    },
    "candy_crush": {
        "env_class": "lazy",
        "config_path": _config_path("custom_03_candy_crush"),
        "action_names": [],  # dynamic: swap(row1,col1,row2,col2) on 8x8 board
        "task": "Match three or more candies in a row/column to clear the board and maximize score.",
        "init_kwargs": _candy_crush_kwargs,
    },
    "tetris": {
        "env_class": "lazy",
        "config_path": _config_path("custom_04_tetris"),
        "action_names": ["no_op", "left", "right", "rotate_left", "rotate_right", "soft_drop", "hard_drop"],
        "task": "Clear as many lines as possible in Tetris by placing tetrominoes strategically.",
        "init_kwargs": _adapter_kwargs,
    },
    "doom": {
        "env_class": "lazy",
        "config_path": _config_path("custom_05_doom"),
        "action_names": ["move_left", "move_right", "attack"],
        "task": "Survive and defeat enemies in Doom by shooting and dodging.",
        "init_kwargs": _doom_kwargs,
    },
    "pokemon_red": {
        "env_class": "lazy",
        "config_path": _config_path("custom_06_pokemon_red"),
        "action_names": ["a", "b", "start", "select", "up", "down", "left", "right"],
        "task": "Progress through Pokemon Red by exploring, battling, and catching Pokemon.",
        "init_kwargs": _pokemon_red_kwargs,
    },
    "super_mario_bros": {
        "env_class": "lazy",
        "config_path": _config_path("retro_01_super_mario_bros"),
        "action_names": ["noop", "right", "right_a", "right_b", "right_a_b", "a", "b", "left", "left_a", "left_b", "left_a_b", "down", "up"],
        "task": "Complete levels in Super Mario Bros by running, jumping, and avoiding enemies.",
        "init_kwargs": _retro_kwargs("super_mario_bros"),
    },
    "ace_attorney": {
        "env_class": "lazy",
        "config_path": _config_path("retro_02_ace_attorney"),
        "action_names": ["a", "b", "l", "r", "up", "down", "left", "right", "no_op"],
        "task": "Solve cases in Ace Attorney by investigating evidence and cross-examining witnesses.",
        "init_kwargs": _ace_attorney_kwargs,
    },
    "nineteen_forty_two": {
        "env_class": "lazy",
        "config_path": _config_path("retro_03_1942"),
        "action_names": ["noop", "right", "right_b", "a", "b", "left", "left_b", "down", "up"],
        "task": "Survive waves of enemy aircraft in 1942 by dodging and shooting.",
        "init_kwargs": _retro_kwargs("nineteen_forty_two"),
    },
    "tic_tac_toe": {
        "env_class": "lazy",
        "config_path": _config_path("zoo_01_tictactoe"),
        "action_names": ["place 0", "place 1", "place 2", "place 3", "place 4", "place 5", "place 6", "place 7", "place 8"],
        "task": "Win at Tic-Tac-Toe by placing marks to get three in a row.",
        "init_kwargs": _adapter_kwargs,
    },
    "texas_holdem": {
        "env_class": "lazy",
        "config_path": _config_path("zoo_02_texasholdem"),
        "action_names": ["call", "raise", "fold", "check"],
        "task": "Win at Texas Hold'em poker by making optimal betting decisions.",
        "init_kwargs": _adapter_kwargs,
    },
    # ── Orak benchmark (krafton-ai/Orak, 12 games, 6 genres) ─────────
    # All use Orak's BaseEnv implementations, loaded via make_orak_env.
    # Puzzle
    "orak_twenty_fourty_eight": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "twenty_fourty_eight" / "config.yaml"),
        "action_names": ["up", "down", "left", "right"],
        "task": "Merge tiles to reach 2048. Score = min(score/20000*100, 100).",
        "init_kwargs": None,
    },
    "orak_baba_is_you": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "baba_is_you" / "config.yaml"),
        "action_names": ["idle", "left", "right", "up", "down"],
        "task": "Solve rule-manipulation puzzles by pushing word blocks. 100=win, 40=WIN exists, 20=WALL broken.",
        "init_kwargs": None,
    },
    # Action
    "orak_super_mario": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "super_mario" / "config.yaml"),
        "action_names": [f"Jump Level : {i}" for i in range(7)],
        "task": "Advance Mario right. Score = x_pos/3161*100.",
        "init_kwargs": None,
    },
    "orak_street_fighter": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "street_fighter" / "config.yaml"),
        "action_names": [
            "Move Closer", "Move Away", "Fireball", "Megapunch", "Hurricane",
            "Low Kick", "Medium Kick", "High Kick", "Jump Closer", "Jump Away",
            "Crouch", "Block", "Low Punch", "Medium Punch", "High Punch",
        ],
        "task": "Defeat opponents in Street Fighter III. Score = stages cleared.",
        "init_kwargs": None,
    },
    # Strategy
    "orak_star_craft": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "star_craft" / "config.yaml"),
        "action_names": [
            "TRAIN PROBE", "TRAIN ZEALOT", "TRAIN STALKER", "BUILD PYLON",
            "BUILD GATEWAY", "BUILD ASSIMILATOR", "BUILD NEXUS",
            "BUILD CYBERNETICSCORE", "RESEARCH WARPGATERESEARCH",
            "RESEARCH CHARGE", "SCOUTING PROBE", "MULTI-ATTACK",
            "MULTI-RETREAT", "CHRONOBOOST NEXUS", "EMPTY ACTION",
        ],
        "task": "Win 1v1 as Protoss vs Zerg bot. Provide 5 sequential macro actions per step.",
        "init_kwargs": None,
    },
    "orak_star_craft_multi": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "star_craft_multi" / "config.yaml"),
        "action_names": [],
        "task": "Win 1v1 StarCraft II (2-player). Provide 5 actions per step.",
        "init_kwargs": None,
    },
    "orak_slay_the_spire": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "slay_the_spire" / "config.yaml"),
        "action_names": ["PLAY", "END", "CHOOSE", "SKIP"],
        "task": "Climb the Spire with card combos. Score = floor reached (max 50).",
        "init_kwargs": None,
    },
    # RPG
    "orak_pokemon_red": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "pokemon_red" / "config.yaml"),
        "action_names": [
            "up", "down", "left", "right", "a", "b", "start", "select",
            "move_to", "interact_with_object", "warp_with_warp_point",
            "continue_dialog", "select_move_in_battle",
            "switch_pkmn_in_battle", "run_away", "use_item_in_battle",
        ],
        "task": "Progress through Pokemon Red storyline milestones (0-12 flags).",
        "init_kwargs": None,
    },
    "orak_darkest_dungeon": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "darkest_dungeon" / "config.yaml"),
        "action_names": ["attack", "heal", "swap", "idle", "skip"],
        "task": "Survive dungeon raids. Score = 0.4*combat + 0.3*survival + 0.3*(1-stress).",
        "init_kwargs": None,
    },
    # Adventure
    "orak_pwaat": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "pwaat" / "config.yaml"),
        "action_names": ["Ok", "Back", "Down", "Up", "Left", "Right", "Present evidence", "Press"],
        "task": "Solve cases in Ace Attorney. Score = milestone rewards.",
        "init_kwargs": None,
    },
    "orak_her_story": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "her_story" / "config.yaml"),
        "action_names": ["Search", "Play Video"],
        "task": "Uncover the story by searching keywords and watching videos. Score = views/272.",
        "init_kwargs": None,
    },
    # Simulation
    "orak_minecraft": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "minecraft" / "config.yaml"),
        "action_names": [],
        "task": "Craft target items in Minecraft via JavaScript async functions.",
        "init_kwargs": None,
    },
    "orak_stardew_valley": {
        "env_class": "orak",
        "config_path": str(_ORAK_MCP_AGENTS / "configs" / "stardew_valley" / "config.yaml"),
        "action_names": [
            "till_soil", "plant_seeds", "water_seeds", "harvest_crops",
            "sell_item", "buy_item", "get_out_of_house", "go_house_and_sleep",
        ],
        "task": "Complete farming tasks in Stardew Valley.",
        "init_kwargs": None,
    },
}

# Max steps per game so cold-start runs until natural end (no truncation before win/lose).
# Used when --max_steps is not passed in run_100_rollouts.py and generate_cold_start_gpt54.py.
COLD_START_MAX_STEPS_NATURAL_END: Dict[str, int] = {
    "twenty_forty_eight": 200,
    "sokoban": 200,
    "candy_crush": 50,
    "tetris": 200,
    "pokemon_red": 200,  # align with orak-2025-starter-kit & Orak benchmark
}


def get_cold_start_max_steps(game_name: str, override: Optional[int] = None) -> int:
    """Return max_steps for cold-start: override if set, else per-game natural-end limit."""
    if override is not None:
        return override
    return COLD_START_MAX_STEPS_NATURAL_END.get(game_name, 200)


# ---------------------------------------------------------------------------
# Lightweight NL wrapper for GamingAgent envs
# ---------------------------------------------------------------------------

class ColdStartEnvWrapper:
    """
    Thin wrapper that adapts GamingAgent envs (which return Observation objects)
    into the standard (obs_str, info) interface used by decision agents.
    """

    def __init__(self, game_name: str, max_steps: int = 100):
        reg = GAME_REGISTRY.get(game_name)
        if reg is None or reg["env_class"] is None:
            raise ValueError(f"Game '{game_name}' not available. Install GamingAgent and check imports.")

        self._action_names = reg["action_names"]
        self._game_name = game_name
        self._max_steps = max_steps
        self._step_count = 0
        self._is_orak = (reg["env_class"] == "orak")

        if self._is_orak:
            from evaluate_orak.orak_nl_wrapper import make_orak_env
            orak_short = game_name.replace("orak_", "")
            self._orak_wrapper = make_orak_env(orak_short, max_steps=max_steps)
            self._env = self._orak_wrapper.env
        else:
            self._orak_wrapper = None
            cache_dir = str(SCRIPT_DIR / "cache" / game_name)
            os.makedirs(cache_dir, exist_ok=True)
            kwargs_fn = reg.get("init_kwargs", _adapter_kwargs)
            env_kwargs = kwargs_fn(cache_dir, reg["config_path"])
            env_class = reg["env_class"]
            if env_class == "lazy":
                env_class = _get_env_class(game_name)
                if env_class is None:
                    raise ValueError(f"Game '{game_name}' env could not be imported.")
            self._env = env_class(**env_kwargs)

        self._dynamic_actions = False
        self._env_action_idx_to_str: Dict[int, str] = {}
        if not self._action_names and hasattr(self._env, 'env_action_idx_to_move') and self._env.env_action_idx_to_move:
            self._dynamic_actions = True
            self._env_action_idx_to_str = dict(self._env.env_action_idx_to_move)
            self._action_names = list(self._env_action_idx_to_str.values())

    def _effective_action_strs(self, info: Dict[str, Any]) -> List[str]:
        """Convert effective_actions (int indices) from env info to action strings."""
        if not self._dynamic_actions:
            return self._action_names
        eff_indices = info.get("effective_actions", [])
        if not eff_indices:
            return self._action_names
        eff_strs = [self._env_action_idx_to_str[i] for i in eff_indices if i in self._env_action_idx_to_str]
        return eff_strs if eff_strs else self._action_names

    def reset(self, seed=None, options=None) -> Tuple[str, Dict[str, Any]]:
        if self._is_orak:
            text, info = self._orak_wrapper.reset(seed=seed)
            self._step_count = 0
            info["game"] = "orak"
            info["raw_obs"] = text
            info["available_actions"] = self._action_names
            return text, info

        obs, info = self._env.reset(seed=seed)
        self._step_count = 0
        text = self._obs_to_text(obs)
        info["raw_obs"] = obs
        info["action_names"] = self._action_names
        info["available_actions"] = self._effective_action_strs(info)
        info["game"] = GAME_GAMINGAGENT
        info["env_name"] = "gamingagent"
        info["game_name"] = self._game_name
        return text, info

    def step(self, action) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._is_orak:
            action_str = str(action).strip() if action is not None else ""
            text, reward, terminated, truncated, info = self._orak_wrapper.step(action_str)
            self._step_count += 1
            info["game"] = "orak"
            info["perf_score"] = info.get("score", 0.0)
            info["raw_obs"] = text
            info["available_actions"] = self._action_names
            return text, reward, terminated, truncated, info

        action_str = str(action).strip() if action is not None else ""
        action_str = self._validate_action(action_str)
        result = self._env.step(agent_action_str=action_str)
        obs, reward, terminated, truncated, info = result[0], result[1], result[2], result[3], result[4]
        self._step_count += 1
        if self._step_count >= self._max_steps:
            truncated = True
        text = self._obs_to_text(obs)
        info["raw_obs"] = obs
        info["action_names"] = self._action_names
        info["available_actions"] = self._effective_action_strs(info)
        info["game"] = GAME_GAMINGAGENT
        info["env_name"] = "gamingagent"
        info["game_name"] = self._game_name
        info["perf_score"] = result[5] if len(result) > 5 else 0.0
        return text, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def _validate_action(self, action_str: str) -> str:
        """Ensure action_str is valid for this game; fall back to a random valid action."""
        if not self._action_names:
            return action_str
        if action_str in self._action_names:
            return action_str
        lower_map = {a.lower(): a for a in self._action_names}
        canonical = lower_map.get(action_str.lower())
        if canonical:
            return canonical
        import random
        fallback = random.choice(self._action_names)
        print(f"    [WARN] Invalid action '{action_str}' for {self._game_name}, "
              f"using random fallback '{fallback}'")
        return fallback

    def _obs_to_text(self, obs) -> str:
        if isinstance(obs, str):
            return obs
        if hasattr(obs, "textual_representation") and obs.textual_representation:
            return str(obs.textual_representation)
        if isinstance(obs, dict):
            return str(obs.get("text", obs.get("textual_representation", str(obs))))
        return str(obs)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_dummy_agent_episode(
    env: ColdStartEnvWrapper,
    game_name: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with the dummy language_agent_action (GPT function calling)."""
    task = GAME_REGISTRY[game_name]["task"]
    action_names = GAME_REGISTRY[game_name]["action_names"]

    obs, info = env.reset()
    raw_obs = info.get("raw_obs")
    curr_available_actions = info.get("available_actions") or action_names
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0

    while step_count < max_steps:
        step_actions = curr_available_actions if curr_available_actions else action_names
        action = language_agent_action(
            state_nl=obs + f"\n\nValid actions: {', '.join(step_actions)}. Choose one.",
            game=GAME_GAMINGAGENT,
            model=model,
            use_function_call=True,
            temperature=0.3,
        )

        if step_actions and str(action) not in step_actions:
            lower_map = {a.lower(): a for a in step_actions}
            canonical = lower_map.get(str(action).lower().strip())
            if canonical:
                action = canonical
            else:
                import random
                action = random.choice(step_actions)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_raw_obs = next_info.get("raw_obs")
        done = terminated or truncated
        total_reward += reward

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            tasks=task,
        )
        exp.idx = step_count
        exp.raw_state = str(raw_obs) if raw_obs is not None else None
        exp.raw_next_state = str(next_raw_obs) if next_raw_obs is not None else None
        exp.available_actions = list(step_actions) if step_actions else None
        exp.interface = {"env_name": "gamingagent", "game_name": game_name}
        experiences.append(exp)

        if verbose:
            print(f"  step {step_count}: action={action}, reward={reward:.2f}, cum={total_reward:.2f}")

        obs = next_obs
        raw_obs = next_raw_obs
        curr_available_actions = next_info.get("available_actions") or action_names
        info = next_info
        step_count += 1

        if done:
            break

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=game_name,
    )
    episode.set_outcome()
    stats = {
        "game": game_name,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "dummy",
    }
    return episode, stats


def run_vlm_agent_episode(
    env: ColdStartEnvWrapper,
    game_name: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with VLMDecisionAgent (prompt decision agent).

    Delegates to run_episode_vlm_agent which returns a fully-populated
    Episode (Experience objects with summary_state, intentions, sub_tasks,
    reward_details) ready for direct ingestion by the skill agents pipeline.
    """
    task = GAME_REGISTRY[game_name]["task"]

    agent = VLMDecisionAgent(model=model, game=GAME_GAMINGAGENT)

    episode = run_episode_vlm_agent(
        env,
        agent=agent,
        task=task,
        max_steps=max_steps,
        verbose=verbose,
    )

    meta = episode.metadata or {}
    stats = {
        "game": game_name,
        "steps": meta.get("steps", len(episode.experiences)),
        "total_reward": episode.get_reward(),
        "total_shaped_reward": episode.get_total_reward(),
        "terminated": meta.get("done", False),
        "truncated": False,
        "model": model,
        "agent_type": "vlm",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Trajectory labeling with GPT-5-mini
# ---------------------------------------------------------------------------

def label_trajectory(episode: Episode, model: str) -> Episode:
    """
    Use GPT-5-mini to generate summaries and intention labels for each experience
    in the episode, and segment into sub-tasks for initial skill seeds.
    """
    print(f"  Labeling trajectory ({len(episode.experiences)} steps) with {model}...")

    for i, exp in enumerate(episode.experiences):
        history = episode.experiences[max(0, i - 3):i]
        try:
            exp.generate_summary()
        except Exception as e:
            exp.summary = f"Step {i}: action={exp.action}"
        try:
            exp.generate_intentions(history)
        except Exception as e:
            exp.intentions = "unknown"

    try:
        episode.generate_summary()
    except Exception:
        episode.summary = f"Episode with {len(episode.experiences)} steps, reward={episode.get_reward():.2f}"

    sub_episodes = episode.separate_into_sub_episodes(outcome_length=3)
    for sub_ep in sub_episodes:
        try:
            sub_ep.generate_summary()
        except Exception:
            pass
        try:
            sub_ep.sub_task_labeling()
        except Exception:
            pass

    return episode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate cold-start data for Game-AI-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", type=str, default="twenty_forty_eight",
                        choices=list(GAME_REGISTRY.keys()),
                        help="Game to generate data for")
    parser.add_argument("--all_games", action="store_true",
                        help="Generate data for all available games")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per game")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="LLM model to use for agent and labeling")
    parser.add_argument("--agent_type", type=str, default="dummy",
                        choices=["dummy", "vlm"],
                        help="Agent type: 'dummy' (language_agent_action) or 'vlm' (VLMDecisionAgent)")
    parser.add_argument("--label", action="store_true",
                        help="Label trajectories with LLM after generation (default: off; use labeling/ for that)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling (default: no labeling)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/data/)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from api_keys import open_router_api_key
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or (open_router_api_key and open_router_api_key.strip()))
    except Exception:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        print("[WARNING] No API key set. LLM calls will fail.")
        print("  Prefer: open_router_api_key in api_keys.py or export OPENROUTER_API_KEY='sk-or-...'")
        print("  Or: export OPENAI_API_KEY='sk-...'")

    games = list(GAME_REGISTRY.keys()) if args.all_games else [args.game]
    available_games = [
        g for g in games
        if GAME_REGISTRY[g]["env_class"] is not None
        and (GAME_REGISTRY[g]["env_class"] == "orak" or GAME_REGISTRY[g]["env_class"] is not None)
    ]

    if not available_games:
        print("[ERROR] No games available. Ensure GamingAgent is installed.")
        sys.exit(1)

    print("=" * 78)
    print("  Cold-Start Data Generation")
    print("=" * 78)
    print(f"  Games:      {', '.join(available_games)}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Model:      {args.model}")
    print(f"  Agent:      {args.agent_type}")
    print(f"  Labeling:   {args.label and not args.no_label}")
    print(f"  Output:     {output_dir}")
    print("=" * 78)

    run_fn = run_vlm_agent_episode if args.agent_type == "vlm" else run_dummy_agent_episode
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for game_name in available_games:
        print(f"\n{'─' * 78}")
        print(f"  Game: {game_name}")
        print(f"{'─' * 78}")

        game_output_dir = output_dir / game_name
        game_output_dir.mkdir(parents=True, exist_ok=True)

        episode_buffer = Episode_Buffer(buffer_size=1000)

        for ep_idx in range(args.episodes):
            print(f"\n  Episode {ep_idx + 1}/{args.episodes}")

            try:
                env = ColdStartEnvWrapper(game_name, max_steps=args.max_steps)
                episode, stats = run_fn(
                    env=env,
                    game_name=game_name,
                    model=args.model,
                    max_steps=args.max_steps,
                    verbose=args.verbose,
                )
                env.close()

                print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

                if args.label and not args.no_label:
                    episode = label_trajectory(episode, args.model)

                episode_buffer.add_episode(episode)
                all_stats.append(stats)

                # Save individual episode
                ep_path = game_output_dir / f"episode_{ep_idx:03d}.json"
                ep_data = episode.to_dict()
                ep_data["metadata"] = stats
                with open(ep_path, "w", encoding="utf-8") as f:
                    json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            except Exception as e:
                print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save episode buffer for this game
        buffer_path = game_output_dir / "episode_buffer.json"
        episode_buffer.save_to_json(str(buffer_path))
        print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    elapsed = time.time() - t0

    # Save overall run summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": args.agent_type,
        "games": available_games,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    summary_path = output_dir / "cold_start_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  COLD-START GENERATION COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Total episodes: {len(all_stats)}")
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
        print(f"  Mean reward:  {sum(rewards) / len(rewards):.2f}")
        print(f"  Mean steps:   {sum(steps) / len(steps):.1f}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Output:       {output_dir}")
    print(f"  Summary:      {summary_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
