"""
Unified game-environment package for Game-AI-Agent.

Includes NL wrappers (convert game state to/from natural language),
Gymnasium-compatible adapters, game configurations, and benchmark runners
for all supported environments.

NL Wrappers:
  - GamingAgentNLWrapper:  GamingAgent / LMGame-Bench (2048, Candy Crush, Tetris)
  - OrakNLWrapper:         Orak environments (Super Mario)
  - OSWorldNLWrapper:      OSWorld desktop automation (Ubuntu/Windows VMs)
  - ALFWorldNLWrapper:     ALFWorld TextWorld household tasks
  - TetrisMacroWrapper:    Tetris macro-action wrapper (placement-level actions)

Evaluation helpers:
  - game_configs:          Per-game default configs (GameConfig, GAME_CONFIGS)
  - gym_like:              Gymnasium adapter for GamingAgent (make_gaming_env)
  - osworld_wrapper:       Gymnasium adapter for OSWorld (OSWorldGymWrapper)
  - run_benchmark:         CLI benchmark runner for LMGame-Bench
  - run_orak_benchmark:    CLI benchmark runner for Orak games
"""

from env_wrappers.gamingagent_nl_wrapper import (
    GamingAgentNLWrapper,
    state_to_natural_language as gamingagent_state_to_nl,
)

from env_wrappers.orak_nl_wrapper import (
    ORAK_GAMES,
    OrakNLWrapper,
    make_orak_env,
)

from env_wrappers.osworld_wrapper import OSWorldGymWrapper, load_task_catalog
from env_wrappers.osworld_nl_wrapper import (
    OSWorldNLWrapper,
    obs_to_natural_language as osworld_obs_to_nl,
    build_osworld_state_summary,
)

from env_wrappers.alfworld_nl_wrapper import (
    ALFWorldNLWrapper,
    alfworld_obs_to_natural_language,
    make_alfworld_env,
)

from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper as TetrisMacroWrapper

from env_wrappers.game_configs import (
    GAME_CONFIGS,
    GameConfig,
    ALL_GAME_NAMES,
    AVAILABLE_GAME_NAMES,
    TOTAL_GAMES,
    AVAILABLE_GAMES,
)

from env_wrappers.gym_like import make_gaming_env, list_games

from env_wrappers.visual_utils import get_obs_image, get_obs_pil_image

__all__ = [
    # GamingAgent
    "GamingAgentNLWrapper",
    "gamingagent_state_to_nl",
    # Orak
    "ORAK_GAMES",
    "OrakNLWrapper",
    "make_orak_env",
    # OSWorld
    "OSWorldGymWrapper",
    "OSWorldNLWrapper",
    "load_task_catalog",
    "osworld_obs_to_nl",
    "build_osworld_state_summary",
    # ALFWorld
    "ALFWorldNLWrapper",
    "alfworld_obs_to_natural_language",
    "make_alfworld_env",
    # Tetris Macro
    "TetrisMacroWrapper",
    # Game configs & Gymnasium adapters
    "GAME_CONFIGS",
    "GameConfig",
    "ALL_GAME_NAMES",
    "AVAILABLE_GAME_NAMES",
    "TOTAL_GAMES",
    "AVAILABLE_GAMES",
    "make_gaming_env",
    "list_games",
    # Visual helpers (cross-env)
    "get_obs_image",
    "get_obs_pil_image",
]
