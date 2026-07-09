"""
Env wrappers: convert environment state to/from natural language for language-model agents.

Available wrappers:
  - OvercookedNLWrapper:    Overcooked cooperative cooking (2 agents)
  - AvalonNLWrapper:       Avalon hidden-role deduction (5-10 agents)
  - DiplomacyNLWrapper:    Diplomacy strategic negotiation (7 agents / powers)
  - GamingAgentNLWrapper:  GamingAgent / LMGame-Bench (2048, Sokoban, Tetris, etc.)
  - VideoGameBenchNLWrapper: VideoGameBench Game Boy games (PyBoy)
"""

from env_wrappers.overcooked_nl_wrapper import (
    OvercookedNLWrapper,
    joint_action_to_indices,
    natural_language_to_action_index,
    state_to_natural_language,
    state_to_natural_language_for_all_agents,
)

from env_wrappers.avalon_nl_wrapper import (
    AvalonNLWrapper,
    state_to_natural_language as avalon_state_to_nl,
    state_to_natural_language_for_all as avalon_state_to_nl_all,
    parse_vote as avalon_parse_vote,
    parse_team as avalon_parse_team,
    parse_target as avalon_parse_target,
)

from env_wrappers.diplomacy_nl_wrapper import (
    DiplomacyNLWrapper,
    state_to_natural_language as diplomacy_state_to_nl,
    state_to_natural_language_for_all as diplomacy_state_to_nl_all,
    parse_orders as diplomacy_parse_orders,
)

from env_wrappers.gamingagent_nl_wrapper import (
    GamingAgentNLWrapper,
    state_to_natural_language as gamingagent_state_to_nl,
)

from env_wrappers.videogamebench_nl_wrapper import (
    VideoGameBenchNLWrapper,
    state_to_natural_language as videogamebench_state_to_nl,
    natural_language_to_action_index as videogamebench_nl_to_action,
    VIDEOGAMEBENCH_BUTTON_NAMES,
)

from env_wrappers.videogamebench_dos_nl_wrapper import (
    VideoGameBenchDOSNLWrapper,
    list_dos_games,
    state_to_natural_language as videogamebench_dos_state_to_nl,
    VIDEOGAMEBENCH_DOS_VALID_KEYS,
)

__all__ = [
    # Overcooked
    "OvercookedNLWrapper",
    "joint_action_to_indices",
    "natural_language_to_action_index",
    "state_to_natural_language",
    "state_to_natural_language_for_all_agents",
    # Avalon
    "AvalonNLWrapper",
    "avalon_state_to_nl",
    "avalon_state_to_nl_all",
    "avalon_parse_vote",
    "avalon_parse_team",
    "avalon_parse_target",
    # Diplomacy
    "DiplomacyNLWrapper",
    "diplomacy_state_to_nl",
    "diplomacy_state_to_nl_all",
    "diplomacy_parse_orders",
    # GamingAgent
    "GamingAgentNLWrapper",
    "gamingagent_state_to_nl",
    # VideoGameBench (Game Boy - deprecated in evaluate_videogamebench)
    "VideoGameBenchNLWrapper",
    "videogamebench_state_to_nl",
    "videogamebench_nl_to_action",
    "VIDEOGAMEBENCH_BUTTON_NAMES",
    # VideoGameBench DOS (DOS games only, no ROMs)
    "VideoGameBenchDOSNLWrapper",
    "list_dos_games",
    "videogamebench_dos_state_to_nl",
    "VIDEOGAMEBENCH_DOS_VALID_KEYS",
]
