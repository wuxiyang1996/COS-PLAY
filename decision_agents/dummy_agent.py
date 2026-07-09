"""
Dummy language agent for multi-game environments: takes state (natural language)
and generates action (natural language) using an LLM (GPT / ask_model).

Supports all three env_wrappers:
  - OvercookedNLWrapper: cooperative cooking (actions: north/south/east/west/stay/interact)
  - AvalonNLWrapper:     hidden-role deduction (actions: team proposals, approve/reject, pass/fail, target)
  - DiplomacyNLWrapper:  strategic negotiation (actions: order strings like "A PAR - BUR")

The agent auto-detects which game is being played from the observation text, or
can be told explicitly via the `game` parameter.

Experience Collection:
  Use `run_episode_with_experience_collection()` to run a full episode and automatically
  collect all experiences in an Episode object following the data structure defined
  in data_structure.experience. Each step creates an Experience object with state,
  action, reward, next_state, and done fields, all stored within the Episode.

Usage with OvercookedNLWrapper:

  from env_wrappers import OvercookedNLWrapper
  from agents.dummy_agent import language_agent_action

  env = OvercookedNLWrapper(env)
  state_nl, info = env.reset()
  action_nl = language_agent_action(state_nl, model="gpt-4o-mini")
  state_nl, reward, term, trunc, info = env.step(action_nl)

Usage with AvalonNLWrapper (single-agent):

  from env_wrappers import AvalonNLWrapper
  from agents.dummy_agent import language_agent_action

  env = AvalonNLWrapper(num_players=5, controlled_player=0)
  obs, info = env.reset()
  action = language_agent_action(obs, model="gpt-4o-mini")
  obs, reward, term, trunc, info = env.step(action)

Usage with AvalonNLWrapper (multi-agent):

  from env_wrappers import AvalonNLWrapper
  from agents.dummy_agent import language_agent_action

  env = AvalonNLWrapper(num_players=5)
  obs, info = env.reset()         # obs: {player_id: nl_str}
  actions = {
      pid: language_agent_action(obs_str, model="gpt-4o-mini")
      for pid, obs_str in obs.items()
  }
  obs, rewards, term, trunc, info = env.step(actions)

Usage with DiplomacyNLWrapper (single-agent):

  from env_wrappers import DiplomacyNLWrapper
  from agents.dummy_agent import language_agent_action

  env = DiplomacyNLWrapper(controlled_power="FRANCE")
  obs, info = env.reset()
  action = language_agent_action(obs, model="gpt-4o-mini")
  obs, reward, term, trunc, info = env.step(action)

Usage with DiplomacyNLWrapper (multi-agent):

  from env_wrappers import DiplomacyNLWrapper
  from agents.dummy_agent import language_agent_action

  env = DiplomacyNLWrapper()
  obs, info = env.reset()         # obs: {power_name: nl_str}
  actions = {
      power: language_agent_action(obs_str, model="gpt-4o-mini")
      for power, obs_str in obs.items()
  }
  obs, rewards, term, trunc, info = env.step(actions)
"""

import json
import re
from typing import List, Optional, Union, Dict, Any

# Optional: use API_func.ask_model for generic model routing
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

# Import Experience and Episode classes, and buffers
try:
    from data_structure.experience import Experience, Episode, Experience_Replay_Buffer, Episode_Buffer
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import experience classes: {e}. Experience collection features will be unavailable.")
    Experience = None
    Episode = None
    Experience_Replay_Buffer = None
    Episode_Buffer = None


# ---------------------------------------------------------------------------
# Game type constants
# ---------------------------------------------------------------------------
GAME_OVERCOOKED = "overcooked"
GAME_AVALON = "avalon"
GAME_DIPLOMACY = "diplomacy"
GAME_GAMINGAGENT = "gamingagent"
GAME_VIDEOGAMEBENCH = "videogamebench"
GAME_VIDEOGAMEBENCH_DOS = "videogamebench_dos"
GAME_ORAK = "orak"
GAME_ORAK_STARCRAFT = "orak_starcraft"
GAME_ORAK_STARCRAFT_MULTI = "orak_starcraft_multi"
GAME_ORAK_MARIO = "orak_mario"
GAME_ORAK_POKEMON = "orak_pokemon"
GAME_ORAK_2048 = "orak_2048"
GAME_ORAK_STREET_FIGHTER = "orak_street_fighter"
GAME_ORAK_SLAY_THE_SPIRE = "orak_slay_the_spire"
GAME_ORAK_DARKEST_DUNGEON = "orak_darkest_dungeon"
GAME_ORAK_PWAAT = "orak_pwaat"
GAME_ORAK_HER_STORY = "orak_her_story"
GAME_ORAK_MINECRAFT = "orak_minecraft"
GAME_ORAK_STARDEW_VALLEY = "orak_stardew_valley"
GAME_ORAK_BABA_IS_YOU = "orak_baba_is_you"


# ---------------------------------------------------------------------------
# Overcooked constants
# ---------------------------------------------------------------------------
OVERCOOKED_VALID_ACTIONS = ("north", "south", "east", "west", "stay", "interact")

OVERCOOKED_SYSTEM_PROMPT = (
    "You are playing Overcooked, a cooperative cooking game. You control one chef.\n"
    "Your only goal is to choose one action each turn. You must reply with exactly one action.\n\n"
    "Valid actions (choose exactly one):\n"
    "- north  (move up)\n"
    "- south  (move down)\n"
    "- east   (move right)\n"
    "- west   (move left)\n"
    "- stay   (do not move)\n"
    "- interact (pick up item, put down item, pick up from dispenser, interact with pot, or serve soup)\n\n"
    "Reply with only the single word: north, south, east, west, stay, or interact. No explanation."
)

OVERCOOKED_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action do you take? Reply with exactly one word: north, south, east, west, stay, or interact."
)


# ---------------------------------------------------------------------------
# Avalon constants
# ---------------------------------------------------------------------------
AVALON_SYSTEM_PROMPT = (
    "You are playing Avalon, a hidden-role social deduction game.\n"
    "You must carefully read the game state and choose your action based on the current phase.\n\n"
    "Phase actions:\n"
    "- Team Selection (you are leader): Reply with player IDs separated by commas, e.g. \"0, 2, 3\"\n"
    "- Team Voting: Reply with exactly \"approve\" or \"reject\"\n"
    "- Quest Voting (you are on the team): Reply with exactly \"pass\" or \"fail\"\n"
    "- Assassination (you are the Assassin): Reply with the player ID to assassinate, e.g. \"2\"\n\n"
    "If you are not the active player (e.g. not the leader in Team Selection, or not on the quest team),\n"
    "reply with \"wait\" (the wrapper handles non-active players automatically).\n\n"
    "Think strategically about your role and what you know. Reply concisely with your action."
)

AVALON_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What is your action? Follow the instructions in the state description above."
)


# ---------------------------------------------------------------------------
# Diplomacy constants
# ---------------------------------------------------------------------------
DIPLOMACY_SYSTEM_PROMPT = (
    "You are playing Diplomacy, a strategic negotiation board game.\n"
    "You control one of seven European powers. Each turn you issue orders for your units.\n\n"
    "Order formats:\n"
    "  Hold:         A PAR H\n"
    "  Move:         A PAR - BUR\n"
    "  Support hold: A MAR S A PAR\n"
    "  Support move: A MAR S A PAR - BUR\n"
    "  Convoy:       F ENG C A LON - BRE\n"
    "  Retreat:      A PAR R MAR\n"
    "  Build:        A PAR B\n"
    "  Disband:      A PAR D\n\n"
    "Reply with your orders as a JSON list of strings, e.g.:\n"
    "[\"A PAR - BUR\", \"F BRE - ENG\", \"A MAR H\"]\n\n"
    "If you have no units or the game is over, reply with an empty list: []\n"
    "Choose orders from the possible orders listed in the state description."
)

DIPLOMACY_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "Submit your orders as a JSON list of order strings."
)


# ---------------------------------------------------------------------------
# GamingAgent constants
# ---------------------------------------------------------------------------
GAMINGAGENT_SYSTEM_PROMPT = (
    "You are playing a game from LMGame-Bench (GamingAgent). "
    "You receive the current game state in text form and must choose one action per turn.\n\n"
    "Read the state description and the list of valid actions. Reply with exactly one valid action. "
    "Use the exact action name from the valid actions list."
)

GAMINGAGENT_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action do you take? Reply with exactly one action from the valid actions list."
)


# ---------------------------------------------------------------------------
# VideoGameBench constants
# ---------------------------------------------------------------------------
VIDEOGAMEBENCH_VALID_ACTIONS = ("no-op", "A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN")

VIDEOGAMEBENCH_SYSTEM_PROMPT = (
    "You are playing a Game Boy game (VideoGameBench). "
    "You see the game screen and must choose one button action per step.\n\n"
    "Valid actions: no-op, A, B, SELECT, START, RIGHT, LEFT, UP, DOWN.\n"
    "Reply with exactly one action. Use the exact name (e.g. 'A', 'UP', 'no-op')."
)

VIDEOGAMEBENCH_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action do you take? Reply with exactly one: no-op, A, B, SELECT, START, RIGHT, LEFT, UP, or DOWN."
)


# ---------------------------------------------------------------------------
# VideoGameBench DOS constants (DOS games only, no Game Boy)
# ---------------------------------------------------------------------------
VIDEOGAMEBENCH_DOS_VALID_KEYS = (
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Space", "Enter", "Escape", "KeyW", "KeyA", "KeyS", "KeyD",
)

VIDEOGAMEBENCH_DOS_SYSTEM_PROMPT = (
    "You are playing a DOS video game (VideoGameBench). "
    "You see the game screen and must choose one keyboard key to press per step.\n\n"
    "Valid keys: ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Space, Enter, Escape, KeyW, KeyA, KeyS, KeyD.\n"
    "Reply with exactly one key name (e.g. ArrowUp, Space, KeyW)."
)

VIDEOGAMEBENCH_DOS_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What key do you press? Reply with exactly one: ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Space, Enter, Escape, KeyW, KeyA, KeyS, or KeyD."
)


# ---------------------------------------------------------------------------
# Orak constants (AIcrowd Game Agent Challenge 2025)
# ---------------------------------------------------------------------------
ORAK_STARCRAFT_SYSTEM_PROMPT = (
    "You are playing StarCraft II as Protoss against a Zerg AI opponent.\n"
    "You receive the current game state and must output exactly 5 actions per step.\n\n"
    "Action categories: TRAIN <unit>, BUILD <structure>, RESEARCH <upgrade>, "
    "SCOUTING <unit>, MULTI-ATTACK, MULTI-RETREAT, CHRONOBOOST <building>, EMPTY ACTION.\n\n"
    "Reply with 5 numbered actions:\n"
    "1: <ACTION>\n2: <ACTION>\n3: <ACTION>\n4: <ACTION>\n5: <ACTION>\n\n"
    "Use exact action names (e.g. TRAIN ZEALOT, BUILD PYLON, RESEARCH CHARGE)."
)

ORAK_STARCRAFT_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "Provide exactly 5 actions in the format:\n"
    "1: <ACTION>\n2: <ACTION>\n3: <ACTION>\n4: <ACTION>\n5: <ACTION>"
)

ORAK_MARIO_SYSTEM_PROMPT = (
    "You are playing Super Mario Bros. Mario auto-runs right; you control jump height.\n"
    "Choose a jump level from 0 (no jump) to 6 (highest jump).\n\n"
    "Reply in format: Jump Level : N  (where N is 0-6)"
)

ORAK_MARIO_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What jump level? Reply: Jump Level : N  (N from 0-6)"
)

ORAK_POKEMON_SYSTEM_PROMPT = (
    "You are playing Pokemon Red. Available tools:\n"
    "- Basic: up, down, left, right, a, b, start, select\n"
    "- Advanced: move_to(x,y), interact_with_object(name), warp_with_warp_point(x,y),\n"
    "  continue_dialog(), select_move_in_battle(move_name), switch_pkmn_in_battle(pokemon),\n"
    "  run_away(), use_item_in_battle(item)\n\n"
    "Reply with exactly one action or tool call."
)

ORAK_POKEMON_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action do you take?"
)

ORAK_2048_SYSTEM_PROMPT = (
    "You are playing 2048. Merge tiles by sliding in one direction.\n"
    "Reply with exactly one: up, down, left, or right."
)

ORAK_2048_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What direction? Reply: up, down, left, or right."
)

ORAK_STREET_FIGHTER_SYSTEM_PROMPT = (
    "You are playing Street Fighter III: 3rd Strike.\n"
    "Choose one meta-instruction per turn from your character's moveset.\n\n"
    "Common moves: Move Closer, Move Away, Fireball, Megapunch, Hurricane, "
    "Low Kick, Medium Kick, High Kick, Low Punch, Medium Punch, High Punch, "
    "Jump Closer, Jump Away, Crouch, Block.\n\n"
    "Reply with exactly one move name."
)

ORAK_STREET_FIGHTER_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What move do you choose? Reply with one move name."
)

ORAK_SLAY_THE_SPIRE_SYSTEM_PROMPT = (
    "You are playing Slay the Spire as a deck-building roguelike.\n"
    "Actions depend on the current screen:\n"
    "- Combat: PLAY <card_index> [target_index], or END to end turn.\n"
    "- Card reward: CHOOSE <card_index> or SKIP.\n\n"
    "Reply with one action per line."
)

ORAK_SLAY_THE_SPIRE_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action? Reply with: PLAY <idx> [target], END, CHOOSE <idx>, or SKIP."
)

ORAK_DARKEST_DUNGEON_SYSTEM_PROMPT = (
    "You are playing Darkest Dungeon. Your party of heroes raids a dungeon.\n"
    "Each turn in combat, choose one action:\n"
    "- attack target X using skill slot Y\n"
    "- heal target X using skill slot Y\n"
    "- swap rank X hero forward/backward by Y\n"
    "- idle (skip turn)\n\n"
    "Reply with exactly one action."
)

ORAK_DARKEST_DUNGEON_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action? Reply: attack/heal/swap/idle with target and skill details."
)

ORAK_PWAAT_SYSTEM_PROMPT = (
    "You are playing Phoenix Wright: Ace Attorney Trilogy.\n"
    "Navigate court scenes by choosing actions:\n"
    "- Ok (advance dialog), Back, Up, Down, Left, Right\n"
    "- Present evidence <index> during cross-examination\n"
    "- Press (press witness on testimony)\n"
    "- Choose option <number> for multiple choice\n\n"
    "Reply with exactly one action."
)

ORAK_PWAAT_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action do you take? Reply with one action."
)

ORAK_HER_STORY_SYSTEM_PROMPT = (
    "You are playing Her Story, a detective FMV game.\n"
    "Search keywords to find video clips, then play them to uncover the story.\n\n"
    "Actions:\n"
    "- Search <keyword> (search the database with a keyword)\n"
    "- Play Video <index> (watch a specific video from search results)\n\n"
    "Reply with exactly one action."
)

ORAK_HER_STORY_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What action? Reply: Search <keyword> or Play Video <index>."
)

ORAK_MINECRAFT_SYSTEM_PROMPT = (
    "You are playing Minecraft. You control a bot and must craft target items.\n"
    "Write a JavaScript async function that takes a `bot` parameter.\n"
    "Available APIs: exploreUntil, mineBlock, craftItem, placeItem, smeltItem, "
    "killMob, useChest, etc.\n\n"
    "Reply with a ```javascript code block containing your async function."
)

ORAK_MINECRAFT_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "Write a JavaScript async function to accomplish the task."
)

ORAK_STARDEW_VALLEY_SYSTEM_PROMPT = (
    "You are playing Stardew Valley. Complete farming tasks using skill expressions.\n"
    "Available skills: till_soil, plant_seeds, water_seeds, harvest_crops, "
    "sell_item, buy_item, get_out_of_house, go_house_and_sleep.\n\n"
    "Reply with a Python list of skill calls, e.g.:\n"
    "[\"till_soil\", \"plant_seeds\", \"water_seeds\"]"
)

ORAK_STARDEW_VALLEY_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What skills do you use? Reply with a Python list of skill calls."
)

ORAK_BABA_IS_YOU_SYSTEM_PROMPT = (
    "You are playing Baba Is You, a rule-manipulation puzzle game.\n"
    "Move BABA (or the controllable entity) to satisfy the WIN condition.\n"
    "Actions: idle, left, right, up, down (optionally with step count, e.g. 'up 3').\n\n"
    "Reply with a sequence of moves, one per line."
)

ORAK_BABA_IS_YOU_USER_TEMPLATE = (
    "Current game state:\n\n{state}\n\n"
    "What moves? Reply with directions (e.g. 'up', 'left 2', 'down')."
)

ORAK_STARCRAFT_MULTI_SYSTEM_PROMPT = ORAK_STARCRAFT_SYSTEM_PROMPT
ORAK_STARCRAFT_MULTI_USER_TEMPLATE = ORAK_STARCRAFT_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Game auto-detection
# ---------------------------------------------------------------------------

def detect_game(state_nl: str) -> str:
    """
    Auto-detect which game is being played from the observation text.

    Returns one of: GAME_OVERCOOKED, GAME_AVALON, GAME_DIPLOMACY, GAME_GAMINGAGENT, GAME_VIDEOGAMEBENCH.
    Falls back to GAME_OVERCOOKED if detection fails.
    """
    if not state_nl or not isinstance(state_nl, str):
        return GAME_OVERCOOKED

    text = state_nl.lower()

    # Orak StarCraft II markers
    if "starcraft" in text and ("protoss" in text or "zerg" in text or "minerals" in text):
        return GAME_ORAK_STARCRAFT
    if "game time" in text and ("supply" in text or "nexus" in text or "pylon" in text):
        return GAME_ORAK_STARCRAFT

    # Orak Super Mario markers (jump-level based)
    if "position of mario" in text or "jump level" in text:
        return GAME_ORAK_MARIO

    # Orak Pokemon Red markers (map-based observation)
    if "map name:" in text and "your position" in text and "warppoint" in text:
        return GAME_ORAK_POKEMON
    if "[full map]" in text and "[current party]" in text:
        return GAME_ORAK_POKEMON

    # Orak 2048 markers
    if "board of 2048" in text:
        return GAME_ORAK_2048

    # Orak Street Fighter III markers
    if "street fighter" in text or ("health" in text and "super bar" in text and "stun" in text):
        return GAME_ORAK_STREET_FIGHTER
    if "move closer" in text and "fireball" in text:
        return GAME_ORAK_STREET_FIGHTER

    # Orak Slay the Spire markers
    if "slay the spire" in text or ("floor" in text and "energy" in text and ("hand:" in text or "deck:" in text)):
        return GAME_ORAK_SLAY_THE_SPIRE
    if "ironclad" in text and ("block" in text or "strike" in text or "defend" in text):
        return GAME_ORAK_SLAY_THE_SPIRE

    # Orak Darkest Dungeon markers
    if "darkest dungeon" in text or ("rank" in text and "stress" in text and "skill slot" in text):
        return GAME_ORAK_DARKEST_DUNGEON
    if "raid" in text and ("hero" in text or "enemy formation" in text) and "stress" in text:
        return GAME_ORAK_DARKEST_DUNGEON

    # Orak Ace Attorney (pwaat) markers
    if "cross-examination" in text or "court record" in text or "testimony" in text:
        return GAME_ORAK_PWAAT
    if "evidence" in text and "profile" in text and ("present" in text or "press" in text):
        return GAME_ORAK_PWAAT

    # Orak Her Story markers
    if "her story" in text or ("search" in text and "play video" in text and "session" in text):
        return GAME_ORAK_HER_STORY

    # Orak Minecraft markers (Voyager-style)
    if "biome:" in text and "nearby blocks" in text and "inventory" in text:
        return GAME_ORAK_MINECRAFT

    # Orak Stardew Valley markers
    if "stardew" in text or ("tilled_soil" in text and ("plant_seeds" in text or "harvest_crops" in text)):
        return GAME_ORAK_STARDEW_VALLEY
    if "season:" in text and "energy:" in text and ("crops" in text or "toolbar" in text):
        return GAME_ORAK_STARDEW_VALLEY

    # Orak Baba Is You markers
    if "baba is you" in text or ("baba" in text and "is" in text and "you" in text and "level" in text):
        return GAME_ORAK_BABA_IS_YOU
    if "active rules" in text and "objects" in text and ("wall" in text or "flag" in text):
        return GAME_ORAK_BABA_IS_YOU

    # VideoGameBench DOS markers
    if "dos game screen" in text or "choose one key" in text:
        return GAME_VIDEOGAMEBENCH_DOS
    if "arrowup" in text or "arrowdown" in text:
        return GAME_VIDEOGAMEBENCH_DOS

    # VideoGameBench Game Boy markers (deprecated in evaluate_videogamebench)
    if "game screen" in text and ("no-op" in text or "choose one action" in text):
        return GAME_VIDEOGAMEBENCH
    if "videogamebench" in text and "dos" in text:
        return GAME_VIDEOGAMEBENCH_DOS

    # GamingAgent / LMGame-Bench markers (textual game state with valid actions)
    if "valid actions:" in text and any(
        m in text for m in ("push", "hard_drop", "board", "sokoban", "2048", "tetris")
    ):
        return GAME_GAMINGAGENT
    if "choose one action" in text and "step" in text and len(text) < 500:
        return GAME_VIDEOGAMEBENCH

    # Diplomacy markers
    if "diplomacy" in text or "supply centers" in text or "orderable" in text:
        return GAME_DIPLOMACY
    if any(p in text for p in ("a par", "f bre", "movement phase", "retreat phase", "adjustment phase")):
        return GAME_DIPLOMACY
    if any(p in text for p in ("austria", "england", "france", "germany", "italy", "russia", "turkey")):
        # Check it looks like Diplomacy context (power status lines)
        if "centers" in text and "units" in text:
            return GAME_DIPLOMACY

    # Avalon markers
    if "avalon" in text or "quest" in text or "assassination" in text:
        return GAME_AVALON
    if any(p in text for p in ("merlin", "percival", "morgana", "mordred", "assassin", "servant", "minion", "oberon")):
        return GAME_AVALON
    if "approve" in text and "reject" in text:
        return GAME_AVALON
    if "team selection" in text or "team voting" in text or "quest voting" in text:
        return GAME_AVALON

    # Overcooked markers (or fallback)
    if any(p in text for p in ("overcooked", "chef", "soup", "pot", "interact", "facing north",
                                "facing south", "facing east", "facing west")):
        return GAME_OVERCOOKED

    return GAME_OVERCOOKED


# ---------------------------------------------------------------------------
# Game-specific prompt selection
# ---------------------------------------------------------------------------

def _get_system_prompt(game: str) -> str:
    """Return the system prompt for the given game type."""
    if game == GAME_AVALON:
        return AVALON_SYSTEM_PROMPT
    if game == GAME_DIPLOMACY:
        return DIPLOMACY_SYSTEM_PROMPT
    if game == GAME_GAMINGAGENT:
        return GAMINGAGENT_SYSTEM_PROMPT
    if game == GAME_VIDEOGAMEBENCH:
        return VIDEOGAMEBENCH_SYSTEM_PROMPT
    if game == GAME_VIDEOGAMEBENCH_DOS:
        return VIDEOGAMEBENCH_DOS_SYSTEM_PROMPT
    if game == GAME_ORAK_STARCRAFT:
        return ORAK_STARCRAFT_SYSTEM_PROMPT
    if game == GAME_ORAK_MARIO:
        return ORAK_MARIO_SYSTEM_PROMPT
    if game == GAME_ORAK_POKEMON:
        return ORAK_POKEMON_SYSTEM_PROMPT
    if game == GAME_ORAK_2048:
        return ORAK_2048_SYSTEM_PROMPT
    if game == GAME_ORAK_STARCRAFT_MULTI:
        return ORAK_STARCRAFT_MULTI_SYSTEM_PROMPT
    if game == GAME_ORAK_STREET_FIGHTER:
        return ORAK_STREET_FIGHTER_SYSTEM_PROMPT
    if game == GAME_ORAK_SLAY_THE_SPIRE:
        return ORAK_SLAY_THE_SPIRE_SYSTEM_PROMPT
    if game == GAME_ORAK_DARKEST_DUNGEON:
        return ORAK_DARKEST_DUNGEON_SYSTEM_PROMPT
    if game == GAME_ORAK_PWAAT:
        return ORAK_PWAAT_SYSTEM_PROMPT
    if game == GAME_ORAK_HER_STORY:
        return ORAK_HER_STORY_SYSTEM_PROMPT
    if game == GAME_ORAK_MINECRAFT:
        return ORAK_MINECRAFT_SYSTEM_PROMPT
    if game == GAME_ORAK_STARDEW_VALLEY:
        return ORAK_STARDEW_VALLEY_SYSTEM_PROMPT
    if game == GAME_ORAK_BABA_IS_YOU:
        return ORAK_BABA_IS_YOU_SYSTEM_PROMPT
    if game == GAME_ORAK:
        return GAMINGAGENT_SYSTEM_PROMPT
    return OVERCOOKED_SYSTEM_PROMPT


def _get_user_prompt(state_nl: str, game: str) -> str:
    """Return the user prompt for the given game type, filled with state."""
    if game == GAME_AVALON:
        return AVALON_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_DIPLOMACY:
        return DIPLOMACY_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_GAMINGAGENT:
        return GAMINGAGENT_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_VIDEOGAMEBENCH:
        return VIDEOGAMEBENCH_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_VIDEOGAMEBENCH_DOS:
        return VIDEOGAMEBENCH_DOS_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_STARCRAFT:
        return ORAK_STARCRAFT_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_MARIO:
        return ORAK_MARIO_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_POKEMON:
        return ORAK_POKEMON_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_2048:
        return ORAK_2048_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_STARCRAFT_MULTI:
        return ORAK_STARCRAFT_MULTI_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_STREET_FIGHTER:
        return ORAK_STREET_FIGHTER_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_SLAY_THE_SPIRE:
        return ORAK_SLAY_THE_SPIRE_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_DARKEST_DUNGEON:
        return ORAK_DARKEST_DUNGEON_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_PWAAT:
        return ORAK_PWAAT_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_HER_STORY:
        return ORAK_HER_STORY_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_MINECRAFT:
        return ORAK_MINECRAFT_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_STARDEW_VALLEY:
        return ORAK_STARDEW_VALLEY_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK_BABA_IS_YOU:
        return ORAK_BABA_IS_YOU_USER_TEMPLATE.format(state=state_nl)
    if game == GAME_ORAK:
        return GAMINGAGENT_USER_TEMPLATE.format(state=state_nl)
    return OVERCOOKED_USER_TEMPLATE.format(state=state_nl)


# ---------------------------------------------------------------------------
# Action extraction: Overcooked
# ---------------------------------------------------------------------------

def _extract_overcooked_action(text: str) -> Optional[str]:
    """Extract one valid Overcooked action from model reply (first matching word)."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip().lower()
    words = re.findall(r"[a-z]+", text)
    for w in words:
        if w in OVERCOOKED_VALID_ACTIONS:
            return w
        if w in ("up", "down", "left", "right"):
            return {"up": "north", "down": "south", "left": "west", "right": "east"}[w]
        if w in ("move", "go", "walk") and len(words) > 1:
            idx = words.index(w) + 1
            if idx < len(words) and words[idx] in OVERCOOKED_VALID_ACTIONS:
                return words[idx]
        if w == "interact" or w.startswith("interact"):
            return "interact"
        if w in ("wait", "hold", "nothing"):
            return "stay"
    return None


# ---------------------------------------------------------------------------
# Action extraction: Avalon
# ---------------------------------------------------------------------------

_APPROVE_WORDS = {"approve", "yes", "accept", "aye", "agree", "yea", "support"}
_REJECT_WORDS = {"reject", "no", "deny", "nay", "disagree", "oppose", "fail"}
_PASS_WORDS = {"pass", "success", "succeed"}
_FAIL_WORDS = {"fail", "sabotage"}
_WAIT_WORDS = {"wait", "waiting", "silent", "nothing"}


def _detect_avalon_phase(state_nl: str) -> str:
    """
    Detect the Avalon phase from the observation text.

    Returns one of: "team_selection", "team_voting", "quest_voting", "assassination", "wait".
    """
    text = state_nl.lower()
    if "team selection" in text:
        if "you are the quest leader" in text:
            return "team_selection"
        return "wait"
    if "team voting" in text:
        if "vote to approve or reject" in text:
            return "team_voting"
        return "wait"
    if "quest voting" in text:
        if "you are on the quest team" in text:
            return "quest_voting"
        return "wait"
    if "assassination" in text:
        if "you are the assassin" in text:
            return "assassination"
        return "wait"
    return "wait"


def _extract_avalon_action(text: str, state_nl: str) -> Union[str, List[int], int, None]:
    """
    Extract an Avalon action from model reply, based on the detected phase.

    Returns:
      - team_selection: comma-separated player IDs as a string, e.g. "0, 2, 3"
      - team_voting:    "approve" or "reject"
      - quest_voting:   "pass" or "fail"
      - assassination:  player ID as a string, e.g. "2"
      - wait:           "wait"
    """
    if not text or not isinstance(text, str):
        return "wait"

    phase = _detect_avalon_phase(state_nl)

    if phase == "wait":
        return "wait"

    reply = text.strip().lower()
    words = re.findall(r"[a-z]+", reply)

    if phase == "team_selection":
        # Extract player IDs (digits) from the reply
        ids = re.findall(r"\d+", text)
        if ids:
            return ", ".join(ids)
        return "0"

    if phase == "team_voting":
        for w in words:
            if w in _APPROVE_WORDS:
                return "approve"
            if w in _REJECT_WORDS:
                return "reject"
        return "approve"

    if phase == "quest_voting":
        for w in words:
            if w in _PASS_WORDS:
                return "pass"
            if w in _FAIL_WORDS:
                return "fail"
        return "pass"

    if phase == "assassination":
        ids = re.findall(r"\d+", text)
        if ids:
            return ids[0]
        return "0"

    return "wait"


# ---------------------------------------------------------------------------
# Action extraction: Diplomacy
# ---------------------------------------------------------------------------

def _parse_valid_actions_from_state(state_nl: str) -> List[str]:
    """Extract valid action names from 'Valid actions: a, b, c' in state_nl."""
    m = re.search(r"[Vv]alid\s+actions?\s*[:\-]\s*(.+?)(?:\n|\.|$)", state_nl)
    if not m:
        return []
    raw = m.group(1).strip()
    actions = [a.strip().lower() for a in re.split(r"[,;]", raw) if a.strip()]
    return actions


def _extract_gamingagent_action(text: str, state_nl: str) -> Optional[str]:
    """Extract one valid GamingAgent action from model reply."""
    valid = _parse_valid_actions_from_state(state_nl)
    if not text or not isinstance(text, str):
        return valid[0] if valid else None
    reply = text.strip().lower()
    words = re.findall(r"[\w_]+", reply)
    for w in words:
        wl = w.lower()
        for v in valid:
            if wl == v.lower() or (len(wl) >= 2 and v.lower().startswith(wl)):
                return v
    if valid:
        return valid[0]
    return None


def _extract_videogamebench_action(text: str) -> Optional[str]:
    """Extract VideoGameBench action (no-op, A, B, etc.) from model reply."""
    if not text or not isinstance(text, str):
        return "no-op"
    reply = text.strip().lower()
    words = re.findall(r"[a-z_\-]+", reply)
    for w in words:
        if w in VIDEOGAMEBENCH_VALID_ACTIONS:
            return w
        if w in ("noop", "none", "nothing", "stay", "wait"):
            return "no-op"
        if w in ("a", "b"):
            return w.upper()
        if w in ("select", "start"):
            return w.upper()
        if w in ("right", "left", "up", "down"):
            return w.upper()
    return "no-op"


def _extract_videogamebench_dos_action(text: str) -> Optional[str]:
    """Extract VideoGameBench DOS key (ArrowUp, Space, etc.) from model reply."""
    if not text or not isinstance(text, str):
        return "Space"
    s = text.strip()
    if s in VIDEOGAMEBENCH_DOS_VALID_KEYS:
        return s
    lower = s.lower()
    key_map = {
        "up": "ArrowUp", "arrowup": "ArrowUp", "↑": "ArrowUp",
        "down": "ArrowDown", "arrowdown": "ArrowDown", "↓": "ArrowDown",
        "left": "ArrowLeft", "arrowleft": "ArrowLeft", "←": "ArrowLeft",
        "right": "ArrowRight", "arrowright": "ArrowRight", "→": "ArrowRight",
        "space": "Space", "spacebar": "Space",
        "enter": "Enter", "return": "Enter",
        "escape": "Escape", "esc": "Escape",
        "w": "KeyW", "keyw": "KeyW",
        "a": "KeyA", "keya": "KeyA",
        "s": "KeyS", "keys": "KeyS",
        "d": "KeyD", "keyd": "KeyD",
    }
    for k, v in key_map.items():
        if k in lower or lower == k:
            return v
    words = re.findall(r"[a-z]+", lower)
    for w in words:
        if w in key_map:
            return key_map[w]
    return "Space"


def _extract_orak_starcraft_action(text: str) -> Optional[str]:
    """Extract StarCraft II multi-action string (5 numbered actions)."""
    if not text or not isinstance(text, str):
        return "1: TRAIN PROBE\n2: BUILD PYLON\n3: EMPTY ACTION\n4: EMPTY ACTION\n5: EMPTY ACTION"
    pattern = r"\d+:\s*<?([^>\n]+)>?"
    matches = re.findall(pattern, text)
    if matches:
        actions = [m.strip() for m in matches[:5]]
        while len(actions) < 5:
            actions.append("EMPTY ACTION")
        return "\n".join(f"{i+1}: {a}" for i, a in enumerate(actions))
    return text.strip()


def _extract_orak_mario_action(text: str) -> Optional[str]:
    """Extract Super Mario jump level."""
    if not text:
        return "Jump Level : 0"
    m = re.search(r"[Jj]ump\s*[Ll]evel\s*[:\-]\s*(\d)", text)
    if m:
        return f"Jump Level : {m.group(1)}"
    digits = re.findall(r"\d", text)
    for d in digits:
        if d in "0123456":
            return f"Jump Level : {d}"
    return "Jump Level : 0"


def _extract_orak_pokemon_action(text: str) -> Optional[str]:
    """Extract Pokemon Red action or tool call."""
    if not text:
        return "a"
    stripped = text.strip()
    basic = ["up", "down", "left", "right", "a", "b", "start", "select"]
    for b in basic:
        if stripped.lower() == b:
            return b
    if "use_tool" in stripped or "move_to" in stripped or "interact_with" in stripped:
        return stripped
    if "continue_dialog" in stripped:
        return stripped
    for b in basic:
        if b in stripped.lower():
            return b
    return stripped


def _extract_orak_2048_action(text: str) -> Optional[str]:
    """Extract 2048 direction."""
    if not text:
        return "up"
    lower = text.strip().lower()
    for d in ["up", "down", "left", "right"]:
        if d in lower:
            return d
    return "up"


def _extract_orak_street_fighter_action(text: str) -> Optional[str]:
    """Extract Street Fighter move name."""
    if not text:
        return "Move Closer"
    moves = [
        "Move Closer", "Move Away", "Fireball", "Megapunch", "Hurricane",
        "Low Kick", "Medium Kick", "High Kick", "Jump Closer", "Jump Away",
        "Crouch", "Block", "Low Punch", "Medium Punch", "High Punch",
    ]
    lower = text.strip().lower()
    for m in moves:
        if m.lower() in lower:
            return m
    return text.strip() if text.strip() else "Move Closer"


def _extract_orak_slay_the_spire_action(text: str) -> Optional[str]:
    """Extract Slay the Spire action (PLAY/END/CHOOSE/SKIP)."""
    if not text:
        return "END"
    stripped = text.strip()
    if re.match(r"(?i)play\s+\d+", stripped):
        return stripped.upper()
    if re.match(r"(?i)end", stripped):
        return "END"
    if re.match(r"(?i)choose\s+\d+", stripped):
        return stripped.upper()
    if re.match(r"(?i)skip", stripped):
        return "SKIP"
    m = re.search(r"(PLAY\s+\d+(?:\s+\d+)?|END|CHOOSE\s+\d+|SKIP)", stripped, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stripped


def _extract_orak_darkest_dungeon_action(text: str) -> Optional[str]:
    """Extract Darkest Dungeon combat action."""
    if not text:
        return "idle"
    lower = text.strip().lower()
    if "idle" in lower or "skip" in lower or "pass" in lower:
        return "idle"
    if re.search(r"attack\s+target\s+\d+\s+using\s+skill\s+slot\s+\d+", lower):
        return text.strip()
    if re.search(r"heal\s+target\s+\d+\s+using\s+skill\s+slot\s+\d+", lower):
        return text.strip()
    if re.search(r"swap\s+rank\s+\d+", lower):
        return text.strip()
    if "attack" in lower:
        return text.strip()
    if "heal" in lower:
        return text.strip()
    return text.strip() if text.strip() else "idle"


def _extract_orak_pwaat_action(text: str) -> Optional[str]:
    """Extract Ace Attorney action (Ok/Back/Present/Press/option index)."""
    if not text:
        return "Ok"
    stripped = text.strip()
    lower = stripped.lower()
    actions = ["ok", "back", "down", "up", "left", "right", "press"]
    for a in actions:
        if lower == a:
            return a.capitalize() if a != "ok" else "Ok"
    if "present" in lower:
        return stripped
    m = re.search(r"(\d+)", stripped)
    if m:
        return m.group(1)
    return stripped if stripped else "Ok"


def _extract_orak_her_story_action(text: str) -> Optional[str]:
    """Extract Her Story action (Search/Play Video)."""
    if not text:
        return "Search murder"
    stripped = text.strip()
    if re.match(r"(?i)search\s+.+", stripped):
        return stripped
    if re.match(r"(?i)play\s+video\s+\d+", stripped):
        return stripped
    m = re.search(r"(?i)(search\s+\S+|play\s+video\s+\d+)", stripped)
    if m:
        return m.group(1)
    return f"Search {stripped}" if stripped else "Search murder"


def _extract_orak_minecraft_action(text: str) -> Optional[str]:
    """Extract Minecraft JavaScript code action."""
    if not text:
        return 'async function action(bot) { await bot.chat("hello"); }'
    m = re.search(r"```(?:javascript|js)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_orak_stardew_valley_action(text: str) -> Optional[str]:
    """Extract Stardew Valley skill list."""
    if not text:
        return '["till_soil"]'
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


def _extract_orak_baba_is_you_action(text: str) -> Optional[str]:
    """Extract Baba Is You movement actions."""
    if not text:
        return "idle"
    lower = text.strip().lower()
    valid = ["idle", "left", "right", "up", "down"]
    for v in valid:
        if lower == v or lower.startswith(v):
            return lower
    m = re.findall(r"(idle|left|right|up|down)(?:\s+(\d+))?", lower)
    if m:
        parts = []
        for direction, count in m:
            parts.append(f"{direction} {count}" if count else direction)
        return "\n".join(parts)
    return text.strip() if text.strip() else "idle"


def _extract_diplomacy_orders(text: str) -> List[str]:
    """
    Extract Diplomacy order strings from model reply.

    Tries JSON parsing first, then falls back to line/semicolon splitting.
    Returns a list of order strings.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()

    # Try to extract a JSON list from the reply
    json_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [str(o).strip() for o in parsed if o and str(o).strip()]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: split by newlines, semicolons, or commas (careful with order syntax)
    # Orders look like: "A PAR - BUR", "F BRE H", etc.
    lines = re.split(r"[;\n]+", text)
    orders = []
    order_pattern = re.compile(
        r"[AF]\s+[A-Z]{3}\s+(?:[-HSCRBD]|S\s+[AF]\s+[A-Z]{3}|C\s+[AF]\s+[A-Z]{3})",
        re.IGNORECASE,
    )
    for line in lines:
        line = line.strip().strip(",").strip("'\"").strip()
        if not line:
            continue
        # Check if line looks like a valid order
        if order_pattern.search(line.upper()):
            orders.append(line.strip())

    return orders


# ---------------------------------------------------------------------------
# Unified action extraction
# ---------------------------------------------------------------------------

def extract_action(text: str, game: str, state_nl: str = "") -> Union[str, List[str], None]:
    """
    Extract an action from model reply text for the given game.

    Args:
        text: Raw LLM reply text.
        game: One of GAME_OVERCOOKED, GAME_AVALON, GAME_DIPLOMACY, GAME_GAMINGAGENT, GAME_VIDEOGAMEBENCH.
        state_nl: Original state observation (needed for Avalon phase detection, GamingAgent valid actions).

    Returns:
        The extracted action in the format expected by the corresponding env wrapper.
    """
    if game == GAME_AVALON:
        return _extract_avalon_action(text, state_nl)
    if game == GAME_DIPLOMACY:
        return _extract_diplomacy_orders(text)
    if game == GAME_GAMINGAGENT:
        return _extract_gamingagent_action(text, state_nl)
    if game == GAME_VIDEOGAMEBENCH:
        return _extract_videogamebench_action(text)
    if game == GAME_VIDEOGAMEBENCH_DOS:
        return _extract_videogamebench_dos_action(text)
    if game == GAME_ORAK_STARCRAFT:
        return _extract_orak_starcraft_action(text)
    if game == GAME_ORAK_MARIO:
        return _extract_orak_mario_action(text)
    if game == GAME_ORAK_POKEMON:
        return _extract_orak_pokemon_action(text)
    if game == GAME_ORAK_2048:
        return _extract_orak_2048_action(text)
    if game == GAME_ORAK_STARCRAFT_MULTI:
        return _extract_orak_starcraft_action(text)
    if game == GAME_ORAK_STREET_FIGHTER:
        return _extract_orak_street_fighter_action(text)
    if game == GAME_ORAK_SLAY_THE_SPIRE:
        return _extract_orak_slay_the_spire_action(text)
    if game == GAME_ORAK_DARKEST_DUNGEON:
        return _extract_orak_darkest_dungeon_action(text)
    if game == GAME_ORAK_PWAAT:
        return _extract_orak_pwaat_action(text)
    if game == GAME_ORAK_HER_STORY:
        return _extract_orak_her_story_action(text)
    if game == GAME_ORAK_MINECRAFT:
        return _extract_orak_minecraft_action(text)
    if game == GAME_ORAK_STARDEW_VALLEY:
        return _extract_orak_stardew_valley_action(text)
    if game == GAME_ORAK_BABA_IS_YOU:
        return _extract_orak_baba_is_you_action(text)
    if game == GAME_ORAK:
        return _extract_gamingagent_action(text, state_nl)
    return _extract_overcooked_action(text)


def _default_action(game: str, state_nl: str = "") -> Union[str, List[str]]:
    """Return a safe default/fallback action for the given game."""
    if game == GAME_AVALON:
        return "approve"
    if game == GAME_DIPLOMACY:
        return []
    if game == GAME_GAMINGAGENT:
        valid = _parse_valid_actions_from_state(state_nl) if state_nl else []
        return valid[0] if valid else "up"
    if game == GAME_VIDEOGAMEBENCH:
        return "no-op"
    if game == GAME_VIDEOGAMEBENCH_DOS:
        return "Space"
    if game == GAME_ORAK_STARCRAFT:
        return "1: TRAIN PROBE\n2: BUILD PYLON\n3: EMPTY ACTION\n4: EMPTY ACTION\n5: EMPTY ACTION"
    if game == GAME_ORAK_MARIO:
        return "Jump Level : 3"
    if game == GAME_ORAK_POKEMON:
        return "a"
    if game == GAME_ORAK_2048:
        return "down"
    if game == GAME_ORAK_STARCRAFT_MULTI:
        return "1: TRAIN PROBE\n2: BUILD PYLON\n3: EMPTY ACTION\n4: EMPTY ACTION\n5: EMPTY ACTION"
    if game == GAME_ORAK_STREET_FIGHTER:
        return "Move Closer"
    if game == GAME_ORAK_SLAY_THE_SPIRE:
        return "END"
    if game == GAME_ORAK_DARKEST_DUNGEON:
        return "idle"
    if game == GAME_ORAK_PWAAT:
        return "Ok"
    if game == GAME_ORAK_HER_STORY:
        return "Search murder"
    if game == GAME_ORAK_MINECRAFT:
        return 'async function action(bot) { await bot.chat("hello"); }'
    if game == GAME_ORAK_STARDEW_VALLEY:
        return '["till_soil"]'
    if game == GAME_ORAK_BABA_IS_YOU:
        return "right"
    if game == GAME_ORAK:
        return "up"
    return "stay"


# ---------------------------------------------------------------------------
# LLM call: ask_model
# ---------------------------------------------------------------------------

def ask_model_action(
    state_nl: str,
    game: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 200,
) -> Union[str, List[str]]:
    """
    Ask the LLM (via ask_model) for an action given the state in natural language.

    Args:
        state_nl: State description string from the env wrapper observation.
        game: Game type (auto-detected if None).
        model: Model name for ask_model (e.g. "gpt-4o", "gpt-4o-mini"). None = default.
        temperature: Lower for more deterministic actions.
        max_tokens: Max reply tokens.

    Returns:
        Action in the format expected by the corresponding env wrapper:
        - Overcooked: one of "north", "south", "east", "west", "stay", "interact"
        - Avalon: "approve"/"reject", "pass"/"fail", team string, or target string
        - Diplomacy: list of order strings
    """
    if game is None:
        game = detect_game(state_nl)

    prompt = _get_user_prompt(state_nl, game)
    if ask_model is None:
        return _default_action(game, state_nl)

    reply = ask_model(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    action = extract_action(reply, game, state_nl)
    return action if action is not None else _default_action(game, state_nl)


# ---------------------------------------------------------------------------
# LLM call: OpenAI GPT with function/tool calling
# ---------------------------------------------------------------------------

def _build_overcooked_tools() -> list:
    """Build OpenAI function-calling tool definition for Overcooked."""
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the single action for your chef in Overcooked this turn.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(OVERCOOKED_VALID_ACTIONS),
                            "description": "One of: north, south, east, west, stay, interact",
                        }
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_avalon_tools() -> list:
    """Build OpenAI function-calling tool definition for Avalon."""
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": (
                    "Choose your action in Avalon based on the current phase. "
                    "For team selection: comma-separated player IDs (e.g. '0, 2, 3'). "
                    "For team voting: 'approve' or 'reject'. "
                    "For quest voting: 'pass' or 'fail'. "
                    "For assassination: a player ID (e.g. '2'). "
                    "If not your turn: 'wait'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": (
                                "Your action: team IDs (e.g. '0, 2, 3'), "
                                "'approve'/'reject', 'pass'/'fail', target ID, or 'wait'"
                            ),
                        }
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_gamingagent_tools(state_nl: str = "") -> list:
    """Build OpenAI function-calling tool for GamingAgent (dynamic valid actions)."""
    valid = _parse_valid_actions_from_state(state_nl)
    enum_actions = valid if valid else ["stay", "up", "down", "left", "right"]
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the action for your turn in the game.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": f"One of the valid actions. Valid: {', '.join(enum_actions)}",
                        }
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_videogamebench_tools() -> list:
    """Build OpenAI function-calling tool for VideoGameBench (Game Boy)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose a button action for the Game Boy game.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(VIDEOGAMEBENCH_VALID_ACTIONS),
                            "description": "One of: no-op, A, B, SELECT, START, RIGHT, LEFT, UP, DOWN",
                        }
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_videogamebench_dos_tools() -> list:
    """Build OpenAI function-calling tool for VideoGameBench DOS games."""
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose a keyboard key for the DOS game.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(VIDEOGAMEBENCH_DOS_VALID_KEYS),
                            "description": "One of: ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Space, Enter, Escape, KeyW, KeyA, KeyS, KeyD",
                        }
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _build_diplomacy_tools() -> list:
    """Build OpenAI function-calling tool definition for Diplomacy."""
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_orders",
                "description": (
                    "Submit your Diplomacy orders for this phase. "
                    "Each order is a string like 'A PAR - BUR' or 'F BRE H'. "
                    "Submit an empty list if you have no orders."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "orders": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of order strings, e.g. ['A PAR - BUR', 'F BRE H']",
                        }
                    },
                    "required": ["orders"],
                },
            },
        }
    ]


def ask_gpt_function_action(
    state_nl: str,
    game: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> Union[str, List[str]]:
    """
    Use OpenAI GPT with function/tool calling to get a structured action.

    Requires openai and api_keys. Returns action appropriate for the game.
    """
    if game is None:
        game = detect_game(state_nl)

    use_router = open_router_api_key and open_router_api_key.strip()
    if openai is None or (not use_router and openai_api_key is None):
        return ask_model_action(state_nl, game=game, model=model, temperature=temperature)

    client_kw = {}
    if use_router:
        client_kw = {"base_url": OPENROUTER_BASE, "api_key": open_router_api_key.strip()}
        openrouter_model = model if "/" in model else f"openai/{model}"
        model = openrouter_model
    else:
        openai.api_key = openai_api_key

    system_prompt = _get_system_prompt(game)
    user_content = _get_user_prompt(state_nl, game)

    # Select tools and function name based on game
    if game == GAME_DIPLOMACY:
        tools = _build_diplomacy_tools()
        func_name = "submit_orders"
        result_key = "orders"
    elif game == GAME_AVALON:
        tools = _build_avalon_tools()
        func_name = "choose_action"
        result_key = "action"
    elif game == GAME_GAMINGAGENT:
        tools = _build_gamingagent_tools(state_nl)
        func_name = "choose_action"
        result_key = "action"
    elif game == GAME_VIDEOGAMEBENCH:
        tools = _build_videogamebench_tools()
        func_name = "choose_action"
        result_key = "action"
    elif game == GAME_VIDEOGAMEBENCH_DOS:
        tools = _build_videogamebench_dos_tools()
        func_name = "choose_action"
        result_key = "action"
    else:
        tools = _build_overcooked_tools()
        func_name = "choose_action"
        result_key = "action"

    try:
        if client_kw:
            client = openai.OpenAI(**client_kw)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": func_name}},
                temperature=temperature,
                max_tokens=300,
            )
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": func_name}},
                temperature=temperature,
                max_tokens=300,
            )
        msg = response.choices[0].message

        if msg.tool_calls and len(msg.tool_calls) > 0:
            tc = msg.tool_calls[0]
            args = getattr(tc, "arguments", None) or getattr(tc.function, "arguments", None) or "{}"
            if isinstance(args, str):
                args = json.loads(args)

            if game == GAME_DIPLOMACY:
                orders = (args or {}).get(result_key, [])
                if isinstance(orders, list):
                    return [str(o) for o in orders if o]
                return _extract_diplomacy_orders(str(orders))

            if game == GAME_OVERCOOKED:
                action = (args or {}).get(result_key, "")
                if action in OVERCOOKED_VALID_ACTIONS:
                    return action

            if game == GAME_GAMINGAGENT:
                action = (args or {}).get(result_key, "")
                if action:
                    return str(action)

            if game == GAME_VIDEOGAMEBENCH:
                action = (args or {}).get(result_key, "no-op")
                if action in VIDEOGAMEBENCH_VALID_ACTIONS:
                    return action
                return "no-op"

            if game == GAME_VIDEOGAMEBENCH_DOS:
                action = (args or {}).get(result_key, "Space")
                if action in VIDEOGAMEBENCH_DOS_VALID_KEYS:
                    return action
                return "Space"

            if game == GAME_AVALON:
                action = (args or {}).get(result_key, "")
                if action:
                    return str(action)

        # Fallback: parse content
        action = extract_action(msg.content or "", game, state_nl)
        return action if action is not None else _default_action(game, state_nl)

    except Exception:
        return ask_model_action(state_nl, game=game, model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def language_agent_action(
    state_nl: str,
    obs=None,
    game: Optional[str] = None,
    model: Optional[str] = None,
    use_function_call: bool = True,
    temperature: float = 0.3,
) -> Union[str, List[str]]:
    """
    Universal language agent: takes state (natural language) and returns action (natural language).

    Compatible with all three env_wrappers:
      - OvercookedNLWrapper: returns one of "north"/"south"/"east"/"west"/"stay"/"interact"
      - AvalonNLWrapper:     returns "approve"/"reject", "pass"/"fail", team IDs, or target ID
      - DiplomacyNLWrapper:  returns a list of order strings

    Args:
        state_nl: Current state as natural language (from wrapper observation).
        obs: Unused; for compatibility with agent_fn(state, obs=obs) signature.
        game: Game type: "overcooked", "avalon", or "diplomacy".
            If None, auto-detected from the observation text.
        model: Model name (e.g. "gpt-4o", "gpt-4o-mini"). None = default in ask_model.
        use_function_call: If True and OpenAI available, use GPT function calling.
        temperature: Sampling temperature.

    Returns:
        Action in the format expected by the corresponding env wrapper.
    """
    if game is None:
        game = detect_game(state_nl)

    if use_function_call and openai is not None and ((open_router_api_key and open_router_api_key.strip()) or openai_api_key is not None):
        return ask_gpt_function_action(
            state_nl,
            game=game,
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )
    return ask_model_action(
        state_nl,
        game=game,
        model=model,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Buffer management class
# ---------------------------------------------------------------------------

class AgentBufferManager:
    """
    Helper class to manage experience and episode buffers for the dummy agent.
    
    This class maintains both Experience_Replay_Buffer and Episode_Buffer,
    making it easy to collect and manage experiences across multiple episodes.
    """
    
    def __init__(
        self,
        experience_buffer_size: int = 10000,
        episode_buffer_size: int = 1000,
    ):
        """
        Initialize buffer manager with specified buffer sizes.
        
        Args:
            experience_buffer_size: Maximum number of experiences to store.
            episode_buffer_size: Maximum number of episodes to store.
        """
        if Experience_Replay_Buffer is None or Episode_Buffer is None:
            raise ImportError(
                "Experience_Replay_Buffer and Episode_Buffer classes must be imported. "
                "Make sure data_structure.experience is available."
            )
        
        self.experience_buffer = Experience_Replay_Buffer(experience_buffer_size)
        self.episode_buffer = Episode_Buffer(episode_buffer_size)
    
    def run_episode(
        self,
        env,
        task: str = "",
        game: Optional[str] = None,
        model: Optional[str] = None,
        use_function_call: bool = True,
        temperature: float = 0.3,
        max_steps: int = 1000,
        verbose: bool = False,
    ) -> Episode:
        """
        Run an episode and automatically add experiences/episodes to buffers.
        
        This is a convenience wrapper around run_episode_with_experience_collection
        that automatically uses this manager's buffers.
        
        Args:
            env: Environment instance.
            task: Task description for the episode.
            game: Game type (auto-detected if None).
            model: Model name for the agent.
            use_function_call: Whether to use GPT function calling.
            temperature: Sampling temperature.
            max_steps: Maximum steps per episode.
            verbose: Whether to print progress.
        
        Returns:
            Episode object containing all collected experiences.
        """
        return run_episode_with_experience_collection(
            env=env,
            task=task,
            game=game,
            model=model,
            use_function_call=use_function_call,
            temperature=temperature,
            max_steps=max_steps,
            verbose=verbose,
            experience_buffer=self.experience_buffer,
            episode_buffer=self.episode_buffer,
        )
    
    def get_experience_buffer(self) -> Experience_Replay_Buffer:
        """Get the experience replay buffer."""
        return self.experience_buffer
    
    def get_episode_buffer(self) -> Episode_Buffer:
        """Get the episode buffer."""
        return self.episode_buffer
    
    def sample_experiences(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from the experience buffer."""
        return self.experience_buffer.sample_experience(batch_size)
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """Sample a batch of episodes from the episode buffer."""
        return self.episode_buffer.sample_episode(batch_size)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffers."""
        return {
            "experience_buffer_size": len(self.experience_buffer),
            "experience_buffer_capacity": self.experience_buffer.buffer_size,
            "episode_buffer_size": len(self.episode_buffer),
            "episode_buffer_capacity": self.episode_buffer.buffer_size,
        }
    
    def save_episode_buffer(self, filepath: str):
        """
        Save the episode buffer to a JSON file.
        
        Args:
            filepath: Path to the JSON file where the episode buffer will be saved.
        """
        self.episode_buffer.save_to_json(filepath)
    
    @classmethod
    def load_from_json(
        cls,
        filepath: str,
        experience_buffer_size: int = 10000,
        episode_buffer_size: Optional[int] = None,
    ):
        """
        Create an AgentBufferManager by loading an episode buffer from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing the episode buffer.
            experience_buffer_size: Size for the new experience buffer (default: 10000).
            episode_buffer_size: Size for the episode buffer. If None, uses size from file or defaults to 1000.
        
        Returns:
            AgentBufferManager instance with loaded episode buffer.
        """
        if Episode_Buffer is None:
            raise ImportError(
                "Episode_Buffer class must be imported. "
                "Make sure data_structure.experience is available."
            )
        
        # Load episode buffer from file
        episode_buffer = Episode_Buffer.load_from_json(filepath, buffer_size=episode_buffer_size)
        
        # Create manager with loaded buffer
        manager = cls(
            experience_buffer_size=experience_buffer_size,
            episode_buffer_size=episode_buffer.buffer_size,
        )
        manager.episode_buffer = episode_buffer
        
        return manager


# ---------------------------------------------------------------------------
# Experience collection wrapper
# ---------------------------------------------------------------------------

def run_episode_with_experience_collection(
    env,
    task: str = "",
    game: Optional[str] = None,
    model: Optional[str] = None,
    use_function_call: bool = True,
    temperature: float = 0.3,
    max_steps: int = 1000,
    verbose: bool = False,
    experience_buffer: Optional[Experience_Replay_Buffer] = None,
    episode_buffer: Optional[Episode_Buffer] = None,
) -> Episode:
    """
    Run a full episode using the language agent and collect all experiences in an Episode.
    
    This function wraps the environment interaction loop and creates Experience objects
    for each step, storing them all in an Episode. Optionally maintains experience and
    episode buffers for experience replay.
    
    Args:
        env: Environment instance (must have reset() and step() methods).
        task: Task description for the episode (optional).
        game: Game type: "overcooked", "avalon", or "diplomacy". Auto-detected if None.
        model: Model name for the agent (e.g. "gpt-4o-mini").
        use_function_call: Whether to use GPT function calling if available.
        temperature: Sampling temperature for the agent.
        max_steps: Maximum number of steps before truncating the episode.
        verbose: Whether to print progress information.
        experience_buffer: Optional Experience_Replay_Buffer to add experiences to.
        episode_buffer: Optional Episode_Buffer to add the completed episode to.
    
    Returns:
        Episode object containing all collected Experience objects.
    
    Raises:
        ImportError: If Experience or Episode classes are not available.
    """
    if Experience is None or Episode is None:
        raise ImportError(
            "Experience and Episode classes must be imported from data_structure.experience. "
            "Make sure the module is available."
        )
    
    # Reset environment
    obs, info = env.reset()
    experiences: List[Experience] = []
    step_count = 0
    done = False
    
    # Determine if multi-agent (obs is dict) or single-agent (obs is string)
    is_multi_agent = isinstance(obs, dict)
    
    # Get initial state
    if is_multi_agent:
        # For multi-agent, use the first observation or combine them
        initial_state = str(obs) if obs else ""
    else:
        initial_state = str(obs) if obs else ""
    
    if game is None:
        game = detect_game(initial_state)
    
    if verbose:
        print(f"Starting episode with game: {game}, task: {task}")
    
    prev_state = initial_state
    
    while not done and step_count < max_steps:
        # Get action(s) from agent
        if is_multi_agent:
            # Multi-agent: get actions for all active players
            actions: Dict[Any, Any] = {}
            active_players = info.get("active_players", list(obs.keys()) if isinstance(obs, dict) else [])
            
            for player_id in active_players:
                player_obs = obs.get(player_id, "") if isinstance(obs, dict) else str(obs)
                if player_obs:
                    action = language_agent_action(
                        player_obs,
                        game=game,
                        model=model,
                        use_function_call=use_function_call,
                        temperature=temperature,
                    )
                    actions[player_id] = action
        else:
            # Single-agent: get one action
            action = language_agent_action(
                prev_state,
                game=game,
                model=model,
                use_function_call=use_function_call,
                temperature=temperature,
            )
            actions = action
        
        # Execute action in environment
        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated
        
        # Convert observations and rewards to strings/numbers for Experience
        state_str = prev_state
        action_str = str(actions) if not isinstance(actions, (str, list)) else actions
        reward_val = reward
        if isinstance(reward, dict):
            # For multi-agent, sum rewards or use a representative value
            reward_val = sum(reward.values()) if reward else 0.0
        
        if is_multi_agent:
            next_state_str = str(next_obs) if isinstance(next_obs, dict) else str(next_obs)
        else:
            next_state_str = str(next_obs) if next_obs else ""
        
        # Create Experience object
        experience = Experience(
            state=state_str,
            action=action_str,
            reward=float(reward_val) if reward_val is not None else 0.0,
            next_state=next_state_str,
            done=bool(done),
            intentions=None,  # Can be filled later
            tasks=task if task else None,
            sub_tasks=None,  # Can be filled later
        )
        experience.idx = step_count
        experiences.append(experience)
        
        if verbose:
            print(f"Step {step_count}: action={action_str}, reward={reward_val}, done={done}")
        
        # Update for next iteration
        obs = next_obs
        info = next_info
        prev_state = next_state_str
        step_count += 1
        
        if done:
            break
    
    # Create Episode with all experiences, populating identifiers from wrapper info.
    episode = Episode(
        experiences=experiences,
        task=task if task else "Unspecified task",
        env_name=info.get("env_name") or game or "",
        game_name=info.get("game_name") or info.get("structured_state", {}).get("game") or "",
    )
    episode.set_outcome()
    
    # Add experiences to experience buffer if provided
    if experience_buffer is not None:
        experience_buffer.add_experience(episode)  # This adds all experiences from the episode
        if verbose:
            print(f"Added {len(experiences)} experiences to experience buffer (size: {len(experience_buffer)})")
    
    # Add episode to episode buffer if provided
    if episode_buffer is not None:
        episode_buffer.add_episode(episode)
        if verbose:
            print(f"Added episode to episode buffer (size: {len(episode_buffer)})")
    
    if verbose:
        print(f"Episode completed: {len(experiences)} steps, total reward: {episode.get_reward()}")
    
    return episode
