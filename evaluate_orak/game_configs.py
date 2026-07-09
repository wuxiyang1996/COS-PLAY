"""
Orak-specific game configurations.

12 games from krafton-ai/Orak across 6 genres, categorised by cost:

  FREE (7 entries, 6 unique games):
    2048, Super Mario, Street Fighter III, StarCraft II, StarCraft II Multi,
    Pokemon Red, Minecraft*

  PAID (6 games):
    Baba Is You, Ace Attorney, Her Story, Darkest Dungeon,
    Slay the Spire, Stardew Valley

  * Minecraft is classified as "free" by Orak but the Java client costs ~$30.

Availability tiers for free games:
  easy   -- pip-only, no external deps, runs anywhere (2048, Super Mario)
  medium -- free account or ROM needed (Street Fighter, Pokemon Red)
  hard   -- complex multi-component setup (StarCraft II, Minecraft)
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OrakGameConfig:
    name: str
    display_name: str
    genre: str
    cost_tier: str              # "free" | "paid"
    max_steps: int
    episodes: int
    available: bool = True
    purchase_required: bool = False
    setup_difficulty: str = ""  # "easy" | "medium" | "hard"
    description: str = ""
    notes: str = ""
    rom_required: bool = False
    platform: str = "any"       # "any" | "windows" | "linux"


ORAK_GAME_CONFIGS: Dict[str, OrakGameConfig] = {
    # ── FREE: Puzzle ─────────────────────────────────────────────────────
    "orak_twenty_fourty_eight": OrakGameConfig(
        name="orak_twenty_fourty_eight",
        display_name="2048 (Orak)",
        genre="puzzle",
        cost_tier="free",
        max_steps=1000,
        episodes=3,
        description="Merge tiles to reach 2048. Score = min(score/20000*100, 100).",
        notes="Pure Python + pygame; no external deps; runs anywhere.",
        setup_difficulty="easy",
    ),
    # ── FREE: Action ─────────────────────────────────────────────────────
    "orak_super_mario": OrakGameConfig(
        name="orak_super_mario",
        display_name="Super Mario (Orak)",
        genre="action",
        cost_tier="free",
        max_steps=100,
        episodes=3,
        description="Advance Mario as far right as possible. Score = x_pos / 3161 * 100.",
        notes="pip install gym-super-mario-bros (ROM bundled). Requires Python 3.10-3.12 for nes-py.",
        setup_difficulty="easy",
    ),
    "orak_street_fighter": OrakGameConfig(
        name="orak_street_fighter",
        display_name="Street Fighter III (Orak)",
        genre="action",
        cost_tier="free",
        max_steps=500,
        episodes=3,
        description="Defeat the opponent in Street Fighter III. Score = stages cleared.",
        notes="Free Diambra account + Docker + sfiii3n.zip ROM.",
        setup_difficulty="medium",
    ),
    # ── FREE: Strategy ───────────────────────────────────────────────────
    "orak_star_craft": OrakGameConfig(
        name="orak_star_craft",
        display_name="StarCraft II (Orak)",
        genre="strategy",
        cost_tier="free",
        max_steps=1000,
        episodes=3,
        description="Win 1v1 as Protoss vs Zerg bot. Provide 5 sequential macro actions per step.",
        notes="Free Battle.net account + SC2 client (free-to-play) + map files + burnysc2.",
        setup_difficulty="hard",
    ),
    "orak_star_craft_multi": OrakGameConfig(
        name="orak_star_craft_multi",
        display_name="StarCraft II Multi (Orak)",
        genre="strategy",
        cost_tier="free",
        max_steps=1000,
        episodes=3,
        description="Win 1v1 StarCraft II (multi-player mode). Provide 5 actions per step.",
        notes="Same setup as orak_star_craft; two-player variant.",
        setup_difficulty="hard",
    ),
    "orak_slay_the_spire": OrakGameConfig(
        name="orak_slay_the_spire",
        display_name="Slay the Spire (Orak)",
        genre="strategy",
        cost_tier="paid",
        max_steps=500,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Climb the Spire, defeat enemies with card combos. Score = floor reached (max 50).",
        notes="Steam purchase ~$25; ModTheSpire + BaseMod + CommunicationMod required.",
        setup_difficulty="medium",
    ),
    # ── FREE: RPG ────────────────────────────────────────────────────────
    "orak_pokemon_red": OrakGameConfig(
        name="orak_pokemon_red",
        display_name="Pokemon Red (Orak)",
        genre="rpg",
        cost_tier="free",
        max_steps=200,
        episodes=3,
        description="Progress through Pokemon Red storyline milestones (0-12 flags).",
        notes="Needs pokered.gbc ROM (legally obtained) + PyBoy + pret/pokered map data.",
        rom_required=True,
        setup_difficulty="medium",
    ),
    "orak_darkest_dungeon": OrakGameConfig(
        name="orak_darkest_dungeon",
        display_name="Darkest Dungeon (Orak)",
        genre="rpg",
        cost_tier="paid",
        max_steps=300,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Survive dungeon raids. Score = 0.4*combat + 0.3*survival + 0.3*(1-stress).",
        notes="Steam purchase ~$25; Windows only; x360ce + Save Editor + Java.",
        setup_difficulty="hard",
        platform="windows",
    ),
    # ── PAID: Adventure ──────────────────────────────────────────────────
    "orak_pwaat": OrakGameConfig(
        name="orak_pwaat",
        display_name="Ace Attorney (Orak)",
        genre="adventure",
        cost_tier="paid",
        max_steps=300,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Solve cases in Ace Attorney. Score = milestone rewards.",
        notes="Steam purchase ~$20; Windows only; BepInEx 5 plugin.",
        setup_difficulty="medium",
        platform="windows",
    ),
    "orak_her_story": OrakGameConfig(
        name="orak_her_story",
        display_name="Her Story (Orak)",
        genre="adventure",
        cost_tier="paid",
        max_steps=400,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Uncover the story by searching keywords and watching videos. Score = videos viewed / 272.",
        notes="Steam purchase ~$6; Unity Doorstop + Harmony plugin.",
        setup_difficulty="medium",
    ),
    # ── FREE: Simulation ─────────────────────────────────────────────────
    "orak_minecraft": OrakGameConfig(
        name="orak_minecraft",
        display_name="Minecraft (Orak)",
        genre="simulation",
        cost_tier="free",
        max_steps=200,
        episodes=3,
        available=False,
        description="Craft target items in Minecraft. Actions are JavaScript async functions.",
        notes="Minecraft Java Edition (~$30 client) + Voyager + Node.js + Fabric mods.",
        setup_difficulty="hard",
    ),
    "orak_stardew_valley": OrakGameConfig(
        name="orak_stardew_valley",
        display_name="Stardew Valley (Orak)",
        genre="simulation",
        cost_tier="paid",
        max_steps=300,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Complete farming tasks in Stardew Valley.",
        notes="Steam purchase ~$15; Windows only; SMAPI + StateExtractor mod.",
        setup_difficulty="medium",
        platform="windows",
    ),
    # ── PAID: Puzzle ─────────────────────────────────────────────────────
    "orak_baba_is_you": OrakGameConfig(
        name="orak_baba_is_you",
        display_name="Baba Is You (Orak)",
        genre="puzzle",
        cost_tier="paid",
        max_steps=200,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Solve Baba Is You puzzle by manipulating rules. 100=win, 40=WIN exists, 20=WALL broken.",
        notes="Steam purchase ~$15; Lua mod scripts required.",
        setup_difficulty="medium",
    ),
}

ORAK_FREE_GAMES: List[str] = sorted(
    k for k, v in ORAK_GAME_CONFIGS.items() if v.cost_tier == "free"
)
ORAK_PAID_GAMES: List[str] = sorted(
    k for k, v in ORAK_GAME_CONFIGS.items() if v.cost_tier == "paid"
)
ORAK_AVAILABLE_GAMES: List[str] = sorted(
    k for k, v in ORAK_GAME_CONFIGS.items() if v.available
)
ORAK_ALL_GAMES: List[str] = sorted(ORAK_GAME_CONFIGS.keys())
