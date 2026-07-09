"""
Per-game default configurations for the LMGame-Bench evaluation suite.

LMGame-Bench (GamingAgent) contains 11 games across 3 categories:

  CUSTOM (4):  2048, Candy Crush, Tetris, Doom
  RETRO  (3):  Super Mario Bros, Ace Attorney, 1942
  ZOO    (2):  Tic-Tac-Toe, Texas Hold'em

Orak benchmark (krafton-ai/Orak) adds 10 games across 6 genres:

  FREE (4):    2048, Super Mario, Street Fighter III, Minecraft
  PAID (6):    Baba Is You, Ace Attorney, Her Story, Darkest Dungeon, Slay the Spire, Stardew Valley

Availability tiers for Orak free games:
  - 2048:             pip-only, no external deps, runs anywhere
  - Super Mario:      pip-only (gym-super-mario-bros bundles the ROM)
  - Street Fighter:   needs Diambra account (free) + Docker + ROM
  - Minecraft:        needs Minecraft Java ($30 client) + Voyager + Node.js
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GameConfig:
    name: str
    display_name: str
    category: str
    max_steps: int
    episodes: int
    available: bool = True
    env_vars: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    notes: str = ""
    rom_required: bool = False
    purchase_required: bool = False
    setup_difficulty: str = ""


GAME_CONFIGS: Dict[str, GameConfig] = {
    # ── Custom games ────────────────────────────────────────────────────
    "twenty_forty_eight": GameConfig(
        name="twenty_forty_eight",
        display_name="2048",
        category="custom",
        max_steps=200,
        episodes=3,
        description="Slide tiles to reach 2048",
        notes="Reward = tile merges score",
    ),
    "candy_crush": GameConfig(
        name="candy_crush",
        display_name="Candy Crush",
        category="custom",
        max_steps=50,
        episodes=3,
        description="Swap adjacent tiles to match 3+",
        notes="Dynamic action space: effective swaps shown per step",
    ),
    "tetris": GameConfig(
        name="tetris",
        display_name="Tetris",
        category="custom",
        max_steps=200,
        episodes=3,
        description="Place falling tetrominoes to clear lines",
        notes="Benchmark uses TetrisMacroActionWrapper (placement-level actions)",
    ),
    "doom": GameConfig(
        name="doom",
        display_name="Doom (Basic)",
        category="custom",
        max_steps=50,
        episodes=2,
        available=False,
        env_vars={"DISPLAY": "", "SDL_VIDEODRIVER": "dummy"},
        description="VizDoom basic scenario: shoot the monster",
        notes="Excluded: text-only mode too limited for meaningful training",
    ),
    # ── Retro games ─────────────────────────────────────────────────────
    "super_mario_bros": GameConfig(
        name="super_mario_bros",
        display_name="Super Mario Bros",
        category="retro",
        max_steps=500,
        episodes=1,
        available=False,
        rom_required=True,
        description="Classic NES platformer",
        notes="Requires SuperMarioBros-Nes ROM via stable-retro",
    ),
    "ace_attorney": GameConfig(
        name="ace_attorney",
        display_name="Ace Attorney",
        category="retro",
        max_steps=200,
        episodes=1,
        available=False,
        rom_required=True,
        description="Visual novel / courtroom adventure (GBA)",
        notes="Requires AceAttorney-GbAdvance ROM via stable-retro",
    ),
    "nineteen_forty_two": GameConfig(
        name="nineteen_forty_two",
        display_name="1942",
        category="retro",
        max_steps=500,
        episodes=1,
        available=False,
        rom_required=True,
        description="Classic NES vertical shoot-em-up",
        notes="Requires 1942-Nes ROM via stable-retro",
    ),

    # ── Orak benchmark games (krafton-ai/Orak) ─────────────────────────
    #
    # FREE games (no purchase required):
    #   2048, Super Mario, Street Fighter III
    # PAID games ($10-$25 one-time purchase each):
    #   Baba Is You, Ace Attorney, Her Story, Darkest Dungeon, Slay the Spire, Stardew Valley
    #
    # ── FREE: Puzzle ────────────────────────────────────────────────────
    "orak_twenty_fourty_eight": GameConfig(
        name="orak_twenty_fourty_eight",
        display_name="2048 (Orak)",
        category="orak_free",
        max_steps=1000,
        episodes=3,
        description="Orak 2048: merge tiles, score = min(score/20000*100, 100)",
        notes="Pure Python + pygame; no external deps; runs anywhere",
        setup_difficulty="easy",
    ),
    # ── FREE: Action ────────────────────────────────────────────────────
    "orak_super_mario": GameConfig(
        name="orak_super_mario",
        display_name="Super Mario (Orak)",
        category="orak_free",
        max_steps=100,
        episodes=3,
        description="Orak Super Mario: jump-level control (0-6), score = x_pos/3161*100",
        notes="pip install gym-super-mario-bros (ROM bundled in package)",
        setup_difficulty="easy",
    ),
    "orak_street_fighter": GameConfig(
        name="orak_street_fighter",
        display_name="Street Fighter III (Orak)",
        category="orak_free",
        max_steps=500,
        episodes=3,
        description="Orak Street Fighter III: character-specific meta-instructions, score = stages cleared",
        notes="Free Diambra account + Docker + sfiii3n.zip ROM",
        setup_difficulty="medium",
    ),
    # ── FREE: Simulation ────────────────────────────────────────────────
    "orak_minecraft": GameConfig(
        name="orak_minecraft",
        display_name="Minecraft (Orak)",
        category="orak_free",
        max_steps=200,
        episodes=3,
        available=False,
        description="Orak Minecraft: craft target items via JavaScript async functions",
        notes="Minecraft Java Edition (~$30 client) + Voyager + Node.js + Fabric mods; listed as free by Orak but client costs money",
        setup_difficulty="hard",
    ),
    # ── PAID: Puzzle ────────────────────────────────────────────────────
    "orak_baba_is_you": GameConfig(
        name="orak_baba_is_you",
        display_name="Baba Is You (Orak)",
        category="orak_paid",
        max_steps=200,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Orak Baba Is You: rule-manipulation puzzle, 100=win/40=WIN exists/20=WALL broken",
        notes="Steam purchase ~$15; Lua mod scripts required",
        setup_difficulty="medium",
    ),
    # ── PAID: Strategy ──────────────────────────────────────────────────
    "orak_slay_the_spire": GameConfig(
        name="orak_slay_the_spire",
        display_name="Slay the Spire (Orak)",
        category="orak_paid",
        max_steps=500,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Orak Slay the Spire: deck-building roguelike, score = floor reached (max 50)",
        notes="Steam purchase ~$25; ModTheSpire + BaseMod + CommunicationMod required",
        setup_difficulty="medium",
    ),
    # ── PAID: RPG ───────────────────────────────────────────────────────
    "orak_darkest_dungeon": GameConfig(
        name="orak_darkest_dungeon",
        display_name="Darkest Dungeon (Orak)",
        category="orak_paid",
        max_steps=300,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Orak Darkest Dungeon: dungeon raid, 0.4*combat + 0.3*survival + 0.3*(1-stress)",
        notes="Steam purchase ~$25; Windows only; x360ce + Save Editor + Java required",
        setup_difficulty="hard",
    ),
    # ── PAID: Adventure ─────────────────────────────────────────────────
    "orak_pwaat": GameConfig(
        name="orak_pwaat",
        display_name="Ace Attorney (Orak)",
        category="orak_paid",
        max_steps=300,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Orak Phoenix Wright: Ace Attorney Trilogy, milestone reward scoring",
        notes="Steam purchase ~$20; Windows only; BepInEx 5 plugin required",
        setup_difficulty="medium",
    ),
    "orak_her_story": GameConfig(
        name="orak_her_story",
        display_name="Her Story (Orak)",
        category="orak_paid",
        max_steps=400,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Orak Her Story: search keywords & watch videos, score = views/272",
        notes="Steam purchase ~$6; Unity Doorstop + Harmony plugin required",
        setup_difficulty="medium",
    ),
    # ── PAID: Simulation ────────────────────────────────────────────────
    "orak_stardew_valley": GameConfig(
        name="orak_stardew_valley",
        display_name="Stardew Valley (Orak)",
        category="orak_paid",
        max_steps=300,
        episodes=3,
        available=False,
        purchase_required=True,
        description="Orak Stardew Valley: farming tasks (cleanup/cultivation/shopping/earn money)",
        notes="Steam purchase ~$15; Windows only; SMAPI + StateExtractor mod required",
        setup_difficulty="medium",
    ),

    # ── OSWorld (desktop automation benchmark) ──────────────────────────
    "osworld": GameConfig(
        name="osworld",
        display_name="OSWorld",
        category="desktop",
        max_steps=15,
        episodes=1,
        description="Open-ended desktop automation tasks in real Ubuntu/Windows VMs",
        notes="Requires OSWorld (pip install desktop-env) + VM provider (vmware/docker/aws). "
              "369 tasks across LibreOffice, Firefox, GIMP, VLC, Thunderbird, VS Code, etc.",
        setup_difficulty="medium",
    ),

    # ── Zoo (multi-agent) games ─────────────────────────────────────────
    "tictactoe": GameConfig(
        name="tictactoe",
        display_name="Tic-Tac-Toe",
        category="zoo",
        max_steps=10,
        episodes=10,
        description="3x3 board game vs random opponent",
        notes="Simple perfect-info game; LLM should win most games",
    ),
    "texasholdem": GameConfig(
        name="texasholdem",
        display_name="Texas Hold'em",
        category="zoo",
        max_steps=30,
        episodes=10,
        description="Heads-up poker vs random opponent",
        notes="Imperfect-info game; single hand per episode",
    ),
}

ALL_GAME_NAMES: List[str] = sorted(GAME_CONFIGS.keys())
AVAILABLE_GAME_NAMES: List[str] = sorted(
    k for k, v in GAME_CONFIGS.items() if v.available
)
UNAVAILABLE_GAME_NAMES: List[str] = sorted(
    k for k, v in GAME_CONFIGS.items() if not v.available
)

TOTAL_GAMES = len(ALL_GAME_NAMES)
AVAILABLE_GAMES = len(AVAILABLE_GAME_NAMES)
