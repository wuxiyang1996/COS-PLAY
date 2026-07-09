"""
VideoGameBench DOS env wrapper: state <-> natural language.

Supports DOS games only (JS-DOS in browser). Excludes Game Boy / PyBoy games.

Wraps the DOS pipeline (DOSGameServer + DOSGameInterface) so that:
- State is a natural language description (step, valid keyboard keys).
- step() accepts string keys (e.g. "ArrowUp", "Space", "KeyW").

Usage:

    from env_wrappers.videogamebench_dos_nl_wrapper import (
        VideoGameBenchDOSNLWrapper,
        list_dos_games,
        state_to_natural_language,
    )
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# DOS keyboard keys (Playwright codes) - common keys for DOS games
VIDEOGAMEBENCH_DOS_VALID_KEYS = [
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Space", "Enter", "Escape",
    "KeyW", "KeyA", "KeyS", "KeyD",
]

# NL phrase -> Playwright key
_NL_TO_KEY: Dict[str, str] = {
    "up": "ArrowUp", "arrowup": "ArrowUp", "↑": "ArrowUp",
    "down": "ArrowDown", "arrowdown": "ArrowDown", "↓": "ArrowDown",
    "left": "ArrowLeft", "arrowleft": "ArrowLeft", "←": "ArrowLeft",
    "right": "ArrowRight", "arrowright": "ArrowRight", "→": "ArrowRight",
    "space": "Space", "spacebar": "Space",
    "enter": "Enter", "return": "Enter",
    "escape": "Escape", "esc": "Escape",
    "w": "KeyW", "keyw": "KeyW", "forward": "KeyW",
    "a": "KeyA", "keya": "KeyA",
    "s": "KeyS", "keys": "KeyS", "back": "KeyS",
    "d": "KeyD", "keyd": "KeyD",
}


def list_dos_games() -> List[str]:
    """Return DOS game names that have HTTP URLs (no ROMs required)."""
    try:
        from src.consts import GAME_URL_MAP
        return [
            k for k, v in GAME_URL_MAP.items()
            if isinstance(v, str) and v.startswith("http")
        ]
    except ImportError:
        return ["doom2", "doom", "quake", "civ", "oregon_trail", "incredible-machine"]


def state_to_natural_language(step: int, valid_keys: Optional[List[str]] = None) -> str:
    """Build NL state for DOS games."""
    keys = valid_keys or VIDEOGAMEBENCH_DOS_VALID_KEYS
    keys_str = ", ".join(keys)
    return (
        f"You see the DOS game screen. Step {step}.\n\n"
        f"Choose one key to press: {keys_str}\n"
        f"(Examples: ArrowUp, ArrowDown, Space, Enter, KeyW, KeyA, KeyS, KeyD, Escape)"
    )


def natural_language_to_key(text: str) -> str:
    """Convert NL action to Playwright key. Falls back to Space if unparseable."""
    if not text or not isinstance(text, str):
        return "Space"
    s = text.strip()
    # Direct match
    if s in VIDEOGAMEBENCH_DOS_VALID_KEYS:
        return s
    lower = s.lower()
    if lower in _NL_TO_KEY:
        return _NL_TO_KEY[lower]
    # "press ArrowUp", "arrow up"
    for prefix in ("press ", "key ", "press_key "):
        if lower.startswith(prefix):
            rest = lower[len(prefix):].strip().replace(" ", "")
            if rest in _NL_TO_KEY:
                return _NL_TO_KEY[rest]
    words = re.split(r"[\s,]+", lower)
    first = words[0] if words else ""
    if first in _NL_TO_KEY:
        return _NL_TO_KEY[first]
    # Partial
    for k, v in _NL_TO_KEY.items():
        if first and (k.startswith(first) or first.startswith(k)):
            return v
    return "Space"


class VideoGameBenchDOSNLWrapper:
    """
    Wraps DOS game state as NL for language-model agents.
    Use with async DOS pipeline (DOSGameServer, DOSGameInterface).
    """

    def __init__(
        self,
        valid_keys: Optional[List[str]] = None,
        game_name: str = "dos_game",
    ):
        self._step_count = 0
        self._valid_keys = valid_keys or VIDEOGAMEBENCH_DOS_VALID_KEYS
        self._game_name = game_name

    def build_state_nl(self) -> str:
        return state_to_natural_language(step=self._step_count, valid_keys=self._valid_keys)

    def parse_action(self, action: str) -> str:
        return natural_language_to_key(action)

    def advance_step(self) -> None:
        self._step_count += 1

    def inject_info(self, info: dict) -> dict:
        """Add env_name and game_name to an info dict for Episode identification."""
        info["env_name"] = "videogamebench_dos"
        info["game_name"] = self._game_name
        return info
