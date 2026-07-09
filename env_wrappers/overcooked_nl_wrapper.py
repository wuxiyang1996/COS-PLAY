"""
Overcooked AI env wrapper: state <-> natural language.

- Converts env state (OvercookedState) into a natural language description.
- Converts agent output (natural language action) into action index 0..5.

Supports single-agent (one controlled agent) or multi-agent (joint action from both agents).
Optional GUI: set show_gui=True to display the game in a pygame window (non-blocking updates).
"""

import re
import time
from typing import Any, List, Optional, Tuple, Union

try:
    from overcooked_ai_py.mdp.actions import Action as OvercookedAction
    _INDEX_TO_ACTION_OBJ = list(OvercookedAction.INDEX_TO_ACTION)
except ImportError:
    _INDEX_TO_ACTION_OBJ = None

# Optional GUI (requires overcooked_ai_py and pygame)
try:
    import pygame
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    _GUI_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore
    StateVisualizer = None  # type: ignore
    _GUI_AVAILABLE = False

# Direction names for orientation and action mapping
_DIRECTION_NAMES = {
    (0, -1): "north",
    (0, 1): "south",
    (1, 0): "east",
    (-1, 0): "west",
}

# Natural language action phrases -> action index (see Action: NORTH=0, SOUTH=1, EAST=2, WEST=3, STAY=4, INTERACT=5)
_NL_ACTION_MAP = [
    ("north", "up", "move north", "go north", "walk north", "↑"),   # 0
    ("south", "down", "move south", "go south", "walk south", "↓"), # 1
    ("east", "right", "move east", "go east", "walk east", "→"),     # 2
    ("west", "left", "move west", "go west", "walk west", "←"),     # 3
    ("stay", "wait", "stay still", "do nothing", "hold"),           # 4
    ("interact", "use", "pick up", "put down", "pickup", "putdown", "interact with"),  # 5
]
# Build a single dict: phrase -> index (lowercased)
_NL_TO_INDEX = {}
for idx, phrases in enumerate(_NL_ACTION_MAP):
    for p in phrases:
        _NL_TO_INDEX[p.strip().lower()] = idx


def _orientation_to_name(orientation: tuple) -> str:
    """(dx, dy) -> 'north'|'south'|'east'|'west'."""
    return _DIRECTION_NAMES.get(tuple(orientation), "unknown")


# ---------------------------------------------------------------------------
# Compact structured state summary (for agent context / retrieval)
# ---------------------------------------------------------------------------

def _compact_held(held) -> str:
    """One-token description of a held object."""
    if held is None:
        return "none"
    name = getattr(held, "name", str(held))
    if name == "soup":
        if getattr(held, "is_ready", False):
            return "soup:ready"
        if getattr(held, "is_cooking", False):
            return "soup:cooking"
        return "soup:raw"
    return name


def _compact_pot_status(objects: dict) -> str:
    """Summarise pot state in ≤20 chars."""
    for pos, obj in objects.items():
        name = getattr(obj, "name", "")
        if name == "soup":
            ings = getattr(obj, "ingredients", [])
            n = len(ings)
            if getattr(obj, "is_ready", False):
                return f"pot:{n}ing:ready"
            if getattr(obj, "is_cooking", False):
                return f"pot:{n}ing:cooking"
            if n > 0:
                return f"pot:{n}ing:waiting"
    return "pot:empty"


def build_structured_state_summary(
    state: Any,
    agent_idx: int = 0,
    horizon: Optional[int] = None,
) -> dict:
    """Build a compact structured dict for the Overcooked state.

    Designed to be fed into ``compact_structured_state()`` from
    ``decision_agents.agent_helper``.

    Returns:
        Dict with short key=value-friendly fields.  Example::

            {"game": "overcooked", "self": "hold:onion near:pot",
             "ally": "hold:dish", "critical": "pot:2ing:cooking",
             "orders": "onion_soup", "time_left": 47}
    """
    players = getattr(state, "players", ())
    if len(players) < 2:
        return {"game": "overcooked", "critical": "invalid_state"}

    you = players[agent_idx]
    other = players[1 - agent_idx]

    y_held = _compact_held(getattr(you, "held_object", None))
    y_pos = getattr(you, "position", (0, 0))
    o_held = _compact_held(getattr(other, "held_object", None))

    objects = getattr(state, "objects", {})
    pot = _compact_pot_status(objects)

    timestep = getattr(state, "timestep", 0)
    time_left = (horizon - timestep) if horizon is not None else None

    orders_raw = getattr(state, "bonus_orders", []) or getattr(state, "all_orders", [])
    orders_str = ",".join(str(o) for o in orders_raw[:3]) if orders_raw else ""

    summary: dict = {"game": "overcooked"}
    summary["self"] = f"hold:{y_held} pos:{y_pos[0]},{y_pos[1]}"
    summary["ally"] = f"hold:{o_held}"
    summary["critical"] = pot
    if orders_str:
        summary["orders"] = orders_str
    if time_left is not None:
        summary["time_left"] = str(time_left)

    return summary


def _describe_held_object(held) -> str:
    if held is None:
        return "nothing"
    name = getattr(held, "name", str(held))
    if name == "soup":
        ingredients = getattr(held, "ingredients", [])
        ing = ", ".join(ingredients) if ingredients else "empty"
        if getattr(held, "is_ready", False):
            status = "ready to deliver"
        elif getattr(held, "is_cooking", False):
            status = "cooking"
        else:
            status = "in pot (not cooking yet)"
        return f"soup ({ing}; {status})"
    return name


def state_to_natural_language(
    state: Any,
    agent_idx: int = 0,
    horizon: Optional[int] = None,
) -> str:
    """
    Convert OvercookedState to a natural language description.

    Args:
        state: OvercookedState (has .players, .objects, .timestep, .bonus_orders, .all_orders).
        agent_idx: Which player index is "you" (0 or 1).
        horizon: Optional max steps; if set, "steps left" is included.

    Returns:
        A string describing the state for the controlled agent.
    """
    lines = []
    players = getattr(state, "players", ())
    if len(players) < 2:
        return "Invalid state: fewer than 2 players."

    # You (controlled agent)
    you = players[agent_idx]
    y_pos = getattr(you, "position", (0, 0))
    y_or = _orientation_to_name(getattr(you, "orientation", (0, -1)))
    y_held = _describe_held_object(getattr(you, "held_object", None))
    lines.append(
        f"You are at position {y_pos}, facing {y_or}, holding {y_held}."
    )

    # Teammate
    other_idx = 1 - agent_idx
    other = players[other_idx]
    o_pos = getattr(other, "position", (0, 0))
    o_or = _orientation_to_name(getattr(other, "orientation", (0, -1)))
    o_held = _describe_held_object(getattr(other, "held_object", None))
    lines.append(
        f"Your teammate is at position {o_pos}, facing {o_or}, holding {o_held}."
    )

    # Objects on the grid (not held)
    objects = getattr(state, "objects", {})
    if objects:
        obj_parts = []
        for pos, obj in objects.items():
            name = getattr(obj, "name", "object")
            if name == "soup":
                ingredients = getattr(obj, "ingredients", [])
                ing = ", ".join(ingredients) if ingredients else "empty"
                if getattr(obj, "is_ready", False):
                    status = "ready"
                elif getattr(obj, "is_cooking", False):
                    status = "cooking"
                else:
                    status = "in pot"
                obj_parts.append(f"  - At {pos}: soup ({ing}; {status})")
            else:
                obj_parts.append(f"  - At {pos}: {name}")
        lines.append("Objects on the grid:")
        lines.extend(obj_parts)
    else:
        lines.append("No objects currently on the grid (any ingredients/dishes may be held).")

    # Time
    timestep = getattr(state, "timestep", 0)
    lines.append(f"Current timestep: {timestep}.")
    if horizon is not None:
        lines.append(f"Steps remaining in episode: {horizon - timestep}.")

    # Orders (optional; can be verbose)
    bonus = getattr(state, "bonus_orders", [])
    all_orders = getattr(state, "all_orders", [])
    if bonus or all_orders:
        try:
            bonus_str = ", ".join(str(o) for o in bonus) if bonus else "none"
            lines.append(f"Bonus orders: {bonus_str}.")
        except Exception:
            lines.append("Current orders are active.")

    lines.append("")
    lines.append(
        "Choose one action: north, south, east, west, stay, or interact."
    )
    return "\n".join(lines)


def natural_language_to_action_index(text: Union[str, int]) -> int:
    """
    Convert natural language action to Overcooked action index 0..5.

    Accepts: int (returned as-is if in range), or str (parsed).
    Valid actions: NORTH=0, SOUTH=1, EAST=2, WEST=3, STAY=4, INTERACT=5.

    Args:
        text: Action as string (e.g. "move north", "interact") or int 0..5.

    Returns:
        Action index in [0, 5].

    Raises:
        ValueError: If text cannot be parsed or int is out of range.
    """
    if isinstance(text, int):
        if 0 <= text <= 5:
            return text
        raise ValueError(f"Action index must be 0..5, got {text}")

    s = (text or "").strip().lower()
    if not s:
        raise ValueError("Empty action string.")

    # Direct lookup
    if s in _NL_TO_INDEX:
        return _NL_TO_INDEX[s]

    # Starts with known action verb (e.g. "interact with pot")
    if s.startswith("interact"):
        return 5
    if s.startswith("stay") or s.startswith("wait") or s.startswith("hold"):
        return 4
    if s.startswith("north") or s.startswith("up"):
        return 0
    if s.startswith("south") or s.startswith("down"):
        return 1
    if s.startswith("east") or s.startswith("right"):
        return 2
    if s.startswith("west") or s.startswith("left"):
        return 3

    # Single word / first word match
    words = re.split(r"[\s,]+", s)
    first = words[0] if words else ""
    for phrase, idx in _NL_TO_INDEX.items():
        if first == phrase or (phrase.startswith(first) and len(first) >= 2):
            return idx

    raise ValueError(
        f"Cannot parse action: {text!r}. "
        "Use one of: north, south, east, west, stay, interact (or equivalent)."
    )


def joint_action_to_indices(
    joint_action: Union[
        Tuple[Union[str, int], Union[str, int]],
        List[Union[str, int]],
        dict,
    ],
) -> Tuple[int, int]:
    """
    Normalize multi-agent joint action to (index0, index1).

    Accepts:
        - (a0, a1) or [a0, a1]: tuple or list of two actions (each int or NL string).
        - {0: a0, 1: a1}: dict mapping agent index to action.

    Returns:
        (idx0, idx1) with each in 0..5.
    """
    if isinstance(joint_action, dict):
        a0 = joint_action.get(0)
        a1 = joint_action.get(1)
        if a0 is None or a1 is None:
            raise ValueError(
                "joint_action dict must have keys 0 and 1."
            )
    elif isinstance(joint_action, (tuple, list)) and len(joint_action) == 2:
        a0, a1 = joint_action[0], joint_action[1]
    else:
        raise ValueError(
            "joint_action must be (a0, a1), [a0, a1], or {0: a0, 1: a1}."
        )
    return (
        natural_language_to_action_index(a0),
        natural_language_to_action_index(a1),
    )


def state_to_natural_language_for_all_agents(
    state: Any,
    horizon: Optional[int] = None,
) -> List[str]:
    """
    Return state description from each agent's perspective (agent 0 and agent 1).

    Returns:
        [description_for_agent_0, description_for_agent_1].
    """
    return [
        state_to_natural_language(state, agent_idx=0, horizon=horizon),
        state_to_natural_language(state, agent_idx=1, horizon=horizon),
    ]


class OvercookedNLWrapper:
    """
    Wraps an Overcooked env so that:
    - Observation is the state as natural language (string, or list of strings in multi-agent mode).
    - step() accepts a single action (single-agent) or joint action (multi-agent).

    Single-agent (multi_agent=False): one controlled agent; step(action) with action int or NL string.
    Multi-agent (multi_agent=True): both agents; step(joint_action) with joint_action =
        (a0, a1), [a0, a1], or {0: a0, 1: a1} (each a_i int or NL string).

    The underlying env must provide info["overcooked_state"]. In single-agent mode it should
    also provide info["agent_idx"]. In multi-agent mode the base env step() should accept
    joint_action as (idx0, idx1) or (Action, Action) if overcooked_ai_py is available.
    """

    def __init__(
        self,
        env: Any,
        horizon: Optional[int] = None,
        multi_agent: bool = False,
        show_gui: bool = False,
        gui_delay_ms: Optional[int] = 100,
    ):
        """
        Args:
            env: Gymnasium-like env. reset() -> (obs, info); step(action or joint_action)
                 returns (obs, reward, terminated, truncated, info). info must contain
                 "overcooked_state". For single-agent, info should contain "agent_idx".
            horizon: Optional; if set, state description includes steps remaining.
                 If None, tries to read from env.base_env.horizon.
            multi_agent: If True, step() expects joint_action (a0, a1) and observations
                 include state NL for each agent (list of two strings).
            show_gui: If True, display the game state in a pygame window after each
                 reset/step (non-blocking). Requires overcooked_ai_py and pygame.
            gui_delay_ms: Delay in milliseconds after each GUI update when show_gui=True.
                 None = no delay. Use e.g. 200–500 for easier viewing.
        """
        self._env = env
        self._horizon = horizon
        self._multi_agent = bool(multi_agent)
        self._show_gui = bool(show_gui)
        self._gui_delay_ms = gui_delay_ms
        if self._horizon is None and hasattr(env, "base_env"):
            self._horizon = getattr(env.base_env, "horizon", None)
        # Lazy GUI state
        self._gui_visualizer: Optional[Any] = None
        self._gui_window: Optional[Any] = None
        self._gui_clock: Optional[Any] = None
        self._gui_closed = False
        if self._show_gui and not _GUI_AVAILABLE:
            import warnings
            warnings.warn(
                "show_gui=True but GUI not available (install overcooked_ai and pygame).",
                UserWarning,
                stacklevel=2,
            )

    @property
    def env(self):
        return self._env

    def _get_base_env(self) -> Optional[Any]:
        """Get env that has .state and .mdp (for rendering)."""
        base = getattr(self._env, "base_env", self._env)
        if hasattr(base, "state") and hasattr(base, "mdp"):
            return base
        return None

    def _render_gui(self, state: Any) -> None:
        """Update GUI window with current state (non-blocking). No-op if show_gui False or GUI unavailable."""
        if not self._show_gui or self._gui_closed or not _GUI_AVAILABLE or pygame is None or StateVisualizer is None:
            return
        base = self._get_base_env()
        if base is None:
            return
        grid = getattr(base.mdp, "terrain_mtx", None)
        if grid is None:
            return
        try:
            if self._gui_visualizer is None:
                self._gui_visualizer = StateVisualizer()
            hud_kw = {}
            if hasattr(base, "game_stats") and base.game_stats:
                gs = base.game_stats
                if "cumulative_sparse_rewards_by_agent" in gs:
                    hud_kw["cumulative_sparse_rewards_by_agent"] = gs["cumulative_sparse_rewards_by_agent"]
                if "cumulative_shaped_rewards_by_agent" in gs:
                    hud_kw["cumulative_shaped_rewards_by_agent"] = gs["cumulative_shaped_rewards_by_agent"]
            hud_data = StateVisualizer.default_hud_data(state, **hud_kw)
            surface = self._gui_visualizer.render_state(state, grid, hud_data=hud_data)
            if self._gui_window is None:
                pygame.init()
                self._gui_clock = pygame.time.Clock()
                self._gui_window = pygame.display.set_mode(
                    surface.get_size(),
                    pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
                )
            self._gui_window.blit(surface, (0, 0))
            pygame.display.flip()
            # Process events (quit, resize) without blocking
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._gui_closed = True
                    pygame.display.quit()
                    pygame.quit()
                    self._gui_window = None
                    return
                if event.type == pygame.VIDEORESIZE:
                    self._gui_window = pygame.display.set_mode(
                        event.dict["size"],
                        pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
                    )
                    self._gui_window.blit(
                        pygame.transform.scale(surface, event.dict["size"]), (0, 0)
                    )
                    pygame.display.flip()
            if self._gui_delay_ms is not None and self._gui_delay_ms > 0:
                time.sleep(self._gui_delay_ms / 1000.0)
        except Exception:
            self._gui_closed = True
            if self._gui_window is not None:
                try:
                    pygame.display.quit()
                    pygame.quit()
                except Exception:
                    pass
                self._gui_window = None

    def _state_to_nl(self, state: Any, agent_idx: int) -> str:
        return state_to_natural_language(
            state, agent_idx=agent_idx, horizon=self._horizon
        )

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        state = info.get("overcooked_state")
        if state is None:
            return obs, info

        if self._multi_agent:
            state_nl_0 = self._state_to_nl(state, 0)
            state_nl_1 = self._state_to_nl(state, 1)
            info["state_natural_language"] = state_nl_0  # primary
            info["state_natural_language_by_agent"] = [state_nl_0, state_nl_1]
            info["structured_state"] = build_structured_state_summary(
                state, agent_idx=0, horizon=self._horizon,
            )
            obs = [state_nl_0, state_nl_1]
        else:
            agent_idx = info.get("agent_idx", 0)
            state_nl = self._state_to_nl(state, agent_idx)
            info["state_natural_language"] = state_nl
            info["structured_state"] = build_structured_state_summary(
                state, agent_idx=agent_idx, horizon=self._horizon,
            )
            obs = state_nl
        self._render_gui(state)
        info["env_name"] = "overcooked"
        info["game_name"] = "overcooked"
        return obs, info

    def step(
        self,
        action: Union[
            str,
            int,
            Tuple[Union[str, int], Union[str, int]],
            List[Union[str, int]],
            dict,
        ],
    ):
        if self._multi_agent:
            idx0, idx1 = joint_action_to_indices(action)
            if _INDEX_TO_ACTION_OBJ is not None:
                joint = (
                    _INDEX_TO_ACTION_OBJ[idx0],
                    _INDEX_TO_ACTION_OBJ[idx1],
                )
            else:
                joint = (idx0, idx1)
            result = self._env.step(joint)
            # Support both gymnasium (obs, r, term, trunc, info) and base (state, r, done, info)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                # Base OvercookedEnv: (next_state, reward, done, env_info)
                next_state = result[0]
                reward = result[1]
                terminated = result[2]
                truncated = False
                info = dict(result[3]) if len(result) > 3 else {}
                info["overcooked_state"] = next_state
                obs = None
        else:
            if isinstance(action, (tuple, list, dict)):
                raise ValueError(
                    "Single-agent mode: pass one action (int or NL string). "
                    "Use multi_agent=True for joint actions."
                )
            if isinstance(action, str):
                action = natural_language_to_action_index(action)
            obs, reward, terminated, truncated, info = self._env.step(action)

        state = info.get("overcooked_state")
        if state is None:
            return obs, reward, terminated, truncated, info

        if self._multi_agent:
            state_nl_0 = self._state_to_nl(state, 0)
            state_nl_1 = self._state_to_nl(state, 1)
            info["state_natural_language"] = state_nl_0
            info["state_natural_language_by_agent"] = [state_nl_0, state_nl_1]
            info["structured_state"] = build_structured_state_summary(
                state, agent_idx=0, horizon=self._horizon,
            )
            obs = [state_nl_0, state_nl_1]
        else:
            agent_idx = info.get("agent_idx", 0)
            state_nl = self._state_to_nl(state, agent_idx)
            info["state_natural_language"] = state_nl
            info["structured_state"] = build_structured_state_summary(
                state, agent_idx=agent_idx, horizon=self._horizon,
            )
            obs = state_nl
        self._render_gui(state)
        info["env_name"] = "overcooked"
        info["game_name"] = "overcooked"
        return obs, reward, terminated, truncated, info

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        # Observation is now a string; many gym setups don't have Text space, so we don't set it.
        return getattr(self._env, "observation_space", None)

    @property
    def show_gui(self) -> bool:
        """Whether GUI updates are enabled."""
        return self._show_gui

    def close_gui(self) -> None:
        """Close the GUI window if open. Safe to call even when GUI was not used."""
        self._gui_closed = True
        if _GUI_AVAILABLE and pygame is not None and self._gui_window is not None:
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._gui_window = None
