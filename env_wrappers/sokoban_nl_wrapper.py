"""
Sokoban-specialized NL wrapper with spatial grid observations, rolling
memory, periodic reflection, and domain-specific prompts.

Enhanced with ideas from GamingAgent:
- Programmatic deadlock detection (corner, wall-line, frozen-box)
- Spatial analysis (Manhattan distances, box-target pairing)
- Restart action (auto-reset when deadlock detected)
- Multi-step planning prompt
- Error taxonomy from GamingAgent's workers.py
- Structured reflection checklist

Ported from ``cold_start/generate_cold_start_sokoban.py`` so that the
same rich prompt architecture is available during Qwen3 evaluation.

Usage::

    from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
    from evaluate_gamingagent.gym_like import make_gaming_env

    base_env = make_gaming_env("sokoban", max_steps=200)
    env = SokobanNLWrapper(base_env)
    obs, info = env.reset()

    # In the agent loop:
    system  = env.system_prompt
    user    = env.build_user_prompt()
    # ... send to LLM ...
    action  = env.parse_action(reply)
    obs, reward, term, trunc, info = env.step(action)
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MEMORY_STEPS = 8

VALID_ACTIONS = [
    "up", "down", "left", "right",
    "push up", "push down", "push left", "push right",
]

VALID_ACTIONS_WITH_RESTART = VALID_ACTIONS + ["restart"]

CHAR_LEGEND = {
    "#": "Wall",
    " ": "Floor",
    "@": "Player",
    "$": "Box",
    "?": "Target (empty)",
    "*": "Box on Target (solved!)",
    "+": "Player on Target",
}

_WALL_CHARS = {"#"}
_BOX_CHARS = {"$", "*"}
_TARGET_CHARS = {"?", "*", "+"}
_FLOOR_CHARS = {" ", "?", "@", "+"}


# ---------------------------------------------------------------------------
# Deadlock detection
# ---------------------------------------------------------------------------

def _is_wall(grid: List[List[str]], r: int, c: int) -> bool:
    """True if (r, c) is out of bounds or a wall."""
    if r < 0 or r >= len(grid):
        return True
    if c < 0 or c >= len(grid[r]):
        return True
    return grid[r][c] in _WALL_CHARS


def _get_targets(grid: List[List[str]]) -> Set[Tuple[int, int]]:
    targets = set()
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in ("?", "*", "+"):
                targets.add((r, c))
    return targets


def detect_corner_deadlocks(
    grid: List[List[str]],
    targets: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Detect boxes in simple corner deadlocks (two perpendicular walls)."""
    if targets is None:
        targets = _get_targets(grid)
    deadlocked: List[Tuple[int, int]] = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch != "$":
                continue
            # Box already on target is fine
            if (r, c) in targets:
                continue
            wall_up = _is_wall(grid, r - 1, c)
            wall_down = _is_wall(grid, r + 1, c)
            wall_left = _is_wall(grid, r, c - 1)
            wall_right = _is_wall(grid, r, c + 1)
            if (wall_up or wall_down) and (wall_left or wall_right):
                deadlocked.append((r, c))
    return deadlocked


def detect_wall_line_deadlocks(
    grid: List[List[str]],
    targets: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Detect boxes against a wall edge where no target exists along that wall."""
    if targets is None:
        targets = _get_targets(grid)
    deadlocked: List[Tuple[int, int]] = []
    n_rows = len(grid)

    for r, row in enumerate(grid):
        n_cols = len(row)
        for c, ch in enumerate(row):
            if ch != "$" or (r, c) in targets:
                continue
            # Already caught by corner detection — skip
            wall_up = _is_wall(grid, r - 1, c)
            wall_down = _is_wall(grid, r + 1, c)
            wall_left = _is_wall(grid, r, c - 1)
            wall_right = _is_wall(grid, r, c + 1)
            if (wall_up or wall_down) and (wall_left or wall_right):
                continue

            # Top wall: check if any target shares this top-wall row
            if wall_up:
                has_target = any((r, cc) in targets for cc in range(n_cols))
                if not has_target:
                    deadlocked.append((r, c))
                    continue
            # Bottom wall
            if wall_down:
                has_target = any((r, cc) in targets for cc in range(n_cols))
                if not has_target:
                    deadlocked.append((r, c))
                    continue
            # Left wall
            if wall_left:
                has_target = any((rr, c) in targets for rr in range(n_rows))
                if not has_target:
                    deadlocked.append((r, c))
                    continue
            # Right wall
            if wall_right:
                has_target = any((rr, c) in targets for rr in range(n_rows))
                if not has_target:
                    deadlocked.append((r, c))
                    continue

    return deadlocked


def detect_all_deadlocks(grid: List[List[str]]) -> Dict[str, List[Tuple[int, int]]]:
    """Run all deadlock detectors. Returns dict keyed by type."""
    targets = _get_targets(grid)
    corners = detect_corner_deadlocks(grid, targets)
    wall_lines = detect_wall_line_deadlocks(grid, targets)
    # Deduplicate wall-line deadlocks already in corners
    corner_set = set(corners)
    wall_only = [pos for pos in wall_lines if pos not in corner_set]
    return {
        "corner": corners,
        "wall_line": wall_only,
    }


def is_deadlocked(grid: List[List[str]]) -> bool:
    """Quick check: any unsolvable deadlock present?"""
    dl = detect_all_deadlocks(grid)
    return bool(dl["corner"] or dl["wall_line"])


# ---------------------------------------------------------------------------
# Spatial analysis
# ---------------------------------------------------------------------------

def _total_min_manhattan(grid: List[List[str]]) -> Optional[int]:
    """Sum of each unsolved box's Manhattan distance to its nearest empty target.

    Returns None if the grid has no unsolved boxes or no empty targets.
    """
    boxes: List[Tuple[int, int]] = []
    targets: List[Tuple[int, int]] = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "$":
                boxes.append((r, c))
            elif ch == "?":
                targets.append((r, c))
    if not boxes or not targets:
        return None
    return sum(
        min(abs(br - tr) + abs(bc - tc) for tr, tc in targets)
        for br, bc in boxes
    )


def compute_spatial_analysis(grid: List[List[str]]) -> str:
    """Compute per-box Manhattan distances, nearest targets, and deadlock warnings."""
    boxes: List[Tuple[int, int]] = []
    targets: List[Tuple[int, int]] = []
    player: Optional[Tuple[int, int]] = None
    solved: List[Tuple[int, int]] = []

    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "$":
                boxes.append((r, c))
            elif ch == "?":
                targets.append((r, c))
            elif ch in ("@", "+"):
                player = (r, c)
            elif ch == "*":
                solved.append((r, c))

    lines: List[str] = []

    if not boxes and solved:
        lines.append("All boxes are on targets — puzzle is SOLVED!")
        return "\n".join(lines)

    # Per-box analysis
    for br, bc in boxes:
        if not targets:
            lines.append(f"  Box({br},{bc}): no empty targets remaining")
            continue
        dists = sorted(
            [(abs(br - tr) + abs(bc - tc), (tr, tc)) for tr, tc in targets]
        )
        nearest_dist, nearest_pos = dists[0]
        lines.append(
            f"  Box({br},{bc}) → nearest target ({nearest_pos[0]},{nearest_pos[1]}) "
            f"dist={nearest_dist}"
        )

    # Total remaining distance (sum of min distances — lower bound)
    if boxes and targets:
        total = sum(
            min(abs(br - tr) + abs(bc - tc) for tr, tc in targets)
            for br, bc in boxes
        )
        lines.append(f"  Total min-distance (lower bound): {total}")

    # Deadlock report
    deadlocks = detect_all_deadlocks(grid)
    corner_dl = deadlocks["corner"]
    wall_dl = deadlocks["wall_line"]
    if corner_dl:
        lines.append(
            f"  *** CORNER DEADLOCK at {corner_dl} — PUZZLE LIKELY UNSOLVABLE ***"
        )
    if wall_dl:
        lines.append(
            f"  *** WALL-LINE DEADLOCK at {wall_dl} — PUZZLE LIKELY UNSOLVABLE ***"
        )

    # Player distance to nearest unsolved box
    if player and boxes:
        p_dists = sorted(
            [(abs(player[0] - br) + abs(player[1] - bc), (br, bc)) for br, bc in boxes]
        )
        lines.append(
            f"  Player({player[0]},{player[1]}) → nearest box "
            f"({p_dists[0][1][0]},{p_dists[0][1][1]}) dist={p_dists[0][0]}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sokoban-specific prompts (enhanced with GamingAgent techniques)
# ---------------------------------------------------------------------------

SOKOBAN_SYSTEM_PROMPT = """\
You are an expert AI player solving Sokoban puzzles. Your goal is to push ALL \
boxes ($) onto ALL target locations (?). A box on a target becomes *.

GRID CHARACTERS:
  #  Wall (impassable)
  .  Floor (empty space, shown as space in the grid)
  @  Player (you)
  $  Box on floor
  ?  Target location (empty — needs a box)
  *  Box on a target (SOLVED — do not move this off the target!)
  +  Player standing on a target

ACTIONS — choose exactly one:
  Movement only:   up, down, left, right
    → Moves the player one step if the destination is floor or empty target.
  Push:            push up, push down, push left, push right
    → Player must be adjacent to a box in that direction AND the cell beyond \
the box must be floor or empty target. Player and box both move one step.
  Restart:         restart
    → Resets the puzzle to its initial state. Use ONLY when you detect an \
unsolvable deadlock (box stuck in corner or against wall with no target).

CRITICAL STRATEGY:
1. PLAN AHEAD: Before moving, mentally simulate the next 3-5 moves. \
Trace the path from each box to the nearest unoccupied target. \
Decide which box to push first based on dependencies.
2. AVOID DEADLOCKS — these error patterns make the puzzle UNSOLVABLE:
   a) CORNER LOCK: Box pushed into a non-target corner (two perpendicular walls) → GAME OVER
   b) WALL-LINE LOCK: Box against a wall where no target exists along that wall → GAME OVER
   c) VERTICAL/HORIZONTAL STACK: Two boxes adjacent along a wall → often UNSOLVABLE
   d) PATH OBSTRUCTION: A box blocks your path to reach other boxes
   e) WRONG DOCK: Box on wrong target blocks the correct box's path
3. POSITIONING: Move the player around boxes to approach from the correct \
side before pushing. Use movement actions (up/down/left/right) to reposition.
4. DO NOT oscillate: If you find yourself repeating the same 2-3 moves, STOP \
and rethink your plan completely.
5. PUSH vs MOVE: Use "push X" ONLY when you are adjacent to a box AND want to \
push it. Use plain movement to navigate without pushing.
6. Once a box is on a target (*), try NOT to move it off unless required by your plan.
7. BOX ORDERING: Push the box that is farthest from all targets first, or push boxes \
that would block other boxes' paths if left in place.
8. RESTART WISELY: If the Spatial Analysis section reports a DEADLOCK, immediately \
use ACTION: restart rather than wasting moves.

MULTI-STEP PLANNING:
Before choosing your action, outline a brief plan:
  PLAN: step1 → step2 → step3
Then execute only the FIRST step of your plan.
If your planned sequence would create a deadlock at ANY step, choose differently.

Respond with EXACTLY this format:
PLAN: <your 3-5 step plan in brief>
REASONING: <your step-by-step reasoning about the current board, which box to \
target, where to position, and what to push>
ACTION: <action>"""

SOKOBAN_USER_TEMPLATE = """\
## Current Sokoban Board (row, col from top-left = 0,0)
```
{grid}
```

## Element Summary
{element_summary}

## Spatial Analysis
{spatial_analysis}

## Recent History ({n_history} steps)
{history}

## Reflection
{reflection}

## Task
Push all boxes ($) onto targets (?). Boxes on targets show as *.
Choose ONE action from: {actions}

PLAN: <your 3-5 step plan>
REASONING: <reasoning>
ACTION: <action>"""

REFLECTION_PROMPT = """\
Analyze the last {n} steps of this Sokoban game:

{trajectory}

Current board:
{board}

Spatial analysis:
{spatial_analysis}

Answer each question in ONE line (total under 80 words):
1. PROGRESS: How many boxes moved closer to targets vs. farther? (+N/-N)
2. DEADLOCK: Any box now in a corner or against a wall without a target? (yes/no, which position)
3. OSCILLATION: Am I repeating a cycle of moves? (yes/no)
4. PLAN: Which box should I push next, to which target, from which direction?
5. RESTART: Should I restart the puzzle? (yes if deadlock detected, no otherwise)"""


# ---------------------------------------------------------------------------
# Board parsing — convert the flat table back to a spatial grid
# ---------------------------------------------------------------------------

def table_obs_to_grid(obs_text: str) -> Optional[List[List[str]]]:
    """Parse the text-table observation back into a 2D character grid.

    The env produces a table like::

        ID  | Item Type    | Position
        --------------------------------
        1   | Wall         | (0, 0)
        2   | Wall         | (1, 0)
        ...

    We reconstruct the grid from these rows.
    """
    rows: Dict[Tuple[int, int], str] = {}
    type_to_char = {
        "wall": "#",
        "worker": "@",
        "box": "$",
        "dock": "?",
        "box on dock": "*",
        "worker on dock": "+",
        "empty": " ",
    }

    for line in obs_text.splitlines():
        line = line.strip()
        if not line or line.startswith("ID") or line.startswith("-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        item_type = parts[1].strip().lower()
        pos_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", parts[2])
        if not pos_match:
            continue
        col, row = int(pos_match.group(1)), int(pos_match.group(2))
        char = type_to_char.get(item_type, "?")
        rows[(row, col)] = char

    if not rows:
        return None

    max_r = max(r for r, _ in rows) + 1
    max_c = max(c for _, c in rows) + 1
    grid = [[" " for _ in range(max_c)] for _ in range(max_r)]
    for (r, c), ch in rows.items():
        grid[r][c] = ch
    return grid


def grid_to_string(grid: List[List[str]]) -> str:
    """Render grid with row/col indices for spatial reasoning."""
    if not grid:
        return "(empty board)"
    max_c = max(len(row) for row in grid)
    # Multi-digit column headers
    col_header = "    " + "".join(f"{c % 10}" for c in range(max_c))
    lines = [col_header]
    for r, row in enumerate(grid):
        lines.append(f"{r:>3} " + "".join(row))
    return "\n".join(lines)


def summarize_elements(grid: List[List[str]]) -> str:
    """List positions of key elements for quick reference."""
    player_pos = []
    boxes = []
    targets = []
    boxes_on_target = []

    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "@":
                player_pos.append((r, c))
            elif ch == "+":
                player_pos.append((r, c))
                targets.append((r, c))
            elif ch == "$":
                boxes.append((r, c))
            elif ch == "?":
                targets.append((r, c))
            elif ch == "*":
                boxes_on_target.append((r, c))

    parts = []
    if player_pos:
        parts.append(f"Player: {player_pos[0]}")
    if boxes:
        parts.append(f"Boxes on floor: {boxes}")
    if targets:
        parts.append(f"Empty targets: {targets}")
    if boxes_on_target:
        parts.append(f"Boxes on target (solved): {boxes_on_target}")
    total_boxes = len(boxes) + len(boxes_on_target)
    total_targets = len(targets) + len(boxes_on_target)
    parts.append(f"Progress: {len(boxes_on_target)}/{total_targets} targets filled")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rolling memory (enhanced)
# ---------------------------------------------------------------------------

class SokobanMemory:
    """Rolling window of recent (action, board_summary, reward) tuples."""

    def __init__(self, max_steps: int = MAX_MEMORY_STEPS):
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.last_reflection: str = "Game just started. Survey the board and plan."
        self.restart_count: int = 0
        self.total_deadlocks_detected: int = 0

    def add(
        self,
        step: int,
        action: str,
        reward: float,
        board_summary: str,
        deadlock_info: str = "",
    ):
        self.history.append({
            "step": step,
            "action": action,
            "reward": reward,
            "summary": board_summary,
            "deadlock": deadlock_info,
        })
        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]

    def format_history(self) -> str:
        if not self.history:
            return "(no previous actions)"
        lines = []
        for h in self.history:
            r_str = f"{h['reward']:+.2f}"
            dl = f" [{h['deadlock']}]" if h.get("deadlock") else ""
            lines.append(f"  Step {h['step']}: {h['action']} → reward {r_str}{dl}")
        return "\n".join(lines)

    def format_trajectory_for_reflection(self) -> str:
        if not self.history:
            return "(no actions yet)"
        lines = []
        for h in self.history:
            dl = f"  Deadlock: {h['deadlock']}" if h.get("deadlock") else ""
            lines.append(
                f"Step {h['step']}: action={h['action']}, "
                f"reward={h['reward']:+.2f}"
            )
            lines.append(f"  Board: {h['summary']}")
            if dl:
                lines.append(dl)
        return "\n".join(lines)

    def detect_oscillation(self, window: int = 6) -> bool:
        """Check if recent actions form a repeating cycle."""
        if len(self.history) < window:
            return False
        recent = [h["action"] for h in self.history[-window:]]
        # Check for 2-cycle: A B A B
        if len(recent) >= 4:
            if recent[-1] == recent[-3] and recent[-2] == recent[-4]:
                return True
        # Check for 3-cycle: A B C A B C
        if len(recent) >= 6:
            if (recent[-1] == recent[-4] and
                    recent[-2] == recent[-5] and
                    recent[-3] == recent[-6]):
                return True
        return False


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_sokoban_action(reply: str) -> Optional[str]:
    """Extract a valid Sokoban action from free-form LLM reply."""
    if not reply:
        return None

    lower = reply.lower()

    # Try ACTION: pattern first
    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", reply, re.IGNORECASE)
    if action_m:
        candidate = action_m.group(1).strip().lower()
        for v in VALID_ACTIONS_WITH_RESTART:
            if v == candidate:
                return v

    # Try move: pattern
    move_m = re.search(r"move:\s*(.+?)(?:\n|$)", lower)
    if move_m:
        candidate = move_m.group(1).strip()
        for v in VALID_ACTIONS_WITH_RESTART:
            if v == candidate:
                return v

    # Fallback: find longest matching action in text
    for v in sorted(VALID_ACTIONS_WITH_RESTART, key=len, reverse=True):
        if v in lower:
            return v

    return None


# ---------------------------------------------------------------------------
# SokobanNLWrapper
# ---------------------------------------------------------------------------

class SokobanNLWrapper:
    """
    Wraps GamingAgentEnv (Sokoban) so observations are spatial-grid NL
    strings with rolling memory and domain-specific prompt support.

    Enhanced features (borrowed from GamingAgent):
    - Programmatic deadlock detection with alerts in observation
    - Spatial analysis (distances, box-target pairing)
    - Restart action (auto-resets puzzle on deadlock)
    - Multi-step planning in prompts
    - Structured reflection with checklist

    Provides:
    - ``system_prompt``: Domain-specific system prompt.
    - ``build_user_prompt()``: Rich user prompt with grid, elements, spatial
      analysis, history, reflection.
    - ``parse_action(reply)``: Extract valid Sokoban action from LLM reply.
    - ``generate_reflection(ask_fn, model)``: Ask the agent model to reflect.
    """

    _has_rich_observation = True

    # Penalty for restarting (discourages frivolous restarts)
    RESTART_PENALTY = -1.0
    # Max restarts before we stop allowing them
    MAX_RESTARTS = 3

    DISTANCE_SHAPING_BONUS = 0.2
    DEADLOCK_SHAPING_PENALTY = -0.5

    def __init__(
        self,
        env: Any,
        max_memory: int = MAX_MEMORY_STEPS,
        reflect_every: int = 3,
        enable_restart: bool = True,
        auto_detect_deadlock: bool = True,
        distance_shaping: bool = True,
    ):
        self._env = env
        self._memory = SokobanMemory(max_steps=max_memory)
        self._reflect_every = reflect_every
        self._step_count = 0
        self._last_grid: Optional[List[List[str]]] = None
        self._last_obs_nl: str = ""
        self._enable_restart = enable_restart
        self._auto_detect_deadlock = auto_detect_deadlock
        self._distance_shaping = distance_shaping
        self._prev_distance: Optional[int] = None
        self._initial_seed: Optional[int] = None
        self._initial_options: Optional[Dict[str, Any]] = None
        self._action_names = (
            VALID_ACTIONS_WITH_RESTART if enable_restart else VALID_ACTIONS
        )
        # Track consecutive deadlocked states for auto-restart
        self._consecutive_deadlock_steps = 0
        self._deadlock_auto_restart_threshold = 3

    @property
    def env(self):
        return self._env

    @property
    def system_prompt(self) -> str:
        return SOKOBAN_SYSTEM_PROMPT

    @property
    def action_names(self) -> List[str]:
        return list(self._action_names)

    @property
    def memory(self) -> SokobanMemory:
        return self._memory

    @property
    def last_grid(self) -> Optional[List[List[str]]]:
        return self._last_grid

    def _process_obs(self, obs_nl: str) -> Tuple[str, Optional[List[List[str]]]]:
        """Parse table obs into grid; return (obs_text, grid_or_none)."""
        grid = table_obs_to_grid(obs_nl)
        if grid is not None:
            self._last_grid = grid
        return obs_nl, grid

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._memory = SokobanMemory(max_steps=self._memory.max_steps)
        self._consecutive_deadlock_steps = 0

        # Remember reset params for restart
        self._initial_seed = seed
        self._initial_options = options

        obs_nl = obs if isinstance(obs, str) else str(obs.get("text", obs))
        self._last_obs_nl, self._last_grid = self._process_obs(obs_nl)

        self._prev_distance = (
            _total_min_manhattan(self._last_grid)
            if self._distance_shaping and self._last_grid is not None
            else None
        )

        info["action_names"] = self._action_names
        info["env_name"] = "gamingagent"
        info["game_name"] = "sokoban"
        if self._last_grid is not None:
            info["grid_string"] = grid_to_string(self._last_grid)
            info["element_summary"] = summarize_elements(self._last_grid)
            info["spatial_analysis"] = compute_spatial_analysis(self._last_grid)
        return self._last_obs_nl, info

    def _handle_restart(self) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Reset the env and return observation with a restart penalty."""
        self._memory.restart_count += 1
        obs, info = self._env.reset(
            seed=self._initial_seed, options=self._initial_options,
        )
        self._consecutive_deadlock_steps = 0
        # Don't reset step count — restarts still cost steps
        self._step_count += 1

        obs_nl = obs if isinstance(obs, str) else str(obs.get("text", obs))
        self._last_obs_nl, self._last_grid = self._process_obs(obs_nl)

        self._prev_distance = (
            _total_min_manhattan(self._last_grid)
            if self._distance_shaping and self._last_grid is not None
            else None
        )

        board_summary = ""
        if self._last_grid is not None:
            board_summary = summarize_elements(self._last_grid)

        self._memory.add(
            self._step_count, "restart", self.RESTART_PENALTY,
            board_summary, deadlock_info="RESTART issued",
        )

        info["action_names"] = self._action_names
        info["env_name"] = "gamingagent"
        info["game_name"] = "sokoban"
        info["step"] = self._step_count
        info["restart"] = True
        if self._last_grid is not None:
            info["grid_string"] = grid_to_string(self._last_grid)
            info["element_summary"] = board_summary
            info["spatial_analysis"] = compute_spatial_analysis(self._last_grid)
        return obs_nl, self.RESTART_PENALTY, False, False, info

    def step(
        self,
        action: Union[str, int, np.integer],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        action_str = str(action).lower().strip()

        # Handle restart action
        if action_str == "restart":
            if (self._enable_restart
                    and self._memory.restart_count < self.MAX_RESTARTS):
                return self._handle_restart()
            # If restarts exhausted, treat as no-op (move up)
            action = "up"

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        obs_nl = obs if isinstance(obs, str) else str(obs.get("text", obs))
        self._last_obs_nl, self._last_grid = self._process_obs(obs_nl)

        board_summary = ""
        deadlock_info = ""
        shaping_bonus = 0.0
        if self._last_grid is not None:
            board_summary = summarize_elements(self._last_grid)
            # Run deadlock detection
            if self._auto_detect_deadlock:
                dl = detect_all_deadlocks(self._last_grid)
                if dl["corner"]:
                    deadlock_info = f"CORNER DEADLOCK at {dl['corner']}"
                    self._memory.total_deadlocks_detected += 1
                    self._consecutive_deadlock_steps += 1
                elif dl["wall_line"]:
                    deadlock_info = f"WALL-LINE DEADLOCK at {dl['wall_line']}"
                    self._memory.total_deadlocks_detected += 1
                    self._consecutive_deadlock_steps += 1
                else:
                    self._consecutive_deadlock_steps = 0

            # Distance-based reward shaping
            if self._distance_shaping and self._prev_distance is not None:
                cur_dist = _total_min_manhattan(self._last_grid)
                if cur_dist is not None:
                    delta = self._prev_distance - cur_dist
                    if delta > 0:
                        shaping_bonus = self.DISTANCE_SHAPING_BONUS * delta
                    elif delta < 0:
                        shaping_bonus = self.DISTANCE_SHAPING_BONUS * delta
                    self._prev_distance = cur_dist
                elif cur_dist is None and self._prev_distance > 0:
                    # All boxes now on targets — puzzle solved, big bonus
                    shaping_bonus = self.DISTANCE_SHAPING_BONUS * self._prev_distance
                    self._prev_distance = 0
                # Deadlock penalty on top of distance shaping
                if deadlock_info:
                    shaping_bonus += self.DEADLOCK_SHAPING_PENALTY

        reward = float(reward) + shaping_bonus
        info["raw_env_reward"] = float(reward) - shaping_bonus
        info["shaping_bonus"] = shaping_bonus

        self._memory.add(
            self._step_count, str(action), reward,
            board_summary, deadlock_info=deadlock_info,
        )

        info["action_names"] = self._action_names
        info["env_name"] = "gamingagent"
        info["game_name"] = "sokoban"
        info["step"] = self._step_count
        info["deadlock_detected"] = bool(deadlock_info)
        info["deadlock_info"] = deadlock_info
        info["restart_count"] = self._memory.restart_count
        if self._last_grid is not None:
            info["grid_string"] = grid_to_string(self._last_grid)
            info["element_summary"] = board_summary
            info["spatial_analysis"] = compute_spatial_analysis(self._last_grid)
        return obs_nl, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_user_prompt(self, reflection_override: Optional[str] = None) -> str:
        """Build the rich user prompt with grid, elements, spatial analysis,
        history, and reflection."""
        if self._last_grid is not None:
            grid_str = grid_to_string(self._last_grid)
            element_summary = summarize_elements(self._last_grid)
            spatial = compute_spatial_analysis(self._last_grid)
        else:
            grid_str = self._last_obs_nl
            element_summary = "(could not parse grid)"
            spatial = "(could not compute — grid parse failed)"

        reflection = reflection_override or self._memory.last_reflection

        # Append oscillation warning if detected
        if self._memory.detect_oscillation():
            reflection += (
                "\n*** WARNING: Oscillation detected — you are repeating moves. "
                "STOP and choose a completely different strategy. ***"
            )

        # Append restart hint if deadlock persists
        if (self._consecutive_deadlock_steps
                >= self._deadlock_auto_restart_threshold
                and self._enable_restart
                and self._memory.restart_count < self.MAX_RESTARTS):
            reflection += (
                "\n*** DEADLOCK PERSISTS for "
                f"{self._consecutive_deadlock_steps} steps. "
                "Consider using ACTION: restart ***"
            )

        return SOKOBAN_USER_TEMPLATE.format(
            grid=grid_str,
            element_summary=element_summary,
            spatial_analysis=spatial,
            n_history=len(self._memory.history),
            history=self._memory.format_history(),
            reflection=reflection,
            actions=", ".join(self._action_names),
        )

    def should_reflect(self) -> bool:
        """Whether it's time for a reflection call (every N steps, after warmup)."""
        return (
            self._step_count > 0
            and len(self._memory.history) >= 3
            and self._step_count % self._reflect_every == 0
        )

    def generate_reflection(
        self,
        ask_fn: Callable[..., str],
        model: str,
    ) -> str:
        """Ask the LLM to reflect on recent history. Updates internal state.

        ``ask_fn`` should have signature
        ``ask_fn(prompt, model=..., temperature=..., max_tokens=...) -> str``.
        """
        if len(self._memory.history) < 3:
            return self._memory.last_reflection

        board_str = (
            grid_to_string(self._last_grid) if self._last_grid else "(no grid)"
        )
        spatial = (
            compute_spatial_analysis(self._last_grid)
            if self._last_grid
            else "(no analysis)"
        )

        prompt = (
            "You are a Sokoban game analyst. Give brief, actionable reflections "
            "using the structured checklist below.\n\n"
            + REFLECTION_PROMPT.format(
                n=len(self._memory.history),
                trajectory=self._memory.format_trajectory_for_reflection(),
                board=board_str,
                spatial_analysis=spatial,
            )
        )

        try:
            reflection = ask_fn(
                prompt, model=model, temperature=0.2, max_tokens=180,
            )
            if reflection and not reflection.startswith("Error"):
                self._memory.last_reflection = reflection
                return reflection
        except Exception:
            pass
        return self._memory.last_reflection

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_action(reply: str) -> str:
        """Extract a valid Sokoban action from LLM reply; falls back to 'up'."""
        action = parse_sokoban_action(reply)
        return action if action else "up"

    @staticmethod
    def parse_reasoning(reply: str) -> Optional[str]:
        """Extract REASONING from LLM reply."""
        m = re.search(
            r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)",
            reply, re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            r"thought\s*:\s*(.+?)(?=\nmove|\Z)",
            reply, re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def parse_plan(reply: str) -> Optional[str]:
        """Extract PLAN from LLM reply."""
        m = re.search(
            r"PLAN\s*:\s*(.+?)(?=\nREASONING|\nACTION|\Z)",
            reply, re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)
