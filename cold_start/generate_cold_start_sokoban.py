#!/usr/bin/env python
"""
Sokoban-specialized cold-start rollout generation using GPT-5.4.

Unlike the generic cold-start script, this leverages the rich prompt
architecture from GamingAgent's Sokoban module: domain-specific system
prompts with rules/strategy, a spatial grid observation format, and a
rolling memory window with per-step reflections.

Output structure (cold_start/output/gpt54_sokoban/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # Default: 60 episodes
    python cold_start/generate_cold_start_sokoban.py

    # Fewer episodes (testing)
    python cold_start/generate_cold_start_sokoban.py --episodes 5

    # Resume an interrupted run
    python cold_start/generate_cold_start_sokoban.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
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
# Imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

from cold_start.generate_cold_start import (
    GAME_REGISTRY,
    ColdStartEnvWrapper,
    get_cold_start_max_steps,
    label_trajectory,
)

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

try:
    from env_wrappers.sokoban_nl_wrapper import (
        compute_spatial_analysis,
        detect_all_deadlocks,
        is_deadlocked,
    )
except ImportError:
    compute_spatial_analysis = None
    detect_all_deadlocks = None
    is_deadlocked = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"
GAME_NAME = "sokoban"
MAX_MEMORY_STEPS = 8

# Character legend for the grid representation
CHAR_LEGEND = {
    "#": "Wall",
    " ": "Floor",
    "@": "Player",
    "$": "Box",
    "?": "Target (empty)",
    "*": "Box on Target (solved!)",
    "+": "Player on Target",
}

# ---------------------------------------------------------------------------
# Sokoban-specialized prompts (adapted from GamingAgent module_prompts.json)
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
    → Player must be adjacent to a box in that direction AND the cell beyond
      the box must be floor or empty target. Player and box both move one step.

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
3. POSITIONING: Move the player around boxes to approach from the correct side \
before pushing. Use movement actions (up/down/left/right) to reposition.
4. DO NOT oscillate: If you find yourself repeating the same 2-3 moves, STOP \
and rethink your plan completely.
5. PUSH vs MOVE: Use "push X" ONLY when you are adjacent to a box AND want to \
push it. Use plain movement to navigate without pushing.
6. Once a box is on a target (*), try NOT to move it off unless required by your plan.
7. BOX ORDERING: Push the box that is farthest from all targets first, or push boxes \
that would block other boxes' paths if left in place.

MULTI-STEP PLANNING:
Before choosing your action, outline a brief plan:
  plan: step1 → step2 → step3
Then execute only the FIRST step of your plan.

Respond with EXACTLY this format:
plan: <your 3-5 step plan in brief>
thought: <your step-by-step reasoning about the current board, which box to \
target, where to position, and what to push>
move: <action>

Example:
plan: position below box → push box up to row 2 → move right to next box
thought: Box at (3,2) needs to go to target at (1,2). I need to get below the \
box first. Moving down to row 4.
move: down"""

REFLECTION_PROMPT = """\
Analyze the last {n} steps of this Sokoban game:

{trajectory}

Current board:
{board}

Spatial analysis:
{spatial_analysis}

Answer each question in ONE line (total under 80 words):
1. PROGRESS: How many boxes moved closer to targets vs. farther? (+N/-N)
2. DEADLOCK: Any box now in a corner or against a wall without a target? (yes/no, which)
3. OSCILLATION: Am I repeating a cycle of moves? (yes/no)
4. PLAN: Which box should I push next, to which target, from which direction?"""

USER_PROMPT_TEMPLATE = """\
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
Choose ONE action from: up, down, left, right, push up, push down, push left, push right

plan: <your 3-5 step plan>
thought: <reasoning>
move: <action>"""


# ---------------------------------------------------------------------------
# Board parsing — convert the flat table back to a spatial grid
# ---------------------------------------------------------------------------

def table_obs_to_grid(obs_text: str) -> Optional[List[List[str]]]:
    """Parse the text-table observation back into a 2D character grid.

    The env produces a table like:
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
    col_header = "    " + "".join(f"{c}" for c in range(max_c))
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
# Memory & reflection
# ---------------------------------------------------------------------------

class SokobanMemory:
    """Rolling window of recent (action, board_summary, reward) tuples."""

    def __init__(self, max_steps: int = MAX_MEMORY_STEPS):
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.last_reflection: str = "Game just started. Survey the board and plan."

    def add(self, step: int, action: str, reward: float, board_summary: str):
        self.history.append({
            "step": step,
            "action": action,
            "reward": reward,
            "summary": board_summary,
        })
        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]

    def format_history(self) -> str:
        if not self.history:
            return "(no previous actions)"
        lines = []
        for h in self.history:
            r_str = f"{h['reward']:+.2f}"
            lines.append(f"  Step {h['step']}: {h['action']} → reward {r_str}")
        return "\n".join(lines)

    def format_trajectory_for_reflection(self) -> str:
        if not self.history:
            return "(no actions yet)"
        lines = []
        for h in self.history:
            lines.append(f"Step {h['step']}: action={h['action']}, reward={h['reward']:+.2f}")
            lines.append(f"  Board: {h['summary']}")
        return "\n".join(lines)


def get_reflection(
    memory: SokobanMemory,
    board_str: str,
    client_kw: Dict[str, Any],
    model: str,
    grid: Optional[List[List[str]]] = None,
) -> str:
    """Ask the model to reflect on recent history (cheap, short call)."""
    if len(memory.history) < 3:
        return memory.last_reflection

    spatial = "(not available)"
    if grid is not None and compute_spatial_analysis is not None:
        spatial = compute_spatial_analysis(grid)

    prompt = REFLECTION_PROMPT.format(
        n=len(memory.history),
        trajectory=memory.format_trajectory_for_reflection(),
        board=board_str,
        spatial_analysis=spatial,
    )

    try:
        client = openai.OpenAI(**client_kw)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Sokoban game analyst. Give brief, actionable reflections using the structured checklist."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        reflection = resp.choices[0].message.content.strip()
        memory.last_reflection = reflection
        return reflection
    except Exception:
        return memory.last_reflection


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "up", "down", "left", "right",
    "push up", "push down", "push left", "push right",
]


def _build_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the single best action for the current Sokoban board state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Step-by-step reasoning about which box to target, player positioning, and deadlock avoidance.",
                        },
                        "action": {
                            "type": "string",
                            "enum": VALID_ACTIONS,
                            "description": "The chosen action.",
                        },
                    },
                    "required": ["thought", "action"],
                },
            },
        }
    ]


def _extract_action_from_text(text: str) -> Optional[str]:
    """Best-effort extraction of a valid action from free-form model reply."""
    if not text:
        return None
    lower = text.lower()
    move_match = re.search(r"move:\s*(.+?)(?:\n|$)", lower)
    if move_match:
        candidate = move_match.group(1).strip()
        for v in VALID_ACTIONS:
            if v == candidate:
                return v
    for v in sorted(VALID_ACTIONS, key=len, reverse=True):
        if v in lower:
            return v
    return None


def sokoban_agent_action(
    grid: List[List[str]],
    memory: SokobanMemory,
    client_kw: Dict[str, Any],
    model: str,
    temperature: float = 0.4,
    do_reflect: bool = True,
) -> Tuple[str, Optional[str]]:
    """Query GPT-5.4 with Sokoban-specific prompts. Returns (action, reasoning)."""
    if openai is None:
        return "up", None

    board_str = grid_to_string(grid)
    element_summary = summarize_elements(grid)

    spatial = "(not available)"
    if compute_spatial_analysis is not None:
        spatial = compute_spatial_analysis(grid)

    reflection = memory.last_reflection
    if do_reflect and len(memory.history) >= 3:
        reflection = get_reflection(
            memory, board_str, client_kw, model, grid=grid,
        )

    user_content = USER_PROMPT_TEMPLATE.format(
        grid=board_str,
        element_summary=element_summary,
        spatial_analysis=spatial,
        n_history=len(memory.history),
        history=memory.format_history(),
        reflection=reflection,
    )

    tools = _build_tools()

    try:
        client = openai.OpenAI(**client_kw)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SOKOBAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            temperature=temperature,
            max_tokens=500,
        )
        msg = response.choices[0].message

        if msg.tool_calls and len(msg.tool_calls) > 0:
            tc = msg.tool_calls[0]
            raw_args = getattr(tc, "arguments", None) or getattr(tc.function, "arguments", None) or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

            action = args.get("action", "")
            thought = args.get("thought", "")

            if action and action.lower() in VALID_ACTIONS:
                return action.lower(), thought
            extracted = _extract_action_from_text(action)
            if extracted:
                return extracted, thought

        content = msg.content or ""
        extracted = _extract_action_from_text(content)
        if extracted:
            return extracted, content
        return "up", None

    except Exception as exc:
        print(f"    [WARN] Sokoban agent call failed ({exc}), using fallback")
        return "up", None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_sokoban_episode(
    env: ColdStartEnvWrapper,
    model: str = MODEL_GPT54,
    max_steps: int = 200,
    temperature: float = 0.4,
    reflect_every: int = 3,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one Sokoban episode with domain-specialized prompts and memory."""
    task = GAME_REGISTRY[GAME_NAME]["task"]
    action_names = GAME_REGISTRY[GAME_NAME]["action_names"]

    use_router = open_router_api_key and open_router_api_key.strip()
    client_kw: Dict[str, Any] = {}
    effective_model = model
    if use_router:
        client_kw = {"base_url": OPENROUTER_BASE, "api_key": open_router_api_key.strip()}
        effective_model = model if "/" in model else f"openai/{model}"
    elif openai_api_key:
        client_kw = {"api_key": openai_api_key}

    obs, info = env.reset()
    memory = SokobanMemory(max_steps=MAX_MEMORY_STEPS)
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        grid = table_obs_to_grid(obs)
        if grid is None:
            if verbose:
                print(f"    [WARN] Could not parse grid at step {step_count}, using raw obs")
            action, reasoning = "up", "Could not parse board"
        else:
            do_reflect = (step_count > 0 and step_count % reflect_every == 0)
            action, reasoning = sokoban_agent_action(
                grid=grid,
                memory=memory,
                client_kw=client_kw,
                model=effective_model,
                temperature=temperature,
                do_reflect=do_reflect,
            )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        board_summary = ""
        next_grid = table_obs_to_grid(next_obs)
        if next_grid:
            board_summary = summarize_elements(next_grid)
        memory.add(step_count, action, reward, board_summary)

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=task,
        )
        exp.idx = step_count - 1
        exp.action_type = "primitive"
        exp.raw_state = str(info.get("raw_obs")) if info.get("raw_obs") is not None else None
        exp.raw_next_state = str(next_info.get("raw_obs")) if next_info.get("raw_obs") is not None else None
        exp.available_actions = list(action_names)
        exp.interface = {"env_name": "gamingagent", "game_name": GAME_NAME}
        experiences.append(exp)

        if verbose:
            reason_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            print(f"  step {step_count}: action={action}, reward={reward:.2f}, "
                  f"cum={total_reward:.2f}, reason={reason_short}")

        obs = next_obs
        info = next_info

        if done:
            break

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=GAME_NAME,
    )
    episode.set_outcome()

    stats = {
        "game": GAME_NAME,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "gpt54_sokoban_specialized",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch rollout helpers (reused from gpt54 script patterns)
# ---------------------------------------------------------------------------

def count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")


def save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_all_episodes(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Run all Sokoban episodes and save outputs."""
    game_dir = output_dir / GAME_NAME
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {GAME_NAME}: {start_idx}/{args.episodes} episodes already done")
            return {"game": GAME_NAME, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {GAME_NAME}: resuming from episode {start_idx}")

    effective_max_steps = args.max_steps if args.max_steps is not None else get_cold_start_max_steps(GAME_NAME)

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [sokoban] Episode {ep_idx + 1}/{args.episodes}")

        try:
            env = ColdStartEnvWrapper(GAME_NAME, max_steps=effective_max_steps)
            episode, stats = run_sokoban_episode(
                env=env,
                model=args.model,
                max_steps=effective_max_steps,
                temperature=args.temperature,
                reflect_every=args.reflect_every,
                verbose=args.verbose,
            )
            env.close()

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            if not args.no_label:
                episode = label_trajectory(episode, args.label_model)

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": GAME_NAME,
                "episode_index": ep_idx,
                "error": str(e),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed = time.time() - t0

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary: Dict[str, Any] = {
        "game": GAME_NAME,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "gpt54_sokoban_specialized",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": effective_max_steps,
        "reflect_every": args.reflect_every,
        "memory_window": MAX_MEMORY_STEPS,
        "labeled": not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    if all_stats:
        rewards = [s.get("total_reward", 0.0) for s in all_stats if "error" not in s]
        steps = [s.get("steps", 0) for s in all_stats if "error" not in s]
        if rewards:
            summary["mean_reward"] = sum(rewards) / len(rewards)
            summary["mean_steps"] = sum(steps) / len(steps)
            summary["max_reward"] = max(rewards)
            summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sokoban-specialized GPT-5.4 cold-start rollouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=60,
                        help="Number of episodes (default: 60)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: per-game natural end = 200)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model (default: {MODEL_GPT54})")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    parser.add_argument("--reflect_every", type=int, default=3,
                        help="Run reflection every N steps (default: 3)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini",
                        help="Model for trajectory labeling (default: gpt-5-mini)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/output/gpt54_sokoban)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54_sokoban"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check API keys
    try:
        from api_keys import open_router_api_key as _or_key
        has_key = bool(
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or (_or_key and _or_key.strip())
        )
    except Exception:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        print("[WARNING] No API key set. LLM calls will fail.")

    # Verify sokoban is available
    if GAME_NAME not in GAME_REGISTRY:
        print(f"[ERROR] '{GAME_NAME}' not in game registry.")
        sys.exit(1)
    if GAME_REGISTRY[GAME_NAME]["env_class"] is None:
        print(f"[ERROR] Sokoban env class not importable.")
        sys.exit(1)

    effective_max_steps = args.max_steps if args.max_steps is not None else get_cold_start_max_steps(GAME_NAME)

    print("=" * 78)
    print("  Sokoban-Specialized GPT-5.4 — Cold-Start Rollout Generation")
    print("=" * 78)
    print(f"  Episodes:       {args.episodes}")
    print(f"  Max steps:      {effective_max_steps}")
    print(f"  Model:          {args.model}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Reflect every:  {args.reflect_every} steps")
    print(f"  Memory window:  {MAX_MEMORY_STEPS} steps")
    print(f"  Labeling:       {not args.no_label} (label model: {args.label_model})")
    print(f"  Resume:         {args.resume}")
    print(f"  Output:         {output_dir}")
    print("=" * 78)

    summary = run_all_episodes(args, output_dir)

    print(f"\n{'=' * 78}")
    print("  SOKOBAN SPECIALIZED ROLLOUTS — COMPLETE")
    print(f"{'=' * 78}")
    if not summary.get("skipped"):
        print(f"  Episodes:    {summary.get('total_episodes', 0)}")
        print(f"  Elapsed:     {summary.get('elapsed_seconds', 0):.1f}s")
        if "mean_reward" in summary:
            print(f"  Avg reward:  {summary['mean_reward']:.2f}")
            print(f"  Max reward:  {summary['max_reward']:.2f}")
            print(f"  Avg steps:   {summary['mean_steps']:.1f}")
    print(f"  Output:      {output_dir / GAME_NAME}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
