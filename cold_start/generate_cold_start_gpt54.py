#!/usr/bin/env python
"""
Cold-start base agent using GPT-5.4 for LM-Game Bench (GamingAgent).

Generates decision-making trajectories using GPT-5.4 as the backbone model.
The base agent uses structured chain-of-thought reasoning before each action,
producing richer Experience data (with reasoning traces stored in intentions)
suitable for downstream skill extraction and co-evolution training.

Output structure (cold_start/output/gpt54/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # All available games, 100 episodes each
    python cold_start/generate_cold_start_gpt54.py

    # Specific game(s)
    python cold_start/generate_cold_start_gpt54.py --games tic_tac_toe sokoban tetris

    # Fewer episodes (for testing)
    python cold_start/generate_cold_start_gpt54.py --episodes 5 --max_steps 30

    # Resume an interrupted run
    python cold_start/generate_cold_start_gpt54.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# GPT-5.4 model constant
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"

# ---------------------------------------------------------------------------
# System prompt: enhanced for GPT-5.4 with chain-of-thought reasoning
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert game-playing agent competing in LM-Game Bench.\n"
    "You receive a textual description of the current game state and must choose exactly one action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Key elements of the current state (positions, scores, threats, opportunities).\n"
    "2. What your goal is and which sub-goal to pursue now.\n"
    "3. How each candidate action moves you toward that goal.\n\n"
    "IMPORTANT: If your recent action history shows that an action had NO EFFECT on the\n"
    "game state (marked as 'NO EFFECT'), you MUST choose a DIFFERENT action. Repeating a\n"
    "no-effect action wastes your limited steps.\n\n"
    "Then call the `choose_action` function with your chosen action.\n"
    "Use the EXACT action name from the valid actions list."
)

USER_TEMPLATE = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "{history_block}"
    "Think step-by-step, then choose one action."
)

# Max number of recent history entries to include in the prompt
_HISTORY_WINDOW = 5
# Force a random different action after this many consecutive no-ops
_MAX_CONSECUTIVE_NOOPS = 2


# ---------------------------------------------------------------------------
# Action extraction helpers (mirrors dummy_agent patterns)
# ---------------------------------------------------------------------------

def _parse_valid_actions(state_nl: str) -> List[str]:
    """Extract valid action names from 'Valid actions: a, b, c' in state."""
    m = re.search(r"[Vv]alid\s+actions?\s*[:\-]\s*(.+?)(?:\n|\.|$)", state_nl)
    if not m:
        return []
    raw = m.group(1).strip()
    return [a.strip().lower() for a in re.split(r"[,;]", raw) if a.strip()]


def _extract_action(text: str, valid_actions: List[str]) -> Optional[str]:
    """Best-effort extraction of a valid action from model reply."""
    if not text:
        return None
    reply = text.strip().lower()
    words = re.findall(r"[\w_]+", reply)
    for w in words:
        for v in valid_actions:
            if w == v.lower():
                return v
    for w in words:
        for v in valid_actions:
            if len(w) >= 2 and v.lower().startswith(w):
                return v
    return valid_actions[0] if valid_actions else None


# ---------------------------------------------------------------------------
# GPT-5.4 function-calling agent
# ---------------------------------------------------------------------------

def _build_tools(action_names: List[str]) -> list:
    """Build OpenAI function-calling tool definition with the game's valid actions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the single action for your turn in the game.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Brief chain-of-thought reasoning for this action.",
                        },
                        "action": {
                            "type": "string",
                            "description": f"One of the valid actions: {', '.join(action_names)}",
                        },
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _format_history_block(recent_history: List[Dict[str, Any]]) -> str:
    """Format recent action history into a prompt block."""
    if not recent_history:
        return ""
    lines = ["Recent action history (newest first):"]
    for entry in reversed(recent_history[-_HISTORY_WINDOW:]):
        effect = "NO EFFECT" if entry["noop"] else f"reward {entry['reward']:+.1f}"
        lines.append(f"  - Action '{entry['action']}' -> {effect}")
    noop_actions = {e["action"] for e in recent_history[-_HISTORY_WINDOW:] if e["noop"]}
    if noop_actions:
        lines.append(
            f"WARNING: Action(s) {noop_actions} had no effect. Choose a DIFFERENT action!\n"
        )
    else:
        lines.append("")
    return "\n".join(lines) + "\n"


def agent_action(
    state_nl: str,
    action_names: List[str],
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
    recent_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[str]]:
    """Query LLM with function calling. Returns (action, reasoning)."""
    use_router = open_router_api_key and open_router_api_key.strip()
    if openai is None or (not use_router and openai_api_key is None):
        return action_names[0] if action_names else "stay", None

    client_kw: Dict[str, Any] = {}
    effective_model = model
    if use_router:
        client_kw = {"base_url": OPENROUTER_BASE, "api_key": open_router_api_key.strip()}
        effective_model = model if "/" in model else f"openai/{model}"
    else:
        client_kw = {"api_key": openai_api_key}

    history_block = _format_history_block(recent_history or [])
    user_content = USER_TEMPLATE.format(
        state=state_nl,
        actions=", ".join(action_names),
        history_block=history_block,
    )
    tools = _build_tools(action_names)

    try:
        client = openai.OpenAI(**client_kw)
        response = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            temperature=temperature,
            max_tokens=400,
        )
        msg = response.choices[0].message

        if msg.tool_calls and len(msg.tool_calls) > 0:
            tc = msg.tool_calls[0]
            raw_args = getattr(tc, "arguments", None) or getattr(tc.function, "arguments", None) or "{}"
            if isinstance(raw_args, str):
                args = json.loads(raw_args)
            else:
                args = raw_args or {}

            action = args.get("action", "")
            reasoning = args.get("reasoning")

            if action:
                lower_map = {a.lower(): a for a in action_names}
                canonical = lower_map.get(action.lower().strip())
                if canonical:
                    return canonical, reasoning
                extracted = _extract_action(action, action_names)
                if extracted:
                    return extracted, reasoning

        content = msg.content or ""
        extracted = _extract_action(content, action_names)
        if extracted:
            return extracted, None
        return (action_names[0] if action_names else "stay"), None

    except Exception as exc:
        print(f"    [WARN] LLM call failed ({exc}), using fallback")
        return (action_names[0] if action_names else "stay"), None


# Keep old name as alias for backward compatibility
gpt54_agent_action = agent_action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _pick_different_action(
    last_action: str, action_names: List[str]
) -> str:
    """Pick a random action that differs from ``last_action``."""
    alternatives = [a for a in action_names if a != last_action]
    return random.choice(alternatives) if alternatives else last_action


def run_gpt54_episode(
    env: ColdStartEnvWrapper,
    game_name: str,
    model: str = MODEL_GPT54,
    max_steps: int = 50,
    temperature: float = 0.4,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with the base agent (supports any model via --model)."""
    task = GAME_REGISTRY[game_name]["task"]
    action_names = GAME_REGISTRY[game_name]["action_names"]

    obs, info = env.reset(seed=seed)
    raw_obs = info.get("raw_obs")
    curr_available_actions = info.get("available_actions") or action_names
    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    recent_history: List[Dict[str, Any]] = []
    consecutive_noops = 0
    last_noop_action: Optional[str] = None

    while step_count < max_steps:
        step_actions = curr_available_actions if curr_available_actions else action_names
        prompt_state = obs + f"\n\nValid actions: {', '.join(step_actions)}. Choose one."

        action, reasoning = agent_action(
            state_nl=prompt_state,
            action_names=step_actions,
            model=model,
            temperature=temperature,
            recent_history=recent_history,
        )

        # Safety valve: if the LLM keeps returning the same no-op action,
        # override with a random different one.
        if (
            consecutive_noops >= _MAX_CONSECUTIVE_NOOPS
            and action == last_noop_action
            and len(step_actions) > 1
        ):
            old = action
            action = _pick_different_action(action, step_actions)
            reasoning = (reasoning or "") + f" [auto-override: '{old}' had no effect {consecutive_noops}x]"
            if verbose:
                print(f"  step {step_count}: OVERRIDE {old} -> {action} (no-op x{consecutive_noops})")

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_raw_obs = next_info.get("raw_obs")
        done = terminated or truncated
        total_reward += reward

        is_noop = (obs == next_obs and reward == 0.0)

        recent_history.append({
            "action": action,
            "reward": reward,
            "noop": is_noop,
        })

        if is_noop and action == last_noop_action:
            consecutive_noops += 1
        elif is_noop:
            consecutive_noops = 1
            last_noop_action = action
        else:
            consecutive_noops = 0
            last_noop_action = None

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.raw_state = str(raw_obs) if raw_obs is not None else None
        exp.raw_next_state = str(next_raw_obs) if next_raw_obs is not None else None
        exp.available_actions = list(step_actions) if step_actions else None
        exp.interface = {"env_name": "gamingagent", "game_name": game_name}
        experiences.append(exp)

        if verbose:
            noop_tag = " [NOOP]" if is_noop else ""
            reason_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            print(f"  step {step_count}: action={action}, reward={reward:.2f}, "
                  f"cum={total_reward:.2f}{noop_tag}, reason={reason_short}")

        obs = next_obs
        raw_obs = next_raw_obs
        curr_available_actions = next_info.get("available_actions") or action_names
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

    noop_count = sum(1 for h in recent_history if h["noop"])
    stats = {
        "game": game_name,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "gpt54_base",
        "noop_steps": noop_count,
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch rollout helpers
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


def save_game_summary(
    game_name: str,
    game_dir: Path,
    all_stats: List[Dict[str, Any]],
    elapsed: float,
    args: argparse.Namespace,
    max_steps_used: Optional[int] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "gpt54_base",
        "total_episodes": len(all_stats),
        "target_episodes": args.episodes,
        "max_steps": max_steps_used if max_steps_used is not None else args.max_steps,
        "labeled": not args.no_label,
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def run_game_rollouts(
    game_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes for one game and save outputs."""
    game_dir = output_dir / game_name
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {game_name}: {start_idx}/{args.episodes} episodes already done")
            return {"game": game_name, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {game_name}: resuming from episode {start_idx}")

    effective_max_steps = args.max_steps if args.max_steps is not None else get_cold_start_max_steps(game_name)

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            env = ColdStartEnvWrapper(game_name, max_steps=effective_max_steps)
            episode, stats = run_gpt54_episode(
                env=env,
                game_name=game_name,
                model=args.model,
                max_steps=effective_max_steps,
                temperature=args.temperature,
                verbose=args.verbose,
                seed=42 + ep_idx,
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
                "game": game_name,
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

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args, max_steps_used=effective_max_steps)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.4 base agent cold-start rollouts for LM-Game Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Games to generate rollouts for (default: all available)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per game (default: 100)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: per-game natural end)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model for the agent (default: {MODEL_GPT54})")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling (faster, unlabeled only)")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini",
                        help="Model used for trajectory labeling (default: gpt-5-mini)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run (skip completed episodes)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/output/gpt54)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        print("  Prefer: open_router_api_key in api_keys.py or export OPENROUTER_API_KEY='sk-or-...'")
        print("  Or: export OPENAI_API_KEY='sk-...'")

    if args.games:
        requested = args.games
    else:
        requested = list(GAME_REGISTRY.keys())

    available_games: List[str] = []
    skipped_games: List[str] = []
    for g in requested:
        if g not in GAME_REGISTRY:
            print(f"[WARNING] Game '{g}' not in registry, skipping.")
            skipped_games.append(g)
            continue
        if GAME_REGISTRY[g]["env_class"] is None:
            print(f"[WARNING] Game '{g}' env class not importable, skipping.")
            skipped_games.append(g)
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available. Ensure GamingAgent is installed.")
        sys.exit(1)

    print("=" * 78)
    print("  GPT-5.4 Base Agent — Cold-Start Rollout Generation")
    print("=" * 78)
    print(f"  Games:       {', '.join(available_games)}")
    if skipped_games:
        print(f"  Skipped:     {', '.join(skipped_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  Max steps:   {'per-game (natural end)' if args.max_steps is None else args.max_steps}")
    print(f"  Model:       {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Labeling:    {not args.no_label} (label model: {args.label_model})")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in available_games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game_name} ({args.episodes} episodes)")
        print(f"{'━' * 78}")

        summary = run_game_rollouts(game_name, args, output_dir)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "gpt54_base",
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,  # None means per-game natural end
        "temperature": args.temperature,
        "labeled": not args.no_label,
        "label_model": args.label_model,
        "total_elapsed_seconds": overall_elapsed,
        "games_completed": list(available_games),
        "games_skipped": skipped_games,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  GPT-5.4 BASE AGENT — BATCH ROLLOUT COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Games processed: {len(available_games)}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped")
    )
    print(f"  Total episodes:  {total_eps}")
    print(f"  Elapsed:         {overall_elapsed:.1f}s")
    print(f"  Output:          {output_dir}")
    print(f"  Master summary:  {master_path}")

    successful = [s for s in game_summaries if not s.get("skipped") and "mean_reward" in s]
    if successful:
        avg_reward = sum(s["mean_reward"] for s in successful) / len(successful)
        avg_steps = sum(s["mean_steps"] for s in successful) / len(successful)
        print(f"  Avg reward:      {avg_reward:.2f}")
        print(f"  Avg steps:       {avg_steps:.1f}")

    print(f"{'=' * 78}")
    print()
    print("  Load into skill pipeline:")
    print("    from cold_start.load_rollouts import load_all_game_rollouts")
    print(f"    rollouts = load_all_game_rollouts('{output_dir}')")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<game>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
