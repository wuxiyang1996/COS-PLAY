#!/usr/bin/env python
"""
Cold-start rollout generation for Orak games (Super Mario, StarCraft II)
using GPT-5.4 with structured chain-of-thought reasoning.

Uses the evaluate_orak.orak_nl_wrapper env wrappers directly.  Output format
matches the rest of cold_start/output/ so data feeds into the skill pipeline
and co-evolution trainer.

Output structure (cold_start/output/gpt54_orak/<game_name>/):
  - episode_NNN.json        Individual episode (Episode.to_dict())
  - episode_buffer.json     All episodes in Episode_Buffer format
  - rollouts.jsonl          Append-friendly JSONL (one Episode per line)
  - rollout_summary.json    Per-game run stats

Usage (from Game-AI-Agent root):

    # --- Super Mario (activate orak-mario first) ---
    source evaluate_orak/setup_orak_mario.sh
    python cold_start/generate_cold_start_orak.py --games super_mario --episodes 5

    # --- StarCraft II (activate orak-sc2 first) ---
    source evaluate_orak/setup_orak_sc2.sh
    python cold_start/generate_cold_start_orak.py --games star_craft --episodes 3

    # Both games (needs matching conda env for each)
    python cold_start/generate_cold_start_orak.py --games super_mario star_craft

    # Quick test
    python cold_start/generate_cold_start_orak.py --games super_mario \\
        --episodes 1 --max_steps 5 -v

    # Resume an interrupted run
    python cold_start/generate_cold_start_orak.py --games star_craft --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
ORAK_SRC = CODEBASE_ROOT.parent / "Orak" / "src"

for _p in [str(CODEBASE_ROOT), str(ORAK_SRC)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

# Quiet SC2 library logs (protocol, controller, main) so cold-start runs are readable
logging.getLogger("sc2").setLevel(logging.WARNING)
try:
    import loguru
    loguru.logger.disable("sc2")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from evaluate_orak.orak_nl_wrapper import make_orak_env, ORAK_GAMES
from data_structure.experience import Experience, Episode, Episode_Buffer

try:
    from cold_start.generate_cold_start import label_trajectory
except ImportError:
    label_trajectory = None

try:
    import openai
    from api_keys import openai_api_key, open_router_api_key
except (ImportError, AttributeError):
    openai = None
    openai_api_key = None  # type: ignore[assignment]
    open_router_api_key = None  # type: ignore[assignment]

try:
    from API_func import OPENROUTER_BASE
except ImportError:
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"

ORAK_COLD_START_GAMES: Dict[str, Dict[str, Any]] = {
    "super_mario": {
        "max_steps": 100,
        "display_name": "Super Mario Bros",
        "task": ORAK_GAMES["super_mario"]["task"],
    },
    "star_craft": {
        "max_steps": 1000,
        "display_name": "StarCraft II (1v1 Protoss)",
        "task": ORAK_GAMES["star_craft"]["task"],
    },
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_MARIO = (
    "You are an expert game-playing agent powered by GPT-5.4, competing in the "
    "Orak benchmark.\n"
    "You receive a textual description of the current Super Mario game state and "
    "must choose exactly one action.\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Mario's position and nearby objects (enemies, blocks, pipes, pits).\n"
    "2. What your immediate sub-goal should be (dodge enemy, jump gap, advance).\n"
    "3. Which jump level best achieves that goal.\n\n"
    "Then call the `choose_action` function with your chosen action.\n"
    "Use the EXACT action name from the valid actions list."
)

_SYSTEM_PROMPT_SC2 = (
    "You are an expert StarCraft II agent powered by GPT-5.4, playing Protoss "
    "vs Zerg in the Orak benchmark.\n"
    "Each turn you must provide EXACTLY 5 sequential macro actions.\n\n"
    "Action categories:\n"
    "  TRAIN: PROBE, ZEALOT, ADEPT, STALKER, SENTRY, HIGHTEMPLAR, DARKTEMPLAR, "
    "VOIDRAY, CARRIER, TEMPEST, ORACLE, PHOENIX, MOTHERSHIP, OBSERVER, IMMORTAL, "
    "WARPPRISM, COLOSSUS, DISRUPTOR  |  MORPH ARCHON\n"
    "  BUILD: PYLON, ASSIMILATOR, NEXUS, GATEWAY, CYBERNETICSCORE, FORGE, "
    "TWILIGHTCOUNCIL, ROBOTICSFACILITY, STARGATE, TEMPLARARCHIVE, DARKSHRINE, "
    "ROBOTICSBAY, FLEETBEACON, PHOTONCANNON, SHIELDBATTERY\n"
    "  RESEARCH: WARPGATERESEARCH, CHARGE, BLINKTECH, ADEPTPIERCINGATTACK, "
    "PSISTORMTECH, EXTENDEDTHERMALLANCE, GRAVITICDRIVE, OBSERVERGRAVITICBOOSTER, "
    "ground/air weapons/armor levels 1-3, shields levels 1-3, etc.\n"
    "  SCOUTING: PROBE, OBSERVER, ZEALOT, PHOENIX\n"
    "  CHRONOBOOST: NEXUS, CYBERNETICSCORE, TWILIGHTCOUNCIL, STARGATE, FORGE\n"
    "  MILITARY: MULTI-ATTACK, MULTI-RETREAT\n"
    "  EMPTY ACTION (no-op)\n\n"
    "Before choosing, briefly reason about:\n"
    "1. Current economy (minerals, gas, supply), army composition, and "
    "opponent threat level.\n"
    "2. Which build order phase you are in and what the next priority is.\n"
    "3. Whether to focus on economy, tech, or aggression this turn.\n\n"
    "Then call the `choose_actions` function with a list of 5 actions.\n"
    "Use EXACT action names from the valid actions list."
)

_USER_TEMPLATE = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "Think step-by-step, then choose one action."
)

_USER_TEMPLATE_SC2 = (
    "Game state:\n\n{state}\n\n"
    "Valid actions: {actions}\n\n"
    "Think step-by-step, then choose exactly 5 sequential macro actions."
)


# ---------------------------------------------------------------------------
# OpenAI client helpers
# ---------------------------------------------------------------------------

def _make_client() -> Optional["openai.OpenAI"]:
    if openai is None:
        return None
    use_router = open_router_api_key and str(open_router_api_key).strip()
    if use_router:
        return openai.OpenAI(base_url=OPENROUTER_BASE, api_key=str(open_router_api_key).strip())
    if openai_api_key:
        return openai.OpenAI(api_key=openai_api_key)
    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        kw: Dict[str, Any] = {"api_key": env_key}
        if os.environ.get("OPENROUTER_API_KEY"):
            kw["base_url"] = OPENROUTER_BASE
        return openai.OpenAI(**kw)
    return None


def _effective_model(model: str) -> str:
    if open_router_api_key and str(open_router_api_key).strip():
        return model if "/" in model else f"openai/{model}"
    return model


def _canonicalize(raw: str, action_names: List[str]) -> str:
    lower_map = {a.lower().strip(): a for a in action_names}
    hit = lower_map.get(raw.lower().strip())
    if hit:
        return hit
    for v in action_names:
        if v.lower() in raw.lower():
            return v
    return action_names[0] if action_names else raw


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _build_tools_single(action_names: List[str]) -> list:
    return [{
        "type": "function",
        "function": {
            "name": "choose_action",
            "description": "Choose the single action for your turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief chain-of-thought reasoning.",
                    },
                    "action": {
                        "type": "string",
                        "description": f"One of: {', '.join(action_names)}",
                    },
                },
                "required": ["action"],
            },
        },
    }]


def _build_tools_sc2(action_names: List[str]) -> list:
    desc = "A valid Protoss macro action from the valid actions list."
    return [{
        "type": "function",
        "function": {
            "name": "choose_actions",
            "description": "Choose exactly 5 sequential macro actions for this turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Brief chain-of-thought reasoning about economy, army, and priorities."},
                    "action_1": {"type": "string", "description": desc},
                    "action_2": {"type": "string", "description": desc},
                    "action_3": {"type": "string", "description": desc},
                    "action_4": {"type": "string", "description": desc},
                    "action_5": {"type": "string", "description": desc},
                },
                "required": ["action_1", "action_2", "action_3", "action_4", "action_5"],
            },
        },
    }]


# ---------------------------------------------------------------------------
# Safe tool-call argument parsing (handles truncated/malformed JSON from LLM)
# ---------------------------------------------------------------------------

def _parse_tool_args(raw: str) -> Dict[str, Any]:
    """Parse tool call arguments JSON; on failure try to fix or extract known keys."""
    if not raw or not isinstance(raw, str):
        return {}
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to fix unterminated string: close string and object if we're inside "..."
    try:
        in_str = raw.rfind('"')
        if in_str > 0 and raw.count('"') % 2 == 1:
            attempt = raw + '"}' if not raw.rstrip().endswith("}") else raw + '"'
            return json.loads(attempt)
    except json.JSONDecodeError:
        pass
    # Extract "key": "value" for known keys (handles truncated/malformed JSON)
    out: Dict[str, Any] = {}
    for key in ["reasoning", "action", "action_1", "action_2", "action_3", "action_4", "action_5"]:
        pat = re.compile(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
        m = pat.search(raw)
        if m:
            out[key] = m.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
    return out


# ---------------------------------------------------------------------------
# GPT-5.4 agent action functions
# ---------------------------------------------------------------------------

def gpt54_mario_action(
    state_nl: str,
    action_names: List[str],
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
) -> Tuple[str, Optional[str]]:
    client = _make_client()
    if client is None:
        return action_names[0] if action_names else "Jump Level: 0", None
    tools = _build_tools_single(action_names)
    user = _USER_TEMPLATE.format(state=state_nl, actions=", ".join(action_names))
    try:
        resp = client.chat.completions.create(
            model=_effective_model(model),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_MARIO},
                {"role": "user", "content": user},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            temperature=temperature,
            max_tokens=400,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            raw = getattr(msg.tool_calls[0], "arguments", None) or \
                  getattr(msg.tool_calls[0].function, "arguments", None) or "{}"
            args = _parse_tool_args(raw) if isinstance(raw, str) else (raw or {})
            return _canonicalize(args.get("action", ""), action_names), args.get("reasoning")
        return _canonicalize(msg.content or "", action_names), None
    except Exception as exc:
        print(f"    [WARN] GPT-5.4 Mario call failed ({exc}), fallback")
        return action_names[0] if action_names else "Jump Level: 0", None


def gpt54_sc2_action(
    state_nl: str,
    action_names: List[str],
    model: str = MODEL_GPT54,
    temperature: float = 0.4,
) -> Tuple[str, Optional[str]]:
    client = _make_client()
    default = "EMPTY ACTION"
    if client is None:
        return "\n".join(f"{i}: {default}" for i in range(1, 6)), None
    tools = _build_tools_sc2(action_names)
    user = _USER_TEMPLATE_SC2.format(state=state_nl, actions=", ".join(action_names))
    try:
        resp = client.chat.completions.create(
            model=_effective_model(model),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_SC2},
                {"role": "user", "content": user},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_actions"}},
            temperature=temperature,
            max_tokens=600,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            raw = getattr(msg.tool_calls[0], "arguments", None) or \
                  getattr(msg.tool_calls[0].function, "arguments", None) or "{}"
            args = _parse_tool_args(raw) if isinstance(raw, str) else (raw or {})
            reasoning = args.get("reasoning")
            acts = [_canonicalize(args.get(f"action_{i}", default), action_names) for i in range(1, 6)]
            return "\n".join(f"{i}: {a}" for i, a in enumerate(acts, 1)), reasoning
        content = msg.content or ""
        return content.strip() or "\n".join(f"{i}: {default}" for i in range(1, 6)), None
    except Exception as exc:
        print(f"    [WARN] SC2 call failed ({exc}), fallback")
        return "\n".join(f"{i}: {default}" for i in range(1, 6)), None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_orak_episode(
    game_name: str,
    model: str = MODEL_GPT54,
    max_steps: int = 100,
    temperature: float = 0.4,
    verbose: bool = False,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode of an Orak game with the GPT-5.4 agent."""
    cfg = ORAK_COLD_START_GAMES[game_name]
    task = cfg["task"]
    is_sc2 = game_name in ("star_craft", "star_craft_multi")

    agent_fn = gpt54_sc2_action if is_sc2 else gpt54_mario_action

    env = make_orak_env(game_name, max_steps=max_steps)
    action_names = env.action_names
    obs, info = env.reset()

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps:
        action, reasoning = agent_fn(
            state_nl=obs,
            action_names=action_names,
            model=model,
            temperature=temperature,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

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
        exp.raw_state = obs
        exp.raw_next_state = next_obs
        exp.available_actions = list(action_names)

        step_info = next_info if isinstance(next_info, dict) else {}
        exp.interface = {
            "env_name": "orak",
            "game_name": game_name,
            "step": step_count,
            "terminated": terminated,
            "truncated": truncated,
            "score": step_info.get("score"),
            "cumulative_reward": total_reward,
        }
        experiences.append(exp)

        if verbose:
            act_short = str(action)[:60]
            reason_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            term_label = "TERM" if terminated else ("TRUNC" if truncated else "")
            print(f"  step {step_count}: action={act_short}, reward={reward:.3f}, "
                  f"cum={total_reward:.3f}, {term_label} reason={reason_short}")

        obs = next_obs
        step_count += 1
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="orak",
        game_name=game_name,
    )
    episode.set_outcome()

    final_info = next_info if (step_count > 0 and isinstance(next_info, dict)) else {}
    stats = {
        "game": game_name,
        "display_name": cfg["display_name"],
        "steps": step_count,
        "total_reward": total_reward,
        "final_score": final_info.get("score", 0),
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "gpt54_orak",
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
) -> Dict[str, Any]:
    good = [s for s in all_stats if "error" not in s]
    summary: Dict[str, Any] = {
        "game": game_name,
        "display_name": ORAK_COLD_START_GAMES[game_name]["display_name"],
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "gpt54_orak",
        "total_episodes": len(good),
        "target_episodes": args.episodes,
        "max_steps": args.max_steps,
        "labeled": args.label and not args.no_label,
        "elapsed_seconds": round(elapsed, 2),
        "episode_stats": all_stats,
    }
    if good:
        rewards = [s["total_reward"] for s in good]
        steps = [s["steps"] for s in good]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def _save_episode_result(
    episode: Episode,
    stats: Dict[str, Any],
    ep_idx: int,
    game_dir: Path,
    jsonl_path: Path,
    episode_buffer: Episode_Buffer,
    io_lock: Lock,
    label: bool = False,
    label_model: str = "gpt-5-mini",
):
    """Save a completed episode to disk (thread-safe)."""
    if label and label_trajectory is not None:
        episode = label_trajectory(episode, label_model)

    ep_data = episode.to_dict()
    ep_data["metadata"] = stats
    ep_path = game_dir / f"episode_{ep_idx:03d}.json"

    with io_lock:
        episode_buffer.add_episode(episode)
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
        save_episode_jsonl(episode, jsonl_path, stats)


def run_game_rollouts(
    game_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run all episodes for one game and save outputs."""
    game_dir = output_dir / game_name
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    max_steps = args.max_steps or ORAK_COLD_START_GAMES[game_name]["max_steps"]

    start_idx = 0
    if args.resume:
        start_idx = count_existing_episodes(game_dir)
        if start_idx >= args.episodes:
            print(f"  [SKIP] {game_name}: {start_idx}/{args.episodes} episodes already done")
            return {"game": game_name, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {game_name}: resuming from episode {start_idx}")

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    workers = getattr(args, "workers", 1) or 1
    # StarCraft II env spawns one SC2 process per episode; multiple instances hang (port/single-binary).
    if game_name in ("star_craft", "star_craft_multi") and workers > 1:
        print(f"  [NOTE] StarCraft II uses 1 worker only (multiple SC2 instances would hang).")
        workers = 1
    io_lock = Lock()

    if workers > 1:
        # ── Parallel episode execution ──────────────────────────────────
        display = ORAK_COLD_START_GAMES[game_name]["display_name"]
        remaining = list(range(start_idx, args.episodes))
        print(f"\n  Running {len(remaining)} episodes with {workers} parallel workers ...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: Dict[Any, int] = {}
            for ep_idx in remaining:
                future = executor.submit(
                    run_orak_episode,
                    game_name=game_name,
                    model=args.model,
                    max_steps=max_steps,
                    temperature=args.temperature,
                    verbose=args.verbose,
                )
                futures[future] = ep_idx

            for future in as_completed(futures):
                ep_idx = futures[future]
                try:
                    episode, stats = future.result()
                    stats["episode_index"] = ep_idx
                    print(f"  [{display}] Episode {ep_idx + 1}/{args.episodes} done — "
                          f"Steps: {stats['steps']}, Reward: {stats['total_reward']:.3f}")
                    _save_episode_result(
                        episode, stats, ep_idx, game_dir, jsonl_path,
                        episode_buffer, io_lock,
                        label=args.label and not args.no_label,
                        label_model=getattr(args, "label_model", "gpt-5-mini"),
                    )
                    all_stats.append(stats)
                except Exception as e:
                    print(f"  [{display}] Episode {ep_idx + 1}/{args.episodes} FAILED: {e}")
                    traceback.print_exc()
                    all_stats.append({
                        "game": game_name,
                        "episode_index": ep_idx,
                        "error": str(e),
                        "steps": 0,
                        "total_reward": 0.0,
                    })
    else:
        # ── Sequential episode execution (original) ────────────────────
        for ep_idx in range(start_idx, args.episodes):
            display = ORAK_COLD_START_GAMES[game_name]["display_name"]
            print(f"\n  [{display}] Episode {ep_idx + 1}/{args.episodes}")

            try:
                episode, stats = run_orak_episode(
                    game_name=game_name,
                    model=args.model,
                    max_steps=max_steps,
                    temperature=args.temperature,
                    verbose=args.verbose,
                )
                stats["episode_index"] = ep_idx
                print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.3f}")

                if args.label and not args.no_label and label_trajectory is not None:
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

    summary = save_game_summary(game_name, game_dir, all_stats, elapsed, args)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.4 cold-start rollouts for Orak games (Super Mario, StarCraft II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Super Mario, 5 episodes (activate orak-mario env first)
  python cold_start/generate_cold_start_orak.py --games super_mario --episodes 5

  # StarCraft II, 3 episodes (activate orak-sc2 env first)
  python cold_start/generate_cold_start_orak.py --games star_craft --episodes 3

  # Quick test
  python cold_start/generate_cold_start_orak.py --games super_mario --episodes 1 --max_steps 5 -v

  # Resume interrupted run
  python cold_start/generate_cold_start_orak.py --games star_craft --resume
""",
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        choices=list(ORAK_COLD_START_GAMES.keys()),
                        help="Orak games to run (default: all)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per game (default: 10)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (None = game default: Mario=100, SC2=1000)")
    parser.add_argument("--model", type=str, default=MODEL_GPT54,
                        help=f"LLM model (default: {MODEL_GPT54})")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default: 0.4)")
    parser.add_argument("--label", action="store_true",
                        help="Label trajectories with LLM (default: off; use labeling/ for that)")
    parser.add_argument("--no_label", action="store_true",
                        help="Skip trajectory labeling")
    parser.add_argument("--label_model", type=str, default="gpt-5-mini",
                        help="Model for labeling (default: gpt-5-mini)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted run (skip completed episodes)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel episode workers (default: 1). "
                             "Each worker runs its own SC2 instance.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: cold_start/output/gpt54_orak)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54_orak"
    output_dir.mkdir(parents=True, exist_ok=True)

    if label_trajectory is None and args.label and not args.no_label:
        print("[INFO] label_trajectory not available; skipping labeling")
        args.no_label = True

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from api_keys import open_router_api_key as _k
            api_key = _k
        except Exception:
            pass
    if not api_key:
        print("[WARNING] No API key found. LLM calls will fall back to defaults.")

    requested = args.games or list(ORAK_COLD_START_GAMES.keys())

    print("=" * 78)
    print("  GPT-5.4 Cold-Start — Orak Games (Super Mario & StarCraft II)")
    print("=" * 78)
    print(f"  Games:       {', '.join(requested)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  Max steps:   {args.max_steps or 'per-game default'}")
    print(f"  Model:       {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Workers:     {args.workers}")
    print(f"  Labeling:    {args.label and not args.no_label}")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for game_name in requested:
        display = ORAK_COLD_START_GAMES[game_name]["display_name"]
        print(f"\n{'━' * 78}")
        print(f"  GAME: {display} ({args.episodes} episodes)")
        print(f"{'━' * 78}")

        summary = run_game_rollouts(game_name, args, output_dir)
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "source": "orak",
        "model": args.model,
        "agent_type": "gpt54_orak",
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "labeled": args.label and not args.no_label,
        "total_elapsed_seconds": round(overall_elapsed, 2),
        "games_completed": requested,
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  GPT-5.4 ORAK COLD-START COMPLETE")
    print(f"{'=' * 78}")
    total_eps = sum(
        s.get("total_episodes", 0) for s in game_summaries if not s.get("skipped")
    )
    print(f"  Games:          {len(requested)}")
    print(f"  Total episodes: {total_eps}")
    print(f"  Elapsed:        {overall_elapsed:.1f}s")
    print(f"  Output:         {output_dir}")
    print(f"  Summary:        {master_path}")

    successful = [s for s in game_summaries if not s.get("skipped") and "mean_reward" in s]
    if successful:
        avg_r = sum(s["mean_reward"] for s in successful) / len(successful)
        avg_s = sum(s["mean_steps"] for s in successful) / len(successful)
        print(f"  Avg reward:     {avg_r:.3f}")
        print(f"  Avg steps:      {avg_s:.1f}")

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
