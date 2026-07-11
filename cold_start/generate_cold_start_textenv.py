#!/usr/bin/env python
"""
Cold-start SFT data generation for text environments (WebShop, ALFWorld).

Generates decision-making trajectories using GPT-5.4 (or any OpenAI-
compatible model).  Output format matches the existing cold_start pipeline
(Episode/Experience + JSONL).

Usage:
    export OPENAI_API_KEY="sk-..."
    cd /workspace/cos-play
    PYTHONPATH=. python cold_start/generate_cold_start_textenv.py \
        --games webshop alfworld \
        --episodes 20 --max_steps 50 --model gpt-5.4 -v
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

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent

if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))

from data_structure.experience import Experience, Episode, Episode_Buffer

import openai

# ---------------------------------------------------------------------------
# System prompts — per environment
# ---------------------------------------------------------------------------
WEBSHOP_SYSTEM_PROMPT = """\
You are an expert online shopping assistant using a text-based e-commerce site.

At each step you see the page content and a list of clickable elements.
You can take exactly ONE of two action types:

  search[<your query>]   — type a search query (e.g. search[red running shoes size 10])
  click[<element text>]  — click a link/button shown on the page (e.g. click[Buy Now])

Strategy:
1. Read the GOAL carefully — note required attributes (color, size, price limit).
2. Use search[] with keywords extracted from the goal.
3. Click the most relevant product from results.
4. Select correct options (size, color), then click "Buy Now".

Respond with ONLY the action string, e.g.:
  search[women's cotton jumpsuit size large under $50]
  click[Buy Now]
Do NOT add any other text outside the action."""

ALFWORLD_SYSTEM_PROMPT = """\
You are an expert household robot in a text-based home simulator (ALFWorld).

At each step you see your surroundings and a list of admissible commands.
You must first THINK about what to do, then choose EXACTLY ONE command.

RESPONSE FORMAT (mandatory):
Think: <your reasoning — what is the task? what object do I need? where should I look?>
Act: <exactly one admissible command>

TASK TYPES & STRATEGIES:

1. "put X in/on Y" (pick_and_place):
   go to locations to find X → take X from <receptacle> → go to Y → put X in/on Y

2. "examine X with desklamp" (look_at_obj_in_light):
   Find X → take X from <receptacle> → go to desklamp location → use desklamp 1

3. "clean X and put in Y" (pick_clean_then_place):
   Find X → take X → go to sinkbasin 1 → clean X with sinkbasin 1 → go to Y → put X in/on Y

4. "heat X and put in Y" (pick_heat_then_place):
   Find X → take X → go to microwave 1 → heat X with microwave 1 → go to Y → put X in/on Y

5. "cool X and put in Y" (pick_cool_then_place):
   Find X → take X → go to fridge 1 → cool X with fridge 1 → go to Y → put X in/on Y

6. "find two X and put in Y" (pick_two_obj_and_place):
   Find first X → take → go to Y → put in/on Y → find second X → take → go to Y → put in/on Y

CRITICAL RULES:
- Read the task FIRST. Identify the EXACT object name and target receptacle.
- Only interact with the object named in the task — do NOT pick up unrelated items.
- If you see the target object at a location, take it IMMEDIATELY.
- If you do NOT see the object, systematically search: go to each receptacle, open containers.
- After "take X from R", go DIRECTLY to the next destination (target or processing station).
- NEVER repeat the same action — if it didn't work, try something else.
- "open" drawers/cabinets/fridge BEFORE looking inside.

EXAMPLE (pick_and_place):
Task: put some spraybottle on garbagecan.

Observation: You are in the middle of a room. You see a countertop 1, a drawer 1, a garbagecan 1...
Think: I need to find a spraybottle and put it on garbagecan 1. Let me check countertop 1 first.
Act: go to countertop 1

Observation: On the countertop 1, you see a candle 1, a spraybottle 1, and a toiletpaper 1.
Think: I found spraybottle 1 on countertop 1. I need to take it.
Act: take spraybottle 1 from countertop 1

Observation: You pick up the spraybottle 1 from the countertop 1.
Think: I have the spraybottle. Now I need to go to garbagecan 1 and put it there.
Act: go to garbagecan 1

Observation: You arrive at garbagecan 1.
Think: I'm at the garbagecan. Now I put the spraybottle in it.
Act: put spraybottle 1 in/on garbagecan 1"""

_HISTORY_WINDOW_WEBSHOP = 5
_HISTORY_WINDOW_ALFWORLD = 15


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _format_history(recent: List[Dict[str, Any]], is_webshop: bool = False) -> str:
    if not recent:
        return ""
    window = _HISTORY_WINDOW_WEBSHOP if is_webshop else _HISTORY_WINDOW_ALFWORLD
    lines = ["Recent actions:"]
    for entry in recent[-window:]:
        obs_preview = entry.get("obs_preview", "")
        tag = f"reward {entry['reward']:+.2f}" if entry["reward"] != 0 else ""
        if obs_preview:
            lines.append(f"  {entry['action']} → {obs_preview}{(' ' + tag) if tag else ''}")
        else:
            lines.append(f"  {entry['action']}{(' ' + tag) if tag else ''}")

    if len(recent) >= 3:
        last_actions = [e["action"] for e in recent[-3:]]
        if len(set(last_actions)) <= 2:
            lines.append("\nWARNING: You are STUCK in a loop! Try a COMPLETELY DIFFERENT action or location.")
            visited = set(e["action"] for e in recent[-8:])
            lines.append(f"Actions tried recently: {', '.join(sorted(visited))}")

    return "\n".join(lines) + "\n\n"


def _call_llm(
    observation: str,
    action_names: List[str],
    system_prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    recent_history: Optional[List[Dict[str, Any]]] = None,
    is_webshop: bool = False,
    chat_messages: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Optional[str]]:
    """Query LLM and return (action, reasoning_or_None).

    For ALFWorld, *chat_messages* carries the full multi-turn conversation
    so the model can track its own reasoning across steps.
    """

    if chat_messages is not None and not is_webshop:
        # Multi-turn ALFWorld: append new observation as user message
        action_list = "\n".join(f"  - {a}" for a in action_names)
        user_msg = (
            f"{observation}\n\n"
            f"Admissible commands:\n{action_list}"
        )
        chat_messages.append({"role": "user", "content": user_msg})
        messages = chat_messages
        max_tokens = 300
    elif is_webshop:
        history_block = _format_history(recent_history or [], is_webshop=True)
        user_content = (
            f"{observation}\n\n"
            f"Available clickable elements: {', '.join(a for a in action_names if a.startswith('click['))}\n"
            f"{'You can also use search[<query>] to search.' if any(a.startswith('search') for a in action_names) else ''}\n\n"
            f"{history_block}"
            f"Choose your action (respond with ONLY the action string):"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        max_tokens = 200
    else:
        history_block = _format_history(recent_history or [], is_webshop=False)
        action_list = "\n".join(f"  - {a}" for a in action_names)
        user_content = (
            f"{observation}\n\n"
            f"Admissible commands:\n{action_list}\n\n"
            f"{history_block}"
            f"Think step by step, then Act."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        max_tokens = 300

    try:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        reply = (response.choices[0].message.content or "").strip()

        if is_webshop:
            action, reasoning = _parse_webshop_reply(reply, action_names)
        else:
            action, reasoning = _parse_alfworld_reply(reply, action_names)

        # For multi-turn, append assistant reply to chat history
        if chat_messages is not None and not is_webshop:
            chat_messages.append({"role": "assistant", "content": reply})

        return action, reasoning

    except Exception as exc:
        print(f"    [WARN] LLM call failed: {exc}")
        if chat_messages is not None and not is_webshop:
            # Remove the user message we added since call failed
            chat_messages.pop()
        return action_names[0] if action_names else "look", None


def _parse_webshop_reply(reply: str, action_names: List[str]) -> Tuple[str, Optional[str]]:
    """Extract a WebShop action from the LLM reply."""
    # Try to find search[...] or click[...] pattern
    m = re.search(r'(search\[.+?\]|click\[.+?\])', reply, re.IGNORECASE)
    if m:
        raw = m.group(1)
        if raw.lower().startswith("search["):
            return raw, None
        # For click, try to match against known clickable elements
        inner = re.search(r'click\[(.+?)\]', raw, re.IGNORECASE)
        if inner:
            click_target = inner.group(1).strip()
            for a in action_names:
                am = re.search(r'click\[(.+?)\]', a)
                if am and am.group(1).strip().lower() == click_target.lower():
                    return a, None
            return f"click[{click_target}]", None

    # Fallback: check if any action name appears in the reply
    for a in action_names:
        if a.lower() in reply.lower() and a.startswith("click["):
            return a, None

    return action_names[0] if action_names else "search[products]", None


def _parse_alfworld_reply(reply: str, action_names: List[str]) -> Tuple[str, Optional[str]]:
    """Extract an ALFWorld action from the ReAct-format LLM reply.

    Expected format: 'Think: <reasoning>\nAct: <command>'
    Falls back to line-by-line matching if Act: tag is missing.
    """
    lower_map = {a.lower().strip(): a for a in action_names}
    reasoning = None

    # Extract reasoning from Think: tag
    think_m = re.search(r"Think:\s*(.+?)(?=\nAct:|\Z)", reply, re.DOTALL | re.IGNORECASE)
    if think_m:
        reasoning = think_m.group(1).strip()

    # Extract action from Act: tag
    act_m = re.search(r"Act:\s*(.+)", reply, re.IGNORECASE)
    if act_m:
        act_str = act_m.group(1).strip().strip('"').strip("'").strip("`")
        if act_str.lower() in lower_map:
            return lower_map[act_str.lower()], reasoning
        # Fuzzy match — find best substring overlap
        for a in action_names:
            if a.lower() in act_str.lower() or act_str.lower() in a.lower():
                return a, reasoning

    # Fallback: exact match on whole reply
    reply_clean = reply.strip().strip('"').strip("'")
    if reply_clean.lower().strip() in lower_map:
        return lower_map[reply_clean.lower().strip()], reasoning

    # Line-by-line check
    for line in reply.split("\n"):
        line_clean = line.strip().strip('"').strip("'").strip("`")
        if line_clean.lower() in lower_map:
            return lower_map[line_clean.lower()], reasoning

    # Substring match
    for a in action_names:
        if a.lower() in reply.lower():
            return a, reasoning

    return action_names[0] if action_names else "look", reasoning


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_webshop_episode(
    api_key: str,
    model: str,
    max_steps: int,
    temperature: float,
    verbose: bool,
    num_products: Optional[int],
    episode_idx: int,
) -> Tuple[Episode, Dict[str, Any]]:
    from env_wrappers.webshop_nl_wrapper import make_webshop_env

    env = make_webshop_env(max_steps=max_steps, num_products=num_products)
    obs, info = env.reset()
    goal = info.get("goal", "")

    experiences: List[Experience] = []
    total_reward = 0.0
    recent_history: List[Dict[str, Any]] = []
    terminated = False
    truncated = False

    for step_i in range(max_steps):
        action_names = info.get("action_names", ["search[query]"])

        action, reasoning = _call_llm(
            observation=obs,
            action_names=action_names,
            system_prompt=WEBSHOP_SYSTEM_PROMPT,
            api_key=api_key,
            model=model,
            temperature=temperature,
            recent_history=recent_history,
            is_webshop=True,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        recent_history.append({"action": action, "reward": reward})

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=goal,
        )
        exp.idx = step_i
        exp.action_type = "text_action"
        exp.available_actions = list(action_names)
        exp.interface = {"env_name": "webshop", "game_name": "webshop"}
        if next_info.get("effect_tags"):
            exp.reward_details = next_info["effect_tags"]
        experiences.append(exp)

        if verbose:
            print(f"    step {step_i}: {action[:60]}, r={reward:.2f}, cum={total_reward:.2f}")

        obs = next_obs
        info = next_info
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=goal,
        env_name="webshop",
        game_name="webshop",
    )
    episode.set_outcome()

    stats = {
        "game": "webshop",
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "model": model,
        "agent_type": "gpt54_textenv",
    }
    return episode, stats


def run_alfworld_episode(
    api_key: str,
    model: str,
    max_steps: int,
    temperature: float,
    verbose: bool,
    split: str,
    episode_idx: int,
    task_types: Optional[List[int]] = None,
) -> Tuple[Episode, Dict[str, Any]]:
    from env_wrappers.alfworld_nl_wrapper import make_alfworld_env

    env = make_alfworld_env(max_steps=max_steps, split=split, task_types=task_types)
    obs, info = env.reset()
    goal = info.get("goal", "")

    # Multi-turn chat: system prompt + explicit task reminder as first message
    chat_messages: List[Dict[str, str]] = [
        {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
        {"role": "user", "content": f"YOUR TASK: {goal}\n\nYou MUST complete this exact task. Do not change or reinterpret it."},
        {"role": "assistant", "content": f"Think: I understand. My task is: {goal}. I will focus on exactly this task.\nAct: I'm ready. Show me the environment."},
    ]

    experiences: List[Experience] = []
    total_reward = 0.0
    recent_history: List[Dict[str, Any]] = []
    terminated = False
    truncated = False

    for step_i in range(max_steps):
        action_names = info.get("action_names", ["look"])

        action, reasoning = _call_llm(
            observation=obs,
            action_names=action_names,
            system_prompt=ALFWORLD_SYSTEM_PROMPT,
            api_key=api_key,
            model=model,
            temperature=temperature,
            recent_history=recent_history,
            is_webshop=False,
            chat_messages=chat_messages,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        obs_preview = next_obs[:120].replace("\n", " ").strip() if next_obs else ""
        recent_history.append({"action": action, "reward": reward, "obs_preview": obs_preview})

        exp = Experience(
            state=obs,
            action=str(action),
            reward=float(reward),
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=goal,
        )
        exp.idx = step_i
        exp.action_type = "text_action"
        exp.available_actions = list(action_names)
        exp.interface = {"env_name": "alfworld", "game_name": "alfworld"}
        if next_info.get("effect_tags"):
            exp.reward_details = next_info["effect_tags"]
        experiences.append(exp)

        if verbose:
            think_preview = f" [{reasoning[:50]}...]" if reasoning else ""
            print(f"    step {step_i}: {action[:60]}, r={reward:.2f}, cum={total_reward:.2f}{think_preview}")

        obs = next_obs
        info = next_info
        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=goal,
        env_name="alfworld",
        game_name="alfworld",
    )
    episode.set_outcome()

    task_type = getattr(env, '_current_task_type', 'unknown')
    stats = {
        "game": "alfworld",
        "task_type": task_type,
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "model": model,
        "agent_type": "gpt54_textenv",
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_game_rollouts(
    game: str,
    episodes: int,
    api_key: str,
    model: str,
    max_steps: int,
    temperature: float,
    verbose: bool,
    output_dir: Path,
    resume: bool,
    num_products: Optional[int] = None,
    split: str = "train",
    task_types: Optional[List[int]] = None,
    game_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    dir_name = game_subdir if game_subdir else game
    game_dir = output_dir / dir_name
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    start_idx = 0
    if resume:
        start_idx = sum(1 for f in game_dir.glob("episode_*.json") if f.name != "episode_buffer.json")
        if start_idx >= episodes:
            print(f"  [SKIP] {game}: {start_idx}/{episodes} already done")
            return {"game": game, "skipped": True, "existing": start_idx}
        if start_idx > 0:
            print(f"  [RESUME] {game}: resuming from episode {start_idx}")

    buffer = Episode_Buffer(buffer_size=episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(start_idx, episodes):
        print(f"\n  [{game}] Episode {ep_idx + 1}/{episodes}")

        try:
            if game == "webshop":
                episode, stats = run_webshop_episode(
                    api_key=api_key, model=model, max_steps=max_steps,
                    temperature=temperature, verbose=verbose,
                    num_products=num_products, episode_idx=ep_idx,
                )
            elif game == "alfworld":
                episode, stats = run_alfworld_episode(
                    api_key=api_key, model=model, max_steps=max_steps,
                    temperature=temperature, verbose=verbose,
                    split=split, episode_idx=ep_idx,
                    task_types=task_types,
                )
            else:
                raise ValueError(f"Unknown game: {game}")

            stats["episode_index"] = ep_idx
            print(f"    => Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            record = episode.to_dict()
            record["rollout_metadata"] = stats
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": game, "episode_index": ep_idx,
                "error": str(e), "steps": 0, "total_reward": 0.0,
            })

    elapsed = time.time() - t0

    buffer_path = game_dir / "episode_buffer.json"
    buffer.save_to_json(str(buffer_path))

    summary: Dict[str, Any] = {
        "game": game,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "total_episodes": len([s for s in all_stats if "error" not in s]),
        "elapsed_seconds": elapsed,
    }
    if all_stats:
        rewards = [s["total_reward"] for s in all_stats if "error" not in s]
        steps = [s["steps"] for s in all_stats if "error" not in s]
        if rewards:
            summary["mean_reward"] = sum(rewards) / len(rewards)
            summary["mean_steps"] = sum(steps) / len(steps)
            summary["max_reward"] = max(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.4 cold-start SFT data for text environments (WebShop, ALFWorld)",
    )
    parser.add_argument("--games", nargs="+", default=["webshop", "alfworld"],
                        choices=["webshop", "alfworld"],
                        help="Which text environments to run (default: both)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per game (default: 20)")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode (default: 50)")
    parser.add_argument("--model", type=str, default="gpt-5.4",
                        help="Model name (default: gpt-5.4)")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--num_products", type=int, default=None,
                        help="WebShop: number of products to load (None=all)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
                        help="ALFWorld data split (default: train)")
    parser.add_argument("--task_types", type=int, nargs="+", default=None,
                        help="ALFWorld task types to include (1-6). Default: all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--game_subdir", type=str, default=None,
                        help="Override subdirectory name (e.g. alfworld_pick_and_place)")

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            sys.path.insert(0, str(CODEBASE_ROOT.parent))
            from keys import OPENAI_API_KEY
            api_key = OPENAI_API_KEY
        except Exception:
            pass
    if not api_key:
        print("[ERROR] No OPENAI_API_KEY found. Set it via env or /workspace/keys.py")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54_textenv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  Text Environment Cold-Start SFT Data Generation")
    print("=" * 68)
    print(f"  Games:      {', '.join(args.games)}")
    print(f"  Episodes:   {args.episodes} per game")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Model:      {args.model}")
    print(f"  API key:    {api_key[:12]}...")
    print(f"  Output:     {output_dir}")
    print("=" * 68)

    t0 = time.time()
    summaries = []

    for game in args.games:
        print(f"\n{'━' * 68}")
        print(f"  GAME: {game} ({args.episodes} episodes)")
        print(f"{'━' * 68}")

        summary = run_game_rollouts(
            game=game,
            episodes=args.episodes,
            api_key=api_key,
            model=args.model,
            max_steps=args.max_steps,
            temperature=args.temperature,
            verbose=args.verbose,
            output_dir=output_dir,
            resume=args.resume,
            num_products=args.num_products,
            split=args.split,
            task_types=args.task_types,
            game_subdir=args.game_subdir,
        )
        summaries.append(summary)

    total_elapsed = time.time() - t0

    master = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "episodes_per_game": args.episodes,
        "total_elapsed": total_elapsed,
        "summaries": summaries,
    }
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(master, f, indent=2, default=str)

    print(f"\n{'=' * 68}")
    print("  BATCH COMPLETE")
    print(f"{'=' * 68}")
    for s in summaries:
        if s.get("skipped"):
            print(f"  {s['game']}: skipped (already done)")
        else:
            mr = s.get("mean_reward", 0)
            ms = s.get("mean_steps", 0)
            print(f"  {s['game']}: {s.get('total_episodes', 0)} episodes, "
                  f"mean_reward={mr:.2f}, mean_steps={ms:.1f}")
    print(f"  Elapsed: {total_elapsed:.1f}s")
    print(f"  Output:  {output_dir}")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
