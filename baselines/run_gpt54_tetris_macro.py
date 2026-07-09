#!/usr/bin/env python
"""
GPT-5.4 baseline on Tetris using the SAME env wrapper chain as training.

Training pipeline (episode_runner.py):
    base_env = make_gaming_env("tetris", max_steps=200)
    env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))

This script replicates that chain exactly, then uses GPT-5.4 (via OpenRouter
or OpenAI) to choose among the placement-level macro actions.  Each action
places one entire piece (rotation + column), matching training semantics.

Usage (from Game-AI-Agent root):
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    python baselines/run_gpt54_tetris_macro.py
    python baselines/run_gpt54_tetris_macro.py --episodes 4 --verbose
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
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

from evaluate_gamingagent.gym_like import make_gaming_env
from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
from trainer.coevolution.config import GAME_MAX_STEPS

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


MODEL_GPT54 = "gpt-5.4"

SYSTEM_PROMPT = (
    "You are an expert Tetris-playing agent.\n"
    "You receive the current board state and a list of numbered placement options.\n"
    "Each placement describes: piece-orientation, column, lines it would clear, "
    "holes it would create, and resulting stack height.\n\n"
    "Call the `choose_action` function with the EXACT placement description.\n"
    "Pick the best placement from the valid actions list."
)

USER_TEMPLATE = (
    "Game state:\n\n{state}\n\n"
    "Valid placements:\n{actions}\n\n"
    "Choose one placement."
)


def _build_tools(action_names: List[str]) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the single placement for the current piece.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": f"One of: {', '.join(action_names[:10])}{'...' if len(action_names) > 10 else ''}",
                        },
                    },
                    "required": ["action"],
                },
            },
        }
    ]


def _extract_action(text: str, valid_actions: List[str]) -> Optional[str]:
    """Best-effort fuzzy match of model output to a valid action string."""
    if not text:
        return None
    text_lower = text.strip().lower()
    for v in valid_actions:
        if v.lower() in text_lower:
            return v
    for v in valid_actions:
        if v.split(" ")[0].lower() in text_lower and v.split("col")[1].split(" ")[0] if "col" in v else "" in text_lower:
            return v
    return valid_actions[0] if valid_actions else None


def gpt54_choose_placement(
    state_nl: str,
    action_names: List[str],
    model: str = MODEL_GPT54,
    temperature: float = 0.3,
) -> Tuple[str, Optional[str]]:
    """Query GPT-5.4 to choose a macro placement action. Returns (action, reasoning)."""
    use_router = open_router_api_key and open_router_api_key.strip()
    if openai is None or (not use_router and openai_api_key is None):
        return action_names[0] if action_names else "hard_drop", None

    client_kw: Dict[str, Any] = {}
    effective_model = model
    if use_router:
        client_kw = {"base_url": OPENROUTER_BASE, "api_key": open_router_api_key.strip()}
        effective_model = model if "/" in model else f"openai/{model}"
    else:
        client_kw = {"api_key": openai_api_key}

    numbered = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))
    user_content = USER_TEMPLATE.format(state=state_nl, actions=numbered)
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
            max_tokens=48,
        )
        msg = response.choices[0].message

        if msg.tool_calls and len(msg.tool_calls) > 0:
            tc = msg.tool_calls[0]
            raw_args = getattr(tc, "arguments", None) or getattr(tc.function, "arguments", None) or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

            action = args.get("action", "")
            reasoning = args.get("reasoning")

            if action:
                for v in action_names:
                    if v.lower() == action.lower().strip():
                        return v, reasoning
                for v in action_names:
                    if action.lower().strip() in v.lower() or v.lower() in action.lower().strip():
                        return v, reasoning
                # Try numeric index (e.g. "3" → action_names[2])
                try:
                    idx = int(action.strip()) - 1
                    if 0 <= idx < len(action_names):
                        return action_names[idx], reasoning
                except (ValueError, TypeError):
                    pass
                extracted = _extract_action(action, action_names)
                if extracted:
                    return extracted, reasoning

        content = msg.content or ""
        extracted = _extract_action(content, action_names)
        if extracted:
            return extracted, None
        return (action_names[0] if action_names else "hard_drop"), None

    except Exception as exc:
        print(f"    [WARN] GPT-5.4 call failed ({exc}), using fallback")
        return (action_names[0] if action_names else "hard_drop"), None


def run_episode(
    max_steps: int,
    model: str,
    temperature: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Run one Tetris episode with training-identical env chain + GPT-5.4."""
    base_env = make_gaming_env("tetris", max_steps=max_steps)
    env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])

    total_reward = 0.0
    step_count = 0
    terminated = truncated = False
    step_details: List[Dict[str, Any]] = []

    while step_count < max_steps:
        if not action_names:
            break

        action, reasoning = gpt54_choose_placement(
            state_nl=obs_nl,
            action_names=action_names,
            model=model,
            temperature=temperature,
        )

        obs_nl, reward, terminated, truncated, info = env.step(action)
        action_names = info.get("action_names", [])
        total_reward += reward
        step_count += 1

        step_info = {
            "step": step_count,
            "action": action,
            "reward": float(reward),
            "cumulative_reward": total_reward,
        }
        if info.get("board_stats"):
            step_info["board_stats"] = info["board_stats"]
        step_details.append(step_info)

        if verbose:
            board = info.get("board_stats", {})
            reason_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            print(f"  step {step_count}: {action}  r={reward:.1f}  cum={total_reward:.1f}"
                  f"  lines={board.get('lines_total', '?')}  h={board.get('stack_height', '?')}"
                  f"  holes={board.get('holes', '?')}")
            if reason_short:
                print(f"    reason: {reason_short}")

        if terminated or truncated:
            break

    env.close()

    return {
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "max_steps": max_steps,
        "step_details": step_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.4 Tetris baseline with training-identical macro actions",
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=GAME_MAX_STEPS.get("tetris", 200))
    parser.add_argument("--model", type=str, default=MODEL_GPT54)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output" / "gpt54_tetris"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GPT-5.4 Tetris Baseline — Macro Actions (training-identical)")
    print("=" * 70)
    print(f"  Env chain:    make_gaming_env → GamingAgentNLWrapper → TetrisMacroActionWrapper")
    print(f"  Max steps:    {args.max_steps} (GAME_MAX_STEPS['tetris'])")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Model:        {args.model}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Output:       {output_dir}")
    print("=" * 70)

    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(args.episodes):
        print(f"\n  Episode {ep_idx + 1}/{args.episodes}")
        try:
            stats = run_episode(
                max_steps=args.max_steps,
                model=args.model,
                temperature=args.temperature,
                verbose=args.verbose,
            )
            stats["episode_index"] = ep_idx
            all_stats.append(stats)
            print(f"    Result: steps={stats['steps']}, reward={stats['total_reward']:.2f}, "
                  f"term={stats['terminated']}, trunc={stats['truncated']}")

            ep_path = output_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "episode_index": ep_idx,
                "error": str(e),
                "steps": 0,
                "total_reward": 0.0,
            })

    elapsed = time.time() - t0

    rewards = [s["total_reward"] for s in all_stats if "error" not in s]
    steps_list = [s["steps"] for s in all_stats if "error" not in s]

    summary = {
        "model": args.model,
        "game": "tetris",
        "env_chain": "make_gaming_env → GamingAgentNLWrapper → TetrisMacroActionWrapper",
        "action_type": "macro_placement",
        "max_steps": args.max_steps,
        "n_episodes": len(rewards),
        "n_failed": len(all_stats) - len(rewards),
        "elapsed_seconds": elapsed,
        "rewards": rewards,
        "steps": steps_list,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "std_reward": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5 if rewards else 0,
        "mean_steps": sum(steps_list) / len(steps_list) if steps_list else 0,
        "episodes": all_stats,
    }
    summary_path = output_dir / "reward_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print("  GPT-5.4 Tetris Baseline — REWARD REPORT")
    print(f"{'=' * 70}")
    print(f"  {'Episode':<10} {'Reward':>10} {'Steps':>8} {'Outcome':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*12}")
    for s in all_stats:
        if "error" in s:
            print(f"  {s['episode_index']+1:<10} {'ERROR':>10} {'':>8} {'failed':>12}")
        else:
            outcome = "terminated" if s["terminated"] else ("truncated" if s["truncated"] else "running")
            print(f"  {s['episode_index']+1:<10} {s['total_reward']:>10.2f} {s['steps']:>8} {outcome:>12}")

    print(f"\n  {'-' * 50}")
    if rewards:
        print(f"  Mean reward:    {summary['mean_reward']:.2f}")
        print(f"  Max reward:     {summary['max_reward']:.2f}")
        print(f"  Min reward:     {summary['min_reward']:.2f}")
        print(f"  Std reward:     {summary['std_reward']:.2f}")
        print(f"  Mean steps:     {summary['mean_steps']:.1f}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"  Output:         {output_dir}")
    print(f"  Summary JSON:   {summary_path}")
    print(f"{'=' * 70}")

    report_path = output_dir / "reward_report.txt"
    with open(report_path, "w") as f:
        f.write(f"GPT-5.4 Tetris Baseline — Macro Actions (training-identical)\n")
        f.write(f"Env: make_gaming_env → GamingAgentNLWrapper → TetrisMacroActionWrapper\n")
        f.write(f"Max steps: {args.max_steps}, Episodes: {len(rewards)}, Model: {args.model}\n\n")
        for s in all_stats:
            if "error" not in s:
                f.write(f"Episode {s['episode_index']+1}: reward={s['total_reward']:.2f}, steps={s['steps']}\n")
        f.write(f"\nMean reward: {summary['mean_reward']:.2f}\n")
        f.write(f"Max reward:  {summary['max_reward']:.2f}\n")
        f.write(f"Min reward:  {summary['min_reward']:.2f}\n")
        f.write(f"Std reward:  {summary['std_reward']:.2f}\n")


if __name__ == "__main__":
    main()
