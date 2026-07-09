#!/usr/bin/env python3
"""Capture real Gym-V observations and send them to GPT-4o for schema generation.

Resets a Gym-V env, takes a few steps, and prints the structured
``<state>…</state>`` schema GPT-4o produces from each rendered frame.

Usage:
    # Run the default suite (Game2048-v0 + Minesweeper-v0, 2 steps each)
    python -m gymv_wrapper.test_schema_gen

    # Custom env list
    python -m gymv_wrapper.test_schema_gen --env Games/Sokoban-v0 --steps 3

    # Save captured screenshots next to this file
    python -m gymv_wrapper.test_schema_gen --save-images

The OpenRouter API key is loaded from ``api_keys.open_router_api_key``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_keys import open_router_api_key  # noqa: E402
from gymv_wrapper.adapter import generate_label as gymv_generate_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = open_router_api_key
MODEL = "openai/gpt-4.1"  # via OpenRouter; swap to "openai/gpt-5.4" when available

DEFAULT_ENVS = ["Games/Game2048-v0", "Games/Minesweeper-v0"]
ACTION_MAP: dict[str, list[str]] = {
    "Games/Game2048-v0": ["[Up]", "[Left]", "[Down]", "[Right]", "[Up]"],
    "Games/Minesweeper-v0": ["reveal 0 0", "reveal 1 1", "reveal 2 2"],
    "Games/Sokoban-v0": ["[Up]", "[Right]", "[Down]", "[Left]", "[Up]"],
    "Games/Wordle-v0": ["CRANE", "SLOTH", "BUMPY"],
}


def capture_gymv_obs(env_id: str, n_steps: int = 3, seed: int = 42) -> list[dict]:
    """Reset a Gym-V env, take ``n_steps``, return obs dicts.

    Each dict has: ``image``, ``text``, ``description``, ``task_id``, ``step``.
    """
    import gym_v

    log.info("Creating Gym-V env: %s", env_id)
    env = gym_v.make(env_id)
    obs_dict, info_dict = env.reset(seed=seed)
    obs = obs_dict["agent_0"]

    description = getattr(env, "description", "") or ""

    results = [{
        "image": obs.image,
        "text": obs.text,
        "description": description,
        "task_id": env_id,
        "step": 0,
    }]

    actions = ACTION_MAP.get(env_id, ["[Up]"] * n_steps)

    for i, action in enumerate(actions[:n_steps]):
        try:
            obs_dict, rew, term, trunc, info = env.step({"agent_0": action})
            obs = obs_dict["agent_0"]
            results.append({
                "image": obs.image,
                "text": obs.text,
                "description": description,
                "task_id": env_id,
                "step": i + 1,
            })
            all_done = term.get("__all__", False) or trunc.get("__all__", False)
            if all_done:
                log.info("  %s terminated at step %d", env_id, i + 1)
                break
        except Exception as e:
            log.warning("  Step %d failed: %s", i + 1, e)
            break

    env.close()
    log.info("  Captured %d observations from %s", len(results), env_id)
    return results


def run_gymv_test(obs_data: dict) -> dict:
    """Send a Gym-V observation to GPT-4o."""
    goal_line = ""
    if obs_data["description"]:
        goal_line = obs_data["description"].strip().split("\n")[0]

    t0 = time.time()
    result = gymv_generate_label(
        obs_data["image"],
        goal=goal_line,
        task_id=obs_data["task_id"],
        step=obs_data["step"],
        game_rules=obs_data["description"],
        obs_text=obs_data["text"] or "",
        model=MODEL,
        api_key=API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


def print_result(label: str, result: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"  model={result['model']}  elapsed={result.get('elapsed_s', '?')}s")
    print(sep)

    if result["schema"]:
        print(result["schema"])
    else:
        print("[NO SCHEMA PARSED]")
        print("Raw output:")
        print(textwrap.indent(result["raw"][:2000], "  "))

    if result["warnings"]:
        print(f"\nWarnings: {result['warnings']}")
    else:
        print("\nValidation: PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test GPT-4o schema generation on real Gym-V observations",
    )
    parser.add_argument(
        "--env", action="append", default=None,
        help="Gym-V env id (repeat to add more). Defaults to "
             "Games/Game2048-v0 and Games/Minesweeper-v0",
    )
    parser.add_argument("--steps", type=int, default=2,
                        help="Number of game steps to capture per env")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true",
                        help="Save captured screenshots next to this file")
    args = parser.parse_args()

    envs = args.env or DEFAULT_ENVS
    out_dir = Path(__file__).parent

    for env_id in envs:
        try:
            observations = capture_gymv_obs(env_id, n_steps=args.steps, seed=args.seed)
        except Exception as e:
            log.error("Failed to capture %s: %s", env_id, e)
            continue

        for obs_data in observations:
            step = obs_data["step"]
            tag = env_id.replace("/", "_")

            if args.save_images and obs_data["image"]:
                fname = out_dir / f"real_{tag}_step{step}.png"
                obs_data["image"].save(str(fname))
                log.info("  Saved %s", fname.name)

            log.info("Sending %s step=%d to %s ...", env_id, step, MODEL)
            result = run_gymv_test(obs_data)
            print_result(f"Gym-V: {env_id} (step {step})", result)

    print("\n" + "=" * 70)
    print("  Done. All observations came from real Gym-V envs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
