#!/usr/bin/env python3
"""Capture real BrowserGym observations and send them to GPT-4o for schema generation.

Resets a ``browsergym/openended`` env on the requested URL and prints the
``<state>…</state>`` schema GPT-4o produces for the rendered page.

Usage:
    # Default: Google + Wikipedia (one step each)
    python -m browsergym_wrapper.test_schema_gen

    # Custom URL
    python -m browsergym_wrapper.test_schema_gen --url https://en.wikipedia.org

    # With a goal hint for the schema
    python -m browsergym_wrapper.test_schema_gen \\
        --url https://en.wikipedia.org/wiki/Reinforcement_learning \\
        --goal "Find the section about temporal difference learning"

    # Save captured screenshots next to this file
    python -m browsergym_wrapper.test_schema_gen --save-images

The OpenRouter API key is loaded from ``api_keys.open_router_api_key``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import api_keys  # noqa: E402
from browsergym_wrapper.adapter import generate_label as browser_generate_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = (
    getattr(api_keys, "open_router_api_key", None)
    or getattr(api_keys, "openrouter_api_key", None)
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)
MODEL = "openai/gpt-4.1"  # via OpenRouter; swap to "openai/gpt-5.4" when available

DEFAULT_URLS: list[tuple[str, str]] = [
    ("https://www.google.com", "Explore https://www.google.com"),
    (
        "https://en.wikipedia.org/wiki/Reinforcement_learning",
        "Find the section about temporal difference learning",
    ),
]


def capture_browser_obs(
    url: str = "https://www.google.com",
    goal: str = "",
    n_steps: int = 0,
) -> list[dict]:
    """Reset a BrowserGym ``openended`` env on ``url`` and return obs dicts."""
    import gymnasium as gym
    import browsergym.core  # noqa: F401 — registers envs

    log.info("Creating BrowserGym env for: %s", url)
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": url},
        headless=True,
    )
    obs, info = env.reset()

    def _to_pil(screenshot):
        if isinstance(screenshot, np.ndarray):
            return Image.fromarray(screenshot)
        return screenshot

    results = [{
        "image": _to_pil(obs["screenshot"]),
        "goal": goal or f"Explore {url}",
        "url": obs.get("url", url),
        "task_id": "browsergym/openended",
        "step": 0,
        "last_action": "",
        "last_action_error": "",
    }]

    for i in range(n_steps):
        obs, rew, term, trunc, info = env.step("noop()")
        results.append({
            "image": _to_pil(obs["screenshot"]),
            "goal": goal or f"Explore {url}",
            "url": obs.get("url", ""),
            "task_id": "browsergym/openended",
            "step": i + 1,
            "last_action": obs.get("last_action", ""),
            "last_action_error": obs.get("last_action_error", ""),
        })
        if term or trunc:
            break

    env.close()
    log.info("  Captured %d observations from BrowserGym", len(results))
    return results


def run_browser_test(obs_data: dict) -> dict:
    """Send a BrowserGym observation to GPT-4o."""
    t0 = time.time()
    result = browser_generate_label(
        obs_data["image"],
        goal=obs_data["goal"],
        task_id=obs_data["task_id"],
        step=obs_data["step"],
        url=obs_data["url"],
        last_action=obs_data.get("last_action", ""),
        last_action_error=obs_data.get("last_action_error", ""),
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
        description="Test GPT-4o schema generation on real BrowserGym pages",
    )
    parser.add_argument("--url", default=None,
                        help="BrowserGym URL (repeat with --extra-url to add more)")
    parser.add_argument("--extra-url", action="append", default=[],
                        help="Additional URLs to test (repeatable)")
    parser.add_argument("--goal", default="",
                        help="Task goal for the primary --url")
    parser.add_argument("--save-images", action="store_true",
                        help="Save captured screenshots next to this file")
    args = parser.parse_args()

    if args.url is None and not args.extra_url:
        urls = DEFAULT_URLS
    else:
        urls = []
        if args.url:
            urls.append((args.url, args.goal or f"Explore {args.url}"))
        for u in args.extra_url:
            urls.append((u, f"Explore {u}"))

    out_dir = Path(__file__).parent

    for url, goal in urls:
        try:
            observations = capture_browser_obs(url=url, goal=goal, n_steps=0)
        except Exception as e:
            log.error("Failed to capture %s: %s", url, e)
            continue

        for obs_data in observations:
            step = obs_data["step"]
            domain = url.split("//")[1].split("/")[0].replace(".", "_")

            if args.save_images and obs_data["image"]:
                fname = out_dir / f"real_browser_{domain}_step{step}.png"
                obs_data["image"].save(str(fname))
                log.info("  Saved %s", fname.name)

            log.info("Sending %s step=%d to %s ...", url, step, MODEL)
            result = run_browser_test(obs_data)
            print_result(f"Browser: {url} (step {step})", result)

    print("\n" + "=" * 70)
    print("  Done. All observations came from real BrowserGym pages.")
    print("=" * 70)


if __name__ == "__main__":
    main()
