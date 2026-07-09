#!/usr/bin/env python3
"""Cross-domain schema-generation runner — delegates to per-wrapper variants.

The Gym-V capture/parse logic now lives in :mod:`gymv_wrapper.test_schema_gen`
and the BrowserGym variant in :mod:`browsergym_wrapper.test_schema_gen`.
This shim keeps ``python vlm_wrapper/test_schema_gen.py`` working as a
combined entry point that runs both domains.

Usage:
    # Both domains
    python vlm_wrapper/test_schema_gen.py

    # Only Gym-V
    python vlm_wrapper/test_schema_gen.py --gymv-only

    # Only BrowserGym
    python vlm_wrapper/test_schema_gen.py --browser-only

    # Custom BrowserGym URL
    python vlm_wrapper/test_schema_gen.py --browser-only --url https://en.wikipedia.org

    # Save captured screenshots
    python vlm_wrapper/test_schema_gen.py --save-images
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from browsergym_wrapper.test_schema_gen import (  # noqa: E402
    DEFAULT_URLS as _BROWSER_DEFAULTS,
    capture_browser_obs,
    print_result as _print_browser_result,
    run_browser_test,
)
from gymv_wrapper.test_schema_gen import (  # noqa: E402
    DEFAULT_ENVS as _GYMV_DEFAULTS,
    capture_gymv_obs,
    print_result as _print_gymv_result,
    run_gymv_test,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test GPT-4o schema generation on real env observations",
    )
    parser.add_argument("--gymv-only", action="store_true")
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--url", default="https://www.google.com",
                        help="BrowserGym URL to test")
    parser.add_argument("--goal", default="", help="BrowserGym task goal")
    parser.add_argument("--save-images", action="store_true",
                        help="Save captured screenshots next to the per-wrapper file")
    parser.add_argument("--gymv-steps", type=int, default=2,
                        help="Number of game steps to capture per env")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_gymv = not args.browser_only
    run_browser = not args.gymv_only

    if run_gymv:
        gymv_out_dir = Path(REPO_ROOT) / "gymv_wrapper"
        for env_id in _GYMV_DEFAULTS:
            try:
                observations = capture_gymv_obs(
                    env_id, n_steps=args.gymv_steps, seed=args.seed,
                )
            except Exception as e:
                log.error("Failed to capture %s: %s", env_id, e)
                continue

            for obs_data in observations:
                step = obs_data["step"]
                tag = env_id.replace("/", "_")
                if args.save_images and obs_data["image"]:
                    fname = gymv_out_dir / f"real_{tag}_step{step}.png"
                    obs_data["image"].save(str(fname))
                    log.info("  Saved %s", fname.name)
                log.info("Sending %s step=%d ...", env_id, step)
                result = run_gymv_test(obs_data)
                _print_gymv_result(f"Gym-V: {env_id} (step {step})", result)

    if run_browser:
        browser_out_dir = Path(REPO_ROOT) / "browsergym_wrapper"

        if args.browser_only or args.url != "https://www.google.com":
            urls = [(args.url, args.goal or f"Explore {args.url}")]
        else:
            urls = list(_BROWSER_DEFAULTS)

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
                    fname = browser_out_dir / f"real_browser_{domain}_step{step}.png"
                    obs_data["image"].save(str(fname))
                    log.info("  Saved %s", fname.name)
                log.info("Sending %s step=%d ...", url, step)
                result = run_browser_test(obs_data)
                _print_browser_result(f"Browser: {url} (step {step})", result)

    print("\n" + "=" * 70)
    print("  Done. All observations came from real environments.")
    print("=" * 70)


if __name__ == "__main__":
    main()
