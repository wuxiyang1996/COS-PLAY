"""Held-out gymv evaluation driver (block C6).

Reuses the trainer's :func:`run_episode_async` so the actor + harness +
bank wiring exactly mirrors training, then aggregates per-episode
rewards into a uniform ``eval_result.json`` for the cross-domain
benchmark table.

Usage::

    python -m scripts.skillbridge_eval.eval_gymv \\
        --run-dir runs/Qwen3.5-9B_20260504_144712 \\
        --games gymv_columns gymv_dynamite_headdy \\
        --episodes-per-game 5 \\
        --max-steps 200 \\
        --vllm-base-url http://localhost:8000/v1 \\
        --output runs/.../eval/gymv_result.json

The trainer must NOT be running on the same vLLM endpoint — start a
dedicated eval-only vLLM with the LoRA adapters from the run's
``lora_adapters/`` directory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Trainer run dir (must contain skillbank/ and lora_adapters/).",
    )
    p.add_argument(
        "--bank-dir", type=Path, default=None,
        help="Skill bank dir.  Defaults to <run-dir>/skillbank.",
    )
    p.add_argument(
        "--games", nargs="+", required=True,
        help="Held-out game slugs to evaluate.  Each must be a valid "
             "trainer game slug (gymv_*, etc.).",
    )
    p.add_argument(
        "--episodes-per-game", type=int, default=5,
        help="Number of evaluation episodes per game.",
    )
    p.add_argument(
        "--max-steps", type=int, default=200,
        help="Per-episode max env steps.",
    )
    p.add_argument(
        "--vllm-base-url", type=str, default="http://localhost:8000/v1",
        help="Already-running vLLM endpoint (with LoRA adapters loaded).",
    )
    p.add_argument(
        "--vllm-base-urls", nargs="*", default=None,
        help="Multiple vLLM endpoints (overrides --vllm-base-url for "
             "load balancing).",
    )
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-9B",
        help="Base model identifier.",
    )
    p.add_argument(
        "--harness-mode",
        choices=["full", "plain-text-skills", "off"],
        default="full",
        help="Block-B1 harness ablation switch.",
    )
    p.add_argument(
        "--actor-bank-cap-k", type=int, default=0,
        help="Block-B5 actor-bank-cap-K (0 = no cap).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.3,
    )
    p.add_argument(
        "--max-concurrent", type=int, default=4,
        help="Max concurrent episodes (vLLM batch capacity bound).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path.  Defaults to <run-dir>/eval/gymv_result_"
             "<timestamp>.json.",
    )
    p.add_argument(
        "--label", type=str, default="skillbridge",
        help="Label for this eval run (e.g. 'skillbridge', "
             "'baseline', 'w/o-harness').  Stored in the result JSON.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _run_one_episode(
    *,
    game: str,
    episode_idx: int,
    max_steps: int,
    vllm_client: Any,
    skill_banks: Dict[str, Any],
    harness_hooks: Dict[str, Any],
    temperature: float,
    actor_bank_cap_k: int,
    thread_executor: ThreadPoolExecutor,
) -> Dict[str, Any]:
    """Run a single eval episode and return a compact summary dict."""
    from trainer.coevolution.episode_runner import run_episode_async

    t0 = time.monotonic()
    result = None
    try:
        result = await run_episode_async(
            game=game,
            max_steps=max_steps,
            vllm_client=vllm_client,
            skill_bank=skill_banks.get(game),
            temperature=temperature,
            executor=thread_executor,
            harness_hook=harness_hooks.get(game),
            # Eval mode: every-step intention regen (matches v12 prod).
            intention_trigger="every-step",
            actor_bank_cap_k=actor_bank_cap_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("eval_gymv: episode %s/%d crashed: %s",
                         game, episode_idx, exc)
        return {
            "game": game,
            "episode_idx": episode_idx,
            "error": repr(exc),
            "wall_time_s": time.monotonic() - t0,
            "total_reward": 0.0,
            "steps": 0,
            "success": False,
        }

    summary = {
        "game": game,
        "episode_idx": episode_idx,
        "episode_id": getattr(result, "episode_id", ""),
        "total_reward": float(getattr(result, "total_reward", 0.0)),
        "raw_env_total_reward": float(getattr(result, "raw_env_total_reward", 0.0)),
        "steps": int(getattr(result, "steps", 0)),
        "success": bool(getattr(result, "success", False)),
        "terminated": bool(getattr(result, "terminated", False)),
        "wall_time_s": time.monotonic() - t0,
    }
    return summary


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    from trainer.coevolution.vllm_client import AsyncVLLMClient
    from trainer.coevolution._harness_hook import SkillHarnessHook
    from scripts.skillbridge_eval.eval_actor import _load_skill_bank

    bank_dir = args.bank_dir or (args.run_dir / "skillbank")
    if not bank_dir.exists():
        logger.warning("Bank dir %s missing — running cold-start mode", bank_dir)

    base_urls = args.vllm_base_urls or [args.vllm_base_url]
    vllm_client = AsyncVLLMClient(
        base_urls=base_urls,
        model=args.model,
        default_temperature=args.temperature,
        default_max_tokens=512,
    )

    # Per-game skill bank — a single merged engine drives all games for
    # simplicity (the underlying SkillQueryEngine doesn't gate by game,
    # the actor uses ``game_name`` as a soft RAG hint).
    merged_engine = _load_skill_bank(bank_dir)
    skill_banks: Dict[str, Any] = {g: merged_engine for g in args.games}

    # Per-game harness hook — uses the per-game bank.jsonl when present.
    harness_hooks: Dict[str, Any] = {}
    if args.harness_mode != "off":
        for g in args.games:
            bank_path = bank_dir / g / "skill_bank.jsonl"
            if not bank_path.exists():
                continue
            try:
                harness_hooks[g] = SkillHarnessHook.for_game(
                    game=g,
                    bank_path=bank_path,
                    domain="gymv",
                    allow_shadow=True,
                    mode=args.harness_mode,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("harness hook for %s failed: %s", g, exc)

    thread_executor = ThreadPoolExecutor(max_workers=max(1, args.max_concurrent))
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def _run_with_sem(g: str, idx: int) -> Dict[str, Any]:
        async with semaphore:
            return await _run_one_episode(
                game=g,
                episode_idx=idx,
                max_steps=args.max_steps,
                vllm_client=vllm_client,
                skill_banks=skill_banks,
                harness_hooks=harness_hooks,
                temperature=args.temperature,
                actor_bank_cap_k=args.actor_bank_cap_k,
                thread_executor=thread_executor,
            )

    coros = [
        _run_with_sem(g, i)
        for g in args.games
        for i in range(args.episodes_per_game)
    ]
    logger.info(
        "eval_gymv: launching %d episodes (%d games × %d eps/game)",
        len(coros), len(args.games), args.episodes_per_game,
    )
    t0 = time.time()
    rows = await asyncio.gather(*coros)
    wall = time.time() - t0
    thread_executor.shutdown(wait=False)

    # Aggregate per-game stats.
    per_game: Dict[str, Dict[str, Any]] = {}
    for g in args.games:
        gr = [r for r in rows if r["game"] == g and "error" not in r]
        rewards = [r["total_reward"] for r in gr]
        steps = [r["steps"] for r in gr]
        successes = [r["success"] for r in gr]
        per_game[g] = {
            "n_episodes": len(gr),
            "mean_reward": float(mean(rewards)) if rewards else 0.0,
            "std_reward": float(stdev(rewards)) if len(rewards) > 1 else 0.0,
            "mean_steps": float(mean(steps)) if steps else 0.0,
            "success_rate": (sum(successes) / len(successes)) if successes else 0.0,
            "n_errors": sum(1 for r in rows if r["game"] == g and "error" in r),
        }

    overall = {
        "n_episodes_total": sum(p["n_episodes"] for p in per_game.values()),
        "mean_reward": float(mean([
            r["total_reward"] for r in rows if "error" not in r
        ])) if rows else 0.0,
        "success_rate_macro": float(mean([
            p["success_rate"] for p in per_game.values()
        ])) if per_game else 0.0,
    }

    return {
        "schema_version": 1,
        "domain": "gymv",
        "label": args.label,
        "run_dir": str(args.run_dir),
        "bank_dir": str(bank_dir) if bank_dir else None,
        "model": args.model,
        "harness_mode": args.harness_mode,
        "actor_bank_cap_k": args.actor_bank_cap_k,
        "episodes_per_game": args.episodes_per_game,
        "max_steps": args.max_steps,
        "wall_time_s": wall,
        "per_game": per_game,
        "overall": overall,
        "rows": rows,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.run_dir.exists():
        logger.error("run-dir %s does not exist", args.run_dir)
        return 1

    try:
        result = asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    out_path = args.output
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = args.run_dir / "eval" / f"gymv_result_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("eval_gymv: wrote %s", out_path)

    # Print a compact summary.
    print("\n=== gymv eval summary ===")
    print(f"label    : {args.label}")
    print(f"games    : {', '.join(args.games)}")
    print(f"episodes : {args.episodes_per_game} per game")
    for g, stats in result["per_game"].items():
        print(
            f"  {g}: n={stats['n_episodes']:3d} "
            f"mean_r={stats['mean_reward']:7.2f} "
            f"std_r={stats['std_reward']:6.2f} "
            f"steps={stats['mean_steps']:5.1f} "
            f"sr={stats['success_rate']:.2%}"
        )
    print(
        f"overall  : mean_r={result['overall']['mean_reward']:.2f} "
        f"sr_macro={result['overall']['success_rate_macro']:.2%} "
        f"wall={result['wall_time_s']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
