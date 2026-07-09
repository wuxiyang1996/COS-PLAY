#!/usr/bin/env python
"""R10 — Multi-LoRA noise sweep using the training-time inference path.

Unlike ``scripts.run_qwen3_8b_eval`` (which makes a single LLM call per step
against the ``action_taking`` LoRA and uses retrieval-only skill selection),
this driver invokes :func:`trainer.coevolution.episode_runner.run_episode_async`,
which is the same orchestrator used during training rollouts.  It routes calls
to ``skill_selection``, ``action_taking``, ``segment``, ``contract``, and
``curator`` LoRAs as appropriate via :class:`AsyncVLLMClient`.

Noise is injected via the existing
``decision_agents.agent_helper._apply_summary_noise`` hook (env-var gated by
``COSPLAY_SUMMARY_NOISE``), which is invoked inside ``build_rag_summary``;
the training rollout path also calls ``build_rag_summary`` per step, so the
hook fires automatically.

Usage::

    python -m ablation_study.r10_noise_sweep_multilora \\
        --vllm-url http://127.0.0.1:8061/v1 \\
        --bank runs/Qwen3-8B_20260321_213813_\\(Candy_crush\\)/best/banks/candy_crush/skill_bank.jsonl \\
        --episodes 8 --max-steps 200 \\
        --output ablation_study/output/r10_multilora_<ts>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMINGAGENT_ROOT = REPO_ROOT.parent / "GamingAgent"
for p in (REPO_ROOT, GAMINGAGENT_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from trainer.coevolution.vllm_client import AsyncVLLMClient  # noqa: E402
from trainer.coevolution.episode_runner import run_episode_async  # noqa: E402
from skill_agents.skill_bank.bank import SkillBankMVP  # noqa: E402


NOISE_CELLS: List[Tuple[str, str]] = [
    ("N0", ""),
    ("N2", "dropout:p=0.25"),
    ("N3", "dropout:p=0.5"),
    ("F-board", "drop:board"),
    ("NUM", "num_noise:0.2"),
]


def _aggregate(rewards: List[float]) -> Dict[str, float]:
    n = len(rewards)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci95": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if n > 1 else 0.0
    se = std / (n ** 0.5) if n > 1 else 0.0
    return {
        "n": n,
        "mean": float(mean),
        "std": float(std),
        "ci95": float(1.96 * se),
        "min": float(min(rewards)),
        "max": float(max(rewards)),
    }


async def _run_cell(
    *,
    name: str,
    noise: str,
    episodes: int,
    max_steps: int,
    vllm_url: str,
    bank: Any,
    temperature: float,
    seed: int,
    concurrency: int,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Run *episodes* candy_crush rollouts with *noise* applied to summaries."""
    os.environ["COSPLAY_SUMMARY_NOISE"] = noise
    os.environ["COSPLAY_SUMMARY_NOISE_SEED"] = str(seed)

    client = AsyncVLLMClient(
        base_url=vllm_url,
        model="Qwen/Qwen3-8B",
        default_temperature=temperature,
        default_max_tokens=512,
        timeout=180.0,
    )
    exe = ThreadPoolExecutor(max_workers=min(episodes, 8))

    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int):
        async with sem:
            t0 = time.monotonic()
            try:
                res = await run_episode_async(
                    game="candy_crush",
                    max_steps=max_steps,
                    vllm_client=client,
                    skill_bank=bank,
                    temperature=temperature,
                    executor=exe,
                )
                wall = time.monotonic() - t0
                return {
                    "idx": idx,
                    "episode_id": res.episode_id,
                    "steps": res.steps,
                    "total_reward": float(res.total_reward),
                    "terminated": bool(res.terminated),
                    "truncated": bool(res.truncated),
                    "skill_switches": int(res.skill_switches),
                    "wall_s": round(wall, 2),
                }
            except Exception as exc:
                logging.exception("[%s] episode %d failed: %s", name, idx, exc)
                return {"idx": idx, "error": str(exc)}

    results = await asyncio.gather(*[_one(i) for i in range(episodes)])
    rewards = [r["total_reward"] for r in results if "total_reward" in r]
    return rewards, results


def main():
    ap = argparse.ArgumentParser(description="R10 multi-LoRA noise sweep")
    ap.add_argument("--vllm-url", required=True,
                    help="vLLM OpenAI base URL with all 5 LoRAs loaded "
                         "(e.g. http://127.0.0.1:8061/v1)")
    ap.add_argument("--bank", required=True, help="Path to skill_bank.jsonl")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Max concurrent episodes per cell")
    ap.add_argument("--cells", nargs="+",
                    default=["N0", "N2", "N3", "F-board", "NUM"],
                    help="Subset of cells to run")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    bank = SkillBankMVP(path=args.bank)
    bank.load(args.bank)
    print(f"[r10-mlora] bank loaded: {len(bank)} skills  ({args.bank})")

    summary: Dict[str, Any] = {
        "config": vars(args),
        "cells": {},
    }

    sweep_t0 = time.time()
    for name, noise in NOISE_CELLS:
        if name not in args.cells:
            continue
        print(f"\n══════════════════════════════════════════════════════════════")
        print(f"  Cell {name}   noise='{noise}'")
        print(f"══════════════════════════════════════════════════════════════")
        t0 = time.time()
        rewards, per_ep = asyncio.run(_run_cell(
            name=name,
            noise=noise,
            episodes=args.episodes,
            max_steps=args.max_steps,
            vllm_url=args.vllm_url,
            bank=bank,
            temperature=args.temperature,
            seed=args.seed,
            concurrency=args.concurrency,
        ))
        stat = _aggregate(rewards)
        stat["wall_s"] = round(time.time() - t0, 1)

        cell_record = {
            "cell": name,
            "noise_spec": noise,
            **stat,
            "rewards": rewards,
            "episodes": per_ep,
        }
        with open(out / f"{name}.json", "w") as f:
            json.dump(cell_record, f, indent=2)

        summary["cells"][name] = {
            "noise_spec": noise,
            **stat,
            "rewards": rewards,
        }

        print(
            f"[r10-mlora] {name}: n={stat['n']} "
            f"mean={stat['mean']:.2f} ±{stat['ci95']:.2f} "
            f"(min={stat['min']:.1f}, max={stat['max']:.1f}, {stat['wall_s']}s)"
        )

    summary["sweep_wall_s"] = round(time.time() - sweep_t0, 1)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    md = ["# R10 Multi-LoRA Noise Sweep\n",
          f"- Bank: `{args.bank}`",
          f"- Episodes/cell: {args.episodes},  max_steps={args.max_steps}",
          f"- Temperature: {args.temperature},  seed={args.seed}",
          "",
          "| Cell | Noise spec | n | mean | ±95% CI | min | max |",
          "|------|------------|---|------|---------|-----|-----|"]
    for name, _ in NOISE_CELLS:
        if name not in summary["cells"]:
            continue
        s = summary["cells"][name]
        md.append(
            f"| {name} | `{s['noise_spec'] or '(none)'}` | "
            f"{s['n']} | {s['mean']:.2f} | ±{s['ci95']:.2f} | "
            f"{s['min']:.1f} | {s['max']:.1f} |"
        )
    (out / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\n[r10-mlora] Done in {summary['sweep_wall_s']}s. → {out}/summary.json")


if __name__ == "__main__":
    main()
