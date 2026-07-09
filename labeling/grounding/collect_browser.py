#!/usr/bin/env python
"""Collect ``(frame, heuristic_schema, gpt4o_schema)`` triples from BrowserGym.

For each browser env / step we capture the rendered screenshot plus the
AXTree, then produce:

1. The **heuristic schema** via ``vlm_wrapper.browser_heuristic.obs_to_schema``
   (deterministic AXTree → schema parse).
2. The **GPT-5.5 vision schema** via
   ``vlm_wrapper.browser_adapter.browser_obs_to_schema`` (vision-first
   teacher).

This is the BrowserGym counterpart to ``collect_gymv.py``.

Usage::

    export OPENAI_API_KEY=sk-...
    export VLM_LABEL_MODEL=gpt-5.5
    conda activate browsergym

    python -m labeling.grounding.collect_browser \\
        --tasks miniwob.click-test miniwob.search-engine \\
        --episodes 2 --max_steps 8 \\
        --output_root labeling/output/grounding/browser

The output layout matches what the Phase-1 ``schema_gen`` SFT data
loader (``trainer.SFT.schema_gen.data_loader``) expects.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("labeling.grounding.collect_browser")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vlm_wrapper.browser_heuristic import obs_to_schema  # noqa: E402


# ----------------------------------------------------------------------
# Per-step record
# ----------------------------------------------------------------------

@dataclass
class BrowserTriple:
    task_id: str
    episode: int
    step: int
    seed: int
    url: str
    goal: str
    frame_path: str
    axtree_text: str
    heuristic_schema: str
    vision_schema: str | None = None
    vision_warnings: list[str] | None = None
    vision_model: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_env(task_id: str, *, headless: bool = True):
    """Lazily import ``browsergym`` so ``--help`` works without it."""
    try:
        import browsergym  # noqa: F401  (registers tasks)
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "browsergym / gymnasium not installed.  Activate the "
            "`browsergym` env (install/INSTALL_BENCHMARKS.md §2)."
        ) from exc
    # BrowserGym task ids are typically registered as ``browsergym/<id>``.
    name = task_id if task_id.startswith("browsergym/") else f"browsergym/{task_id}"
    return gym.make(name, headless=headless)


def _flatten_axtree(obs: dict[str, Any]) -> str:
    """Return a truncated AXTree string suitable for prompt context."""
    axt = obs.get("axtree_object")
    if not axt:
        return ""
    try:
        from browsergym.utils.obs import flatten_axtree_to_str  # type: ignore
        return flatten_axtree_to_str(axt)
    except Exception:
        # Fall back to a JSON dump — much chunkier but still parseable.
        try:
            return json.dumps(axt)[:4000]
        except Exception:
            return str(axt)[:4000]


def _save_screenshot(obs: dict[str, Any], path: Path) -> bool:
    """Persist the screenshot to ``path``; return True if written."""
    shot = obs.get("screenshot")
    if shot is None:
        return False
    if isinstance(shot, np.ndarray):
        img = Image.fromarray(shot)
    elif isinstance(shot, Image.Image):
        img = shot
    else:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return True


def _extract_goal(obs: dict[str, Any]) -> str:
    if obs.get("goal"):
        return obs["goal"]
    parts = obs.get("goal_object") or []
    return " ".join(
        m.get("text", "") for m in parts if isinstance(m, dict)
        and m.get("type") == "text"
    )


# ----------------------------------------------------------------------
# Episode driver
# ----------------------------------------------------------------------

def collect_one_episode(
    task_id: str,
    *,
    episode_idx: int,
    seed: int,
    max_steps: int,
    output_root: Path,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    skip_vision: bool,
    headless: bool,
    max_entities: int = 25,
) -> list[BrowserTriple]:
    env = _make_env(task_id, headless=headless)
    triples: list[BrowserTriple] = []

    safe_task = task_id.replace("/", "_").replace(".", "_")
    ep_dir = output_root / safe_task / f"ep_{episode_idx:03d}"
    frames_dir = ep_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    triples_path = ep_dir / "triples.jsonl"
    fh = open(triples_path, "a", encoding="utf-8")

    try:
        obs, _info = env.reset(seed=seed)
        for step_idx in range(max_steps):
            frame_path = frames_dir / f"step_{step_idx:03d}.png"
            ok = _save_screenshot(obs, frame_path)
            if not ok:
                logger.warning(
                    "%s ep%d step%d: no screenshot in obs — skipping",
                    task_id, episode_idx, step_idx,
                )
                break

            axtree_text = _flatten_axtree(obs)
            heuristic = obs_to_schema(
                obs, step=step_idx, task_id=task_id,
                max_entities=max_entities,
            )

            vision_schema: str | None = None
            vision_warnings: list[str] | None = None
            vision_model: str | None = None
            error: str | None = None
            t0 = time.time()
            if not skip_vision:
                try:
                    from vlm_wrapper.browser_adapter import (
                        browser_obs_to_schema,
                    )
                    out = browser_obs_to_schema(
                        obs,
                        step=step_idx,
                        task_id=task_id,
                        axtree_text=axtree_text,
                        max_entities=max_entities,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    vision_schema = out.get("schema")
                    vision_warnings = out.get("warnings")
                    vision_model = out.get("model")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "%s ep%d step%d: vision-LLM call failed: %s",
                        task_id, episode_idx, step_idx, error,
                    )

            triple = BrowserTriple(
                task_id=task_id,
                episode=episode_idx,
                step=step_idx,
                seed=seed,
                url=obs.get("url", ""),
                goal=_extract_goal(obs),
                frame_path=str(
                    frame_path.relative_to(_REPO_ROOT)
                    if str(frame_path).startswith(str(_REPO_ROOT))
                    else frame_path
                ),
                axtree_text=axtree_text[:4000],
                heuristic_schema=heuristic,
                vision_schema=vision_schema,
                vision_warnings=vision_warnings,
                vision_model=vision_model,
                error=error,
                elapsed_s=round(time.time() - t0, 2),
            )
            triples.append(triple)
            fh.write(json.dumps(triple.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

            # Step the env with a no-op so we just sweep the rendered
            # state; a richer episode could replay teacher actions, but
            # for grounding-label collection coverage of *frames* is
            # what matters.
            try:
                step_out = env.step("noop()")
            except Exception:
                # Some BrowserGym envs require a structured action — try
                # the simplest move that the dispatcher always accepts.
                try:
                    step_out = env.step("scroll(0, 100)")
                except Exception as exc:
                    logger.info(
                        "%s ep%d step%d: env.step failed (%s); ending episode",
                        task_id, episode_idx, step_idx, exc,
                    )
                    break
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            done = (
                step_out[2] if isinstance(step_out, tuple) and len(step_out) > 2
                else False
            )
            if done:
                logger.info(
                    "%s ep%d ended at step %d", task_id, episode_idx, step_idx,
                )
                break
    finally:
        fh.close()
        try:
            env.close()
        except Exception:
            pass

    return triples


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect (frame, heuristic, vision) triples from BrowserGym",
    )
    p.add_argument(
        "--tasks", nargs="+", required=True,
        help="BrowserGym task ids, e.g. miniwob.click-test "
             "webarena.shopping.143",
    )
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=8)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_entities", type=int, default=25)
    p.add_argument(
        "--output_root", default="labeling/output/grounding/browser",
    )
    p.add_argument(
        "--model", default=os.environ.get("VLM_LABEL_MODEL", "gpt-5.5"),
    )
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument(
        "--skip_vision", action="store_true",
        help="Only emit heuristic schemas (no vision-LLM calls).",
    )
    p.add_argument(
        "--no_headless", action="store_true",
        help="Run browser with a visible window (debugging).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_vision and not args.api_key:
        logger.warning(
            "OPENAI_API_KEY is not set and --skip_vision is False — vision "
            "calls will fail.  Pass --skip_vision to collect heuristic only."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "tasks": [],
        "n_triples": 0,
        "n_with_vision": 0,
        "n_errors": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for task_id in args.tasks:
        task_summary: dict[str, Any] = {
            "task_id": task_id,
            "n_episodes": 0,
            "n_steps": 0,
            "n_with_vision": 0,
            "n_errors": 0,
        }
        for ep in range(args.episodes):
            seed = args.seed_start + ep
            try:
                triples = collect_one_episode(
                    task_id,
                    episode_idx=ep,
                    seed=seed,
                    max_steps=args.max_steps,
                    output_root=output_root,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    skip_vision=args.skip_vision,
                    headless=not args.no_headless,
                    max_entities=args.max_entities,
                )
            except Exception as exc:
                logger.error(
                    "%s ep%d failed: %s: %s",
                    task_id, ep, type(exc).__name__, exc,
                )
                task_summary["n_errors"] += 1
                continue
            task_summary["n_episodes"] += 1
            task_summary["n_steps"] += len(triples)
            task_summary["n_with_vision"] += sum(
                1 for t in triples if t.vision_schema
            )
            task_summary["n_errors"] += sum(1 for t in triples if t.error)

        summary["tasks"].append(task_summary)
        summary["n_triples"] += task_summary["n_steps"]
        summary["n_with_vision"] += task_summary["n_with_vision"]
        summary["n_errors"] += task_summary["n_errors"]

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(
        "Done — %d triples (%d with vision, %d errors).  Summary at %s",
        summary["n_triples"], summary["n_with_vision"],
        summary["n_errors"], summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
