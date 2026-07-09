#!/usr/bin/env python
"""Collect ``(frame, heuristic_schema, gpt4o_schema)`` triples from Gym-V.

Drives one or more Gym-V environments for a fixed number of steps each,
captures the rendered frame plus the text observation, and produces:

1. The **heuristic schema** via ``gymv_wrapper.heuristic.text_to_schema``
   — the cheap, deterministic parse of ``obs.text`` + ``env.description``.
2. The **GPT-5.5 vision schema** via
   ``gymv_wrapper.adapter.generate_label`` — the slow, expensive
   teacher that grounds purely from pixels.

Both schemas are stored alongside the saved frame so cross-validation
can run later without burning more API quota.

Usage::

    export OPENAI_API_KEY=sk-...
    export VLM_LABEL_MODEL=gpt-5.5
    conda activate vlm_benchmarks   # (or gymv if you prefer)

    python -m labeling.grounding.collect_gymv \\
        --envs Games/Game2048-v0 Games/Sokoban-v0 \\
        --episodes 3 --max_steps 12 \\
        --output_root labeling/output/grounding/gymv

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("labeling.grounding.collect_gymv")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gymv_wrapper.heuristic import text_to_schema  # noqa: E402


# ----------------------------------------------------------------------
# Per-step record
# ----------------------------------------------------------------------

@dataclass
class GymVTriple:
    """One ``(frame, heuristic, vision)`` triple from a Gym-V step.

    ``frame_path`` is repo-relative for portability across hosts;
    ``error`` is non-empty when the vision-LLM call failed (we still keep
    the frame + heuristic so cross-validation can compute heuristic-only
    coverage).
    """

    env_id: str
    episode: int
    step: int
    seed: int
    obs_text: str
    valid_actions: list[str]
    description: str
    frame_path: str
    heuristic_schema: str
    vision_schema: str | None = None
    vision_warnings: list[str] | None = None
    vision_model: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Episode driver
# ----------------------------------------------------------------------

def _make_env(env_id: str):
    """Lazily import gym_v so this script can ``--help`` without it."""
    try:
        import gym_v  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "gym_v is not installed in this env.  Activate `gymv` (see "
            "install/INSTALL_BENCHMARKS.md §1) or install gym-v in the "
            "vlm_benchmarks env."
        ) from exc
    return gym_v.make(env_id)


def _save_frame(image, path: Path) -> None:
    """Write a PIL image to disk, creating intermediate dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _normalise_obs(obs: Any) -> tuple[Any, str, list[str]]:
    """Return ``(image, text, valid_actions)`` from a Gym-V observation.

    Gym-V wraps multi-agent obs as ``{"agent_0": Observation(...)}``;
    older single-agent envs return the Observation directly.
    """
    if isinstance(obs, dict):
        first_key = next(iter(obs))
        obs = obs[first_key]
    image = getattr(obs, "image", None)
    text = getattr(obs, "text", "") or ""
    valid_actions = list(
        getattr(obs, "metadata", {}).get("valid_actions", []) or []
    )
    return image, text, valid_actions


def collect_one_episode(
    env_id: str,
    *,
    episode_idx: int,
    seed: int,
    max_steps: int,
    output_root: Path,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    skip_vision: bool,
    max_entities: int = 20,
) -> list[GymVTriple]:
    """Run one episode and return its triples (also written to disk)."""
    env = _make_env(env_id)
    triples: list[GymVTriple] = []

    description = getattr(env, "description", "") or ""
    safe_env = env_id.replace("/", "_")
    ep_dir = output_root / safe_env / f"ep_{episode_idx:03d}"
    frames_dir = ep_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    triples_path = ep_dir / "triples.jsonl"
    fh = open(triples_path, "a", encoding="utf-8")

    try:
        obs, _info = env.reset(seed=seed)
        for step_idx in range(max_steps):
            image, text, valid_actions = _normalise_obs(obs)
            if image is None:
                logger.warning(
                    "%s ep%d step%d: no image in obs, skipping",
                    env_id, episode_idx, step_idx,
                )
                break

            frame_path = frames_dir / f"step_{step_idx:03d}.png"
            _save_frame(image, frame_path)

            heuristic = text_to_schema(
                obs_text=text,
                description=description,
                task_id=env_id,
                step=step_idx,
                max_entities=max_entities,
                include_actions=True,
            )

            vision_schema: str | None = None
            vision_warnings: list[str] | None = None
            vision_model: str | None = None
            error: str | None = None
            t0 = time.time()
            if not skip_vision:
                try:
                    from gymv_wrapper.adapter import generate_label
                    out = generate_label(
                        image=image,
                        goal="",
                        task_id=env_id,
                        step=step_idx,
                        game_rules=description,
                        obs_text=text,
                        valid_actions=valid_actions,
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
                        env_id, episode_idx, step_idx, error,
                    )

            triple = GymVTriple(
                env_id=env_id,
                episode=episode_idx,
                step=step_idx,
                seed=seed,
                obs_text=text,
                valid_actions=valid_actions,
                description=description,
                frame_path=str(
                    frame_path.relative_to(_REPO_ROOT)
                    if str(frame_path).startswith(str(_REPO_ROOT))
                    else frame_path
                ),
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

            # Step the env.  We pick a deterministic action so episodes
            # are reproducible — Phase-0 just needs *coverage*, not
            # optimal play.
            action = (
                valid_actions[step_idx % len(valid_actions)]
                if valid_actions else 0
            )
            try:
                step_out = env.step({"agent_0": action})
            except Exception:
                # Single-agent fallback (older Gym-V envs).
                step_out = env.step(action)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
            done = (
                step_out[2] if isinstance(step_out, tuple) and len(step_out) > 2
                else False
            )
            if done:
                logger.info(
                    "%s ep%d ended at step %d", env_id, episode_idx, step_idx,
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
        description="Collect (frame, heuristic, vision) triples from Gym-V",
    )
    p.add_argument(
        "--envs", nargs="+", required=True,
        help="Gym-V env ids, e.g. Games/Game2048-v0 Games/Sokoban-v0",
    )
    p.add_argument("--episodes", type=int, default=3,
                   help="Episodes per env")
    p.add_argument("--max_steps", type=int, default=12)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--max_entities", type=int, default=20)
    p.add_argument(
        "--output_root", default="labeling/output/grounding/gymv",
        help="Root directory for the collected triples + frames",
    )
    p.add_argument(
        "--model", default=os.environ.get("VLM_LABEL_MODEL", "gpt-5.5"),
    )
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument(
        "--skip_vision", action="store_true",
        help="Only emit heuristic schemas — useful for dry runs without API "
             "quota.",
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

    summary = {
        "envs": [],
        "n_triples": 0,
        "n_with_vision": 0,
        "n_errors": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for env_id in args.envs:
        env_summary: dict[str, Any] = {
            "env_id": env_id,
            "n_episodes": 0,
            "n_steps": 0,
            "n_with_vision": 0,
            "n_errors": 0,
        }
        for ep in range(args.episodes):
            seed = args.seed_start + ep
            try:
                triples = collect_one_episode(
                    env_id,
                    episode_idx=ep,
                    seed=seed,
                    max_steps=args.max_steps,
                    output_root=output_root,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    skip_vision=args.skip_vision,
                    max_entities=args.max_entities,
                )
            except Exception as exc:
                logger.error(
                    "%s ep%d failed: %s: %s",
                    env_id, ep, type(exc).__name__, exc,
                )
                env_summary["n_errors"] += 1
                continue
            env_summary["n_episodes"] += 1
            env_summary["n_steps"] += len(triples)
            env_summary["n_with_vision"] += sum(
                1 for t in triples if t.vision_schema
            )
            env_summary["n_errors"] += sum(1 for t in triples if t.error)

        summary["envs"].append(env_summary)
        summary["n_triples"] += env_summary["n_steps"]
        summary["n_with_vision"] += env_summary["n_with_vision"]
        summary["n_errors"] += env_summary["n_errors"]

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
