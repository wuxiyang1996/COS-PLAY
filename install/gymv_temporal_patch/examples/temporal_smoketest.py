"""Smoke test for Temporal/* (stable-retro) envs with the new wrappers.

Loads each registered Temporal env, runs ``reset`` + a handful of steps with
canned actions, and verifies that the multimodal :class:`gym_v.Observation`
contains both an image and a non-trivial text. With ``--save-frames`` the
first frame and a frame-stacked tile from each env are written to
``examples/temporal_smoketest_out/<EnvId>/``.

Usage:
    python examples/temporal_smoketest.py --save-frames

Skips any env whose ROM hasn't been imported into stable-retro.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from PIL import Image

import gym_v
from gym_v.wrappers import (
    FrameStack,
    GrayscaleObservation,
    HistoryRecorder,
    ResizeObservation,
    StochasticFrameSkip,
    TextStateAugmenter,
)


TEMPORAL_IDS = sorted(eid for eid in gym_v.registry if eid.startswith("Temporal/"))

CANNED_ACTIONS = ["NOOP", "RIGHT", "A", "B+RIGHT", "START", "UP+A"]


def _rom_available(env_id: str) -> bool:
    """Return True iff the underlying ROM resolves in stable-retro."""
    import stable_retro  # local import keeps the module importable without it

    spec = gym_v.spec(env_id)
    game = (spec.kwargs or {}).get("game", "")
    if not game:
        return False
    try:
        stable_retro.data.get_romfile_path(game)
        return True
    except FileNotFoundError:
        if game.endswith("-v0"):
            return False
        try:
            stable_retro.data.get_romfile_path(f"{game}-v0")
            return True
        except FileNotFoundError:
            return False


def _build_env(env_id: str, *, with_wrappers: bool):
    """Create the env and optionally stack the new visual + text wrappers."""
    env = gym_v.make(env_id)
    if not with_wrappers:
        return env
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = ResizeObservation(env, size=(160, 112))
    env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStack(env, num_stack=4)
    env = HistoryRecorder(env, include_history_in_info=True, max_turns=64)
    env = TextStateAugmenter(env)
    return env


def _tile_frames(frames, *, gap: int = 4) -> Image.Image:
    if not frames:
        raise ValueError("empty frames list")
    h = max(f.size[1] for f in frames)
    w_total = sum(f.size[0] for f in frames) + gap * (len(frames) - 1)
    canvas = Image.new("RGB", (w_total, h), (0, 0, 0))
    x = 0
    for f in frames:
        canvas.paste(f.convert("RGB"), (x, 0))
        x += f.size[0] + gap
    return canvas


def run_env(env_id: str, *, save_dir: Path | None) -> dict:
    """Run reset + a few steps; return a status dict."""
    result: dict = {"env_id": env_id, "ok": False}

    if not _rom_available(env_id):
        result.update({"skipped": True, "reason": "ROM not imported"})
        return result

    raw_env = _build_env(env_id, with_wrappers=False)
    try:
        obs, info = raw_env.reset(seed=0)
        agent_id = next(iter(obs))
        first_frame = obs[agent_id].image
        first_text = obs[agent_id].text
        if first_frame is None or not first_text:
            raise RuntimeError("reset produced incomplete observation")

        for i, act in enumerate(CANNED_ACTIONS):
            obs, reward, terminated, truncated, info = raw_env.step(
                {agent_id: act}
            )
            if terminated.get("__all__") or truncated.get("__all__"):
                break

        last_text = obs[agent_id].text
        last_meta = obs[agent_id].metadata
        result.update(
            {
                "ok": True,
                "image_size": first_frame.size,
                "text_first": first_text,
                "text_last": last_text,
                "frame_index": last_meta.get("frame_index"),
                "episode_reward": last_meta.get("episode_reward"),
            }
        )

        if save_dir is not None:
            out = save_dir / env_id.replace("/", "_")
            out.mkdir(parents=True, exist_ok=True)
            first_frame.save(out / "00_reset.png")
            obs[agent_id].image.save(out / "01_after_steps.png")
    finally:
        raw_env.close()

    # Now exercise the wrappers on a fresh env.
    wrapped = _build_env(env_id, with_wrappers=True)
    try:
        obs, info = wrapped.reset(seed=0)
        agent_id = next(iter(obs))
        stacked = obs[agent_id].image
        if not isinstance(stacked, list) or len(stacked) != 4:
            raise RuntimeError(
                f"FrameStack should yield list of 4 PIL Images, got {type(stacked)}"
            )
        if any(img.size != (160, 112) for img in stacked):
            raise RuntimeError(
                f"ResizeObservation broken; sizes={[i.size for i in stacked]}"
            )
        for act in CANNED_ACTIONS:
            obs, reward, terminated, truncated, info = wrapped.step(
                {agent_id: act}
            )
            if terminated.get("__all__") or truncated.get("__all__"):
                break

        result["wrapped_text_last"] = obs[agent_id].text
        result["wrapped_image_kind"] = "list[Image]" if isinstance(
            obs[agent_id].image, list
        ) else type(obs[agent_id].image).__name__
        result["wrapped_image_count"] = (
            len(obs[agent_id].image)
            if isinstance(obs[agent_id].image, list)
            else 1
        )

        if save_dir is not None:
            out = save_dir / env_id.replace("/", "_")
            tile = _tile_frames(obs[agent_id].image)
            tile.save(out / "02_wrapped_framestack.png")
    finally:
        wrapped.close()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Write sample frames to examples/temporal_smoketest_out/.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only the listed env ids (default: all Temporal/*).",
    )
    args = parser.parse_args()

    targets = args.only if args.only else TEMPORAL_IDS
    save_dir = (
        Path(__file__).resolve().parent / "temporal_smoketest_out"
        if args.save_frames
        else None
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for env_id in targets:
        print(f"=== {env_id} ===")
        try:
            r = run_env(env_id, save_dir=save_dir)
        except Exception:
            print(traceback.format_exc())
            failures.append(env_id)
            continue
        if r.get("skipped"):
            print(f"  SKIPPED ({r['reason']})")
            continue
        if not r.get("ok"):
            print(f"  FAILED: {r}")
            failures.append(env_id)
            continue
        print(f"  image_size:   {r['image_size']}")
        print(f"  text_first:   {r['text_first']}")
        print(f"  text_last:    {r['text_last']}")
        print(f"  ep_reward:    {r.get('episode_reward')}")
        print(f"  frame_index:  {r.get('frame_index')}")
        print(f"  wrapped img:  {r.get('wrapped_image_kind')} "
              f"x{r.get('wrapped_image_count')}")
        print(f"  wrapped text: {r['wrapped_text_last']}")

    if failures:
        print(f"\n{len(failures)} env(s) failed: {failures}")
        return 1
    print("\nAll envs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
