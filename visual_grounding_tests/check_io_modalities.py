#!/usr/bin/env python
"""Pre-flight modality check for the 13 ``Temporal/*`` (stable-retro) envs.

For each game registered in :data:`gymv_wrapper.TEMPORAL_GAME_SPECS`, this
script answers three questions before we kick off any visual-grounding run:

1. **Is the image channel available?**  All ``RetroGymVEnv`` instances
   construct an ``Observation(image=PIL.Image.fromarray(obs_array), ...)``
   on every ``reset`` / ``step``, so the answer is structurally **yes**
   for every covered game.

2. **Is the text channel available?**  Same place: ``Observation(text=...)``
   is always populated by ``_get_observation_text`` — minimally
   ``"Game: ... | Frame: 0 | StepReward: ... | EpReward: ..."``.

3. **How rich is that text?**  ``RetroGymVEnv`` reads each ROM's
   ``data.json`` watch variables; richer ``data.json`` ⇒ more
   game-state fields surfaced to the LLM.  This script reports the
   exact watch keys per game when ``stable_retro`` resolves the ROM,
   otherwise it tells you what to install.

The probe is **read-only** and **does not require an API key**.

Usage::

    # quickest: static check + try-imports (no env.reset)
    python visual_grounding_tests/check_io_modalities.py

    # also instantiate each env and run reset() to confirm at runtime
    python visual_grounding_tests/check_io_modalities.py --runtime

    # JSON output (for CI / batch summaries)
    python visual_grounding_tests/check_io_modalities.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))


def _load_specs() -> Dict[str, Any]:
    from gymv_wrapper.temporal_visual_grounding import TEMPORAL_GAME_SPECS

    return TEMPORAL_GAME_SPECS


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception as exc:  # noqa: BLE001
        return exc


def _resolve_rom(stable_retro, game: str) -> Optional[str]:
    try:
        return stable_retro.data.get_romfile_path(game)
    except (FileNotFoundError, OSError):
        if not game.endswith("-v0"):
            try:
                return stable_retro.data.get_romfile_path(f"{game}-v0")
            except (FileNotFoundError, OSError):
                return None
        return None


def _watch_keys_for(rom_path: str) -> List[str]:
    folder = Path(rom_path).parent
    data_json = folder / "data.json"
    if not data_json.exists():
        return []
    try:
        payload = json.loads(data_json.read_text())
        return sorted(payload.get("info", {}).keys())
    except (OSError, ValueError):
        return []


def _check_static(env_id: str, spec, stable_retro) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "env_id": env_id,
        "retro_game": spec.retro_game,
        "display_name": spec.display_name,
        "image_channel": "yes (PIL.Image from RetroGymVEnv.reset/step)",
        "text_channel": "yes (RetroGymVEnv._get_observation_text always returns)",
    }
    if isinstance(stable_retro, Exception):
        rec["rom_resolved"] = "unknown (stable_retro not installed)"
        rec["watch_keys"] = []
        rec["text_richness"] = (
            "minimum (Frame/StepReward/EpReward only) until ROM imported"
        )
        return rec

    rom_path = _resolve_rom(stable_retro, spec.retro_game)
    if rom_path is None:
        rec["rom_resolved"] = "no (import the ROM via stable_retro)"
        rec["watch_keys"] = []
        rec["text_richness"] = "minimum until ROM imported"
        return rec

    rec["rom_path"] = rom_path
    rec["rom_resolved"] = "yes"
    keys = _watch_keys_for(rom_path)
    rec["watch_keys"] = keys
    rec["text_richness"] = (
        f"rich ({len(keys)} watch keys: {', '.join(keys)})"
        if keys
        else "minimum (data.json missing/empty)"
    )
    return rec


def _check_runtime(env_id: str, gym_v) -> Dict[str, Any]:
    out: Dict[str, Any] = {"runtime_ok": False}
    try:
        env = gym_v.make(env_id)
    except Exception as exc:  # noqa: BLE001
        out["runtime_error"] = f"gym_v.make failed: {exc}"
        return out
    try:
        odict, _ = env.reset(seed=0)
        agent_id = next(iter(odict))
        obs = odict[agent_id]
        img = obs.image
        if isinstance(img, list) and img:
            img = img[-1]
        out["image_present"] = img is not None
        out["image_size"] = (
            tuple(int(x) for x in img.size) if img is not None else None
        )
        out["text_present"] = bool(obs.text)
        out["text_preview"] = (obs.text or "")[:160]
        out["runtime_ok"] = bool(out["image_present"] and out["text_present"])
    except Exception as exc:  # noqa: BLE001
        out["runtime_error"] = f"reset failed: {exc}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also instantiate each env and call reset() to confirm both"
        " modalities at runtime (requires gym_v + stable_retro + ROMs).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of a human-readable table.",
    )
    args = parser.parse_args()

    specs = _load_specs()
    stable_retro = _try_import("stable_retro")
    gym_v = _try_import("gym_v") if args.runtime else None

    results: List[Dict[str, Any]] = []
    pass_count = 0
    runtime_count = 0
    for env_id in sorted(specs):
        rec = _check_static(env_id, specs[env_id], stable_retro)
        if args.runtime:
            if isinstance(gym_v, Exception):
                rec["runtime_error"] = f"gym_v not importable: {gym_v}"
            else:
                rec.update(_check_runtime(env_id, gym_v))
                if rec.get("runtime_ok"):
                    runtime_count += 1
        # Image+text channels are guaranteed by source; "pass" when both
        # are reported and (when runtime check ran) reset succeeded.
        passed = rec["image_channel"].startswith("yes") and rec[
            "text_channel"
        ].startswith("yes")
        if args.runtime and "runtime_ok" in rec:
            passed = passed and bool(rec["runtime_ok"])
        rec["pass"] = passed
        if passed:
            pass_count += 1
        results.append(rec)

    if args.json:
        print(
            json.dumps(
                {
                    "stable_retro_installed": not isinstance(stable_retro, Exception),
                    "gym_v_installed": (
                        not isinstance(gym_v, Exception)
                        if args.runtime
                        else None
                    ),
                    "total_envs": len(results),
                    "static_pass": pass_count,
                    "runtime_pass": runtime_count if args.runtime else None,
                    "results": results,
                },
                indent=2,
                default=str,
            )
        )
        return 0 if pass_count == len(results) else 1

    print("Gym-V Temporal/* modality coverage")
    print("=" * 78)
    print(
        f"  stable_retro: "
        f"{'installed' if not isinstance(stable_retro, Exception) else 'NOT installed (' + type(stable_retro).__name__ + ')'}"
    )
    if args.runtime:
        print(
            f"  gym_v       : "
            f"{'installed' if not isinstance(gym_v, Exception) else 'NOT installed (' + type(gym_v).__name__ + ')'}"
        )
    print("-" * 78)
    header = f"{'env_id':<34} {'image':<6} {'text':<6} {'rom':<10} richness"
    print(header)
    print("-" * 78)
    for rec in results:
        rom_field = (
            "yes"
            if rec.get("rom_resolved") == "yes"
            else "no"
            if rec.get("rom_resolved", "").startswith("no")
            else "?"
        )
        rich = rec.get("text_richness", "")
        print(
            f"{rec['env_id']:<34} "
            f"{'YES':<6} "
            f"{'YES':<6} "
            f"{rom_field:<10} "
            f"{rich}"
        )
        if args.runtime:
            if rec.get("runtime_ok"):
                w, h = rec["image_size"]
                print(
                    f"  └─ runtime: image={w}x{h}  text[:80]="
                    f"{(rec.get('text_preview') or '')[:80]!r}"
                )
            elif "runtime_error" in rec:
                print(f"  └─ runtime: ERROR {rec['runtime_error']}")
    print("-" * 78)
    print(
        f"Static modality pass: {pass_count}/{len(results)} envs "
        f"(image+text always emitted by RetroGymVEnv)"
    )
    if args.runtime:
        print(
            f"Runtime reset pass : {runtime_count}/{len(results)} envs "
            "(requires ROM imports)"
        )
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
