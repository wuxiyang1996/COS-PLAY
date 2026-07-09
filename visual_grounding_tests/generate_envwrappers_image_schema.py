#!/usr/bin/env python
"""
env_wrappers **image-only** → ``<state>`` schema rollouts for
2048 / Candy Crush / Tetris / Super Mario.

Steps each game via its env-wrapper of record:

  - 2048 / Candy Crush / Tetris : ``make_gaming_env(..., observation_mode="both")``
                                  composed with :class:`GamingAgentNLWrapper`.
  - Super Mario                 : ``make_orak_env("super_mario")``.

At every step the wrapper's *rendered frame* is asked of a vision LLM
(GPT-5.5 by default) which returns a canonical ``<state>…</state>`` block
following the cross-domain ``vlm_wrapper.schema`` spec. The wrapper's
natural-language text is shipped only as **auxiliary context** — the
image is the PRIMARY input. The result is directly comparable against
the text-head schema produced by ``generate_envwrappers_text_schema.py``.

For each step we save:
  1. ``images/step_NNN.png``    — the visual state of the env (PRIMARY)
  2. ``obs_text``               — wrapper text (auxiliary, for reference)
  3. ``schema_image_llm``       — schema produced by the image (vision) head
  4. ``schema_text_heuristic``  — deterministic ``gymv_wrapper.heuristic``
                                  grounding (no API; cheap baseline)
  5. ``schema_canonical``       — deterministic per-game canonical schema
                                  (modality-invariant ground truth)

Output layout (under ``visual_grounding_tests/output/envwrappers_image/``)::

    <game>/<run_id>_ep<NNN>/
      images/step_NNN.png   — one PNG per step (the actual VLM input)
      steps.jsonl           — one JSON record per step
      run_summary.json      — timing/counts/model id

Usage examples (from ``Multi-hop-Reasoning-VLM-Agent`` or repo root with
that on ``PYTHONPATH``)::

    # All four games, default few-step, GPT-5.5
    export OPENAI_API_KEY=...
    python visual_grounding_tests/generate_envwrappers_image_schema.py

    # Only 2048 and tetris, dry-run (no API; canonical + heuristic + frames)
    python visual_grounding_tests/generate_envwrappers_image_schema.py \\
        --games twenty_forty_eight tetris --dry_run --max_steps 3

    # Override the model
    python visual_grounding_tests/generate_envwrappers_image_schema.py \\
        --model gpt-5.5 --max_steps 4 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))

logger = logging.getLogger("envwrappers_image_schema")


# ── Models / defaults ────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("VLM_ENVWRAP_IMAGE_MODEL", "gpt-5.5")
DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_ENVWRAP_IMAGE_MAX_TOKENS", "1200"))
DEFAULT_TEMPERATURE = float(os.environ.get("VLM_ENVWRAP_IMAGE_TEMPERATURE", "0.2"))
DEFAULT_OUTPUT_TAG = "envwrappers_image"

_REASONING_TOKEN_MULTIPLIER = int(os.environ.get("VLM_ENVWRAP_REASONING_MULT", "5"))
_REASONING_TOKEN_FLOOR = int(os.environ.get("VLM_ENVWRAP_REASONING_FLOOR", "6000"))


# Games we support and the action we will repeat per step (kept dumb on
# purpose: this is a grounding/labeling test, not a benchmark).
GAME_DEFAULT_ACTION: Dict[str, str] = {
    "twenty_forty_eight": "up",
    "candy_crush": "",          # filled in dynamically — first valid action
    "tetris": "hard_drop",
    "super_mario": "Jump Level: 0",
}

GAME_DESCRIPTIONS: Dict[str, str] = {
    "twenty_forty_eight": (
        "Play 2048 on a 4x4 grid. Slide tiles up/down/left/right to merge "
        "matching numbers; goal is to create a 2048 tile. Larger tiles "
        "score higher."
    ),
    "candy_crush": (
        "Match-3 puzzle on a colored grid. Swap two adjacent candies to "
        "form lines of 3+ same colors, which clear them and earn points. "
        "Limited number of moves per episode."
    ),
    "tetris": (
        "Classic Tetris. Move and rotate falling tetrominoes to fill rows; "
        "complete rows clear and earn points. Game ends when the stack "
        "reaches the top."
    ),
    "super_mario": (
        "Super Mario Bros (NES). Move Mario to the right, jump over pits "
        "and enemies. Score = horizontal distance progressed."
    ),
}


# ── Lazy / optional imports ─────────────────────────────────────────────

def _import_schema_helpers():
    """Import the cross-domain schema helpers (image path needs ``build_user_message``)."""
    from vlm_wrapper.schema import (
        build_system_prompt,
        build_user_message,
        encode_image_b64,
        parse_schema_output,
        semantic_validate,
        validate_schema,
    )
    return {
        "build_system_prompt": build_system_prompt,
        "build_user_message": build_user_message,
        "encode_image_b64": encode_image_b64,
        "parse_schema_output": parse_schema_output,
        "semantic_validate": semantic_validate,
        "validate_schema": validate_schema,
    }


def _import_api_func():
    """Import OpenAI client/model helpers from API_func.py (project-wide)."""
    try:
        from API_func import effective_openai_model, make_openai_client
        return make_openai_client, effective_openai_model
    except ImportError:
        return None, None


def _import_heuristic_text_to_schema():
    try:
        from gymv_wrapper.heuristic import text_to_schema
        return text_to_schema
    except Exception:
        return None


def _import_canonical_schema():
    """Deterministic per-game canonical schema generator.

    Returned as ``(make_canonical_schema, canonical_label_hint,
    max_entities_by_game)``. Any unavailable entry degrades to ``None`` so
    callers can fall back to the legacy free-form path.
    """
    try:
        from visual_grounding_tests.canonical_schema import (
            MAX_ENTITIES_BY_GAME,
            canonical_label_hint,
            make_canonical_schema,
        )
        return make_canonical_schema, canonical_label_hint, MAX_ENTITIES_BY_GAME
    except Exception:
        return None, None, {}


# ── PIL / numpy helpers ─────────────────────────────────────────────────

def _to_pil(image: Any):
    """Best-effort conversion of any frame-like object to a PIL.Image (RGB)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        arr = np.asarray(image)
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


def _save_frame(image: Any, path: Path) -> Optional[str]:
    pil = _to_pil(image)
    if pil is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pil.save(str(path), format="PNG")
        return str(path)
    except Exception:
        return None


# ── env_wrappers integration ────────────────────────────────────────────

def _build_env(game: str, max_steps: int) -> Tuple[Any, Dict[str, Any]]:
    """Create the wrapped env for a game and return (wrapped_env, meta)."""
    if game in {"twenty_forty_eight", "candy_crush", "tetris"}:
        from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
        from env_wrappers.gym_like import make_gaming_env

        gym = make_gaming_env(
            game,
            max_steps=max_steps,
            observation_mode="both",
            render_mode="rgb_array" if game == "tetris" else None,
            load_image_array=True,
        )
        env = GamingAgentNLWrapper(gym, include_action_hint=False, game_name=game)
        return env, {
            "wrapper": "GamingAgentNLWrapper",
            "underlying": "make_gaming_env",
            "task": GAME_DESCRIPTIONS.get(game, ""),
        }

    if game == "super_mario":
        from env_wrappers.orak_nl_wrapper import make_orak_env
        env = make_orak_env(
            "super_mario",
            max_steps=max_steps,
            input_modality="text_image",
            save_frames=True,
        )
        return env, {
            "wrapper": "OrakNLWrapper",
            "underlying": "make_orak_env",
            "task": GAME_DESCRIPTIONS.get(game, ""),
        }

    raise ValueError(f"Unsupported game: {game}")


def _pick_action(game: str, info: Dict[str, Any], step: int) -> str:
    """Pick the per-step action.  Defaults to a constant from
    ``GAME_DEFAULT_ACTION``; for ``candy_crush`` we round-robin over the
    valid actions exposed in ``info['action_names']``."""
    if game == "candy_crush":
        names: List[str] = list(info.get("action_names") or [])
        if not names:
            return "swap (0,0) right"
        return names[step % len(names)]
    return GAME_DEFAULT_ACTION.get(game, "")


def _extract_visual(info: Dict[str, Any], obs_text: str) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Pull the current visual frame out of the wrapper's info dict.

    Returns ``(pil_image, np_array)``. Falls back to ``env_wrappers``'s
    cross-env helpers, then to manual digging into ``info['raw_obs']``.
    The image head needs an actual frame at every step — if both fail
    we record a no-image error in the per-step result.
    """
    pil = None
    arr = None
    try:
        from env_wrappers.visual_utils import get_obs_image, get_obs_pil_image
        arr = get_obs_image(info)
        pil = get_obs_pil_image(info)
    except Exception:
        pass

    if arr is None and isinstance(info.get("raw_obs"), dict):
        raw = info["raw_obs"]
        for key in ("frame", "image", "screenshot"):
            cand = raw.get(key)
            if cand is not None:
                try:
                    arr = np.asarray(cand)
                    break
                except Exception:
                    arr = None

    if pil is None and arr is not None:
        pil = _to_pil(arr)

    return pil, arr


# ── OpenAI client + cross-model retry (gpt-5.x reasoning vs classic) ────

def _build_client_and_model(
    *, model: str, api_key: Optional[str], base_url: Optional[str],
):
    """Return ``(client, routed_model)`` or ``(None, model)`` on failure."""
    make_openai_client, effective_openai_model = _import_api_func()

    try:
        if make_openai_client is not None:
            client = make_openai_client(api_key=api_key, base_url=base_url)
        else:
            client = None
    except Exception:
        client = None

    if client is None:
        try:
            import openai
            kw: Dict[str, Any] = {}
            if api_key:
                kw["api_key"] = api_key
            if base_url:
                kw["base_url"] = base_url
            client = openai.OpenAI(**kw) if (api_key or base_url) else openai.OpenAI()
        except Exception:
            return None, model

    if api_key or base_url:
        routed = model
    elif effective_openai_model is not None:
        try:
            routed = effective_openai_model(model)
        except Exception:
            routed = model
    else:
        routed = model
    return client, routed


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> str:
    """Call ``client.chat.completions.create`` with cross-model token args.

    Reasoning models (``gpt-5.x``, ``o*``) require ``max_completion_tokens``
    and reject custom temperature; classic models accept the legacy names.
    We try the legacy names first and transparently retry once on the
    "Unsupported parameter" error path. Reasoning models also burn many
    tokens on internal thinking, so on retry we scale the budget by
    ``VLM_ENVWRAP_REASONING_MULT`` (default 5×) and floor at
    ``VLM_ENVWRAP_REASONING_FLOOR`` (default 6000).
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        needs_retry = (
            "max_completion_tokens" in msg
            or ("max_tokens" in msg and "Unsupported" in msg)
            or ("temperature" in msg and "Unsupported" in msg)
        )
        if not needs_retry:
            raise
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        kwargs["max_completion_tokens"] = max(
            _REASONING_TOKEN_FLOOR,
            max_tokens * _REASONING_TOKEN_MULTIPLIER,
        )
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


# ── Schema validation wrapper ──────────────────────────────────────────

def _validate_schema_text(
    schema_text: Optional[str],
    *,
    image_size: Optional[Tuple[int, int]],
    helpers: Dict[str, Any],
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    if not schema_text:
        return [], None
    warnings = list(helpers["validate_schema"](schema_text))
    vres = helpers["semantic_validate"](
        schema_text, domain="gymv", image_size=image_size,
    )
    validation = vres.as_dict()
    warnings = warnings + list(vres.warnings) + list(vres.errors)
    return warnings, validation


# ── Image head: vision VLM (image + optional text) → schema ────────────

def generate_image_schema_llm(
    *,
    image: Any,
    obs_text: str,
    game: str,
    task_id: str,
    goal: str,
    step: int,
    valid_actions: Optional[List[str]],
    helpers: Dict[str, Any],
    client: Any,
    routed_model: str,
    fallback_model: str,
    temperature: float,
    max_tokens: int,
    retries: int = 1,
    canonical_hint: Optional[str] = None,
    max_entities: int = 20,
) -> Dict[str, Any]:
    """Call the VLM with **image (+ optional text grounding)** to produce
    a ``<state>`` schema.

    The image is the primary input; ``obs_text`` is shipped as auxiliary
    context only. Reuses the same cross-domain ``vlm_wrapper.schema``
    system prompt as the text head so the two outputs are comparable.

    ``canonical_hint`` and ``max_entities`` have the same role as in the
    text head: when provided they pin the label vocabulary / position
    units / per-game entity cap.
    """
    pil = _to_pil(image)
    if pil is None:
        return {
            "schema": None,
            "raw": "Error: no image available",
            "warnings": ["no_image_for_visual_call"],
            "validation": None,
            "model": fallback_model,
            "model_routed": routed_model,
            "head": "image",
        }

    system = helpers["build_system_prompt"]("gymv", max_entities=max_entities)
    if canonical_hint:
        system = system + "\n\n" + canonical_hint

    extra_parts: List[str] = []
    extra_parts.append(f"Game rules:\n{GAME_DESCRIPTIONS.get(game, '') or task_id}")
    if obs_text:
        extra_parts.append(
            f"Environment text state (auxiliary — for reference only):\n{obs_text}"
        )
    if valid_actions:
        extra_parts.append(
            "Valid actions for this environment (you MUST copy these strings "
            "verbatim into <actions>; do NOT rename or reformat):\n"
            + "\n".join(f"  - {a}" for a in valid_actions)
        )
    extra_context = "\n\n".join(extra_parts)

    user_content = helpers["build_user_message"](
        pil,
        domain="gymv",
        task_id=task_id,
        goal=goal,
        step=step,
        extra_context=extra_context,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    schema = None
    for attempt in range(1, retries + 2):
        try:
            raw = _chat_completion(
                client,
                model=routed_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            schema = helpers["parse_schema_output"](raw)
            if schema:
                break
            logger.warning("[image-VLM] attempt %d: no <state> block", attempt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[image-VLM] attempt %d failed: %s", attempt, exc)
            raw = f"Error: {exc!r}"

    warnings, validation = _validate_schema_text(
        schema, image_size=pil.size, helpers=helpers,
    )

    return {
        "schema": schema,
        "raw": raw,
        "warnings": warnings,
        "validation": validation,
        "model": fallback_model,
        "model_routed": routed_model,
        "image_size": list(pil.size),
        "head": "image",
    }


# ── Episode runner ──────────────────────────────────────────────────────

def run_one_episode(
    game: str,
    *,
    max_steps: int,
    out_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    api_key: Optional[str],
    base_url: Optional[str],
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    helpers = _import_schema_helpers()
    text_to_schema = _import_heuristic_text_to_schema()
    (
        make_canonical_schema,
        canonical_label_hint,
        max_entities_by_game,
    ) = _import_canonical_schema()

    if dry_run:
        client, routed_model = None, model
    else:
        client, routed_model = _build_client_and_model(
            model=model, api_key=api_key, base_url=base_url,
        )
        if client is None:
            print(
                "[WARNING] No OpenAI/OpenRouter client could be built — "
                "set OPENAI_API_KEY / OPENROUTER_API_KEY, pass --api_key, "
                "or use --dry_run. Continuing with heuristic only."
            )

    env, env_meta = _build_env(game, max_steps=max_steps)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    nl, info = env.reset()
    records: List[Dict[str, Any]] = []
    total_reward = 0.0
    image_ok = image_err = 0

    t0 = time.time()
    for step in range(max_steps):
        obs_text = info.get("raw_obs", {}).get("text") if isinstance(info.get("raw_obs"), dict) else None
        if not obs_text:
            obs_text = nl

        pil, _arr = _extract_visual(info, obs_text or "")
        img_rel = None
        img_size: Optional[Tuple[int, int]] = None
        if pil is not None:
            img_path = images_dir / f"step_{step:03d}.png"
            saved = _save_frame(pil, img_path)
            if saved:
                img_rel = str(img_path.relative_to(out_dir))
            try:
                img_size = (int(pil.size[0]), int(pil.size[1]))
            except Exception:
                img_size = None

        valid_actions: Optional[List[str]] = None
        an = info.get("action_names") or env_meta.get("action_names")
        if an:
            valid_actions = [str(a) for a in an][:25]

        task_id = f"{env_meta.get('underlying', 'env')}/{game}"
        goal = (env_meta.get("task") or GAME_DESCRIPTIONS.get(game, "")).split("\n")[0]

        heuristic_schema = None
        if text_to_schema is not None:
            try:
                heuristic_schema = text_to_schema(
                    obs_text=obs_text or "",
                    description=env_meta.get("task", ""),
                    task_id=task_id,
                    step=step,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("heuristic text_to_schema failed: %s", exc)

        canonical_schema: Optional[str] = None
        canonical_hint: Optional[str] = None
        if make_canonical_schema is not None:
            try:
                canonical_schema = make_canonical_schema(
                    game=game,
                    info=info,
                    task_id=task_id,
                    goal=goal,
                    step=step,
                    actions=valid_actions,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[canonical] %s step %d failed: %s", game, step, exc)
            if canonical_label_hint is not None:
                try:
                    canonical_hint = canonical_label_hint(game)
                except Exception:
                    canonical_hint = None

        per_game_max_entities = (max_entities_by_game or {}).get(game, 20)

        image_result: Optional[Dict[str, Any]] = None
        if not dry_run and client is not None:
            try:
                image_result = generate_image_schema_llm(
                    image=pil,
                    obs_text=obs_text or "",
                    game=game,
                    task_id=task_id,
                    goal=goal,
                    step=step,
                    valid_actions=valid_actions,
                    helpers=helpers,
                    client=client,
                    routed_model=routed_model,
                    fallback_model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    canonical_hint=canonical_hint,
                    max_entities=per_game_max_entities,
                )
                if image_result and image_result.get("schema"):
                    image_ok += 1
                else:
                    image_err += 1
            except Exception as exc:  # noqa: BLE001
                image_err += 1
                image_result = {
                    "schema": None,
                    "raw": f"Error: {exc!r}",
                    "warnings": [str(exc)],
                    "validation": None,
                    "model": model,
                    "head": "image",
                }
                if verbose:
                    traceback.print_exc()

        record: Dict[str, Any] = {
            "step": step,
            "game": game,
            "wrapper": env_meta.get("wrapper"),
            "task_id": task_id,
            "image_path": img_rel,
            "image_size": list(img_size) if img_size else None,
            "obs_text": obs_text,
            "valid_actions": valid_actions,
            "schema_canonical": canonical_schema,
            "schema_text_heuristic": heuristic_schema,
            "schema_image_llm": image_result,
            "model_scheduled": routed_model,
            "dry_run": dry_run,
        }
        records.append(record)

        if verbose:
            ok = bool(image_result and image_result.get("schema"))
            print(
                f"    step {step:>2}: image={'yes' if img_rel else 'no '} "
                f"vision_ok={'yes' if ok else 'no '}"
            )

        if step >= max_steps - 1:
            break

        action = _pick_action(game, info, step)
        try:
            nl, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward or 0.0)
            if bool(terminated) or bool(truncated):
                break
        except Exception as exc:  # noqa: BLE001
            print(f"  [step error] {game} step={step} action={action!r}: {exc}")
            if verbose:
                traceback.print_exc()
            break

    try:
        env.close()
    except Exception:
        pass

    elapsed = time.time() - t0
    stats: Dict[str, Any] = {
        "game": game,
        "wrapper": env_meta.get("wrapper"),
        "task": env_meta.get("task"),
        "head": "image",
        "max_steps": max_steps,
        "steps_recorded": len(records),
        "total_reward": total_reward,
        "elapsed_seconds": round(elapsed, 3),
        "model": model,
        "model_routed": routed_model,
        "dry_run": dry_run,
        "image_schema_ok": image_ok,
        "image_schema_fail": image_err,
    }
    return records, stats


# ── CLI ────────────────────────────────────────────────────────────────

def _resolve_games(requested: List[str]) -> List[str]:
    canonical = list(GAME_DEFAULT_ACTION.keys())
    aliases: Dict[str, str] = {
        "2048": "twenty_forty_eight",
        "twentyfortyeight": "twenty_forty_eight",
        "twenty_forty_eight": "twenty_forty_eight",
        "candy": "candy_crush",
        "candy_crush": "candy_crush",
        "candycrush": "candy_crush",
        "tetris": "tetris",
        "mario": "super_mario",
        "super_mario": "super_mario",
        "supermario": "super_mario",
    }
    out: List[str] = []
    for g in requested:
        key = g.strip().lower()
        if key in aliases:
            out.append(aliases[key])
        elif g in canonical:
            out.append(g)
        else:
            print(f"[SKIP] Unknown game requested: {g}")
    seen = set()
    deduped: List[str] = []
    for g in out:
        if g not in seen:
            deduped.append(g)
            seen.add(g)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "env_wrappers: rendered frames → <state> schema (image-only VLM head) "
            "for 2048 / Candy Crush / Tetris / Super Mario."
        ),
    )
    parser.add_argument(
        "--games",
        type=str,
        nargs="+",
        default=list(GAME_DEFAULT_ACTION.keys()),
        help="Games to cover (aliases: 2048, mario, etc.).",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"VLM model id (default: {DEFAULT_MODEL}; env override: VLM_ENVWRAP_IMAGE_MODEL).",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip API calls; only run env, save frames, and emit canonical + heuristic schemas.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    games = _resolve_games(args.games)
    if not games:
        print("[FATAL] No valid games selected. Available: "
              + ", ".join(GAME_DEFAULT_ACTION.keys()))
        sys.exit(2)

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "output" / DEFAULT_OUTPUT_TAG
    )
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "head": "image",
        "model": args.model,
        "dry_run": args.dry_run,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "games": games,
    }
    all_summaries: List[Dict[str, Any]] = []

    for game in games:
        for ep in range(args.episodes):
            ep_dir = out_root / game / f"{run_id}_ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  -> {game}  episode {ep + 1}/{args.episodes}  -> {ep_dir}")
            try:
                records, stats = run_one_episode(
                    game,
                    max_steps=args.max_steps,
                    out_dir=ep_dir,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    dry_run=args.dry_run,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    verbose=args.verbose,
                )
                stats["episode"] = ep
                with open(ep_dir / "steps.jsonl", "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                with open(ep_dir / "run_summary.json", "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
                print(
                    f"     wrote {len(records)} steps  "
                    f"image_ok={stats.get('image_schema_ok', 0)}"
                )
                all_summaries.append({**stats, "path": str(ep_dir)})
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {game} ep{ep}: {exc}")
                traceback.print_exc()
                all_summaries.append(
                    {"game": game, "episode": ep, "error": str(exc)},
                )

    master["runs"] = all_summaries
    master_path = out_root / f"batch_{run_id}.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Batch summary: {master_path}")


if __name__ == "__main__":
    main()
