#!/usr/bin/env python
"""
BrowserGym **heuristic + tool-loop** → ``<state>`` schema rollouts.

Captures a stream of BrowserGym observations (real ``browsergym/openended``
or a synthetic shopping page from :mod:`browsergym_wrapper.example`), and
at every step produces two cross-comparable ``<state>...</state>`` schemas:

  1. ``schema_heuristic`` — :func:`browsergym_wrapper.heuristic.obs_to_schema`
     deterministically walks the AXTree + ``extra_element_properties``.
     Fast, free, no LLM.  This is the **AXTree-grounded baseline** used by
     ``cascaded_ground`` to validate the vision head.

  2. ``schema_tool_loop`` — :func:`vlm_wrapper.tool_loop.run_tool_loop`
     hands the **visual state** (the page screenshot) to a VLM that may
     call any of the browser tools registered by
     :func:`browsergym_wrapper.tools.build_browser_registry`
     (``query_element_bbox``, ``search_elements``, ``get_som_elements``,
     ``check_relation``, …).  The VLM identifies entities visually, then
     queries the live AXTree for ground-truth bboxes / states / relations
     before emitting the canonical schema.  The full tool trace is
     persisted for SFT data.

For each step we record:
  - ``images/step_NNN.png``      — the visual state (PRIMARY tool-loop input)
  - ``url``, ``goal``, ``focused_element_bid``, ``last_action(_error)``
  - ``schema_heuristic``         — AXTree → schema (always emitted)
  - ``schema_tool_loop``         — screenshot + tools → VLM → schema
  - ``schema_grounding``         — OmniParser-v2 → schema (optional)
  - ``schema_vision_only``       — single-shot screenshot → VLM (optional)

Output layout (under ``visual_grounding_tests/output/browsergym/``)::

    <url_safe>/<run_id>_ep<NNN>/
      images/step_NNN.png
      steps.jsonl                — one JSON record per step
      run_summary.json           — timing/counts/model id

Usage examples (run from ``Multi-hop-Reasoning-VLM-Agent`` or repo root with
that on ``PYTHONPATH``)::

    # Synthetic shopping page (no browsergym install, no API needed)
    python visual_grounding_tests/generate_browsergym_schema.py \\
        --synthetic --dry_run --max_steps 1

    # Synthetic + tool-loop head (uses api_keys.open_router_api_key by default)
    python visual_grounding_tests/generate_browsergym_schema.py \\
        --synthetic --max_steps 1

    # Real browsergym pages (Google + Wikipedia), heuristic + tool-loop
    python visual_grounding_tests/generate_browsergym_schema.py \\
        --urls https://www.google.com https://en.wikipedia.org/wiki/Reinforcement_learning \\
        --max_steps 2 -v

    # Add the OmniParser grounding head and the single-shot vision head
    python visual_grounding_tests/generate_browsergym_schema.py \\
        --synthetic --grounding --vision_only --max_steps 1

The OpenRouter / OpenAI key is sourced (in order) from ``--api_key``,
``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``, or ``api_keys.open_router_api_key``.
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
REPO_ROOT = CODEBASE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("browsergym_schema")


# ── Models / defaults ────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("VLM_BROWSERGYM_MODEL", "openai/gpt-4.1")
DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_BROWSERGYM_MAX_TOKENS", "1500"))
DEFAULT_TEMPERATURE = float(os.environ.get("VLM_BROWSERGYM_TEMPERATURE", "0.2"))
DEFAULT_MAX_ROUNDS = int(os.environ.get("VLM_BROWSERGYM_MAX_ROUNDS", "5"))
DEFAULT_OUTPUT_TAG = "browsergym"

_REASONING_TOKEN_MULTIPLIER = int(os.environ.get("VLM_BROWSERGYM_REASONING_MULT", "5"))
_REASONING_TOKEN_FLOOR = int(os.environ.get("VLM_BROWSERGYM_REASONING_FLOOR", "6000"))


# ── Lazy / optional imports ─────────────────────────────────────────────

def _import_heuristic():
    from browsergym_wrapper.heuristic import obs_to_schema
    return obs_to_schema


def _import_tools_and_loop():
    from browsergym_wrapper.tools import build_browser_registry
    from vlm_wrapper.tool_loop import run_tool_loop
    return build_browser_registry, run_tool_loop


def _import_vision_only_adapter():
    try:
        from browsergym_wrapper.adapter import browser_obs_to_schema
        return browser_obs_to_schema
    except Exception:
        return None


def _import_grounding():
    try:
        from browsergym_wrapper.grounding import grounding_obs_to_schema
        return grounding_obs_to_schema
    except Exception:
        return None


def _import_schema_helpers():
    from vlm_wrapper.schema import (
        count_entities,
        parse_schema_output,
        semantic_validate,
        validate_schema,
    )
    return {
        "count_entities": count_entities,
        "parse_schema_output": parse_schema_output,
        "semantic_validate": semantic_validate,
        "validate_schema": validate_schema,
    }


def _import_synthetic_obs():
    from browsergym_wrapper.example import build_browsergym_obs
    return build_browsergym_obs


def _resolve_api_key(explicit: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(api_key, base_url)`` resolved from CLI / env / api_keys.py.

    Precedence:
      1. Explicit ``--api_key``.
      2. ``OPENROUTER_API_KEY`` (env).
      3. ``OPENAI_API_KEY`` (env).
      4. ``api_keys.open_router_api_key`` / ``api_keys.openai_api_key``
         (next to repo root).

    When the key looks like an OpenRouter key (or env override is set),
    ``base_url`` is set to ``https://openrouter.ai/api/v1``.
    """
    or_url = "https://openrouter.ai/api/v1"

    if explicit:
        base = or_url if explicit.startswith("sk-or-") else None
        return explicit, base

    or_env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if or_env:
        return or_env, or_url

    oa_env = os.environ.get("OPENAI_API_KEY", "").strip()
    if oa_env:
        return oa_env, None

    try:
        import api_keys  # type: ignore
    except Exception:
        api_keys = None  # type: ignore[assignment]

    if api_keys is not None:
        for attr, base in (
            ("open_router_api_key", or_url),
            ("openrouter_api_key", or_url),
            ("openai_api_key", None),
        ):
            v = getattr(api_keys, attr, "")
            if isinstance(v, str) and v.strip():
                return v.strip(), base

    return None, None


# ── PIL / numpy helpers ─────────────────────────────────────────────────

def _to_pil(image: Any):
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


# ── BrowserGym observation capture ─────────────────────────────────────

def _url_safe(url: str) -> str:
    """Best-effort filesystem-safe slug for a URL."""
    s = url.lower()
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in "-._/":
            out.append("_")
    slug = "".join(out).strip("_")[:120] or "page"
    return slug


def _capture_synthetic_episode(max_steps: int) -> List[Dict[str, Any]]:
    """Build a list of observations from the synthetic shopping page.

    The same fake obs is replayed ``max_steps`` times with a step counter
    bumped — perfect for a local smoke-test that exercises both heads
    without booting a browser.
    """
    build_synth = _import_synthetic_obs()
    base_obs = build_synth()

    out = []
    for s in range(max_steps):
        clone = dict(base_obs)
        clone["screenshot"] = base_obs["screenshot"]
        clone["last_action"] = base_obs.get("last_action", "") if s == 0 else "noop()"
        clone["last_action_error"] = ""
        out.append(clone)
    return out


def _capture_browsergym_episode(
    url: str,
    max_steps: int,
    *,
    headless: bool,
) -> List[Dict[str, Any]]:
    """Reset ``browsergym/openended`` on ``url`` and step ``noop()`` ``max_steps`` times.

    Returns a list of observation dicts (same shape as
    ``BrowserEnv._get_obs()``).
    """
    import gymnasium as gym
    import browsergym.core  # noqa: F401  -- registers the env id

    logger.info("Booting browsergym/openended: %s", url)
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": url},
        headless=headless,
    )

    obs, _info = env.reset()
    captured: List[Dict[str, Any]] = [dict(obs)]

    for i in range(max(0, max_steps - 1)):
        try:
            obs, _r, term, trunc, _info = env.step("noop()")
        except Exception as exc:  # noqa: BLE001
            logger.warning("step %d failed (%s); stopping", i, exc)
            break
        captured.append(dict(obs))
        if term or trunc:
            break

    try:
        env.close()
    except Exception:
        pass
    return captured


# ── AXTree flattening (for the optional vision-only head) ──────────────

def _flatten_axtree(obs: Dict[str, Any], max_chars: int = 3000) -> str:
    """Flatten ``obs['axtree_object']`` into a compact text dump.

    Uses ``browsergym.utils.obs.flatten_axtree_to_str`` when available;
    otherwise emits a minimal fallback so the vision-only head still has
    *some* AXTree grounding.  Truncated to ``max_chars``.
    """
    axtree = obs.get("axtree_object")
    if axtree is None:
        return ""

    try:
        from browsergym.utils.obs import flatten_axtree_to_str  # type: ignore
        text = flatten_axtree_to_str(axtree, extra_properties=obs.get("extra_element_properties", {}))
    except Exception:
        # Fallback: minimal flattening without the helper.
        lines: List[str] = []
        for node in axtree.get("nodes", [])[:80]:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            bid = node.get("browsergym_id", "")
            if not (role and (name or bid)):
                continue
            lines.append(f"[{bid}] {role}: {name[:60]}".rstrip())
        text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


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
    try:
        vres = helpers["semantic_validate"](
            schema_text, domain="browser", image_size=image_size,
        )
    except Exception as exc:  # noqa: BLE001
        return warnings + [f"semantic_validate_failed: {exc}"], None
    validation = vres.as_dict()
    warnings = warnings + list(vres.warnings) + list(vres.errors)
    return warnings, validation


# ── Episode runner ─────────────────────────────────────────────────────

def run_one_episode(
    url_or_tag: str,
    *,
    observations: List[Dict[str, Any]],
    out_dir: Path,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    max_rounds: int,
    max_entities: int,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    do_grounding: bool,
    do_vision_only: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obs_to_schema = _import_heuristic()
    helpers = _import_schema_helpers()

    build_browser_registry = run_tool_loop = None
    if not dry_run and api_key:
        try:
            build_browser_registry, run_tool_loop = _import_tools_and_loop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool loop unavailable (%s); skipping tool head", exc)
            build_browser_registry = run_tool_loop = None

    grounding_obs_to_schema = _import_grounding() if do_grounding else None
    browser_obs_to_schema = _import_vision_only_adapter() if do_vision_only else None

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "heuristic_ok": 0, "heuristic_fail": 0,
        "tool_ok": 0, "tool_fail": 0,
        "grounding_ok": 0, "grounding_fail": 0,
        "vision_only_ok": 0, "vision_only_fail": 0,
    }

    t_episode = time.time()

    for step, obs in enumerate(observations):
        # 1. Save the visual state
        img_rel: Optional[str] = None
        img_size: Optional[Tuple[int, int]] = None
        pil = _to_pil(obs.get("screenshot"))
        if pil is not None:
            img_path = images_dir / f"step_{step:03d}.png"
            saved = _save_frame(pil, img_path)
            if saved:
                img_rel = str(img_path.relative_to(out_dir))
            try:
                img_size = (int(pil.size[0]), int(pil.size[1]))
            except Exception:
                img_size = None

        # Goal extraction (mirrors heuristic._extract_goal)
        goal = obs.get("goal", "") or ""
        if not goal:
            goal_obj = obs.get("goal_object", ()) or ()
            goal = " ".join(
                m.get("text", "") for m in goal_obj if m.get("type") == "text"
            )

        task_id = f"browsergym/{_url_safe(obs.get('url', url_or_tag))}"

        # 2. Heuristic head (always)
        heuristic_t0 = time.time()
        heuristic_schema: Optional[str] = None
        heuristic_warnings: List[str] = []
        try:
            heuristic_schema = obs_to_schema(
                obs, step=step, task_id=task_id, max_entities=max_entities,
            )
            heuristic_warnings, _ = _validate_schema_text(
                heuristic_schema, image_size=img_size, helpers=helpers,
            )
            counts["heuristic_ok"] += 1
        except Exception as exc:  # noqa: BLE001
            counts["heuristic_fail"] += 1
            heuristic_warnings = [f"heuristic_failed: {exc}"]
            if verbose:
                traceback.print_exc()
        heuristic_elapsed = round(time.time() - heuristic_t0, 3)

        # 3. Tool-loop head (visual state + browser tools → VLM → schema)
        tool_result: Optional[Dict[str, Any]] = None
        if (
            not dry_run
            and pil is not None
            and api_key
            and build_browser_registry is not None
            and run_tool_loop is not None
        ):
            tool_t0 = time.time()
            try:
                registry = build_browser_registry(obs)
                tool_loop_result = run_tool_loop(
                    pil,
                    domain="browser",
                    registry=registry,
                    goal=goal,
                    task_id=task_id,
                    step=step,
                    extra_context=(
                        f"URL: {obs.get('url', '')}\n"
                        "You have AXTree-backed grounding tools (query_element_bbox, "
                        "search_elements, get_som_elements, check_relation, "
                        "get_element_tree, list_entities, get_page_info, "
                        "get_state_flags, list_valid_actions). Identify entities in "
                        "the screenshot first, then call the tools to fetch ground-"
                        "truth bids, bounding boxes, and states before emitting the "
                        "final <state>...</state> schema."
                    ),
                    max_entities=max_entities,
                    max_rounds=max_rounds,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                tool_warnings, tool_validation = _validate_schema_text(
                    tool_loop_result.get("schema"),
                    image_size=img_size,
                    helpers=helpers,
                )
                tool_result = {
                    "schema": tool_loop_result.get("schema"),
                    "raw": tool_loop_result.get("raw", ""),
                    "warnings": tool_warnings,
                    "validation": tool_validation,
                    "model": tool_loop_result.get("model", model),
                    "rounds": tool_loop_result.get("rounds"),
                    "tool_trace": tool_loop_result.get("tool_trace", []),
                    "elapsed_seconds": round(time.time() - tool_t0, 3),
                    "head": "tool_loop",
                }
                if tool_result["schema"]:
                    counts["tool_ok"] += 1
                else:
                    counts["tool_fail"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["tool_fail"] += 1
                tool_result = {
                    "schema": None,
                    "raw": f"Error: {exc!r}",
                    "warnings": [f"tool_loop_failed: {exc}"],
                    "validation": None,
                    "model": model,
                    "rounds": 0,
                    "tool_trace": [],
                    "elapsed_seconds": round(time.time() - tool_t0, 3),
                    "head": "tool_loop",
                }
                if verbose:
                    traceback.print_exc()

        # 4. Optional: OmniParser-v2 grounding head (no LLM)
        grounding_record: Optional[Dict[str, Any]] = None
        if grounding_obs_to_schema is not None and pil is not None:
            g_t0 = time.time()
            try:
                gres = grounding_obs_to_schema(
                    obs, step=step, task_id=task_id, max_entities=max_entities,
                )
                gr_warnings, gr_validation = _validate_schema_text(
                    gres.get("schema"), image_size=img_size, helpers=helpers,
                )
                grounding_record = {
                    "schema": gres.get("schema"),
                    "warnings": gr_warnings,
                    "validation": gr_validation,
                    "model": gres.get("model", "omniparser-v2"),
                    "n_elements": len(gres.get("elements") or []),
                    "elapsed_seconds": round(time.time() - g_t0, 3),
                    "head": "grounding",
                }
                if grounding_record["schema"]:
                    counts["grounding_ok"] += 1
                else:
                    counts["grounding_fail"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["grounding_fail"] += 1
                grounding_record = {
                    "schema": None,
                    "warnings": [f"grounding_failed: {exc}"],
                    "validation": None,
                    "model": "omniparser-v2",
                    "elapsed_seconds": round(time.time() - g_t0, 3),
                    "head": "grounding",
                }
                if verbose:
                    traceback.print_exc()

        # 5. Optional: vision-only single-shot head (screenshot + AXTree text → VLM)
        vision_only_record: Optional[Dict[str, Any]] = None
        if (
            not dry_run
            and browser_obs_to_schema is not None
            and pil is not None
            and api_key
        ):
            v_t0 = time.time()
            try:
                axtree_text = _flatten_axtree(obs)
                vres = browser_obs_to_schema(
                    obs,
                    step=step,
                    task_id=task_id,
                    axtree_text=axtree_text,
                    max_entities=max_entities,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )
                vo_warnings, vo_validation = _validate_schema_text(
                    vres.get("schema"), image_size=img_size, helpers=helpers,
                )
                vision_only_record = {
                    "schema": vres.get("schema"),
                    "raw": (vres.get("raw") or "")[:4000],
                    "warnings": vo_warnings,
                    "validation": vo_validation,
                    "model": vres.get("model", model),
                    "elapsed_seconds": round(time.time() - v_t0, 3),
                    "head": "vision_only",
                }
                if vision_only_record["schema"]:
                    counts["vision_only_ok"] += 1
                else:
                    counts["vision_only_fail"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["vision_only_fail"] += 1
                vision_only_record = {
                    "schema": None,
                    "warnings": [f"vision_only_failed: {exc}"],
                    "validation": None,
                    "model": model,
                    "elapsed_seconds": round(time.time() - v_t0, 3),
                    "head": "vision_only",
                }
                if verbose:
                    traceback.print_exc()

        record: Dict[str, Any] = {
            "step": step,
            "url": obs.get("url", url_or_tag),
            "task_id": task_id,
            "goal": goal,
            "focused_element_bid": obs.get("focused_element_bid", "") or None,
            "last_action": obs.get("last_action", "") or None,
            "last_action_error": obs.get("last_action_error", "") or None,
            "image_path": img_rel,
            "image_size": list(img_size) if img_size else None,
            "n_axtree_nodes": (
                len((obs.get("axtree_object") or {}).get("nodes", []))
                if obs.get("axtree_object") is not None else 0
            ),
            "schema_heuristic": {
                "schema": heuristic_schema,
                "warnings": heuristic_warnings,
                "elapsed_seconds": heuristic_elapsed,
                "head": "heuristic",
            },
            "schema_tool_loop": tool_result,
            "schema_grounding": grounding_record,
            "schema_vision_only": vision_only_record,
            "model_scheduled": model,
            "dry_run": dry_run,
        }
        records.append(record)

        if verbose:
            ents = helpers["count_entities"](heuristic_schema) if heuristic_schema else 0
            tool_n = (
                helpers["count_entities"](tool_result["schema"])
                if (tool_result and tool_result.get("schema")) else 0
            )
            print(
                f"    step {step:>2} url={obs.get('url', '')[:60]:<60s} "
                f"image={'yes' if img_rel else 'no '} "
                f"heur_ents={ents:>2}  "
                f"tool_ents={tool_n:>2}"
            )

    elapsed = round(time.time() - t_episode, 3)
    summary: Dict[str, Any] = {
        "url_or_tag": url_or_tag,
        "head_set": [
            "heuristic",
            "tool_loop" if (api_key and not dry_run) else None,
            "grounding" if do_grounding else None,
            "vision_only" if (do_vision_only and api_key and not dry_run) else None,
        ],
        "steps_recorded": len(records),
        "elapsed_seconds": elapsed,
        "model": model,
        "dry_run": dry_run,
        **counts,
    }
    summary["head_set"] = [h for h in summary["head_set"] if h]
    return records, summary


# ── CLI ────────────────────────────────────────────────────────────────

DEFAULT_URLS: List[str] = [
    "https://www.google.com",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "BrowserGym: heuristic + tool-loop -> <state> schema rollouts. "
            "Records the visual state (PNG) and both heuristic (AXTree) and "
            "tool-augmented vision schemas at every step."
        ),
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=None,
        help=(
            "BrowserGym start URLs to capture (one episode each). "
            f"Default: {DEFAULT_URLS}."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Skip browsergym; use the synthetic shopping page from "
            "browsergym_wrapper.example. Useful for offline smoke-tests."
        ),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--max_entities", type=int, default=20)
    parser.add_argument(
        "--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help="Max VLM tool-call rounds before forcing a final schema.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Default: {DEFAULT_MODEL} (env: VLM_BROWSERGYM_MODEL).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip API calls; only run env, save frames, and emit heuristic schemas.",
    )
    parser.add_argument(
        "--grounding",
        action="store_true",
        help="Also run the OmniParser-v2 grounding head (heavy deps required).",
    )
    parser.add_argument(
        "--vision_only",
        action="store_true",
        help="Also run the single-shot vision adapter (no tools, AXTree-as-text only).",
    )
    parser.add_argument(
        "--no_headless",
        action="store_true",
        help="Render the browser visibly (default is headless).",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    api_key, base_url = _resolve_api_key(args.api_key)
    if args.base_url:
        base_url = args.base_url

    if args.dry_run:
        api_key = None  # forced off — heuristic only
        base_url = None
    elif api_key is None:
        print(
            "[NOTE] No API key found (checked --api_key, OPENROUTER_API_KEY, "
            "OPENAI_API_KEY, api_keys.py).  Continuing with heuristic only; "
            "tool-loop and vision-only heads will be skipped."
        )

    if args.synthetic:
        targets: List[Tuple[str, str]] = [("synthetic_shopmart", "synthetic")]
    else:
        urls = args.urls or DEFAULT_URLS
        targets = [(_url_safe(u), u) for u in urls]

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "output" / DEFAULT_OUTPUT_TAG
    )
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "head_summary": "heuristic + tool_loop"
                        + (" + grounding" if args.grounding else "")
                        + (" + vision_only" if args.vision_only else ""),
        "model": args.model,
        "dry_run": args.dry_run,
        "synthetic": args.synthetic,
        "episodes_per_target": args.episodes,
        "max_steps": args.max_steps,
        "targets": [{"tag": t, "url": u} for t, u in targets],
        "runs": [],
    }

    for tag, url in targets:
        for ep in range(args.episodes):
            ep_dir = out_root / tag / f"{run_id}_ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  -> {tag}  episode {ep + 1}/{args.episodes}  -> {ep_dir}")

            try:
                if args.synthetic:
                    observations = _capture_synthetic_episode(args.max_steps)
                else:
                    observations = _capture_browsergym_episode(
                        url, args.max_steps, headless=not args.no_headless,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  [capture-error] {tag}: {exc}")
                if args.verbose:
                    traceback.print_exc()
                master["runs"].append({"tag": tag, "url": url, "error": str(exc)})
                continue

            if not observations:
                print(f"  [capture-empty] {tag}: no observations")
                master["runs"].append(
                    {"tag": tag, "url": url, "error": "no_observations"},
                )
                continue

            try:
                records, stats = run_one_episode(
                    url,
                    observations=observations,
                    out_dir=ep_dir,
                    model=args.model,
                    api_key=api_key,
                    base_url=base_url,
                    max_rounds=args.max_rounds,
                    max_entities=args.max_entities,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    dry_run=args.dry_run,
                    do_grounding=args.grounding,
                    do_vision_only=args.vision_only,
                    verbose=args.verbose,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [run-error] {tag} ep{ep}: {exc}")
                if args.verbose:
                    traceback.print_exc()
                master["runs"].append(
                    {"tag": tag, "url": url, "episode": ep, "error": str(exc)},
                )
                continue

            stats["episode"] = ep
            stats["url"] = url

            steps_path = ep_dir / "steps.jsonl"
            with steps_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            with (ep_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

            print(
                f"     wrote {len(records)} steps  "
                f"heuristic_ok={stats.get('heuristic_ok', 0)}  "
                f"tool_ok={stats.get('tool_ok', 0)}"
            )
            master["runs"].append({**stats, "path": str(ep_dir)})

    master_path = out_root / f"batch_{run_id}.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Batch summary: {master_path}")


if __name__ == "__main__":
    main()
