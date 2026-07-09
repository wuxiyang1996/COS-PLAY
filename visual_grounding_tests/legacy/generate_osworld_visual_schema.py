#!/usr/bin/env python
"""
OSWorld **visual** → ``<state>`` schema rollouts.

Renders the OSWorld desktop (xlang-ai/OSWorld), captures the screenshot
+ accessibility tree + terminal at every step via the
``osworld_wrapper`` package, and emits a canonical ``<state>...</state>``
block for each step using the same cross-domain machinery the BrowserGym
and ``env_wrappers`` rollouts already exercise:

  - **schema_vision_only** — :func:`osworld_wrapper.adapter.osworld_obs_to_schema`
    (vision LLM head — single-shot screenshot + a11y/terminal as text
    grounding).  This is the canonical default for ``cascaded_ground(domain="desktop")``.
  - **schema_grounding**   — :func:`osworld_wrapper.grounding.grounding_osworld_obs_to_schema`
    (OmniParser-v2 head — local YOLO + OCR + Florence-2; no API).
    Optional, requires the heavyweight vision extras.
  - **schema_tool_loop**   — :func:`vlm_wrapper.tool_loop.run_tool_loop`
    over :func:`osworld_wrapper.tools.build_osworld_registry` (multi-turn
    tool-augmented head: VLM identifies elements visually, then queries
    the OS accessibility tree to get pixel coordinates / state flags).

Capture sources (mirrors the BrowserGym ``--synthetic`` / ``--urls``
split so this script is runnable offline):

  - ``--synthetic``  uses :func:`scripts.test_vlm_parsers._synthesize_desktop`
    to fabricate a desktop screenshot + a tiny synthetic AT-SPI-style
    a11y XML, no VM required.  Useful for smoke-testing the schema
    pipeline.
  - ``--task_catalog`` boots the real OSWorld DesktopEnv via
    :class:`env_wrappers.osworld_wrapper.OSWorldGymWrapper` (Docker /
    VMware / AWS provider — see ``--provider``) and steps ``noop``-style
    pyautogui actions to capture a real desktop trace.

Output layout (under ``visual_grounding_tests/output/osworld/``)::

    <task_slug>/<run_id>_ep<NNN>/
      images/step_<NNN>.png   — one PNG per step (the actual VLM input)
      steps.jsonl             — one JSON record per step
      run_summary.json        — timing / counts / model id

Usage examples (run from ``Multi-hop-Reasoning-VLM-Agent`` or repo root with
that on ``PYTHONPATH``)::

    # Synthetic desktop, dry-run (no API; only saves frames + records).
    python visual_grounding_tests/generate_osworld_visual_schema.py \\
        --synthetic --dry_run --max_steps 2

    # Synthetic desktop, vision-only head against OpenAI / OpenRouter.
    export OPENAI_API_KEY=...
    python visual_grounding_tests/generate_osworld_visual_schema.py \\
        --synthetic --max_steps 2 -v

    # Add the OmniParser-v2 grounding head and the multi-turn tool-loop head.
    python visual_grounding_tests/generate_osworld_visual_schema.py \\
        --synthetic --grounding --tool_loop --max_steps 2

    # Real OSWorld via Docker provider (requires the OSWorld VM image).
    python visual_grounding_tests/generate_osworld_visual_schema.py \\
        --task_catalog ../OSWorld/evaluation_examples/test_small.json \\
        --provider docker --max_steps 3

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
CODEBASE_ROOT = SCRIPT_DIR.parent.parent
if str(CODEBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEBASE_ROOT))
REPO_ROOT = CODEBASE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("osworld_visual_schema")


# ── Models / defaults ────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("VLM_OSWORLD_MODEL", "openai/gpt-4.1")
DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_OSWORLD_MAX_TOKENS", "1500"))
DEFAULT_TEMPERATURE = float(os.environ.get("VLM_OSWORLD_TEMPERATURE", "0.2"))
DEFAULT_MAX_ROUNDS = int(os.environ.get("VLM_OSWORLD_MAX_ROUNDS", "5"))
DEFAULT_OUTPUT_TAG = "osworld"

_REASONING_TOKEN_MULTIPLIER = int(os.environ.get("VLM_OSWORLD_REASONING_MULT", "5"))
_REASONING_TOKEN_FLOOR = int(os.environ.get("VLM_OSWORLD_REASONING_FLOOR", "6000"))


# ── Lazy / optional imports ─────────────────────────────────────────────

def _import_osworld_vision_adapter():
    """Head 2: vision LLM single-shot adapter."""
    from osworld_wrapper.adapter import osworld_obs_to_schema
    return osworld_obs_to_schema


def _import_osworld_grounding_adapter():
    """Head 3: OmniParser-v2 grounding adapter.  Optional."""
    try:
        from osworld_wrapper.grounding import grounding_osworld_obs_to_schema
        return grounding_osworld_obs_to_schema
    except Exception:
        return None


def _import_osworld_tools():
    """Head 4 helpers: build_osworld_registry + run_tool_loop."""
    from osworld_wrapper.tools import build_osworld_registry
    from vlm_wrapper.tool_loop import run_tool_loop
    return build_osworld_registry, run_tool_loop


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


def _import_synth_desktop():
    """Fabricate a desktop screenshot via the existing test helper."""
    try:
        from scripts.test_vlm_parsers import _synthesize_desktop
        return _synthesize_desktop
    except Exception:
        return None


def _import_osworld_wrappers():
    """Real OSWorld env: requires Docker / VMware / AWS provider running."""
    try:
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper, load_task_catalog
        return OSWorldGymWrapper, load_task_catalog
    except Exception:
        return None, None


# ── API key resolution (mirrors generate_browsergym_schema.py) ─────────

def _resolve_api_key(explicit: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(api_key, base_url)`` resolved from CLI / env / api_keys.py.

    Precedence:
      1. Explicit ``--api_key``.
      2. ``OPENROUTER_API_KEY`` (env).
      3. ``OPENAI_API_KEY`` (env).
      4. ``api_keys.open_router_api_key`` / ``api_keys.openai_api_key``
         (next to repo root).
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


# ── Synthetic OSWorld observation factory ──────────────────────────────

_SYNTHETIC_A11Y_XML = """\
<accessibilitytree>
  <node roleName="application" name="GNOME Shell" />
  <node roleName="frame" name="Files" screencoord="(60, 80)" size="(64, 64)" showing="True" visible="True" />
  <node roleName="push button" name="Files" screencoord="(60, 80)" size="(64, 64)" showing="True" visible="True" enabled="True" />
  <node roleName="push button" name="Trash" screencoord="(60, 190)" size="(64, 64)" showing="True" visible="True" enabled="True" />
  <node roleName="push button" name="Firefox" screencoord="(60, 410)" size="(64, 64)" showing="True" visible="True" enabled="True" />
  <node roleName="push button" name="Terminal" screencoord="(60, 520)" size="(64, 64)" showing="True" visible="True" enabled="True" />
  <node roleName="frame" name="Terminal - bash" screencoord="(120, 120)" size="(600, 340)" showing="True" visible="True" />
  <node roleName="text" name="user@linux:~$" screencoord="(135, 160)" size="(120, 20)" showing="True" visible="True" />
  <node roleName="menu bar" name="" screencoord="(0, 0)" size="(1280, 28)" showing="True" visible="True" />
  <node roleName="menu item" name="Activities" screencoord="(16, 6)" size="(80, 22)" showing="True" visible="True" enabled="True" />
  <node roleName="menu item" name="Applications" screencoord="(110, 6)" size="(100, 22)" showing="True" visible="True" enabled="True" />
  <node roleName="menu item" name="Places" screencoord="(220, 6)" size="(70, 22)" showing="True" visible="True" enabled="True" />
  <node roleName="tool bar" name="taskbar" screencoord="(0, 770)" size="(1280, 30)" showing="True" visible="True" />
</accessibilitytree>
"""

_SYNTHETIC_TERMINAL = (
    "user@linux:~$ ls\n"
    "Desktop  Documents  Downloads  Pictures\n"
    "user@linux:~$ \n"
)

_SYNTHETIC_INSTRUCTION = (
    "Open the Files application and create a new folder called 'reports' on the Desktop."
)


def _build_synthetic_obs() -> Dict[str, Any]:
    """Fabricate an OSWorld observation dict for offline smoke-tests.

    Reuses :func:`scripts.test_vlm_parsers._synthesize_desktop` for the
    image so the synthetic frame matches the one the unit tests already
    assert against. Falls back to a flat-grey image if PIL / the helper
    are unavailable.
    """
    synth = _import_synth_desktop()
    if synth is not None:
        try:
            pil = synth()
            arr = np.asarray(pil.convert("RGB"))
        except Exception:
            arr = np.full((800, 1280, 3), 38, dtype=np.uint8)
    else:
        arr = np.full((800, 1280, 3), 38, dtype=np.uint8)

    return {
        "screenshot": arr,
        "accessibility_tree": _SYNTHETIC_A11Y_XML,
        "terminal": _SYNTHETIC_TERMINAL,
        "instruction": _SYNTHETIC_INSTRUCTION,
        "last_action": "",
        "last_action_error": "",
    }


def _capture_synthetic_episode(max_steps: int) -> List[Dict[str, Any]]:
    """Build a list of synthetic observations.

    The same fabricated obs is replayed ``max_steps`` times with
    ``last_action`` filled in after the first step — perfect for a local
    smoke-test that exercises every head without booting a VM.
    """
    base = _build_synthetic_obs()
    out: List[Dict[str, Any]] = []
    for step in range(max(1, max_steps)):
        clone = dict(base)
        clone["last_action"] = "" if step == 0 else "pyautogui.click(60, 410)"
        out.append(clone)
    return out


# ── Real OSWorld observation capture ───────────────────────────────────

def _capture_real_episode(
    *,
    task: Dict[str, Any],
    max_steps: int,
    provider: str,
    headless: bool,
    require_a11y_tree: bool,
    require_terminal: bool,
    screen_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Boot OSWorld DesktopEnv via OSWorldGymWrapper and step ``noop``-style.

    Each step issues ``WAIT`` so we capture frames without nudging the
    desktop — pure visual-grounding test, not a benchmark.
    """
    OSWorldGymWrapper, _load_catalog = _import_osworld_wrappers()
    if OSWorldGymWrapper is None:
        raise RuntimeError(
            "OSWorldGymWrapper not importable.  Install OSWorld or run "
            "with --synthetic."
        )

    logger.info("Booting OSWorldGymWrapper (provider=%s)…", provider)
    env = OSWorldGymWrapper(
        provider_name=provider,
        headless=headless,
        max_steps=max(2, max_steps + 1),
        require_a11y_tree=require_a11y_tree,
        require_terminal=require_terminal,
        screen_size=screen_size,
        task_catalog=[task],
    )

    captured: List[Dict[str, Any]] = []
    obs, _info = env.reset()
    captured.append({**dict(obs), "last_action": "", "last_action_error": ""})

    for i in range(max(0, max_steps - 1)):
        try:
            obs, _r, term, trunc, step_info = env.step("WAIT")
        except Exception as exc:  # noqa: BLE001
            logger.warning("step %d failed (%s); stopping", i, exc)
            break
        captured.append({
            **dict(obs),
            "last_action": "WAIT",
            "last_action_error": (step_info or {}).get("error", "") or "",
        })
        if term or trunc:
            break

    try:
        env.close()
    except Exception:
        pass
    return captured


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
            schema_text, domain="desktop", image_size=image_size,
        )
    except Exception as exc:  # noqa: BLE001
        return warnings + [f"semantic_validate_failed: {exc}"], None
    validation = vres.as_dict()
    warnings = warnings + list(vres.warnings) + list(vres.errors)
    return warnings, validation


# ── Head wrappers ──────────────────────────────────────────────────────

def _run_vision_head(
    obs: Dict[str, Any],
    *,
    step: int,
    task_id: str,
    max_entities: int,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
    helpers: Dict[str, Any],
    img_size: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    """Single-shot vision LLM head — :func:`osworld_obs_to_schema`."""
    osworld_obs_to_schema = _import_osworld_vision_adapter()
    t0 = time.time()
    try:
        result = osworld_obs_to_schema(
            obs,
            step=step,
            task_id=task_id,
            max_entities=max_entities,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": None,
            "raw": f"Error: {exc!r}",
            "warnings": [f"vision_head_failed: {exc}"],
            "validation": None,
            "model": model,
            "elapsed_seconds": round(time.time() - t0, 3),
            "head": "vision_only",
        }
    warnings, validation = _validate_schema_text(
        result.get("schema"), image_size=img_size, helpers=helpers,
    )
    return {
        "schema": result.get("schema"),
        "raw": (result.get("raw") or "")[:4000],
        "warnings": warnings + list(result.get("warnings", [])),
        "validation": validation if validation else result.get("validation"),
        "model": result.get("model", model),
        "elapsed_seconds": round(time.time() - t0, 3),
        "head": "vision_only",
    }


def _run_grounding_head(
    obs: Dict[str, Any],
    *,
    step: int,
    task_id: str,
    max_entities: int,
    helpers: Dict[str, Any],
    img_size: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    """OmniParser-v2 head — :func:`grounding_osworld_obs_to_schema`."""
    grounding_osworld_obs_to_schema = _import_osworld_grounding_adapter()
    if grounding_osworld_obs_to_schema is None:
        return {
            "schema": None,
            "warnings": ["grounding_head_unavailable"],
            "validation": None,
            "model": "omniparser-v2",
            "elapsed_seconds": 0.0,
            "head": "grounding",
        }
    t0 = time.time()
    try:
        gres = grounding_osworld_obs_to_schema(
            obs, step=step, task_id=task_id, max_entities=max_entities,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": None,
            "warnings": [f"grounding_head_failed: {exc}"],
            "validation": None,
            "model": "omniparser-v2",
            "elapsed_seconds": round(time.time() - t0, 3),
            "head": "grounding",
        }
    warnings, validation = _validate_schema_text(
        gres.get("schema"), image_size=img_size, helpers=helpers,
    )
    return {
        "schema": gres.get("schema"),
        "warnings": warnings + list(gres.get("warnings", [])),
        "validation": validation,
        "model": gres.get("model", "omniparser-v2"),
        "n_elements": len(gres.get("elements") or []),
        "elapsed_seconds": round(time.time() - t0, 3),
        "head": "grounding",
    }


def _run_tool_loop_head(
    obs: Dict[str, Any],
    pil: Any,
    *,
    step: int,
    task_id: str,
    goal: str,
    max_entities: int,
    max_rounds: int,
    model: str,
    api_key: str,
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
    helpers: Dict[str, Any],
    img_size: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    """Multi-turn tool-augmented head over the OSWorld a11y registry."""
    build_osworld_registry, run_tool_loop = _import_osworld_tools()
    t0 = time.time()
    try:
        registry = build_osworld_registry(
            a11y_tree_xml=obs.get("accessibility_tree", "") or "",
            instruction=obs.get("instruction", "") or "",
            terminal_output=obs.get("terminal", "") or "",
        )
        tool_loop_result = run_tool_loop(
            pil,
            domain="desktop",
            registry=registry,
            goal=goal,
            task_id=task_id,
            step=step,
            extra_context=(
                "You have OS-level grounding tools (query_os_element, "
                "query_entity_pos, get_state_flags) backed by the desktop "
                "accessibility tree. Identify entities visually first, "
                "then call query_os_element with the visible name to fetch "
                "ground-truth pixel coordinates and state flags before "
                "emitting the final <state>...</state> schema. Use absolute "
                "pixel coordinates in the <targets> section so pyautogui "
                "actions click the right element."
            ),
            max_entities=max_entities,
            max_rounds=max_rounds,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": None,
            "raw": f"Error: {exc!r}",
            "warnings": [f"tool_loop_failed: {exc}"],
            "validation": None,
            "model": model,
            "rounds": 0,
            "tool_trace": [],
            "elapsed_seconds": round(time.time() - t0, 3),
            "head": "tool_loop",
        }
    warnings, validation = _validate_schema_text(
        tool_loop_result.get("schema"), image_size=img_size, helpers=helpers,
    )
    return {
        "schema": tool_loop_result.get("schema"),
        "raw": (tool_loop_result.get("raw") or "")[:4000],
        "warnings": warnings,
        "validation": validation,
        "model": tool_loop_result.get("model", model),
        "rounds": tool_loop_result.get("rounds"),
        "tool_trace": tool_loop_result.get("tool_trace", []),
        "elapsed_seconds": round(time.time() - t0, 3),
        "head": "tool_loop",
    }


# ── Episode runner ─────────────────────────────────────────────────────

def run_one_episode(
    task_slug: str,
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
    do_tool_loop: bool,
    do_vision_only: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    helpers = _import_schema_helpers()

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "vision_only_ok": 0, "vision_only_fail": 0,
        "grounding_ok": 0, "grounding_fail": 0,
        "tool_ok": 0, "tool_fail": 0,
    }
    t_episode = time.time()

    for step, obs in enumerate(observations):
        # 1. Save the visual state.
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

        instruction = obs.get("instruction", "") or ""
        goal = instruction.strip().split("\n")[0] if instruction else ""
        task_id = f"osworld/{task_slug}"

        # 2. Vision-only head (schema_vision_only) — single-shot VLM.
        vision_only_record: Optional[Dict[str, Any]] = None
        if do_vision_only and not dry_run and api_key and pil is not None:
            vision_only_record = _run_vision_head(
                obs,
                step=step,
                task_id=task_id,
                max_entities=max_entities,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                helpers=helpers,
                img_size=img_size,
            )
            if vision_only_record.get("schema"):
                counts["vision_only_ok"] += 1
            else:
                counts["vision_only_fail"] += 1
            if verbose and vision_only_record.get("schema") is None:
                logger.warning(
                    "[vision] step %d: %s",
                    step,
                    "; ".join(str(w) for w in vision_only_record.get("warnings", []))[:200],
                )

        # 3. OmniParser-v2 grounding head (schema_grounding) — no LLM.
        grounding_record: Optional[Dict[str, Any]] = None
        if do_grounding and pil is not None:
            grounding_record = _run_grounding_head(
                obs,
                step=step,
                task_id=task_id,
                max_entities=max_entities,
                helpers=helpers,
                img_size=img_size,
            )
            if grounding_record.get("schema"):
                counts["grounding_ok"] += 1
            else:
                counts["grounding_fail"] += 1

        # 4. Tool-loop head (schema_tool_loop) — multi-turn VLM + a11y tools.
        tool_record: Optional[Dict[str, Any]] = None
        if do_tool_loop and not dry_run and api_key and pil is not None:
            tool_record = _run_tool_loop_head(
                obs,
                pil,
                step=step,
                task_id=task_id,
                goal=goal,
                max_entities=max_entities,
                max_rounds=max_rounds,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                helpers=helpers,
                img_size=img_size,
            )
            if tool_record.get("schema"):
                counts["tool_ok"] += 1
            else:
                counts["tool_fail"] += 1

        a11y_xml = obs.get("accessibility_tree", "") or ""
        a11y_lines = a11y_xml.count("\n")

        record: Dict[str, Any] = {
            "step": step,
            "task_id": task_id,
            "task_slug": task_slug,
            "instruction": instruction,
            "goal": goal,
            "image_path": img_rel,
            "image_size": list(img_size) if img_size else None,
            "n_a11y_lines": a11y_lines,
            "has_terminal": bool(obs.get("terminal")),
            "last_action": obs.get("last_action") or None,
            "last_action_error": obs.get("last_action_error") or None,
            "schema_vision_only": vision_only_record,
            "schema_grounding": grounding_record,
            "schema_tool_loop": tool_record,
            "model_scheduled": model,
            "dry_run": dry_run,
        }
        records.append(record)

        if verbose:
            v_ents = (
                helpers["count_entities"](vision_only_record["schema"])
                if (vision_only_record and vision_only_record.get("schema")) else 0
            )
            t_ents = (
                helpers["count_entities"](tool_record["schema"])
                if (tool_record and tool_record.get("schema")) else 0
            )
            g_ents = (
                helpers["count_entities"](grounding_record["schema"])
                if (grounding_record and grounding_record.get("schema")) else 0
            )
            print(
                f"    step {step:>2}  image={'yes' if img_rel else 'no '}  "
                f"vision_ents={v_ents:>2}  ground_ents={g_ents:>2}  "
                f"tool_ents={t_ents:>2}"
            )

    elapsed = round(time.time() - t_episode, 3)
    head_set = []
    if do_vision_only and api_key and not dry_run:
        head_set.append("vision_only")
    if do_grounding:
        head_set.append("grounding")
    if do_tool_loop and api_key and not dry_run:
        head_set.append("tool_loop")

    summary: Dict[str, Any] = {
        "task_slug": task_slug,
        "head_set": head_set,
        "steps_recorded": len(records),
        "elapsed_seconds": elapsed,
        "model": model,
        "dry_run": dry_run,
        **counts,
    }
    return records, summary


# ── CLI ────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Best-effort filesystem-safe slug for a task id / instruction."""
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "-._/ ":
            out.append("_")
    slug = "".join(out).strip("_")[:120]
    return slug or "task"


def _load_tasks_from_catalog(
    catalog_path: Path,
    *,
    task_id: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Load ``OSWorld`` task configs from a JSON catalog (test_small.json…)."""
    _Wrapper, load_task_catalog = _import_osworld_wrappers()
    if load_task_catalog is not None:
        try:
            tasks = load_task_catalog(catalog_path, limit=None)
        except Exception:
            tasks = []
    else:
        tasks = []

    if not tasks:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            tasks = []
            for v in data.values():
                if isinstance(v, list):
                    tasks.extend(v)
                elif isinstance(v, dict):
                    tasks.extend(v.values())
        elif isinstance(data, list):
            tasks = list(data)
        else:
            tasks = []

    if task_id:
        tasks = [t for t in tasks if t.get("id") == task_id]

    if limit and limit > 0:
        tasks = tasks[:limit]

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OSWorld desktop -> <state> schema rollouts.  Renders the OSWorld "
            "desktop (real DesktopEnv via Docker / VMware / AWS, or a "
            "synthetic stub) and records vision / grounding / tool-loop "
            "schemas at every step."
        ),
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Skip OSWorld; use a synthesized desktop screenshot + a tiny "
            "AT-SPI-style a11y XML.  Useful for offline smoke-tests."
        ),
    )
    src.add_argument(
        "--task_catalog",
        type=str,
        default=None,
        help=(
            "Path to an OSWorld task catalog JSON "
            "(e.g. ../OSWorld/evaluation_examples/test_small.json).  "
            "Boots OSWorldGymWrapper for each task."
        ),
    )

    parser.add_argument(
        "--task_id",
        default=None,
        help="Filter the task catalog to a single task ID.",
    )
    parser.add_argument(
        "--task_limit",
        type=int,
        default=2,
        help="Max number of tasks to take from the catalog (default 2).",
    )
    parser.add_argument(
        "--provider",
        default="docker",
        help="OSWorld provider: docker / vmware / virtualbox / aws (default docker).",
    )
    parser.add_argument(
        "--no_headless",
        action="store_true",
        help="Render the OSWorld VM visibly (default is headless).",
    )
    parser.add_argument(
        "--screen_width", type=int, default=1280,
        help="OSWorld VM screen width (default 1280).",
    )
    parser.add_argument(
        "--screen_height", type=int, default=800,
        help="OSWorld VM screen height (default 800).",
    )
    parser.add_argument(
        "--no_a11y_tree", action="store_true",
        help="Do NOT request the accessibility tree from OSWorld (faster).",
    )
    parser.add_argument(
        "--no_terminal", action="store_true",
        help="Do NOT request terminal output from OSWorld (faster, default).",
    )

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--max_entities", type=int, default=25)
    parser.add_argument(
        "--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help="Max VLM tool-call rounds before forcing a final schema.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Default: {DEFAULT_MODEL} (env: VLM_OSWORLD_MODEL).",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip API calls; only run env, save frames, and emit grounding (if enabled).",
    )
    parser.add_argument(
        "--no_vision",
        action="store_true",
        help="Skip the single-shot vision head.",
    )
    parser.add_argument(
        "--grounding",
        action="store_true",
        help="Also run the OmniParser-v2 grounding head (heavy deps required).",
    )
    parser.add_argument(
        "--tool_loop",
        action="store_true",
        help="Also run the multi-turn tool-loop head (uses query_os_element).",
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
        api_key = None
        base_url = None
    elif api_key is None and (not args.no_vision or args.tool_loop):
        print(
            "[NOTE] No API key found (checked --api_key, OPENROUTER_API_KEY, "
            "OPENAI_API_KEY, api_keys.py).  Vision and tool-loop heads will be "
            "skipped; only --grounding (if enabled) will produce schemas."
        )

    # Default to synthetic when no source is selected, so the script is
    # runnable offline without OSWorld installed.
    if not args.synthetic and not args.task_catalog:
        args.synthetic = True

    targets: List[Tuple[str, Dict[str, Any]]] = []
    if args.synthetic:
        targets = [("synthetic_desktop", {"id": "synthetic_desktop"})]
    else:
        catalog_path = Path(args.task_catalog).expanduser().resolve()
        if not catalog_path.exists():
            print(f"[FATAL] Task catalog not found: {catalog_path}")
            sys.exit(2)
        tasks = _load_tasks_from_catalog(
            catalog_path,
            task_id=args.task_id,
            limit=args.task_limit,
        )
        if not tasks:
            print(f"[FATAL] No tasks loaded from {catalog_path}.")
            sys.exit(2)
        targets = [(_slugify(t.get("id") or t.get("instruction") or "task"), t) for t in tasks]

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "output" / DEFAULT_OUTPUT_TAG
    )
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    head_summary_parts = []
    if not args.no_vision:
        head_summary_parts.append("vision_only")
    if args.grounding:
        head_summary_parts.append("grounding")
    if args.tool_loop:
        head_summary_parts.append("tool_loop")
    head_summary = " + ".join(head_summary_parts) or "(none)"

    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "head_summary": head_summary,
        "model": args.model,
        "dry_run": args.dry_run,
        "synthetic": args.synthetic,
        "provider": None if args.synthetic else args.provider,
        "episodes_per_target": args.episodes,
        "max_steps": args.max_steps,
        "targets": [{"slug": s, "task_id": t.get("id", "")} for s, t in targets],
        "runs": [],
    }

    for slug, task in targets:
        for ep in range(args.episodes):
            ep_dir = out_root / slug / f"{run_id}_ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  -> {slug}  episode {ep + 1}/{args.episodes}  -> {ep_dir}")

            try:
                if args.synthetic:
                    observations = _capture_synthetic_episode(args.max_steps)
                else:
                    observations = _capture_real_episode(
                        task=task,
                        max_steps=args.max_steps,
                        provider=args.provider,
                        headless=not args.no_headless,
                        require_a11y_tree=not args.no_a11y_tree,
                        require_terminal=not args.no_terminal,
                        screen_size=(args.screen_width, args.screen_height),
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  [capture-error] {slug}: {exc}")
                if args.verbose:
                    traceback.print_exc()
                master["runs"].append({"slug": slug, "task_id": task.get("id", ""), "error": str(exc)})
                continue

            if not observations:
                print(f"  [capture-empty] {slug}: no observations")
                master["runs"].append(
                    {"slug": slug, "task_id": task.get("id", ""), "error": "no_observations"},
                )
                continue

            try:
                records, stats = run_one_episode(
                    slug,
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
                    do_tool_loop=args.tool_loop,
                    do_vision_only=not args.no_vision,
                    verbose=args.verbose,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [run-error] {slug} ep{ep}: {exc}")
                if args.verbose:
                    traceback.print_exc()
                master["runs"].append(
                    {"slug": slug, "task_id": task.get("id", ""), "episode": ep, "error": str(exc)},
                )
                continue

            stats["episode"] = ep
            stats["task_id"] = task.get("id", "")

            steps_path = ep_dir / "steps.jsonl"
            with steps_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            with (ep_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

            print(
                f"     wrote {len(records)} steps  "
                f"vision_ok={stats.get('vision_only_ok', 0)}  "
                f"grounding_ok={stats.get('grounding_ok', 0)}  "
                f"tool_ok={stats.get('tool_ok', 0)}"
            )
            master["runs"].append({**stats, "path": str(ep_dir)})

    master_path = out_root / f"batch_{run_id}.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Batch summary: {master_path}")


if __name__ == "__main__":
    main()
