#!/usr/bin/env python
"""
OSWorld **image-only** → ``<state>`` schema rollouts.

Captures OSWorld desktop observations (real DesktopEnv via Docker /
VMware / AWS, or a synthetic stub) and at every step asks a vision LLM
for the canonical ``<state>...</state>`` schema *from the screenshot
alone* — the accessibility tree is **not** in the prompt (it's only
saved on disk for reference / comparison).

For each step we save:
  1. ``images/step_<NNN>.png``    — the visual state of the env (PRIMARY VLM input)
  2. ``schema_image_llm``          — schema produced by the vision LLM
  3. ``schema_grounding``          — schema produced by OmniParser-v2
                                     (``--grounding``, optional)
  4. ``schema_xml_heuristic``      — deterministic AT-SPI XML walker
                                     baseline (always recorded so vision
                                     vs. heuristic schemas are
                                     side-by-side comparable; this is
                                     :func:`osworld_wrapper.heuristic.obs_to_schema`)

Output layout (under ``visual_grounding_tests/output/osworld_image/``)::

    <task_slug>/<run_id>_ep<NNN>/
      images/step_<NNN>.png   — one PNG per step (the actual VLM input)
      steps.jsonl             — one JSON record per step
      run_summary.json        — timing / counts / model id

Usage examples::

    # Synthetic desktop, dry-run (no API; only heuristic baseline + frames).
    python visual_grounding_tests/generate_osworld_image_schema.py \\
        --synthetic --dry_run --max_steps 2

    # Synthetic desktop, vision head against OpenAI / OpenRouter.
    export OPENAI_API_KEY=...
    python visual_grounding_tests/generate_osworld_image_schema.py \\
        --synthetic --max_steps 2 -v

    # Add the OmniParser-v2 grounding head (heavy local deps).
    python visual_grounding_tests/generate_osworld_image_schema.py \\
        --synthetic --grounding --max_steps 2

    # Real OSWorld via Docker provider (requires the OSWorld VM image).
    python visual_grounding_tests/generate_osworld_image_schema.py \\
        --task_catalog ../OSWorld/evaluation_examples/test_small.json \\
        --provider docker --max_steps 3
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

logger = logging.getLogger("osworld_image_schema")


# ── Models / defaults ────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("VLM_OSWORLD_IMAGE_MODEL", "openai/gpt-4.1")
DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_OSWORLD_IMAGE_MAX_TOKENS", "1500"))
DEFAULT_TEMPERATURE = float(os.environ.get("VLM_OSWORLD_IMAGE_TEMPERATURE", "0.2"))
DEFAULT_OUTPUT_TAG = "osworld_image"


# ── Lazy / optional imports ─────────────────────────────────────────────

def _import_vision_adapter():
    """Single-shot vision LLM adapter."""
    from osworld_wrapper.adapter import generate_label, osworld_obs_to_schema
    return generate_label, osworld_obs_to_schema


def _import_grounding_adapter():
    """OmniParser-v2 grounding adapter. Optional."""
    try:
        from osworld_wrapper.grounding import grounding_osworld_obs_to_schema
        return grounding_osworld_obs_to_schema
    except Exception:
        return None


def _import_heuristic():
    from osworld_wrapper.heuristic import obs_to_schema
    return obs_to_schema


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
    try:
        from scripts.test_vlm_parsers import _synthesize_desktop
        return _synthesize_desktop
    except Exception:
        return None


def _import_osworld_wrappers():
    try:
        from env_wrappers.osworld_wrapper import OSWorldGymWrapper, load_task_catalog
        return OSWorldGymWrapper, load_task_catalog
    except Exception:
        return None, None


# ── API key resolution ─────────────────────────────────────────────────

def _resolve_api_key(explicit: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
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
#
# The XML matches the namespaced AT-SPI shape OSWorld actually returns.
# It's only used for the heuristic baseline; the vision LLM never sees it.

_SYNTHETIC_A11Y_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<root xmlns:st="https://accessibility.ubuntu.example.org/ns/state"
      xmlns:cp="https://accessibility.ubuntu.example.org/ns/component"
      xmlns:attr="https://accessibility.ubuntu.example.org/ns/attributes">
  <application name="GNOME Shell" st:visible="true" st:showing="true">
    <menu-bar name="" st:visible="true" st:showing="true"
              cp:screencoord="(0, 0)" cp:size="(1280, 28)">
      <menu-item name="Activities" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(16, 6)" cp:size="(80, 22)" />
      <menu-item name="Applications" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(110, 6)" cp:size="(100, 22)" />
      <menu-item name="Places" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(220, 6)" cp:size="(70, 22)" />
    </menu-bar>
    <push-button name="Files" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(60, 80)" cp:size="(64, 64)" />
    <push-button name="Trash" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(60, 190)" cp:size="(64, 64)" />
    <push-button name="Firefox" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(60, 410)" cp:size="(64, 64)" />
    <push-button name="Terminal" st:visible="true" st:showing="true" st:enabled="true"
                 cp:screencoord="(60, 520)" cp:size="(64, 64)" />
    <frame name="Terminal — bash" st:visible="true" st:showing="true"
           cp:screencoord="(120, 120)" cp:size="(600, 340)">
      <text name="prompt" st:visible="true" st:showing="true" st:editable="true"
            cp:screencoord="(135, 160)" cp:size="(400, 24)">user@linux:~$ ls</text>
    </frame>
    <tool-bar name="taskbar" st:visible="true" st:showing="true"
              cp:screencoord="(0, 770)" cp:size="(1280, 30)" />
  </application>
</root>
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
    OSWorldGymWrapper, _ = _import_osworld_wrappers()
    if OSWorldGymWrapper is None:
        raise RuntimeError(
            "OSWorldGymWrapper not importable. Install OSWorld or run with "
            "--synthetic."
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


# ── Schema validation ──────────────────────────────────────────────────

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


# ── Image-only vision head ─────────────────────────────────────────────

def generate_image_schema_llm(
    *,
    obs: Dict[str, Any],
    pil: Any,
    task_id: str,
    step: int,
    helpers: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    max_tokens: int,
    max_entities: int,
    img_size: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    """Image-PRIMARY VLM head — calls
    :func:`osworld_wrapper.adapter.generate_label` directly so we control
    exactly what (if any) text grounding is shipped alongside the image.

    We pass only the task instruction as auxiliary context — the
    accessibility tree is deliberately **NOT** included so this head's
    schema is comparable to a real visual-grounding agent that hasn't
    yet looked up element coordinates from the OS a11y service.
    """
    if pil is None:
        return {
            "schema": None,
            "raw": "Error: no image available",
            "warnings": ["no_image_for_visual_call"],
            "validation": None,
            "model": model,
            "head": "image_llm",
        }

    generate_label, _ = _import_vision_adapter()
    t0 = time.time()
    try:
        result = generate_label(
            pil,
            instruction=obs.get("instruction", "") or "",
            task_id=task_id,
            step=step,
            a11y_tree_xml="",       # image-only head: no XML grounding text
            terminal_output="",
            last_action=obs.get("last_action", "") or "",
            last_action_error=obs.get("last_action_error", "") or "",
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
            "warnings": [f"image_llm_failed: {exc}"],
            "validation": None,
            "model": model,
            "elapsed_seconds": round(time.time() - t0, 3),
            "head": "image_llm",
        }

    schema = result.get("schema")
    warnings, validation = _validate_schema_text(
        schema, image_size=img_size, helpers=helpers,
    )
    return {
        "schema": schema,
        "raw": (result.get("raw") or "")[:4000],
        "warnings": warnings + list(result.get("warnings", [])),
        "validation": validation if validation else result.get("validation"),
        "model": result.get("model", model),
        "elapsed_seconds": round(time.time() - t0, 3),
        "head": "image_llm",
    }


# ── OmniParser grounding head ──────────────────────────────────────────

def _run_grounding_head(
    obs: Dict[str, Any],
    *,
    step: int,
    task_id: str,
    max_entities: int,
    helpers: Dict[str, Any],
    img_size: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    grounding_osworld_obs_to_schema = _import_grounding_adapter()
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
            "warnings": [f"grounding_failed: {exc}"],
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


# ── Episode runner ─────────────────────────────────────────────────────

def run_one_episode(
    task_slug: str,
    *,
    observations: List[Dict[str, Any]],
    out_dir: Path,
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    max_entities: int,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
    do_image_llm: bool,
    do_grounding: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    obs_to_schema = _import_heuristic()
    helpers = _import_schema_helpers()

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "image_llm_ok": 0, "image_llm_fail": 0,
        "grounding_ok": 0, "grounding_fail": 0,
        "heuristic_ok": 0, "heuristic_fail": 0,
    }
    t_episode = time.time()

    for step, obs in enumerate(observations):
        # 1. Save the visual state — the PRIMARY input for the image head.
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
        task_id = f"osworld/{task_slug}"

        # 2. Heuristic baseline (always emitted so the image schema is
        #    side-by-side comparable against a deterministic ground truth).
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

        # 3. Image-only vision LLM head.
        image_llm_record: Optional[Dict[str, Any]] = None
        if do_image_llm and not dry_run and api_key and pil is not None:
            image_llm_record = generate_image_schema_llm(
                obs=obs,
                pil=pil,
                task_id=task_id,
                step=step,
                helpers=helpers,
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_entities=max_entities,
                img_size=img_size,
            )
            if image_llm_record.get("schema"):
                counts["image_llm_ok"] += 1
            else:
                counts["image_llm_fail"] += 1

        # 4. Optional: OmniParser-v2 grounding head.
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

        record: Dict[str, Any] = {
            "step": step,
            "task_id": task_id,
            "task_slug": task_slug,
            "instruction": instruction,
            "image_path": img_rel,
            "image_size": list(img_size) if img_size else None,
            "n_a11y_chars": len(obs.get("accessibility_tree", "") or ""),
            "has_terminal": bool(obs.get("terminal")),
            "last_action": obs.get("last_action") or None,
            "last_action_error": obs.get("last_action_error") or None,
            "schema_image_llm": image_llm_record,
            "schema_grounding": grounding_record,
            "schema_xml_heuristic": {
                "schema": heuristic_schema,
                "warnings": heuristic_warnings,
                "elapsed_seconds": heuristic_elapsed,
                "head": "heuristic",
            },
            "model_scheduled": model,
            "dry_run": dry_run,
        }
        records.append(record)

        if verbose:
            i_ents = (
                helpers["count_entities"](image_llm_record["schema"])
                if (image_llm_record and image_llm_record.get("schema")) else 0
            )
            g_ents = (
                helpers["count_entities"](grounding_record["schema"])
                if (grounding_record and grounding_record.get("schema")) else 0
            )
            h_ents = (
                helpers["count_entities"](heuristic_schema) if heuristic_schema else 0
            )
            print(
                f"    step {step:>2}  image={'yes' if img_rel else 'no '}  "
                f"image_ents={i_ents:>2}  ground_ents={g_ents:>2}  "
                f"heur_ents={h_ents:>2}"
            )

    elapsed = round(time.time() - t_episode, 3)
    head_set = ["heuristic"]
    if do_image_llm and api_key and not dry_run:
        head_set.append("image_llm")
    if do_grounding:
        head_set.append("grounding")
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
    _Wrapper, load_task_catalog = _import_osworld_wrappers()
    tasks: List[Dict[str, Any]] = []
    if load_task_catalog is not None:
        try:
            tasks = load_task_catalog(catalog_path, limit=None)
        except Exception:
            tasks = []
    if not tasks:
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    tasks.extend(v)
                elif isinstance(v, dict):
                    tasks.extend(v.values())
        elif isinstance(data, list):
            tasks = list(data)
    if task_id:
        tasks = [t for t in tasks if t.get("id") == task_id]
    if limit and limit > 0:
        tasks = tasks[:limit]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OSWorld image-only -> <state> schema rollouts. The screenshot "
            "is the PRIMARY input to a vision LLM (and optionally to "
            "OmniParser-v2); the AT-SPI accessibility tree is recorded for "
            "the deterministic heuristic baseline only and is NOT passed to "
            "the vision model."
        ),
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--synthetic", action="store_true",
        help="Skip OSWorld; use a synthetic desktop screenshot + namespaced AT-SPI XML stub.",
    )
    src.add_argument(
        "--task_catalog", type=str, default=None,
        help="Path to an OSWorld task catalog JSON.",
    )

    parser.add_argument("--task_id", default=None)
    parser.add_argument("--task_limit", type=int, default=2)
    parser.add_argument("--provider", default="docker")
    parser.add_argument("--no_headless", action="store_true")
    parser.add_argument("--screen_width", type=int, default=1280)
    parser.add_argument("--screen_height", type=int, default=800)
    parser.add_argument(
        "--no_a11y_tree", action="store_true",
        help="Don't request the accessibility tree from OSWorld (disables the heuristic baseline).",
    )
    parser.add_argument("--no_terminal", action="store_true")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--max_entities", type=int, default=25)

    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Default: {DEFAULT_MODEL} (env: VLM_OSWORLD_IMAGE_MODEL).",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)

    parser.add_argument(
        "--dry_run", action="store_true",
        help="Skip API calls; only run env, save frames, and emit heuristic baseline.",
    )
    parser.add_argument(
        "--no_vision", action="store_true",
        help="Skip the single-shot vision head (heuristic + grounding only).",
    )
    parser.add_argument(
        "--grounding", action="store_true",
        help="Also run the OmniParser-v2 grounding head (heavy local deps).",
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
    elif api_key is None and not args.no_vision:
        print(
            "[NOTE] No API key found (checked --api_key, OPENROUTER_API_KEY, "
            "OPENAI_API_KEY, api_keys.py).  Vision head will be skipped; "
            "only the heuristic baseline (and --grounding if enabled) will run."
        )

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
            catalog_path, task_id=args.task_id, limit=args.task_limit,
        )
        if not tasks:
            print(f"[FATAL] No tasks loaded from {catalog_path}.")
            sys.exit(2)
        targets = [
            (_slugify(t.get("id") or t.get("instruction") or "task"), t)
            for t in tasks
        ]

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else SCRIPT_DIR / "output" / DEFAULT_OUTPUT_TAG
    )
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    head_summary_parts = ["heuristic"]
    if not args.no_vision:
        head_summary_parts.append("image_llm")
    if args.grounding:
        head_summary_parts.append("grounding")

    master: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "head_summary": " + ".join(head_summary_parts),
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
                master["runs"].append(
                    {"slug": slug, "task_id": task.get("id", ""), "error": str(exc)},
                )
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
                    max_entities=args.max_entities,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    dry_run=args.dry_run,
                    do_image_llm=not args.no_vision,
                    do_grounding=args.grounding,
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

            with (ep_dir / "steps.jsonl").open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            with (ep_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

            print(
                f"     wrote {len(records)} steps  "
                f"image_ok={stats.get('image_llm_ok', 0)}  "
                f"grounding_ok={stats.get('grounding_ok', 0)}  "
                f"heuristic_ok={stats.get('heuristic_ok', 0)}"
            )
            master["runs"].append({**stats, "path": str(ep_dir)})

    master_path = out_root / f"batch_{run_id}.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Batch summary: {master_path}")


if __name__ == "__main__":
    main()
