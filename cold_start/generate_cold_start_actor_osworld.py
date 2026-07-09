#!/usr/bin/env python
"""
Cold-start actor-agent rollouts for **OSWorld** (gpt-5.5 vision pipeline).

Pipeline (one outer step):

  1. Pull a multimodal observation from a real OSWorld ``DesktopEnv``
     (Docker / VMware / AWS-backed Ubuntu KVM guest) wrapped in
     :class:`env_wrappers.osworld_wrapper.OSWorldGymWrapper`:

       - ``screenshot``          NumPy ``HxWx3 uint8`` framebuffer
       - ``accessibility_tree``  namespaced AT-SPI / UI-Automation XML
       - ``terminal``            recent VM terminal output
       - ``instruction``         natural-language task description

  2. Visual schema (deterministic fallback) —
     :func:`osworld_wrapper.heuristic.obs_to_schema` walks the namespaced
     AT-SPI XML into the canonical ``<state>...</state>`` block. Free,
     reproducible, ~6 ms; always emitted as a fallback.

  3. Visual schema (VLM, primary) — gpt-5.5 (vision) reads the desktop
     **screenshot** plus the AXTree-as-text grounding context and emits
     the canonical schema via :mod:`vlm_wrapper.schema`. The screenshot
     is the primary input; the AT-SPI tree rides along so the VLM can
     produce real pixel coordinates that match accessible elements.

  4. Action selection — gpt-5.5 reads the schema + a candidate-action
     list (common ``pyautogui.*`` skeletons + the special tokens
     ``DONE`` / ``FAIL`` / ``WAIT``) and picks ONE action via OpenAI
     function calling. The function-call schema separates ``action_type``
     / ``x``, ``y`` / ``text`` / ``key`` / ``keys`` so the actor can
     also fill text into an input box, press hotkeys, scroll, etc.

  5. ``env.step(action_string)`` → ``(obs, reward, term, trunc, info)``.
     Each :class:`Experience` carries the schema, raw VLM output,
     reasoning, action, reward, terminate / truncate, eval score (when
     available), and the saved screenshot path.

Companion to:
  - ``cold_start/generate_cold_start_actor.py`` (env_wrappers, 4 games)
  - ``cold_start/generate_cold_start_actor_gymv.py`` (gym-v Temporal)
  - ``cold_start/generate_cold_start_actor_browsergym.py`` (BrowserGym)

Same Episode/Experience output format, but driven through OSWorld's live
``DesktopEnv`` screenshot + AT-SPI obs API. The Docker provider is the
recommended default — it auto-boots the Ubuntu KVM guest from the
``Ubuntu.qcow2`` blob in ``./docker_vm_data/`` (see
``osworld_wrapper/README.md`` for VM install).

Output layout (``<codebase_root>/Cold-start-out-osworld/<domain>/<safe_task_id>/``):

  - ``episode_NNN.json``        individual Episode (Episode.to_dict())
  - ``episode_buffer.json``     Episode_Buffer (loadable for trainer)
  - ``rollouts.jsonl``          append-only JSONL, one Episode per line
  - ``rollout_summary.json``    per-task stats
  - ``frames/<ep>/step_NNN.png``  rendered screenshots fed to the VLM
  - ``frames/<ep>/step_NNN.json`` per-step sidecar (action / schema / eval)

Usage::

    export OPENAI_API_KEY="sk-..."          # or OPENROUTER_API_KEY

    # Default: 1 episode of the smoke task on Docker, 50 steps (OSWorld
    #          standard cap), --resume, frames saved (PNG + sidecar JSON),
    #          headless VM, gpt-5.5 vision
    python cold_start/generate_cold_start_actor_osworld.py

    # All 10 OSWorld domains, 1 episode per task, 50 steps each (the
    # published OSWorld evaluation cap — gives gimp/office tasks room to
    # finish multi-dialog flows that 15 steps could not).
    python cold_start/generate_cold_start_actor_osworld.py \\
        --task_catalog /workspace/OSWorld/evaluation_examples/test_small.json \\
        --episodes 1 --max_steps 50 -v

    # Restrict to 2 domains
    python cold_start/generate_cold_start_actor_osworld.py \\
        --task_catalog /workspace/OSWorld/evaluation_examples/test_small.json \\
        --domains chrome os --episodes 1 --max_steps 50 -v

    # Specific task ids
    python cold_start/generate_cold_start_actor_osworld.py \\
        --task_catalog /workspace/OSWorld/evaluation_examples/test_small.json \\
        --task_ids 5ea617a3-0e86-4ba6-aab2-dac9aa2e8d57 \\
        --max_steps 50 -v

    # List task domains/IDs without running
    python cold_start/generate_cold_start_actor_osworld.py \\
        --task_catalog /workspace/OSWorld/evaluation_examples/test_small.json \\
        --list_tasks

    # Skip frame persistence (rollouts.jsonl + episode_NNN.json still record
    # the schema and action — just no PNGs on disk)
    python cold_start/generate_cold_start_actor_osworld.py --no_save_frames

Pre-reqs:
  - ``osworld`` conda env (``bash install/install_osworld.sh``)
  - Docker daemon running + ``happysixd/osworld-docker`` image pulled
  - ``./docker_vm_data/Ubuntu.qcow2`` (~23 GB) extracted at the launch cwd
  - OpenAI/OpenRouter API key (``api_keys.py`` next to the repo root works)

Hard-wired modes (NO opt-out):
  - The VM is **always headless** (no GUI on the host; an Xvfb display is
    used for any host-side rendering hooks).
  - The VLM (gpt-5.5 vision) **always** produces the schema; the
    deterministic AT-SPI heuristic is kept as a fallback only.
  - Frames are saved to disk **by default** (pass ``--no_save_frames``
    to skip when disk pressure matters).

Default-on improvements (opt-out flags exist for ablations):
  - **Set-of-Marks visual grounding** (``--no_som`` to disable): every
    interactive AT-SPI element gets a numbered red box drawn on the
    screenshot, and ``click_element(id=N)`` enters the action vocab.
    The harness translates the SoM verb back to ``pyautogui.click(cx,
    cy)`` before stepping the env. Published baselines move ~5% →
    ~18% pass@1 with this single change.
  - **Anti-loop early-DONE** (``--loop_repeat_threshold 999`` to
    disable): force ``DONE`` when the agent repeats the same action
    with no reward — kills 30+-step "click → escape → click" timeouts.
  - **DONE-nudge prompt** (``--done_nudge_step 999`` to disable): adds
    a "stop verifying, commit to DONE" reminder to the action prompt
    once the trajectory has run for >12 steps.
  - **Anti-noop replan**: when the previous step changed nothing, the
    next action prompt is told the action had no effect so the model
    picks a different element / strategy.

Expected pass rates (calibration reference, with SoM enabled):
  - chrome (settings, tab management) .... ~30% pass@1 with max_steps=50
  - libreoffice_writer / impress / calc .... ~15% pass@1 with max_steps=50
  - gimp / vlc / vs_code / thunderbird .... ~5-15% pass@1 with max_steps=50
  - os (file/system shortcuts) ............ ~25% pass@1 with max_steps=50
  - multi_apps ............................ ~5%  pass@1 with max_steps=50
Published vision-only baselines for the same protocol with SoM cluster
around 18-24% overall (GPT-4V+SoM ~18%, Claude-3.5 Sonnet+SoM ~24%);
without SoM the same backbones drop to 5-15%. At max_steps=15 expect
to roughly halve those numbers (the long-tail of multi-dialog tasks
runs out of budget). Bumping ``--episodes 3`` and averaging gives a
tighter estimate at 3x the wall-clock.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make the codebase + workspace root importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in [str(CODEBASE_ROOT), str(WORKSPACE_ROOT)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_api_keys_from_file() -> Optional[Path]:
    """Seed ``os.environ`` from a sibling ``api_keys.py`` if present.

    Looked-up locations (first hit wins):
      1. ``$COSPLAY_API_KEYS_FILE``
      2. ``cold_start/api_keys.py``
      3. ``<codebase_root>/api_keys.py``
      4. ``<codebase_root>/../api_keys.py``  (workspace root)

    The file may define any of:
      ``openrouter_api_key``, ``openai_api_key``, ``claude_api_key``
    Existing env vars are NOT overwritten.
    """
    import importlib.util

    candidates: List[Path] = []
    env_override = os.environ.get("COSPLAY_API_KEYS_FILE")
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend([
        SCRIPT_DIR / "api_keys.py",
        CODEBASE_ROOT / "api_keys.py",
        CODEBASE_ROOT.parent / "api_keys.py",
    ])

    for path in candidates:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        try:
            spec = importlib.util.spec_from_file_location("_cosplay_api_keys", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to load api_keys file {path}: {exc}", file=sys.stderr)
            continue

        mapping = {
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
            "claude_api_key": "ANTHROPIC_API_KEY",
        }
        for attr, env_name in mapping.items():
            val = getattr(mod, attr, None)
            if isinstance(val, str) and val.strip() and not os.environ.get(env_name):
                os.environ[env_name] = val.strip()
        return path
    return None


_API_KEYS_FILE_USED = _bootstrap_api_keys_from_file()


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

import openai

try:
    from common.models import BACKBONE_SFT_TEACHER_MODEL as _SFT_TEACHER_MODEL
except Exception:  # pragma: no cover
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_MODEL = _SFT_TEACHER_MODEL  # gpt-5.5

try:
    from API_func import OPENROUTER_BASE, make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    make_openai_client = None
    effective_openai_model = None

logger = logging.getLogger("cold_start.actor_osworld")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 10 OSWorld domains (matches the directory names under
# ``OSWorld/evaluation_examples/examples/``).
ALL_OSWORLD_DOMAINS: List[str] = [
    "chrome",
    "gimp",
    "libreoffice_calc",
    "libreoffice_impress",
    "libreoffice_writer",
    "multi_apps",
    "os",
    "thunderbird",
    "vlc",
    "vs_code",
]

# Default task catalog — ships with OSWorld; ~84 tasks across all 10 domains.
DEFAULT_TASK_CATALOG = "/workspace/OSWorld/evaluation_examples/test_small.json"

# How many outer steps per episode. The OSWorld-Verified protocol
# allows up to ~150 steps for the hardest tasks; 50 was the original
# default but it truncates a large fraction of the long-tail multi-app
# / multi-menu LibreOffice / GIMP / VLC tasks that need 30-60 steps to
# reach the evaluator. The May-2026 cold-start run had 46% of episodes
# truncate at step 50 with eval_score=None — i.e. nearly half of all
# pipeline cost was wasted on tasks where the agent ran out of budget
# before it could declare DONE. 75 is a balance: enough headroom for
# the mid-horizon tasks (file operations, format conversions, multi-
# dialog wizards) while keeping the worst-case wall-clock per task
# bounded. Bump higher (e.g. 100-150) only when running the full
# OSWorld-Verified leaderboard eval; the schema-from-vision call is
# the dominant cost so doubling steps roughly doubles per-task spend.
DEFAULT_MAX_STEPS = 75
# Episodes per task when ``--episodes`` is unset.
DEFAULT_EPISODES = 1
# Anti-noop: force a different action after this many consecutive steps
# whose state is identical AND no error fired.
_MAX_CONSECUTIVE_NOOPS = 2
# How many recent action results to surface in the action prompt.
_HISTORY_WINDOW = 5

# ─── Anti-loop early termination ──────────────────────────────────────────
# Reasoning models (gpt-5.x, o1/o3/o4) tend to over-verify on OSWorld:
# they keep clicking the same button / pressing escape instead of emitting
# DONE when the goal is satisfied. The loop detector watches a rolling
# window of recent actions and force-emits DONE when all of these hold:
#   (a) the same action string appears at least
#       ``DEFAULT_LOOP_REPEAT_THRESHOLD`` times within the last
#       ``DEFAULT_LOOP_WINDOW`` steps,
#   (b) total reward over that window is 0 (no progress),
#   (c) the trajectory has already advanced past
#       ``DEFAULT_LOOP_MIN_STEP`` (don't bail too early).
# Forcing DONE lets the OSWorld evaluator score the current state — even
# if it scores 0 we save 30+ wasted steps per loopy trajectory.
DEFAULT_LOOP_WINDOW = 5
DEFAULT_LOOP_REPEAT_THRESHOLD = 3
DEFAULT_LOOP_MIN_STEP = 8

# ─── DONE-nudge ───────────────────────────────────────────────────────────
# Once a trajectory has run for this many steps without DONE, the action
# prompt gets a hard reminder telling the model to emit DONE if the goal
# already appears satisfied. Reasoning models respond well to an explicit
# "stop verifying — commit to DONE" instruction; without it they keep
# inventing extra confirmation clicks.
#
# Calibration note (2026-05-01 cold-start run, max_steps=50): firing at
# step 12 caused 123/250 episodes to emit DONE-but-failed (49% of all
# episodes). The nudge interacts badly with the LLM-generated
# ``progress=0.X`` schema field — the same model writes a hallucinated
# "progress 0.9" and then the prompt asks it to commit. Pushing the
# threshold past the typical solve length (most successful episodes
# DONE between step 9 and 30; only the tail past 35 is genuine "stuck")
# eliminates the premature-DONE wave while keeping the loop-breaker.
DEFAULT_DONE_NUDGE_STEP = 35

# ─── Set-of-Marks (SoM) visual grounding ─────────────────────────────────
# SoM is the single biggest known lever for OSWorld pass-rate. The pipeline:
#   1. extract every interactive AT-SPI element with a bbox,
#   2. draw a numbered red box around each one on the screenshot,
#   3. add ``click_element(id=N)`` verbs to the candidate vocabulary,
#   4. on execute, translate ``click_element(N)`` → ``pyautogui.click(cx, cy)``.
# Published baselines move from ~5% (raw-pixel) → ~18% pass@1 with the
# same VLM backbone once SoM is enabled. The annotated screenshot is what
# both the schema-VLM and the action-VLM see; the saved frames also show
# the boxes so failure cases are easy to debug visually. Disable with
# ``--no_som`` if you want to A/B against the raw-pixel ablation.
DEFAULT_USE_SOM = True
DEFAULT_SOM_MAX_ELEMENTS = 25
# Default token budgets.
_ACTION_MAX_TOKENS = 500
_SCHEMA_MAX_TOKENS = 4000
# Reasoning models burn output tokens on hidden thinking — give them more.
_SCHEMA_MAX_TOKENS_REASONING = 12000
# Cap on entities per schema.
_DEFAULT_MAX_ENTITIES = 25
# Cap on candidate-action list passed to the actor LLM.
_MAX_CANDIDATE_ACTIONS = 22
# Default screen size (matches OSWorld smoke-test default).
_DEFAULT_SCREEN_W = 1280
_DEFAULT_SCREEN_H = 800

# OSWorld terminal tokens — episode ends when the agent emits any of these.
_TERMINAL_ACTIONS = ("DONE", "FAIL", "WAIT")

# Reasoning-model detection (gpt-5.x, o1/o3/o4).
_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:gpt-5(?:[\.\-]\w+)?|o[134](?:[\.\-]\w+)?)(?:$|[^\w])",
    re.IGNORECASE,
)


def _is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI-style reasoning models (gpt-5.x, o1/o3/o4)."""
    if not model:
        return False
    return bool(_REASONING_MODEL_RE.search(model))


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _safe_id(s: str, *, maxlen: int = 80) -> str:
    """Filesystem-safe slug; preserves alnum / '-' / '_' / '.'."""
    s = (s or "").strip() or "task"
    out: List[str] = []
    for ch in s:
        if ch.isalnum() or ch in "-._":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") [:maxlen] or "task"


# ---------------------------------------------------------------------------
# Image / observation helpers
# ---------------------------------------------------------------------------

def _to_pil(image: Any):
    """Coerce ``obs['screenshot']`` (np HxWx3 / PIL / bytes) into PIL RGB."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        try:
            from io import BytesIO
            return Image.open(BytesIO(image)).convert("RGB")
        except Exception:
            return None
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


def _pil_to_data_url(pil: Any) -> Optional[str]:
    """PIL image → data:image/png;base64,... URL, for VLM ``image_url`` parts.

    Used by the opt-in SelfVerifier (improvement #6); not used on
    every step (the schema-VLM caller has its own image-encoding
    path that lives in vlm_wrapper.schema). Returns None if the
    image cannot be encoded.
    """
    if pil is None:
        return None
    try:
        from io import BytesIO
        import base64
        buf = BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:  # noqa: BLE001
        logger.debug("[verify] _pil_to_data_url failed: %s", exc)
        return None


def _truncate(s: Optional[str], n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "...[truncated]"


# ---------------------------------------------------------------------------
# OpenAI client / model routing
# ---------------------------------------------------------------------------

def _build_client_and_route(
    *, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """Return ``(client, routed_model)`` or ``(None, model)`` on failure."""
    client = None
    if make_openai_client is not None:
        try:
            client = make_openai_client(api_key=api_key, base_url=base_url)
        except Exception:
            client = None

    if client is None:
        kw: Dict[str, Any] = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        try:
            client = openai.OpenAI(**kw) if kw else openai.OpenAI()
        except Exception:
            return None, model

    if api_key or base_url:
        return client, model
    if effective_openai_model is not None:
        try:
            return client, effective_openai_model(model)
        except Exception:
            return client, model
    return client, model


_VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high")


def _maybe_disable_thinking_kwargs(model: Any, tool_choice: Any) -> Dict[str, Any]:
    """Return ``extra_body`` kwargs to disable Qwen ``thinking`` mode.

    Qwen3/3.5/3.6 multimodal flagships (e.g. ``qwen/qwen3.5-plus-20260420``,
    ``qwen/qwen3.6-plus``) ship with thinking-mode ON.  When a strict
    ``tool_choice={"type":"function",...}`` payload is sent, the upstream
    DashScope endpoint rejects it with HTTP 400
    (``InvalidParameter ... in thinking mode``).  Our actor pipeline
    relies on strict tool-choice to force structured action JSON, so the
    only safe option is to turn thinking OFF.

    We forward both supported parameter names so the same payload works
    whether the endpoint is local-vLLM or DashScope/OpenRouter:

      - DashScope / OpenRouter:    ``extra_body.enable_thinking = False``
      - vLLM-OpenAI-compat:        ``extra_body.chat_template_kwargs
                                     .enable_thinking = False``

    Each server silently ignores the parameter it does not recognise, so
    the payload is portable.  No-ops for non-Qwen models and for calls
    that don't set ``tool_choice``.
    """
    if not isinstance(model, str):
        return {}
    if "qwen" not in model.lower():
        return {}
    if tool_choice is None:
        return {}
    return {
        "extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[list] = None,
    tool_choice: Any = None,
    reasoning_effort: Optional[str] = None,
):
    """Cross-model chat-completion wrapper (handles gpt-5.x / o1/o3/o4).

    ``reasoning_effort`` ∈ {minimal, low, medium, high} is forwarded only
    for reasoning models; ignored otherwise.  ``minimal`` suppresses
    hidden thinking tokens — recommended default for cold-start data
    generation since the SFT student only learns from the visible
    ``<state>`` and action JSON.
    """
    if _is_reasoning_model(model):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max(6000, max_tokens * 4),
        }
        # OpenAI hard-rejects ``reasoning_effort`` together with
        # ``tools`` on /v1/chat/completions for the gpt-5.x family
        # (returns HTTP 400: "Function tools with reasoning_effort
        # are not supported for gpt-5.4 in /v1/chat/completions.
        # Please use /v1/responses instead."). The /v1/responses
        # migration is a much bigger refactor; for now we silently
        # drop ``reasoning_effort`` whenever the call ships tools so
        # the action-LLM step does not 400-fail and degrade to the
        # candidate-list fallback. Schema-VLM calls (which are
        # tool-less) keep ``reasoning_effort`` and benefit from it.
        # Detection: any tool field set + an OpenAI gpt-5.x model
        # routed via the Chat Completions API. OpenRouter tunnels
        # the same model under ``openai/gpt-5.x`` and historically
        # accepts the parameter — only strip on direct OpenAI.
        is_direct_openai_gpt5 = (
            isinstance(model, str)
            and model.lower().startswith("gpt-5")
            and "/" not in model  # OpenRouter ids contain a "/"
        )
        tools_present = (tools is not None) or (tool_choice is not None)
        suppress_reasoning = (
            reasoning_effort is not None
            and is_direct_openai_gpt5
            and tools_present
        )
        if reasoning_effort and not suppress_reasoning:
            if reasoning_effort not in _VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort must be one of {_VALID_REASONING_EFFORTS}, "
                    f"got {reasoning_effort!r}"
                )
            kwargs["reasoning_effort"] = reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        kwargs.update(_maybe_disable_thinking_kwargs(model, tool_choice))
        return client.chat.completions.create(**kwargs)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    kwargs.update(_maybe_disable_thinking_kwargs(model, tool_choice))

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc)
        if not (
            "max_completion_tokens" in msg
            or ("max_tokens" in msg and "Unsupported" in msg)
            or ("temperature" in msg and "Unsupported" in msg)
        ):
            raise
        kwargs.pop("max_tokens", None)
        kwargs.pop("temperature", None)
        kwargs["max_completion_tokens"] = max(6000, max_tokens * 5)
        return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Lazy imports for optional deps
# ---------------------------------------------------------------------------

def _import_osworld_heuristic():
    """Heuristic schema head: AT-SPI XML → <state>."""
    from osworld_wrapper.heuristic import obs_to_schema as os_obs_to_schema
    return os_obs_to_schema


def _import_osworld_tools():
    """Tool registry for AT-SPI element queries (optional)."""
    try:
        from osworld_wrapper.tools import build_osworld_registry
        return build_osworld_registry
    except Exception as exc:
        logger.debug("osworld_wrapper.tools unavailable: %s", exc)
        return None


def _import_osworld_gym_wrapper():
    """Gymnasium-style wrapper around OSWorld's DesktopEnv."""
    from env_wrappers.osworld_wrapper import OSWorldGymWrapper, load_task_catalog
    return OSWorldGymWrapper, load_task_catalog


def _import_osworld_steering():
    """Lazy import of the opt-in steering helpers (improvements #3/#4/#6).

    Lives in a separate module so the OSWorld actor's main loop is
    unchanged when the corresponding flags are off. ``None`` here
    means the user did not enable any of the three subsystems — every
    call site treats ``None`` as "feature disabled, skip the hook".
    """
    try:
        from cold_start.osworld_steering import (
            MemorySummary,
            ReflexionTrigger,
            SelfVerifier,
        )
        return {
            "MemorySummary": MemorySummary,
            "ReflexionTrigger": ReflexionTrigger,
            "SelfVerifier": SelfVerifier,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[steering] cold_start.osworld_steering unavailable; "
            "advanced steering flags will be silently ignored: %s",
            exc,
        )
        return None


def _import_osworld_skill_retrieval():
    """Lazy import of the opt-in skill-bank retrieval helper (improvement #7)."""
    try:
        from cold_start.osworld_skill_retrieval import SkillBankRetriever
        return {"SkillBankRetriever": SkillBankRetriever}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[retrieval] cold_start.osworld_skill_retrieval unavailable; "
            "--skill_bank_path will be silently ignored: %s",
            exc,
        )
        return None


def _import_som_helpers():
    """Set-of-Marks helpers: AT-SPI XML → numbered overlay + verb table.

    Returns ``None`` on any import failure (Pillow missing, module
    not yet on PYTHONPATH, …) so the actor can fall back to the
    raw-pixel ablation rather than crashing.
    """
    try:
        from osworld_wrapper.som import (
            extract_som_elements,
            draw_som_overlay,
            format_som_table,
            som_action_strings,
            som_action_to_pyautogui,
        )
        return {
            "extract": extract_som_elements,
            "draw": draw_som_overlay,
            "format": format_som_table,
            "verbs": som_action_strings,
            "translate": som_action_to_pyautogui,
        }
    except Exception as exc:
        logger.warning(
            "[som] osworld_wrapper.som unavailable, SoM grounding disabled: %s",
            exc,
        )
        return None


def _import_schema_helpers():
    """``vlm_wrapper.schema`` — shared cross-domain prompt builder."""
    try:
        from vlm_wrapper.schema import (
            build_system_prompt,
            build_user_message,
            parse_schema_output,
        )
        return {
            "build_system_prompt": build_system_prompt,
            "build_user_message": build_user_message,
            "parse_schema_output": parse_schema_output,
        }
    except Exception as exc:
        logger.debug("vlm_wrapper.schema unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Stage 1 — visual schema generation (gpt-5.5 vision)
# ---------------------------------------------------------------------------

# Pre-compiled patterns for the lenient schema parser.
_LENIENT_STATE_OPEN_RE = re.compile(r"<state\b[^>]*>", re.IGNORECASE)
_LENIENT_STATE_CLOSE_RE = re.compile(r"</state\s*>", re.IGNORECASE)
_LENIENT_FENCE_RE = re.compile(
    r"^\s*```(?:xml|html|text|state)?\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LENIENT_SECTION_RE = re.compile(
    r"<(entities|attributes|affordances|relations|state_flags|targets|actions)\b",
    re.IGNORECASE,
)


def _lenient_parse_schema(raw: str, strict_parser) -> Tuple[Optional[str], str]:
    """Salvage a ``<state>...</state>`` schema from messy VLM output."""
    if not raw:
        return None, ""

    parsed = strict_parser(raw)
    if parsed:
        return parsed, "strict"

    cleaned = _LENIENT_FENCE_RE.sub("", raw).strip()
    if cleaned and cleaned != raw:
        parsed = strict_parser(cleaned)
        if parsed:
            return parsed, "fenced"

    candidate = cleaned or raw

    open_m = _LENIENT_STATE_OPEN_RE.search(candidate)
    close_m = _LENIENT_STATE_CLOSE_RE.search(candidate)
    if open_m and not close_m:
        salvaged = candidate[open_m.start():].rstrip() + "\n</state>"
        parsed = strict_parser(salvaged)
        if parsed:
            return parsed, "truncated"

    if not open_m and _LENIENT_SECTION_RE.search(candidate):
        salvaged = "<state>\n" + candidate.strip() + "\n</state>"
        parsed = strict_parser(salvaged)
        if parsed:
            return parsed, "untagged"

    return None, ""


def _flatten_a11y(xml_str: str, max_chars: int = 3500) -> str:
    """Truncate the namespaced AT-SPI XML so the action-LLM prompt stays small."""
    if not xml_str:
        return ""
    s = xml_str.strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n...[truncated]"
    return s


def generate_schema_from_image(
    *,
    pil_image,
    obs: Dict[str, Any],
    task_id: str,
    goal: str,
    step: int,
    candidate_actions: List[str],
    client: Any,
    routed_model: str,
    schema_helpers: Dict[str, Any],
    canonical_fallback: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = _SCHEMA_MAX_TOKENS,
    max_entities: int = _DEFAULT_MAX_ENTITIES,
    som_elements: Optional[List[Any]] = None,
    som_helpers: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call gpt-5.5 (vision) on the screenshot to produce a ``<state>`` schema.

    The screenshot is the **primary** input; the AT-SPI XML rides along
    as grounding context (so the VLM can produce real pixel coordinates
    matching accessible elements). Falls back to ``canonical_fallback``
    on failure or unparseable output.
    """
    if pil_image is None or schema_helpers is None or client is None:
        return {
            "schema": canonical_fallback,
            "raw": "",
            "source": "fallback_canonical" if canonical_fallback else "no_image_or_client",
            "error": None,
        }

    system = schema_helpers["build_system_prompt"]("desktop", max_entities=max_entities)
    if som_elements:
        system = (
            f"{system}\n\n"
            "The screenshot has numbered red boxes drawn over every "
            "interactive element (Set-of-Marks visual grounding). The "
            "id you populate in <actions> MUST be a SoM verb when the "
            "target is one of the numbered boxes: "
            "``click_element(id=N)``, "
            "``double_click_element(id=N)``, "
            "``right_click_element(id=N)``, or "
            "``type_into_element(id=N, text='...')``. The harness "
            "translates N to the box centre. Only emit raw "
            "``pyautogui.click(x, y)`` when NO numbered box covers the "
            "target. Hotkeys (``pyautogui.hotkey('ctrl', 's')`` etc.) "
            "and ``DONE`` / ``FAIL`` / ``WAIT`` are also valid."
        )
    else:
        system = (
            f"{system}\n\n"
            "Valid OSWorld actions are pyautogui commands "
            "(pyautogui.click(x, y), pyautogui.doubleClick(x, y), "
            "pyautogui.typewrite('text'), pyautogui.hotkey('ctrl', 's'), "
            "pyautogui.scroll(-3), pyautogui.press('enter')) plus the "
            "special tokens DONE / FAIL / WAIT. Use absolute pixel "
            "coordinates for click / move targets — the screenshot is "
            "the source of truth."
        )

    extra_parts: List[str] = []
    instr = obs.get("instruction") or ""
    if instr:
        extra_parts.append(f"Task instruction: {instr}")
    last_action = obs.get("last_action") or ""
    if last_action:
        extra_parts.append(f"Last action: {last_action}")
    last_err = obs.get("last_action_error") or ""
    if last_err:
        extra_parts.append(f"Last action error: {_truncate(last_err, 200)}")
    a11y_text = _flatten_a11y(obs.get("accessibility_tree", "") or "")
    if a11y_text:
        extra_parts.append(
            "Accessibility tree (for element grounding, truncated):\n"
            f"{a11y_text}"
        )
    terminal = obs.get("terminal") or ""
    if terminal:
        tail = "\n".join(terminal.strip().splitlines()[-12:])
        if tail:
            extra_parts.append("Terminal tail:\n" + tail)
    if candidate_actions:
        extra_parts.append(
            "Candidate actions (you MUST copy one verbatim into <actions>; "
            "do NOT rename or reformat):\n"
            + "\n".join(f"  - {a}" for a in candidate_actions[:_MAX_CANDIDATE_ACTIONS])
        )
    # Set-of-Marks element table — same numbering as the red boxes drawn
    # on ``pil_image``. Surfacing it as text lets the VLM pick an ID
    # even when the red box would otherwise overlap a busy region.
    if som_elements and som_helpers is not None:
        try:
            som_table = som_helpers["format"](som_elements)
        except Exception as exc:
            logger.debug("som_helpers.format failed: %s", exc)
            som_table = None
        if som_table:
            extra_parts.append(som_table)
    extra_context = "\n\n".join(extra_parts)

    user_content = schema_helpers["build_user_message"](
        pil_image,
        domain="desktop",
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
    parsed: Optional[str] = None
    err: Optional[str] = None
    finish_reason: Optional[str] = None
    recovery: str = ""
    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        if resp.choices:
            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
        parsed, recovery = _lenient_parse_schema(
            raw, schema_helpers["parse_schema_output"]
        )
    except Exception as exc:
        err = repr(exc)
        logger.warning("[schema-VLM] step %d failed: %s", step, exc)

    base_meta: Dict[str, Any] = {
        "raw": raw,
        "raw_full_len": len(raw),
        "finish_reason": finish_reason,
        "recovery": recovery,
        "error": err,
    }

    if parsed:
        if recovery and recovery != "strict":
            logger.info(
                "[schema-VLM] step %d salvaged via '%s' (finish_reason=%s, raw_len=%d)",
                step, recovery, finish_reason, len(raw),
            )
        return {"schema": parsed, "source": "vlm", **base_meta}

    if raw and finish_reason == "length":
        logger.warning(
            "[schema-VLM] step %d response truncated (finish_reason=length, "
            "raw_len=%d) — consider raising _SCHEMA_MAX_TOKENS",
            step, len(raw),
        )

    return {
        "schema": canonical_fallback,
        "source": "fallback_canonical" if canonical_fallback else "vlm_no_schema",
        **base_meta,
    }


# ---------------------------------------------------------------------------
# Action vocabulary construction
# ---------------------------------------------------------------------------

# Standard navigation / system actions OSWorld always accepts.
_GLOBAL_DESKTOP_ACTIONS: List[str] = [
    "pyautogui.scroll(-3)",
    "pyautogui.scroll(3)",
    "pyautogui.press('enter')",
    "pyautogui.press('escape')",
    "pyautogui.hotkey('ctrl', 's')",
    "pyautogui.hotkey('ctrl', 'c')",
    "pyautogui.hotkey('ctrl', 'v')",
    "pyautogui.hotkey('alt', 'tab')",
    "WAIT",
    "DONE",
    "FAIL",
]

# Lenient validator for pyautogui-style strings, special tokens, and
# Set-of-Marks verbs (``click_element(id=N)`` etc.). Multi-statement
# pyautogui sequences separated by ``;`` are accepted because the SoM
# translator emits ``click + typewrite`` as a single action.
_PYAUTOGUI_ACTION_RE = re.compile(
    r"^\s*(?:"
    r"DONE|FAIL|WAIT"
    r"|pyautogui\.[A-Za-z_]+\([^\n]*\)(?:\s*;\s*pyautogui\.[A-Za-z_]+\([^\n]*\))*"
    r"|(?:click|double_?click|right_?click|type_?(?:text_?)?(?:into_?)?)_?element"
    r"\([^\n)]*\)"
    r")\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _list_a11y_clickables(
    a11y_tree_xml: str, instruction: str, *,
    max_results: int = _MAX_CANDIDATE_ACTIONS,
) -> List[Dict[str, Any]]:
    """Walk the AT-SPI XML and emit per-element click suggestions.

    Returns a list of dicts with ``action`` / ``role`` / ``name`` /
    ``pos`` so the actor LLM has concrete targets to choose from. We
    rely on the same NS / state / component helpers used by
    :func:`osworld_wrapper.heuristic.obs_to_schema` so namespace handling
    matches across all three OS flavours.
    """
    if not a11y_tree_xml or not a11y_tree_xml.strip():
        return []

    try:
        from osworld_wrapper.heuristic import (
            INTERACTIVE_ROLES, _bbox_from, _states_from, _strip_ns,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("a11y helpers import failed: %s", exc)
        return []

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(a11y_tree_xml)
    except Exception:
        return []

    instr_lc = (instruction or "").lower()
    out: List[Tuple[float, Dict[str, Any]]] = []

    for el in root.iter():
        role = _strip_ns(el.tag)
        if role not in INTERACTIVE_ROLES:
            continue
        states = _states_from(el)
        if "visible" not in states or "showing" not in states:
            continue

        bbox = _bbox_from(el)
        if not bbox:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        cx, cy = x + w // 2, y + h // 2

        name = (el.get("name") or "").strip()
        label = name or role

        score = 0.0
        if name and instr_lc:
            for tok in re.findall(r"[a-zA-Z]{3,}", name.lower()):
                if tok in instr_lc:
                    score += 1.0
        # Nudges: focused / armed / pressed elements are more relevant.
        if "focused" in states:
            score += 0.5
        if "armed" in states:
            score += 0.3

        action = f"pyautogui.click({cx}, {cy})"
        out.append((score, {
            "action": action,
            "role": role,
            "name": _truncate(label, 80),
            "pos": (cx, cy),
            "bbox": (x, y, w, h),
            "states": states,
        }))

    # Stable: by score desc, then by reading order (top-to-bottom, left-to-right).
    out.sort(key=lambda kv: (-kv[0], kv[1]["pos"][1], kv[1]["pos"][0]))
    seen: set = set()
    suggestions: List[Dict[str, Any]] = []
    for _, m in out:
        key = m["action"]
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(m)
        if len(suggestions) >= max_results:
            break
    return suggestions


def _build_candidate_actions(
    *, obs: Dict[str, Any],
    som_elements: Optional[List[Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build candidate actions for this step.

    Combines (priority order):
      - Set-of-Marks click verbs (``click_element(id=N)``) — preferred
        when ``som_elements`` is non-empty, since the VLM picks IDs
        more accurately than coordinates.
      - Top accessible interactive elements (``pyautogui.click(x,y)``
        derived from AT-SPI ``cp:screencoord`` + ``cp:size``) — kept
        as a fallback so the model can still issue raw-coordinate
        clicks when no SoM box covers what it needs.
      - A small global vocab of nav / hotkey / DONE / FAIL / WAIT actions.

    Returns ``(strings, meta)`` — meta carries role / name / pos for the
    actor prompt.
    """
    a11y = obs.get("accessibility_tree", "") or ""
    instruction = obs.get("instruction", "") or ""

    strings: List[str] = []
    meta: List[Dict[str, Any]] = []
    seen: set = set()

    # SoM verbs first — most reliable grounding once boxes are drawn.
    if som_elements:
        for el in som_elements[:12]:
            verb = f"click_element(id={el.som_id})"
            if verb in seen:
                continue
            seen.add(verb)
            strings.append(verb)
            meta.append({
                "action": verb,
                "role": el.role,
                "name": el.label,
                "pos": (el.bbox[0], el.bbox[1], el.bbox[2], el.bbox[3]),
                "som_id": el.som_id,
            })
        # One typing template — VLM substitutes the id and text.
        # Only advertised if at least one typable element is on screen
        # (entry / text / combo-box / spin-button).
        from osworld_wrapper.som import _is_typable
        if any(_is_typable(e.role) for e in som_elements):
            verb = "type_into_element(id=N, text='...')"
            if verb not in seen:
                seen.add(verb)
                strings.append(verb)
                meta.append({
                    "action": verb,
                    "role": "som-template",
                    "name": "fill text into element id N",
                })

    # When SoM is active and offers ≥ 4 numbered targets, the raw
    # ``pyautogui.click(x, y)`` candidates derived from AT-SPI are
    # redundant (every box already has a click_element verb pointing at
    # its centre) and they actively pull the model toward emitting raw
    # coords instead of SoM IDs. Empirically, gpt-5.x picks raw
    # candidates ~85% of the time when both forms are offered, which
    # nullifies the SoM uplift. Drop the raw clicks in that regime; the
    # global hotkey/escape/scroll vocabulary still survives below.
    if not (som_elements and len(som_elements) >= 4):
        suggestions = _list_a11y_clickables(a11y, instruction, max_results=10)
        for entry in suggestions:
            a = entry["action"]
            if a in seen:
                continue
            seen.add(a)
            strings.append(a)
            meta.append({
                "action": a,
                "role": entry.get("role"),
                "name": entry.get("name"),
                "pos": entry.get("pos"),
            })

    for a in _GLOBAL_DESKTOP_ACTIONS:
        if a in seen:
            continue
        seen.add(a)
        strings.append(a)
        meta.append({"action": a, "role": "system", "name": None})

    return strings[:_MAX_CANDIDATE_ACTIONS], meta[:_MAX_CANDIDATE_ACTIONS]


# ---------------------------------------------------------------------------
# Stage 2 — schema-driven action selection (gpt-5.5)
# ---------------------------------------------------------------------------

_ACTOR_SYSTEM_PROMPT = (
    "You are an Actor Agent for the COS-PLAY desktop-agent pipeline, "
    "driving a real OSWorld DesktopEnv (Ubuntu / Windows / macOS guest "
    "VM with real pyautogui actuation).\n"
    "On every step you receive a structured ``<state>...</state>`` schema "
    "describing the visual state of the desktop (entities have pixel "
    "bounding boxes), plus a list of candidate actions that combines "
    "a11y-derived click targets with global hotkeys + the special "
    "tokens DONE / FAIL / WAIT.\n\n"
    "EVERY ``choose_action`` call MUST include a short ``subgoal`` "
    "string naming the immediate intent in 5-10 words "
    "(e.g. 'open File menu', 'locate Export PDF entry', "
    "'type query into address bar', 'confirm save dialog'). The "
    "subgoal is the unit of *plan decomposition* — it should change "
    "as you progress through the task. Two consecutive steps with "
    "the *same* subgoal mean the previous step did not advance you, "
    "so the second step should try a different action. The harness "
    "uses the subgoal sequence as the segmentation anchor when it "
    "lifts your trajectory into reusable Skills, so be precise.\n\n"
    "Set-of-Marks grounding (preferred when a numbered box covers the "
    "target): when the screenshot has numbered red bounding boxes "
    "drawn over interactive elements and the user prompt lists those "
    "IDs, prefer ``click_element(id=N)`` or "
    "``type_into_element(id=N, text='...')`` — the harness translates "
    "N to the element's bbox centre at execute time. SoM IS NOT "
    "EXHAUSTIVE: it only labels the AT-SPI elements the heuristic "
    "could enumerate. Many real targets (deeply-nested menu entries, "
    "text inside a document body, image regions, file-manager files) "
    "have no SoM ID. When the target you need has no numbered box, "
    "DO NOT give up — issue a raw ``pyautogui.click(x, y)`` using the "
    "bbox centre from the schema's ``<entities>`` block, or a "
    "``pyautogui.hotkey(...)`` that achieves the same effect (e.g. "
    "``Ctrl+Shift+E`` for LibreOffice 'Export as PDF', ``Alt+F`` to "
    "open the File menu).\n\n"
    "Your job:\n"
    "1. Reason briefly (≤3 sentences) about the schema: which entity / "
    "control matters, what is the current sub-goal, and why one action "
    "best advances the user's instruction.\n"
    "2. Pick EXACTLY ONE action by calling the ``choose_action`` "
    "function. Order of preference:\n"
    "   (a) a SoM ``click_element(id=N)`` candidate that matches the "
    "       intended target;\n"
    "   (b) a raw ``pyautogui.click(x, y)`` at the schema bbox centre, "
    "       or a ``pyautogui.hotkey(...)`` keyboard equivalent — both "
    "       are valid even when not in the candidate list;\n"
    "   (c) ``WAIT`` to let an animation / load finish.\n"
    "3. Only emit ``DONE`` when the user's instruction is FULLY and "
    "OBJECTIVELY satisfied — what the user can verify on-screen, NOT "
    "what the schema's ``progress=`` field claims (that field is "
    "generated by the same model and is not a reliable signal). "
    "``FAIL`` is a LAST RESORT — only use it after you have exhausted "
    "menu navigation, hotkeys, AND raw-coordinate clicks. Emitting "
    "FAIL because 'no SoM id matches' is incorrect; the SoM is "
    "advisory, not exhaustive.\n\n"
    "If recent action history shows an action had NO EFFECT (state "
    "unchanged AND no error), choose a DIFFERENT action this turn — "
    "try a NEARBY numbered box, a different ID, a keyboard shortcut, "
    "or a raw click at a NEW (x, y) before giving up.\n\n"
    "Always respond by calling the ``choose_action`` function."
)


def _build_action_tools(candidate_actions: List[str]) -> list:
    """OpenAI function-calling tool definition for desktop action selection."""
    enum_types = [
        "click", "double_click", "right_click", "move_to",
        "drag", "type", "hotkey", "press",
        "scroll_down", "scroll_up", "wait", "done", "fail",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": (
                    "Choose a single OSWorld pyautogui action (or DONE / FAIL / "
                    "WAIT) for this turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subgoal": {
                            "type": "string",
                            "description": (
                                "REQUIRED. 5-10 word naming of the "
                                "immediate intent driving this step "
                                "(e.g. 'open File menu', "
                                "'locate Export PDF entry', "
                                "'type query into address bar'). "
                                "Used as the segmentation anchor when "
                                "the harness lifts the trajectory into "
                                "reusable Skills. Two consecutive steps "
                                "with the same subgoal mean the prior "
                                "step did not advance progress — switch "
                                "to a different action this turn."
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Brief chain-of-thought (≤3 sentences) "
                                "grounded in the schema entities."
                            ),
                        },
                        "action_string": {
                            "type": "string",
                            "description": (
                                "Verbatim pyautogui action string (preferred). "
                                "Examples: 'pyautogui.click(820, 412)', "
                                "'pyautogui.typewrite(\"hello\", interval=0.05)', "
                                "'pyautogui.hotkey(\"ctrl\", \"s\")', "
                                "'DONE', 'FAIL', 'WAIT'. Prefer a SoM "
                                "candidate when one matches; otherwise a "
                                "raw 'pyautogui.click(x, y)' / 'pyautogui."
                                "hotkey(...)' is also valid. Do NOT emit "
                                "FAIL just because no SoM id matches — "
                                "fall back to raw coordinates from the "
                                "schema's <entities> bboxes. Candidate "
                                "list (advisory): "
                                + ", ".join(candidate_actions[:_MAX_CANDIDATE_ACTIONS])
                            ),
                        },
                        "action_type": {
                            "type": "string",
                            "enum": enum_types,
                            "description": (
                                "Structured action type. Used when "
                                "action_string is missing or invalid."
                            ),
                        },
                        "x": {
                            "type": "integer",
                            "description": (
                                "Absolute screen X pixel for click / "
                                "double_click / right_click / move_to / drag."
                            ),
                        },
                        "y": {
                            "type": "integer",
                            "description": (
                                "Absolute screen Y pixel for click / "
                                "double_click / right_click / move_to / drag."
                            ),
                        },
                        "dx": {
                            "type": "integer",
                            "description": "Horizontal pixel delta for drag.",
                        },
                        "dy": {
                            "type": "integer",
                            "description": "Vertical pixel delta for drag.",
                        },
                        "text": {
                            "type": "string",
                            "description": (
                                "Text payload for type (typewrite). "
                                "Plain ASCII; avoid newlines unless typing "
                                "them deliberately."
                            ),
                        },
                        "key": {
                            "type": "string",
                            "description": (
                                "Single key for press, e.g. 'enter', "
                                "'escape', 'tab', 'space', 'down'."
                            ),
                        },
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Hotkey combo for hotkey, e.g. "
                                "['ctrl','s'] or ['alt','tab']."
                            ),
                        },
                        "scroll_clicks": {
                            "type": "integer",
                            "description": (
                                "Number of mouse-wheel clicks for "
                                "scroll_down / scroll_up. Positive."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }
    ]


def _format_history_block(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Recent action history (newest last):"]
    for entry in history[-_HISTORY_WINDOW:]:
        effect = (
            "ERROR" if entry.get("error")
            else ("NO EFFECT" if entry.get("noop") else f"reward {entry.get('reward', 0.0):+.2f}")
        )
        lines.append(
            f"  - {entry.get('action')!r} -> {effect}"
            + (f" (err: {entry.get('error_text','')[:80]})" if entry.get("error") else "")
        )
    noop_actions = sorted({e["action"] for e in history[-_HISTORY_WINDOW:] if e.get("noop")})
    if noop_actions:
        lines.append(
            f"WARNING: Action(s) {noop_actions} had no effect. Pick a DIFFERENT action."
        )
    return "\n".join(lines) + "\n"


def _structured_to_action_string(args: Dict[str, Any]) -> Optional[str]:
    """Build a pyautogui action string from structured function-call args."""
    atype = (args.get("action_type") or "").lower().strip()
    x = args.get("x")
    y = args.get("y")
    dx = args.get("dx")
    dy = args.get("dy")
    text = args.get("text", "")
    key = (args.get("key") or "").strip()
    keys = args.get("keys") or []
    clicks = args.get("scroll_clicks")

    def _esc(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace("'", "\\'")

    if atype in ("click", "left_click") and isinstance(x, int) and isinstance(y, int):
        return f"pyautogui.click({x}, {y})"
    if atype == "double_click" and isinstance(x, int) and isinstance(y, int):
        return f"pyautogui.doubleClick({x}, {y})"
    if atype == "right_click" and isinstance(x, int) and isinstance(y, int):
        return f"pyautogui.rightClick({x}, {y})"
    if atype == "move_to" and isinstance(x, int) and isinstance(y, int):
        return f"pyautogui.moveTo({x}, {y})"
    if atype == "drag" and isinstance(x, int) and isinstance(y, int):
        ddx = int(dx) if isinstance(dx, int) else 0
        ddy = int(dy) if isinstance(dy, int) else 0
        return (
            f"pyautogui.moveTo({x}, {y}); "
            f"pyautogui.dragRel({ddx}, {ddy}, duration=0.5)"
        )
    if atype == "type" and text:
        return f"pyautogui.typewrite('{_esc(str(text))}', interval=0.03)"
    if atype == "press" and key:
        return f"pyautogui.press('{_esc(key)}')"
    if atype == "hotkey" and isinstance(keys, list) and keys:
        kparts = ", ".join(f"'{_esc(str(k))}'" for k in keys if k)
        if kparts:
            return f"pyautogui.hotkey({kparts})"
    if atype in ("scroll_down", "scroll_up"):
        n = int(clicks) if isinstance(clicks, int) else 3
        n = max(1, min(20, n))
        sign = -n if atype == "scroll_down" else n
        return f"pyautogui.scroll({sign})"
    if atype == "wait":
        return "WAIT"
    if atype == "done":
        return "DONE"
    if atype == "fail":
        return "FAIL"
    return None


def _validate_action_string(action: str) -> bool:
    """Best-effort sanity check that ``action`` is a valid OSWorld action."""
    if not action:
        return False
    return bool(_PYAUTOGUI_ACTION_RE.match(action.strip()))


def select_action(
    *,
    schema_text: Optional[str],
    obs: Dict[str, Any],
    candidate_actions: List[str],
    candidate_meta: List[Dict[str, Any]],
    task: str,
    step: int,
    history: List[Dict[str, Any]],
    client: Any,
    routed_model: str,
    temperature: float = 0.4,
    max_tokens: int = _ACTION_MAX_TOKENS,
    done_nudge_step: int = DEFAULT_DONE_NUDGE_STEP,
    som_elements: Optional[List[Any]] = None,
    som_helpers: Optional[Dict[str, Any]] = None,
    last_action_was_noop: bool = False,
    last_failed_action: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    prior_subgoals: Optional[List[str]] = None,
    memory_block: Optional[str] = None,
    reflection_block: Optional[str] = None,
    retrieved_skills_block: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, Optional[str], Optional[str]]:
    """Call gpt-5.5 with the schema → ``(action, reasoning, raw, error, subgoal)``.

    The four optional ``*_block`` args are opt-in steering text that
    the caller (``run_actor_episode``) builds when the corresponding
    feature flags are on. Each is rendered as a clearly-tagged
    section in the user prompt; passing ``None`` (the default) means
    the section is omitted and the prompt matches the pre-feature
    template byte-for-byte.
    """
    if not candidate_actions:
        candidate_actions = list(_GLOBAL_DESKTOP_ACTIONS)
        candidate_meta = [
            {"action": a, "role": "system", "name": None}
            for a in _GLOBAL_DESKTOP_ACTIONS
        ]
    if client is None:
        return None, None, "", "no_client", None

    history_block = _format_history_block(history)
    schema_block = (
        schema_text.strip() if schema_text else
        "(no schema available — fall back to the AT-SPI tree text below)"
    )

    candidate_lines = []
    for i, m in enumerate(candidate_meta[:_MAX_CANDIDATE_ACTIONS], start=1):
        suffix = ""
        if m.get("role") and m.get("name"):
            suffix = f"  # {m['role']}: {str(m['name'])[:60]}"
        elif m.get("role"):
            suffix = f"  # {m['role']}"
        candidate_lines.append(f"  {i}. {m['action']}{suffix}")
    candidate_block = "Candidate actions:\n" + "\n".join(candidate_lines)

    user_parts = [
        f"Task: {task}",
        f"Step: {step}",
        "",
        "Structured state schema:",
        schema_block,
        "",
        candidate_block,
    ]

    # ─── Opt-in retrieved skills (improvement #7) ────────────────────────
    # Rendered just below the candidate-action list so the actor sees
    # the demonstrations in the same scope where it picks the action.
    if retrieved_skills_block:
        user_parts.extend([
            "",
            "=== RETRIEVED SKILLS (in-context demos) ===",
            retrieved_skills_block,
            "Use the retrieved protocols as REFERENCES for action "
            "shape and ordering, not as literal copy-paste. The "
            "current task's UI may have moved/renamed elements.",
        ])

    # ─── Opt-in memory summary (improvement #3) ──────────────────────────
    if memory_block:
        user_parts.extend([
            "",
            "=== MEMORY (running summary of recent steps) ===",
            memory_block,
        ])

    # ─── Opt-in reflection on consecutive no-ops (improvement #4) ───────
    if reflection_block:
        user_parts.extend([
            "",
            "=== REFLECTION (last action streak failed) ===",
            reflection_block,
            "Pick ONE of the proposed alternatives this turn — do "
            "NOT re-issue the action that just failed.",
        ])

    # ─── Set-of-Marks element table ──────────────────────────────────────
    # When SoM is active the screenshot the model already saw at the
    # schema step has numbered red boxes around every clickable; mirror
    # that same numbering as a text table here so the model can pick by
    # ID without re-deriving coordinates from pixels.
    if som_elements and som_helpers is not None:
        try:
            som_table = som_helpers["format"](som_elements)
        except Exception as exc:
            logger.debug("som_helpers.format failed: %s", exc)
            som_table = None
        if som_table:
            user_parts.extend([
                "",
                som_table,
                "",
                "=== ACTION FORMAT (mandatory when SoM is active) ===",
                "Click any UI element by emitting EXACTLY "
                "``click_element(id=N)`` where N is the badge number "
                "from the screenshot / table above. To type into a "
                "[typable] element use "
                "``type_into_element(id=N, text='...')``. "
                "DO NOT emit raw ``pyautogui.click(x, y)`` when a "
                "numbered box covers the target — VLMs hallucinate "
                "coordinates, the box centre is the ground truth. "
                "If NO numbered box covers what you need, fall back "
                "to a hotkey from the candidate list (escape / "
                "scroll / ctrl+s / etc.) BEFORE resorting to raw "
                "coordinates.",
            ])

    if not schema_text:
        a11y_text = _flatten_a11y(obs.get("accessibility_tree", "") or "", max_chars=2000)
        if a11y_text:
            user_parts.extend([
                "",
                "Accessibility tree (since no schema was parsed):",
                a11y_text,
            ])

    # ─── Anti-noop replan ────────────────────────────────────────────────
    # If the last action did NOT change the screen (and produced no
    # error), repeating it is almost certainly a waste. Tell the model
    # that explicitly so it picks a different element / strategy.
    if last_action_was_noop and last_failed_action:
        user_parts.extend([
            "",
            "=== LAST ACTION HAD NO EFFECT ===",
            f"Previous action ``{last_failed_action}`` did not change the "
            f"screen and produced no error. The element is probably "
            f"covered by an overlay, off-screen, or the wrong target. "
            f"Choose a DIFFERENT action this turn — try a nearby "
            f"numbered box, a hotkey, scrolling, or pressing escape "
            f"first. Do NOT re-issue the same coordinates.",
        ])

    user_parts.extend([
        "",
        history_block.strip(),
        "",
        "Pick the BEST action and call ``choose_action``. When SoM "
        "boxes are present, copy a ``click_element(id=N)`` candidate "
        "verbatim into ``action_string``. Otherwise prefer "
        "pyautogui.click(x, y) using bbox centres from the schema, or "
        "specify ``action_type`` + the relevant structured fields. "
        "Always include a ``subgoal`` (5-10 words) describing the "
        "immediate intent.",
    ])

    # ─── Subgoal trail ──────────────────────────────────────────────────
    # Show the last few subgoals back to the model so it knows what
    # plan-level steps have been declared. If the same subgoal has
    # appeared 2+ times recently it's a strong signal the previous
    # action did not advance — the prompt nudges the model to either
    # change the action OR change the subgoal (re-decompose).
    if prior_subgoals:
        recent_sg = [s for s in prior_subgoals[-5:] if s]
        if recent_sg:
            user_parts.extend([
                "",
                "Recent subgoals (newest last): "
                + " → ".join(s[:60] for s in recent_sg),
            ])
            tail = recent_sg[-3:] if len(recent_sg) >= 3 else recent_sg
            if len(tail) >= 2 and len(set(tail)) == 1:
                user_parts.append(
                    "WARNING: the last "
                    f"{len(tail)} subgoals are identical — either the "
                    "previous attempts did not advance progress (so "
                    "pick a MATERIALLY different action this turn) or "
                    "the subgoal is too coarse (so refine it into a "
                    "more specific sub-step before acting)."
                )

    # ─── DONE-nudge ───────────────────────────────────────────────────────
    # Long-running stuck trajectories sometimes need a reminder, but the
    # earlier "trust the LLM-generated progress field" wording caused
    # 49% of the May-2026 run's episodes to emit a premature DONE.
    # We now only fire the nudge late in the trajectory AND only when
    # there is an objective signal that the agent is stuck (action
    # repetition with reward=0). The text deliberately avoids citing
    # the schema's ``progress=`` field — that value is hallucinated by
    # the same VLM that's about to be asked to commit, so it's a
    # circular signal.
    if step >= done_nudge_step:
        repeats = 0
        if history:
            recent = history[-DEFAULT_LOOP_WINDOW:]
            counts: Dict[str, int] = {}
            for h in recent:
                a = (h.get("action") or "").strip()
                if a:
                    counts[a] = counts.get(a, 0) + 1
            repeats = max(counts.values()) if counts else 0
        if repeats >= 2:
            user_parts.extend([
                "",
                f"=== LOOP-BREAKER (step {step}) ===",
                f"You have repeated the same action {repeats} times in the "
                f"last {DEFAULT_LOOP_WINDOW} steps with reward=0. Pick a "
                "MATERIALLY different action (different element, different "
                "menu path, different keyboard shortcut). If you have "
                "exhausted reasonable options for this goal, emit FAIL. "
                "Only emit DONE if the on-screen result objectively "
                "matches what the goal asked for — do NOT use the schema's "
                "``progress=`` field as the trigger; it is generated by "
                "the same model writing this turn's action and is not a "
                "reliable verification signal.",
            ])
    user_content = "\n".join(p for p in user_parts if p is not None)

    tools = _build_action_tools(candidate_actions)

    raw = ""
    err: Optional[str] = None
    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=[
                {"role": "system", "content": _ACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "choose_action"}},
            reasoning_effort=reasoning_effort,
        )
        choice = resp.choices[0]
        msg = choice.message
        raw = msg.content or ""

        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            raw_args = (
                getattr(tc, "arguments", None)
                or getattr(getattr(tc, "function", None), "arguments", None)
                or "{}"
            )
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            reasoning = args.get("reasoning") or None
            subgoal_raw = (args.get("subgoal") or "").strip()
            subgoal = subgoal_raw[:120] if subgoal_raw else None
            action_string = (args.get("action_string") or "").strip()
            if action_string and _validate_action_string(action_string):
                return action_string, reasoning, raw or json.dumps(args), None, subgoal
            structured = _structured_to_action_string(args)
            if structured and _validate_action_string(structured):
                return structured, reasoning, raw or json.dumps(args), None, subgoal
            cand = action_string.strip()
            m = re.match(r"^\s*(\d+)\s*\.?\s*$", cand)
            if m and 1 <= int(m.group(1)) <= len(candidate_actions):
                return (
                    candidate_actions[int(m.group(1)) - 1], reasoning,
                    raw or json.dumps(args), None, subgoal,
                )

        for cand in candidate_actions:
            if cand in raw:
                return cand, None, raw, None, None

    except Exception as exc:
        err = repr(exc)
        logger.warning("[action-LLM] step %d failed: %s", step, exc)

    return None, None, raw, err, None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _is_noop(prev_obs: Dict[str, Any], next_obs: Dict[str, Any]) -> bool:
    """Best-effort no-op detection.

    AT-SPI XML length unchanged AND no last_action_error AND screenshot
    arrays equal ⇒ no-op.
    """
    err = (next_obs.get("last_action_error") or "").strip()
    if err:
        return False

    prev_a11y = prev_obs.get("accessibility_tree") or ""
    next_a11y = next_obs.get("accessibility_tree") or ""
    if len(prev_a11y) != len(next_a11y):
        return False

    prev_img = prev_obs.get("screenshot")
    next_img = next_obs.get("screenshot")
    if isinstance(prev_img, np.ndarray) and isinstance(next_img, np.ndarray):
        if prev_img.shape != next_img.shape:
            return False
        try:
            return bool(np.array_equal(prev_img, next_img))
        except Exception:
            return False

    return prev_a11y == next_a11y


def _pick_different(action: str, candidates: List[str]) -> str:
    alts = [a for a in candidates if a != action]
    return random.choice(alts) if alts else action


def _detect_action_loop(
    history: List[Dict[str, Any]],
    *,
    window: int = DEFAULT_LOOP_WINDOW,
    repeat_threshold: int = DEFAULT_LOOP_REPEAT_THRESHOLD,
) -> Tuple[bool, Optional[str], int]:
    """Detect whether the agent is stuck in a non-progressing action loop.

    Returns ``(is_looping, repeated_action, repeat_count)``. The loop is
    flagged when, within the last ``window`` steps:
      - the same action string appears ``repeat_threshold`` or more times,
      - AND the cumulative reward over that window is 0 (no progress).

    The window-size guard means the detector is a no-op until at least
    ``repeat_threshold`` steps have been recorded; the calling site
    additionally gates on ``DEFAULT_LOOP_MIN_STEP`` to avoid premature
    aborts on tasks that legitimately take a few clicks to wire up.
    """
    if not history or len(history) < repeat_threshold:
        return False, None, 0
    recent = history[-window:]
    window_reward = sum(float(h.get("reward", 0.0) or 0.0) for h in recent)
    if window_reward > 0.0:
        return False, None, 0
    counts: Dict[str, int] = {}
    for h in recent:
        a = (h.get("action") or "").strip()
        if not a:
            continue
        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return False, None, 0
    top_action, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count >= repeat_threshold:
        return True, top_action, top_count
    return False, None, 0


def _resolve_task_domain(task_cfg: Dict[str, Any], default: str = "unknown") -> str:
    """Try to recover the OSWorld domain (chrome / gimp / …) for a task.

    OSWorld's test_*.json buckets tasks by domain; the resolved task
    config doesn't always carry that label, but the catalog loader
    sometimes injects ``domain`` and the task ``snapshot`` field is a
    reliable per-domain marker on the OSWorld side.
    """
    if not isinstance(task_cfg, dict):
        return default
    for k in ("domain", "snapshot", "category"):
        v = task_cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def run_actor_episode(
    *,
    env: Any,
    task_cfg: Dict[str, Any],
    domain: str,
    safe_task_id: str,
    max_steps: int,
    client: Any,
    routed_model: str,
    fallback_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    use_vision: bool,
    temperature_action: float,
    temperature_schema: float,
    max_entities: int,
    frames_dir: Optional[Path],
    seed: Optional[int],
    verbose: bool,
    loop_window: int = DEFAULT_LOOP_WINDOW,
    loop_repeat_threshold: int = DEFAULT_LOOP_REPEAT_THRESHOLD,
    loop_min_step: int = DEFAULT_LOOP_MIN_STEP,
    done_nudge_step: int = DEFAULT_DONE_NUDGE_STEP,
    use_som: bool = DEFAULT_USE_SOM,
    som_max_elements: int = DEFAULT_SOM_MAX_ELEMENTS,
    reasoning_effort: Optional[str] = None,
    steering: Optional[Dict[str, Any]] = None,
    retriever: Optional[Any] = None,
    retriever_top_k: int = 3,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one OSWorld episode end-to-end and return ``(Episode, stats)``.

    ``steering`` is an optional dict ``{"memory": ?, "reflector": ?,
    "verifier": ?}`` carrying instances built in ``main()`` from the
    opt-in flags ``--enable_memory`` / ``--enable_reflection`` /
    ``--enable_self_verify``. Any key may be ``None`` independently;
    when all three are ``None`` (or ``steering`` itself is ``None``)
    this function executes the original main loop bit-for-bit.

    ``retriever`` is the opt-in SkillBankRetriever from
    ``--skill_bank_path``; ``None`` → no retrieval, no in-context
    skill demos.
    """
    osworld_obs_to_schema_heuristic = _import_osworld_heuristic()
    som_helpers = _import_som_helpers() if use_som else None
    if use_som and som_helpers is None:
        # SoM was requested but Pillow / module unavailable — quietly
        # fall back to the raw-pixel ablation. Caller's verbose log
        # will already carry the import warning.
        use_som = False

    if seed is not None:
        random.seed(seed)

    obs, info = env.reset(
        seed=seed,
        options={"task_config": task_cfg},
    )

    instruction = obs.get("instruction") or task_cfg.get("instruction", "")
    task_id = task_cfg.get("id", safe_task_id)
    task = (
        f"OSWorld task ({domain}): {instruction}"
        if instruction else f"OSWorld task ({domain}): {task_id}"
    )
    goal = instruction or task_id

    # ─── Opt-in steering instances (#3 / #4 / #6) ────────────────────────
    # ``steering`` is a dict carrying optional MemorySummary,
    # ReflexionTrigger, SelfVerifier. Each key may be missing or
    # None → that subsystem is disabled. When all are None this
    # block becomes a no-op and the loop matches the pre-feature
    # path exactly.
    memory = (steering or {}).get("memory")
    reflector = (steering or {}).get("reflector")
    verifier = (steering or {}).get("verifier")

    # ─── Opt-in skill retrieval (#7) ─────────────────────────────────────
    # Retrieval fires ONCE at episode start — the in-context demos do
    # not change mid-episode. The block is reused on every step's
    # action prompt. Empty string (or None) → no demos rendered.
    retrieved_skills_block: Optional[str] = None
    n_retrieved_skills = 0
    if retriever is not None:
        try:
            skills = retriever.retrieve(
                instruction=goal, domain=domain, top_k=retriever_top_k,
            )
            if skills:
                retrieved_skills_block = retriever.format_for_prompt(skills)
                n_retrieved_skills = len(skills)
                if verbose:
                    print(
                        f"  [retrieve] {n_retrieved_skills} in-context "
                        f"skills loaded for episode"
                    )
        except Exception as exc:
            logger.warning(
                "[retrieve] retrieval failed for task %s: %s",
                task_id, exc,
            )

    experiences: List[Experience] = []
    history: List[Dict[str, Any]] = []
    consecutive_noops = 0
    last_noop_action: Optional[str] = None
    schema_calls = 0
    schema_ok = 0
    action_llm_ok = 0
    action_llm_fail = 0
    som_steps_with_elements = 0
    som_actions_translated = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    eval_score: Optional[float] = None
    # Subgoal trail (P1) — populated step-by-step from the actor's
    # ``choose_action(subgoal=...)`` arg. Persisted into each
    # Experience.metadata so the skill-bank lift pipeline can use the
    # subgoal as the segmentation anchor instead of re-deriving it
    # from raw actions.
    subgoals: List[str] = []

    last_action: str = ""
    last_action_error: str = ""
    last_action_was_noop: bool = False

    t0 = time.time()
    try:
        for step in range(max_steps):
            obs_with_history = dict(obs)
            obs_with_history["last_action"] = last_action
            obs_with_history["last_action_error"] = last_action_error

            # 1. Pull the screenshot.
            pil_raw = _to_pil(obs_with_history.get("screenshot"))

            # 1b. Set-of-Marks: extract clickable elements from the
            # AT-SPI tree and overlay numbered red boxes on the
            # screenshot. The annotated image is what we send to the
            # VLM AND what we save to disk so failure modes are easy
            # to inspect visually.
            som_elements: List[Any] = []
            if use_som and som_helpers is not None:
                a11y_xml = obs_with_history.get("accessibility_tree", "") or ""
                try:
                    som_elements = som_helpers["extract"](
                        a11y_xml, max_elements=som_max_elements,
                    )
                except Exception as exc:
                    logger.debug("[som] extract failed: %s", exc)
                    som_elements = []
            if som_elements:
                som_steps_with_elements += 1

            pil = pil_raw
            if pil_raw is not None and som_elements and som_helpers is not None:
                try:
                    pil = som_helpers["draw"](pil_raw, som_elements)
                except Exception as exc:
                    logger.debug("[som] draw_overlay failed: %s", exc)
                    pil = pil_raw

            img_path: Optional[str] = None
            if pil is not None and frames_dir is not None:
                img_path = _save_frame(pil, frames_dir / f"step_{step:03d}.png")

            # 2. Heuristic schema (deterministic AT-SPI walker).
            try:
                canonical_schema = osworld_obs_to_schema_heuristic(
                    obs_with_history, step=step, task_id=task_id,
                    max_entities=max_entities,
                )
            except Exception as exc:
                logger.debug("heuristic obs_to_schema failed: %s", exc)
                canonical_schema = None

            # 3. Candidate-action vocabulary (SoM verbs first when active).
            candidate_actions, candidate_meta = _build_candidate_actions(
                obs=obs_with_history,
                som_elements=som_elements,
            )

            # 4. Visual schema (vision call): screenshot → schema.
            schema_text: Optional[str] = None
            schema_meta: Dict[str, Any] = {
                "schema": None, "raw": "", "source": "skipped", "error": None,
            }
            if (
                use_vision and pil is not None
                and schema_helpers is not None and client is not None
            ):
                schema_budget = (
                    _SCHEMA_MAX_TOKENS_REASONING
                    if _is_reasoning_model(routed_model)
                    else _SCHEMA_MAX_TOKENS
                )
                schema_meta = generate_schema_from_image(
                    pil_image=pil,
                    obs=obs_with_history,
                    task_id=task_id,
                    goal=goal,
                    step=step,
                    candidate_actions=candidate_actions,
                    client=client,
                    routed_model=routed_model,
                    schema_helpers=schema_helpers,
                    canonical_fallback=canonical_schema,
                    temperature=temperature_schema,
                    max_tokens=schema_budget,
                    max_entities=max_entities,
                    som_elements=som_elements,
                    som_helpers=som_helpers,
                    reasoning_effort=reasoning_effort,
                )
                schema_calls += 1
                if schema_meta.get("source") == "vlm":
                    schema_ok += 1
                schema_text = schema_meta.get("schema")
            else:
                schema_text = canonical_schema
                schema_meta = {
                    "schema": canonical_schema,
                    "raw": "",
                    "source": "canonical" if canonical_schema else "text_only",
                    "error": None,
                }

            # 4b. Build opt-in steering blocks BEFORE the action call so
            #     each can land in the same prompt. Each call is a
            #     no-op when its subsystem is disabled (returns None).
            memory_block: Optional[str] = None
            if memory is not None:
                try:
                    memory_block = memory.maybe_refresh(
                        step=step, task=task,
                        history=history, subgoals=subgoals,
                    )
                except Exception as exc:
                    logger.debug("[memory] maybe_refresh failed: %s", exc)
                    memory_block = None

            reflection_block: Optional[str] = None
            if reflector is not None:
                try:
                    reflector.maybe_reflect(
                        step=step,
                        consecutive_noops=consecutive_noops,
                        last_action=last_action,
                        task=task,
                        recent_subgoals=subgoals,
                        recent_history=history,
                    )
                    reflection_block = reflector.consume_for(step)
                except Exception as exc:
                    logger.debug("[reflect] maybe_reflect failed: %s", exc)
                    reflection_block = None

            # 5. Action selection (text-only call: schema → action).
            action, reasoning, action_raw, action_err, subgoal = select_action(
                schema_text=schema_text,
                obs=obs_with_history,
                candidate_actions=candidate_actions,
                candidate_meta=candidate_meta,
                task=task,
                step=step,
                history=history,
                client=client,
                routed_model=routed_model,
                temperature=temperature_action,
                done_nudge_step=done_nudge_step,
                som_elements=som_elements,
                som_helpers=som_helpers,
                last_action_was_noop=last_action_was_noop,
                last_failed_action=last_action if last_action_was_noop else None,
                reasoning_effort=reasoning_effort,
                prior_subgoals=subgoals,
                memory_block=memory_block,
                reflection_block=reflection_block,
                retrieved_skills_block=retrieved_skills_block,
            )
            subgoals.append(subgoal or "")
            if action is not None:
                action_llm_ok += 1
            else:
                action_llm_fail += 1
                action = candidate_actions[0] if candidate_actions else "WAIT"
                if last_noop_action == action and len(candidate_actions) > 1:
                    action = _pick_different(action, candidate_actions)

            # 6. Anti-NOOP override.
            if (
                consecutive_noops >= _MAX_CONSECUTIVE_NOOPS
                and action == last_noop_action
                and len(candidate_actions) > 1
            ):
                old_action = action
                action = _pick_different(action, candidate_actions)
                reasoning = (
                    (reasoning or "")
                    + f" [auto-override: '{old_action}' was no-op {consecutive_noops}x]"
                )
                if verbose:
                    print(f"  step {step}: anti-noop override {old_action!r} -> {action!r}")

            # 6a. Translate Set-of-Marks verbs back to pyautogui calls.
            # The model emitted ``click_element(id=N)`` against the
            # numbered overlay; we look up element N and rewrite to
            # ``pyautogui.click(cx, cy)`` so the env can execute it.
            som_action_original: Optional[str] = None
            if (
                use_som and som_helpers is not None and som_elements
                and isinstance(action, str)
            ):
                try:
                    translated = som_helpers["translate"](action, som_elements)
                except Exception as exc:
                    logger.debug("[som] translate failed: %s", exc)
                    translated = None
                if translated:
                    som_action_original = action
                    action = translated
                    som_actions_translated += 1
                    if verbose:
                        print(
                            f"  step {step}: som-translate "
                            f"{som_action_original!r} -> {action!r}"
                        )

            # 6b. Anti-loop override — force DONE when the agent has been
            # repeating the same action with no reward progress. This kills
            # the "click button → escape → click button → escape" failure
            # mode observed with reasoning models that refuse to commit.
            # Forcing DONE gives the OSWorld evaluator a chance to score
            # the current state; if the goal is partially satisfied we
            # still get credit, otherwise we save 30+ wasted steps per
            # loopy trajectory.
            if step >= loop_min_step and action != "DONE":
                is_looping, repeated_action, repeat_count = _detect_action_loop(
                    history,
                    window=loop_window,
                    repeat_threshold=loop_repeat_threshold,
                )
                if is_looping:
                    old_action = action
                    action = "DONE"
                    reasoning = (
                        (reasoning or "")
                        + f" [loop-abort: '{repeated_action}' "
                          f"repeated {repeat_count}x in last "
                          f"{loop_window} steps with reward=0; "
                          f"force-emitting DONE so the evaluator can "
                          f"score the current state]"
                    )
                    if verbose:
                        print(
                            f"  step {step}: loop-abort {old_action!r} -> "
                            f"DONE  (saw {repeat_count}x "
                            f"{repeated_action!r} in last {loop_window})"
                        )

            # 6c. Opt-in self-verification (improvement #6).
            # Before letting an actor-emitted DONE commit, ask a fresh
            # vision LLM call whether the screenshot objectively
            # satisfies the task. On NO, downgrade to ``WAIT`` so the
            # episode loop continues and the actor gets one more
            # chance. Skipped (no extra cost, no behaviour change)
            # when ``--enable_self_verify`` is off.
            self_verify_outcome: Optional[str] = None
            self_verify_reason: Optional[str] = None
            if (
                verifier is not None
                and isinstance(action, str)
                and action.strip().upper() == "DONE"
            ):
                # Use the SAME annotated PIL frame the actor saw so
                # the verifier judges the same evidence the actor did.
                data_url = _pil_to_data_url(pil)
                try:
                    is_done, reason = verifier.verify_done(
                        task=task, screenshot_data_url=data_url,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[self-verify] crashed: %s", exc)
                    is_done, reason = True, "verifier_crashed"
                self_verify_reason = reason
                if is_done:
                    self_verify_outcome = "yes"
                else:
                    self_verify_outcome = "no"
                    if verbose:
                        print(
                            f"  step {step}: self-verify rejected DONE "
                            f"({reason[:60] if reason else '?'}) — "
                            f"downgrading to WAIT"
                        )
                    action = "WAIT"
                    reasoning = (
                        (reasoning or "")
                        + f" [self-verify: NO ({reason[:80]}); "
                          f"downgraded DONE→WAIT]"
                    )

            # 7. Step the env.
            try:
                next_obs, reward, terminated, truncated, step_info = env.step(action)
            except Exception as exc:
                logger.error(
                    "[%s] step %d env.step(%r) failed: %s",
                    task_id, step, action, exc,
                )
                if verbose:
                    traceback.print_exc()
                last_action = action
                last_action_error = str(exc)
                # Record a synthetic experience so the trajectory is
                # debuggable, then break out of the loop.
                next_obs = obs
                reward = 0.0
                terminated = True
                truncated = False
                step_info = {"step_error": str(exc)}

            r = float(reward or 0.0)
            total_reward += r
            done = bool(terminated) or bool(truncated)

            # OSWorld doesn't surface per-step errors directly — fall back
            # to step_info if the wrapper exposed any.
            error_text = ""
            if isinstance(step_info, dict):
                if step_info.get("step_error"):
                    error_text = str(step_info["step_error"])
                elif step_info.get("error"):
                    error_text = str(step_info["error"])
            # Pull the eval score whenever the wrapper auto-evaluated, not
            # just on agent-emitted DONE. ``OSWorldGymWrapper.step()``
            # already calls ``env.evaluate()`` on every ``terminated``
            # return (which includes both DONE and FAIL), so reading it
            # only on DONE silently dropped 23% of episodes (the FAIL
            # ones) from ``mean_eval_score`` in the May-2026 cold-start
            # run. Truncated episodes are handled after the loop below.
            if isinstance(step_info, dict):
                step_eval = step_info.get("eval_score")
                if isinstance(step_eval, (int, float)):
                    eval_score = float(step_eval)

            is_noop = _is_noop(obs_with_history, next_obs)
            history.append({
                "action": action,
                "reward": r,
                "noop": is_noop,
                "error": bool(error_text),
                "error_text": error_text,
            })

            if is_noop and action == last_noop_action:
                consecutive_noops += 1
            elif is_noop:
                consecutive_noops = 1
                last_noop_action = action
            else:
                consecutive_noops = 0
                last_noop_action = None

            # 8. Build the Experience record.
            obs_summary = (
                f"task={task_id} domain={domain} step={step} "
                f"a11y_chars={len(obs_with_history.get('accessibility_tree','') or '')} "
                f"goal={_truncate(goal, 120)}"
            )
            next_obs_summary = (
                f"task={task_id} domain={domain} step={step+1} "
                f"a11y_chars={len(next_obs.get('accessibility_tree','') or '')} "
                f"err={_truncate(error_text, 120) or 'null'}"
            )
            exp = Experience(
                state=obs_summary,
                action=str(action),
                reward=r,
                next_state=next_obs_summary,
                done=done,
                intentions=reasoning,
                tasks=task,
            )
            exp.idx = step
            exp.action_type = "primitive"
            exp.raw_state = obs_summary[:4000]
            exp.raw_next_state = next_obs_summary[:4000]
            exp.available_actions = list(candidate_actions)
            exp.interface = {
                "env_name": "osworld",
                "game_name": safe_task_id,
                "domain": domain,
                "task_id": task_id,
                "wrapper": "OSWorldGymWrapper",
            }

            extras: Dict[str, Any] = {
                "schema": schema_text,
                "schema_source": schema_meta.get("source"),
                "schema_error": schema_meta.get("error"),
                "schema_canonical": canonical_schema,
                "candidate_actions": list(candidate_actions),
                "candidate_meta": candidate_meta[:_MAX_CANDIDATE_ACTIONS],
                "is_noop": is_noop,
                "error_text": error_text or None,
                "som_active": bool(use_som and som_helpers is not None),
                "som_n_elements": len(som_elements),
                "som_action_original": som_action_original,
                "task_id": task_id,
                "domain": domain,
                "subgoal": subgoal,
            }
            # Opt-in steering provenance (only populated when the
            # corresponding flag was on; otherwise these fields are
            # absent so the on-disk shape matches the pre-feature
            # path for default runs).
            if memory_block:
                extras["memory_block_used"] = True
            if reflection_block:
                extras["reflection_block_used"] = True
            if retrieved_skills_block:
                extras["retrieved_skills_used"] = True
                extras["retrieved_skills_count"] = n_retrieved_skills
            if self_verify_outcome:
                extras["self_verify_outcome"] = self_verify_outcome
                extras["self_verify_reason"] = self_verify_reason
            if schema_meta.get("finish_reason") is not None:
                extras["schema_finish_reason"] = schema_meta.get("finish_reason")
            if schema_meta.get("recovery"):
                extras["schema_recovery"] = schema_meta.get("recovery")
            schema_raw = schema_meta.get("raw") or ""
            if schema_raw:
                extras["schema_raw_excerpt"] = schema_raw[:4000]
                extras["schema_raw_full_len"] = schema_meta.get(
                    "raw_full_len", len(schema_raw)
                )
            if action_raw:
                extras["action_raw"] = action_raw[:1000]
            if action_err:
                extras["action_error"] = action_err
            if img_path:
                extras["frame_path"] = img_path
            if eval_score is not None:
                extras["eval_score"] = eval_score
            exp.extras = extras
            existing_meta = getattr(exp, "metadata", None) or {}
            if isinstance(existing_meta, dict):
                existing_meta = dict(existing_meta)
            else:
                existing_meta = {}
            existing_meta.update(extras)
            exp.metadata = existing_meta
            experiences.append(exp)

            # Sidecar JSON next to each frame so the PNG is self-describing.
            if img_path and frames_dir is not None:
                try:
                    sidecar = {
                        "step": step,
                        "task_id": task_id,
                        "domain": domain,
                        "action": action,
                        "subgoal": subgoal,
                        "action_raw": (action_raw or "")[:1000],
                        "action_error": action_err,
                        "reward": r,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "is_noop": is_noop,
                        "schema_source": schema_meta.get("source"),
                        "schema_error": schema_meta.get("error"),
                        "schema": schema_text,
                        "candidate_actions": list(candidate_actions),
                        "frame_path": img_path,
                        "eval_score": eval_score,
                        "som_active": bool(use_som and som_helpers is not None),
                        "som_n_elements": len(som_elements),
                        "som_action_original": som_action_original,
                        "som_table": (
                            som_helpers["format"](som_elements)
                            if som_elements and som_helpers is not None
                            else None
                        ),
                    }
                    sidecar_path = frames_dir / f"step_{step:03d}.json"
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar, f, indent=2, ensure_ascii=False, default=str)
                except Exception as exc:
                    logger.debug("frame sidecar write failed at step %d: %s", step, exc)

            if verbose:
                r_short = (
                    (reasoning[:80] + "...")
                    if reasoning and len(reasoning) > 80 else reasoning
                )
                tag = (
                    " [ERR]" if error_text else
                    (" [NOOP]" if is_noop else "")
                )
                print(
                    f"  step {step:>3}: action={action!r:<40} "
                    f"reward={r:+.2f} cum={total_reward:+.2f}{tag} "
                    f"schema={schema_meta.get('source')} reason={r_short}"
                )

            last_action = action
            last_action_error = error_text
            last_action_was_noop = bool(is_noop and not error_text)
            obs = next_obs
            if done:
                break

    finally:
        # NB: do not close env here — same env instance is reused across
        # multiple episodes by run_task_rollouts to amortise VM boot cost.
        pass

    # If the episode ended in truncation (max_steps reached without an
    # agent-emitted DONE / FAIL), the wrapper does NOT auto-evaluate.
    # Force one final ``env.evaluate()`` so the metric covers ALL
    # episodes — not just the agent's self-declared DONEs. In the
    # May-2026 cold-start run this hid 17% of episodes from
    # ``mean_eval_score``; the few that the env quietly already
    # satisfied were silently scored as None.
    if eval_score is None and truncated and not last_action_error:
        try:
            if hasattr(env, "evaluate"):
                eval_score = float(env.evaluate())
                if verbose:
                    print(
                        f"  [truncated] post-hoc env.evaluate() = "
                        f"{eval_score}"
                    )
        except Exception as exc:
            logger.debug("post-hoc env.evaluate() failed: %s", exc)

    elapsed = time.time() - t0

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="osworld",
        game_name=safe_task_id,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "domain": domain,
        "task_id": task_id,
        "safe_task_id": safe_task_id,
        "wrapper": "OSWorldGymWrapper",
        "macro": False,
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "elapsed_seconds": round(elapsed, 2),
        "model": fallback_model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_osworld",
        "use_vision": use_vision,
        "schema_calls": schema_calls,
        "schema_ok": schema_ok,
        "action_llm_ok": action_llm_ok,
        "action_llm_fail": action_llm_fail,
        "noop_steps": sum(1 for h in history if h["noop"]),
        "error_steps": sum(1 for h in history if h.get("error")),
        "eval_score": eval_score,
        "use_som": bool(use_som and som_helpers is not None),
        "som_steps_with_elements": som_steps_with_elements,
        "som_actions_translated": som_actions_translated,
        # P1 subgoal trail — one string per outer step, parallel-indexed
        # to ``experiences``. Empty strings mean the actor failed to
        # emit a subgoal that step (or the LLM call itself failed).
        # The skill-bank lift uses this as the segmentation anchor.
        "subgoal_trail": list(subgoals),
        "subgoal_unique": len({s for s in subgoals if s}),
    }
    # Opt-in steering provenance — present only when a flag was set.
    if memory is not None:
        try:
            stats["memory_stats"] = memory.stats()
        except Exception:  # noqa: BLE001
            pass
    if reflector is not None:
        try:
            stats["reflection_stats"] = reflector.stats()
        except Exception:  # noqa: BLE001
            pass
    if verifier is not None:
        try:
            stats["self_verify_stats"] = verifier.stats()
        except Exception:  # noqa: BLE001
            pass
    if retriever is not None:
        stats["retrieved_skills_count"] = n_retrieved_skills
    return episode, stats


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _count_existing_episodes(task_dir: Path) -> int:
    if not task_dir.exists():
        return 0
    return sum(
        1 for f in task_dir.glob("episode_*.json")
        if f.name != "episode_buffer.json"
    )


def _save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_task_rollouts(
    task_cfg: Dict[str, Any],
    *,
    args: argparse.Namespace,
    env: Any,
    output_dir: Path,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all episodes for one OSWorld task and persist outputs."""
    task_id = task_cfg.get("id") or _safe_id(task_cfg.get("instruction", ""))
    domain = _resolve_task_domain(task_cfg, default=getattr(args, "_inferred_domain", "unknown"))
    safe = _safe_id(task_id)
    domain_dir = output_dir / _safe_id(domain)
    task_dir = domain_dir / safe
    task_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = task_dir / "rollouts.jsonl"

    target_episodes = args.episodes
    effective_max_steps = args.max_steps
    label = f"{domain}/{task_id}"

    start_idx = 0
    if args.resume:
        start_idx = _count_existing_episodes(task_dir)
        if start_idx >= target_episodes:
            print(f"  [SKIP] {label}: {start_idx}/{target_episodes} episodes already done")
            return {
                "task_id": task_id,
                "domain": domain,
                "safe_task_id": safe,
                "skipped": True,
                "existing": start_idx,
                "target_episodes": target_episodes,
            }
        if start_idx > 0:
            print(f"  [RESUME] {label}: starting from episode {start_idx}")

    buffer = Episode_Buffer(buffer_size=target_episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t_target = time.time()

    save_frames = getattr(args, "_save_frames", not getattr(args, "no_save_frames", False))

    for ep_idx in range(start_idx, target_episodes):
        print(f"\n  [{label}] Episode {ep_idx + 1}/{target_episodes}")
        try:
            frames_dir = (
                task_dir / "frames" / f"ep_{ep_idx:03d}"
                if save_frames else None
            )

            episode, stats = run_actor_episode(
                env=env,
                task_cfg=task_cfg,
                domain=domain,
                safe_task_id=safe,
                max_steps=effective_max_steps,
                client=client,
                routed_model=routed_model,
                fallback_model=args.model,
                schema_helpers=schema_helpers,
                use_vision=True,  # vision is mandatory in this pipeline
                temperature_action=args.temperature_action,
                temperature_schema=args.temperature_schema,
                max_entities=args.max_entities,
                frames_dir=frames_dir,
                seed=42 + ep_idx,
                verbose=args.verbose,
                loop_window=args.loop_window,
                loop_repeat_threshold=args.loop_repeat_threshold,
                loop_min_step=args.loop_min_step,
                done_nudge_step=args.done_nudge_step,
                use_som=getattr(args, "_use_som", DEFAULT_USE_SOM),
                som_max_elements=getattr(
                    args, "som_max_elements", DEFAULT_SOM_MAX_ELEMENTS
                ),
                reasoning_effort=getattr(args, "reasoning_effort", None),
                # Opt-in features — None when the corresponding flag
                # is off, in which case run_actor_episode runs the
                # original main loop bit-for-bit.
                steering=getattr(args, "_steering", None),
                retriever=getattr(args, "_retriever", None),
                retriever_top_k=getattr(args, "skill_retrieval_top_k", 3),
            )
            stats["episode_index"] = ep_idx
            print(
                f"    steps={stats['steps']:>3} "
                f"reward={stats['total_reward']:+.2f} "
                f"eval={stats.get('eval_score')!s:<5} "
                f"schema_ok={stats['schema_ok']}/{stats['schema_calls']} "
                f"action_ok={stats['action_llm_ok']} (fail={stats['action_llm_fail']}) "
                f"noops={stats['noop_steps']} errs={stats['error_steps']}"
            )

            buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            with open(task_dir / f"episode_{ep_idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
            _save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as exc:
            print(f"    [ERROR] episode {ep_idx + 1} failed: {exc}")
            traceback.print_exc()
            all_stats.append({
                "task_id": task_id,
                "domain": domain,
                "episode_index": ep_idx,
                "error": str(exc),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed_target = time.time() - t_target
    buffer.save_to_json(str(task_dir / "episode_buffer.json"))
    print(f"\n  Saved {len(buffer)} episodes for {label} in {elapsed_target:.1f}s")

    summary: Dict[str, Any] = {
        "task_id": task_id,
        "domain": domain,
        "safe_task_id": safe,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_osworld",
        "wrapper": "OSWorldGymWrapper",
        "instruction": task_cfg.get("instruction", ""),
        "target_episodes": target_episodes,
        "completed_episodes": len([s for s in all_stats if "error" not in s]),
        "use_vision": True,
        "use_som": getattr(args, "_use_som", DEFAULT_USE_SOM),
        "som_max_elements": getattr(
            args, "som_max_elements", DEFAULT_SOM_MAX_ELEMENTS
        ),
        "save_frames": save_frames,
        "max_steps": effective_max_steps,
        "elapsed_seconds": round(elapsed_target, 2),
        "episode_stats": all_stats,
    }
    rewards = [s["total_reward"] for s in all_stats if "error" not in s]
    steps_list = [s["steps"] for s in all_stats if "error" not in s]
    valid_runs = [s for s in all_stats if "error" not in s]
    eval_scores = [
        s.get("eval_score") for s in valid_runs
        if isinstance(s.get("eval_score"), (int, float))
    ]
    if rewards:
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
    if steps_list:
        summary["mean_steps"] = sum(steps_list) / len(steps_list)
    # Honest pass@1: count any episode with eval_score>0 as solved, and
    # any unsigned (None) episode as 0. The ``mean_eval_score`` field
    # is preserved for backward-compat (it averages only scored
    # episodes), but ``pass_rate`` / ``solved`` / ``unscored`` give the
    # full picture.
    if eval_scores:
        summary["mean_eval_score"] = sum(eval_scores) / len(eval_scores)
    if valid_runs:
        summary["solved"] = sum(
            1 for s in valid_runs
            if isinstance(s.get("eval_score"), (int, float))
            and float(s["eval_score"]) > 0
        )
        summary["unscored"] = sum(
            1 for s in valid_runs if s.get("eval_score") is None
        )
        summary["pass_rate"] = summary["solved"] / len(valid_runs)

    with open(task_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Catalog → (domain, task_cfg) resolution
# ---------------------------------------------------------------------------

def _load_task_catalog_grouped(
    catalog_path: Path,
    *,
    domains: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    limit_per_domain: Optional[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Load OSWorld test_*.json catalog and return ``[(domain, task_cfg), ...]``.

    The on-disk format is ``{"<domain>": ["<task_id>", ...]}`` where each
    string ID resolves to ``examples/<domain>/<id>.json``. We resolve
    them all here so we keep the domain label attached to each task.
    """
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Task catalog not found: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples_root = catalog_path.parent / "examples"

    selected_domains = list(domains) if domains else None
    selected_ids: Optional[set] = set(task_ids) if task_ids else None

    out: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(data, dict):
        for domain_name, ids in data.items():
            if selected_domains and domain_name not in selected_domains:
                continue
            entries: List[Any] = []
            if isinstance(ids, list):
                entries = list(ids)
            elif isinstance(ids, dict):
                entries = list(ids.values())
            else:
                continue
            kept_for_domain = 0
            for entry in entries:
                task_cfg: Optional[Dict[str, Any]] = None
                if isinstance(entry, dict):
                    task_cfg = entry
                elif isinstance(entry, str):
                    cand = examples_root / domain_name / f"{entry}.json"
                    if cand.exists():
                        try:
                            with open(cand, "r", encoding="utf-8") as fh:
                                task_cfg = json.load(fh)
                        except Exception as exc:
                            logger.warning(
                                "failed to load task %s/%s: %s",
                                domain_name, entry, exc,
                            )
                if task_cfg is None:
                    continue
                if selected_ids and task_cfg.get("id") not in selected_ids:
                    continue
                # Inject the bucket name so downstream callers can group by it.
                task_cfg.setdefault("domain", domain_name)
                out.append((domain_name, task_cfg))
                kept_for_domain += 1
                if limit_per_domain and kept_for_domain >= limit_per_domain:
                    break
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            domain_name = entry.get("domain") or _resolve_task_domain(entry)
            if selected_domains and domain_name not in selected_domains:
                continue
            if selected_ids and entry.get("id") not in selected_ids:
                continue
            entry.setdefault("domain", domain_name)
            out.append((domain_name, entry))

    return out


def _print_catalog_summary(catalog_path: Path):
    """Pretty-print the task catalog (groups + counts)."""
    pairs = _load_task_catalog_grouped(catalog_path)
    by_domain: Dict[str, List[str]] = {}
    for d, t in pairs:
        by_domain.setdefault(d, []).append(t.get("id", "<no-id>"))
    total = sum(len(v) for v in by_domain.values())
    print(f"OSWorld task catalog: {catalog_path}")
    print(f"Domains: {len(by_domain)} | Total tasks: {total}")
    for d in sorted(by_domain):
        ids = by_domain[d]
        sample = ids[0] if ids else ""
        print(f"  {d:<20s} {len(ids):>4d} tasks   e.g. {sample}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_screen_size(s: str) -> Tuple[int, int]:
    """Parse 'WxH' (or 'W,H') → (W, H)."""
    if not s:
        return (_DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H)
    sep = "x" if "x" in s.lower() else ","
    try:
        w_str, h_str = s.lower().split(sep)
        return (int(w_str.strip()), int(h_str.strip()))
    except Exception:
        return (_DEFAULT_SCREEN_W, _DEFAULT_SCREEN_H)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start actor-agent rollouts using gpt-5.5 visual grounding "
            "+ schema-driven action selection over OSWorld DesktopEnv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task_catalog", type=str, default=DEFAULT_TASK_CATALOG,
        help=(
            "Path to an OSWorld test_*.json catalog (default: "
            f"{DEFAULT_TASK_CATALOG}). The catalog is a "
            "{domain: [task_id, ...]} mapping; each task_id resolves to "
            "examples/<domain>/<id>.json."
        ),
    )
    parser.add_argument(
        "--domains", type=str, nargs="+", default=None,
        help=(
            "Restrict to specific OSWorld domains. Choose from: "
            f"{', '.join(ALL_OSWORLD_DOMAINS)}. Default: all in the catalog."
        ),
    )
    parser.add_argument(
        "--task_ids", type=str, nargs="+", default=None,
        help="Restrict to specific OSWorld task IDs (UUID strings).",
    )
    parser.add_argument(
        "--tasks_per_domain", type=int, default=None,
        help="Max tasks per domain (useful for smoke tests).",
    )
    parser.add_argument(
        "--list_tasks", action="store_true",
        help="Print task catalog summary (domains + counts) and exit.",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Episodes per task (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max outer steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--reasoning_effort", "--reasoning-effort",
        type=str, default=None,
        choices=list(_VALID_REASONING_EFFORTS),
        help=(
            "OpenAI reasoning_effort knob for gpt-5.x / o1 / o3 / o4. "
            "One of {minimal, low, medium, high}. Default: unset (= "
            "OpenAI default 'medium'). Recommended for cold-start data "
            "generation: 'minimal' — the SFT student only learns from "
            "the visible <state> and pyautogui action, so hidden "
            "thinking is wasted spend."
        ),
    )
    parser.add_argument(
        "--max_entities", type=int, default=_DEFAULT_MAX_ENTITIES,
        help=f"Cap on entities per schema (default: {_DEFAULT_MAX_ENTITIES})",
    )
    parser.add_argument(
        "--loop_window", type=int, default=DEFAULT_LOOP_WINDOW,
        help=f"Anti-loop: rolling window size when scanning for repeated "
             f"actions (default: {DEFAULT_LOOP_WINDOW}).",
    )
    parser.add_argument(
        "--loop_repeat_threshold", type=int,
        default=DEFAULT_LOOP_REPEAT_THRESHOLD,
        help=f"Anti-loop: trigger force-DONE when the same action appears "
             f"this many times in --loop_window with reward=0 "
             f"(default: {DEFAULT_LOOP_REPEAT_THRESHOLD}). Set very high "
             f"(e.g. 999) to disable loop detection.",
    )
    parser.add_argument(
        "--loop_min_step", type=int, default=DEFAULT_LOOP_MIN_STEP,
        help=f"Anti-loop: do not abort earlier than this step "
             f"(default: {DEFAULT_LOOP_MIN_STEP}).",
    )
    parser.add_argument(
        "--done_nudge_step", type=int, default=DEFAULT_DONE_NUDGE_STEP,
        help=f"DONE-nudge: starting at this step, the action prompt gets "
             f"an explicit 'commit to DONE if goal is satisfied' reminder "
             f"(default: {DEFAULT_DONE_NUDGE_STEP}). Set very high "
             f"(e.g. 999) to disable.",
    )
    parser.add_argument(
        "--no_som", action="store_true",
        help=(
            "Disable Set-of-Marks visual grounding (numbered red boxes "
            "drawn over interactive AT-SPI elements). SoM is the single "
            "biggest known lever for OSWorld pass-rate (~5%% to ~18%% "
            "on the same VLM backbone in published baselines). Disable "
            "ONLY when running the raw-pixel ablation."
        ),
    )
    parser.add_argument(
        "--som_max_elements", type=int, default=DEFAULT_SOM_MAX_ELEMENTS,
        help=f"Max numbered boxes drawn on the SoM overlay "
             f"(default: {DEFAULT_SOM_MAX_ELEMENTS}). Lower this if the "
             f"overlay looks visually crowded; raise it for dense UIs.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Backbone model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature_action", type=float, default=0.4,
        help="Sampling temperature for the action call (default: 0.4)",
    )
    parser.add_argument(
        "--temperature_schema", type=float, default=0.2,
        help="Sampling temperature for the visual schema call (default: 0.2)",
    )
    parser.add_argument(
        "--no_save_frames", action="store_true",
        help=(
            "Skip persisting the per-step PNGs + step_NNN.json sidecars. "
            "By default this pipeline saves every frame the VLM sees to "
            "<domain>/<task_id>/frames/ep_NNN/step_NNN.png with a sidecar "
            "step_NNN.json carrying action / reward / schema / eval / "
            "candidates — useful since the VM is always headless. Pass "
            "this flag if disk pressure matters; the schema and action "
            "are still recorded in rollouts.jsonl + episode_NNN.json."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip episodes that already have an episode_NNN.json on disk.",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Override OPENAI_API_KEY / OPENROUTER_API_KEY for this run.",
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="Override the OpenAI base URL (e.g. for a custom proxy).",
    )
    parser.add_argument(
        "--provider_name", type=str, default="docker",
        choices=["docker", "vmware", "aws", "virtualbox"],
        help="OSWorld VM provider (default: docker).",
    )
    parser.add_argument(
        "--path_to_vm", type=str, default=None,
        help="Provider-specific path to the VM image (vmware/virtualbox).",
    )
    parser.add_argument(
        "--os_type", type=str, default="Ubuntu",
        choices=["Ubuntu", "Windows", "macOS"],
        help="Guest OS (default: Ubuntu).",
    )
    parser.add_argument(
        "--action_space_type", type=str, default="pyautogui",
        choices=["pyautogui", "computer_13"],
        help="OSWorld action space (default: pyautogui).",
    )
    parser.add_argument(
        "--screen_size", type=str,
        default=f"{_DEFAULT_SCREEN_W}x{_DEFAULT_SCREEN_H}",
        help="VM screen resolution as WxH (default: %(default)s).",
    )
    parser.add_argument(
        "--pause_after_action", type=float, default=2.0,
        help="Seconds to wait after each action (default: 2.0).",
    )
    parser.add_argument(
        "--client_password", type=str, default="",
        help="VM sudo/login password (optional).",
    )
    parser.add_argument(
        "--enable_proxy", action="store_true",
        help="Enable proxy support for tasks that need it.",
    )
    parser.add_argument(
        "--no_a11y_tree", action="store_true",
        help="Disable accessibility-tree fetching (faster but the schema "
             "loses element bboxes).",
    )
    parser.add_argument(
        "--no_terminal", action="store_true",
        help="Disable terminal-output fetching.",
    )
    parser.add_argument(
        "--no_auto_evaluate", action="store_true",
        help="Don't call DesktopEnv.evaluate() on DONE/FAIL.",
    )
    parser.add_argument(
        "--reuse_env", action="store_true",
        help=(
            "Keep the same DesktopEnv instance across all tasks — the "
            "VM is rebooted between tasks via reset(task_config=...). "
            "Saves ~28 s per task. ON by default; pass --no_reuse_env "
            "to start a fresh VM per task."
        ),
    )
    parser.add_argument(
        "--no_reuse_env", action="store_true",
        help="Boot a fresh DesktopEnv per task (paranoid mode).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: <codebase_root>/Cold-start-out-osworld)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details (action, reward, schema source).",
    )
    # ─── Opt-in advanced steering (improvements #3 / #4 / #6) ────────────
    # These three live in cold_start/osworld_steering.py so the main
    # loop is unchanged when the flags are off. Each costs an extra
    # LLM round-trip per fire and is OSWorld-specific (depends on
    # AT-SPI / pyautogui / desktop semantics) — never enable on other
    # corpora.
    parser.add_argument(
        "--enable_memory", "--enable-memory", action="store_true",
        help=(
            "OPT-IN (default off). Every K=5 steps, run a small LLM "
            "summarisation call over the recent (subgoal, action, "
            "reward) trail and inject the result as a <memory> block "
            "on subsequent action prompts. Combats the long-horizon "
            "'lost-in-trajectory' failure mode where the actor "
            "forgets which subgoals it has already completed. "
            "OSWorld-only — implemented in osworld_steering."
            "MemorySummary."
        ),
    )
    parser.add_argument(
        "--memory_refresh_every", "--memory-refresh-every",
        type=int, default=5,
        help="K parameter for --enable_memory (default: 5 steps).",
    )
    parser.add_argument(
        "--enable_reflection", "--enable-reflection", action="store_true",
        help=(
            "OPT-IN (default off). When the actor has 2+ consecutive "
            "no-op steps (action did NOT change state and produced "
            "no error), fire a one-shot 'why did this fail? give 3 "
            "alternatives' LLM call and inject the result as a "
            "<reflection> block on the next prompt. Per-streak; "
            "does not re-fire until the streak breaks. OSWorld-only "
            "— implemented in osworld_steering.ReflexionTrigger."
        ),
    )
    parser.add_argument(
        "--reflection_streak", "--reflection-streak",
        type=int, default=2,
        help=(
            "Consecutive no-op count that triggers --enable_reflection "
            "(default: 2). Lower = more reflections fired (more cost, "
            "more recovery); higher = fewer (less cost, less rescue)."
        ),
    )
    parser.add_argument(
        "--enable_self_verify", "--enable-self-verify",
        action="store_true",
        help=(
            "OPT-IN (default off). Before accepting an actor-emitted "
            "DONE, fire a vision LLM call asking 'does this "
            "screenshot objectively satisfy the task?'. Only commit "
            "DONE on YES; on NO, downgrade to WAIT so the loop "
            "continues. Catches premature-DONE hallucinations that "
            "the schema's progress= field can also produce. "
            "OSWorld-only — implemented in osworld_steering."
            "SelfVerifier."
        ),
    )

    # ─── Opt-in skill-bank retrieval (improvement #7) ────────────────────
    parser.add_argument(
        "--skill_bank_path", "--skill-bank-path",
        type=str, default=None,
        help=(
            "OPT-IN (default unset). Path to a skill_bank.jsonl "
            "produced by labeling/extract_skillbank_gpt54.py or "
            "skill_transfer_test/extract/full_v5/<corpus>/. When "
            "set, at the START of every episode the harness "
            "retrieves the top-K most relevant skills (BM25 over "
            "skill name + protocol summary against the task "
            "instruction) and injects their compact protocol as "
            "in-context demonstrations in the actor user prompt. "
            "This is the eval-side hook for the skill-bank research "
            "story (RQ3 transfer matrix)."
        ),
    )
    parser.add_argument(
        "--skill_retrieval_top_k", "--skill-retrieval-top-k",
        type=int, default=3,
        help=(
            "Top-K parameter for --skill_bank_path (default: 3). "
            "Larger K trades context budget for recall."
        ),
    )

    parser.add_argument(
        "--eval_mode", "--eval-mode",
        type=str, nargs="?", const="medium", default=None,
        choices=["off", "low", "medium", "high", "max"],
        help=(
            "Switch to eval-grade defaults for benchmark numbers. "
            "Four tiers (cost ↑ → score ↑):\n"
            "  ``low`` (cheapest eval-grade): reasoning_effort=low, "
            "temperature_action=0.0, temperature_schema=0.0, "
            "done_nudge_step=999, max_steps=75. ~2x faster than "
            "``medium`` on gpt-5.x; minimal pass-rate hit (-2-3pp) "
            "for the cross-model OSWorld baseline use case where the "
            "actor task is mostly numbered-SoM pattern-matching, not "
            "deep multi-hop reasoning.\n"
            "  ``medium`` (default when --eval_mode is bare): same as "
            "``low`` but reasoning_effort=medium. ~2x token spend "
            "vs ``low`` on gpt-5.x; +2-3pp pass-rate.\n"
            "  ``high``: same as ``medium`` plus reasoning_effort=high. "
            "~2-3x token spend on the schema/action calls; expect "
            "+3-5pp pass-rate over ``medium`` on multi-step tasks.\n"
            "  ``max``: same as ``high`` plus max_steps=100. Use for "
            "the long-tail GIMP / VLC / multi-app workflows that need "
            "60+ steps. Highest spend, highest published number.\n"
            "Note: ``reasoning_effort`` is silently dropped by the "
            "driver for non-OpenAI-reasoning models (Claude, Gemini, "
            "Qwen3-VL on OpenRouter) — for those families the tier "
            "only changes temperature / done_nudge / max_steps.\n"
            "Without this flag the script keeps the cold-start data "
            "generation defaults (minimal effort, temperature 0.4, "
            "nudge at step 35) which intentionally trade pass-rate "
            "for trajectory diversity. Individual flags still win — "
            "use ``--eval_mode low --temperature_action 0.2`` to "
            "layer one explicit override on top of a preset."
        ),
    )

    args = parser.parse_args()

    # ``--eval_mode`` re-binds defaults BEFORE the rest of the file
    # reads them; explicit user overrides (e.g. ``--reasoning_effort
    # high`` passed alongside ``--eval_mode medium``) still take
    # precedence because the user-supplied values were set on the
    # namespace by argparse first and we only overwrite argparse's
    # defaults below. The three tiers (medium / high / max) are
    # documented in the --eval_mode help string.
    eval_tier = getattr(args, "eval_mode", None)
    if eval_tier and eval_tier != "off":
        if args.reasoning_effort is None:
            # Tier → reasoning_effort. The ``low`` tier is the cheapest
            # eval-grade preset added 2026-05-03 for the cross-model
            # OSWorld baseline; it is ~2x cheaper than ``medium`` on
            # gpt-5.x and roughly matches the schema-VLM / action-LLM
            # workload (numbered-SoM pattern matching, not deep
            # reasoning). For non-OpenAI-reasoning models the value is
            # silently dropped by ``_chat_completion``.
            args.reasoning_effort = {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "max": "high",
            }[eval_tier]
        # argparse defaults are 0.4 / 0.2 — only overwrite if the user
        # left them at the defaults.
        if args.temperature_action == 0.4:
            args.temperature_action = 0.0
        if args.temperature_schema == 0.2:
            args.temperature_schema = 0.0
        if args.done_nudge_step == DEFAULT_DONE_NUDGE_STEP:
            args.done_nudge_step = 999
        # ``max`` tier raises max_steps for long-horizon tasks. We
        # only override the argparse default (75) — if the user
        # explicitly set --max_steps, we keep their value.
        if eval_tier == "max" and args.max_steps == DEFAULT_MAX_STEPS:
            args.max_steps = 100

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    catalog_path = Path(args.task_catalog).expanduser().resolve()

    if args.list_tasks:
        _print_catalog_summary(catalog_path)
        return

    # Probe required imports up-front so we fail fast.
    try:
        OSWorldGymWrapper, _ = _import_osworld_gym_wrapper()
    except Exception as exc:
        print(
            "[FATAL] OSWorld / desktop_env stack not importable.\n"
            f"        {exc}\n"
            "        Install via:  bash install/install_osworld.sh\n"
            "        Then activate the env:  conda activate osworld"
        )
        sys.exit(2)

    # Resolve task list.
    try:
        tasks = _load_task_catalog_grouped(
            catalog_path,
            domains=args.domains,
            task_ids=args.task_ids,
            limit_per_domain=args.tasks_per_domain,
        )
    except FileNotFoundError as exc:
        print(f"[FATAL] {exc}")
        sys.exit(2)

    if not tasks:
        print("[ERROR] No tasks resolved from the catalog. Check --domains / "
              "--task_ids / --task_catalog.")
        sys.exit(2)

    # Output dir.
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else CODEBASE_ROOT / "Cold-start-out-osworld"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    has_key = bool(
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not has_key:
        print(
            "[FATAL] This pipeline requires the gpt-5.5 vision call on every "
            "step (vision is mandatory). Set one of:\n"
            "    export OPENAI_API_KEY='sk-...'\n"
            "    export OPENROUTER_API_KEY='sk-or-...'\n"
            "  (or drop the key in api_keys.py next to the repo root, or "
            "pass --api_key on the CLI)."
        )
        sys.exit(2)

    client, routed_model = _build_client_and_route(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    if client is None:
        print(
            "[WARNING] No OpenAI/OpenRouter client could be built — "
            "actor will fall through to deterministic defaults."
        )

    schema_helpers = _import_schema_helpers()
    if schema_helpers is None:
        print(
            "[FATAL] vlm_wrapper.schema is not importable; the gpt-5.5 "
            "visual schema call cannot run. Make sure the codebase root "
            f"({CODEBASE_ROOT}) is on PYTHONPATH and vlm_wrapper is "
            "installed."
        )
        sys.exit(2)

    screen_size = _parse_screen_size(args.screen_size)
    reuse_env = (not args.no_reuse_env)  # default ON
    save_frames = (not args.no_save_frames)  # default ON
    use_som = (not args.no_som)  # default ON

    # ─── Build opt-in steering / retrieval (improvements #3/#4/#6/#7) ────
    # The whole block is no-op when none of the four flags is set —
    # ``args._steering`` ends up as ``None`` and ``args._retriever``
    # stays ``None``, so ``run_actor_episode`` runs the original
    # main loop bit-for-bit.
    steering: Optional[Dict[str, Any]] = None
    enabled_steering: List[str] = []
    if (
        getattr(args, "enable_memory", False)
        or getattr(args, "enable_reflection", False)
        or getattr(args, "enable_self_verify", False)
    ):
        steer_mod = _import_osworld_steering()
        if steer_mod is None:
            print(
                "[WARNING] Advanced steering flags requested but "
                "cold_start.osworld_steering could not be imported; "
                "running with steering disabled."
            )
        else:
            steering = {}
            if getattr(args, "enable_memory", False):
                steering["memory"] = steer_mod["MemorySummary"](
                    client=client,
                    routed_model=routed_model,
                    chat_completion_fn=_chat_completion,
                    refresh_every=getattr(args, "memory_refresh_every", 5),
                )
                enabled_steering.append("memory")
            if getattr(args, "enable_reflection", False):
                steering["reflector"] = steer_mod["ReflexionTrigger"](
                    client=client,
                    routed_model=routed_model,
                    chat_completion_fn=_chat_completion,
                    trigger_streak=getattr(args, "reflection_streak", 2),
                )
                enabled_steering.append("reflection")
            if getattr(args, "enable_self_verify", False):
                steering["verifier"] = steer_mod["SelfVerifier"](
                    client=client,
                    routed_model=routed_model,
                    chat_completion_fn=_chat_completion,
                )
                enabled_steering.append("self_verify")
    args._steering = steering

    retriever = None
    if getattr(args, "skill_bank_path", None):
        retr_mod = _import_osworld_skill_retrieval()
        if retr_mod is None:
            print(
                "[WARNING] --skill_bank_path set but "
                "cold_start.osworld_skill_retrieval could not be "
                "imported; running without retrieval."
            )
        else:
            try:
                retriever = retr_mod["SkillBankRetriever"](
                    bank_path=args.skill_bank_path
                )
                print(
                    f"  Skill retrieval:      ON  "
                    f"({retriever.n_loaded} skills loaded, "
                    f"top_k={args.skill_retrieval_top_k})"
                )
            except Exception as exc:
                print(
                    f"[WARNING] Skill bank load failed at "
                    f"{args.skill_bank_path}: {exc} — disabling retrieval."
                )
                retriever = None
    args._retriever = retriever

    print("=" * 78)
    print("  Cold-Start Actor Agent — OSWorld + gpt-5.5  (vision-required, headless)")
    print("=" * 78)
    if _API_KEYS_FILE_USED is not None:
        print(f"  API keys file:        {_API_KEYS_FILE_USED}")
    print(f"  Catalog:              {catalog_path}")
    print(f"  Tasks resolved:       {len(tasks)} across "
          f"{len({d for d,_ in tasks})} domain(s)")
    if args.domains:
        print(f"  Domain filter:        {', '.join(args.domains)}")
    if args.task_ids:
        print(f"  Task ID filter:       {', '.join(args.task_ids)}")
    if args.tasks_per_domain:
        print(f"  Tasks/domain cap:     {args.tasks_per_domain}")
    print(f"  Episodes (per task):  {args.episodes}")
    print(f"  Max steps:            {args.max_steps}")
    print(f"  Max entities:         {args.max_entities}")
    print(f"  Model (configured):   {args.model}")
    print(f"  Model (routed):       {routed_model}")
    print(f"  Vision schema:        ON (mandatory; gpt-5.5 every step)")
    print(f"  Set-of-Marks:         "
          + ("ON  (numbered red boxes; click_element(id=N) action verbs)"
             if use_som else "OFF (--no_som ablation)"))
    if use_som:
        print(f"  SoM max elements:     {args.som_max_elements}")
    print(f"  Provider:             {args.provider_name}")
    print(f"  OS type:              {args.os_type}")
    print(f"  Screen size:          {screen_size[0]}x{screen_size[1]}")
    print(f"  Headless:             ON (mandatory)")
    print(f"  A11y tree:            {not args.no_a11y_tree}")
    print(f"  Terminal:             {not args.no_terminal}")
    print(f"  Auto-evaluate (DONE): {not args.no_auto_evaluate}")
    print(f"  Reuse env across tasks: {reuse_env}")
    if enabled_steering:
        print(
            f"  Advanced steering:    ON  ({', '.join(enabled_steering)})"
        )
    else:
        print(f"  Advanced steering:    OFF (default)")
    print(f"  Save frames:          "
          + ("ON  (PNG + step_NNN.json sidecar — default)"
             if save_frames else "OFF (--no_save_frames)"))
    print(f"  Resume:               {args.resume}")
    print(f"  Output:               {output_dir}")
    print("=" * 78)

    # Mirror the resolved knobs onto args so run_task_rollouts() can read them
    # without re-parsing.
    args._save_frames = save_frames
    args._use_som = use_som

    # Boot ONE shared env for the whole run when reuse_env is on; otherwise
    # spin up a fresh env per task. Headless is hard-wired ON.
    shared_env: Any = None
    if reuse_env:
        try:
            shared_env = OSWorldGymWrapper(
                provider_name=args.provider_name,
                path_to_vm=args.path_to_vm,
                os_type=args.os_type,
                action_space_type=args.action_space_type,
                headless=True,
                max_steps=args.max_steps,
                require_a11y_tree=not args.no_a11y_tree,
                require_terminal=not args.no_terminal,
                auto_evaluate=not args.no_auto_evaluate,
                screen_size=screen_size,
                pause_after_action=args.pause_after_action,
                client_password=args.client_password,
                enable_proxy=args.enable_proxy,
                # task_catalog: we drive task selection ourselves via
                # reset(options={"task_config": ...}), so the wrapper's
                # internal catalog can stay empty.
                task_catalog=[t for _, t in tasks],
            )
        except Exception as exc:
            print(f"[FATAL] Could not boot OSWorldGymWrapper: {exc}")
            traceback.print_exc()
            sys.exit(2)

    overall_t0 = time.time()
    task_summaries: List[Dict[str, Any]] = []

    try:
        for i, (domain, task_cfg) in enumerate(tasks):
            task_id = task_cfg.get("id", "<no-id>")
            print(f"\n{'━' * 78}")
            print(f"  TASK {i+1}/{len(tasks)} — {domain}/{task_id}")
            inst = task_cfg.get("instruction", "") or ""
            if inst:
                print(f"  Instruction: {_truncate(inst, 220)}")
            print(f"{'━' * 78}")

            args._inferred_domain = domain  # exposed on args for run_task_rollouts

            env_to_use = shared_env
            if not reuse_env:
                try:
                    env_to_use = OSWorldGymWrapper(
                        provider_name=args.provider_name,
                        path_to_vm=args.path_to_vm,
                        os_type=args.os_type,
                        action_space_type=args.action_space_type,
                        headless=True,
                        max_steps=args.max_steps,
                        require_a11y_tree=not args.no_a11y_tree,
                        require_terminal=not args.no_terminal,
                        auto_evaluate=not args.no_auto_evaluate,
                        screen_size=screen_size,
                        pause_after_action=args.pause_after_action,
                        client_password=args.client_password,
                        enable_proxy=args.enable_proxy,
                        task_catalog=[task_cfg],
                    )
                except Exception as exc:
                    print(f"  [FATAL] OSWorldGymWrapper boot for {task_id}: {exc}")
                    traceback.print_exc()
                    task_summaries.append({
                        "task_id": task_id, "domain": domain,
                        "error": f"env boot: {exc}",
                    })
                    continue

            try:
                summary = run_task_rollouts(
                    task_cfg,
                    args=args,
                    env=env_to_use,
                    output_dir=output_dir,
                    client=client,
                    routed_model=routed_model,
                    schema_helpers=schema_helpers,
                )
            except Exception as exc:
                traceback.print_exc()
                summary = {
                    "task_id": task_id, "domain": domain,
                    "error": str(exc),
                }
            task_summaries.append(summary)

            if not reuse_env and env_to_use is not None:
                try:
                    env_to_use.close()
                except Exception:
                    pass

    finally:
        if shared_env is not None:
            try:
                shared_env.close()
            except Exception:
                pass

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_osworld",
        "use_vision": True,
        "use_som": use_som,
        "som_max_elements": args.som_max_elements,
        "save_frames": save_frames,
        "task_catalog": str(catalog_path),
        "domains": sorted({d for d, _ in tasks}),
        "task_count": len(tasks),
        "episodes_per_task": args.episodes,
        "max_steps": args.max_steps,
        "max_entities": args.max_entities,
        "temperature_action": args.temperature_action,
        "temperature_schema": args.temperature_schema,
        "provider_name": args.provider_name,
        "os_type": args.os_type,
        "screen_size": list(screen_size),
        "headless": True,
        "elapsed_seconds": round(overall_elapsed, 2),
        "per_task_summaries": task_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ACTOR COLD-START (OSWORLD) — BATCH COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Tasks processed:  {len(task_summaries)}")
    completed = [
        s for s in task_summaries
        if not s.get("skipped") and "completed_episodes" in s
    ]
    total_eps = sum(s["completed_episodes"] for s in completed)
    print(f"  Total episodes:   {total_eps}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s "
          f"({overall_elapsed / 60.0:.1f} min)")
    print(f"  Output:           {output_dir}")
    print(f"  Master summary:   {master_path}")
    if completed:
        means = [s["mean_reward"] for s in completed if "mean_reward" in s]
        eval_means = [s["mean_eval_score"] for s in completed if "mean_eval_score" in s]
        steps_means = [s["mean_steps"] for s in completed if "mean_steps" in s]
        if means:
            print(f"  Avg reward:       {sum(means) / len(means):.3f}")
        if eval_means:
            print(f"  Avg eval score:   {sum(eval_means) / len(eval_means):.3f} "
                  f"(over {len(eval_means)} task(s) with DONE)")
        if steps_means:
            print(f"  Avg steps:        {sum(steps_means) / len(steps_means):.1f}")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<domain>/<safe_task_id>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
